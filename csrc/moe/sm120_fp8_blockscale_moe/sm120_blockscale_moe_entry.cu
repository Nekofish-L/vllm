// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Entry point for SM120 FP8 block-scale MOE GEMM kernels.
// Provides C++ functions callable from PyTorch custom ops for:
//   1. sm120_fp8_blockscale_moe_gemm: Grouped GEMM with FP8 blockwise scaling
//   2. sm120_fp8_blockscale_quant_a: Online BF16→FP8 quantization
//
// The GEMM uses CUTLASS v4.4.2 CollectiveBuilder with SM120's blockwise
// scaling support (Sm120BlockwiseScaleConfig) and GroupProblemShape for
// per-expert grouped GEMM.
//
// These functions are registered as torch ops in csrc/moe/torch_bindings.cpp.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

#include "core/registration.h"
#include "sm120_fp8_quant.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass_extensions/common.hpp"

using namespace cute;

// ============================================================================
// Helper kernel: set up per-expert pointer arrays for grouped GEMM
// ============================================================================
template <typename ElementAB, typename ElementD, typename ElementSF>
__global__ void setup_group_gemm_pointers(
    ElementAB** a_ptrs, ElementAB** b_ptrs, ElementD** out_ptrs,
    ElementSF** a_sf_ptrs, ElementSF** b_sf_ptrs, int64_t* a_strides,
    int64_t* b_strides, int64_t* c_strides, int32_t* problem_sizes_out,
    // Base pointers
    ElementAB* a_base, ElementAB* b_base, ElementD* out_base,
    ElementSF* a_sf_base, ElementSF* b_sf_base,
    // Layout info
    const int64_t* token_offset, int num_experts, int N, int K) {
  int expert_id = threadIdx.x;
  if (expert_id >= num_experts) return;

  int64_t start = token_offset[expert_id];
  int64_t end = token_offset[expert_id + 1];
  int m = static_cast<int>(end - start);

  // A: [M_total, K] row-major, per-expert slice starts at row 'start'
  a_ptrs[expert_id] = a_base + start * K;

  // B: [num_experts, N, K] — each expert has its own [N, K] weight
  b_ptrs[expert_id] = b_base + expert_id * static_cast<int64_t>(N) * K;

  // Output: [M_total, N] row-major
  out_ptrs[expert_id] = out_base + start * N;

  // A scale factors: contiguous column-major [M_total, ceil(K/128)].
  // ptr_SFA[i] points to the start of expert i's rows in the global buffer.
  // The shared layout_SFA (computed on host with M=M_total) provides the
  // stride (1, M_total) so offsets are correct within the global buffer.
  a_sf_ptrs[expert_id] = a_sf_base + start;

  // B scale factors: [num_experts, N, ceil(K/128)] — each expert's scale
  // buffer is independent and identically laid out.
  int scale_k = (K + 127) / 128;
  b_sf_ptrs[expert_id] = b_sf_base + expert_id * N * scale_k;

  // Strides: stored as int64_t, binary-compatible with
  // Stride<int64_t, _1, _0> (EBO makes _1 and _0 zero-sized).
  a_strides[expert_id] = K;  // RowMajor A [M, K]: M-stride = K
  b_strides[expert_id] = K;  // ColumnMajor B [N, K]: N-stride = K
  c_strides[expert_id] = N;  // RowMajor output [M, N]: M-stride = N

  // Problem sizes: [M, N, K] per expert
  problem_sizes_out[expert_id * 3 + 0] = m;
  problem_sizes_out[expert_id * 3 + 1] = N;
  problem_sizes_out[expert_id * 3 + 2] = K;
}

// ============================================================================
// SM120 FP8 Blockwise Grouped GEMM using CUTLASS CollectiveBuilder
// ============================================================================

void sm120_fp8_blockscale_moe_gemm(torch::Tensor& output,
                                   torch::Tensor const& a_fp8,
                                   torch::Tensor const& b_fp8,
                                   torch::Tensor const& a_scales,
                                   torch::Tensor const& b_scales,
                                   torch::Tensor const& token_offset) {
  TORCH_CHECK(a_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "A must be FP8 E4M3");
  TORCH_CHECK(b_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "B must be FP8 E4M3");
  TORCH_CHECK(token_offset.scalar_type() == at::ScalarType::Long,
              "token_offset must be int64");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::BFloat16,
              "Output must be BFloat16");

  int M_total = a_fp8.size(0);
  int K = a_fp8.size(1);
  int num_experts = b_fp8.size(0);
  int N = b_fp8.size(1);

  TORCH_CHECK(b_fp8.size(2) == K, "K dimension mismatch between A and B");
  TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");

  if (M_total == 0) return;

  // --- CUTLASS types ---
  using ProblemShape =
      cutlass::gemm::GroupProblemShape<Shape<int32_t, int32_t, int32_t>>;
  using ElementAB = cutlass::float_e4m3_t;
  using ElementD = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using ElementBlockScale = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = LayoutC;

  static constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  // Scale config: blockwise scaling with granularity (1, 128, 128) matching
  // quantization block size
  using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<
      1, 128, 128, cute::UMMA::Major::MN, cute::UMMA::Major::K>;

  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  // Tile and schedule configuration
  using MmaTileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;

  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;

  using FusionOperation = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementAccumulator, ElementD, ElementAccumulator>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, MmaTileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, ElementD, LayoutC*, AlignmentC, ElementD,
          LayoutD*, AlignmentD, EpilogueSchedule,
          FusionOperation>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementAB, cute::tuple<LayoutA*, LayoutSFA>,
          AlignmentA, ElementAB, cute::tuple<LayoutB*, LayoutSFB>, AlignmentB,
          ElementAccumulator, MmaTileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                           CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;
  using UnderlyingProblemShape = ProblemShape::UnderlyingProblemShape;

  auto device = a_fp8.device();
  auto stream = at::cuda::getCurrentCUDAStream(a_fp8.get_device());

  // Allocate per-expert pointer arrays
  auto opts_long = torch::TensorOptions().device(device).dtype(torch::kLong);
  auto opts_int = torch::TensorOptions().device(device).dtype(torch::kInt);
  auto opts_byte = torch::TensorOptions().device(device).dtype(torch::kByte);

  auto a_ptrs = torch::empty({num_experts}, opts_long);
  auto b_ptrs = torch::empty({num_experts}, opts_long);
  auto out_ptrs = torch::empty({num_experts}, opts_long);
  auto a_sf_ptrs = torch::empty({num_experts}, opts_long);
  auto b_sf_ptrs = torch::empty({num_experts}, opts_long);
  auto a_strides_t = torch::empty({num_experts}, opts_long);
  auto b_strides_t = torch::empty({num_experts}, opts_long);
  auto c_strides_t = torch::empty({num_experts}, opts_long);
  auto problem_sizes = torch::empty({num_experts * 3}, opts_int);

  // Launch pointer setup kernel
  setup_group_gemm_pointers<ElementAB, ElementD, ElementBlockScale>
      <<<1, num_experts, 0, stream>>>(
          reinterpret_cast<ElementAB**>(a_ptrs.data_ptr()),
          reinterpret_cast<ElementAB**>(b_ptrs.data_ptr()),
          reinterpret_cast<ElementD**>(out_ptrs.data_ptr()),
          reinterpret_cast<ElementBlockScale**>(a_sf_ptrs.data_ptr()),
          reinterpret_cast<ElementBlockScale**>(b_sf_ptrs.data_ptr()),
          static_cast<int64_t*>(a_strides_t.data_ptr()),
          static_cast<int64_t*>(b_strides_t.data_ptr()),
          static_cast<int64_t*>(c_strides_t.data_ptr()),
          static_cast<int32_t*>(problem_sizes.data_ptr()),
          // Base pointers
          reinterpret_cast<ElementAB*>(a_fp8.data_ptr()),
          reinterpret_cast<ElementAB*>(b_fp8.data_ptr()),
          reinterpret_cast<ElementD*>(output.data_ptr()),
          static_cast<ElementBlockScale*>(a_scales.data_ptr()),
          static_cast<ElementBlockScale*>(b_scales.data_ptr()),
          // Layout
          static_cast<int64_t const*>(token_offset.data_ptr()), num_experts, N,
          K);

  // Compute shared scale factor layouts on the host.
  // The SM120 blockwise PtrArray mainloop uses a SINGLE shared layout
  // (not per-group layout pointers), unlike OpClassBlockScaledTensorOp.
  // For A scales: the global buffer is column-major [M_total, K/128], so
  // using M_total as M gives the correct stride (1, M_total).
  // For B scales: layout depends only on N and K (same for all experts).
  LayoutSFA layout_sfa =
      ScaleConfig::tile_atom_to_shape_SFA(make_shape(M_total, N, K, 1));
  LayoutSFB layout_sfb =
      ScaleConfig::tile_atom_to_shape_SFB(make_shape(M_total, N, K, 1));

  // Set up CUTLASS GEMM arguments
  Gemm gemm_op;

  UnderlyingProblemShape* problem_sizes_as_shapes =
      static_cast<UnderlyingProblemShape*>(problem_sizes.data_ptr());

  typename GemmKernel::MainloopArguments mainloop_args{
      reinterpret_cast<const ElementAB**>(a_ptrs.data_ptr()),
      static_cast<StrideA*>(a_strides_t.data_ptr()),
      reinterpret_cast<const ElementAB**>(b_ptrs.data_ptr()),
      static_cast<StrideB*>(b_strides_t.data_ptr()),
      reinterpret_cast<const ElementBlockScale**>(a_sf_ptrs.data_ptr()),
      layout_sfa,
      reinterpret_cast<const ElementBlockScale**>(b_sf_ptrs.data_ptr()),
      layout_sfb};

  typename GemmKernel::EpilogueArguments epilogue_args{
      {},
      nullptr,
      static_cast<StrideC*>(c_strides_t.data_ptr()),
      reinterpret_cast<ElementD**>(out_ptrs.data_ptr()),
      static_cast<StrideD*>(c_strides_t.data_ptr())};

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = a_fp8.get_device();
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes_as_shapes, nullptr},
      mainloop_args,
      epilogue_args,
      hw_info};

  TORCH_CHECK(gemm_op.can_implement(args) == cutlass::Status::kSuccess,
              "SM120 FP8 blockscale MOE GEMM: CUTLASS cannot implement "
              "the given problem shape");

  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto workspace =
      torch::empty({static_cast<int64_t>(workspace_size)}, opts_byte);

  auto status = gemm_op.initialize(args, workspace.data_ptr());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "SM120 FP8 blockscale MOE GEMM: initialization failed");

  status = gemm_op.run(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "SM120 FP8 blockscale MOE GEMM: execution failed");
}

// ============================================================================
// Online BF16 → FP8 + FP32 scale quantization for MOE activations
// ============================================================================

void sm120_fp8_blockscale_quant_a(torch::Tensor& fp8_output,
                                  torch::Tensor& scale_output,
                                  torch::Tensor const& input,
                                  torch::Tensor const& token_offset,
                                  int64_t num_experts) {
  TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16,
              "Input must be BFloat16");
  TORCH_CHECK(token_offset.scalar_type() == at::ScalarType::Long,
              "token_offset must be int64");

  int64_t M_total = input.size(0);
  int64_t K = input.size(1);

  TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");

  if (M_total == 0) return;

  constexpr int WarpsPerBlock = 4;
  constexpr int ThreadsPerBlock = WarpsPerBlock * 32;

  // Grid: x = K / 512 blocks, y = ceil(M_total / WarpsPerBlock)
  int64_t k_blocks = (K + 511) / 512;
  int64_t y_blocks = (M_total + WarpsPerBlock - 1) / WarpsPerBlock;

  dim3 grid(k_blocks, y_blocks, 1);
  dim3 block(ThreadsPerBlock, 1, 1);

  // Shared memory for token_offset array.
  // Typical MOE models have at most ~256 experts, so this is well within
  // the 48KB default shared memory limit (256+1)*8 = ~2KB.
  int smem_size = (num_experts + 1) * sizeof(int64_t);
  TORCH_CHECK(smem_size <= 48 * 1024,
              "sm120_fp8_blockscale_quant_a: too many experts (", num_experts,
              "), shared memory exceeds 48KB limit");

  // Scale K stride: M_total (column-major layout [M, ceil(K/128)])
  int64_t scale_k_stride = M_total;

  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  using namespace sm120_blockscaled_gemm;
  scale_1x128_kernel_sm120<__nv_bfloat16, __nv_fp8_e4m3, WarpsPerBlock>
      <<<grid, block, smem_size, stream>>>(
          reinterpret_cast<__nv_fp8_e4m3*>(fp8_output.data_ptr()),
          static_cast<float*>(scale_output.data_ptr()),
          reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr()),
          static_cast<int64_t const*>(token_offset.data_ptr()), num_experts, K,
          scale_k_stride);
}

// Register implementations with PyTorch dispatch
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("sm120_fp8_blockscale_moe_gemm", &sm120_fp8_blockscale_moe_gemm);
  m.impl("sm120_fp8_blockscale_quant_a", &sm120_fp8_blockscale_quant_a);
}
