/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/csrc/stable/tensor.h>

#include "libtorch_stable/torch_utils.h"

#include "cutlass_extensions/common.hpp"

#include "cutlass/cutlass.h"

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/packed_stride.hpp"

#include "core/math.hpp"

using namespace cute;

#define CHECK_TYPE(x, st, m)             \
  STD_TORCH_CHECK(x.scalar_type() == st, \
                  ": Inconsistency of torch::stable::Tensor type:", m)
#define CHECK_TH_CUDA(x, m) \
  STD_TORCH_CHECK(x.is_cuda(), m, ": must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x, m) \
  STD_TORCH_CHECK(x.is_contiguous(), m, ": must be contiguous")
#define CHECK_INPUT(x, st, m) \
  CHECK_TH_CUDA(x, m);        \
  CHECK_CONTIGUOUS(x, m);     \
  CHECK_TYPE(x, st, m)

constexpr auto FLOAT4_E2M1X2 = torch::headeronly::ScalarType::Byte;
constexpr auto SF_DTYPE = torch::headeronly::ScalarType::Float8_e4m3fn;

// Configuration for swapAB (used when M <= 64 for better GPU utilization)
// Uses larger K tile (256) for better throughput on small batch sizes
struct sm120_fp4_config_swapab {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _128, _256>;
  using PerSmTileShape_MNK = Shape<_128, _128, _256>;
};

// Configuration for M in (64, 256]
struct sm120_fp4_config_M256 {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _128, _128>;
  using PerSmTileShape_MNK = Shape<_128, _128, _128>;
};

// Configuration for M in (256, inf)
struct sm120_fp4_config_default {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_256, _128, _128>;
  using PerSmTileShape_MNK = Shape<_256, _128, _128>;
};

template <typename Config, typename OutType, bool swap_ab_ = false>
struct Fp4GemmSm120 {
  static constexpr bool swap_ab = swap_ab_;

  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutATag_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutATag>::type;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  using LayoutBTag_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutBTag>::type;
  static constexpr int AlignmentB = 32;

  using ElementD = OutType;
  using ElementC = OutType;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutCTag_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutCTag>::type;
  using LayoutDTag = cutlass::layout::RowMajor;
  using LayoutDTag_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutDTag>::type;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using MmaTileShape = typename Config::MmaTileShape;
  using ClusterShape = typename Config::ClusterShape;
  using PerSmTileShape_MNK = typename Config::PerSmTileShape_MNK;

  // Conditionally use transposed C/D layouts when swap_ab is enabled
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, ElementC,
          conditional_t<swap_ab, LayoutCTag_Transpose, LayoutCTag>, AlignmentC,
          ElementD, conditional_t<swap_ab, LayoutDTag_Transpose, LayoutDTag>,
          AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  // Conditionally swap A/B operands and their layouts in the mainloop
  using CollectiveMainloop = conditional_t<
      swap_ab,
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementB, LayoutBTag_Transpose, AlignmentB,
          ElementA, LayoutATag_Transpose, AlignmentA, ElementAccumulator,
          MmaTileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp,
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename GemmConfig>
typename GemmConfig::Gemm::Arguments args_from_options(
    torch::stable::Tensor& D, torch::stable::Tensor const& A,
    torch::stable::Tensor const& B, torch::stable::Tensor const& A_sf,
    torch::stable::Tensor const& B_sf, torch::stable::Tensor const& alpha,
    int M, int N, int K) {
  static constexpr bool swap_ab = GemmConfig::swap_ab;
  using Gemm = typename GemmConfig::Gemm;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementD = typename Gemm::ElementD;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementCompute = float;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using Sm1xxBlkScaledConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  // When swap_ab, the GEMM problem becomes (N, M, K) instead of (M, N, K)
  int m_eff = swap_ab ? N : M;
  int n_eff = swap_ab ? M : N;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m_eff, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n_eff, K, 1});
  auto stride_D =
      cutlass::make_cute_packed_stride(StrideD{}, {m_eff, n_eff, 1});

  auto prob_shape = cute::make_shape(m_eff, n_eff, K, 1);
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(prob_shape);
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(prob_shape);

  // When swap_ab: kernel's A operand gets B data, B operand gets A data,
  // and scale factor pointers are swapped accordingly
  auto* a_data = swap_ab ? static_cast<ElementA const*>(B.data_ptr())
                         : static_cast<ElementA const*>(A.data_ptr());
  auto* b_data = swap_ab ? static_cast<ElementB const*>(A.data_ptr())
                         : static_cast<ElementB const*>(B.data_ptr());
  auto* sfa_data = swap_ab ? static_cast<ElementSFA const*>(B_sf.data_ptr())
                           : static_cast<ElementSFA const*>(A_sf.data_ptr());
  auto* sfb_data = swap_ab ? static_cast<ElementSFB const*>(A_sf.data_ptr())
                           : static_cast<ElementSFB const*>(B_sf.data_ptr());

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m_eff, n_eff, K, 1},
      {a_data, stride_A, b_data, stride_B, sfa_data, layout_SFA, sfb_data,
       layout_SFB},
      {{},
       static_cast<ElementD const*>(D.data_ptr()),
       stride_D,
       static_cast<ElementD*>(D.data_ptr()),
       stride_D}};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<ElementCompute const*>(alpha.data_ptr());

  return arguments;
}

template <typename GemmConfig>
void runGemm(torch::stable::Tensor& D, torch::stable::Tensor const& A,
             torch::stable::Tensor const& B, torch::stable::Tensor const& A_sf,
             torch::stable::Tensor const& B_sf,
             torch::stable::Tensor const& alpha, int M, int N, int K,
             cudaStream_t stream) {
  using Gemm = typename GemmConfig::Gemm;
  Gemm gemm;

  auto arguments =
      args_from_options<GemmConfig>(D, A, B, A_sf, B_sf, alpha, M, N, K);

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace =
      torch::stable::empty(workspace_size, torch::headeronly::ScalarType::Byte,
                           std::nullopt, A.device());

  CUTLASS_CHECK(gemm.can_implement(arguments));

  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));

  CUTLASS_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}

void cutlass_fp4_bf16_gemm_dispatch(torch::stable::Tensor& D,
                                    torch::stable::Tensor const& A,
                                    torch::stable::Tensor const& B,
                                    torch::stable::Tensor const& A_sf,
                                    torch::stable::Tensor const& B_sf,
                                    torch::stable::Tensor const& alpha, int m,
                                    int n, int k, cudaStream_t stream) {
  uint32_t const mp2 = std::max(static_cast<uint32_t>(16), next_pow_2(m));
  if (mp2 <= 64) {
    // SwapAB for small M to improve GPU utilization during decode
    runGemm<Fp4GemmSm120<sm120_fp4_config_swapab, cutlass::bfloat16_t,
                         /*swap_ab=*/true>>(D, A, B, A_sf, B_sf, alpha, m, n, k,
                                            stream);
  } else if (mp2 <= 256) {
    runGemm<Fp4GemmSm120<sm120_fp4_config_M256, cutlass::bfloat16_t>>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  } else {
    runGemm<Fp4GemmSm120<sm120_fp4_config_default, cutlass::bfloat16_t>>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  }
}

void cutlass_fp4_f16_gemm_dispatch(torch::stable::Tensor& D,
                                   torch::stable::Tensor const& A,
                                   torch::stable::Tensor const& B,
                                   torch::stable::Tensor const& A_sf,
                                   torch::stable::Tensor const& B_sf,
                                   torch::stable::Tensor const& alpha, int m,
                                   int n, int k, cudaStream_t stream) {
  uint32_t const mp2 = std::max(static_cast<uint32_t>(16), next_pow_2(m));
  if (mp2 <= 64) {
    // SwapAB for small M to improve GPU utilization during decode
    runGemm<Fp4GemmSm120<sm120_fp4_config_swapab, cutlass::half_t,
                         /*swap_ab=*/true>>(D, A, B, A_sf, B_sf, alpha, m, n, k,
                                            stream);
  } else if (mp2 <= 256) {
    runGemm<Fp4GemmSm120<sm120_fp4_config_M256, cutlass::half_t>>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  } else {
    runGemm<Fp4GemmSm120<sm120_fp4_config_default, cutlass::half_t>>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  }
}

void cutlass_scaled_fp4_mm_sm120a(torch::stable::Tensor& D,
                                  torch::stable::Tensor const& A,
                                  torch::stable::Tensor const& B,
                                  torch::stable::Tensor const& A_sf,
                                  torch::stable::Tensor const& B_sf,
                                  torch::stable::Tensor const& alpha) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  CHECK_INPUT(A, FLOAT4_E2M1X2, "a");
  CHECK_INPUT(B, FLOAT4_E2M1X2, "b");

  CHECK_INPUT(A_sf, SF_DTYPE, "scale_a");
  CHECK_INPUT(B_sf, SF_DTYPE, "scale_b");

  CHECK_INPUT(alpha, torch::headeronly::ScalarType::Float, "alpha");

  STD_TORCH_CHECK(A.dim() == 2, "a must be a matrix");
  STD_TORCH_CHECK(B.dim() == 2, "b must be a matrix");
  STD_TORCH_CHECK(A.size(1) == B.size(1),
                  "a and b shapes cannot be multiplied (", A.size(0), "x",
                  A.size(1), " and ", B.size(0), "x", B.size(1), ")");

  auto const m = A.size(0);
  auto const n = B.size(0);
  auto const k = A.size(1) * 2;

  constexpr int alignment = 32;
  STD_TORCH_CHECK(k % alignment == 0, "Expected k to be divisible by ",
                  alignment, ", but got a shape: (", A.size(0), "x", A.size(1),
                  "), k: ", k, ".");
  STD_TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ",
                  alignment, ", but got b shape: (", B.size(0), "x", B.size(1),
                  ").");

  auto round_up = [](int x, int y) { return (x + y - 1) / y * y; };
  int rounded_m = round_up(m, 128);
  int rounded_n = round_up(n, 128);
  // Since k is divisible by 32 (alignment), k / 16 is guaranteed to be an
  // integer.
  int rounded_k = round_up(k / 16, 4);

  STD_TORCH_CHECK(A_sf.dim() == 2, "scale_a must be a matrix");
  STD_TORCH_CHECK(B_sf.dim() == 2, "scale_b must be a matrix");
  STD_TORCH_CHECK(A_sf.size(1) == B_sf.size(1),
                  "scale_a and scale_b shapes cannot be multiplied (",
                  A_sf.size(0), "x", A_sf.size(1), " and ", B_sf.size(0), "x",
                  B_sf.size(1), ")");
  STD_TORCH_CHECK(A_sf.size(0) == rounded_m && A_sf.size(1) == rounded_k,
                  "scale_a must be padded and swizzled to a shape (", rounded_m,
                  "x", rounded_k, "), but got a shape (", A_sf.size(0), "x",
                  A_sf.size(1), ")");
  STD_TORCH_CHECK(B_sf.size(0) == rounded_n && B_sf.size(1) == rounded_k,
                  "scale_b must be padded and swizzled to a shape (", rounded_n,
                  "x", rounded_k, "), but got a shape (", B_sf.size(0), "x",
                  B_sf.size(1), ")");

  auto out_dtype = D.scalar_type();
  const torch::stable::accelerator::DeviceGuard device_guard(
      A.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(A.get_device_index());

  if (out_dtype == torch::headeronly::ScalarType::BFloat16) {
    return cutlass_fp4_bf16_gemm_dispatch(D, A, B, A_sf, B_sf, alpha, m, n, k,
                                          stream);
  } else if (out_dtype == torch::headeronly::ScalarType::Half) {
    return cutlass_fp4_f16_gemm_dispatch(D, A, B, A_sf, B_sf, alpha, m, n, k,
                                         stream);
  } else {
    STD_TORCH_CHECK(false, "Unsupported output data type of nvfp4 mm sm120 (",
                    out_dtype, ")");
  }
#else
  STD_TORCH_CHECK(false,
                  "Unsupported CUTLASS version. Set VLLM_CUTLASS_SRC_DIR to "
                  "a CUTLASS 3.8 source directory to enable support.");
#endif  // defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
}
