// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Entry point for SM120 FP8 block-scale MOE GEMM kernels.
// Provides C++ functions callable from PyTorch custom ops for:
//   1. sm120_fp8_blockscale_moe_gemm: Grouped GEMM with FP8 blockscale
//   2. sm120_fp8_blockscale_quant_a: Online BF16→FP8 quantization
//
// These functions are registered as torch ops in csrc/moe/torch_bindings.cpp.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include "core/registration.h"
#include "sm120_fp8_moe_gemm.cuh"
#include "sm120_fp8_quant.cuh"
#include "sm120_utils.cuh"

namespace sm120_blockscaled_gemm {

// Default tile configuration: TileM=32, TileN=128, Stages=4
using DefaultBuilder = SM120BlockScaledBuilder<32, 128, 4>;
using DefaultMoeKernel = SM120BlockScaledMoeKernel<DefaultBuilder>;

// Kernel launch wrapper for MOE GEMM
__global__ void __launch_bounds__(DefaultMoeKernel::MaxThreadsPerBlock,
                                  DefaultMoeKernel::MinBlocksPerMultiprocessor)
    sm120_moe_gemm_kernel(DefaultMoeKernel::Params params) {
  extern __shared__ char smem_buf[];
  DefaultMoeKernel kernel;
  kernel(params, smem_buf);
}

}  // namespace sm120_blockscaled_gemm

// C++ interface: SM120 FP8 block-scale MOE GEMM
//
// Arguments:
//   output    : [M_total, N]       BF16 output tensor
//   a_fp8     : [M_total, K]       FP8 E4M3 quantized activations
//   b_fp8     : [num_experts, N, K] FP8 E4M3 expert weights
//   a_scales  : [sf_K, pad(M_total)] int32 packed E8M0 scales for A
//   b_scales  : [num_experts, pad(N), sf_K] int32 packed E8M0 scales for B
//   token_offset : [num_experts + 1]  int64 cumulative token offsets
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

  int M_total = a_fp8.size(0);
  int K = a_fp8.size(1);
  int num_experts = b_fp8.size(0);
  int N = b_fp8.size(1);

  TORCH_CHECK(b_fp8.size(2) == K, "K dimension mismatch between A and B");
  TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");

  using namespace sm120_blockscaled_gemm;
  using KT = DefaultBuilder;
  using Kernel = DefaultMoeKernel;

  auto problem_shape = cute::make_shape(M_total, N, K, num_experts);

  // Build arguments
  typename Kernel::Arguments args{
      reinterpret_cast<KT::ElementA*>(a_fp8.data_ptr()),
      cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(M_total * K)),
      reinterpret_cast<KT::ElementB*>(b_fp8.data_ptr()),
      cute::make_stride(int64_t(K), cute::Int<1>{}, int64_t(N * K)),
      reinterpret_cast<KT::ElementSFLoad*>(a_scales.data_ptr()),
      typename KT::StrideSFA{},
      reinterpret_cast<KT::ElementSFLoad*>(b_scales.data_ptr()),
      typename KT::StrideSFB{},
      reinterpret_cast<KT::ElementD*>(output.data_ptr()),
      cute::make_stride(int64_t(N), cute::Int<1>{}, int64_t(M_total * N)),
      static_cast<int64_t*>(token_offset.data_ptr()),
  };

  auto params = Kernel::to_underlying_arguments(problem_shape, args);
  auto grid = Kernel::get_grid_shape(params);
  auto block = Kernel::get_block_shape();

  auto stream = at::cuda::getCurrentCUDAStream(a_fp8.get_device());

  // Set dynamic shared memory size
  int smem_size = Kernel::kSmemSize;
  if (smem_size > 48 * 1024) {
    cudaFuncSetAttribute(sm120_moe_gemm_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);
  }

  sm120_moe_gemm_kernel<<<grid, block, smem_size, stream>>>(params);
}

// C++ interface: Online BF16 → FP8 + E8M0 quantization for MOE activations
//
// Arguments:
//   fp8_output    : [M_total, K]       FP8 E4M3 output
//   scale_output  : [sf_K, scale_ld]   int32 packed E8M0 scales
//   input         : [M_total, K]       BF16 input
//   token_offset  : [num_experts + 1]  int64 cumulative token offsets
//   num_experts   : number of experts
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
              "sm120_fp8_blockscale_quant_a: too many experts (",
              num_experts, "), shared memory exceeds 48KB limit");

  // Scale leading dimension (padded M)
  int64_t m_padded =
      (M_total + num_experts * 3) / 4 * 4;  // match compute_padded_offset
  int64_t scale_leading_dim = m_padded;

  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  using namespace sm120_blockscaled_gemm;
  scale_1x128_kernel_sm120<__nv_bfloat16, __nv_fp8_e4m3, WarpsPerBlock>
      <<<grid, block, smem_size, stream>>>(
          reinterpret_cast<__nv_fp8_e4m3*>(fp8_output.data_ptr()),
          reinterpret_cast<int32_t*>(scale_output.data_ptr()),
          reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr()),
          static_cast<int64_t const*>(token_offset.data_ptr()),
          num_experts, K, scale_leading_dim);
}

// Register implementations with PyTorch dispatch
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("sm120_fp8_blockscale_moe_gemm", &sm120_fp8_blockscale_moe_gemm);
  m.impl("sm120_fp8_blockscale_quant_a", &sm120_fp8_blockscale_quant_a);
}
