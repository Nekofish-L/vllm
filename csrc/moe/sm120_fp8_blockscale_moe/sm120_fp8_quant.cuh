// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from TensorRT-LLM SM120 blockscale quantization kernel.
//
// Online BF16-to-FP8 quantization kernel with FP32 scale generation for
// SM120 MOE GEMM. Each warp processes one token's quantization block (512
// elements along K), computing per-128-element FP32 scales and quantizing
// BF16 to FP8 E4M3.
//
// Scale layout matches CUTLASS Sm120BlockwiseScaleConfig's deduce_layoutSFA:
//   column-major [M, ceil(K/128)]

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace sm120_blockscaled_gemm {

__device__ __forceinline__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

// Online BF16-to-FP8 quantization kernel for MOE activations.
//
// Quantizes BF16 activation tokens to FP8 E4M3 format with FP32 block
// scales. Each warp handles one token, processing kElemsPerWarp
// elements (4 blocks of kBlockSize=128 elements) per K-block.
//
// Scale output is stored as float in a column-major layout [M, ceil(K/128)]
// matching CUTLASS Sm120BlockwiseScaleConfig's deduce_layoutSFA.
//
// Template parameters:
//   InputType   - Input element type (e.g., __nv_bfloat16)
//   OutputType  - Output element type (e.g., __nv_fp8_e4m3)
//   WarpsPerBlock - Number of warps per thread block (default 4)
template <typename InputType, typename OutputType, int WarpsPerBlock = 4>
__global__ void scale_1x128_kernel_sm120(
    OutputType* __restrict__ fp8_output,
    float* __restrict__ scale_output,
    InputType const* __restrict__ input,
    int64_t const* __restrict__ token_offset, int64_t num_experts,
    int64_t size_k, int64_t scale_k_stride) {
  extern __shared__ char shared_memory[];
  int64_t* smem_token_offset = reinterpret_cast<int64_t*>(shared_memory);

  // Load token_offset into shared memory
  for (int i = threadIdx.x; i <= num_experts; i += blockDim.x) {
    smem_token_offset[i] = token_offset[i];
  }
  __syncthreads();

  // Get actual token_num from token_offset[num_experts]
  const int64_t token_num = smem_token_offset[num_experts];

  int const warp_id = threadIdx.x >> 5;
  int const lane_id = threadIdx.x & 31;

  const int64_t k_block_idx = blockIdx.x;
  const int64_t grid_stride =
      static_cast<int64_t>(gridDim.y) * WarpsPerBlock;

  for (int64_t token_idx =
           static_cast<int64_t>(blockIdx.y) * WarpsPerBlock + warp_id;
       token_idx < token_num; token_idx += grid_stride) {
    // Named constants for block quantization geometry
    // kBlockSize: quantization block size (128 elements per scale)
    // kBlocksPerWarp: blocks processed per warp (4 blocks × 128 = 512 elems)
    // kElemsPerThread: elements loaded per thread (32 threads × 16 = 512)
    constexpr int kBlockSize = 128;
    constexpr int kBlocksPerWarp = 4;
    constexpr int kElemsPerWarp = kBlockSize * kBlocksPerWarp;  // 512
    constexpr int kElemsPerThread = kElemsPerWarp / 32;         // 16

    // Check if this thread's data is within k bounds
    int const k_offset =
        (k_block_idx * kElemsPerWarp + lane_id * kElemsPerThread);

    // 1. Load 16 BF16 elements per thread (512 per warp)
    auto const cur_input_ptr = reinterpret_cast<double4 const*>(
        input + token_idx * size_k + k_offset);

    constexpr int kLoadNumElems =
        sizeof(double4) / sizeof(InputType);  // 16 for BF16

    union LoadTrick {
      double4 pack;
      InputType v[kLoadNumElems];
    };

    LoadTrick load_trick;
    load_trick.pack = k_offset < size_k ? cur_input_ptr[0] : double4{};

    // 2.1 Find max abs element in 16 elements per thread
    InputType max_elem = InputType(0.0f);
#pragma unroll
    for (int i = 0; i < kLoadNumElems; i++) {
      max_elem = __hmax(max_elem, __habs(load_trick.v[i]));
    }

    // 2.2 Find max in 8-lane group (128 elements = 1 quantization block)
    float amax = float(max_elem);
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, 4, 8));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, 2, 8));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, 1, 8));
    amax = fmaxf(amax, 1e-10f);

    // 3. Compute scale = amax / 448.0 (FP8 E4M3 max value)
    float scale = amax * reciprocal_approximate_ftz(448.0f);

    // Compute quant_scale = 1 / scale (for quantization)
    float quant_scale = reciprocal_approximate_ftz(scale);

    // 4.1 Quantize and store FP8 output
    constexpr int kStoreNumElems =
        sizeof(float4) / sizeof(OutputType);  // 16 for FP8

    union StoreTrick {
      float4 pack;
      OutputType v[kStoreNumElems];
    };

    StoreTrick store_trick;
    store_trick.pack = float4{};

#pragma unroll
    for (int i = 0; i < kStoreNumElems; i++) {
      store_trick.v[i] = OutputType(float(load_trick.v[i]) * quant_scale);
    }

    auto cur_output_ptr = reinterpret_cast<float4*>(
        fp8_output + token_idx * size_k + k_offset);

    if (k_offset < size_k) {
      cur_output_ptr[0] = store_trick.pack;
    }

    // 4.2 Store FP32 scale from lane 0 of each 8-lane group
    // Scale layout is column-major [M, ceil(K/128)]:
    //   scale_output[token_idx + k_scale_idx * scale_k_stride]
    // This matches CUTLASS Sm120BlockwiseScaleConfig's deduce_layoutSFA
    int group_lane = lane_id % 8;
    int group_id = lane_id / 8;  // 0..3 for 4 groups per warp

    if (group_lane == 0) {
      int k_scale_idx = k_block_idx * kBlocksPerWarp + group_id;
      int64_t scale_idx = token_idx + k_scale_idx * scale_k_stride;
      scale_output[scale_idx] = scale;
    }
  }
}

}  // namespace sm120_blockscaled_gemm
