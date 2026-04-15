// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from TensorRT-LLM SM120 blockscale quantization kernel.
//
// Online BF16-to-FP8 quantization kernel with E8M0 scale generation for
// SM120 MOE GEMM. Each warp processes one token's quantization block (512
// elements along K), computing per-128-element UE8M0 scales and quantizing
// BF16 to FP8 E4M3.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "sm120_utils.cuh"

namespace sm120_blockscaled_gemm {

// Compute reciprocal of 2^(exp-127) for UE8M0 scale
__device__ __forceinline__ float exp2f_rcp(uint8_t exp) {
  constexpr uint32_t FP32_EXPONENT_BIAS = 127;
  return (exp == 0) ? 1.0f
                    : exp2f(FP32_EXPONENT_BIAS - static_cast<float>(exp));
}

__device__ __forceinline__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

// Online BF16-to-FP8 quantization kernel for MOE activations.
//
// Quantizes BF16 activation tokens to FP8 E4M3 format with UE8M0 (E8M0)
// block scales. Each warp handles one token, processing kElemsPerWarp
// elements (4 blocks of kBlockSize=128 elements) per K-block.
//
// Scale output is packed as int32 (4 x UE8M0 per int32) in a MOE-aware
// layout using compute_padded_offset for per-expert alignment.
//
// Template parameters:
//   InputType   - Input element type (e.g., __nv_bfloat16)
//   OutputType  - Output element type (e.g., __nv_fp8_e4m3)
//   WarpsPerBlock - Number of warps per thread block (default 4)
template <typename InputType, typename OutputType, int WarpsPerBlock = 4>
__global__ void scale_1x128_kernel_sm120(
    OutputType* __restrict__ fp8_output,
    int32_t* __restrict__ scale_output,
    InputType const* __restrict__ input,
    int64_t const* __restrict__ token_offset, int64_t num_experts,
    int64_t size_k, int64_t scale_leading_dim) {
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
    // Binary search to find expert_idx:
    // token_offset[expert_idx] <= token_idx < token_offset[expert_idx + 1]
    int64_t expert_idx = 0;
    {
      int left = 0;
      int right = num_experts - 1;
      while (left < right) {
        int mid = (left + right + 1) >> 1;
        if (smem_token_offset[mid] <= token_idx) {
          left = mid;
        } else {
          right = mid - 1;
        }
      }
      expert_idx = left;
    }

    // Local token index within this expert
    const int64_t local_token_idx =
        token_idx - smem_token_offset[expert_idx];

    // Named constants for block quantization geometry
    // kBlockSize: quantization block size (128 elements per scale)
    // kBlocksPerWarp: blocks processed per warp (4 blocks × 128 = 512 elems)
    // kElemsPerThread: elements loaded per thread (32 threads × 16 = 512)
    constexpr int kBlockSize = 128;
    constexpr int kBlocksPerWarp = 4;
    constexpr int kElemsPerWarp = kBlockSize * kBlocksPerWarp;  // 512
    constexpr int kElemsPerThread = kElemsPerWarp / 32;  // 16

    // Check if this thread's data is within k bounds
    int const k_offset = (k_block_idx * kElemsPerWarp + lane_id * kElemsPerThread);

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

    // 3. Compute UE8M0 scale and quant_scale
    float dequant_scale_raw = amax * reciprocal_approximate_ftz(448.0f);
    __nv_fp8_e8m0 ue8m0_scale;
    ue8m0_scale.__x = __nv_cvt_float_to_e8m0(dequant_scale_raw,
                                               __NV_SATFINITE,
                                               cudaRoundPosInf);
    float quant_scale = exp2f_rcp(ue8m0_scale.__x);

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

    // 4.2 Pack scales from lane 0, 8, 16, 24 and store
    uint32_t s0 =
        __shfl_sync(0xFFFFFFFF, (uint32_t)ue8m0_scale.__x, 0);
    uint32_t s1 =
        __shfl_sync(0xFFFFFFFF, (uint32_t)ue8m0_scale.__x, 8);
    uint32_t s2 =
        __shfl_sync(0xFFFFFFFF, (uint32_t)ue8m0_scale.__x, 16);
    uint32_t s3 =
        __shfl_sync(0xFFFFFFFF, (uint32_t)ue8m0_scale.__x, 24);

    if (lane_id == 0) {
      uint32_t packed_scale = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);

      const int64_t scale_padded_offset = compute_padded_offset(
          static_cast<int64_t>(smem_token_offset[expert_idx]),
          expert_idx);

      auto cur_scale_ptr =
          scale_output + k_block_idx * scale_leading_dim + scale_padded_offset;
      *reinterpret_cast<uint32_t*>(&cur_scale_ptr[local_token_idx]) =
          packed_scale;
    }
  }
}

}  // namespace sm120_blockscaled_gemm
