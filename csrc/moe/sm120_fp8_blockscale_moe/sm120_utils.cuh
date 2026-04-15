// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from TensorRT-LLM SM120 blockscale utilities.
//
// This file provides the SM120BlockScaledBuilder template and utilities for
// SM120 FP8 block-scale MOE GEMM. It defines TMA descriptors, scale layouts,
// shared memory structures, and tile configurations for SM120's blockscaled
// MMA instructions.

#pragma once

#include <cstdint>

#include "cute/tensor.hpp"
#include "cute/algorithm/copy.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/array.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/mma_sm120.h"

using namespace cute;
using namespace cutlass;

namespace sm120_blockscaled_gemm {

// Compute padded offset for MOE-aware scale layout alignment.
// Ensures per-expert scale storage is aligned to 4-element boundaries,
// because 4 UE8M0 values are packed into each int32.
// IMPORTANT: The alignment constant must match the Python-side
// _SCALE_ALIGN in sm120_fp8_blockscale_moe.py.
template <typename T_offset, typename T_index>
CUTE_HOST_DEVICE static T_offset compute_padded_offset(T_offset offset,
                                                       T_index problem_idx) {
  // 4 UE8M0 values per int32 → 4-element alignment
  constexpr T_offset alignment = 4;
  return (offset + problem_idx * (alignment - 1)) / alignment * alignment;
}

// SM120 Blockscaled Builder: defines types, layouts, TMA descriptors, and
// shared memory structures for FP8 blockscale GEMM on SM120.
//
// Template parameters:
//   TileM_  - Tile size in M dimension (default 32)
//   TileN_  - Tile size in N dimension (default 128)
//   Stages_ - Pipeline depth for AB loading (default 4)
template <int TileM_ = 32, int TileN_ = 128, int Stages_ = 4>
struct SM120BlockScaledBuilder {
  // Element types
  using ElementA = cute::float_e4m3_t;
  using ElementB = cute::float_e4m3_t;
  using ElementSFLoad = int32_t;  // E8M0 packed as int32
  using ElementAccum = float;
  using ElementD = cute::bfloat16_t;

  // Pipeline configuration
  static constexpr int AB_Stages = Stages_;
  static constexpr int SF_Stages = 1;
  static constexpr int kTileM = TileM_;
  static constexpr int kTileN = TileN_;
  static constexpr int kSFVecSize = 128;  // 1x128 quantization block
  static constexpr int kTileSF = 1;
  static constexpr int kTileK = 128;
  static constexpr int kNumTileKPerSF = 512 / kTileK;
  static constexpr int kNumStagePerSF = kNumTileKPerSF / AB_Stages;
  static_assert(kNumStagePerSF > 0 && kNumStagePerSF <= 2,
                "kNumStagePerSF must be 1 or 2");
  static_assert(kNumTileKPerSF % AB_Stages == 0,
                "kNumTileKPerSF must be divisible by AB_Stages");

  using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
  using ScaleTileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileSF>>;
  using ClusterShape = Shape<_1, _1, _1>;
  using ProblemShape = Shape<int, int, int, int>;

  // MMA configuration
  using PermMmaTileM = Int<32>;
  using PermMmaTileN = Layout<Shape<_8, _4, _4>, Stride<_1, _32, _8>>;
  using PermMmaTileK = Underscore;
  using MMA_Atom = MMA_Atom<SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
      float_e4m3_t, float_e4m3_t, float, float_ue8m0_t, 32>>;
  using TiledMma =
      TiledMMA<MMA_Atom, Layout<Shape<_2, _4, _1>, Stride<_4, _1, _0>>,
               Tile<PermMmaTileM, PermMmaTileN, PermMmaTileK>>;

  static_assert(kTileM % cute::size(PermMmaTileM{}) == 0,
                "TileM must be divisible by PermMmaTileM");
  static_assert(kTileN % cute::size(PermMmaTileN{}) == 0,
                "TileN must be divisible by PermMmaTileN");

  static constexpr int kNumMathThreads = size(TiledMma::ThrLayoutVMNK{});
  static constexpr int kNumMathWarps = kNumMathThreads / 32;

  CUTE_HOST_DEVICE
  static auto ceil_div(int const& x, int const& y) {
    return (x + y - 1) / y;
  }

  CUTE_HOST_DEVICE
  static auto get_tma_aligned_size(int const& x) {
    // TMA requires 4-element alignment for int32 (E8M0 packed)
    return ((x + 3) / 4) * 4;
  }

  // Scale factor layout deduction
  CUTE_HOST_DEVICE
  static auto deduce_sfa_layout(ProblemShape const& problem_shape) {
    auto M = cute::get<0>(problem_shape);
    auto N = cute::get<1>(problem_shape);
    auto K = cute::get<2>(problem_shape);
    auto L = cute::get<3>(problem_shape);
    int64_t scale_m = static_cast<int64_t>(get_tma_aligned_size(M));
    int64_t scale_k = static_cast<int64_t>(ceil_div(K, 128 * 4));
    return make_layout(make_shape(scale_m, scale_k, L),
                       make_stride(Int<1>{}, scale_m,
                                   scale_m * scale_k));  // column major
  }

  CUTE_HOST_DEVICE
  static auto deduce_sfb_layout(ProblemShape const& problem_shape) {
    auto M = cute::get<0>(problem_shape);
    auto N = cute::get<1>(problem_shape);
    auto K = cute::get<2>(problem_shape);
    auto L = cute::get<3>(problem_shape);
    int64_t scale_n = static_cast<int64_t>(get_tma_aligned_size(N));
    int64_t scale_k = static_cast<int64_t>(ceil_div(K, 128 * 4));
    return make_layout(make_shape(scale_n, scale_k, L),
                       make_stride(Int<1>{}, scale_n,
                                   scale_n * scale_k));  // column major
  }

  // Scale fragment helpers for SM120 blockscaled MMA
  template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
  CUTE_HOST_DEVICE static constexpr auto thrfrg_SFA(SFATensor&& sfatensor,
                                                    TiledMMA<Atom, TiledThr,
                                                             TiledPerm>& mma) {
    using AtomLayoutSFA_TV =
        typename MMA_Atom::Traits::ALayout_SF;  // ((ThrV,FrgV),(RestM,RestK))
    auto tiled_atom_sfa = make_tile(
        make_layout(size<0, 0>(AtomLayoutSFA_TV{})),
        make_layout(make_shape(size<1, 0>(AtomLayoutSFA_TV{}),
                               size<1, 1>(AtomLayoutSFA_TV{}))));
    auto tv_atom_sfa =
        tiled_atom_sfa.compose(AtomLayoutSFA_TV{}, _);

    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
    auto thr_tile = make_tile(
        _,
        make_tile(make_layout(size<1>(thr_layout_vmnk)),
                  make_layout(size<3>(thr_layout_vmnk))));
    auto thr_tensor = zipped_divide(tv_atom_sfa, thr_tile);

    auto perm_tile = make_tile(_, make_tile(_, _));
    auto perm_tensor =
        thr_tensor.compose(perm_tile, _);

    return perm_tensor.compose(sfatensor, _);
  }

  template <class SFATensor, class ThrMma>
  CUTE_HOST_DEVICE static constexpr auto partition_fragment_SFA(
      SFATensor&& sfatensor, ThrMma& thread_mma) {
    auto thr_tensor = make_tensor(static_cast<SFATensor&&>(sfatensor).data(),
                                  thrfrg_SFA(sfatensor.layout(), thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    return thr_tensor(
        _, make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)), _);
  }

  CUTE_HOST_DEVICE
  static auto get_layoutSFA_TV(TiledMma& mma) {
    auto tile_shape_mnk = tile_shape(mma);
    auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), _1{}));
    auto thr_tensor = thrfrg_SFA(ref_A, mma);
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
    auto atile = make_tile(
        _,
        make_tile(
            make_layout(make_shape(size<1>(thr_layout_vmnk),
                                   size<3>(thr_layout_vmnk)),
                        make_stride(Int<0>{}, Int<1>{})),
            _));
    auto tv_sfa = thr_tensor.compose(atile, _);
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
    auto tv_layout = tv_sfa.compose(thridx_2_thrid, _);
    return tv_layout;
  }

  template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
  CUTE_HOST_DEVICE static constexpr auto thrfrg_SFB(SFBTensor&& sfbtensor,
                                                    TiledMMA<Atom, TiledThr,
                                                             TiledPerm>& mma) {
    using AtomLayoutSFB_TV = typename MMA_Atom::Traits::BLayout_SF;
    auto tiled_atom_sfb = make_tile(
        make_layout(size<0, 0>(AtomLayoutSFB_TV{})),
        make_layout(make_shape(size<1, 0>(AtomLayoutSFB_TV{}),
                               size<1, 1>(AtomLayoutSFB_TV{}))));
    auto tv_atom_sfb = tiled_atom_sfb.compose(AtomLayoutSFB_TV{}, _);
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
    auto thr_tile = make_tile(
        _,
        make_tile(make_layout(size<2>(thr_layout_vmnk)),
                  make_layout(size<3>(thr_layout_vmnk))));
    auto thr_tensor = zipped_divide(tv_atom_sfb, thr_tile);
    auto perm_tile = make_tile(_, make_tile(_, _));
    auto perm_tensor = thr_tensor.compose(perm_tile, _);
    return perm_tensor.compose(sfbtensor, _);
  }

  template <class SFBTensor, class ThrMma>
  CUTE_HOST_DEVICE static constexpr auto partition_fragment_SFB(
      SFBTensor&& sfbtensor, ThrMma& thread_mma) {
    auto thr_tensor = make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(),
                                  thrfrg_SFB(sfbtensor.layout(), thread_mma));
    auto thr_vmnk = thread_mma.thr_vmnk_;
    return thr_tensor(
        _, make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)), _);
  }

  CUTE_HOST_DEVICE
  static auto get_layoutSFB_TV(TiledMma& mma) {
    auto tile_shape_mnk = tile_shape(mma);
    auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), _1{}));
    auto thr_tensor = thrfrg_SFB(ref_B, mma);
    auto thr_layout_vmnk = mma.get_thr_layout_vmnk();
    auto btile = make_tile(
        _,
        make_tile(
            make_layout(make_shape(size<1>(thr_layout_vmnk),
                                   size<2>(thr_layout_vmnk)),
                        make_stride(Int<0>{}, Int<1>{})),
            _));
    auto tv_sfb = thr_tensor.compose(btile, _);
    auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
    auto tv_layout = tv_sfb.compose(thridx_2_thrid, _);
    return tv_layout;
  }

  // Transform scale fragments for qmma (quantized MMA)
  template <class Tensor>
  CUTE_HOST_DEVICE static auto transform_fragment_for_qmma(Tensor& tensor) {
    return make_tensor(tensor.data(),
                       make_layout(shape<0>(tensor.layout()),
                                   shape<1>(tensor.layout()),
                                   make_shape(Int<kNumStagePerSF>{},
                                              Int<AB_Stages>{})));
  }

  // Shared memory copy atoms
  using SmemCopyAtomA =
      Copy_Atom<SM75_U32x4_LDSM_N, ElementA>;
  using SmemCopyAtomB =
      Copy_Atom<SM75_U16x8_LDSM_T, ElementB>;
  using SmemCopyAtomSF =
      Copy_Atom<DefaultCopy, int32_t>;

  // Shared memory layouts
  using SmemLayoutAtomA = decltype(composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<_8, Shape<_32, _4>>, Stride<_32, Stride<_1, _256>>>{}));
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}),
                 Int<AB_Stages>{}),
      Step<_1, _2, _3>{}));

  using SmemLayoutAtomB = decltype(composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<_8, Shape<_32, _4>>, Stride<_32, Stride<_1, _256>>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}),
                 Int<AB_Stages>{}),
      Step<_1, _2, _3>{}));

  using SmemLayoutAtomSFA =
      Layout<Shape<Int<kTileM>, _1>, Stride<_1, Int<kTileM>>>;
  using SmemLayoutSFA = decltype(tile_to_shape(
      SmemLayoutAtomSFA{},
      make_shape(shape<0>(ScaleTileShape{}), shape<2>(ScaleTileShape{}),
                 Int<SF_Stages>{}),
      Step<_1, _2, _3>{}));

  using SmemLayoutAtomSFB =
      Layout<Shape<Int<kTileN>, _1>, Stride<_1, Int<kTileN>>>;
  using SmemLayoutSFB = decltype(tile_to_shape(
      SmemLayoutAtomSFB{},
      make_shape(shape<1>(ScaleTileShape{}), shape<2>(ScaleTileShape{}),
                 Int<SF_Stages>{}),
      Step<_1, _2, _3>{}));

  // TMA config
  using StrideA = Stride<int64_t, Int<1>, int64_t>;
  using StrideB = Stride<int64_t, Int<1>, int64_t>;

  using TMA_A = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(recast_ptr<ElementA>(nullptr),
                  repeat_like(StrideA{}, int64_t(0)), StrideA{}),
      SmemLayoutA{}(_, _, Int<0>{}),
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})), _1{}));

  using TMA_B = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(recast_ptr<ElementB>(nullptr),
                  repeat_like(StrideB{}, int64_t(0)), StrideB{}),
      SmemLayoutB{}(_, _, Int<0>{}),
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})), _1{}));

  // Scale TMA
  using StrideSFA = Stride<Int<1>, int64_t, int64_t>;
  using StrideSFB = Stride<Int<1>, int64_t, int64_t>;

  using TMA_SFA = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(recast_ptr<ElementSFLoad>(nullptr),
                  repeat_like(StrideSFA{}, int64_t(0)), StrideSFA{}),
      SmemLayoutSFA{}(_, _, cute::Int<0>{}),
      make_shape(shape<0>(ScaleTileShape{}), shape<2>(ScaleTileShape{})),
      _1{}));

  using TMA_SFB = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(recast_ptr<ElementSFLoad>(nullptr),
                  repeat_like(StrideSFB{}, int64_t(0)), StrideSFB{}),
      SmemLayoutSFB{}(_, _, cute::Int<0>{}),
      make_shape(shape<1>(ScaleTileShape{}), shape<2>(ScaleTileShape{})),
      _1{}));

  // TMA transaction sizes
  static constexpr uint32_t TmaTransactionBytesA =
      static_cast<uint32_t>(sizeof(ElementA) * size(SmemLayoutA{}(_, _, _0{})));
  static constexpr uint32_t TmaTransactionBytesB =
      static_cast<uint32_t>(sizeof(ElementB) * size(SmemLayoutB{}(_, _, _0{})));
  static constexpr uint32_t TmaABTransactionBytes =
      TmaTransactionBytesA + TmaTransactionBytesB;

  static constexpr uint32_t TmaTransactionBytesSFA =
      static_cast<uint32_t>(sizeof(ElementSFLoad) *
                            size(SmemLayoutSFA{}(_, _, _0{})));
  static constexpr uint32_t TmaTransactionBytesSFB =
      static_cast<uint32_t>(sizeof(ElementSFLoad) *
                            size(SmemLayoutSFB{}(_, _, _0{})));
  static constexpr uint32_t TmaSFTransactionBytes =
      TmaTransactionBytesSFA + TmaTransactionBytesSFB;

  // TMA store
  using StrideD = Stride<int64_t, Int<1>, int64_t>;
  using EpilogueTile_MN = Shape<Int<kTileM>, Int<kTileN>>;

  using CopyAtomC = Copy_Atom<SM90_U32x2_STSM_N, cutlass::half_t>;
  using SmemLayoutAtomD = decltype(composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<_8, Shape<_4, _16>>, Stride<_64, Stride<_16, _1>>>{}));
  using SmemLayoutD = decltype(tile_to_shape(
      SmemLayoutAtomD{},
      make_shape(shape<0>(EpilogueTile_MN{}), shape<1>(EpilogueTile_MN{}),
                 Int<1>{})));
  using CopyOpR2S = SM90_U32x2_STSM_N;
  using CopyOpS2G = SM90_TMA_STORE;
  using TMA_D = decltype(make_tma_copy_C_sm90(
      CopyOpS2G{},
      make_tensor(make_gmem_ptr(static_cast<ElementD*>(nullptr)),
                  repeat_like(StrideD{}, int64_t(0)), StrideD{}),
      take<0, 2>(SmemLayoutD{}), EpilogueTile_MN{}));

  // MOE store (register → smem → gmem path, no TMA store)
  using SmemAtomLayoutO = decltype(composition(
      Swizzle<3, 3, 3>{},
      Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{}));
  using SmemLayoutO = decltype(
      tile_to_shape(SmemAtomLayoutO{}, Shape<Int<kTileM>, Int<kTileN>>{}));
  using SmemCopyAtomR2S = Copy_Atom<AutoVectorizingCopy, ElementD>;
  using SmemCopyAtomS2R = Copy_Atom<UniversalCopy<uint128_t>, ElementD>;
  using GmemCopyAtomR2G = SmemCopyAtomS2R;
  using TiledCopyS2G = decltype(make_tiled_copy(
      SmemCopyAtomS2R{},
      Layout<Shape<_32, _8>, Stride<_8, _1>>{},
      Layout<Shape<_1, _8>>{}));

  // Shared memory structures
  struct SharedStorageLoad : cute::aligned_struct<128, _0> {
    alignas(1024) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>>
        smem_A;
    alignas(1024) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>>
        smem_B;
    alignas(1024)
        cute::ArrayEngine<ElementSFLoad, cute::cosize_v<SmemLayoutSFA>>
            smem_SFA;
    alignas(1024)
        cute::ArrayEngine<ElementSFLoad, cute::cosize_v<SmemLayoutSFB>>
            smem_SFB;
  };

  struct SharedStorageStore : cute::aligned_struct<128, _0> {
    alignas(1024) cute::ArrayEngine<ElementD, cute::cosize_v<SmemLayoutD>>
        smem_D;
  };

  struct SharedStorageMoeStore : cute::aligned_struct<128, _0> {
    alignas(1024) cute::ArrayEngine<ElementD, cute::cosize_v<SmemLayoutO>>
        smem_O;
  };

  union TensorStorage {
    SharedStorageLoad load;
    SharedStorageStore store;
  };

  union TensorStorageMoe {
    SharedStorageLoad load;
    SharedStorageMoeStore store;
  };

  // Barrier types
  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBarrier = cutlass::arch::ClusterBarrier;
  using ProducerBarrierType = FullBarrier::ValueType;
  using ConsumerBarrierType = EmptyBarrier::ValueType;

  struct BarrierStorage {
    FullBarrier ab_full_mbar[AB_Stages];
    EmptyBarrier ab_empty_mbar[AB_Stages];
    FullBarrier sf_full_mbar[SF_Stages];
    EmptyBarrier sf_empty_mbar[SF_Stages];
    EmptyBarrier store_full_mbar[SF_Stages];
    EmptyBarrier store_empty_mbar[SF_Stages];
  };
};

}  // namespace sm120_blockscaled_gemm
