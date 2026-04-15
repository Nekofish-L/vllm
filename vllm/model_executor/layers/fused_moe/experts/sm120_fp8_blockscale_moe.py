# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SM120 FP8 block-scale MOE expert backend for Blackwell GeForce (SM120).

Uses a custom persistent-thread CUTLASS kernel with SM120's blockscaled
MMA instructions (FP8 E4M3 data + UE8M0 block scales). The kernel performs:
  1. Online BF16→FP8 quantization of activations with E8M0 scale generation
  2. Grouped GEMM across expert groups using persistent-thread scheduling
  3. BF16 output with bounds-checked gmem writes

Data flow:
  BF16 tokens → quant_a → FP8 + E8M0 scales
  FP8 tokens × FP8 weights (with E8M0 scales) → BF16 output
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    deepgemm_moe_permute,
    deepgemm_unpermute_and_reduce,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _compute_padded_offset(offset: int, problem_idx: int) -> int:
    """Compute padded offset for MOE-aware scale layout alignment.
    Must match the C++ compute_padded_offset function."""
    alignment = 4
    return (offset + problem_idx * (alignment - 1)) // alignment * alignment


def _compute_aligned_M(
    M: int,
    num_topk: int,
    local_num_experts: int,
    alignment: int,
    expert_tokens_meta: mk.ExpertTokensMetadata | None,
) -> int:
    """Compute aligned total M accounting for per-expert alignment padding."""
    if expert_tokens_meta is not None:
        return int(expert_tokens_meta.expert_num_tokens.sum().item())
    # Rough estimate: each expert gets (M * topk / E) tokens, aligned up
    M_total = M * num_topk
    per_expert = (M_total + local_num_experts - 1) // local_num_experts
    per_expert = ((per_expert + alignment - 1) // alignment) * alignment
    return per_expert * local_num_experts


class SM120BlockscaleMoEExperts(mk.FusedMoEExpertsModular):
    """SM120 FP8 block-scale MOE expert implementation.

    Uses custom CUTLASS kernels for SM120's blockscaled MMA instructions
    with FP8 E4M3 data and UE8M0 (E8M0) block scales.
    """

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        assert quant_config.quant_dtype == torch.float8_e4m3fn
        assert quant_config.block_shape is not None
        assert quant_config.block_shape == [128, 128]
        assert not quant_config.per_act_token_quant
        assert not quant_config.per_out_ch_quant

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return p.is_cuda() and p.is_device_capability(120)

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (
            kFp8Static128BlockSym,
            kFp8Dynamic128Sym,
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [MoEActivation.SILU, MoEActivation.SWIGLUSTEP]

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        # Currently supports only non-distributed execution
        return moe_parallel_config.ep_size == 1

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        return act_dtype

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        M_total = M * topk
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M_total, max(activation_out_dim, K))
        workspace2 = (M_total, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def _quantize_a_for_sm120(
        self,
        a_bf16: torch.Tensor,
        token_offset: torch.Tensor,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Online BF16→FP8+E8M0 quantization for MOE activations.

        Returns:
            a_fp8: [M_total, K] FP8 E4M3 quantized activations
            a_scales: [sf_K, scale_ld] int32 packed E8M0 scales
        """
        M_total, K = a_bf16.shape
        assert K % 128 == 0

        a_fp8 = torch.empty(
            (M_total, K),
            dtype=torch.float8_e4m3fn,
            device=a_bf16.device,
        )

        # Scale layout: [sf_K, scale_leading_dim]
        sf_K = (K + 511) // 512
        # Padded M for scale alignment: the MOE scale layout packs 4 UE8M0
        # values per int32, so each expert's scale offset must be 4-aligned.
        # compute_padded_offset(offset, idx) = (offset + idx * 3) // 4 * 4
        # The worst case adds (num_experts - 1) * 3 padding elements, hence
        # scale_ld >= M_total + (num_experts - 1) * 3 rounded up to 4.
        _SCALE_ALIGN = 4  # Must match C++ compute_padded_offset alignment
        scale_ld = (
            (M_total + num_experts * (_SCALE_ALIGN - 1)) // _SCALE_ALIGN * _SCALE_ALIGN
        )
        a_scales = torch.zeros(
            (sf_K, scale_ld),
            dtype=torch.int32,
            device=a_bf16.device,
        )

        ops.sm120_fp8_blockscale_quant_a(
            a_fp8, a_scales, a_bf16, token_offset, num_experts
        )
        return a_fp8, a_scales

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        """
        Execute the SM120 FP8 block-scale MOE GEMM.

        Pipeline:
          1. Permute tokens by expert assignment
          2. Quantize A (BF16→FP8+E8M0) with MOE-aware scale layout
          3. GEMM1: A_fp8 × W1_fp8 → intermediate (BF16)
          4. Activation + quantize intermediate
          5. GEMM2: intermediate_fp8 × W2_fp8 → output (BF16)
          6. Unpermute and reduce with topk_weights
        """
        assert a1q_scale is not None
        assert self.w1_scale is not None
        assert self.w2_scale is not None

        a1q = hidden_states  # pre-quantized FP8 if available, or BF16
        _, N, K = w1.size()
        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        M = topk_ids.size(0)
        topk = topk_ids.size(1)
        M_total = M * topk

        assert w2.size(1) == K
        assert w1.dtype == torch.float8_e4m3fn
        assert w2.dtype == torch.float8_e4m3fn

        # Step 1: Permute tokens by expert assignment
        # Use deepgemm permute utilities for contiguous expert grouping
        a1q_perm = _resize_cache(workspace13.view(dtype=a1q.dtype), (M_total, K))
        a1q, a1q_scale, expert_ids, inv_perm = deepgemm_moe_permute(
            aq=a1q,
            aq_scale=a1q_scale,
            topk_ids=topk_ids,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            expert_tokens_meta=expert_tokens_meta,
            aq_out=a1q_perm,
        )

        # Build token_offset from expert_ids
        # token_offset[i] = number of tokens for experts 0..i-1 (cumulative)
        expert_num_tokens = torch.zeros(
            local_num_experts, dtype=torch.long, device=a1q.device
        )
        if expert_tokens_meta is not None:
            for i in range(local_num_experts):
                expert_num_tokens[i] = expert_tokens_meta.expert_num_tokens[i].item()
        else:
            # Count from expert_ids
            for i in range(local_num_experts):
                mask = expert_ids == i
                expert_num_tokens[i] = mask.sum().item()

        token_offset = torch.zeros(
            local_num_experts + 1, dtype=torch.long, device=a1q.device
        )
        torch.cumsum(expert_num_tokens, dim=0, out=token_offset[1:])
        actual_M_total = int(token_offset[-1].item())

        # Step 2: Quantize A (BF16→FP8+E8M0) for GEMM1
        # If a1q is already FP8, we still need the E8M0 scales
        if a1q.dtype == torch.bfloat16:
            a1_fp8, a1_scales = self._quantize_a_for_sm120(
                a1q[:actual_M_total], token_offset, local_num_experts
            )
        else:
            # Already FP8, use provided scales (convert to E8M0 layout if needed)
            a1_fp8 = a1q[:actual_M_total]
            a1_scales = a1q_scale  # Assume compatible format

        # Step 3: GEMM1: a1_fp8 × w1 → mm1_out (BF16)
        mm1_out = _resize_cache(workspace2, (actual_M_total, N))
        ops.sm120_fp8_blockscale_moe_gemm(
            mm1_out, a1_fp8, w1, a1_scales, self.w1_scale, token_offset
        )

        # Step 4: Activation + quantize intermediate for GEMM2
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        act_out = _resize_cache(workspace13, (actual_M_total, activation_out_dim))
        apply_moe_activation(activation, act_out, mm1_out)

        # Quantize activated output for GEMM2
        a2_fp8, a2_scales = self._quantize_a_for_sm120(
            act_out[:actual_M_total], token_offset, local_num_experts
        )

        # Step 5: GEMM2: a2_fp8 × w2 → mm2_out (BF16)
        mm2_out = _resize_cache(workspace2, (actual_M_total, K))
        ops.sm120_fp8_blockscale_moe_gemm(
            mm2_out, a2_fp8, w2, a2_scales, self.w2_scale, token_offset
        )

        # Step 6: Unpermute and reduce
        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)

        deepgemm_unpermute_and_reduce(
            a=mm2_out,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            inv_perm=inv_perm,
            expert_map=expert_map,
            output=output,
        )
