# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the SM120 FP8 block-scale MOE expert backend.

Covers:
1. Utility functions: compute_aligned_M, expert_num_tokens_round_up_and_sum,
   count_expert_num_tokens
2. SM120BlockscaleMoEExperts class: static capability checks, construction
   validation, workspace_shapes correctness
3. Regression guard: torch.bincount fails on -1 padded expert_ids

Run: pytest tests/kernels/moe/test_sm120_expert_token_counting.py -v
"""

import math

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    compute_aligned_M,
    expert_num_tokens_round_up_and_sum,
)
from vllm.model_executor.layers.fused_moe.experts.sm120_fp8_blockscale_moe import (
    SM120BlockscaleMoEExperts,
)
from vllm.model_executor.layers.fused_moe.utils import count_expert_num_tokens
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)

BLOCK_SHAPE = [128, 128]


def _make_moe_config(
    num_experts: int = 8,
    experts_per_token: int = 2,
    hidden_dim: int = 256,
    intermediate_size: int = 512,
) -> FusedMoEConfig:
    """Helper to build a FusedMoEConfig for SM120 tests."""
    return FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        hidden_dim=hidden_dim,
        intermediate_size_per_partition=intermediate_size,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=512,
    )


def _make_quant_config(
    num_experts: int = 8,
    N: int = 512,
    K: int = 256,
) -> FusedMoEQuantConfig:
    """Helper to build an FP8 block-scale quant config for SM120 tests."""
    block_n, block_k = BLOCK_SHAPE
    n_tiles_w1 = math.ceil((2 * N) / block_n)
    k_tiles_w1 = math.ceil(K / block_k)
    n_tiles_w2 = math.ceil(K / block_n)
    k_tiles_w2 = math.ceil(N / block_k)

    w1_scale = torch.ones(
        num_experts, n_tiles_w1, k_tiles_w1, device="cuda", dtype=torch.float32
    )
    w2_scale = torch.ones(
        num_experts, n_tiles_w2, k_tiles_w2, device="cuda", dtype=torch.float32
    )

    return FusedMoEQuantConfig.make(
        quant_dtype=torch.float8_e4m3fn,
        block_shape=BLOCK_SHAPE,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )


def _make_sm120_experts(
    num_experts: int = 8,
    N: int = 512,
    K: int = 256,
) -> SM120BlockscaleMoEExperts:
    """Construct an SM120BlockscaleMoEExperts instance for testing."""
    moe_config = _make_moe_config(
        num_experts=num_experts, hidden_dim=K, intermediate_size=N
    )
    quant_config = _make_quant_config(num_experts=num_experts, N=N, K=K)
    return SM120BlockscaleMoEExperts(moe_config=moe_config, quant_config=quant_config)


# ===========================================================================
# Section 1: SM120BlockscaleMoEExperts class-level tests
# ===========================================================================


class TestSM120StaticMethods:
    """Tests for SM120BlockscaleMoEExperts static capability queries."""

    def test_activation_format_is_standard(self):
        assert (
            SM120BlockscaleMoEExperts.activation_format()
            == mk.FusedMoEActivationFormat.Standard
        )

    def test_supports_no_act_and_mul_is_false(self):
        assert SM120BlockscaleMoEExperts._supports_no_act_and_mul() is False

    def test_supports_silu_activation(self):
        assert SM120BlockscaleMoEExperts._supports_activation(MoEActivation.SILU)

    def test_supports_swiglustep_activation(self):
        assert SM120BlockscaleMoEExperts._supports_activation(MoEActivation.SWIGLUSTEP)

    @pytest.mark.parametrize(
        "activation",
        [MoEActivation.GELU, MoEActivation.IDENTITY],
    )
    def test_rejects_unsupported_activations(self, activation: MoEActivation):
        assert not SM120BlockscaleMoEExperts._supports_activation(activation)

    def test_supports_correct_quant_scheme(self):
        assert SM120BlockscaleMoEExperts._supports_quant_scheme(
            kFp8Static128BlockSym, kFp8Dynamic128Sym
        )

    def test_rejects_wrong_quant_scheme(self):
        assert not SM120BlockscaleMoEExperts._supports_quant_scheme(None, None)
        assert not SM120BlockscaleMoEExperts._supports_quant_scheme(
            kFp8Dynamic128Sym, kFp8Dynamic128Sym
        )

    def test_supports_ep_size_1_only(self):
        no_parallel = FusedMoEParallelConfig.make_no_parallel()
        assert SM120BlockscaleMoEExperts._supports_parallel_config(no_parallel)

    def test_rejects_ep_size_gt_1(self):
        ep_config = FusedMoEParallelConfig(
            tp_size=1, dp_size=1, ep_size=2, tp_rank=0, dp_rank=0, ep_rank=0
        )
        assert not SM120BlockscaleMoEExperts._supports_parallel_config(ep_config)


class TestSM120Construction:
    """Tests for SM120BlockscaleMoEExperts construction and validation."""

    def test_construction_succeeds_with_valid_config(self):
        experts = _make_sm120_experts()
        assert experts is not None
        assert experts.supports_expert_map() is True

    def test_construction_rejects_non_fp8_quant(self):
        moe_config = _make_moe_config()
        bad_quant = FusedMoEQuantConfig.make(
            quant_dtype=torch.int8,
            block_shape=BLOCK_SHAPE,
            w1_scale=torch.ones(8, 8, 2, device="cuda"),
            w2_scale=torch.ones(8, 2, 4, device="cuda"),
        )
        with pytest.raises(AssertionError):
            SM120BlockscaleMoEExperts(moe_config=moe_config, quant_config=bad_quant)

    def test_construction_rejects_wrong_block_shape(self):
        moe_config = _make_moe_config()
        bad_quant = FusedMoEQuantConfig.make(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[64, 64],
            w1_scale=torch.ones(8, 16, 4, device="cuda"),
            w2_scale=torch.ones(8, 4, 8, device="cuda"),
        )
        with pytest.raises(AssertionError):
            SM120BlockscaleMoEExperts(moe_config=moe_config, quant_config=bad_quant)

    def test_construction_rejects_per_act_token_quant(self):
        moe_config = _make_moe_config()
        bad_quant = FusedMoEQuantConfig.make(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=True,
            w1_scale=torch.ones(8, 1, 1, device="cuda"),
            w2_scale=torch.ones(8, 1, 1, device="cuda"),
        )
        with pytest.raises(AssertionError):
            SM120BlockscaleMoEExperts(moe_config=moe_config, quant_config=bad_quant)


class TestSM120WorkspaceShapes:
    """Tests for workspace_shapes correctness."""

    @pytest.mark.parametrize("M", [1, 32, 128, 256])
    @pytest.mark.parametrize("topk", [1, 2, 4])
    def test_workspace_shapes_no_meta(self, M: int, topk: int):
        """workspace_shapes must produce buffers large enough for the aligned
        M_sum that deepgemm_moe_permute will compute."""
        E, N, K = 8, 512, 256
        experts = _make_sm120_experts(num_experts=E, N=N, K=K)

        ws1, ws2, out = experts.workspace_shapes(
            M=M,
            N=2 * N,
            K=K,
            topk=topk,
            global_num_experts=E,
            local_num_experts=E,
            expert_tokens_meta=None,
            activation=MoEActivation.SILU,
        )

        # Output must be (M, K)
        assert out == (M, K)

        # First dim of workspace1 and workspace2 must be >= M*topk
        assert ws1[0] >= M * topk
        assert ws2[0] >= M * topk

        # Must be aligned to block_m=128
        assert ws1[0] % 128 == 0
        assert ws2[0] % 128 == 0

        # Second dim of workspace1 must fit max(activation_out_dim, K)
        # For SILU: activation_out_dim = 2*N / 2 = N
        assert ws1[1] >= max(N, K)

        # Second dim of workspace2 must fit max(2*N, K)
        assert ws2[1] >= max(2 * N, K)

    def test_workspace_shapes_with_meta(self):
        """workspace_shapes with expert_tokens_meta should use the actual
        per-expert counts for a tighter allocation."""
        E, N, K = 4, 512, 256
        experts = _make_sm120_experts(num_experts=E, N=N, K=K)

        expert_counts = torch.tensor([10, 20, 5, 30], dtype=torch.int32)
        meta = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_counts.to("cuda"),
            expert_num_tokens_cpu=expert_counts,
        )

        ws1, ws2, out = experts.workspace_shapes(
            M=65,
            N=2 * N,
            K=K,
            topk=1,
            global_num_experts=E,
            local_num_experts=E,
            expert_tokens_meta=meta,
            activation=MoEActivation.SILU,
        )

        expected_M_sum = expert_num_tokens_round_up_and_sum(expert_counts, 128)
        assert ws1[0] == expected_M_sum
        assert ws2[0] == expected_M_sum
        assert out == (65, K)

    def test_finalize_returns_no_op(self):
        """SM120 uses TopKWeightAndReduceNoOP for its weight/reduce step."""
        from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
            TopKWeightAndReduceNoOP,
        )

        experts = _make_sm120_experts()
        wr = experts.finalize_weight_and_reduce_impl()
        assert isinstance(wr, TopKWeightAndReduceNoOP)

    def test_workspace_dtype_passthrough(self):
        """workspace_dtype should return the same dtype as the input."""
        experts = _make_sm120_experts()
        assert experts.workspace_dtype(torch.bfloat16) == torch.bfloat16
        assert experts.workspace_dtype(torch.float16) == torch.float16


# ===========================================================================
# Section 2: Utility function tests
# ===========================================================================

# ---- Tests for compute_aligned_M ----


@pytest.mark.parametrize("M", [1, 8, 32, 127, 256])
@pytest.mark.parametrize("topk", [1, 2, 4, 8])
@pytest.mark.parametrize("local_num_experts", [4, 8, 16, 64])
@pytest.mark.parametrize("alignment", [1, 64, 128])
def test_compute_aligned_M_no_meta(
    M: int, topk: int, local_num_experts: int, alignment: int
):
    """compute_aligned_M without expert_tokens_meta returns a value that
    is >= M*topk and properly aligned."""
    result = compute_aligned_M(
        M, topk, local_num_experts, alignment, expert_tokens_meta=None
    )
    # Must be at least as large as the number of token-expert assignments
    assert result >= M * topk
    # Must be aligned
    assert result % alignment == 0


@pytest.mark.parametrize("alignment", [1, 64, 128])
def test_compute_aligned_M_with_meta(alignment: int):
    """compute_aligned_M with expert_tokens_meta uses the actual per-expert
    token counts instead of the worst-case estimate."""
    expert_num_tokens = torch.tensor([10, 20, 30, 5], dtype=torch.int32)
    meta = mk.ExpertTokensMetadata(
        expert_num_tokens=expert_num_tokens.to("cuda"),
        expert_num_tokens_cpu=expert_num_tokens,
    )
    result = compute_aligned_M(
        M=65,
        num_topk=1,
        local_num_experts=4,
        alignment=alignment,
        expert_tokens_meta=meta,
    )
    expected = expert_num_tokens_round_up_and_sum(
        expert_num_tokens, alignment=alignment
    )
    assert result == expected


# ---- Tests for expert_num_tokens_round_up_and_sum ----


def test_expert_num_tokens_round_up_and_sum_basic():
    """Each expert's token count should be rounded up to the alignment."""
    tokens = torch.tensor([1, 2, 3], dtype=torch.int64)
    # With alignment=128, each rounds up to 128, so total = 3 * 128 = 384
    assert expert_num_tokens_round_up_and_sum(tokens, alignment=128) == 384


def test_expert_num_tokens_round_up_and_sum_exact():
    """Exact multiples of alignment should not be padded further."""
    tokens = torch.tensor([128, 256], dtype=torch.int64)
    assert expert_num_tokens_round_up_and_sum(tokens, alignment=128) == 384


def test_expert_num_tokens_round_up_and_sum_zero():
    """Zero tokens for an expert should remain zero after alignment."""
    tokens = torch.tensor([0, 10, 0], dtype=torch.int64)
    # 0 rounds to 0, 10 rounds to 128
    assert expert_num_tokens_round_up_and_sum(tokens, alignment=128) == 128


# ---- Tests for count_expert_num_tokens with topk_ids ----


@pytest.mark.parametrize("num_tokens", [1, 8, 64, 128, 333])
@pytest.mark.parametrize("topk", [1, 2, 4])
@pytest.mark.parametrize("num_experts", [4, 8, 16])
def test_count_expert_num_tokens_matches_reference(
    num_tokens: int, topk: int, num_experts: int
):
    """count_expert_num_tokens on topk_ids should match a simple CPU
    reference counting."""
    if topk > num_experts:
        pytest.skip("topk > num_experts")

    torch.manual_seed(42)
    # Build topk_ids: each row has `topk` unique expert assignments
    topk_ids = torch.empty((num_tokens, topk), dtype=torch.int64, device="cpu")
    for i in range(num_tokens):
        topk_ids[i] = torch.randperm(num_experts)[:topk]

    topk_ids_cuda = topk_ids.to("cuda")

    # Reference: count on CPU
    ref_counts = torch.zeros(num_experts, dtype=torch.int32)
    for eid in topk_ids.flatten():
        ref_counts[eid.item()] += 1

    # Actual
    actual_counts = count_expert_num_tokens(topk_ids_cuda, num_experts, expert_map=None)

    torch.testing.assert_close(ref_counts.to("cuda"), actual_counts, atol=0, rtol=0)


# ---- Test that bincount would fail on expert_ids with -1 padding ----


def test_bincount_fails_on_negative_expert_ids():
    """Demonstrate that torch.bincount fails when expert_ids contains -1.
    This is the bug that was fixed by switching to count_expert_num_tokens."""
    # Simulate expert_ids from deepgemm_moe_permute with -1 padding
    expert_ids = torch.tensor([0, 1, -1, 2, -1, -1], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="bincount"):
        torch.bincount(expert_ids, minlength=3)


def test_count_expert_num_tokens_with_expert_map():
    """count_expert_num_tokens should correctly handle expert_map for
    expert-parallel scenarios."""
    num_global_experts = 8
    num_local_experts = 4
    ep_rank = 1  # owns experts 4, 5, 6, 7

    # Build expert_map: maps global expert index -> local expert index
    expert_map = torch.full((num_global_experts,), -1, dtype=torch.int32, device="cuda")
    start = ep_rank * num_local_experts
    for i in range(num_local_experts):
        expert_map[start + i] = i

    # topk_ids reference: tokens assigned to various global experts
    topk_ids = torch.tensor(
        [[4, 6], [5, 7], [0, 4], [3, 5]],  # token 0-3, topk=2
        dtype=torch.int64,
        device="cuda",
    )

    counts = count_expert_num_tokens(topk_ids, num_local_experts, expert_map)

    # Expected local counts:
    #   local 0 (global 4): tokens 0, 2 -> 2
    #   local 1 (global 5): tokens 1, 3 -> 2
    #   local 2 (global 6): token 0 -> 1
    #   local 3 (global 7): token 1 -> 1
    expected = torch.tensor([2, 2, 1, 1], dtype=torch.int32, device="cuda")
    torch.testing.assert_close(counts, expected, atol=0, rtol=0)


# ---- Test the full token_offset computation pattern ----


@pytest.mark.parametrize("num_tokens", [1, 16, 64])
@pytest.mark.parametrize("topk", [1, 2])
@pytest.mark.parametrize("num_experts", [4, 8])
def test_token_offset_computation(num_tokens: int, topk: int, num_experts: int):
    """Test the full token_offset computation pattern used in
    SM120BlockscaleMoEExperts.apply after the fix."""
    if topk > num_experts:
        pytest.skip("topk > num_experts")

    torch.manual_seed(42)
    topk_ids = torch.empty((num_tokens, topk), dtype=torch.int64, device="cpu")
    for i in range(num_tokens):
        topk_ids[i] = torch.randperm(num_experts)[:topk]

    topk_ids_cuda = topk_ids.to("cuda")

    # Replicate the fixed code path
    expert_num_tokens = count_expert_num_tokens(
        topk_ids_cuda, num_experts, expert_map=None
    )

    token_offset = torch.zeros(num_experts + 1, dtype=torch.long, device="cuda")
    torch.cumsum(expert_num_tokens.to(torch.long), dim=0, out=token_offset[1:])
    actual_M_total = int(token_offset[-1].item())

    # actual_M_total should equal total token-expert assignments
    assert actual_M_total == num_tokens * topk

    # token_offset should be monotonically non-decreasing
    assert torch.all(token_offset[1:] >= token_offset[:-1]).item()

    # First element should be 0
    assert token_offset[0].item() == 0
