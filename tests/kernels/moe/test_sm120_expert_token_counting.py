# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the SM120 expert token counting fix.

Verifies that:
1. compute_aligned_M produces correct aligned workspace sizes
2. count_expert_num_tokens correctly counts tokens from topk_ids
   (replacing the old torch.bincount(expert_ids) approach which failed
   when expert_ids contained -1 padding values from deepgemm_moe_permute)
3. token_offset cumsum computation works end-to-end with both
   expert_tokens_meta and fallback paths

Run: pytest tests/kernels/moe/test_sm120_expert_token_counting.py -v
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    compute_aligned_M,
    expert_num_tokens_round_up_and_sum,
)
from vllm.model_executor.layers.fused_moe.utils import count_expert_num_tokens

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
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk

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

    # Build expert_map: maps global expert index → local expert index
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
    #   local 0 (global 4): tokens 0, 2 → 2
    #   local 1 (global 5): tokens 1, 3 → 2
    #   local 2 (global 6): token 0 → 1
    #   local 3 (global 7): token 1 → 1
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
