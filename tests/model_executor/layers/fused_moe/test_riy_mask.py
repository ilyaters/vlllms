# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the RIY runtime mask / load-time prune helpers."""

import json

import pytest
import torch

from vllm.model_executor.layers.fused_moe.riy import (
    RiyMaskState,
    apply_riy_mask,
    build_riy_prune_logit_mask,
    build_riy_prune_map,
    load_riy_profile,
    pruned_experts_for_layer,
)


class TestApplyRiyMask:
    def test_no_mask_is_noop(self):
        topk_ids = torch.tensor([[0, 1], [2, 3]])
        topk_weights = torch.tensor([[0.5, 0.5], [0.3, 0.7]])
        mask = torch.zeros(4, dtype=torch.bool)
        out = apply_riy_mask(topk_weights, topk_ids, mask)
        # No masked experts -> weights unchanged.
        assert torch.allclose(out, topk_weights)

    def test_masked_zeroed_and_renormalized(self):
        topk_ids = torch.tensor([[0, 1]])
        topk_weights = torch.tensor([[0.4, 0.6]])
        mask = torch.tensor([True, False, False, False])  # expert 0 masked
        out = apply_riy_mask(topk_weights, topk_ids, mask)
        # expert 0 zeroed; remaining weight 0.6 renormalized to 1.0
        assert out[0, 0].item() == 0.0
        assert out[0, 1].item() == pytest.approx(1.0)

    def test_all_masked_token_no_nan(self):
        """E18: a token whose entire top_k is masked must NOT produce NaN."""
        topk_ids = torch.tensor([[0, 1]])
        topk_weights = torch.tensor([[0.5, 0.5]])
        mask = torch.tensor([True, True])  # both masked
        out = apply_riy_mask(topk_weights, topk_ids, mask)
        assert torch.isfinite(out).all()
        # All zeros (no renormalization).
        assert out.sum().item() == 0.0

    def test_partial_row_all_masked_others_ok(self):
        # Row 0 fully masked, row 1 partially masked.
        topk_ids = torch.tensor([[0, 1], [0, 2]])
        topk_weights = torch.tensor([[0.5, 0.5], [0.2, 0.8]])
        mask = torch.tensor([True, True, False])  # experts 0,1 masked
        out = apply_riy_mask(topk_weights, topk_ids, mask)
        assert torch.isfinite(out).all()
        # Row 0: all zeros
        assert out[0].sum().item() == 0.0
        # Row 1: expert 0 zeroed, expert 2 (0.8) renormalized to 1.0
        assert out[1, 0].item() == 0.0
        assert out[1, 1].item() == pytest.approx(1.0)


class TestRiyMaskState:
    def test_preallocated_tensors(self):
        """E11: tensors for all layers pre-allocated, always returned."""
        state = RiyMaskState(
            num_experts=4, num_layers=3, device=torch.device("cpu")
        )
        # Every layer returns a tensor (zero = all-allowed), even before
        # any mask is set.
        for lid in range(3):
            t = state.get_mask_tensor(lid)
            assert t.dtype == torch.bool
            assert t.shape == (4,)
            assert not t.any()

    def test_set_mask_in_place(self):
        state = RiyMaskState(
            num_experts=4, num_layers=2, device=torch.device("cpu")
        )
        t0_before = state.get_mask_tensor(0)
        state.set_mask([[0, 1], [1, 3]])
        t0_after = state.get_mask_tensor(0)
        # E11: same tensor object (address stable) — in-place update.
        assert t0_before is t0_after
        assert t0_after[1].item() is True
        assert t0_after[0].item() is False
        t1 = state.get_mask_tensor(1)
        assert t1[3].item() is True

    def test_clear_mask(self):
        state = RiyMaskState(
            num_experts=4, num_layers=1, device=torch.device("cpu")
        )
        state.set_mask([[0, 0]])
        assert state.get_mask_tensor(0)[0].item() is True
        state.clear_mask()
        assert not state.get_mask_tensor(0).any()
        assert state.get_mask() == []

    def test_get_mask_roundtrip(self):
        state = RiyMaskState(
            num_experts=8, num_layers=5, device=torch.device("cpu")
        )
        pairs = [[0, 1], [3, 7], [4, 0]]
        state.set_mask(pairs)
        assert sorted(state.get_mask()) == sorted(pairs)

    def test_out_of_range_layer_falls_back(self):
        state = RiyMaskState(
            num_experts=4, num_layers=2, device=torch.device("cpu")
        )
        # Out-of-range layer returns layer 0's tensor (no-op zero mask).
        t = state.get_mask_tensor(99)
        assert t.shape == (4,)
        assert not t.any()

    def test_no_new_allocations_after_init(self):
        """E11: set_mask must not replace dict entries (address stability)."""
        state = RiyMaskState(
            num_experts=4, num_layers=2, device=torch.device("cpu")
        )
        refs = {lid: state.get_mask_tensor(lid) for lid in range(2)}
        state.set_mask([[0, 0], [0, 1], [1, 2]])
        state.clear_mask()
        state.set_mask([[1, 3]])
        for lid in range(2):
            assert state.get_mask_tensor(lid) is refs[lid]


class TestBuildRiyPruneMap:
    def test_compaction_contiguous_and_minus_one(self):
        expert_map, compact = build_riy_prune_map(6, {1, 4})
        assert compact == 4
        # Pruned ids -> -1
        assert expert_map[1].item() == -1
        assert expert_map[4].item() == -1
        # Kept ids -> contiguous 0..3 in order
        kept = [expert_map[i].item() for i in [0, 2, 3, 5]]
        assert kept == [0, 1, 2, 3]

    def test_no_prune(self):
        expert_map, compact = build_riy_prune_map(4, set())
        assert compact == 4
        assert expert_map.tolist() == [0, 1, 2, 3]

    def test_logit_mask(self):
        m = build_riy_prune_logit_mask(4, {2})
        assert m.dtype == torch.float32
        assert m[0].item() == 0.0
        assert torch.isinf(m[2]) and m[2].item() < 0
        assert m[3].item() == 0.0


class TestLoadRiyProfile:
    def test_load_pruned_experts_format(self, tmp_path):
        path = tmp_path / "profile.json"
        path.write_text(json.dumps({"pruned_experts": [[0, 1], [3, 2]]}))
        prof = load_riy_profile(str(path))
        assert prof["pruned_experts"] == [[0, 1], [3, 2]]

    def test_load_layers_format(self, tmp_path):
        path = tmp_path / "profile.json"
        path.write_text(json.dumps({"layers": {"0": [1, 2], "3": [4]}}))
        prof = load_riy_profile(str(path))
        assert sorted(prof["pruned_experts"]) == sorted(
            [[0, 1], [0, 2], [3, 4]]
        )

    def test_cached(self, tmp_path):
        path = tmp_path / "profile.json"
        path.write_text(json.dumps({"pruned_experts": [[0, 0]]}))
        a = load_riy_profile(str(path))
        b = load_riy_profile(str(path))
        assert a is b  # cached

    def test_none_path(self):
        assert load_riy_profile(None) is None
        assert load_riy_profile("") is None

    def test_malformed_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(ValueError, match="Failed to load"):
            load_riy_profile(str(path))

    def test_missing_key_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"foo": 1}))
        with pytest.raises(ValueError, match="missing"):
            load_riy_profile(str(path))

    def test_pruned_experts_for_layer(self, tmp_path):
        prof = {"pruned_experts": [[0, 1], [0, 2], [3, 4]]}
        assert pruned_experts_for_layer(prof, 0) == {1, 2}
        assert pruned_experts_for_layer(prof, 3) == {4}
        assert pruned_experts_for_layer(prof, 1) == set()
        assert pruned_experts_for_layer(None, 0) == set()
        assert pruned_experts_for_layer(prof, None) == set()
