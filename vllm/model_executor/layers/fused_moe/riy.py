# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""RIY (Routed-expert Intervention Yield) — runtime masking and load-time
pruning of routed MoE experts.

This module unifies the ``observe`` (routed-experts stats) and ``act``
(runtime mask / load-time prune) axes onto a single telemetry pipeline.

Two orthogonal ``act`` modes:
    - **runtime mask** (reversible, no VRAM savings): zero out the routing
      weights of selected experts and renormalize the rest, in-place on
      ``topk_weights``. Driven by :class:`RiyMaskState` over RPC.
    - **load-time prune** (permanent VRAM savings): compact pruned experts
      out of the expert map so their weights are never allocated. Driven by
      :func:`build_riy_prune_map` inside ``ExpertMapManager``.

See ``plans/riy_routed_experts_unification.md`` for the full design.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

import torch


def apply_riy_mask(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Zero routing weights of masked experts and renormalize.

    Args:
        topk_weights: ``(N, top_k)`` routing weights.
        topk_ids: ``(N, top_k)`` expert ids in **logical/global** space
            (i.e. before EPLB physical mapping).
        mask: ``(num_experts,)`` boolean tensor where ``True`` marks a
            **pruned/masked** expert. Must be on the same device as
            ``topk_weights``.

    Returns:
        ``(N, top_k)`` weights with masked entries set to 0 and the
        remaining entries renormalized to sum to 1 per row.

    Notes:
        - **E18 (NaN safety):** if a token's entire top_k selection is
          masked, the row sum is 0 and renormalization would produce NaN.
          Such rows are left as all-zeros (the experts still execute with
          weight 0, contributing nothing — see M2) instead of NaNing the
          model. This is graph-safe (static ops, no data-dependent Python
          branching).
        - ``topk_ids`` is **not** modified: masked experts still execute
          their FFN with weight 0 (runtime mask gives no throughput/VRAM
          savings, only reversible quality gating).
    """
    # Gather the per-(token, slot) pruned flag.
    pruned = mask[topk_ids]  # (N, top_k) bool
    topk_weights = topk_weights.masked_fill(pruned, 0.0)
    rowsum = topk_weights.sum(dim=-1, keepdim=True)  # (N, 1)
    safe = rowsum > 0
    # Where rowsum > 0 renormalize; where rowsum == 0 leave zeros (no NaN).
    topk_weights = torch.where(
        safe,
        topk_weights / rowsum.clamp_min(1e-12),
        topk_weights,
    )
    return topk_weights


class RiyMaskState:
    """Per-worker holder of runtime expert masks. Updated via RPC.

    Pre-allocates one zero (all-allowed) boolean mask tensor per layer on
    the target device in :meth:`__init__` (E11). Masks are updated
    strictly in-place (``zero_`` + index assignment) so tensor addresses
    stay stable across CUDA Graph capture/replay — the
    ``apply_riy_mask`` branch always executes identically under the graph.

    Thread-safe: updated from RPC threads, read from the forward stream.
    """

    def __init__(
        self,
        num_experts: int,
        num_layers: int,
        device: torch.device,
    ) -> None:
        self._lock = threading.Lock()
        self._num_experts = int(num_experts)
        self._num_layers = int(num_layers)
        self._device = device
        # Logical (layer, expert) pairs that are masked.
        self._mask: set[tuple[int, int]] = set()
        # E11: pre-allocate zero (all-allowed) masks for ALL layers on the
        # target device before CUDA Graph capture. Addresses are stable;
        # updates are strictly in-place. A lazy alloc / ``.to(device)``
        # that replaces the dict entry would break the graph (changing
        # address), and an empty mask at capture time would bake the
        # ``apply_riy_mask`` branch as a no-op.
        self._mask_tensors: dict[int, torch.Tensor] = {
            lid: torch.zeros(num_experts, dtype=torch.bool, device=device)
            for lid in range(num_layers)
        }

    @property
    def num_experts(self) -> int:
        return self._num_experts

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def device(self) -> torch.device:
        return self._device

    def set_mask(self, pruned_experts: list[list[int]]) -> None:
        """Replace the entire mask set and rebuild per-layer tensors.

        Args:
            pruned_experts: list of ``[layer_idx, expert_idx]`` pairs.
        """
        with self._lock:
            self._mask = {(int(p[0]), int(p[1])) for p in pruned_experts}
            self._rebuild_locked()

    def get_mask(self) -> list[list[int]]:
        """Return the current mask as a list of ``[layer, expert]`` pairs."""
        with self._lock:
            return [[lid, eid] for lid, eid in self._mask]

    def clear_mask(self) -> None:
        """Remove all masks (allow every expert)."""
        with self._lock:
            self._mask = set()
            self._rebuild_locked()

    def _rebuild_locked(self) -> None:
        # Rebuild per-layer tensors IN-PLACE (zero_ + index) to keep
        # addresses stable for CUDA Graph. No new allocations and no
        # ``.to(device)`` that would replace a dict entry.
        for t in self._mask_tensors.values():
            t.zero_()
        for layer_idx, expert_idx in self._mask:
            if (
                0 <= layer_idx < self._num_layers
                and 0 <= expert_idx < self._num_experts
            ):
                self._mask_tensors[layer_idx][expert_idx] = True

    def get_mask_tensor(self, layer_idx: int) -> torch.Tensor:
        """Return the mask tensor for ``layer_idx``.

        Always returns a tensor (never ``None``): a zero (all-allowed)
        tensor when the layer has no masks. This lets the
        ``apply_riy_mask`` branch always execute uniformly under the
        graph (E11). ``layer_idx`` out of range falls back to layer 0
        (a no-op zero mask).
        """
        with self._lock:
            return self._mask_tensors.get(
                layer_idx,
                self._mask_tensors[0],
            )


# ---------------------------------------------------------------------------
# Load-time prune helpers
# ---------------------------------------------------------------------------

def build_riy_prune_map(
    global_num_experts: int,
    pruned_experts: set[int],
) -> tuple[torch.Tensor, int]:
    """Build a compacted expert map for load-time pruning.

    Args:
        global_num_experts: total number of (logical) experts.
        pruned_experts: set of global expert ids to remove.

    Returns:
        ``(expert_map, num_kept)`` where ``expert_map`` is a
        ``(global_num_experts,)`` int32 tensor mapping each kept global id
        to a contiguous local id in ``[0, num_kept)`` and pruned ids to
        ``-1``. ``num_kept`` is the count of kept experts
        (``global_num_experts - len(pruned_experts)``).
    """
    expert_map = torch.full(
        (global_num_experts,), -1, dtype=torch.int32
    )
    compact = 0
    for i in range(global_num_experts):
        if i not in pruned_experts:
            expert_map[i] = compact
            compact += 1
    return expert_map, compact


def build_riy_prune_logit_mask(
    global_num_experts: int,
    pruned_experts: set[int],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the ``router_logits`` additive mask for load-time pruning.

    Returns a ``(global_num_experts,)`` tensor that is ``0.0`` for kept
    experts and ``-inf`` for pruned ones, so ``router_logits + mask``
    suppresses pruned experts before top-k.
    """
    mask = torch.zeros(global_num_experts, dtype=dtype)
    if pruned_experts:
        mask[list(pruned_experts)] = float("-inf")
    return mask


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------

_RIY_PROFILE_CACHE: dict[str, dict[str, Any]] = {}


def load_riy_profile(path: str | None) -> dict[str, Any] | None:
    """Load a RIY profile JSON, with a process-wide cache.

    Accepted formats (both supported for compatibility):

    1. ``{"pruned_experts": [[layer, expert], ...]}`` — the canonical
       RIY format.
    2. ``{"layers": {"<layer>": [expert, ...], ...}}`` — alternative
       layer-keyed format.

    Returns ``None`` when ``path`` is falsy. Raises ``ValueError`` on a
    malformed profile.
    """
    if not path:
        return None

    env_path = os.environ.get("RIY_EXPERT_PROFILE")
    if not path and env_path:
        path = env_path
    if not path:
        return None

    if path in _RIY_PROFILE_CACHE:
        return _RIY_PROFILE_CACHE[path]

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to load RIY profile '{path}': {e}") from e

    profile = _normalize_profile(raw, path)
    _RIY_PROFILE_CACHE[path] = profile
    return profile


def _normalize_profile(raw: dict[str, Any], path: str) -> dict[str, Any]:
    """Normalize the loaded JSON into the canonical
    ``{"pruned_experts": [[layer, expert], ...]}`` form.
    """
    pruned: list[list[int]] = []
    if "pruned_experts" in raw:
        for pair in raw["pruned_experts"]:
            if len(pair) != 2:
                raise ValueError(
                    f"RIY profile '{path}': pruned_experts entries must be "
                    f"[layer, expert] pairs, got {pair}"
                )
            pruned.append([int(pair[0]), int(pair[1])])
    elif "layers" in raw:
        for layer_str, experts in raw["layers"].items():
            layer = int(layer_str)
            for e in experts:
                pruned.append([layer, int(e)])
    else:
        raise ValueError(
            f"RIY profile '{path}': missing 'pruned_experts' or 'layers' key"
        )
    return {"pruned_experts": pruned}


def pruned_experts_for_layer(
    profile: dict[str, Any] | None, layer_idx: int | None
) -> set[int]:
    """Filter a profile down to the pruned expert ids for one layer.

    Returns an empty set when ``profile`` is ``None`` or ``layer_idx`` is
    ``None``.
    """
    if profile is None or layer_idx is None:
        return set()
    return {
        int(e)
        for lid, e in profile.get("pruned_experts", [])
        if int(lid) == int(layer_idx)
    }
