# RIY — Expert Masking & Pruning

RIY (Routed-expert Intervention Yield) unifies **observe** and **act** for MoE
routed experts on top of the
[routed-experts statistics](routed_experts_stats.md) pipeline:

- **observe** — collect per-token routing (expert IDs + routing weights) and
  aggregate global activation counts and weight sums.
- **act (runtime)** — reversibly mask experts at serving time via REST API,
  without restarting.
- **act (load-time)** — permanently prune experts from the expert map at model
  load for VRAM savings (quantization-agnostic).

## Modes

Three orthogonal flags drive the pipeline:

| Flag | Effect | VRAM savings | Reversible |
|------|--------|--------------|------------|
| `--enable-routed-experts-stats` | Capture + aggregate activation counts **and** weight sums | No | N/A (observe) |
| `--enable-routed-experts-mask` | Bind the runtime `apply_riy_mask` hook + mask RPC channel | No | Yes (per request) |
| `--riy-expert-profile <file>` | Compact pruned experts out of the expert map at load | Yes (permanent) | No (restart to change) |

## Observe: collecting statistics

Run the server with stats on (use `--enforce-eager` while profiling for the
cleanest capture):

```bash
vllm serve <moe_model> \
    --enable-routed-experts-stats \
    --enforce-eager
```

`GET /v1/routed-experts/stats` now returns **two** axes:

- `expert_activation_counts` / `layer_expert_activation_counts` — how often
  each expert is selected (frequency).
- `expert_weight_sums` / `layer_expert_weight_sums` — the summed routing
  weight per expert (contribution).
- `most_weighted_experts` / `least_weighted_experts` — top/bottom by weight
  sum (percentages use `total_weight_sum` as the denominator, separate from
  the activation counts).

Use the two axes together to categorize experts: e.g. *specialists* =
rarely activated but high weight; *redundant* = rarely activated and low
weight (prune candidates).

## Act (runtime): reversible masking

Start with the runtime-mask flag (CUDA Graphs can stay on):

```bash
vllm serve <moe_model> --enable-routed-experts-mask
```

Then control the mask via REST:

```bash
# Mask experts (layer, expert) pairs — zero + renormalize their weights.
curl -X POST http://localhost:8000/v1/routed-experts/mask \
  -H 'Content-Type: application/json' \
  -d '{"pruned_experts": [[5, 3], [5, 7], [12, 1]]}'

# Read the current mask.
curl http://localhost:8000/v1/routed-experts/mask

# Clear the mask (allow every expert again).
curl -X DELETE http://localhost:8000/v1/routed-experts/mask
```

Semantics:

- Masked experts' routing weights are zeroed and the remaining weights are
  renormalized per token. The expert FFN still executes (weight 0 → no
  contribution), so this gives **no throughput/VRAM savings** — it is a
  reversible quality/A-B control.
- A token whose entire top-k selection is masked is left with all-zero
  weights (no NaN) rather than dividing by zero.
- The mask works in **logical** expert space (before EPLB mapping).

## Act (load-time): permanent pruning

Build a profile JSON, e.g.:

```json
{"pruned_experts": [[5, 3], [5, 7], [12, 1]]}
```

(or the layer-keyed form `{"layers": {"5": [3, 7], "12": [1]}}`).

Serve with the profile:

```bash
vllm serve <moe_model> --riy-expert-profile profile.json
```

At load time, pruned experts are compacted out of the expert map so their
weights are never allocated — permanent VRAM savings, quantization-agnostic.

You can also apply a profile at runtime as a **reversible runtime mask**
(useful for evaluating a profile before committing to a restart):

```bash
curl -X POST http://localhost:8000/v1/routed-experts/profile/load \
  -H 'Content-Type: application/json' \
  -d '{"path": "profile.json"}'
```

> Note: permanent VRAM-saving load-time pruning requires the
> `--riy-expert-profile` flag at **startup**. The `profile/load` endpoint
> applies the profile as a reversible runtime mask (the expert map cannot be
> re-compacted post-init without a reload).

## Limitations (validated at startup)

- **Expert parallelism (EP > 1)** is not supported for `--riy-expert-profile`
  or `--enable-routed-experts-mask`.
- **Monolithic MoE kernels** are not supported: `prune_logit_mask` and
  `apply_riy_mask` only act on the modular routing path
  (`BaseRouter._select_experts`). Monolithic kernels route internally and
  bypass these hooks.
- **Fused shared experts (ROCm AITER)** are incompatible with load-time
  pruning (compaction breaks the shared-expert concatenation).
- **EPLB** is incompatible with load-time pruning.
- The MoE kernel must support `expert_map` remapping for load-time pruning
  (supported: `fused_moe`, `fused_marlin`, `deep_gemm`, `gpt_oss_triton`,
  `rocm_aiter`, `xpu`, unquantized). Kernels that ignore `expert_map` would
  index compacted weights out of range.
- Each layer must keep at least `top_k` experts after pruning.
- Under `use_grouped_topk`, no expert group may be fully pruned (would NaN
  softmax).

## Tutorial: end-to-end workflow

The key insight of RIY is that **profiling and serving are separate steps**.
You measure on the full model with your real workload, build a profile, then
serve with that profile applied — without touching the checkpoint.

```
1.  Start the model fully loaded (no mask, no profile).
2.  Enable stats collection (or it auto-starts on first request).
3.  Reset stats for a clean slate.
4.  Run your actual workload (real prompts, real traffic mix).
5.  Read stats / watch the TUI → identify dead / rare / redundant experts.
6.  Apply a candidate mask live (reversible) and verify quality holds.
7.  Export the mask as a profile JSON.
8.  Restart with --riy-expert-profile → permanent VRAM savings.
9.  (Optional) share the profile — it is quantization-agnostic.
```

### Step 1 — Profile on the full model

Accurate routing statistics require the **complete, unmodified** model loaded
with full VRAM. No savings yet — profiling is a one-time cost.

```bash
# Profiling phase: stats + enforce-eager (CUDA Graph replay skips the
# capture hook, so stats must run in eager mode). TP/PP as needed.
vllm serve Qwen/Qwen3-30B-A3B \
    --enable-routed-experts-stats \
    --enforce-eager
```

Run your real traffic (chat completions, batch inference, replays of
production logs). The more representative the workload, the better the
profile.

### Step 2 — Read the stats

```bash
# Aggregated, human-readable (top/bottom experts by count and by weight)
curl -s "http://localhost:8000/v1/routed-experts/stats?limit=20" | jq

# Or watch live with the TUI (separate terminal)
python tools/riy_live.py --host localhost --port 8000
```

Two axes are reported per expert:

- **activation count** — how often the expert is selected (frequency);
- **weight sum** — total routing weight assigned to it (contribution).

Use them together to categorize each expert (see the table below).

### Step 3 — Try a mask live (reversible)

Before committing to a restart, apply the candidate prune set as a runtime
mask and verify output quality does not degrade:

```bash
# Mask a set of (layer, expert) pairs — weights zeroed + renormalized.
curl -X POST http://localhost:8000/v1/routed-experts/mask \
  -H 'Content-Type: application/json' \
  -d '{"pruned_experts": [[5,3],[5,7],[12,1],[12,9]]}'

# Read it back
curl http://localhost:8000/v1/routed-experts/mask

# Clear and try a different set
curl -X DELETE http://localhost:8000/v1/routed-experts/mask
```

Runtime masking gives **no VRAM/throughput savings** (the FFN still runs with
weight 0) — it is a reversible A/B check. If quality holds, the set is safe
to prune permanently.

### Step 4 — Export the profile

From the TUI, press `p` (enter a prune percentage — ranks experts by
frequency + contribution and masks the bottom N%), then `e` to export the
current mask as `riy_filter.<timestamp>.json`. Or build the JSON by hand /
from `stats` output:

```json
{
  "version": 1,
  "model": "Qwen/Qwen3-30B-A3B",
  "workload": "german-administrative",
  "pruned_experts": [[5,3],[5,7],[12,1],[12,9]]
}
```

The profile is just a list of `(layer, expert)` index pairs — no weights, no
activations, no model data. It is **quantization-agnostic**: the same profile
works on BF16, FP8, INT4.

### Step 5 — Serve with the profile (permanent savings)

```bash
# Serving phase: profile applied at load (compacted expert map → less VRAM),
# CUDA Graphs back on for throughput.
vllm serve Qwen/Qwen3-30B-A3B \
    --riy-expert-profile riy_filter.20260707_120000.json
```

At load time, pruned experts are compacted out of the expert map so their
weights are **never allocated**. The original HuggingFace checkpoint is used
as-is — no conversion, no re-quantization, no export step. The freed VRAM
goes to KV cache / larger batches.

You can also re-apply any profile at runtime as a reversible mask (useful to
re-evaluate a saved profile against new traffic without a restart):

```bash
curl -X POST http://localhost:8000/v1/routed-experts/profile/load \
  -H 'Content-Type: application/json' \
  -d '{"path": "riy_filter.20260707_120000.json"}'
```

> Note: permanent load-time pruning requires the `--riy-expert-profile` flag
> at **startup**. The `profile/load` endpoint applies the profile as a
> reversible runtime mask (the expert map cannot be re-compacted post-init
> without a reload).

## Expert categories

Combine the two axes to decide what to prune:

| Frequency | Contribution | Assessment |
|-----------|-------------|------------|
| Never | — | **Dead** — safe to prune |
| Rare | Low | **Candidate** — prune |
| Rare | High | **Specialist** — workload-dependent, caution |
| Frequent | Low | **Redundant** — candidate |
| Frequent | High | **Essential** — keep |

- **dead** — zero activations across the whole workload;
- **rare** — less than 1% of the most active expert;
- **low** — 1–10% of the most active;
- **active** — more than 10% of the most active;
- **prunable** — dead + rare (conservative prune set);
- **specialists** — rare frequency but high contribution (do **not** prune
  blindly — they may be critical for specific inputs).

## `riy live` TUI

A curses dashboard that renders every `(layer, expert)` cell on one screen.
Fill density = activation frequency (log scale); color = routing-weight
contribution (log scale). The bottom bar shows summary statistics and the
current prune level.

```
  riy live | Qwen/Qwen3-30B-A3B | localhost:8000 | L:48 E:128 | 2.0s | stats:on | ?=help
          |0    0    0    0    0    0    0    0    1    1    1    1    1
          |0    1    2    3    4    5    6    7    8    9    0    1    2
  L0001   |▓░▓█░░▒░░·░░▒░·░░░░░▒·░░░░·░░·░░░░▒░░··░░░░··░░░░░▓░··░░
  L0002   |█▒▓▓░░▒░░·░░▒░░░░░░░▒·░░░░·░░·░░░░▒░░··░░░░··░░░░░▓░··░░
  ...
  dead:1840(15%)  rare:4211  low:3892  active:2345  |  prunable:6051(49%)  specialists:23
  freq: · dead  ░ rare  ▒ low  ▓ high  █ dominant  |  color=contribution: ■■■■■
  prunable: [████████████████████████░░░░░░░░░░░░░░░░░░░░░░░] 49%  ACTIVE: 20% live:2457X
  q=quit  r=reset  p=prune  e=export  m=mask  s=save  g=enable  d=disable  ?=help
```

### Cell symbols

| Symbol | Meaning |
|--------|---------|
| `·` | Dead — never activated |
| `░` | Rare — < 1% of max |
| `▒` | Low — 1–10% of max |
| `▓` | High — frequently activated |
| `█` | Dominant — most activated |
| `X` | Masked — pruned by the current runtime mask |

### Color (contribution)

Dark/dim → cyan → green → orange → red (bold). Both scales are logarithmic
so rare experts stay visible instead of being crushed to zero by a few
dominant ones. Underlined cells are shared experts.

### Keybindings

| Key | Action |
|-----|--------|
| `p` | **Prune** — enter a target percentage (0–100%); ranks experts by frequency + contribution, masks the bottom N% live |
| `e` | **Export** — save the current mask as `riy_filter.<timestamp>.json` |
| `r` | **Reset** stats counters |
| `s` | **Save** raw stats to `riy_stats_export.json` |
| `m` | Toggle the mask overlay (`X` on masked cells) |
| `g` / `d` | Enable / disable stats collection |
| `j` / `k` | Scroll layers down / up |
| `h` / `l` | Scroll expert blocks left / right |
| `[` / `]` | Decrease / increase block size |
| `+` / `-` | Increase / decrease refresh interval |
| `?` | Help screen |
| `q` | Quit |

```bash
python tools/riy_live.py                          # default: localhost:8000
python tools/riy_live.py --host myhost --port 8000
python tools/riy_live.py --interval 1.0 --block 16
python tools/riy_live.py --demo                   # synthetic data, no vLLM needed
```

## Your personal REAP

Generic pruning tools (e.g. Cerebras REAP) decide which experts to cut based
on generic benchmarks. RIY lets you **measure on your own workload and
decide yourself**:

| | Cerebras REAP | vLLM RIY |
|--|--------------|---------|
| Calibration data | Generic benchmarks | Your workload |
| Output | Static pruned model | Profile JSON, model unchanged |
| Reversibility | No | Yes, any time (runtime mask) |
| Quantization-dependent | Yes | No — same profile, any quant |
| Automatic decisions | Yes | No — operator decides |

Because profiles are plain index lists with no model data:

- **Domain-specific.** A German law office, a Japanese game studio, and a
  medical research lab would each produce different profiles for the same
  model — and each would be optimal for their use case.
- **Stackable with quantization.** Profile a BF16 model, then apply the same
  profile to an INT4 or FP8 version. Expert indices do not change across
  quantization formats.
- **Shareable.** Publish profiles on HuggingFace for others to use:

  ```
  your-org/riy-profiles/
    Qwen3-30B-A3B/
      german-administrative-35pct.json
      general-coding-20pct.json
      japanese-customer-support-40pct.json
  ```

  Each profile documents its workload, prune percentage, and evaluation
  results. Users pick the profile that matches their use case — or create
  their own.

- **No vendor lock-in.** Profiles work on any vLLM installation with the RIY
  patch. The model on HuggingFace stays untouched.

## API summary

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/v1/routed-experts/stats` | Aggregated activation + weight stats |
| POST | `/v1/routed-experts/stats/reset` | Clear accumulated stats |
| POST | `/v1/routed-experts/stats/disable` | Pause collection |
| POST | `/v1/routed-experts/stats/enable` | Resume collection |
| GET | `/v1/routed-experts/mask` | Read the runtime mask |
| POST | `/v1/routed-experts/mask` | Set the runtime mask |
| DELETE | `/v1/routed-experts/mask` | Clear the runtime mask |
| POST | `/v1/routed-experts/profile/load` | Apply a profile (runtime mask) |

The router is attached when any of `--enable-routed-experts-stats`,
`--enable-routed-experts-mask`, or `--riy-expert-profile` is set.

## See also

- [Routed Experts Statistics](routed_experts_stats.md) — the underlying
  observe pipeline.
