# Routed Experts Statistics

This document describes how to use the routed-experts statistics feature in vLLM to monitor which experts are activated most frequently in MoE (Mixture of Experts) models.

## Overview

When serving MoE models, vLLM can collect global statistics about which experts are activated across all requests. This data is exposed via REST API endpoints and can be used to:

- Understand expert utilization patterns
- Identify underutilized or overutilized experts
- Compute load balance metrics (Jain's fairness index)
- Debug routing issues

## Enabling Statistics Collection

Start the vLLM server with the `--enable-routed-experts-stats` flag:

```bash
vllm serve <moe_model> \
    --enable-routed-experts-stats
```

For example:

```bash
vllm serve deepseek-ai/DeepSeek-V2 \
    --enable-routed-experts-stats \
    --tensor-parallel-size 2
```

## API Endpoints

Once enabled, the following REST API endpoints are available:

### GET /v1/routed-experts/stats

Returns current expert activation statistics.

**Query Parameters:**
- `limit` (int, optional): Maximum number of entries in `most_activated_experts` / `least_activated_experts` (default: 10, max: 100)
- `include_zeros` (bool, optional): Include experts with zero activations in sorted lists (default: false)

**Example Response:**

```json
{
    "total_tokens_processed": 15000,
    "total_requests_processed": 42,
    "is_collecting": true,
    "num_experts": 256,
    "num_layers": 60,
    "top_k": 8,
    "load_balance_score": 0.85,
    "expert_activation_counts": {
        "0": 1234,
        "1": 5678,
        "42": 5432
    },
    "layer_expert_activation_counts": {
        "0": {"0": 100, "1": 200},
        "1": {"0": 150, "3": 300}
    },
    "most_activated_experts": [
        {"expert_id": 1, "count": 5678, "percentage": 12.3},
        {"expert_id": 42, "count": 5432, "percentage": 11.8}
    ],
    "least_activated_experts": [
        {"expert_id": 255, "count": 10, "percentage": 0.02}
    ]
}
```

### POST /v1/routed-experts/stats/reset

Resets all accumulated statistics to zero.

**Example Response:**

```json
{
    "status": "ok",
    "message": "Statistics reset"
}
```

### POST /v1/routed-experts/stats/disable

Pauses statistics collection (keeps accumulated data).

**Example Response:**

```json
{
    "status": "ok",
    "message": "Statistics collection disabled"
}
```

### POST /v1/routed-experts/stats/enable

Resumes statistics collection after it was disabled.

**Example Response:**

```json
{
    "status": "ok",
    "message": "Statistics collection enabled"
}
```

## Using the API

### Using curl

```bash
# Get current stats
curl http://localhost:8000/v1/routed-experts/stats

# Get stats with limit
curl "http://localhost:8000/v1/routed-experts/stats?limit=5"

# Reset stats
curl -X POST http://localhost:8000/v1/routed-experts/stats/reset

# Disable collection
curl -X POST http://localhost:8000/v1/routed-experts/stats/disable

# Re-enable collection
curl -X POST http://localhost:8000/v1/routed-experts/stats/enable
```

### Using Python

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1")

# Get stats
response = client.get("/routed-experts/stats")
print(response.json())

# Reset stats
client.post("/routed-experts/stats/reset")
```

## Metrics Explained

### expert_activation_counts

A dictionary mapping expert ID to total number of activations across all layers. Each activation represents one expert being selected for one token in one layer.

### layer_expert_activation_counts

A nested dictionary mapping layer ID to expert activation counts for that specific layer. Useful for analyzing per-layer routing patterns.

### load_balance_score

Jain's fairness index measuring how evenly experts are utilized:

- **1.0**: Perfect balance (all experts activated equally)
- **0.5**: Moderate imbalance
- **~0**: Extreme imbalance (single expert handles all requests)

Formula: `J = (sum(x_i))^2 / (n * sum(x_i^2))`

### most_activated_experts / least_activated_experts

Lists of top/bottom experts by activation count, with percentage of total activations.

## Limitations

The routed-experts stats feature has the following limitations:

1. **Expert Parallelism (EP)**: Not supported when `enable_expert_parallel=True`
2. **Context Parallelism (CP)**: Not supported when using DCP or PCP
3. **KV Connectors**: Not supported when using disaggregation or KV offload
4. **Non-MoE Models**: Works but will show zero activations since there are no routed experts

## Architecture

The feature consists of the following components:

- **RoutedExpertsStatsCollector**: Accumulates statistics in the scheduler process
- **EngineClient Protocol**: Defines the interface for accessing stats
- **API Router**: Exposes REST endpoints for stats access

Statistics are collected per-scheduler-step using numpy vectorization for minimal overhead (<1ms per step for typical batch sizes).

## Weight sums

In addition to activation counts, the collector aggregates **routing weight
sums** per expert (`expert_weight_sums`, `layer_expert_weight_sums`) and
exposes `most_weighted_experts` / `least_weighted_experts` lists. Use the two
axes together to find specialists (rare + high weight) and prune candidates
(rare + low weight). See [RIY — Expert Masking & Pruning](riy.md) for the
`act` modes (runtime mask + load-time prune) built on this telemetry.