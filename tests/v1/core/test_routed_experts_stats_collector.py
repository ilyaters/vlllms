# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for RoutedExpertsStatsCollector."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from vllm.v1.core.sched.routed_experts_stats_collector import (
    ExpertStatsSnapshot,
    RoutedExpertsStatsCollector,
    aggregate_snapshots,
    snapshot_to_dict,
)


class TestRoutedExpertsStatsCollector:
    """Tests for RoutedExpertsStatsCollector class."""

    def test_init_validates_num_experts(self):
        with pytest.raises(ValueError, match="num_experts must be > 0"):
            RoutedExpertsStatsCollector(num_experts=0, num_layers=2, top_k=2)

    def test_init_validates_num_layers(self):
        with pytest.raises(ValueError, match="num_layers must be > 0"):
            RoutedExpertsStatsCollector(num_experts=8, num_layers=0, top_k=2)

    def test_init_validates_top_k(self):
        with pytest.raises(ValueError, match="top_k must be > 0"):
            RoutedExpertsStatsCollector(num_experts=8, num_layers=2, top_k=0)

    def test_init_success(self):
        collector = RoutedExpertsStatsCollector(
            num_experts=8, num_layers=2, top_k=2
        )
        assert collector.num_experts == 8
        assert collector.num_layers == 2
        assert collector.top_k == 2
        assert collector.is_enabled is True

    def test_record_batch_empty(self):
        collector = RoutedExpertsStatsCollector(
            num_experts=8, num_layers=2, top_k=2
        )
        # Empty array should not raise
        collector.record_batch(np.array([], dtype=np.int64).reshape(0, 2, 2))
        stats = collector.get_stats()
        assert stats.total_tokens_processed == 0
        assert stats.total_requests_processed == 0
        assert stats.expert_activation_counts == {}

    def test_record_batch_none(self):
        collector = RoutedExpertsStatsCollector(
            num_experts=8, num_layers=2, top_k=2
        )
        collector.record_batch(None)
        stats = collector.get_stats()
        assert stats.total_tokens_processed == 0

    def test_record_batch_basic(self):
        collector = RoutedExpertsStatsCollector(
            num_experts=8, num_layers=2, top_k=2
        )
        # Shape: (num_tokens=3, num_layers=2, top_k=2)
        routing_data = np.array([
            [[0, 1], [2, 3]],
            [[1, 0], [3, 2]],
            [[4, 5], [6, 7]],
        ], dtype=np.int64)
        collector.record_batch(routing_data)

        stats = collector.get_stats()
        assert stats.total_tokens_processed == 3
        # Expert 0: 2 times (from [0,0] and [1,0])
        assert stats.expert_activation_counts[0] == 2
        # Expert 1: 2 times (from [0,1] and [1,1])
        assert stats.expert_activation_counts[1] == 2
        # Expert 2: 2 times
        assert stats.expert_activation_counts[2] == 2
        # Expert 3: 2 times
        assert stats.expert_activation_counts[3] == 2
        # Expert 4: 1 time
        assert stats.expert_activation_counts[4] == 1
        # Expert 5: 1 time
        assert stats.expert_activation_counts[5] == 1
        # Expert 6: 1 time
        assert stats.expert_activation_counts[6] == 1
        # Expert 7: 1 time
        assert stats.expert_activation_counts[7] == 1

    def test_record_batch_with_invalid_ids(self):
        collector = RoutedExpertsStatsCollector(
            num_experts=8, num_layers=2, top_k=2
        )
        # Include -1 as padding
        routing_data = np.array([
            [[0, 1], [2, -1]],
            [[1, -1], [-1, -1]],
        ], dtype=np.int64)
        collector.record_batch(routing_data)

        stats = collector.get_stats()
        # -1 entries should be ignored
        assert -1 not in stats.expert_activation_counts
        assert stats.expert_activation_counts[0] == 2
        assert stats.expert_activation_counts[1] == 2
        assert stats.expert_activation_counts[2] == 1

    def test_record_batch_out_of_range_ids(self):
        collector = RoutedExpertsStatsCollector(
            num_experts=4, num_layers=2, top_k=2
        )
        # Expert ID 10 is out of range (num_experts=4)
        routing_data = np.array([
            [[0, 1], [2, 10]],
        ], dtype=np.int64)
        collector.record_batch(routing_data)

        stats = collector.get_stats()
        # Expert 10 should be filtered out
        assert 10 not in stats.expert_activation_counts
        assert stats.expert_activation_counts[0] == 1
        assert stats.expert_activation_counts[1] == 1
        assert stats.expert_activation_counts[2] == 1

    def test_record_batch_layer_offset(self):
        collector = RoutedExpertsStatsCollector(
            num_experts=8, num_layers=4, top_k=2
        )
        # Simulate PP where this rank has layers 2-3 (offset=2)
        routing_data = np.array([
            [[0, 1], [2, 3]],
        ], dtype=np.int64)
        collector.record_batch(routing_data, layer_offset=2)

        stats = collector.get_stats()
        # Should be recorded at layer 2 and 3, not 0 and 1
        assert 2 in stats.layer_expert_activation_counts
        assert 3 in stats.layer_expert_activation_counts
        assert 0 not in stats.layer_expert_activation_counts
        assert 1 not in stats.layer_expert_activation_counts

    def test_record_batch_invalid_shape(self):
        collector = RoutedExpertsStatsCollector(
            num_experts=8, num_layers=2, top_k=2
        )
        # 2D array instead of 3D should be ignored
        routing_data = np.array([[0, 1, 2, 3]], dtype=np.int64)
        collector.record_batch(routing_data)

        stats = collector.get_stats()
        assert stats.total_tokens_processed == 0

    def test_record_request(self):
        collector = RoutedExpertsStatsCollector(
            num_experts=8, num_layers=2, top_k=2
        )
        collector.record_request()
        collector.record_request()
        collector.record_request()

        stats = collector.get_stats()
        assert stats.total_requests_processed == 3

    def test_record_request_disabled(self):
        collector = RoutedExpertsStatsCollector(
            num_experts=8, num_layers=2, top_k=2
        )
        collector.disable()
        collector.record_batch(np.array([[[0, 1]]], dtype=np.int64))
        collector.record_request()

        stats = collector.get_stats()
        assert stats.total_tokens_processed == 0
        assert stats.total_requests_processed == 0

    def test_disable_enable(self):
        collector = RoutedExpertsStatsCollector(
            num_experts=8, num_layers=2, top_k=2
        )
        assert collector.is_enabled is True

        collector.disable()
        assert collector.is_enabled is False

        collector.enable()
        assert collector.is_enabled is True

    def test_reset(self):
        collector = RoutedExpertsStatsCollector(
            num_experts=8, num_layers=2, top_k=2
        )
        routing_data = np.array([
            [[0, 1], [2, 3]],
            [[4, 5], [6, 7]],
        ], dtype=np.int64)
        collector.record_batch(routing_data)
        collector.record_request()

        collector.reset()

        stats = collector.get_stats()
        assert stats.total_tokens_processed == 0
        assert stats.total_requests_processed == 0
        assert stats.expert_activation_counts == {}
        assert stats.layer_expert_activation_counts == {}

    def test_load_balance_score_perfect(self):
        collector = RoutedExpertsStatsCollector(
            num_experts=4, num_layers=1, top_k=1
        )
        # Perfectly balanced: each expert activated same number of times
        routing_data = np.array([
            [[0], [1], [2], [3]],
        ], dtype=np.int64).reshape(4, 1, 1)
        collector.record_batch(routing_data)

        stats = collector.get_stats()
        assert stats.load_balance_score == 1.0

    def test_load_balance_score_unbalanced(self):
        collector = RoutedExpertsStatsCollector(
            num_experts=4, num_layers=1, top_k=1
        )
        # Highly unbalanced: one expert dominates
        routing_data = np.array([
            [[0], [0], [0], [0]],
        ], dtype=np.int64).reshape(4, 1, 1)
        collector.record_batch(routing_data)

        stats = collector.get_stats()
        # Should be close to 0.25 (1/4) for worst case
        assert 0.2 < stats.load_balance_score < 0.3

    def test_load_balance_score_empty(self):
        collector = RoutedExpertsStatsCollector(
            num_experts=4, num_layers=1, top_k=1
        )
        stats = collector.get_stats()
        # Empty should return 1.0
        assert stats.load_balance_score == 1.0

    def test_thread_safety(self):
        collector = RoutedExpertsStatsCollector(
            num_experts=64, num_layers=8, top_k=4
        )

        def worker():
            for _ in range(100):
                routing_data = np.random.randint(
                    0, 64, size=(16, 8, 4), dtype=np.int64
                )
                collector.record_batch(routing_data)
                collector.record_request()

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = collector.get_stats()
        # With 4 threads × 100 iterations × 16 tokens = 6400 tokens
        assert stats.total_tokens_processed == 6400
        # With 4 threads × 100 iterations × 1 request = 400 requests
        assert stats.total_requests_processed == 400

    def test_concurrent_disable_enable(self):
        collector = RoutedExpertsStatsCollector(
            num_experts=8, num_layers=2, top_k=2
        )

        def toggle():
            for _ in range(50):
                collector.disable()
                collector.enable()

        threads = [threading.Thread(target=toggle) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not crash and should end in a valid state
        assert collector.is_enabled in (True, False)


class TestAggregateSnapshots:
    """Tests for aggregate_snapshots function."""

    def test_aggregate_empty(self):
        result = aggregate_snapshots([])
        assert result.total_tokens_processed == 0
        assert result.total_requests_processed == 0
        assert result.expert_activation_counts == {}
        assert result.num_experts == 0

    def test_aggregate_single(self):
        snapshots = [
            ExpertStatsSnapshot(
                total_tokens_processed=100,
                total_requests_processed=5,
                expert_activation_counts={0: 50, 1: 50},
                layer_expert_activation_counts={0: {0: 50}, 1: {1: 50}},
                top_k=2,
                num_layers=2,
                num_experts=8,
                is_collecting=True,
                load_balance_score=0.9,
            )
        ]
        result = aggregate_snapshots(snapshots)
        assert result.total_tokens_processed == 100
        assert result.total_requests_processed == 5
        assert result.expert_activation_counts == {0: 50, 1: 50}

    def test_aggregate_multiple_ranks(self):
        snapshots = [
            ExpertStatsSnapshot(
                total_tokens_processed=100,
                total_requests_processed=5,
                expert_activation_counts={0: 50, 1: 50},
                layer_expert_activation_counts={0: {0: 50}},
                top_k=2,
                num_layers=2,
                num_experts=8,
                is_collecting=True,
                load_balance_score=0.9,
                dp_rank=0,
            ),
            ExpertStatsSnapshot(
                total_tokens_processed=200,
                total_requests_processed=10,
                expert_activation_counts={0: 100, 1: 100},
                layer_expert_activation_counts={0: {0: 100}},
                top_k=2,
                num_layers=2,
                num_experts=8,
                is_collecting=True,
                load_balance_score=0.85,
                dp_rank=1,
            ),
        ]
        result = aggregate_snapshots(snapshots)
        assert result.total_tokens_processed == 300
        assert result.total_requests_processed == 15
        assert result.expert_activation_counts == {0: 150, 1: 150}
        assert result.dp_rank is None  # Aggregated result has no rank

    def test_aggregate_load_balance_recomputed(self):
        snapshots = [
            ExpertStatsSnapshot(
                total_tokens_processed=100,
                total_requests_processed=5,
                expert_activation_counts={0: 100},
                layer_expert_activation_counts={},
                top_k=1,
                num_layers=1,
                num_experts=2,
                is_collecting=True,
                load_balance_score=0.5,  # Should be ignored
            ),
            ExpertStatsSnapshot(
                total_tokens_processed=100,
                total_requests_processed=5,
                expert_activation_counts={1: 100},
                layer_expert_activation_counts={},
                top_k=1,
                num_layers=1,
                num_experts=2,
                is_collecting=True,
                load_balance_score=0.5,  # Should be ignored
            ),
        ]
        result = aggregate_snapshots(snapshots)
        # After aggregation: {0: 100, 1: 100} - perfectly balanced
        assert result.load_balance_score == 1.0


class TestSnapshotToDict:
    """Tests for snapshot_to_dict function."""

    def test_basic_conversion(self):
        snapshot = ExpertStatsSnapshot(
            total_tokens_processed=100,
            total_requests_processed=5,
            expert_activation_counts={0: 50, 1: 30, 2: 20},
            layer_expert_activation_counts={0: {0: 50}, 1: {1: 30}},
            top_k=2,
            num_layers=2,
            num_experts=8,
            is_collecting=True,
            load_balance_score=0.85,
        )
        result = snapshot_to_dict(snapshot)

        assert result["total_tokens_processed"] == 100
        assert result["total_requests_processed"] == 5
        assert result["is_collecting"] is True
        assert result["num_experts"] == 8
        assert result["num_layers"] == 2
        assert result["top_k"] == 2
        assert result["load_balance_score"] == 0.85
        # Keys should be strings for JSON
        assert "0" in result["expert_activation_counts"]
        assert result["expert_activation_counts"]["0"] == 50

    def test_most_activated_experts(self):
        snapshot = ExpertStatsSnapshot(
            total_tokens_processed=100,
            total_requests_processed=5,
            expert_activation_counts={0: 50, 1: 30, 2: 20},
            layer_expert_activation_counts={},
            top_k=2,
            num_layers=2,
            num_experts=8,
            is_collecting=True,
            load_balance_score=0.85,
        )
        result = snapshot_to_dict(snapshot, limit=2)

        assert len(result["most_activated_experts"]) == 2
        # Should be sorted by count descending
        assert result["most_activated_experts"][0]["expert_id"] == 0
        assert result["most_activated_experts"][0]["count"] == 50
        assert result["most_activated_experts"][0]["percentage"] == 50.0

    def test_least_activated_experts(self):
        snapshot = ExpertStatsSnapshot(
            total_tokens_processed=100,
            total_requests_processed=5,
            expert_activation_counts={0: 50, 1: 30, 2: 20},
            layer_expert_activation_counts={},
            top_k=2,
            num_layers=2,
            num_experts=8,
            is_collecting=True,
            load_balance_score=0.85,
        )
        result = snapshot_to_dict(snapshot, limit=2)

        assert len(result["least_activated_experts"]) == 2
        # Should be sorted by count ascending
        assert result["least_activated_experts"][0]["expert_id"] == 2
        assert result["least_activated_experts"][0]["count"] == 20

    def test_include_zeros(self):
        snapshot = ExpertStatsSnapshot(
            total_tokens_processed=100,
            total_requests_processed=5,
            expert_activation_counts={0: 50},
            layer_expert_activation_counts={},
            top_k=2,
            num_layers=2,
            num_experts=4,
            is_collecting=True,
            load_balance_score=0.85,
        )
        result = snapshot_to_dict(snapshot, limit=4, include_zeros=True)

        # Should include expert 1, 2, 3 with count 0
        assert len(result["most_activated_experts"]) == 4
        assert len(result["least_activated_experts"]) == 4

    def test_limit_clamping(self):
        snapshot = ExpertStatsSnapshot(
            total_tokens_processed=100,
            total_requests_processed=5,
            expert_activation_counts={i: 10 for i in range(10)},
            layer_expert_activation_counts={},
            top_k=2,
            num_layers=2,
            num_experts=16,
            is_collecting=True,
            load_balance_score=0.85,
        )
        # limit > 100 should be clamped to 100
        result = snapshot_to_dict(snapshot, limit=200)
        assert len(result["most_activated_experts"]) <= 100

        # limit < 1 should be clamped to 1
        result = snapshot_to_dict(snapshot, limit=0)
        assert len(result["most_activated_experts"]) == 1

    def test_layer_counts_string_keys(self):
        snapshot = ExpertStatsSnapshot(
            total_tokens_processed=100,
            total_requests_processed=5,
            expert_activation_counts={0: 50},
            layer_expert_activation_counts={0: {0: 50}, 1: {1: 30}},
            top_k=2,
            num_layers=2,
            num_experts=8,
            is_collecting=True,
            load_balance_score=0.85,
        )
        result = snapshot_to_dict(snapshot)

        # Layer keys should be strings
        assert "0" in result["layer_expert_activation_counts"]
        assert "1" in result["layer_expert_activation_counts"]