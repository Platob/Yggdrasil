"""Unit tests for :meth:`HTTPResponseBatch._scatter_partition_count`.

Pure sizing logic — no Spark required, so this runs everywhere.
"""
from __future__ import annotations

import pytest

from yggdrasil.http_.response_batch import HTTPResponseBatch

_count = HTTPResponseBatch._scatter_partition_count


class TestScatterPartitionCount:

    def test_single_node_one_partition_per_core(self):
        # n_executors == 0 → single node → one partition per task slot.
        # In-partition threads handle the I/O concurrency, so no oversubscribe.
        assert _count(1000, 8, 0) == 8

    def test_multi_node_light_2x_for_straggler_rebalance(self):
        # n_executors > 0 → multi node → light ×2 so the scheduler can
        # rebalance stragglers across machines.
        assert _count(1000, 8, 4) == 16

    def test_multi_node_uses_more_partitions_than_single_node(self):
        # Same cores, dedicated executors → strictly more partitions.
        assert _count(1000, 16, 2) > _count(1000, 16, 0)

    def test_clamped_to_number_of_misses(self):
        # Never spawn more partitions than there are requests to send.
        assert _count(3, 64, 8) == 3

    def test_never_below_one(self):
        # A single miss still yields one partition.
        assert _count(1, 8, 4) == 1

    @pytest.mark.parametrize("n_executors", [0, 1, 16])
    def test_zero_cores_floored_to_one(self, n_executors):
        # Degenerate cluster_cores (0 / negative) is floored to 1 core before
        # oversubscription, so we still produce at least one partition.
        assert _count(1000, 0, n_executors) >= 1

    def test_fallback_default_single_node(self):
        # Mirrors the probe's except-branch defaults (cores=8, executors=0).
        assert _count(1000, 8, 0) == 8

    # -- autoscaling / max workers ------------------------------------------

    def test_autoscaling_sizes_against_max_capacity(self):
        # Only 1 worker (4 cores) warm now, but the cluster can scale to 10
        # workers × 4 cores = 40. Size against the ceiling, ×2 multi-node = 80.
        assert _count(1000, 4, 1, max_executors=10, executor_cores=4) == 80

    def test_autoscaling_from_zero_treated_as_multi_node(self):
        # No executors registered yet, but max_workers > 0 → multi-node, sized
        # against the ceiling (8 × 4 = 32) at ×2 = 64, not the single-node ×1.
        assert _count(1000, 1, 0, max_executors=8, executor_cores=4) == 64

    def test_max_capacity_below_current_keeps_current(self):
        # If the live defaultParallelism already exceeds the computed ceiling
        # (stale/partial conf), never shrink below current cores.
        assert _count(1000, 64, 8, max_executors=2, executor_cores=4) == 128

    def test_max_workers_without_executor_cores_ignored_for_sizing(self):
        # Can't compute a ceiling without per-executor cores, so fall back to
        # current cores — but max_workers > 0 still marks it multi-node (×2).
        assert _count(1000, 8, 0, max_executors=4, executor_cores=0) == 16

    def test_single_node_unaffected_by_zero_max(self):
        # Explicit zero max workers keeps the single-node ×1 factor.
        assert _count(1000, 8, 0, max_executors=0, executor_cores=0) == 8
