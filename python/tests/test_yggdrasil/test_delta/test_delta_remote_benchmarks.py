"""Remote-path benchmarks for Delta operations.

Measures S3 call counts and latency for typical Delta workflows over
the mock S3 backend. Validates that caching optimizations (commit
content cache, listing extension, _last_checkpoint cache) reduce
remote round trips.

Run with:
    python -m pytest tests/test_yggdrasil/test_delta/test_delta_remote_benchmarks.py -v -s
"""
from __future__ import annotations

import time

import pyarrow as pa

from yggdrasil.enums import Mode
from yggdrasil.delta.io import DeltaOptions
from yggdrasil.delta.tests import DeltaTestCase
from tests.test_yggdrasil.test_delta.test_delta_s3 import (
    _InMemoryS3,
    _s3_delta_folder,
)


class _CallCounter:
    """Wraps an _InMemoryS3 to count method calls."""

    def __init__(self, s3: _InMemoryS3) -> None:
        self.s3 = s3
        self.counts: dict[str, int] = {}
        self._wrap("head_object")
        self._wrap("get_object")
        self._wrap("put_object")
        self._wrap("delete_object")
        self._wrap("delete_objects")
        orig_pag = s3.get_paginator
        def counted_pag(op):
            p = orig_pag(op)
            orig_paginate = p.paginate
            def counted_paginate(**kw):
                self.counts[f"list({op})"] = self.counts.get(f"list({op})", 0) + 1
                return orig_paginate(**kw)
            p.paginate = counted_paginate
            return p
        s3.get_paginator = counted_pag

    def _wrap(self, name: str) -> None:
        original = getattr(self.s3, name)
        def wrapper(**kw):
            self.counts[name] = self.counts.get(name, 0) + 1
            return original(**kw)
        setattr(self.s3, name, wrapper)

    def reset(self) -> None:
        self.counts.clear()

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def report(self, label: str) -> None:
        parts = [f"{k}={v}" for k, v in sorted(self.counts.items()) if v > 0]
        print(f"  {label}: {self.total} calls ({', '.join(parts)})")


class TestRemoteCallCounts(DeltaTestCase):
    """Verify S3 call efficiency for common Delta workflows."""

    def setUp(self) -> None:
        super().setUp()
        self.s3 = _InMemoryS3()
        self.counter = _CallCounter(self.s3)
        self.bucket = "bench"
        from yggdrasil.io.nested.delta.log import _content_cache
        _content_cache.clear()

    def _folder(self, name: str = "t"):
        return _s3_delta_folder(self.s3, self.bucket, f"test/{name}")

    def test_initial_write_call_count(self) -> None:
        """Initial write: 1 list + 1 head (exists check) + 1 get (_last_ck) + 2 put (parquet + commit)."""
        d = self._folder()
        self.counter.reset()
        d.write_arrow_table(pa.table({"id": [1, 2, 3]}))
        print(f"\n--- Initial write ---")
        self.counter.report("calls")
        self.assertLessEqual(self.counter.total, 6)

    def test_read_after_write_reuses_cache(self) -> None:
        """Read right after write should use cached listing + content."""
        d = self._folder()
        d.write_arrow_table(pa.table({"id": list(range(100))}))
        self.counter.reset()
        out = d.read_arrow_table()
        print(f"\n--- Read after write ({out.num_rows} rows) ---")
        self.counter.report("calls")
        # Should be: 1-2 get_object (commit content cached, only parquet read)
        self.assertLessEqual(self.counter.total, 3)

    def test_second_read_hits_snapshot_cache(self) -> None:
        """Second read on same instance: snapshot cached, only parquet I/O."""
        d = self._folder()
        d.write_arrow_table(pa.table({"id": list(range(50))}))
        d.read_arrow_table()
        self.counter.reset()
        d.read_arrow_table()
        print(f"\n--- Second read (snapshot cached) ---")
        self.counter.report("calls")
        self.assertLessEqual(self.counter.total, 2)

    def test_n_appends_no_relist(self) -> None:
        """N sequential appends should NOT re-list the log directory."""
        d = self._folder()
        d.write_arrow_table(pa.table({"id": [0]}))
        self.counter.reset()
        for i in range(10):
            d.write_arrow_batches(
                pa.table({"id": [i + 1]}).to_batches(),
                options=DeltaOptions(mode=Mode.APPEND, checkpoint_interval=0),
            )
        print(f"\n--- 10 appends ---")
        self.counter.report("calls")
        list_calls = self.counter.counts.get("list(list_objects_v2)", 0)
        # With extend_listing, no re-listing needed between appends
        self.assertLessEqual(list_calls, 1,
            f"Expected ≤1 list call for 10 appends, got {list_calls}")

    def test_read_after_10_appends(self) -> None:
        """Read after 10 appends: commit JSON content cache should help."""
        d = self._folder()
        d.write_arrow_table(pa.table({"id": [0]}))
        for i in range(10):
            d.write_arrow_batches(
                pa.table({"id": [i + 1]}).to_batches(),
                options=DeltaOptions(mode=Mode.APPEND, checkpoint_interval=0),
            )
        self.counter.reset()
        out = d.read_arrow_table()
        print(f"\n--- Read after 10 appends ({out.num_rows} rows) ---")
        self.counter.report("calls")
        # Commit JSON files should be cached from the write phase.
        # Only the 11 parquet files need reading.
        get_calls = self.counter.counts.get("get_object", 0)
        self.assertLessEqual(get_calls, 12)

    def test_fresh_instance_read_with_warm_content_cache(self) -> None:
        """New DeltaFolder on same table: listing cold, content warm."""
        d = self._folder()
        d.write_arrow_table(pa.table({"id": list(range(5))}))
        d.read_arrow_table()  # warm the content cache

        self.counter.reset()
        d2 = _s3_delta_folder(self.s3, self.bucket, "test/t")
        out = d2.read_arrow_table()
        print(f"\n--- Fresh instance, warm cache ({out.num_rows} rows) ---")
        self.counter.report("calls")
        # Content cache hits save commit JSON reads; listing may or
        # may not need a round trip depending on S3Path singleton cache.
        self.assertLessEqual(self.counter.total, 5)

    def test_checkpoint_reduces_commit_reads(self) -> None:
        """After checkpoint, reading a fresh instance should read the
        checkpoint + only the tail commits, not all historical commits."""
        d = self._folder()
        for i in range(11):
            mode = Mode.AUTO if i == 0 else Mode.APPEND
            d.write_arrow_batches(
                pa.table({"id": [i]}).to_batches(),
                options=DeltaOptions(mode=mode, checkpoint_interval=10),
            )

        from yggdrasil.io.nested.delta.log import _content_cache
        _content_cache.clear()

        self.counter.reset()
        d2 = _s3_delta_folder(self.s3, self.bucket, "test/t")
        out = d2.read_arrow_table()
        print(f"\n--- Post-checkpoint fresh read ({out.num_rows} rows) ---")
        self.counter.report("calls")
        # Should read: 1 list, 1 _last_checkpoint, 1 checkpoint parquet,
        # 1 tail commit (v10), 11 data parquets
        get_calls = self.counter.counts.get("get_object", 0)
        self.assertLessEqual(get_calls, 15,
            f"Expected ≤15 gets (checkpoint skips old commits), got {get_calls}")

    def test_overwrite_call_count(self) -> None:
        d = self._folder()
        d.write_arrow_table(pa.table({"id": [1, 2]}))
        self.counter.reset()
        d.write_arrow_table(
            pa.table({"id": [99]}),
            options=DeltaOptions(mode=Mode.OVERWRITE),
        )
        print(f"\n--- Overwrite ---")
        self.counter.report("calls")
        self.assertLessEqual(self.counter.total, 5)


class TestRemoteLatencyBenchmark(DeltaTestCase):
    """Wall-clock benchmarks over mock S3 (no network latency,
    measures pure overhead of the Delta protocol)."""

    def setUp(self) -> None:
        super().setUp()
        self.s3 = _InMemoryS3()
        self.bucket = "bench"
        from yggdrasil.io.nested.delta.log import _content_cache
        _content_cache.clear()

    def _folder(self, name: str = "t"):
        return _s3_delta_folder(self.s3, self.bucket, f"perf/{name}")

    def _time(self, fn, label: str, repeat: int = 1) -> float:
        times = []
        for _ in range(repeat):
            start = time.perf_counter()
            fn()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        avg = sum(times) / len(times)
        print(f"  {label}: {avg:.4f}s (avg of {repeat})")
        return avg

    def test_write_100_rows(self) -> None:
        print("\n--- Write 100 rows over mock S3 ---")
        t = pa.table({"id": list(range(100)), "val": [f"v{i}" for i in range(100)]})

        def write():
            d = _s3_delta_folder(_InMemoryS3(), "b", f"t{time.time_ns()}/")
            d.write_arrow_table(t)
        self._time(write, "write", repeat=5)

    def test_read_100_rows(self) -> None:
        d = self._folder()
        d.write_arrow_table(pa.table({"id": list(range(100))}))

        print("\n--- Read 100 rows over mock S3 ---")
        def read():
            d.refresh()
            return d.read_arrow_table()
        self._time(read, "cold read", repeat=5)

        def cached_read():
            return d.read_arrow_table()
        self._time(cached_read, "cached read", repeat=10)

    def test_20_appends_then_read(self) -> None:
        print("\n--- 20 appends then read over mock S3 ---")
        d = self._folder()

        def appends():
            for i in range(20):
                mode = Mode.AUTO if i == 0 else Mode.APPEND
                d.write_arrow_batches(
                    pa.table({"id": [i]}).to_batches(),
                    options=DeltaOptions(mode=mode, checkpoint_interval=0),
                )
        self._time(appends, "20 appends", repeat=1)

        def read_all():
            return d.read_arrow_table()
        self._time(read_all, "read 20 files", repeat=3)

    def test_checkpoint_write_and_replay(self) -> None:
        print("\n--- 11 writes + checkpoint + replay over mock S3 ---")
        def cycle():
            s3 = _InMemoryS3()
            d = _s3_delta_folder(s3, "b", f"ck{time.time_ns()}/")
            for i in range(11):
                mode = Mode.AUTO if i == 0 else Mode.APPEND
                d.write_arrow_batches(
                    pa.table({"id": list(range(i*10, (i+1)*10))}).to_batches(),
                    options=DeltaOptions(mode=mode, checkpoint_interval=10),
                )
            d2 = _s3_delta_folder(s3, "b", f"ck{time.time_ns() - 1}/")
            # Use same s3 backend, different prefix would miss — use original
            return d.read_arrow_table()
        # Just time the write phase
        s3 = _InMemoryS3()
        d = _s3_delta_folder(s3, "b", "ck/")
        def writes():
            for i in range(11):
                mode = Mode.AUTO if i == 0 else Mode.APPEND
                d.write_arrow_batches(
                    pa.table({"id": list(range(i*10, (i+1)*10))}).to_batches(),
                    options=DeltaOptions(mode=mode, checkpoint_interval=10),
                )
        self._time(writes, "11 writes + checkpoint", repeat=1)

        def replay():
            d2 = _s3_delta_folder(s3, "b", "ck/")
            return d2.read_arrow_table()
        self._time(replay, "checkpoint replay", repeat=3)
