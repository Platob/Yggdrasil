"""Remote-path call-count + latency benchmarks for Delta operations.

Drives the Delta protocol over :class:`MemRemotePath` (a dict-backed
:class:`~yggdrasil.path.remote_path.RemotePath`) and asserts that the caching
optimisations (commit-content cache, listing extension, ``_last_checkpoint``
cache, checkpoint replay) cut yggdrasil's *own* remote round-trips — the
``read`` / ``list`` / ``stat`` IO primitives the Path layer issues, regardless
of whether the backend is S3, a UC volume, or anything else. No boto3 wire is
simulated: ``MemRemotePath.calls`` is the source of truth.

Run with:
    python -m pytest tests/test_yggdrasil/test_delta/test_delta_remote_benchmarks.py -v -s
"""
from __future__ import annotations

import time

import pyarrow as pa

from yggdrasil.enums import Mode
from yggdrasil.delta.io import DeltaOptions
from yggdrasil.delta.tests import DeltaTestCase
from tests.test_yggdrasil.test_delta._mem_remote import MemRemotePath, mem_delta_folder


def _reset_calls() -> None:
    MemRemotePath.calls.clear()


def _calls() -> "dict[str, int]":
    return dict(MemRemotePath.calls)


def _total() -> int:
    return sum(MemRemotePath.calls.values())


def _report(label: str) -> None:
    parts = [f"{k}={v}" for k, v in sorted(MemRemotePath.calls.items()) if v > 0]
    print(f"  {label}: {_total()} io-calls ({', '.join(parts)})")


class TestRemoteCallCounts(DeltaTestCase):
    """Verify remote-IO efficiency for common Delta workflows. Counts the
    Path layer's ``read`` / ``list`` / ``stat`` / ``upload`` primitives."""

    def setUp(self) -> None:
        super().setUp()
        MemRemotePath.reset()
        self.bucket = "bench"
        from yggdrasil.io.delta.log import _content_cache
        _content_cache.clear()

    def _folder(self, name: str = "t"):
        return mem_delta_folder(self.bucket, f"test/{name}_{time.time_ns()}/")

    def test_initial_write_call_count(self) -> None:
        """Initial write stays a handful of IO calls (exists check + last-
        checkpoint probe + parquet/commit uploads)."""
        d = self._folder()
        _reset_calls()
        d.write_arrow_table(pa.table({"id": [1, 2, 3]}))
        _report("initial write")
        self.assertLessEqual(_total(), 12)

    def test_read_after_write_reuses_cache(self) -> None:
        """Read right after write reuses the cached listing + commit content
        — only the parquet bytes are fetched."""
        d = self._folder()
        d.write_arrow_table(pa.table({"id": list(range(100))}))
        _reset_calls()
        out = d.read_arrow_table()
        _report(f"read after write ({out.num_rows} rows)")
        self.assertLessEqual(_calls().get("read", 0), 4)

    def test_second_read_hits_snapshot_cache(self) -> None:
        """Second read on the same instance: snapshot cached, only parquet IO."""
        d = self._folder()
        d.write_arrow_table(pa.table({"id": list(range(50))}))
        d.read_arrow_table()
        _reset_calls()
        d.read_arrow_table()
        _report("second read (snapshot cached)")
        self.assertLessEqual(_calls().get("read", 0), 3)
        self.assertLessEqual(_calls().get("list", 0), 1)

    def test_n_appends_no_relist(self) -> None:
        """N sequential appends should NOT re-list the log directory."""
        d = self._folder()
        d.write_arrow_table(pa.table({"id": [0]}))
        _reset_calls()
        for i in range(10):
            d.write_arrow_batches(
                pa.table({"id": [i + 1]}).to_batches(),
                options=DeltaOptions(mode=Mode.APPEND, checkpoint_interval=0),
            )
        _report("10 appends")
        list_calls = _calls().get("list", 0)
        self.assertLessEqual(
            list_calls, 1,
            f"Expected ≤1 list call for 10 appends, got {list_calls}",
        )

    def test_read_after_10_appends(self) -> None:
        """Read after 10 appends: commit-content cache means only the 11
        parquet files need fetching, not the commit JSONs."""
        d = self._folder()
        d.write_arrow_table(pa.table({"id": [0]}))
        for i in range(10):
            d.write_arrow_batches(
                pa.table({"id": [i + 1]}).to_batches(),
                options=DeltaOptions(mode=Mode.APPEND, checkpoint_interval=0),
            )
        _reset_calls()
        out = d.read_arrow_table()
        _report(f"read after 10 appends ({out.num_rows} rows)")
        # 11 parquet files (≤2 reads each: size probe + body).
        self.assertLessEqual(_calls().get("read", 0), 24)

    def test_fresh_instance_read_with_warm_content_cache(self) -> None:
        """New DeltaFolder on the same table: listing cold, content warm."""
        prefix = f"test/fresh_{time.time_ns()}/"
        d = mem_delta_folder(self.bucket, prefix)
        d.write_arrow_table(pa.table({"id": list(range(5))}))
        d.read_arrow_table()

        _reset_calls()
        d2 = mem_delta_folder(self.bucket, prefix)
        out = d2.read_arrow_table()
        _report(f"fresh instance, warm cache ({out.num_rows} rows)")
        self.assertLessEqual(_total(), 12)

    def test_checkpoint_reduces_commit_reads(self) -> None:
        """After a checkpoint, a fresh read fetches the checkpoint + only the
        tail commits, not every historical commit."""
        prefix = f"test/ck_{time.time_ns()}/"
        d = mem_delta_folder(self.bucket, prefix)
        for i in range(11):
            mode = Mode.AUTO if i == 0 else Mode.APPEND
            d.write_arrow_batches(
                pa.table({"id": [i]}).to_batches(),
                options=DeltaOptions(mode=mode, checkpoint_interval=10),
            )

        from yggdrasil.io.delta.log import _content_cache
        _content_cache.clear()

        _reset_calls()
        d2 = mem_delta_folder(self.bucket, prefix)
        out = d2.read_arrow_table()
        _report(f"post-checkpoint fresh read ({out.num_rows} rows)")
        read_calls = _calls().get("read", 0)
        self.assertLessEqual(
            read_calls, 30,
            f"Expected ≤30 reads (checkpoint skips old commits), got {read_calls}",
        )

    def test_overwrite_call_count(self) -> None:
        d = self._folder()
        d.write_arrow_table(pa.table({"id": [1, 2]}))
        _reset_calls()
        d.write_arrow_table(
            pa.table({"id": [99]}),
            options=DeltaOptions(mode=Mode.OVERWRITE),
        )
        _report("overwrite")
        self.assertLessEqual(_total(), 12)


class TestRemoteLatencyBenchmark(DeltaTestCase):
    """Wall-clock benchmarks over the in-memory remote path (no network
    latency — measures the pure overhead of the Delta protocol)."""

    def setUp(self) -> None:
        super().setUp()
        MemRemotePath.reset()
        self.bucket = "bench"
        from yggdrasil.io.delta.log import _content_cache
        _content_cache.clear()

    def _folder(self, name: str = "t"):
        return mem_delta_folder(self.bucket, f"perf/{name}")

    def _time(self, fn, label: str, repeat: int = 1) -> float:
        times = []
        for _ in range(repeat):
            start = time.perf_counter()
            fn()
            times.append(time.perf_counter() - start)
        avg = sum(times) / len(times)
        print(f"  {label}: {avg:.4f}s (avg of {repeat})")
        return avg

    def test_write_100_rows(self) -> None:
        print("\n--- Write 100 rows over in-memory remote ---")
        t = pa.table({"id": list(range(100)), "val": [f"v{i}" for i in range(100)]})

        def write():
            d = mem_delta_folder("b", f"t{time.time_ns()}/")
            d.write_arrow_table(t)
        self._time(write, "write", repeat=5)

    def test_read_100_rows(self) -> None:
        d = self._folder()
        d.write_arrow_table(pa.table({"id": list(range(100))}))

        print("\n--- Read 100 rows over in-memory remote ---")
        def read():
            d.refresh()
            return d.read_arrow_table()
        self._time(read, "cold read", repeat=5)

        def cached_read():
            return d.read_arrow_table()
        self._time(cached_read, "cached read", repeat=10)

    def test_20_appends_then_read(self) -> None:
        print("\n--- 20 appends then read over in-memory remote ---")
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
        print("\n--- 11 writes + checkpoint + replay over in-memory remote ---")
        d = mem_delta_folder("b", "ck/")

        def writes():
            for i in range(11):
                mode = Mode.AUTO if i == 0 else Mode.APPEND
                d.write_arrow_batches(
                    pa.table({"id": list(range(i * 10, (i + 1) * 10))}).to_batches(),
                    options=DeltaOptions(mode=mode, checkpoint_interval=10),
                )
        self._time(writes, "11 writes + checkpoint", repeat=1)

        def replay():
            d2 = mem_delta_folder("b", "ck/")
            return d2.read_arrow_table()
        self._time(replay, "checkpoint replay", repeat=3)
