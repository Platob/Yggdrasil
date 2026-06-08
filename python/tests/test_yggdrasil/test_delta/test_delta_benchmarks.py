"""Benchmarks comparing yggdrasil DeltaFolder with deltalake.

Measures read/write throughput and latency for various table sizes
and configurations. Not a unit test — run with:

    python -m pytest tests/test_yggdrasil/test_delta/test_delta_benchmarks.py -v -s
"""
from __future__ import annotations

import os
import time
import unittest

from yggdrasil.enums import Mode
from yggdrasil.delta.io import DeltaOptions
from yggdrasil.delta.tests import DeltaTestCase


def _has_deltalake() -> bool:
    try:
        import deltalake  # noqa: F401
        return True
    except ImportError:
        return False


def _generate_table(pa, n: int, n_cols: int = 5):
    """Generate a pyarrow table with n rows and n_cols columns."""
    import random
    data = {"id": list(range(n))}
    for c in range(n_cols - 1):
        data[f"col_{c}"] = [random.random() for _ in range(n)]
    return pa.table(data)


class _BenchmarkMixin:
    """Shared benchmarking helpers."""

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


@unittest.skipUnless(_has_deltalake(), "deltalake package not installed")
class TestWriteBenchmark(DeltaTestCase, _BenchmarkMixin):
    """Write throughput benchmark."""

    def test_write_small_table(self) -> None:
        """1K rows write comparison."""
        import deltalake

        t = _generate_table(self.pa, 1000)
        print("\n--- Write 1K rows ---")

        def ygg_write():
            p = str(self.tmp_path / f"ygg_{time.time_ns()}")
            os.makedirs(p, exist_ok=True)
            from yggdrasil.io.delta.delta_folder import DeltaFolder
            d = DeltaFolder(path=p)
            d.write_arrow_table(t)

        def dl_write():
            p = str(self.tmp_path / f"dl_{time.time_ns()}")
            deltalake.write_deltalake(p, t)

        ygg_time = self._time(ygg_write, "yggdrasil", repeat=3)
        dl_time = self._time(dl_write, "deltalake", repeat=3)
        ratio = ygg_time / dl_time if dl_time > 0 else float("inf")
        print(f"  ratio (ygg/dl): {ratio:.2f}x")

    def test_write_medium_table(self) -> None:
        """100K rows write comparison."""
        import deltalake

        t = _generate_table(self.pa, 100_000)
        print("\n--- Write 100K rows ---")

        def ygg_write():
            p = str(self.tmp_path / f"ygg_{time.time_ns()}")
            os.makedirs(p, exist_ok=True)
            from yggdrasil.io.delta.delta_folder import DeltaFolder
            d = DeltaFolder(path=p)
            d.write_arrow_table(t)

        def dl_write():
            p = str(self.tmp_path / f"dl_{time.time_ns()}")
            deltalake.write_deltalake(p, t)

        ygg_time = self._time(ygg_write, "yggdrasil", repeat=3)
        dl_time = self._time(dl_write, "deltalake", repeat=3)
        ratio = ygg_time / dl_time if dl_time > 0 else float("inf")
        print(f"  ratio (ygg/dl): {ratio:.2f}x")

    def test_write_large_table(self) -> None:
        """1M rows write comparison."""
        import deltalake

        t = _generate_table(self.pa, 1_000_000, n_cols=10)
        print("\n--- Write 1M rows ---")

        def ygg_write():
            p = str(self.tmp_path / f"ygg_{time.time_ns()}")
            os.makedirs(p, exist_ok=True)
            from yggdrasil.io.delta.delta_folder import DeltaFolder
            d = DeltaFolder(path=p)
            d.write_arrow_table(t)

        def dl_write():
            p = str(self.tmp_path / f"dl_{time.time_ns()}")
            deltalake.write_deltalake(p, t)

        ygg_time = self._time(ygg_write, "yggdrasil", repeat=2)
        dl_time = self._time(dl_write, "deltalake", repeat=2)
        ratio = ygg_time / dl_time if dl_time > 0 else float("inf")
        print(f"  ratio (ygg/dl): {ratio:.2f}x")


@unittest.skipUnless(_has_deltalake(), "deltalake package not installed")
class TestReadBenchmark(DeltaTestCase, _BenchmarkMixin):
    """Read throughput benchmark."""

    def _setup_table(self, n: int, engine: str = "both"):
        import deltalake

        t = _generate_table(self.pa, n)

        ygg_path = str(self.tmp_path / "ygg_read")
        os.makedirs(ygg_path, exist_ok=True)
        from yggdrasil.io.delta.delta_folder import DeltaFolder
        d = DeltaFolder(path=ygg_path)
        d.write_arrow_table(t)

        dl_path = str(self.tmp_path / "dl_read")
        deltalake.write_deltalake(dl_path, t)

        return ygg_path, dl_path

    def test_read_small_table(self) -> None:
        import deltalake
        from yggdrasil.io.delta.delta_folder import DeltaFolder

        ygg_path, dl_path = self._setup_table(1000)
        print("\n--- Read 1K rows ---")

        def ygg_read():
            d = DeltaFolder(path=ygg_path)
            d.refresh()
            return d.read_arrow_table()

        def dl_read():
            dt = deltalake.DeltaTable(dl_path)
            return dt.to_pyarrow_table()

        ygg_time = self._time(ygg_read, "yggdrasil", repeat=5)
        dl_time = self._time(dl_read, "deltalake", repeat=5)
        ratio = ygg_time / dl_time if dl_time > 0 else float("inf")
        print(f"  ratio (ygg/dl): {ratio:.2f}x")

    def test_read_medium_table(self) -> None:
        import deltalake
        from yggdrasil.io.delta.delta_folder import DeltaFolder

        ygg_path, dl_path = self._setup_table(100_000)
        print("\n--- Read 100K rows ---")

        def ygg_read():
            d = DeltaFolder(path=ygg_path)
            d.refresh()
            return d.read_arrow_table()

        def dl_read():
            dt = deltalake.DeltaTable(dl_path)
            return dt.to_pyarrow_table()

        ygg_time = self._time(ygg_read, "yggdrasil", repeat=3)
        dl_time = self._time(dl_read, "deltalake", repeat=3)
        ratio = ygg_time / dl_time if dl_time > 0 else float("inf")
        print(f"  ratio (ygg/dl): {ratio:.2f}x")


@unittest.skipUnless(_has_deltalake(), "deltalake package not installed")
class TestAppendBenchmark(DeltaTestCase, _BenchmarkMixin):
    """Append throughput benchmark — many small commits."""

    def test_append_many_commits(self) -> None:
        import deltalake
        from yggdrasil.io.delta.delta_folder import DeltaFolder

        n_commits = 20
        batch_size = 1000
        print(f"\n--- {n_commits} appends of {batch_size} rows ---")

        def ygg_append():
            p = str(self.tmp_path / f"ygg_append_{time.time_ns()}")
            os.makedirs(p, exist_ok=True)
            d = DeltaFolder(path=p)
            for i in range(n_commits):
                t = self.pa.table({"id": list(range(i * batch_size, (i + 1) * batch_size))})
                mode = Mode.AUTO if i == 0 else Mode.APPEND
                d.write_arrow_batches(
                    t.to_batches(),
                    options=DeltaOptions(mode=mode, checkpoint_interval=0),
                )

        def dl_append():
            p = str(self.tmp_path / f"dl_append_{time.time_ns()}")
            for i in range(n_commits):
                t = self.pa.table({"id": list(range(i * batch_size, (i + 1) * batch_size))})
                mode = "error" if i == 0 else "append"
                deltalake.write_deltalake(p, t, mode=mode)

        ygg_time = self._time(ygg_append, "yggdrasil", repeat=2)
        dl_time = self._time(dl_append, "deltalake", repeat=2)
        ratio = ygg_time / dl_time if dl_time > 0 else float("inf")
        print(f"  ratio (ygg/dl): {ratio:.2f}x")


@unittest.skipUnless(_has_deltalake(), "deltalake package not installed")
class TestCheckpointBenchmark(DeltaTestCase, _BenchmarkMixin):
    """Checkpoint write + replay benchmark."""

    def test_checkpoint_v1_write_and_replay(self) -> None:
        from yggdrasil.io.delta.delta_folder import DeltaFolder

        print("\n--- V1 Checkpoint (10 commits + checkpoint + replay) ---")

        def run():
            p = str(self.tmp_path / f"ckpt_v1_{time.time_ns()}")
            os.makedirs(p, exist_ok=True)
            d = DeltaFolder(path=p)
            for i in range(11):
                t = self.pa.table({"id": list(range(i * 100, (i + 1) * 100))})
                mode = Mode.AUTO if i == 0 else Mode.APPEND
                d.write_arrow_batches(
                    t.to_batches(),
                    options=DeltaOptions(mode=mode, checkpoint_interval=10, checkpoint_kind="v1"),
                )
            d2 = DeltaFolder(path=p)
            return d2.read_arrow_table()

        self._time(run, "v1 checkpoint cycle", repeat=3)

    def test_checkpoint_v2_write_and_replay(self) -> None:
        from yggdrasil.io.delta.delta_folder import DeltaFolder

        print("\n--- V2 Checkpoint (10 commits + checkpoint + replay) ---")

        def run():
            p = str(self.tmp_path / f"ckpt_v2_{time.time_ns()}")
            os.makedirs(p, exist_ok=True)
            d = DeltaFolder(path=p)
            for i in range(11):
                t = self.pa.table({"id": list(range(i * 100, (i + 1) * 100))})
                mode = Mode.AUTO if i == 0 else Mode.APPEND
                d.write_arrow_batches(
                    t.to_batches(),
                    options=DeltaOptions(mode=mode, checkpoint_interval=10, checkpoint_kind="v2"),
                )
            d2 = DeltaFolder(path=p)
            return d2.read_arrow_table()

        self._time(run, "v2 checkpoint cycle", repeat=3)


class TestDeletionVectorBenchmark(DeltaTestCase, _BenchmarkMixin):
    """DV encode/decode benchmark."""

    def test_dv_encode_decode_small(self) -> None:
        from yggdrasil.io.delta.deletion_vector import (
            _encode_dv_payload,
            _decode_payload,
        )

        rows = list(range(0, 100, 2))
        print("\n--- DV encode/decode 50 rows ---")

        def encode():
            return _encode_dv_payload(rows)

        def decode():
            payload = _encode_dv_payload(rows)
            return _decode_payload(payload)

        self._time(encode, "encode", repeat=100)
        self._time(decode, "decode", repeat=100)

    def test_dv_encode_decode_large(self) -> None:
        from yggdrasil.io.delta.deletion_vector import (
            _encode_dv_payload,
            _decode_payload,
        )

        rows = list(range(0, 50000, 3))
        print(f"\n--- DV encode/decode {len(rows)} rows (Roaring) ---")

        def encode():
            return _encode_dv_payload(rows)

        def decode():
            payload = _encode_dv_payload(rows)
            return _decode_payload(payload)

        self._time(encode, "encode", repeat=10)
        self._time(decode, "decode", repeat=10)

    def test_dv_mask_batch_benchmark(self) -> None:
        from yggdrasil.io.delta.deletion_vector import (
            DeletionVector,
            DeletionVectorDescriptor,
            mask_batch_with_dv,
        )

        n = 100_000
        batch = self.pa.record_batch({"id": list(range(n))})
        descriptor = DeletionVectorDescriptor(
            storage_type="i", path_or_inline_dv="", size_in_bytes=0,
        )
        dv = DeletionVector(
            descriptor=descriptor,
            deleted_rows=frozenset(range(0, n, 10)),
        )
        print(f"\n--- Mask batch {n} rows, 10% deleted ---")

        def mask():
            return mask_batch_with_dv(batch, dv)

        self._time(mask, "mask_batch", repeat=10)
