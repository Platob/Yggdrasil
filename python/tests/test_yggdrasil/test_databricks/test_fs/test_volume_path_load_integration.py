"""Concurrency + high-volume live integration for :class:`VolumePath`.

Skipped unless ``DATABRICKS_HOST`` is set *and* the caller exports
``DATABRICKS_INTEGRATION_VOLUME_DIR`` (a writable UC volume) — same gate as
:mod:`test_volume_path_integration`. Exercises the Files-API transport under
parallel load and large payloads:

* many files written then read back concurrently (round-trip under fan-out),
* concurrent range reads of one large object (the ranged-GET path under
  contention),
* a large single-PUT file round-trip verified by sha256,
* a large *streaming* Parquet round-trip (the spill→``_upload_stream`` path),
* (opt-in) a multipart upload above ``MULTIPART_MIN_SIZE``.

Sizes are env-tunable so the default run stays bounded over a real network::

    DATABRICKS_INTEGRATION_CONC_FILES     (default 24)
    DATABRICKS_INTEGRATION_CONC_WORKERS   (default 8)
    DATABRICKS_INTEGRATION_LARGE_MB       (default 32)
    DATABRICKS_INTEGRATION_PARQUET_ROWS   (default 1_000_000)
    DATABRICKS_INTEGRATION_RUN_MULTIPART  (set → run the ≥100 MiB multipart test)
"""
from __future__ import annotations

import hashlib
import os
import secrets
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import ClassVar

from yggdrasil.databricks.fs import VolumePath
from yggdrasil.io.io_stats import IOKind

from .. import DatabricksIntegrationCase

__all__ = ["TestVolumeLoadIntegration"]

_MIB = 1024 * 1024


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


class TestVolumeLoadIntegration(DatabricksIntegrationCase):
    """Parallel + large-payload round-trips through the Files API."""

    base_root: ClassVar[str]
    root: ClassVar[VolumePath]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        base = os.environ.get(
            "DATABRICKS_INTEGRATION_VOLUME_DIR",
            "/Volumes/trading_tgp_dev/unittest/unittest/scratch",
        ).strip()
        if not base:
            raise unittest.SkipTest(
                "DATABRICKS_INTEGRATION_VOLUME_DIR is not set — export a writable "
                "UC volume directory, e.g. /Volumes/<catalog>/<schema>/<volume>/scratch."
            )
        cls.base_root = base.rstrip("/")
        cls.root = VolumePath(
            f"{cls.base_root}/load-{secrets.token_hex(4)}", client=cls.client,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            cls.root.remove(recursive=True, missing_ok=True)
        finally:
            super().tearDownClass()

    # ------------------------------------------------------------------
    # Concurrency
    # ------------------------------------------------------------------

    def test_concurrent_writes_then_reads(self) -> None:
        n = _env_int("DATABRICKS_INTEGRATION_CONC_FILES", 24)
        workers = _env_int("DATABRICKS_INTEGRATION_CONC_WORKERS", 8)
        payloads = {i: secrets.token_bytes(64 * 1024) for i in range(n)}
        # Pre-create the parent so the fan-out tests file concurrency, not a
        # thundering herd of parent-directory creations.
        (self.root / "conc").mkdir(parents=True, exist_ok=True)

        def write(i: int) -> int:
            (self.root / "conc" / f"f-{i}.bin").write_bytes(payloads[i])
            return i

        with ThreadPoolExecutor(max_workers=workers) as ex:
            written = sorted(f.result() for f in as_completed(ex.submit(write, i) for i in range(n)))
        self.assertEqual(written, list(range(n)))

        def read(i: int) -> tuple[int, bytes]:
            return i, (self.root / "conc" / f"f-{i}.bin").read_bytes()

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed(ex.submit(read, i) for i in range(n)):
                i, data = fut.result()
                self.assertEqual(data, payloads[i], f"file {i} corrupted under concurrent IO")

    def test_concurrent_range_reads_of_one_large_file(self) -> None:
        size = 8 * _MIB
        payload = secrets.token_bytes(size)
        p = self.root / "concread" / "big.bin"
        p.write_bytes(payload)
        p.invalidate_singleton()  # cold read — don't ride the write-side cache
        reader = self.root / "concread" / "big.bin"

        windows = [
            (0, 4096),
            (size // 2, 65536),
            (size - 4096, 4096),
            (1234, 4096),
            (size // 3, 1 << 20),
            (size - (1 << 20), 1 << 20),
        ]

        def rd(window: tuple[int, int]) -> tuple[int, int, bytes]:
            off, length = window
            return off, length, bytes(reader.read_mv(length, off))

        # Repeat the window set so many threads hit the ranged-GET / page cache
        # concurrently.
        with ThreadPoolExecutor(max_workers=8) as ex:
            for fut in as_completed(ex.submit(rd, w) for w in windows * 4):
                off, length, got = fut.result()
                self.assertEqual(got, payload[off:off + length], f"range {off}:{off + length}")

    # ------------------------------------------------------------------
    # High volume
    # ------------------------------------------------------------------

    def test_large_file_round_trip(self) -> None:
        mb = _env_int("DATABRICKS_INTEGRATION_LARGE_MB", 32)
        payload = secrets.token_bytes(mb * _MIB)
        digest = hashlib.sha256(payload).hexdigest()
        name = f"{mb}mb.bin"
        p = self.root / "large" / name
        p.write_bytes(payload)
        self.assertEqual(p.stat().size, len(payload))

        p.invalidate_singleton()
        got = (self.root / "large" / name).read_bytes()
        self.assertEqual(len(got), len(payload))
        self.assertEqual(hashlib.sha256(got).hexdigest(), digest)

    def test_large_parquet_streaming_round_trip(self) -> None:
        import pyarrow as pa

        from yggdrasil.io.parquet_file import ParquetFile

        rows = _env_int("DATABRICKS_INTEGRATION_PARQUET_ROWS", 1_000_000)
        table = pa.table({
            "id": pa.array(range(rows), type=pa.int64()),
            "v": pa.array(range(0, rows * 2, 2), type=pa.int64()),
        })
        p = self.root / "parquet" / "big.parquet"
        # Goes through arrow_output_stream → temp-file spill → _upload_stream,
        # so the encoded payload is streamed, not materialised whole.
        ParquetFile(holder=p).write_arrow_table(table)
        self.assertEqual(p.stat().kind, IOKind.FILE)

        p.invalidate_singleton()
        back = ParquetFile(holder=self.root / "parquet" / "big.parquet").read_arrow_table()
        self.assertEqual(back.num_rows, rows)
        self.assertEqual(back.column("v").to_pylist()[:3], [0, 2, 4])

    def test_multipart_upload_round_trip(self) -> None:
        # Above the single-PUT ceiling → presigned concurrent multipart parts.
        size = VolumePath.MULTIPART_MIN_SIZE + _MIB
        payload = secrets.token_bytes(size)
        digest = hashlib.sha256(payload).hexdigest()
        p = self.root / "multipart" / "huge.bin"
        p.write_bytes(payload)
        self.assertEqual(p.stat().size, size)

        p.invalidate_singleton()
        got = (self.root / "multipart" / "huge.bin").read_bytes()
        self.assertEqual(hashlib.sha256(got).hexdigest(), digest)
