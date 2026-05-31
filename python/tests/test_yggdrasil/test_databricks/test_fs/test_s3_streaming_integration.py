"""S3 streaming-upload live integration, reached through Databricks storage.

Exercises the optimized S3 upload — spill the Arrow/Parquet encode to a temp
file and stream it to S3 via boto's managed transfer (``upload_fileobj``), so a
multi-GB write stays bounded in memory — against real cloud storage vended by
Databricks credentials:

* a UC **volume**'s backing S3 storage (``Volume.storage_path``), and
* a UC **table**'s backing S3 storage (``Table.storage_path``).

Skipped unless ``DATABRICKS_HOST`` + ``DATABRICKS_INTEGRATION_VOLUME_DIR`` are
set. The table case additionally needs ``DATABRICKS_INTEGRATION_TABLE`` (a full
``catalog.schema.table`` whose storage is S3). Each test writes under a unique
sub-prefix and removes it on the way out. Sizes are env-tunable
(``DATABRICKS_INTEGRATION_S3_LARGE_MB`` default 16,
``DATABRICKS_INTEGRATION_S3_PARQUET_ROWS`` default 500_000).
"""
from __future__ import annotations

import os
import secrets
import unittest

from yggdrasil.databricks.fs import VolumePath
from yggdrasil.enums import Mode

from .. import DatabricksIntegrationCase

__all__ = ["TestS3StreamingViaDatabricks"]

_MIB = 1024 * 1024


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _is_s3(path) -> bool:
    try:
        return path is not None and str(path.full_path()).startswith("s3")
    except Exception:
        return False


class TestS3StreamingViaDatabricks(DatabricksIntegrationCase):
    """The S3 ``_upload_stream`` path against real cloud storage, addressed via
    a UC volume's and a UC table's storage locations."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.volume_dir = os.environ.get(
            "DATABRICKS_INTEGRATION_VOLUME_DIR",
            "/Volumes/trading_tgp_dev/unittest/unittest/scratch",
        ).strip().rstrip("/")
        if not cls.volume_dir:
            raise unittest.SkipTest("DATABRICKS_INTEGRATION_VOLUME_DIR is not set.")

    def _exercise_s3_streaming(self, s3root) -> None:
        import pyarrow as pa

        from yggdrasil.io.parquet_file import ParquetFile

        base = s3root / f"ygg-s3-stream-{secrets.token_hex(4)}"
        blob = base / "blob.bin"
        parquet = base / "data.parquet"
        try:
            # Large raw object: boto managed transfer (multipart for big).
            mb = _env_int("DATABRICKS_INTEGRATION_S3_LARGE_MB", 16)
            payload = secrets.token_bytes(mb * _MIB)
            blob.write_bytes(payload)
            self.assertEqual(blob.read_bytes(), payload)

            # Streaming Parquet: arrow_output_stream spills to a temp file,
            # _upload_stream hands it to upload_fileobj — no whole-payload
            # materialisation.
            rows = _env_int("DATABRICKS_INTEGRATION_S3_PARQUET_ROWS", 500_000)
            table = pa.table({"id": pa.array(range(rows), type=pa.int64())})
            ParquetFile(holder=parquet).write_arrow_table(table)
            parquet.invalidate_singleton()  # cold read, don't ride the write cache
            back = ParquetFile(holder=parquet).read_arrow_table()
            self.assertEqual(back.num_rows, rows)
        finally:
            for p in (blob, parquet):
                try:
                    p.unlink()
                except Exception:
                    pass

    def test_volume_backing_s3_streaming(self) -> None:
        volume = VolumePath(self.volume_dir, client=self.client).volume
        s3root = volume.storage_path(mode=Mode.OVERWRITE)
        if not _is_s3(s3root):
            raise unittest.SkipTest("Volume is not backed by S3 — skipping.")
        self._exercise_s3_streaming(s3root)

    def test_table_backing_s3_streaming(self) -> None:
        full = os.environ.get("DATABRICKS_INTEGRATION_TABLE", "").strip()
        if not full:
            raise unittest.SkipTest(
                "DATABRICKS_INTEGRATION_TABLE is not set — export a full "
                "catalog.schema.table backed by S3 to run this."
            )
        s3root = self.client.tables.table(full).storage_path()
        if not _is_s3(s3root):
            raise unittest.SkipTest("Table is not backed by S3 — skipping.")
        self._exercise_s3_streaming(s3root / "ygg-s3-test")
