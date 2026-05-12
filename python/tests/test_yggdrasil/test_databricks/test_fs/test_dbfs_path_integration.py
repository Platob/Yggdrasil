"""Live-integration tests for :class:`DBFSPath`.

Skipped unless ``DATABRICKS_HOST`` (and matching credentials) are
exported via the standard SDK env vars. Each test creates a unique
sub-directory under :envvar:`DATABRICKS_INTEGRATION_DBFS_DIR`
(default ``/dbfs/tmp/yggdrasil-integration``) and tears it down
on the way out — so concurrent runs don't collide and a partial
failure leaves at most one orphan tree behind.
"""

from __future__ import annotations

import os
import secrets
from typing import ClassVar

from yggdrasil.databricks.fs import DBFSPath
from yggdrasil.io.io_stats import IOKind

from .. import DatabricksIntegrationCase


__all__ = ["TestDBFSPathIntegration"]


class TestDBFSPathIntegration(DatabricksIntegrationCase):
    """Round-trip a fresh ``/dbfs/tmp/...`` directory against the
    real DBFS API: mkdir → write → stat → read → list → delete."""

    base_root: ClassVar[str]
    root: ClassVar[DBFSPath]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.base_root = os.environ.get(
            "DATABRICKS_INTEGRATION_DBFS_DIR",
            "/dbfs/tmp/yggdrasil-integration",
        ).rstrip("/")
        cls.root = DBFSPath(
            f"{cls.base_root}/run-{secrets.token_hex(4)}",
            client=cls.client,
        )
        cls.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            cls.root.remove(recursive=True, missing_ok=True)
        finally:
            super().tearDownClass()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def test_round_trip(self) -> None:
        path = self.root / "hello.bin"
        payload = b"hello-dbfs-" + secrets.token_bytes(8)
        path.write_bytes(payload)

        stat = path._stat_uncached()
        self.assertEqual(stat.kind, IOKind.FILE)
        self.assertEqual(stat.size, len(payload))
        self.assertEqual(path.read_bytes(), payload)

    def test_iterdir_finds_written_child(self) -> None:
        path = self.root / "listing" / "f.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

        children = list(path.parent.iterdir())
        full_paths = {c.full_path() for c in children}
        self.assertIn(path.full_path(), full_paths)

    def test_unlink_then_stat_missing(self) -> None:
        path = self.root / "to-delete.bin"
        path.write_bytes(b"bye")
        path.unlink()
        path._invalidate_stat_cache()
        self.assertIs(path._stat_uncached().kind, IOKind.MISSING)

    def test_open_context(self):
        with (self.root / "context.txt").open("wb") as f:
            f.write(b"hello context")

        with (self.root / "context.txt").open("rb") as f:
            content = f.read()
            self.assertEqual(content, b"hello context")
