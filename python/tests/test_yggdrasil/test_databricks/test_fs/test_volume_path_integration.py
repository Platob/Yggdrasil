"""Live-integration tests for :class:`VolumePath`.

Skipped unless ``DATABRICKS_HOST`` is set *and* the caller exports
:envvar:`DATABRICKS_INTEGRATION_VOLUME_DIR` (e.g.
``/Volumes/main/default/scratch``) — Volume tests need a Unity
Catalog volume the test identity has write access to, and there's
no safe default. Each test builds a unique sub-directory and
removes it on the way out.
"""

from __future__ import annotations

import os
import secrets
import unittest
from typing import ClassVar

from yggdrasil.databricks.fs import VolumePath
from yggdrasil.io.io_stats import IOKind

from .. import DatabricksIntegrationCase


__all__ = ["TestVolumePathIntegration"]


class TestVolumePathIntegration(DatabricksIntegrationCase):
    """Round-trip an existing UC volume sub-directory through the
    Files API."""

    base_root: ClassVar[str]
    root: ClassVar[VolumePath]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        base = os.environ.get(
            "DATABRICKS_INTEGRATION_VOLUME_DIR",
            "/Volumes/trading_tgp_dev/unittest/unittest/scratch"
        ).strip()
        if not base:
            raise unittest.SkipTest(
                "DATABRICKS_INTEGRATION_VOLUME_DIR is not set — skipping. "
                "Export it to a writable UC volume directory, e.g. "
                "/Volumes/<catalog>/<schema>/<volume>/scratch."
            )
        cls.base_root = base.rstrip("/")
        cls.root = VolumePath(
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
        payload = b"hello-volumes-" + secrets.token_bytes(8)
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

    def test_staging_path_round_trip(self) -> None:
        """:meth:`VolumePath.staging_path` is the SQL-engine helper —
        check that the minted path actually round-trips against the
        live Files API."""
        # Borrow the configured root's catalog / schema so the
        # staging helper writes somewhere we have access to.
        parts = self.base_root.lstrip("/").split("/")
        # Expected shape: ``Volumes/<cat>/<sch>/<vol>/...``.
        if len(parts) < 4 or parts[0] != "Volumes":
            self.skipTest(
                f"DATABRICKS_INTEGRATION_VOLUME_DIR={self.base_root!r} is not "
                "a Unity Catalog volume path; skipping staging_path probe."
            )
        catalog, schema = parts[1], parts[2]

        staged = VolumePath.staging_path(
            catalog_name=catalog,
            schema_name=schema,
            resource_name="integration",
            client=self.client,
            temporary=False,
        )
        try:
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(b"staged")
            self.assertEqual(staged.read_bytes(), b"staged")
        finally:
            staged.unlink(missing_ok=True)

    def test_storage_location(self):
        path = self.root / "storage_location.bin"

        assert path.storage_path() == f"/Volumes/{self.catalog}/{self.schema}/integration/storage_location.bin"