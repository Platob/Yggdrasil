"""Live-integration tests for :class:`WorkspacePath`.

Skipped unless ``DATABRICKS_HOST`` is set *and* the caller exports
:envvar:`DATABRICKS_INTEGRATION_WORKSPACE_DIR` (e.g.
``/Workspace/Users/me@example.com/integration``) — Workspace tests
need a directory the running identity can mutate, and there's no
safe global default. Each test creates a unique sub-directory and
removes it afterwards.
"""

from __future__ import annotations

import os
import secrets
import unittest
from typing import ClassVar

from yggdrasil.databricks.fs import WorkspacePath
from yggdrasil.io.io_stats import IOKind

from .. import DatabricksIntegrationCase


__all__ = ["TestWorkspacePathIntegration"]


class TestWorkspacePathIntegration(DatabricksIntegrationCase):

    base_root: ClassVar[str]
    root: ClassVar[WorkspacePath]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        base = os.environ.get("DATABRICKS_INTEGRATION_WORKSPACE_DIR", "").strip()
        if not base:
            raise unittest.SkipTest(
                "DATABRICKS_INTEGRATION_WORKSPACE_DIR is not set — skipping. "
                "Export it to a writable Workspace directory, e.g. "
                "/Workspace/Users/<me>/integration."
            )
        cls.base_root = base.rstrip("/")
        cls.root = WorkspacePath(
            f"{cls.base_root}/run-{secrets.token_hex(4)}",
            workspace=cls.workspace,
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
        path = self.root / "hello.txt"
        payload = b"hello-workspace-" + secrets.token_bytes(8)
        path.write_bytes(payload)

        stat = path._stat_uncached()
        self.assertEqual(stat.kind, IOKind.FILE)
        self.assertEqual(path.read_bytes(), payload)

    def test_iterdir_finds_written_child(self) -> None:
        path = self.root / "listing" / "f.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

        children = list(path.parent.iterdir())
        full_paths = {c.full_path() for c in children}
        self.assertIn(path.full_path(), full_paths)

    def test_unlink_then_stat_missing(self) -> None:
        path = self.root / "to-delete.txt"
        path.write_bytes(b"bye")
        path.unlink()
        path._invalidate_stat_cache()
        self.assertIs(path._stat_uncached().kind, IOKind.MISSING)
