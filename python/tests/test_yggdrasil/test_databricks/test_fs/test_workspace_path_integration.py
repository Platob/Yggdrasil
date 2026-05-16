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
        base = os.environ.get(
            "DATABRICKS_INTEGRATION_WORKSPACE_DIR",
            "/Workspace/Users/<me>/integration"
        ).strip()
        if not base:
            raise unittest.SkipTest(
                "DATABRICKS_INTEGRATION_WORKSPACE_DIR is not set — skipping. "
                "Export it to a writable Workspace directory, e.g. "
                f"/Workspace/Users/<me>/integration."
            )
        cls.base_root = base.rstrip("/")
        cls.root = WorkspacePath(
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
        path.invalidate_singleton()
        self.assertIs(path._stat_uncached().kind, IOKind.MISSING)

    def test_open_context(self):
        with (self.root / "context.txt").open("wb") as f:
            f.write(b"hello context")

        with (self.root / "context.txt").open("rb") as f:
            content = f.read()
            self.assertEqual(content, b"hello context")

    # ------------------------------------------------------------------
    # remove() — directory with contents + at the run-scoped root
    # ------------------------------------------------------------------

    def test_remove_directory_with_contents_recursive(self) -> None:
        """``remove(recursive=True)`` drops every entry under the
        directory and then the directory itself."""
        sub = self.root / "rm-with-contents"
        (sub / "a.txt").parent.mkdir(parents=True, exist_ok=True)
        (sub / "a.txt").write_bytes(b"a")
        (sub / "b.txt").write_bytes(b"b")
        (sub / "nested" / "c.txt").parent.mkdir(parents=True, exist_ok=True)
        (sub / "nested" / "c.txt").write_bytes(b"c")

        sub.remove(recursive=True, missing_ok=False)
        sub.invalidate_singleton()

        self.assertIs(sub._stat_uncached().kind, IOKind.MISSING)
        self.assertIs(self.root._stat_uncached().kind, IOKind.DIRECTORY)

    def test_remove_root_recursive_then_recreate(self) -> None:
        """``remove(recursive=True)`` on the run-scoped root drops the
        whole scratch tree. Rebuilt afterwards so the rest of the
        test class can keep using ``self.root``."""
        (self.root / "leaf.txt").write_bytes(b"leaf")
        (self.root / "dir" / "deep.txt").parent.mkdir(
            parents=True, exist_ok=True,
        )
        (self.root / "dir" / "deep.txt").write_bytes(b"deep")

        self.root.remove(recursive=True, missing_ok=False)
        self.root.invalidate_singleton()
        self.assertIs(self.root._stat_uncached().kind, IOKind.MISSING)

        self.root.mkdir(parents=True, exist_ok=True)
        self.assertIs(self.root._stat_uncached().kind, IOKind.DIRECTORY)

    def test_remove_missing_ok_on_ghost_path(self) -> None:
        """``remove(missing_ok=True)`` against a never-created path
        succeeds quietly — the no-op branch teardown relies on."""
        ghost = self.root / "never-created"
        ghost.remove(recursive=True, missing_ok=True)
        ghost.invalidate_singleton()
        self.assertIs(ghost._stat_uncached().kind, IOKind.MISSING)
