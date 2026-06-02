"""Shared scaffolding for the Databricks filesystem **integration** tests.

Every backend — :class:`VolumePath`, :class:`WorkspacePath`,
:class:`DBFSPath` — is exercised against a *real* workspace (the
"prefer real calls" contract; no mocks). They all need the same three
things, which used to be hand-rolled three different ways per file:

* a writable, per-run scratch directory that is torn down on the way out,
* a few cold-handle / sample-data helpers, and
* a backend-call counter so chattiness regressions are caught.

:class:`FsIntegrationCase` provides all of that; a subclass only has to
mint ``cls.root`` (a provisioned, writable directory) in
``setUpClass``. :class:`FsRoundTripMixin` then layers the
backend-agnostic CRUD / remove / open contract on top, so each backend's
integration file is just "provision a root + inherit the contract".

Provisioning defaults are unified on ``trading`` / ``unittest`` / ``tmp``
(override via :envvar:`DATABRICKS_INTEGRATION_CATALOG` / ``_SCHEMA`` /
``_VOLUME` for volumes, ``_WORKSPACE_DIR`` / ``_DBFS_DIR`` for the
others). A permission error while provisioning degrades to a skip rather
than a failure, so the suite stays green on a workspace the test identity
can only partially write to.
"""
from __future__ import annotations

import collections
import secrets
import unittest
from contextlib import contextmanager
from typing import ClassVar

from yggdrasil.io.io_stats import IOKind

from .. import DatabricksIntegrationCase


__all__ = ["FsIntegrationCase", "FsRoundTripMixin"]


class FsIntegrationCase(DatabricksIntegrationCase):
    """Base for path integration tests: a provisioned scratch ``root``.

    Subclasses set ``cls.root`` to a writable directory in
    ``setUpClass`` (after ``super().setUpClass()``); teardown removes it
    recursively. ``ext`` is the leaf suffix used by the shared
    round-trip body so each backend reads naturally (``.bin`` vs
    ``.txt``). ``checks_size`` gates the ``stat().size`` assertion —
    the Workspace API doesn't report a byte size for every object.
    """

    root: ClassVar
    ext: ClassVar[str] = "bin"
    checks_size: ClassVar[bool] = True

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            root = getattr(cls, "root", None)
            if root is not None:
                root.remove(recursive=True, missing_ok=True)
        finally:
            super().tearDownClass()

    # -- helpers -------------------------------------------------------
    def _fresh(self, name: str):
        """A child handle with no warmed caches, so call counts and
        ``stat`` reflect the operation under test, not a prior one."""
        child = self.root / name
        child.invalidate_singleton()
        return child

    @contextmanager
    def _count(self):
        """Count :meth:`HTTPSession.fetch` calls by HTTP method for the
        wrapped block — the single transport every Files/DBFS/Workspace
        API op funnels through."""
        from yggdrasil.http_.session import HTTPSession

        calls: collections.Counter = collections.Counter()
        orig = HTTPSession.fetch

        def wrapper(s, method, url, **k):
            calls[method] += 1
            return orig(s, method, url, **k)

        HTTPSession.fetch = wrapper
        try:
            yield calls
        finally:
            HTTPSession.fetch = orig

    @staticmethod
    def _table():
        """A small, typed Arrow table reused by the tabular round-trips."""
        import pyarrow as pa

        return pa.table({
            "id": pa.array([1, 2, 3], pa.int64()),
            "v": pa.array([1.5, 2.5, 3.5], pa.float64()),
            "g": pa.array(["a", "b", "a"], pa.string()),
        })


class FsRoundTripMixin:
    """Backend-agnostic CRUD / remove / open contract.

    Mixed into one ``TestCase`` per backend alongside
    :class:`FsIntegrationCase`; every method drives only ``self.root``
    so the same body validates every path implementation identically.
    """

    def test_round_trip(self) -> None:
        path = self.root / f"hello.{self.ext}"
        payload = b"hello-" + secrets.token_bytes(8)
        path.write_bytes(payload)

        stat = path._stat_uncached()
        self.assertEqual(stat.kind, IOKind.FILE)
        if self.checks_size:
            self.assertEqual(stat.size, len(payload))
        self.assertEqual(path.read_bytes(), payload)

    def test_iterdir_finds_written_child(self) -> None:
        path = self.root / "listing" / f"f.{self.ext}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

        full_paths = {c.full_path() for c in path.parent.iterdir()}
        self.assertIn(path.full_path(), full_paths)

    def test_unlink_then_stat_missing(self) -> None:
        path = self.root / f"to-delete.{self.ext}"
        path.write_bytes(b"bye")
        path.unlink()
        path.invalidate_singleton()
        self.assertIs(path._stat_uncached().kind, IOKind.MISSING)

    def test_open_context(self) -> None:
        path = self.root / "context.txt"
        with path.open("wb") as f:
            f.write(b"hello context")
        with self._fresh("context.txt").open("rb") as f:
            self.assertEqual(f.read(), b"hello context")

    def test_remove_directory_with_contents_recursive(self) -> None:
        """``remove(recursive=True)`` clears every entry under the
        directory and then the directory itself — the teardown contract."""
        sub = self.root / "rm-with-contents"
        (sub / f"a.{self.ext}").parent.mkdir(parents=True, exist_ok=True)
        (sub / f"a.{self.ext}").write_bytes(b"a")
        (sub / f"b.{self.ext}").write_bytes(b"b")
        (sub / "nested" / f"c.{self.ext}").parent.mkdir(
            parents=True, exist_ok=True,
        )
        (sub / "nested" / f"c.{self.ext}").write_bytes(b"c")

        sub.remove(recursive=True, missing_ok=False)
        sub.invalidate_singleton()

        self.assertIs(sub._stat_uncached().kind, IOKind.MISSING)
        self.assertIs(self.root._stat_uncached().kind, IOKind.DIRECTORY)

    def test_remove_root_recursive_then_recreate(self) -> None:
        """``remove(recursive=True)`` on the run-scoped root drops the
        whole scratch tree; rebuilt afterwards so later tests still have
        ``self.root``."""
        (self.root / f"leaf.{self.ext}").write_bytes(b"leaf")
        deep = self.root / "dir" / f"deep.{self.ext}"
        deep.parent.mkdir(parents=True, exist_ok=True)
        deep.write_bytes(b"deep")

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
