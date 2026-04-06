"""Unit tests for DatabricksPath — pure (no API calls) and pathlib-like API.

These run without a Databricks connection; they verify parsing, path algebra,
and the new pathlib.PurePosixPath-compatible properties.
"""
from __future__ import annotations

import unittest

from yggdrasil.databricks.fs.path import (
    DatabricksPath,
    DBFSPath,
    WorkspacePath,
    VolumePath,
    TablePath,
    DatabricksStatResult,
)
from yggdrasil.databricks.fs.path_kind import DatabricksPathKind


# ══════════════════════════════════════════════════════════════════════════
# Parsing
# ══════════════════════════════════════════════════════════════════════════

class TestParsing(unittest.TestCase):
    """Test DatabricksPath.parse produces the right subclass and parts."""

    def test_dbfs_bare(self):
        p = DatabricksPath.parse("/dbfs/tmp/foo")
        self.assertIsInstance(p, DBFSPath)
        self.assertEqual(p.parts, ["tmp", "foo"])
        self.assertEqual(p.kind, DatabricksPathKind.DBFS)

    def test_dbfs_scheme(self):
        p = DatabricksPath.parse("dbfs:/mnt/data/file.csv")
        self.assertIsInstance(p, DBFSPath)
        self.assertEqual(p.parts, ["mnt", "data", "file.csv"])

    def test_workspace_bare(self):
        p = DatabricksPath.parse("/Workspace/Users/me/nb.py")
        self.assertIsInstance(p, WorkspacePath)
        self.assertEqual(p.parts, ["Users", "me", "nb.py"])

    def test_volumes_bare(self):
        p = DatabricksPath.parse("/Volumes/cat/schema/vol/data.parquet")
        self.assertIsInstance(p, VolumePath)
        self.assertEqual(p.parts, ["cat", "schema", "vol", "data.parquet"])

    def test_tables_bare(self):
        p = DatabricksPath.parse("/Tables/cat/schema/tbl")
        self.assertIsInstance(p, TablePath)
        self.assertEqual(p.parts, ["cat", "schema", "tbl"])

    def test_empty_returns_empty_dbfs(self):
        p = DatabricksPath.parse("")
        self.assertIsInstance(p, DBFSPath)
        self.assertEqual(p.parts, [])

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            DatabricksPath.parse("/some/unknown/path")

    def test_pass_through(self):
        p1 = DatabricksPath.parse("/Volumes/a/b/c")
        p2 = DatabricksPath.parse(p1)
        self.assertIs(p1, p2)

    def test_url_with_host(self):
        p = DatabricksPath.parse("https://myhost.databricks.com/#workspace/Users/me/nb")
        self.assertIsInstance(p, WorkspacePath)
        self.assertEqual(p.parts, ["Users", "me", "nb"])


# ══════════════════════════════════════════════════════════════════════════
# pathlib.PurePosixPath-compatible properties
# ══════════════════════════════════════════════════════════════════════════

class TestPathlibPureProperties(unittest.TestCase):
    """Verify the PurePosixPath-like properties on DatabricksPath."""

    def setUp(self):
        self.p = DatabricksPath.parse("/Volumes/cat/schema/vol/dir/file.tar.gz")

    def test_name(self):
        self.assertEqual(self.p.name, "file.tar.gz")

    def test_suffix(self):
        self.assertEqual(self.p.suffix, ".gz")

    def test_suffixes(self):
        self.assertEqual(self.p.suffixes, [".tar", ".gz"])

    def test_stem(self):
        self.assertEqual(self.p.stem, "file.tar")

    def test_parent(self):
        parent = self.p.parent
        self.assertIsInstance(parent, VolumePath)
        self.assertEqual(parent.parts, ["cat", "schema", "vol", "dir"])

    def test_parents(self):
        parents = self.p.parents
        self.assertGreater(len(parents), 0)
        self.assertEqual(parents[0].name, "dir")
        self.assertEqual(parents[-1].parts, [])

    def test_name_no_suffix(self):
        p = DatabricksPath.parse("/dbfs/dir/README")
        self.assertEqual(p.suffix, "")
        self.assertEqual(p.stem, "README")
        self.assertEqual(p.suffixes, [])

    def test_name_dotfile(self):
        p = DatabricksPath.parse("/dbfs/dir/.hidden")
        self.assertEqual(p.name, ".hidden")
        self.assertEqual(p.suffixes, [])

    def test_empty_name(self):
        p = DatabricksPath.parse("")
        self.assertEqual(p.name, "")


class TestPathlibJoinAndAlter(unittest.TestCase):
    """Test joinpath, with_name, with_suffix, with_stem."""

    def setUp(self):
        self.p = DatabricksPath.parse("/Volumes/cat/schema/vol/file.parquet")

    def test_truediv(self):
        child = self.p / "sub" / "file.csv"
        self.assertIsInstance(child, VolumePath)
        self.assertEqual(child.name, "file.csv")

    def test_joinpath(self):
        child = self.p.parent.joinpath("sub", "file.csv")
        self.assertEqual(child.name, "file.csv")
        self.assertIn("sub", child.parts)

    def test_with_name(self):
        q = self.p.with_name("data.csv")
        self.assertEqual(q.name, "data.csv")
        self.assertEqual(q.parent.parts, self.p.parent.parts)

    def test_with_suffix(self):
        q = self.p.with_suffix(".csv")
        self.assertEqual(q.name, "file.csv")

    def test_with_suffix_remove(self):
        q = self.p.with_suffix("")
        self.assertEqual(q.name, "file")

    def test_with_stem(self):
        q = self.p.with_stem("output")
        self.assertEqual(q.name, "output.parquet")

    def test_with_name_empty_raises(self):
        p = DatabricksPath.parse("")
        with self.assertRaises(ValueError):
            p.with_name("x")

    def test_with_suffix_bad(self):
        with self.assertRaises(ValueError):
            self.p.with_suffix("csv")  # missing dot


class TestPathlibMatch(unittest.TestCase):
    """Test match, is_relative_to, relative_to."""

    def test_match_name(self):
        p = DatabricksPath.parse("/Volumes/cat/schema/vol/data.parquet")
        self.assertTrue(p.match("*.parquet"))
        self.assertFalse(p.match("*.csv"))

    def test_match_wildcard(self):
        p = DatabricksPath.parse("/dbfs/dir/foo_bar.json")
        self.assertTrue(p.match("foo_*"))

    def test_is_relative_to_true(self):
        base = DatabricksPath.parse("/Volumes/cat/schema/vol")
        child = DatabricksPath.parse("/Volumes/cat/schema/vol/dir/file.txt")
        self.assertTrue(child.is_relative_to(base))

    def test_is_relative_to_false(self):
        base = DatabricksPath.parse("/Volumes/cat/schema/vol")
        other = DatabricksPath.parse("/Volumes/cat/other/vol")
        self.assertFalse(other.is_relative_to(base))

    def test_is_relative_to_cross_type(self):
        a = DatabricksPath.parse("/dbfs/foo")
        b = DatabricksPath.parse("/Workspace/foo")
        self.assertFalse(a.is_relative_to(b))

    def test_relative_to(self):
        base = DatabricksPath.parse("/Volumes/cat/schema/vol")
        child = DatabricksPath.parse("/Volumes/cat/schema/vol/dir/file.txt")
        rel = child.relative_to(base)
        self.assertEqual(rel.parts, ["dir", "file.txt"])

    def test_relative_to_raises(self):
        a = DatabricksPath.parse("/Volumes/cat/schema/vol")
        b = DatabricksPath.parse("/Volumes/cat/other/vol")
        with self.assertRaises(ValueError):
            b.relative_to(a)


# ══════════════════════════════════════════════════════════════════════════
# full_path / __fspath__ / __str__
# ══════════════════════════════════════════════════════════════════════════

class TestFullPath(unittest.TestCase):

    def test_dbfs(self):
        p = DatabricksPath.parse("/dbfs/mnt/data")
        self.assertEqual(p.full_path(), "/dbfs/mnt/data")
        self.assertEqual(str(p), "/dbfs/mnt/data")

    def test_workspace(self):
        p = DatabricksPath.parse("/Workspace/Users/me")
        self.assertEqual(p.full_path(), "/Workspace/Users/me")

    def test_volumes(self):
        p = DatabricksPath.parse("/Volumes/cat/schema/vol/file.csv")
        self.assertEqual(p.full_path(), "/Volumes/cat/schema/vol/file.csv")

    def test_tables(self):
        p = DatabricksPath.parse("/Tables/cat/schema/tbl")
        self.assertEqual(p.full_path(), "/Tables/cat/schema/tbl")

    def test_fspath(self):
        import os
        p = DatabricksPath.parse("/dbfs/tmp/file")
        self.assertEqual(os.fspath(p), "/dbfs/tmp/file")


# ══════════════════════════════════════════════════════════════════════════
# Equality / hash / comparison
# ══════════════════════════════════════════════════════════════════════════

class TestEquality(unittest.TestCase):

    def test_eq_same_type(self):
        a = DatabricksPath.parse("/dbfs/tmp/a")
        b = DatabricksPath.parse("/dbfs/tmp/a")
        self.assertEqual(a, b)

    def test_ne_different_parts(self):
        a = DatabricksPath.parse("/dbfs/tmp/a")
        b = DatabricksPath.parse("/dbfs/tmp/b")
        self.assertNotEqual(a, b)

    def test_ne_different_type(self):
        a = DatabricksPath.parse("/dbfs/a")
        b = DatabricksPath.parse("/Workspace/a")
        self.assertNotEqual(a, b)

    def test_eq_string(self):
        p = DatabricksPath.parse("/dbfs/tmp/a")
        self.assertEqual(p, "/dbfs/tmp/a")

    def test_hash_stable(self):
        a = DatabricksPath.parse("/Volumes/c/s/v/file.csv")
        b = DatabricksPath.parse("/Volumes/c/s/v/file.csv")
        self.assertEqual(hash(a), hash(b))


# ══════════════════════════════════════════════════════════════════════════
# DatabricksStatResult
# ══════════════════════════════════════════════════════════════════════════

class TestStatResult(unittest.TestCase):

    def test_fields(self):
        s = DatabricksStatResult(st_size=100, st_mtime=1.0, st_mode=0o755)
        self.assertEqual(s.st_size, 100)
        self.assertEqual(s.st_mtime, 1.0)
        self.assertEqual(s.st_mode, 0o755)

    def test_subscriptable(self):
        s = DatabricksStatResult(st_size=42, st_mtime=2.5, st_mode=0)
        self.assertEqual(s[0], 42)
        self.assertEqual(s[1], 2.5)

    def test_frozen(self):
        s = DatabricksStatResult()
        with self.assertRaises(AttributeError):
            s.st_size = 1


# ══════════════════════════════════════════════════════════════════════════
# UC-specific: VolumePath decomposition
# ══════════════════════════════════════════════════════════════════════════

class TestVolumePathUC(unittest.TestCase):

    def test_sql_parts_full(self):
        p = DatabricksPath.parse("/Volumes/cat/schema/vol/dir/file.csv")
        cat, sch, vol, rel = p.sql_volume_or_table_parts()
        self.assertEqual(cat, "cat")
        self.assertEqual(sch, "schema")
        self.assertEqual(vol, "vol")
        self.assertEqual(rel, ["dir", "file.csv"])

    def test_sql_parts_short(self):
        p = DatabricksPath.parse("/Volumes/cat")
        cat, sch, vol, rel = p.sql_volume_or_table_parts()
        self.assertEqual(cat, "cat")
        self.assertIsNone(sch)
        self.assertIsNone(vol)
        self.assertEqual(rel, [])


class TestTablePathUC(unittest.TestCase):

    def test_sql_parts(self):
        p = DatabricksPath.parse("/Tables/cat/schema/tbl")
        cat, sch, tbl, rel = p.sql_volume_or_table_parts()
        self.assertEqual(cat, "cat")
        self.assertEqual(sch, "schema")
        self.assertEqual(tbl, "tbl")
        self.assertEqual(rel, [])


# ══════════════════════════════════════════════════════════════════════════
# extension / file_type fallback
# ══════════════════════════════════════════════════════════════════════════

class TestExtension(unittest.TestCase):

    def test_extension(self):
        p = DatabricksPath.parse("/dbfs/dir/file.csv")
        self.assertEqual(p.extension, "csv")

    def test_extension_none(self):
        p = DatabricksPath.parse("/dbfs/dir/README")
        self.assertEqual(p.extension, "")


# ══════════════════════════════════════════════════════════════════════════
# Shutdown-hook registration for temporary paths
# ══════════════════════════════════════════════════════════════════════════

class TestShutdownHook(unittest.TestCase):
    """Verify that temporary paths register / unregister the shutdown exiter."""

    def _registry(self):
        from yggdrasil.environ.shutdown import shutdown_registry
        return shutdown_registry

    # ── helpers ──────────────────────────────────────────────────────────

    def _make_temp(self, path: str = "/dbfs/tmp/ygg_test_temp") -> DBFSPath:
        """Parse a non-temporary path and return a temporary copy via dc.replace."""
        import dataclasses as dc
        p = DatabricksPath.parse(path)
        # Use dc.replace so __post_init__ is called with temporary=True
        return dc.replace(p, temporary=True)

    # ── tests ─────────────────────────────────────────────────────────────

    def test_non_temporary_no_hook(self):
        """Non-temporary paths must not register a shutdown hook."""
        p = DatabricksPath.parse("/dbfs/tmp/normal")
        self.assertIsNone(p._shutdown_hook)
        self.assertFalse(self._registry().is_registered(p._unsafe_remove))

    def test_temporary_parse_registers_hook(self):
        """parse(temporary=True) registers a shutdown hook."""
        p = DatabricksPath.parse("/dbfs/tmp/ygg_shutdown_test", temporary=True)
        try:
            self.assertIsNotNone(p._shutdown_hook)
            self.assertTrue(self._registry().is_registered(p._unsafe_remove))
        finally:
            p._unregister_shutdown_remove()

    def test_dataclass_replace_temporary_registers_hook(self):
        """dc.replace(…, temporary=True) triggers __post_init__ and registers a hook."""
        p = self._make_temp()
        try:
            self.assertIsNotNone(p._shutdown_hook)
            self.assertTrue(self._registry().is_registered(p._unsafe_remove))
        finally:
            p._unregister_shutdown_remove()

    def test_remove_unregisters_hook(self):
        """Calling remove() must unregister the shutdown hook before any I/O."""
        p = DatabricksPath.parse("/dbfs/tmp/ygg_remove_unregister", temporary=True)
        self.assertTrue(self._registry().is_registered(p._unsafe_remove))

        # Patch is_file / is_dir to avoid real API calls
        p._is_file = False
        p._is_dir = False
        p.remove()

        self.assertIsNone(p._shutdown_hook)
        self.assertFalse(self._registry().is_registered(p._unsafe_remove))

    def test_close_wait_unregisters_hook(self):
        """close(wait=True) delegates to remove(), which unregisters the hook."""
        p = DatabricksPath.parse("/dbfs/tmp/ygg_close_wait", temporary=True)
        self.assertTrue(self._registry().is_registered(p._unsafe_remove))

        p._is_file = False
        p._is_dir = False
        p.close(wait=True)

        self.assertFalse(self._registry().is_registered(p._unsafe_remove))

    def test_close_nowait_unregisters_hook_immediately(self):
        """close(wait=False) must unregister the hook synchronously before spawning a thread."""
        import time
        p = DatabricksPath.parse("/dbfs/tmp/ygg_close_nowait", temporary=True)
        self.assertTrue(self._registry().is_registered(p._unsafe_remove))

        p._is_file = False
        p._is_dir = False
        p.close(wait=False)

        # Hook must be gone immediately (not after thread completion)
        self.assertIsNone(p._shutdown_hook)
        self.assertFalse(self._registry().is_registered(p._unsafe_remove))
        time.sleep(0.05)  # let background thread finish cleanly

    def test_parse_upgrades_existing_path_to_temporary(self):
        """parse() called on an already-parsed path with temporary=True arms the hook."""
        p = DatabricksPath.parse("/dbfs/tmp/ygg_upgrade")
        self.assertIsNone(p._shutdown_hook)

        p2 = DatabricksPath.parse(p, temporary=True)
        try:
            self.assertIs(p2, p)  # parse() returns the same object
            self.assertTrue(p.temporary)
            self.assertIsNotNone(p._shutdown_hook)
            self.assertTrue(self._registry().is_registered(p._unsafe_remove))
        finally:
            p._unregister_shutdown_remove()

    def test_double_register_is_idempotent(self):
        """Calling _register_shutdown_remove() twice must not create two hooks."""
        p = DatabricksPath.parse("/dbfs/tmp/ygg_double", temporary=True)
        hook_first = p._shutdown_hook
        p._register_shutdown_remove()  # second call — must be a no-op
        try:
            self.assertIs(p._shutdown_hook, hook_first)
        finally:
            p._unregister_shutdown_remove()

    def test_unregister_idempotent(self):
        """Calling _unregister_shutdown_remove() twice must not raise."""
        p = DatabricksPath.parse("/dbfs/tmp/ygg_unreg2", temporary=True)
        p._unregister_shutdown_remove()
        p._unregister_shutdown_remove()  # second call — must be harmless

    def test_parent_is_not_temporary(self):
        """parent property must never inherit the temporary flag or hook."""
        p = DatabricksPath.parse("/dbfs/tmp/ygg_parent_test", temporary=True)
        try:
            parent = p.parent
            self.assertFalse(parent.temporary)
            self.assertIsNone(parent._shutdown_hook)
        finally:
            p._unregister_shutdown_remove()


if __name__ == "__main__":
    unittest.main(verbosity=2)

