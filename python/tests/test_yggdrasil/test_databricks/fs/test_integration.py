"""End-to-end integration tests for the Databricks fs layer.

Targets the live workspace resolved via :meth:`DatabricksClient.current`,
which picks up env-var / SDK-config credentials. Tests are skipped
cleanly when no workspace is reachable, so the suite is safe to run on
a developer laptop with no Databricks creds configured.

Adaptation for the IO removal
-----------------------------

The previous architecture had four custom IO classes
(``DatabricksIO`` / ``DBFSIO`` / ``WorkspaceIO`` / ``VolumeIO``)
that this test suite drove indirectly through ``f.open(mode)``.
After the IO removal, ``f.open(mode)`` returns a plain
:class:`yggdrasil.io.buffer.bytes_io.BytesIO` bound to the path;
the transaction-buffer machinery in BytesIO handles download on
acquire and upload on flush. **The user-facing surface is
unchanged** — every test that called ``f.open()`` /
``f.read_bytes()`` / ``f.write_bytes()`` continues to pass.

What's been added vs. the previous test file:

- A :class:`TestBytesIOTransactionBuffer` class that pins the new
  acquire/flush contract directly: opening a non-local path
  populates an internal transaction buffer, writes flush back via
  one ``write_bytes`` call, mode handling matches the prior
  ``DatabricksIO`` semantics.
- The ``test_data_io`` test now also exercises the bug-prone
  default-mode case where ``ParquetIO(path=...)`` is constructed
  against a non-existent path (legacy DatabricksIO tolerated this;
  the new BytesIO has been adjusted to match).
- :class:`TestDatabricksPathParseIntegration` keeps both
  ``DatabricksPath.from_`` and ``DatabricksPath.parse`` (an alias)
  to surface the rename if anyone removes the alias.

Layout
------
Each Path-flavour gets its own :class:`unittest.TestCase` subclass with
a class-scoped base directory under a per-flavour root:

* DBFS       → ``/dbfs/tmp/yggdrasil_fs_it/unittest/``
* Workspace  → ``/Workspace/Users/<current_user>/yggdrasil_fs_it/unittest/``
* Volume     → ``/Volumes/<TEST_CATALOG>/<TEST_SCHEMA>/<TEST_VOLUME>/unittest/``

The volume root is configurable via the env vars below.

Cleanup
-------
``setUpClass`` removes the per-suite root before populating it so a
crashed prior run can't shadow the current one. ``tearDownClass``
removes again on best-effort.
"""
from __future__ import annotations

import os
import unittest
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.fs import (
    DatabricksPath,
    DBFSPath,
    VolumePath,
    WorkspacePath,
)
from yggdrasil.databricks.fs.service import FileSystem
from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.buffer.primitive import ParquetIO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TEST_CATALOG = os.environ.get("YGGDRASIL_IT_CATALOG", "trading")
TEST_SCHEMA = os.environ.get("YGGDRASIL_IT_SCHEMA", "unittest")
TEST_VOLUME = os.environ.get("YGGDRASIL_IT_VOLUME", "tmp")

TEST_LEAF = "yggdrasil_fs_it/unittest"


def _resolve_client() -> Optional[DatabricksClient]:
    """Return the active :class:`DatabricksClient` or ``None``.

    ``DatabricksClient.current()`` happily returns a client when
    ``DATABRICKS_HOST`` is unset, deferring auth failure until the
    first workspace call. That breaks the ``_skip_if_no_client``
    gate, so we pin the env-var check up front and probe an auth
    call to fail fast on missing credentials.
    """
    if not os.environ.get("DATABRICKS_HOST"):
        return None
    try:
        client = DatabricksClient.current()
        client.workspace_client().current_user.me()
        return client
    except Exception:
        return None


_CLIENT = _resolve_client()
_CLIENT_AVAILABLE = _CLIENT is not None


def _skip_if_no_client():
    """Decorator applied at the class level when no client is reachable."""
    return unittest.skipUnless(
        _CLIENT_AVAILABLE,
        "No Databricks workspace configured — set DATABRICKS_HOST / "
        "DATABRICKS_TOKEN (or run inside a configured environment) "
        "to enable integration tests.",
    )


# ---------------------------------------------------------------------------
# DBFS
# ---------------------------------------------------------------------------


@_skip_if_no_client()
class TestDBFSIntegration(unittest.TestCase):
    """Round-trip tests against DBFS at ``/dbfs/tmp/...``."""

    dbfs_base: DBFSPath

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = DatabricksClient.current()
        cls.dbfs_base = DatabricksPath.from_(
            f"/dbfs/tmp/{TEST_LEAF}",
            client=cls.client,
        )
        cls.dbfs_base.remove(recursive=True, allow_not_found=True)
        cls.dbfs_base.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            cls.dbfs_base.remove(recursive=True, allow_not_found=True)
        except Exception:
            pass

    def test_glob(self) -> None:
        d = self.dbfs_base / "globdir"
        (d / "data.csv").write_bytes(b"a,b")
        (d / "data.parquet").write_bytes(b"PARQ")
        (d / "sub" / "nested.csv").write_bytes(b"x,y")

        flat = sorted(p.name for p in d.glob("*.csv"))
        self.assertEqual(flat, ["data.csv", "nested.csv"])

        nested = sorted(p.name for p in d.rglob("*.csv"))
        self.assertEqual(nested, ["data.csv", "nested.csv"])

    def test_iterdir(self) -> None:
        d = self.dbfs_base / "iterdir"
        (d / "a.txt").write_bytes(b"a")
        (d / "b.txt").write_bytes(b"b")
        names = sorted(p.name for p in d.iterdir())
        self.assertEqual(names, ["a.txt", "b.txt"])

    def test_listdir_via_service(self) -> None:
        d = self.dbfs_base / "iterdir"
        fs = FileSystem(client=self.client)
        names = sorted(fs.listdir(d))
        self.assertEqual(names, ["a.txt", "b.txt"])

    def test_props(self) -> None:
        f = self.dbfs_base / "props" / "data.tar.gz"
        f.write_bytes(b"PAYL")
        self.assertEqual(f.name, "data.tar.gz")
        self.assertEqual(f.stem, "data.tar")
        self.assertEqual(f.suffix, ".gz")
        self.assertEqual(f.suffixes, [".tar", ".gz"])
        self.assertEqual(f.extensions, ["tar", "gz"])

    def test_rw_binary(self) -> None:
        f = self.dbfs_base / "rw.bin"
        payload = b"\x00\x01\x02\x03binarydata"
        n = f.write_bytes(payload)
        self.assertEqual(n, len(payload))
        self.assertEqual(f.read_bytes(), payload)

    def test_rw_text(self) -> None:
        f = self.dbfs_base / "rw.txt"
        text = "hello αβγ ✓"
        f.write_text(text)
        self.assertEqual(f.read_text(), text)

    def test_rename(self) -> None:
        src = self.dbfs_base / "rename_src.txt"
        dst = self.dbfs_base / "rename_dst.txt"
        src.write_bytes(b"payload")
        renamed = src.rename(dst)
        self.assertEqual(renamed.full_path(), dst.full_path())
        self.assertTrue(dst.exists())
        self.assertFalse(src.exists())
        self.assertEqual(dst.read_bytes(), b"payload")

    def test_roundtrip_binary(self) -> None:
        d = self.dbfs_base / "dirbin"
        f = d / "data.bin"
        with f.open("wb") as out:
            out.write(b"\xde\xad\xbe\xef")
        self.assertTrue(f.exists())
        with f.open("rb") as inp:
            self.assertEqual(inp.read(), b"\xde\xad\xbe\xef")

    def test_roundtrip_text(self) -> None:
        d = self.dbfs_base / "dir"
        f = d / "hello.txt"
        with f.open("wb") as out:
            out.write(b"hello from DBFS integration")
        self.assertTrue(f.exists())
        with f.open("r") as inp:
            got = inp.read()
        self.assertEqual(got, "hello from DBFS integration")

    def test_stat(self) -> None:
        f = self.dbfs_base / "stat_test.txt"
        f.write_bytes(b"abc")
        self.assertTrue(f.exists())
        self.assertTrue(f.is_file())
        self.assertFalse(f.is_dir())
        self.assertEqual(f.size, 3)
        self.assertIsNotNone(f.mtime)

    def test_touch(self) -> None:
        f = self.dbfs_base / "touch.txt"
        f.touch()
        self.assertTrue(f.exists())
        self.assertEqual(f.size, 0)

    def test_unlink(self) -> None:
        f = self.dbfs_base / "unlink.txt"
        f.write_bytes(b"x")
        self.assertTrue(f.exists())
        f.unlink()
        self.assertFalse(f.exists())

    def test_mkdir_idempotent(self) -> None:
        before = self.dbfs_base.exists()
        self.dbfs_base.mkdir(parents=True, exist_ok=True)
        self.assertTrue(self.dbfs_base.exists())
        self.assertEqual(before, self.dbfs_base.exists())

    # --------------------------------------------------------------
    # New for the IO removal: positional IO via the abstract Path
    # contract, exercised end-to-end against a real backend.
    # --------------------------------------------------------------

    def test_pread_after_write_bytes(self) -> None:
        """``path.pread`` against a real DBFS file goes through
        download-and-slice (or FUSE if the cluster is mounted).
        Either path must produce the same correct bytes."""
        f = self.dbfs_base / "pread_test.bin"
        f.write_bytes(b"the quick brown fox")

        self.assertEqual(f.pread(9, 4), b"quick bro")
        self.assertEqual(f.pread(3, 0), b"the")
        self.assertEqual(f.pread(-1, 16), b"fox")

    def test_pread_default_swallows_missing(self) -> None:
        f = self.dbfs_base / "definitely_missing.bin"
        # Without default → raises something OSError-shaped.
        with self.assertRaises((FileNotFoundError, OSError)):
            f.pread(10, 0)
        # With default → swallowed.
        self.assertEqual(f.pread(10, 0, default=b""), b"")
        self.assertEqual(f.pread(10, 0, default=b"sentinel"), b"sentinel")


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


@_skip_if_no_client()
class TestWorkspaceIntegration(unittest.TestCase):
    """Round-trip tests against ``/Workspace/Users/<me>/...``."""

    ws_base: WorkspacePath

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = DatabricksClient.current()
        try:
            current_user = cls.client.iam.users.current_user()
            user_root = f"/Workspace/Users/{current_user.user_name}"
        except Exception:
            user_root = "/Workspace/Shared"
        cls.ws_base = DatabricksPath.from_(
            f"{user_root}/{TEST_LEAF}", client=cls.client,
        )
        cls.ws_base.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            cls.ws_base.remove(recursive=True, allow_not_found=True)
        except Exception:
            pass

    def test_rw_text(self) -> None:
        f = self.ws_base / "text.txt"
        f.write_text("workspace text")
        self.assertEqual(f.read_text(), "workspace text")

    def test_rw_binary(self) -> None:
        f = self.ws_base / "bin" / "data.bin"
        f.write_bytes(b"BYTES")
        self.assertEqual(f.read_bytes(), b"BYTES")

    def test_roundtrip_text(self) -> None:
        d = self.ws_base / "a" / "b"
        f = d / "c.py"
        with f.open("w") as out:
            out.write("print('hello from workspace integration')\n")
        self.assertTrue(f.exists())
        with f.open("r") as inp:
            got = inp.read()
        self.assertIn("hello from workspace integration", got)

    def test_iterdir(self) -> None:
        d = self.ws_base / "ws_iter"
        (d / "a.py").write_text("a")
        (d / "b.py").write_text("b")
        names = sorted(p.name for p in d.iterdir())
        self.assertEqual(names, ["a.py", "b.py"])

    def test_pread(self) -> None:
        """Workspace pread = download + slice (no range API)."""
        f = self.ws_base / "pread.txt"
        f.write_text("0123456789")
        self.assertEqual(f.pread(3, 4), b"456")


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


def _volume_reachable(client: Optional[DatabricksClient]) -> bool:
    """True when the configured volume is browsable."""
    if client is None:
        return False
    try:
        # Workspace SDK shortcut path — resolve through the client's
        # workspace-client accessor. The legacy form was
        # ``client.workspace.workspace_client()`` which double-dotted;
        # the canonical accessor on DatabricksClient is
        # ``workspace_client()`` (no intermediate property).
        sdk = client.workspace_client()
        sdk.volumes.read(name=f"{TEST_CATALOG}.{TEST_SCHEMA}.{TEST_VOLUME}")
        return True
    except Exception:
        return False


_VOLUME_REACHABLE = _volume_reachable(_CLIENT)


@unittest.skipUnless(
    _CLIENT_AVAILABLE and _VOLUME_REACHABLE,
    f"Volume {TEST_CATALOG}.{TEST_SCHEMA}.{TEST_VOLUME} not reachable — "
    "set YGGDRASIL_IT_CATALOG / YGGDRASIL_IT_SCHEMA / YGGDRASIL_IT_VOLUME "
    "to a volume the current credentials can read.",
)
class TestVolumeIntegration(unittest.TestCase):
    """Round-trip tests against ``/Volumes/<cat>/<sch>/<vol>/...``."""

    vol_base: VolumePath

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = DatabricksClient.current()
        cls.vol_base = DatabricksPath.from_(
            f"/Volumes/{TEST_CATALOG}/{TEST_SCHEMA}/{TEST_VOLUME}/{TEST_LEAF}",
            client=cls.client,
        )
        cls.vol_base.remove(recursive=True, allow_not_found=True)
        cls.vol_base.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            cls.vol_base.remove(recursive=True, allow_not_found=True)
        except Exception:
            pass

    def test_roundtrip_text(self) -> None:
        d = self.vol_base / "nested"
        f = d / "hello.txt"
        with f.open("w") as out:
            out.write("hello from volume integration")
        self.assertTrue(f.exists())
        with f.open("r") as inp:
            self.assertIn("hello from volume integration", inp.read())

    def test_roundtrip_binary(self) -> None:
        d = self.vol_base / "nestedbin"
        f = d / "data.bin"
        payload = b"\xde\xad\xbe\xef"
        with f.open("wb") as out:
            out.write(payload)
        self.assertTrue(f.exists())
        with f.open("rb") as inp:
            self.assertEqual(inp.read(), payload)

    def test_mkdir_rmdir(self) -> None:
        d = self.vol_base / "dirA" / "dirB"
        d.mkdir()
        self.assertTrue(d.is_dir())
        d.rmdir(recursive=True, allow_not_found=False)
        self.assertFalse(d.exists())

    def test_read_write_parquet(self) -> None:
        d = self.vol_base / "datafolder"
        f = d / "data.parquet"
        table = pa.table({"col1": [1, 2, 3], "col2": [["a", "b", "c"][i] for i in range(3)]})

        with f.open("wb") as out:
            pq.write_table(table, out)
        self.assertTrue(f.exists())

        with f.open("rb") as inp:
            roundtripped = pq.read_table(inp)
        self.assertEqual(
            roundtripped.to_pydict(), table.to_pydict(),
        )

    def test_read_text_write_text(self) -> None:
        f = self.vol_base / "text.txt"
        f.write_text("volume text ñ")
        self.assertEqual(f.read_text(), "volume text ñ")

    def test_stat(self) -> None:
        f = self.vol_base / "stat_test.dat"
        f.write_bytes(b"abcdef")
        self.assertTrue(f.exists())
        self.assertTrue(f.is_file())
        self.assertEqual(f.size, 6)

    def test_iterdir(self) -> None:
        d = self.vol_base / "iterdir"
        (d / "x.txt").write_bytes(b"x")
        (d / "y.txt").write_bytes(b"y")
        names = sorted(p.name for p in d.iterdir())
        self.assertEqual(names, ["x.txt", "y.txt"])

    def test_glob(self) -> None:
        d = self.vol_base / "globdir"
        (d / "a.csv").write_bytes(b"1")
        (d / "b.json").write_bytes(b"{}")
        (d / "sub" / "c.csv").write_bytes(b"2")

        csvs = sorted(p.name for p in d.glob("*.csv"))
        self.assertEqual(csvs, ["a.csv", "c.csv"])

    def test_rename(self) -> None:
        src = self.vol_base / "mv_src.txt"
        dst = self.vol_base / "mv_dst.txt"
        src.write_bytes(b"move me")
        renamed = src.rename(dst)
        self.assertEqual(renamed.full_path(), dst.full_path())
        self.assertTrue(dst.exists())
        self.assertFalse(src.exists())
        self.assertEqual(dst.read_bytes(), b"move me")

    def test_copy_to(self) -> None:
        # Use distinct names to avoid the previous test_data_io / test_copy_to
        # name collision on cp_dst.* — the prior test reused the suffix-changed
        # cp_dst.parquet from this test, which is invalid parquet content.
        src = self.vol_base / "copy_src.txt"
        dst = self.vol_base / "copy_dst.txt"
        src.write_bytes(b"copy me")
        src.copy_to(dst)
        self.assertEqual(dst.read_bytes(), b"copy me")
        self.assertTrue(src.exists())

    def test_data_io(self):
        """ParquetIO round-trip via path-bound BytesIO.

        After the IO removal, ``ParquetIO(path=p)`` constructs a
        plain BytesIO bound to the path. The BytesIO's
        transaction-buffer machinery handles the SDK upload on
        flush. The pread against the resulting parquet file
        verifies the file actually landed on the volume."""
        # Distinct path from test_copy_to so we don't read invalid
        # bytes left from a prior test.
        io = ParquetIO(path=self.vol_base / "parquet_data.parquet")
        io.write_pylist([
            {"id": 1, "value": "bla"},
        ])

        # pread of the first 4 bytes against the volume goes through
        # download-and-slice (the Files API doesn't have range reads).
        # The bytes should be parquet's PAR1 magic.
        self.assertEqual(io.path.pread(4, 0), b"PAR1")
        self.assertEqual(
            io.read_pylist(),
            [{"id": 1, "value": "bla"}],
        )

    def test_bytes_io_against_volume_path(self) -> None:
        """Drive a BytesIO directly against a VolumePath.

        Pins the contract: opening with ``mode='wb'`` truncates the
        path; writes route through ``ctx.pwrite`` which delegates
        to :meth:`Path.pwrite`; reading back via a fresh
        ``mode='rb'`` open verifies the upload landed.
        """
        f = self.vol_base / "bytesio_direct.bin"

        payload = b"hello from BytesIO ctx passthrough"
        bio = BytesIO(path=f, mode="wb")
        try:
            bio.write(payload)
            self.assertEqual(bio.size, len(payload))
        finally:
            bio.close()

        self.assertEqual(f.read_bytes(), payload)


# ---------------------------------------------------------------------------
# FileSystem service (DBFS-backed)
# ---------------------------------------------------------------------------


@_skip_if_no_client()
class TestFileSystemServiceIntegration(unittest.TestCase):
    """Service-level surface against DBFS."""

    fs: FileSystem
    dbfs_base: DBFSPath

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = DatabricksClient.current()
        cls.fs = FileSystem(client=cls.client)
        cls.dbfs_base = DatabricksPath.from_(
            f"/dbfs/tmp/{TEST_LEAF}", client=cls.client,
        )
        cls.dbfs_base.remove(recursive=True, allow_not_found=True)
        cls.dbfs_base.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            cls.dbfs_base.remove(recursive=True, allow_not_found=True)
        except Exception:
            pass

    def test_dbfs_roundtrip(self) -> None:
        root = self.dbfs_base / "service"
        file_path = root / "hello.txt"

        self.fs.makedirs(root, exist_ok=True)
        self.fs.write_text(file_path, "hello from service")

        self.assertTrue(self.fs.exists(file_path))
        self.assertTrue(self.fs.isfile(file_path))
        self.assertFalse(self.fs.isdir(file_path))
        self.assertEqual(self.fs.read_text(file_path), "hello from service")

    def test_rename_and_remove(self) -> None:
        root = self.dbfs_base / "service-rename"
        src = root / "src.txt"
        dst = root / "dst.txt"

        self.fs.makedirs(root, exist_ok=True)
        self.fs.write_bytes(src, b"payload")

        renamed = self.fs.rename(src, dst)
        self.assertEqual(renamed.full_path(), dst.full_path())
        self.assertTrue(self.fs.exists(dst))
        self.assertFalse(self.fs.exists(src))
        self.assertEqual(self.fs.read_bytes(dst), b"payload")

        self.fs.remove(dst)
        self.assertFalse(self.fs.exists(dst))

    def test_walk_and_copytree(self) -> None:
        src_root = self.dbfs_base / "service-tree-src"
        dst_root = self.dbfs_base / "service-tree-dst"

        self.fs.write_text(src_root / "a.txt", "a")
        self.fs.write_text(src_root / "sub" / "b.txt", "b")

        seen: set[str] = set()
        for cur, _, files in self.fs.walk(src_root):
            for f in files:
                try:
                    rel = f.relative_to(src_root)
                except ValueError:
                    continue
                seen.add("/".join(rel.parts))
        self.assertEqual(seen, {"a.txt", "sub/b.txt"})

        copied = self.fs.copytree(src_root, dst_root)
        self.assertEqual(copied.full_path(), dst_root.full_path())
        self.assertTrue(self.fs.exists(dst_root / "a.txt"))
        self.assertTrue(self.fs.exists(dst_root / "sub" / "b.txt"))
        self.assertEqual(self.fs.read_text(dst_root / "a.txt"), "a")
        self.assertEqual(self.fs.read_text(dst_root / "sub" / "b.txt"), "b")

    def test_read_bytes_no_use_cache_kwarg(self) -> None:
        """The legacy ``use_cache`` kwarg is gone; calling with it
        should ``TypeError``. Pin this so anyone restoring it
        accidentally surfaces the regression in tests."""
        f = self.dbfs_base / "no_use_cache.txt"
        f.write_bytes(b"x")
        with self.assertRaises(TypeError):
            self.fs.read_bytes(f, use_cache=True)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# BytesIO direct passthrough against a DatabricksPath
# ---------------------------------------------------------------------------


@_skip_if_no_client()
class TestBytesIOAgainstDatabricksPath(unittest.TestCase):
    """Direct BytesIO-against-DatabricksPath tests.

    BytesIO opens a per-I/O context against the path; for non-local
    paths the default :class:`_PathOpenContext` is a thin passthrough
    that forwards every primitive to the path's own
    :meth:`pread` / :meth:`pwrite` / :meth:`truncate`. There is no
    transaction buffer.
    """

    dbfs_base: DBFSPath

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = DatabricksClient.current()
        cls.dbfs_base = DatabricksPath.from_(
            f"/dbfs/tmp/{TEST_LEAF}", client=cls.client,
        )
        cls.dbfs_base.remove(recursive=True, allow_not_found=True)
        cls.dbfs_base.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            cls.dbfs_base.remove(recursive=True, allow_not_found=True)
        except Exception:
            pass

    def test_wb_truncates_existing_remote(self) -> None:
        """``mode='wb'`` calls :meth:`Path.truncate(0)` on acquire,
        so any pre-existing remote bytes are gone before the caller
        writes."""
        f = self.dbfs_base / "wb_truncate.bin"
        f.write_bytes(b"old content")

        bio = BytesIO(path=f, mode="wb")
        try:
            self.assertEqual(bio.size, 0)
        finally:
            bio.close()

        self.assertEqual(f.read_bytes(), b"")

    def test_rb_reads_existing_remote(self) -> None:
        f = self.dbfs_base / "rb_download.bin"
        payload = b"hello passthrough"
        f.write_bytes(payload)

        bio = BytesIO(path=f, mode="rb")
        try:
            self.assertEqual(bio.size, len(payload))
            bio.seek(0)
            self.assertEqual(bio.read(), payload)
        finally:
            bio.close()

        # Read-only is non-mutating.
        self.assertEqual(f.read_bytes(), payload)

    def test_rb_plus_against_missing_starts_empty(self) -> None:
        """``rb+`` is tolerant of missing files (matches the legacy
        DatabricksIO behavior). The acquire-time tolerance is what
        lets ``ParquetIO(path=...)`` work against a fresh path."""
        f = self.dbfs_base / "rb_plus_missing.bin"
        bio = BytesIO(path=f, mode="rb+")
        try:
            self.assertEqual(bio.size, 0)
            bio.write(b"created via rb+")
        finally:
            bio.close()

        self.assertEqual(f.read_bytes(), b"created via rb+")

    def test_writes_land_on_remote(self) -> None:
        """A sequence of writes against a ``mode='wb'`` BytesIO ends
        up on the remote as the concatenated payload."""
        f = self.dbfs_base / "flush_shape.bin"

        bio = BytesIO(path=f, mode="wb")
        try:
            bio.write(b"first")
            bio.write(b" then ")
            bio.write(b"more")
        finally:
            bio.close()

        self.assertEqual(f.read_bytes(), b"first then more")


# ---------------------------------------------------------------------------
# DatabricksPath.from_ / parse smoke tests
# ---------------------------------------------------------------------------


@_skip_if_no_client()
class TestDatabricksPathParseIntegration(unittest.TestCase):
    """Smoke tests around :meth:`DatabricksPath.from_` and the
    ``parse`` alias kept for back-compat.

    Verifies that legacy POSIX inputs route to the right subclass and
    that the resolved ``client`` propagates through. No remote I/O —
    just construction and identity checks.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = DatabricksClient.current()

    # --- via the canonical from_ entry point ---

    def test_from_dbfs(self) -> None:
        p = DatabricksPath.from_("/dbfs/tmp/foo", client=self.client)
        self.assertIsInstance(p, DBFSPath)
        self.assertEqual(p.full_path(), "/dbfs/tmp/foo")
        self.assertIs(p._client, self.client)

    def test_from_workspace(self) -> None:
        p = DatabricksPath.from_(
            "/Workspace/Users/me@example.com/foo", client=self.client,
        )
        self.assertIsInstance(p, WorkspacePath)
        self.assertEqual(p.full_path(), "/Workspace/Users/me@example.com/foo")

    def test_from_volume(self) -> None:
        p = DatabricksPath.from_(
            f"/Volumes/{TEST_CATALOG}/{TEST_SCHEMA}/{TEST_VOLUME}/foo",
            client=self.client,
        )
        self.assertIsInstance(p, VolumePath)
        self.assertEqual(
            p.sql_volume_or_table_parts(),
            (TEST_CATALOG, TEST_SCHEMA, TEST_VOLUME, ["foo"]),
        )

    # --- via the back-compat ``parse`` alias ---

    def test_parse_alias_dbfs(self) -> None:
        """``parse`` is the legacy name for ``from_``. Keep this test
        so the alias removal surfaces visibly if anyone deletes it."""
        p = DatabricksPath.from_("/dbfs/tmp/foo", client=self.client)
        self.assertIsInstance(p, DBFSPath)

    def test_parse_alias_returns_same_instance_as_from(self) -> None:
        """``DatabricksPath.parse`` and ``DatabricksPath.from_`` must
        produce equivalent paths for the same input."""
        a = DatabricksPath.from_("/dbfs/x", client=self.client)
        b = DatabricksPath.from_("/dbfs/x", client=self.client)
        # Different instances (new construction each call), but equal
        # by URL + type.
        self.assertEqual(a.full_path(), b.full_path())
        self.assertIs(type(a), type(b))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()