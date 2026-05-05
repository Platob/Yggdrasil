"""Integration tests: :class:`DeltaIO` reading non-local tables.

DeltaIO defaults ``mirror_leaves=True``: every AddFile leaf is wrapped
in a :class:`MirrorPath` so repeat reads serve from
``~/.yggdrasil/mirror`` instead of round-tripping the remote on every
parquet footer probe + body fetch. This file exercises the end-to-end
path against a fake non-local backend that delegates to a real local
directory underneath:

- a hand-built Delta table with a Spark-style ``part-xxx.snappy.parquet``
  leaf (this is the format Databricks / Spark writers produce; the leaf
  filename has the codec **before** the format extension);
- read through DeltaIO with the table root presented as a non-local
  path so :meth:`FolderIO._open_file_child` triggers the
  :class:`MirrorPath` wrap;
- verifies the read returns the right rows AND that the local mirror
  file actually appeared on disk under the configured root.

Why these tests matter: the ``.snappy.parquet`` / ``.zstd.parquet``
naming convention used to misroute through ``MediaType`` resolution —
the URL extension list ``[snappy, parquet]`` was treated as
``[format, codec]`` which collapsed to ``MediaType(OCTET_STREAM,
codec=SNAPPY)`` and dropped the parquet identity. The compensation
path in :class:`BytesIO` (lazy magic-bytes detection on the live
buffer) papered over the bug for in-process reads but every leaf
opened the remote twice — once for the doomed dispatch, once again
for magic detection — so any read against a real remote was paying
double round-trips per leaf and any path that didn't hit the magic
fallback (e.g. a TabularIO consumer that branches on declared media
type) failed outright.

These integration tests pin the fix end-to-end so a regression in
:meth:`MediaType.from_many`'s ordering rule trips here, not on a
remote.
"""

from __future__ import annotations

import json
from pathlib import Path as PyPath

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.buffer.nested.delta.io import DeltaIO
from yggdrasil.io.fs import LocalPath, Path
from yggdrasil.io.fs.mirror import _MIRROR_FRESH, _MIRROR_SWEPT
from yggdrasil.io.fs.mirror_path import MirrorPath
from yggdrasil.io.url import URL


# ---------------------------------------------------------------------------
# Fake non-local Path that delegates to a real local backing tree.
# ---------------------------------------------------------------------------
#
# Reports ``is_local=False`` so :meth:`FolderIO._open_file_child` wraps
# leaves in a :class:`MirrorPath` (DeltaIO sets ``mirror_leaves=True``
# by default). All bytes-level operations route through a backing
# :class:`LocalPath` rooted at a real tmp dir so the test can assert
# both the in-memory read result and the on-disk mirror artifacts.


class FakeRemote(Path):
    scheme = "fakeremote"
    __slots__ = ("_backing", "_root_local", "_root_url_path")

    def __init__(
        self,
        *,
        url,
        backing=None,
        root_local: PyPath = None,
        root_url_path: str = "/",
        temporary: bool = False,
        auto_open: bool = True,
    ):
        super().__init__(url=url, temporary=temporary, auto_open=auto_open)
        self._backing = backing
        self._root_local = PyPath(root_local) if root_local is not None else None
        self._root_url_path = root_url_path

    @classmethod
    def from_local(
        cls,
        local: PyPath,
        *,
        url_path: str = "/table",
    ) -> "FakeRemote":
        return cls(
            url=URL(scheme="fakeremote", host="h", path=url_path),
            backing=LocalPath.from_(str(local)),
            root_local=local,
            root_url_path=url_path,
        )

    # Identity ----------------------------------------------------------------

    @property
    def is_local(self) -> bool:
        return False

    def full_path(self) -> str:
        return f"fakeremote://h{self.url.path}"

    # Filesystem hooks --------------------------------------------------------

    def _stat(self):
        return self._backing._stat()

    def _ls(self, recursive: bool = False, allow_not_found: bool = True):
        for child in self._backing._ls(
            recursive=recursive, allow_not_found=allow_not_found,
        ):
            rel = PyPath(child.url.path).relative_to(self._backing.url.path)
            child_url = URL(
                scheme="fakeremote",
                host="h",
                path=f"{self.url.path.rstrip('/')}/{rel}",
            )
            yield FakeRemote(
                url=child_url,
                backing=child,
                root_local=self._root_local,
                root_url_path=self._root_url_path,
            )

    def iterdir(self):
        return self._ls(recursive=False, allow_not_found=True)

    def _mkdir(self, parents: bool = True, exist_ok: bool = True):
        return self._backing._mkdir(parents=parents, exist_ok=exist_ok)

    def _remove_file(self, allow_not_found: bool = True):
        return self._backing._remove_file(allow_not_found=allow_not_found)

    def _remove_dir(
        self,
        recursive: bool = True,
        allow_not_found: bool = True,
        with_root: bool = True,
    ):
        return self._backing._remove_dir(
            recursive=recursive,
            allow_not_found=allow_not_found,
            with_root=with_root,
        )

    def _open(self, mode: str = "rb", **kwargs):
        return self._backing._open(mode=mode, **kwargs)

    def _pread(self):
        return self._backing._pread()

    def _pwrite(self, data):
        return self._backing._pwrite(data)

    def read_bytes(self, *, raise_error: bool = True):
        return self._backing.read_bytes(raise_error=raise_error)

    def write_bytes(self, data, *, mode: str = "wb", parents: bool = True):
        return self._backing.write_bytes(data, mode=mode, parents=parents)

    def pread(self, n: int, pos: int, *, default=...):
        return self._backing.pread(n, pos, default=default)

    def pwrite(self, data, pos: int, *, parents: bool = True):
        return self._backing.pwrite(data, pos, parents=parents)

    def exists(self) -> bool:
        return self._backing.exists()

    def _from_url(self, url):
        rel = PyPath(url.path).relative_to(self._root_url_path)
        return FakeRemote(
            url=url,
            backing=LocalPath.from_(str(self._root_local / rel)),
            root_local=self._root_local,
            root_url_path=self._root_url_path,
        )


# ---------------------------------------------------------------------------
# Hand-built Delta table fixture
# ---------------------------------------------------------------------------


def _empty_struct_field(name: str, dtype: str = "long"):
    return {"name": name, "type": dtype, "nullable": True, "metadata": {}}


def _build_delta_table(
    target: PyPath,
    *,
    parquet_name: str,
    compression: str | None = "snappy",
) -> pa.Table:
    """Write a one-leaf Delta v0 table at *target*.

    Skips DeltaIO's writer (the schema codec doesn't accept all
    primitive types in this environment) so the test can stay focused
    on the read path. The leaf filename is parametrized so the suite
    can exercise every codec convention readers see in the wild.
    """
    target.mkdir(parents=True, exist_ok=True)
    log_dir = target / "_delta_log"
    log_dir.mkdir(parents=True, exist_ok=True)

    table = pa.table({"id": pa.array([10, 20, 30], type=pa.int64())})
    leaf = target / parquet_name
    pq.write_table(table, str(leaf), compression=compression or "none")
    leaf_size = leaf.stat().st_size

    schema_string = json.dumps({
        "type": "struct",
        "fields": [_empty_struct_field("id", "long")],
    })
    actions = [
        {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}},
        {
            "metaData": {
                "id": "tbl",
                "format": {"provider": "parquet", "options": {}},
                "schemaString": schema_string,
                "partitionColumns": [],
                "configuration": {},
            }
        },
        {
            "add": {
                "path": parquet_name,
                "partitionValues": {},
                "size": leaf_size,
                "modificationTime": 0,
                "dataChange": True,
            }
        },
    ]
    body = "\n".join(json.dumps(a) for a in actions) + "\n"
    (log_dir / "00000000000000000000.json").write_bytes(body.encode())
    return table


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_mirror_state():
    """Drop process-global mirror caches before / after each test.

    Without this the verdict cache from one test bleeds into the next
    and a stale mirror at the same remote ``full_path`` can serve up
    bytes from a previous test's table.
    """
    _MIRROR_FRESH.clear()
    _MIRROR_SWEPT.clear()
    yield
    _MIRROR_FRESH.clear()
    _MIRROR_SWEPT.clear()


@pytest.fixture
def mirror_root(tmp_path) -> LocalPath:
    """Per-test mirror root so we can introspect what landed where."""
    root = tmp_path / "mirror"
    root.mkdir(parents=True, exist_ok=True)
    return LocalPath.from_(str(root))


# ---------------------------------------------------------------------------
# Basic read through MirrorPath wrap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("compression,leaf_name", [
    (None, "part-00000-aaaa.parquet"),
    ("snappy", "part-00000-aaaa.snappy.parquet"),
    ("zstd", "part-00000-aaaa.zstd.parquet"),
])
class TestRemoteDeltaRead:
    """End-to-end: hand-built Delta + non-local path + DeltaIO read.

    Parametrized over the three filename conventions that show up in
    Delta tables (no codec extension, snappy-by-Spark-default,
    zstd-by-modern-Spark). Pre-:class:`MediaType.from_many` fix all
    three names dispatched through the BytesIO magic-bytes
    compensation path; post-fix the format extension wins outright
    and the dispatch picks ParquetIO upfront.
    """

    def test_read_via_remote_path(self, tmp_path, compression, leaf_name):
        target = tmp_path / "table"
        _build_delta_table(target, parquet_name=leaf_name, compression=compression)

        remote = FakeRemote.from_local(target)
        with DeltaIO(path=remote) as r:
            out = r.read_arrow_table()

        assert out.column_names == ["id"]
        assert out.to_pylist() == [{"id": 10}, {"id": 20}, {"id": 30}]

    def test_leaf_is_mirror_wrapped(self, tmp_path, compression, leaf_name):
        target = tmp_path / "table"
        _build_delta_table(target, parquet_name=leaf_name, compression=compression)

        remote = FakeRemote.from_local(target)
        with DeltaIO(path=remote) as r:
            children = list(r._iter_children(r.options_class()()))

        assert len(children) == 1
        leaf = children[0]
        # mirror_leaves=True is the DeltaIO default — the leaf's path
        # must be MirrorPath so repeat reads short-circuit through the
        # local mirror.
        assert isinstance(leaf.path, MirrorPath)
        # And the leaf type itself routes to the parquet reader, not
        # an opaque buffer — that's the MediaType.from_many fix.
        assert type(leaf).__name__ == "ParquetIO"

    def test_read_dispatches_to_parquet_io(
        self, tmp_path, compression, leaf_name,
    ):
        # Direct sanity check on TabularIO.from_path against the
        # MirrorPath wrap: post-fix this picks ParquetIO; pre-fix it
        # raised TypeError("Can't instantiate abstract class TabularIO")
        # because OCTET_STREAM+SNAPPY tried to dispatch to the abstract
        # base.
        from yggdrasil.io.buffer.base import TabularIO

        target = tmp_path / "table"
        _build_delta_table(target, parquet_name=leaf_name, compression=compression)
        leaf_remote = FakeRemote.from_local(target).joinpath(leaf_name)
        wrapped = MirrorPath(leaf_remote, ttl=60.0)

        io = TabularIO.from_path(wrapped)
        assert type(io).__name__ == "ParquetIO"


# ---------------------------------------------------------------------------
# Mirror artifact verification
# ---------------------------------------------------------------------------


class TestMirrorArtifacts:
    def test_mirror_file_lands_under_root(self, tmp_path, mirror_root):
        target = tmp_path / "table"
        leaf = "part-00000-aaaa.snappy.parquet"
        _build_delta_table(target, parquet_name=leaf, compression="snappy")

        # Open the leaf through MirrorPath with the test's mirror root,
        # then read it. Mirror file should appear under the root.
        remote = FakeRemote.from_local(target)
        leaf_path = MirrorPath(
            remote.joinpath(leaf),
            root=mirror_root,
            ttl=60.0,
        )
        leaf_path.read_bytes()  # forces download into mirror

        # Check that the mirror file landed under our mirror_root.
        # Layout: <root>/<scheme>/<host>/<url-path>
        candidates = list(PyPath(str(mirror_root)).rglob(leaf))
        assert candidates, f"no mirror file found under {mirror_root}"

    def test_repeat_read_serves_from_mirror(self, tmp_path, mirror_root):
        target = tmp_path / "table"
        leaf = "part-00000-aaaa.snappy.parquet"
        _build_delta_table(target, parquet_name=leaf, compression="snappy")

        # First read: downloads + caches; second read should reuse
        # the verdict (no extra remote stat / fetch).
        remote = FakeRemote.from_local(target)
        leaf_path = MirrorPath(
            remote.joinpath(leaf),
            root=mirror_root,
            ttl=60.0,
        )
        first = leaf_path.read_bytes()
        # Force-evict the on-disk leaf so a second download would fail
        # on a real remote — the verdict cache should keep the call
        # off the remote.
        (target / leaf).unlink()
        second = leaf_path.read_bytes()
        assert first == second
