"""RemotePath tests using a stub backend backed by a plain dict."""
from __future__ import annotations

import time
from typing import Any, ClassVar, Iterator
from unittest.mock import patch

import pyarrow as pa
import pytest

from yggdrasil.dataclasses import WaitingConfig
from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.enums import Mode
from yggdrasil.io.io_stats import IOKind, IOStats
from yggdrasil.path.remote_path import RemotePath
from yggdrasil.url import URL


# ---------------------------------------------------------------------------
# Stub backend — dict-backed RemotePath for hermetic testing
# ---------------------------------------------------------------------------


class _StubRemotePath(RemotePath):
    _STORAGE: ClassVar[dict[str, bytes]] = {}
    _INSTANCES: ClassVar[ExpiringDict] = ExpiringDict(
        default_ttl=300.0,
        max_size=10_000,
    )

    # ``scheme`` is set after class body (below) so ``__init_subclass__``
    # skips the Scheme enum coercion, yet ``IO.__new__`` sees a truthy
    # value and treats the class as a storage leaf (not a cursor).

    def __init__(
        self,
        data: Any = None,
        *,
        url: URL | None = None,
        singleton_ttl: Any = ...,
        **kwargs: Any,
    ) -> None:
        if url is None and isinstance(data, str):
            url = URL(scheme="stub", host="store", path=data)
        del singleton_ttl
        super().__init__(url=url, singleton_ttl=False, **kwargs)

    def full_path(self) -> str:
        return f"stub://{self.url.host}{self.url.path}"

    # -- backend primitives ------------------------------------------------

    def _stat_uncached(self) -> IOStats:
        key = self.url.path
        if key in self._STORAGE:
            return IOStats(
                size=len(self._STORAGE[key]),
                kind=IOKind.FILE,
                mtime=time.time(),
            )
        # Check if any keys start with key + "/" for directory semantics.
        prefix = key.rstrip("/") + "/"
        if any(k.startswith(prefix) for k in self._STORAGE):
            return IOStats(size=0, kind=IOKind.DIRECTORY, mtime=0.0)
        return IOStats(size=0, kind=IOKind.MISSING, mtime=0.0)

    def _read_mv(self, n: int, pos: int) -> memoryview:
        key = self.url.path
        data = self._STORAGE.get(key)
        if data is None:
            raise FileNotFoundError(key)
        if n < 0:
            return memoryview(data[pos:])
        return memoryview(data[pos : pos + n])

    def _upload(self, content: bytes) -> int:
        key = self.url.path
        self._STORAGE[key] = content
        return len(content)

    def _ls(
        self,
        recursive: bool = False,
        *,
        singleton_ttl: Any = False,
    ) -> Iterator["_StubRemotePath"]:
        prefix = self.url.path.rstrip("/") + "/"
        seen: set[str] = set()
        for k in sorted(self._STORAGE):
            if not k.startswith(prefix):
                continue
            remainder = k[len(prefix):]
            if not recursive:
                top = remainder.split("/")[0]
                if top in seen:
                    continue
                seen.add(top)
                yield _StubRemotePath(prefix + top, singleton_ttl=False)
            else:
                yield _StubRemotePath(k, singleton_ttl=False)

    def _mkdir(self, parents: bool, exist_ok: bool) -> None:
        pass

    def _remove_file(self, missing_ok: bool, wait: WaitingConfig) -> None:
        key = self.url.path
        if key in self._STORAGE:
            del self._STORAGE[key]
        elif not missing_ok:
            raise FileNotFoundError(key)

    def _remove_dir(
        self,
        recursive: bool,
        missing_ok: bool,
        wait: WaitingConfig,
    ) -> None:
        prefix = self.url.path.rstrip("/") + "/"
        to_delete = [k for k in self._STORAGE if k.startswith(prefix)]
        if not to_delete and not missing_ok:
            raise FileNotFoundError(self.url.path)
        for k in to_delete:
            del self._STORAGE[k]


# Post-class scheme assignment: truthy so IO.__new__ treats it as a storage
# leaf, but never passed through Scheme.from_() / __init_subclass__.
_StubRemotePath.scheme = "stub"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_stub_state():
    _StubRemotePath._STORAGE.clear()
    _StubRemotePath._INSTANCES = ExpiringDict(
        default_ttl=300.0,
        max_size=10_000,
    )
    yield
    _StubRemotePath._STORAGE.clear()
    _StubRemotePath._INSTANCES = ExpiringDict(
        default_ttl=300.0,
        max_size=10_000,
    )


def _make(path: str = "/data.bin", **kwargs: Any) -> _StubRemotePath:
    return _StubRemotePath(path, singleton_ttl=False, **kwargs)


# ---------------------------------------------------------------------------
# TestRemoteReadWrite
# ---------------------------------------------------------------------------


class TestRemoteReadWrite:

    def test_write_bytes_then_read_back(self) -> None:
        p = _make("/rw/file.bin")
        p.write_bytes(b"hello world")
        assert bytes(p.read_mv(-1, 0)) == b"hello world"

    def test_write_at_position_zero_then_read(self) -> None:
        p = _make("/rw/pos0.bin")
        p.write_mv(memoryview(b"ABCDE"), 0, overwrite=True)
        assert bytes(p.read_mv(5, 0)) == b"ABCDE"

    def test_size_matches_written_data(self) -> None:
        p = _make("/rw/size.bin")
        payload = b"x" * 128
        p.write_bytes(payload)
        assert p.size == 128

    def test_overwrite_replaces_content(self) -> None:
        p = _make("/rw/overwrite.bin")
        p.write_bytes(b"first")
        p.write_mv(memoryview(b"second"), 0, overwrite=True)
        assert bytes(p.read_mv(-1, 0)) == b"second"

    def test_read_missing_raises(self) -> None:
        # Whole-blob contract: the backend primitive raises on a missing
        # object (no silent empty-buffer swallow).
        p = _make("/rw/ghost.bin")
        with pytest.raises(FileNotFoundError):
            p._read_mv(10, 0)

    def test_write_then_stat_shows_correct_size(self) -> None:
        p = _make("/rw/stat_size.bin")
        p.write_bytes(b"0123456789")
        p.invalidate_singleton()
        stat = p._stat()
        assert stat.size == 10
        assert stat.kind == IOKind.FILE


# ---------------------------------------------------------------------------
# TestRemoteWriteBuffer
# ---------------------------------------------------------------------------


class TestRemoteWriteBuffer:
    """An *acquired* window (``with path:`` / ``open("wb")``) coalesces
    writes into a single upload; closed writes upload straight through."""

    def test_acquired_writes_flush_once(self) -> None:
        p = _StubRemotePath("/buf/dirty.bin", singleton_ttl=False)
        _StubRemotePath._STORAGE["/buf/dirty.bin"] = b"\x00" * 64

        call_count = 0
        original_upload = _StubRemotePath._upload

        def counting_upload(self_inner, content):
            nonlocal call_count
            call_count += 1
            return original_upload(self_inner, content)

        with patch.object(_StubRemotePath, "_upload", counting_upload):
            with p:
                p.write_mv(memoryview(b"PATCHED"), 0)
                p.write_mv(memoryview(b"tail"), 60)
        # Both splices coalesce into one upload on release.
        assert call_count == 1
        stored = _StubRemotePath._STORAGE["/buf/dirty.bin"]
        assert stored[:7] == b"PATCHED"
        assert stored[60:64] == b"tail"

    def test_acquired_read_after_write_sees_buffer(self) -> None:
        p = _StubRemotePath("/buf/raw.bin", singleton_ttl=False)
        with p:
            p.write_mv(memoryview(b"hello"), 0, overwrite=True)
            # Read-after-write within the handle is served from the buffer.
            assert bytes(p.read_mv(-1, 0)) == b"hello"
        assert _StubRemotePath._STORAGE["/buf/raw.bin"] == b"hello"

    def test_closed_write_uploads_immediately(self) -> None:
        p = _StubRemotePath("/buf/closed.bin", singleton_ttl=False)
        call_count = 0
        original_upload = _StubRemotePath._upload

        def counting_upload(self_inner, content):
            nonlocal call_count
            call_count += 1
            return original_upload(self_inner, content)

        with patch.object(_StubRemotePath, "_upload", counting_upload):
            p.write_bytes(b"hello")
        assert call_count == 1
        assert _StubRemotePath._STORAGE["/buf/closed.bin"] == b"hello"


# ---------------------------------------------------------------------------
# TestRemoteTabular
# ---------------------------------------------------------------------------


class TestRemoteTabular:

    def _make_ipc(self, path: str = "/tab/data.ipc") -> Any:
        p = _StubRemotePath(path, singleton_ttl=False)
        return p.as_media("arrow")

    def _make_parquet(self, path: str = "/tab/data.parquet") -> Any:
        p = _StubRemotePath(path, singleton_ttl=False)
        return p.as_media("parquet")

    def test_write_table_arrow_then_read_back(self) -> None:
        leaf = self._make_ipc()
        table = pa.table({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.num_rows == 3
        assert result.column("x").to_pylist() == [1, 2, 3]
        assert result.column("y").to_pylist() == ["a", "b", "c"]

    def test_write_parquet_read_arrow_roundtrip(self) -> None:
        leaf = self._make_parquet()
        table = pa.table({"id": [10, 20], "val": ["foo", "bar"]})
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.num_rows == 2
        assert result.column("id").to_pylist() == [10, 20]

    def test_collect_schema_from_remote_file(self) -> None:
        leaf = self._make_ipc("/tab/schema.ipc")
        table = pa.table({
            "a": pa.array([1, 2], type=pa.int64()),
            "b": pa.array(["x", "y"], type=pa.utf8()),
        })
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        schema = leaf.collect_schema()
        assert "a" in schema
        assert "b" in schema

    def test_overwrite_replaces_table(self) -> None:
        leaf = self._make_ipc("/tab/over.ipc")
        leaf.write_arrow_table(pa.table({"v": [1, 2, 3]}), mode=Mode.OVERWRITE)
        assert leaf.read_arrow_table().num_rows == 3
        leaf.write_arrow_table(pa.table({"v": [99]}), mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.num_rows == 1
        assert result.column("v").to_pylist() == [99]

    def test_large_table_roundtrip(self) -> None:
        leaf = self._make_ipc("/tab/large.ipc")
        n = 1000
        table = pa.table({
            "id": list(range(n)),
            "val": [f"row_{i}" for i in range(n)],
        })
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.num_rows == n
        assert result.column("id").to_pylist()[-1] == n - 1


# ---------------------------------------------------------------------------
# TestRemoteStat
# ---------------------------------------------------------------------------


class TestRemoteStat:

    def test_stat_returns_iostats_with_size_and_mtime(self) -> None:
        p = _make("/stat/probe.bin")
        _StubRemotePath._STORAGE["/stat/probe.bin"] = b"twelve bytes"
        stat = p._stat()
        assert stat.size == 12
        assert stat.kind == IOKind.FILE
        assert stat.mtime > 0

    def test_stat_cache_avoids_repeated_backend_calls(self) -> None:
        p = _make("/stat/cached.bin")
        _StubRemotePath._STORAGE["/stat/cached.bin"] = b"data"

        call_count = 0
        original_stat = _StubRemotePath._stat_uncached

        def counting_stat(self_inner):
            nonlocal call_count
            call_count += 1
            return original_stat(self_inner)

        with patch.object(_StubRemotePath, "_stat_uncached", counting_stat):
            _ = p._stat()
            _ = p._stat()
            _ = p._stat()
        assert call_count == 1, "Stat cache should prevent repeated backend calls"

    def test_invalidate_singleton_clears_stat_cache(self) -> None:
        p = _make("/stat/inval.bin")
        _StubRemotePath._STORAGE["/stat/inval.bin"] = b"hello"
        _ = p._stat()
        assert p._stat_cached is not None

        p.invalidate_singleton()
        assert p._stat_cached is None


# ---------------------------------------------------------------------------
# TestRemoteSingleton
# ---------------------------------------------------------------------------


class TestRemoteSingleton:

    def test_same_url_same_instance_within_ttl(self) -> None:
        a = _StubRemotePath(
            url=URL(scheme="stub", host="store", path="/single/a.bin"),
            singleton_ttl=300,
        )
        b = _StubRemotePath(
            url=URL(scheme="stub", host="store", path="/single/a.bin"),
            singleton_ttl=300,
        )
        assert a is b

    def test_different_url_different_instance(self) -> None:
        a = _StubRemotePath(
            url=URL(scheme="stub", host="store", path="/single/a.bin"),
            singleton_ttl=300,
        )
        b = _StubRemotePath(
            url=URL(scheme="stub", host="store", path="/single/b.bin"),
            singleton_ttl=300,
        )
        assert a is not b

    def test_expired_ttl_creates_new_instance(self) -> None:
        a = _StubRemotePath(
            url=URL(scheme="stub", host="store", path="/single/expire.bin"),
            singleton_ttl=0.001,
        )
        # Let the TTL expire.
        time.sleep(0.05)
        b = _StubRemotePath(
            url=URL(scheme="stub", host="store", path="/single/expire.bin"),
            singleton_ttl=0.001,
        )
        assert a is not b


# ---------------------------------------------------------------------------
# TestRemoteDirectory
# ---------------------------------------------------------------------------


class TestRemoteDirectory:

    def test_mkdir_and_iterdir(self) -> None:
        d = _make("/dir/parent")
        d.mkdir()
        # Plant two children in the storage.
        _StubRemotePath._STORAGE["/dir/parent/one.txt"] = b"1"
        _StubRemotePath._STORAGE["/dir/parent/two.txt"] = b"2"
        children = sorted(c.name for c in d.iterdir())
        assert children == ["one.txt", "two.txt"]

    def test_remove_file_deletes(self) -> None:
        p = _make("/dir/doomed.bin")
        _StubRemotePath._STORAGE["/dir/doomed.bin"] = b"bye"
        assert p.exists()
        p.remove()
        assert "/dir/doomed.bin" not in _StubRemotePath._STORAGE

    def test_remove_dir_deletes_prefix(self) -> None:
        _StubRemotePath._STORAGE["/dir/sub/a.txt"] = b"a"
        _StubRemotePath._STORAGE["/dir/sub/b.txt"] = b"b"
        _StubRemotePath._STORAGE["/dir/other.txt"] = b"keep"
        d = _make("/dir/sub")
        d.remove(recursive=True)
        assert "/dir/sub/a.txt" not in _StubRemotePath._STORAGE
        assert "/dir/sub/b.txt" not in _StubRemotePath._STORAGE
        assert "/dir/other.txt" in _StubRemotePath._STORAGE

    def test_remove_ignores_stale_missing_cache(self) -> None:
        # A read primes the stat cache as MISSING, then the object appears
        # on the backend (another instance, node, or external tool). The
        # delete must observe current state — not the stale snapshot — and
        # actually remove the object instead of silently no-op'ing.
        p = _make("/dir/race.bin")
        assert not p.exists()  # caches MISSING within the TTL window
        _StubRemotePath._STORAGE["/dir/race.bin"] = b"appeared"
        p.remove()
        assert "/dir/race.bin" not in _StubRemotePath._STORAGE

    def test_exists_fresh_after_remove(self) -> None:
        p = _make("/dir/gone.bin")
        _StubRemotePath._STORAGE["/dir/gone.bin"] = b"here"
        assert p.exists()
        p.remove()
        assert not p.exists()

    def test_unlink_ignores_stale_missing_cache(self) -> None:
        p = _make("/dir/unlink_race.bin")
        assert not p.exists()  # caches MISSING
        _StubRemotePath._STORAGE["/dir/unlink_race.bin"] = b"appeared"
        p.unlink()
        assert "/dir/unlink_race.bin" not in _StubRemotePath._STORAGE
