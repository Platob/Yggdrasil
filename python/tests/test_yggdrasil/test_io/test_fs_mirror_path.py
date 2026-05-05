"""Tests for :class:`yggdrasil.io.fs.mirror_path.MirrorPath`."""

from __future__ import annotations

import time

import pytest

from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.fs import LocalPath, Path
from yggdrasil.io.fs.mirror import _MIRROR_FRESH, _MIRROR_SWEPT
from yggdrasil.io.fs.mirror_path import MirrorPath
from yggdrasil.io.path_stat import PathKind, PathStats
from yggdrasil.io.url import URL


# ---------------------------------------------------------------------------
# Fake non-local Path backed by an in-memory dict (with operation counters).
# ---------------------------------------------------------------------------


class FakeRemote(Path):
    scheme = "fakeremote2"
    __slots__ = ()

    @property
    def is_local(self) -> bool:
        return False

    def full_path(self) -> str:
        host = self.url.host or "h"
        return f"fakeremote2://{host}{self.url.path or '/'}"

    def _stat(self) -> PathStats:
        FAKE_COUNTERS["stat"] = FAKE_COUNTERS.get("stat", 0) + 1
        entry = FAKE_STORE.get(self.full_path())
        if entry is None:
            return PathStats(kind=PathKind.MISSING)
        data, mtime = entry
        return PathStats(kind=PathKind.FILE, size=len(data), mtime=float(mtime))

    def _ls(self, recursive=False, allow_not_found=True):
        return iter(())

    def _mkdir(self, parents=True, exist_ok=True):
        return None

    def _remove_file(self, allow_not_found=True):
        FAKE_COUNTERS["remove"] = FAKE_COUNTERS.get("remove", 0) + 1
        FAKE_STORE.pop(self.full_path(), None)

    def _remove_dir(self, recursive=True, allow_not_found=True, with_root=True):
        return None

    def _open(self, mode="rb", **kwargs):
        return BytesIO(self, mode=mode)

    def _pread(self):
        FAKE_COUNTERS["pread"] = FAKE_COUNTERS.get("pread", 0) + 1
        entry = FAKE_STORE.get(self.full_path())
        if entry is None:
            raise FileNotFoundError(self.full_path())
        data, _ = entry
        bio = BytesIO()
        bio.open()
        if data:
            bio.write(data)
            bio.seek(0)
        return bio

    def _pwrite(self, data):
        # Optional sleep so async upload races are observable in tests.
        time.sleep(FAKE_UPLOAD_DELAY[0])
        FAKE_COUNTERS["write_bytes"] = FAKE_COUNTERS.get("write_bytes", 0) + 1
        if not data.opened:
            data.open()
        size = data.size
        payload = data.pread(size, 0) if size else b""
        FAKE_STORE[self.full_path()] = (bytes(payload), time.time())
        return len(payload)

    def pread(self, n, pos, *, default=...):
        FAKE_COUNTERS["pread"] = FAKE_COUNTERS.get("pread", 0) + 1
        entry = FAKE_STORE.get(self.full_path())
        if entry is None:
            if default is ...:
                raise FileNotFoundError(self.full_path())
            return default
        data, _ = entry
        if n < 0:
            return data[pos:]
        return data[pos:pos + n]

    def read_bytes(self, *, raise_error=True):
        FAKE_COUNTERS["read_bytes"] = FAKE_COUNTERS.get("read_bytes", 0) + 1
        entry = FAKE_STORE.get(self.full_path())
        if entry is None:
            if raise_error:
                raise FileNotFoundError(self.full_path())
            return b""
        return entry[0]

    def write_bytes(self, data, *, mode="wb", parents=True):
        # Optional sleep so async upload races are observable in tests.
        time.sleep(FAKE_UPLOAD_DELAY[0])
        FAKE_COUNTERS["write_bytes"] = FAKE_COUNTERS.get("write_bytes", 0) + 1
        FAKE_STORE[self.full_path()] = (bytes(data), time.time())
        return len(data)

    def _from_url(self, url):
        return FakeRemote(url=url)


FAKE_STORE: dict = {}
FAKE_COUNTERS: dict = {}
FAKE_UPLOAD_DELAY: list = [0.0]


@pytest.fixture(autouse=True)
def _reset():
    FAKE_STORE.clear()
    FAKE_COUNTERS.clear()
    FAKE_UPLOAD_DELAY[0] = 0.0
    _MIRROR_FRESH.clear()
    _MIRROR_SWEPT.clear()
    yield
    FAKE_STORE.clear()
    FAKE_COUNTERS.clear()
    FAKE_UPLOAD_DELAY[0] = 0.0
    _MIRROR_FRESH.clear()
    _MIRROR_SWEPT.clear()


@pytest.fixture
def remote() -> FakeRemote:
    p = FakeRemote(url=URL(scheme="fakeremote2", host="b", path="/d/file.bin"))
    FAKE_STORE[p.full_path()] = (b"upstream", 1700000000.0)
    return p


@pytest.fixture
def mirror_root(tmp_path) -> LocalPath:
    return LocalPath.from_(tmp_path)


# ---------------------------------------------------------------------------
# Identity / wrapping
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_full_path_matches_remote(self, remote, mirror_root):
        m = MirrorPath(remote, root=mirror_root)
        assert m.full_path() == remote.full_path()

    def test_remote_property(self, remote, mirror_root):
        m = MirrorPath(remote, root=mirror_root)
        assert m.remote is remote

    def test_idempotent_wrap(self, remote, mirror_root):
        m = MirrorPath(remote, root=mirror_root)
        assert MirrorPath(m) is m

    def test_url_matches_remote(self, remote, mirror_root):
        m = MirrorPath(remote, root=mirror_root)
        assert m.url == remote.url
        assert m.name == remote.name

    def test_handles_returns_false(self):
        # Never claims via dispatch — must be built explicitly.
        assert MirrorPath.handles("fakeremote2://h/x") is False

    def test_path_factory_method(self, remote, mirror_root):
        m = remote.as_mirror(root=mirror_root, ttl=30.0)
        assert isinstance(m, MirrorPath)
        assert m.mirror_ttl == 30.0


# ---------------------------------------------------------------------------
# Reads — through the local mirror
# ---------------------------------------------------------------------------


class TestReads:
    def test_read_bytes_uses_mirror(self, remote, mirror_root):
        m = MirrorPath(remote, root=mirror_root)
        assert m.read_bytes() == b"upstream"
        assert FAKE_COUNTERS["read_bytes"] == 1

    def test_repeat_read_no_round_trip(self, remote, mirror_root):
        m = MirrorPath(remote, root=mirror_root)
        m.read_bytes()
        FAKE_COUNTERS.clear()
        m.read_bytes()
        assert FAKE_COUNTERS.get("read_bytes", 0) == 0
        assert FAKE_COUNTERS.get("stat", 0) == 0

    def test_pread_routes_through_mirror(self, remote, mirror_root):
        m = MirrorPath(remote, root=mirror_root)
        assert m.pread(4, 0) == b"upst"
        assert m.pread(4, 4) == b"ream"

    def test_read_text(self, remote, mirror_root):
        m = MirrorPath(remote, root=mirror_root)
        assert m.read_text() == "upstream"

    def test_open_io_read(self, remote, mirror_root):
        m = MirrorPath(remote, root=mirror_root)
        with m.open_io("rb") as fh:
            assert fh.read() == b"upstream"


# ---------------------------------------------------------------------------
# Writes — local first, async upstream
# ---------------------------------------------------------------------------


class TestWrites:
    def test_write_bytes_lands_locally_immediately(self, remote, mirror_root):
        m = MirrorPath(remote, root=mirror_root)
        m.write_bytes(b"new-bytes")
        # Mirror file exists immediately.
        assert m.mirror_local.exists()
        assert m.mirror_local.read_bytes() == b"new-bytes"

    def test_write_then_flush_uploads(self, remote, mirror_root):
        m = MirrorPath(remote, root=mirror_root)
        m.write_bytes(b"new-bytes")
        m.flush()
        assert FAKE_STORE[remote.full_path()][0] == b"new-bytes"
        assert m.pending_uploads == 0

    def test_pending_count_reflects_in_flight(self, remote, mirror_root):
        FAKE_UPLOAD_DELAY[0] = 0.05
        m = MirrorPath(remote, root=mirror_root)
        m.write_bytes(b"v1")
        # Right after enqueue, the upload may still be running.
        assert m.pending_uploads >= 0
        m.flush()
        assert m.pending_uploads == 0

    def test_pwrite_uploads_full_local(self, remote, mirror_root):
        m = MirrorPath(remote, root=mirror_root)
        m.pwrite(b"AAAA", 0)
        m.flush()
        # pwrite reuses a freshly-mirrored copy of the upstream object.
        # Original was b"upstream" (8 bytes); patching pos=0..3 yields
        # b"AAAAream".
        assert FAKE_STORE[remote.full_path()][0] == b"AAAAream"

    def test_write_stream(self, remote, mirror_root):
        m = MirrorPath(remote, root=mirror_root)
        src = BytesIO(b"streamed")
        m.write_stream(src)
        m.flush()
        assert FAKE_STORE[remote.full_path()][0] == b"streamed"

    def test_truncate(self, remote, mirror_root):
        m = MirrorPath(remote, root=mirror_root)
        m.read_bytes()  # prime mirror
        m.truncate(4)
        m.flush()
        assert FAKE_STORE[remote.full_path()][0] == b"upst"


# ---------------------------------------------------------------------------
# Lifecycle — close drains
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_close_drains_uploads(self, remote, mirror_root):
        FAKE_UPLOAD_DELAY[0] = 0.02
        m = MirrorPath(remote, root=mirror_root)
        m.write_bytes(b"bye")
        m.close()
        assert FAKE_STORE[remote.full_path()][0] == b"bye"

    def test_context_manager_drains(self, remote, mirror_root):
        FAKE_UPLOAD_DELAY[0] = 0.02
        with MirrorPath(remote, root=mirror_root) as m:
            m.write_bytes(b"ctx")
        assert FAKE_STORE[remote.full_path()][0] == b"ctx"

    def test_remove_drops_local_mirror(self, remote, mirror_root):
        m = MirrorPath(remote, root=mirror_root)
        m.read_bytes()
        local = m.mirror_local
        assert local.exists()
        m._remove_file()
        assert not local.exists()
        assert remote.full_path() not in FAKE_STORE


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestErrors:
    def test_upload_failure_recorded(self, remote, mirror_root):
        m = MirrorPath(remote, root=mirror_root)

        original = FakeRemote.write_bytes

        def boom(self, *a, **kw):
            raise OSError("upstream is angry")

        try:
            FakeRemote.write_bytes = boom
            m.write_bytes(b"x")
            m.flush()
            assert isinstance(m.last_upload_error, OSError)
        finally:
            FakeRemote.write_bytes = original

    def test_flush_raise_error(self, remote, mirror_root):
        m = MirrorPath(remote, root=mirror_root)
        original = FakeRemote.write_bytes

        def boom(self, *a, **kw):
            raise OSError("nope")

        try:
            FakeRemote.write_bytes = boom
            m.write_bytes(b"x")
            with pytest.raises(OSError):
                m.flush(raise_error=True)
        finally:
            FakeRemote.write_bytes = original
