"""Tests for :meth:`Path.local_mirror` and the mirror sweep machinery."""

from __future__ import annotations

import json
import time

import pytest

from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.fs import LocalPath, Path
from yggdrasil.io.fs.mirror import (
    _MIRROR_FRESH,
    _MIRROR_SWEPT,
    default_mirror_root,
    mirror_path_for,
    reset_mirror_sweep_state,
    sweep_mirror_root,
)
from yggdrasil.io.path_stat import PathKind, PathStats
from yggdrasil.io.url import URL


# ---------------------------------------------------------------------------
# Fake non-local Path backed by an in-memory dict.
# ---------------------------------------------------------------------------


class FakeRemote(Path):
    """In-memory ``Path`` that pretends to be remote.

    Held by ``FAKE_STORE``: a dict from full_path → (bytes, mtime).
    Each `_stat` / `read_bytes` call increments a counter so tests
    can assert "no extra round-trip happened."
    """

    scheme = "fakeremote"
    __slots__ = ()

    @property
    def is_local(self) -> bool:
        return False

    def full_path(self) -> str:
        host = self.url.host or "h"
        path = self.url.path or "/"
        return f"fakeremote://{host}{path}"

    def _stat(self) -> PathStats:
        FAKE_COUNTERS["stat"] = FAKE_COUNTERS.get("stat", 0) + 1
        entry = FAKE_STORE.get(self.full_path())
        if entry is None:
            return PathStats(kind=PathKind.MISSING)
        data, mtime = entry
        return PathStats(
            kind=PathKind.FILE, size=len(data), mtime=float(mtime),
        )

    def _ls(self, recursive=False, allow_not_found=True):
        return iter(())

    def _mkdir(self, parents=True, exist_ok=True):
        return None

    def _remove_file(self, allow_not_found=True):
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
        FAKE_STORE[self.full_path()] = (bytes(data), time.time())
        return len(data)

    def _from_url(self, url):
        return FakeRemote(url=url)


FAKE_STORE: dict = {}
FAKE_COUNTERS: dict = {}


@pytest.fixture(autouse=True)
def _reset_state():
    """Each test starts with a clean fake store and clean caches."""
    FAKE_STORE.clear()
    FAKE_COUNTERS.clear()
    _MIRROR_FRESH.clear()
    _MIRROR_SWEPT.clear()
    yield
    FAKE_STORE.clear()
    FAKE_COUNTERS.clear()
    _MIRROR_FRESH.clear()
    _MIRROR_SWEPT.clear()


@pytest.fixture
def remote() -> FakeRemote:
    p = FakeRemote(url=URL(scheme="fakeremote", host="bucket", path="/data/file.bin"))
    FAKE_STORE[p.full_path()] = (b"hello-mirror", 1700000000.0)
    return p


# ---------------------------------------------------------------------------
# mirror_path / default_mirror_root
# ---------------------------------------------------------------------------


class TestMirrorPath:
    def test_default_root_under_home(self):
        root = default_mirror_root()
        assert isinstance(root, LocalPath)
        assert root.full_path().endswith("/.yggdrasil/mirror") or (
            "\\.yggdrasil\\mirror" in root.full_path()
        )

    def test_mirror_path_for_remote(self, tmp_path, remote):
        local = mirror_path_for(remote, root=LocalPath.from_(tmp_path))
        assert isinstance(local, LocalPath)
        # Layout: <root>/<scheme>/<host>/<rel>
        assert local.full_path().endswith("/fakeremote/bucket/data/file.bin")

    def test_mirror_path_for_local_is_identity(self, tmp_path):
        p = LocalPath.from_(tmp_path / "x.txt")
        assert mirror_path_for(p) is p

    def test_path_helper_method(self, tmp_path, remote):
        local = remote.mirror_path(root=LocalPath.from_(tmp_path))
        assert isinstance(local, LocalPath)


# ---------------------------------------------------------------------------
# local_mirror — fetch, hit, refresh
# ---------------------------------------------------------------------------


class TestLocalMirrorFetch:
    def test_local_path_returns_self(self, tmp_path):
        p = LocalPath.from_(tmp_path / "x.txt")
        p.write_bytes(b"hi")
        assert p.local_mirror() is p

    def test_first_call_downloads(self, tmp_path, remote):
        local = remote.local_mirror(root=LocalPath.from_(tmp_path))
        assert local.exists()
        assert local.read_bytes() == b"hello-mirror"
        assert FAKE_COUNTERS["read_bytes"] == 1

    def test_second_call_hits_in_process_cache(self, tmp_path, remote):
        root = LocalPath.from_(tmp_path)
        remote.local_mirror(root=root)
        FAKE_COUNTERS.clear()
        # Second call within TTL: no stat, no read_bytes
        remote.local_mirror(root=root)
        assert FAKE_COUNTERS.get("stat", 0) == 0
        assert FAKE_COUNTERS.get("read_bytes", 0) == 0

    def test_ttl_zero_stats_every_call(self, tmp_path, remote):
        root = LocalPath.from_(tmp_path)
        remote.local_mirror(root=root, ttl=0)
        FAKE_COUNTERS.clear()
        remote.local_mirror(root=root, ttl=0)
        # Stat fires; download skipped because sidecar matches.
        assert FAKE_COUNTERS["stat"] >= 1
        assert FAKE_COUNTERS.get("read_bytes", 0) == 0

    def test_remote_change_triggers_refresh(self, tmp_path, remote):
        root = LocalPath.from_(tmp_path)
        remote.local_mirror(root=root)
        # Mutate remote out-of-band, drop process verdict.
        FAKE_STORE[remote.full_path()] = (b"v2-content", 1700000999.0)
        remote.invalidate_mirror()
        FAKE_COUNTERS.clear()
        local = remote.local_mirror(root=root)
        assert local.read_bytes() == b"v2-content"
        assert FAKE_COUNTERS["read_bytes"] == 1

    def test_unchanged_remote_skips_download(self, tmp_path, remote):
        root = LocalPath.from_(tmp_path)
        remote.local_mirror(root=root)
        remote.invalidate_mirror()
        FAKE_COUNTERS.clear()
        # Same size + mtime → sidecar matches → no download
        remote.local_mirror(root=root)
        assert FAKE_COUNTERS.get("read_bytes", 0) == 0
        assert FAKE_COUNTERS["stat"] >= 1

    def test_force_refresh_redownloads(self, tmp_path, remote):
        root = LocalPath.from_(tmp_path)
        remote.local_mirror(root=root)
        FAKE_COUNTERS.clear()
        remote.local_mirror(root=root, force_refresh=True)
        assert FAKE_COUNTERS["read_bytes"] == 1

    def test_sidecar_records_stats(self, tmp_path, remote):
        root = LocalPath.from_(tmp_path)
        local = remote.local_mirror(root=root)
        sidecar = local.parent / f".{local.name}.ygmirror.json"
        assert sidecar.exists()
        data = json.loads(sidecar.read_bytes().decode("utf-8"))
        assert data["size"] == len(b"hello-mirror")
        assert data["kind"] == "file"
        assert float(data["mtime"]) == 1700000000.0

    def test_missing_remote_raises(self, tmp_path):
        p = FakeRemote(url=URL(scheme="fakeremote", host="b", path="/missing"))
        with pytest.raises(FileNotFoundError):
            p.local_mirror(root=LocalPath.from_(tmp_path))

    def test_serves_stale_when_remote_unreachable(self, tmp_path, remote):
        root = LocalPath.from_(tmp_path)
        local = remote.local_mirror(root=root)
        remote.invalidate_mirror()
        # Drop remote and patch _stat to raise
        FAKE_STORE.pop(remote.full_path())
        original_stat = FakeRemote._stat

        def boom(self):
            raise OSError("network down")

        try:
            FakeRemote._stat = boom
            again = remote.local_mirror(root=root)
            assert again.full_path() == local.full_path()
            assert again.read_bytes() == b"hello-mirror"
        finally:
            FakeRemote._stat = original_stat


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


class TestSweep:
    def test_sweep_removes_old_files(self, tmp_path):
        root = LocalPath.from_(tmp_path)
        old = root / "old.bin"
        new = root / "new.bin"
        old.write_bytes(b"old")
        new.write_bytes(b"new")
        # Backdate `old` to two weeks ago.
        import os as _os
        long_ago = time.time() - 14 * 24 * 3600
        _os.utime(old.full_path(), (long_ago, long_ago))
        ran = sweep_mirror_root(
            root=root, max_age=7 * 24 * 3600, force=True,
        )
        assert ran is True
        assert not old.exists()
        assert new.exists()

    def test_sweep_is_rate_limited(self, tmp_path):
        root = LocalPath.from_(tmp_path)
        root.mkdir(parents=True, exist_ok=True)
        first = sweep_mirror_root(root=root)
        second = sweep_mirror_root(root=root)
        assert first is True
        assert second is False  # rate-limited

    def test_reset_sweep_state(self, tmp_path):
        root = LocalPath.from_(tmp_path)
        root.mkdir(parents=True, exist_ok=True)
        sweep_mirror_root(root=root)
        reset_mirror_sweep_state()
        assert sweep_mirror_root(root=root) is True

    def test_local_mirror_triggers_sweep_once(self, tmp_path, remote):
        root = LocalPath.from_(tmp_path)
        # Plant an old file under the (yet-to-exist) mirror tree.
        old = root / "fakeremote" / "bucket" / "stale.bin"
        old.parent.mkdir(parents=True, exist_ok=True)
        old.write_bytes(b"old")
        import os as _os
        long_ago = time.time() - 14 * 24 * 3600
        _os.utime(old.full_path(), (long_ago, long_ago))

        remote.local_mirror(root=root, max_age=7 * 24 * 3600)
        assert not old.exists()

    def test_verdict_cache_hit_skips_sweep(self, tmp_path, remote, monkeypatch):
        """Hot loops must not pay for the sweep machinery on cache hits.

        After the first call has stamped the verdict cache, a follow-up
        :meth:`local_mirror` must short-circuit without calling
        :func:`sweep_mirror_root` at all.
        """
        from yggdrasil.io.fs import mirror as mirror_mod

        root = LocalPath.from_(tmp_path)
        # Warm the cache (this call is allowed to sweep).
        remote.local_mirror(root=root)

        sweep_calls = []
        original = mirror_mod.sweep_mirror_root

        def counting_sweep(*args, **kwargs):
            sweep_calls.append((args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(mirror_mod, "sweep_mirror_root", counting_sweep)
        # Cache-hit path: must not invoke sweep.
        remote.local_mirror(root=root)
        assert sweep_calls == []
