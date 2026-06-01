"""Cluster-mount fast path tests for :class:`VolumePath`.

Inside a Databricks runtime, ``/Volumes/<cat>/<sch>/<vol>/...`` is
exposed as a native FUSE mount. :class:`VolumePath` transparently
detects the mount and short-circuits ``_stat`` / ``_read_mv`` /
``_ls`` / ``_upload`` onto stdlib filesystem calls so reads and writes
run at kernel speed instead of paying a Files API round trip per
operation.

The mount detection lives behind
:func:`yggdrasil.databricks.fs.volume_path._local_mount_available`
which checks ``DatabricksClient.is_in_databricks_environment()`` AND
``os.path.isdir('/Volumes')`` — both have to be true. Tests
monkey-patch the helper to emulate "on cluster" without actually
running inside DBR.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from yggdrasil.databricks.fs import volume_path as vp_module
from yggdrasil.databricks.fs.volume_path import VolumePath
from yggdrasil.databricks.volume.volumes import Volumes
from yggdrasil.io.io_stats import IOKind

from tests.test_yggdrasil.test_databricks._files_fake import wire_files_session


@pytest.fixture(autouse=True)
def reset_volume_path_caches():
    VolumePath._INSTANCES.clear()
    vp_module._reset_local_mount_probe()
    yield
    VolumePath._INSTANCES.clear()
    vp_module._reset_local_mount_probe()


@pytest.fixture
def fake_volume_root(tmp_path, monkeypatch):
    """Stage a fake ``/Volumes/cat/sch/vol`` tree under ``tmp_path``
    and redirect :attr:`VolumePath.api_path` to it. The fast path keys
    off ``api_path`` for every IO call, so rewriting that one property
    is enough to make every syscall hit the test directory."""
    root = tmp_path / "Volumes" / "cat" / "sch" / "vol"
    root.mkdir(parents=True)

    monkeypatch.setattr(
        vp_module, "_local_mount_available", lambda: True,
    )

    original = VolumePath.api_path

    def fake_api_path(self):
        # Rewrite ``/Volumes/...`` to point under ``tmp_path``.
        full = original.fget(self)
        return str(tmp_path / full.lstrip("/"))

    monkeypatch.setattr(VolumePath, "api_path", property(fake_api_path))
    return root


@pytest.fixture
def service():
    # Wire the Files-API HTTP seam so the off-cluster fallback tests
    # route through the fake session; on-cluster tests never touch it.
    client = wire_files_session(MagicMock())
    svc = MagicMock(spec=Volumes)
    svc.client = client
    return svc


# ===========================================================================
# Probe semantics
# ===========================================================================


class TestLocalMountProbe:

    def test_probe_returns_false_when_not_in_runtime(self, monkeypatch):
        # The DATABRICKS_RUNTIME_VERSION env var is the gate — without
        # it the probe must short-circuit to False regardless of
        # ``/Volumes`` existing on disk.
        from yggdrasil.databricks.client import DatabricksClient
        monkeypatch.setattr(
            DatabricksClient, "is_in_databricks_environment",
            staticmethod(lambda: False),
        )
        assert vp_module._local_mount_available() is False

    def test_probe_returns_false_when_volumes_dir_missing(
        self, monkeypatch,
    ):
        from yggdrasil.databricks.client import DatabricksClient
        monkeypatch.setattr(
            DatabricksClient, "is_in_databricks_environment",
            staticmethod(lambda: True),
        )
        # Off-cluster CI machines don't have ``/Volumes`` — the
        # ``isdir`` check is the second gate.
        import os
        monkeypatch.setattr(os.path, "isdir", lambda p: False)
        vp_module._reset_local_mount_probe()
        assert vp_module._local_mount_available() is False

    def test_probe_result_is_cached(self, monkeypatch):
        # Once the probe answers, repeated calls must not re-check the
        # environment — this matters on the hot stat path (called once
        # per ``exists`` / ``size``).
        from yggdrasil.databricks.client import DatabricksClient
        calls = {"n": 0}

        def is_runtime():
            calls["n"] += 1
            return False

        monkeypatch.setattr(
            DatabricksClient, "is_in_databricks_environment",
            staticmethod(is_runtime),
        )
        vp_module._local_mount_available()
        vp_module._local_mount_available()
        vp_module._local_mount_available()
        assert calls["n"] == 1


# ===========================================================================
# _stat_uncached uses os.stat when mounted
# ===========================================================================


class TestStatFastPath:

    def test_stat_file_uses_local_mount(
        self, fake_volume_root, service,
    ):
        (fake_volume_root / "data.parquet").write_bytes(b"x" * 17)
        p = VolumePath(
            "/Volumes/cat/sch/vol/data.parquet", service=service,
        )
        stats = p._stat_uncached()
        assert stats.kind is IOKind.FILE
        assert stats.size == 17
        # The Files API mock should NOT have been touched — the kernel
        # mount took the whole call.
        service.client.workspace_client.assert_not_called()

    def test_stat_directory_uses_local_mount(
        self, fake_volume_root, service,
    ):
        (fake_volume_root / "subdir").mkdir()
        p = VolumePath(
            "/Volumes/cat/sch/vol/subdir", service=service,
        )
        stats = p._stat_uncached()
        assert stats.kind is IOKind.DIRECTORY
        service.client.workspace_client.assert_not_called()

    def test_stat_missing_returns_missing(
        self, fake_volume_root, service,
    ):
        p = VolumePath(
            "/Volumes/cat/sch/vol/nope.bin", service=service,
        )
        stats = p._stat_uncached()
        assert stats.kind is IOKind.MISSING
        service.client.workspace_client.assert_not_called()


# ===========================================================================
# _read_mv reads off the kernel mount
# ===========================================================================


class TestReadMvFastPath:

    def test_whole_file_read(self, fake_volume_root, service):
        (fake_volume_root / "data.bin").write_bytes(b"hello-volume")
        p = VolumePath(
            "/Volumes/cat/sch/vol/data.bin", service=service,
        )
        mv = p._read_mv(-1, 0)
        assert bytes(mv) == b"hello-volume"
        service.client.workspace_client.assert_not_called()

    def test_positional_read_honours_offset_and_length(
        self, fake_volume_root, service,
    ):
        (fake_volume_root / "data.bin").write_bytes(b"abcdefghij")
        p = VolumePath(
            "/Volumes/cat/sch/vol/data.bin", service=service,
        )
        mv = p._read_mv(3, 4)
        # ``files.download`` doesn't expose range reads; the kernel
        # mount does — verify we ride it.
        assert bytes(mv) == b"efg"

    def test_read_missing_raises_filenotfound(
        self, fake_volume_root, service,
    ):
        p = VolumePath(
            "/Volumes/cat/sch/vol/missing.bin", service=service,
        )
        with pytest.raises(FileNotFoundError):
            p._read_mv(-1, 0)


# ===========================================================================
# _ls walks the kernel mount
# ===========================================================================


class TestLsFastPath:

    def test_ls_returns_children_with_seeded_stat_cache(
        self, fake_volume_root, service,
    ):
        (fake_volume_root / "a.bin").write_bytes(b"AA")
        (fake_volume_root / "b.bin").write_bytes(b"BBBB")
        (fake_volume_root / "sub").mkdir()
        p = VolumePath(
            "/Volumes/cat/sch/vol", service=service,
        )
        children = list(p._ls())
        names = sorted(c.name for c in children)
        assert names == ["a.bin", "b.bin", "sub"]
        # The seeded ``_stat_cached`` from the fast path means ``size``
        # is a local hit, not a workspace round trip.
        by_name = {c.name: c for c in children}
        assert by_name["a.bin"]._stat_cached.kind is IOKind.FILE
        assert by_name["a.bin"]._stat_cached.size == 2
        assert by_name["sub"]._stat_cached.kind is IOKind.DIRECTORY
        service.client.workspace_client.assert_not_called()

    def test_ls_recursive_walks_subtree(
        self, fake_volume_root, service,
    ):
        (fake_volume_root / "sub").mkdir()
        (fake_volume_root / "sub" / "leaf.bin").write_bytes(b"L")
        p = VolumePath(
            "/Volumes/cat/sch/vol", service=service,
        )
        children = list(p._ls(recursive=True))
        names = sorted(c.name for c in children)
        assert names == ["leaf.bin", "sub"]


# ===========================================================================
# _upload writes through the kernel mount
# ===========================================================================


class TestUploadFastPath:

    def test_upload_bytes_writes_to_mount(
        self, fake_volume_root, service,
    ):
        p = VolumePath(
            "/Volumes/cat/sch/vol/new.bin", service=service,
        )
        n = p._upload(b"hello")
        assert n == 5
        assert (fake_volume_root / "new.bin").read_bytes() == b"hello"
        service.client.workspace_client.assert_not_called()

    def test_upload_creates_missing_parents(
        self, fake_volume_root, service,
    ):
        p = VolumePath(
            "/Volumes/cat/sch/vol/deep/sub/new.bin", service=service,
        )
        p._upload(b"deep-write")
        assert (
            fake_volume_root / "deep" / "sub" / "new.bin"
        ).read_bytes() == b"deep-write"

    def test_upload_stream_rewinds_to_zero(
        self, fake_volume_root, service,
    ):
        # A stream parked at EOF (a just-written buffer the caller didn't
        # rewind) must still upload the whole object — the mount path used
        # to read from the current position and truncate it to empty, while
        # the off-cluster PUT stayed correct. Both must write all the bytes.
        import io as _io

        buf = _io.BytesIO(b"payload-bytes")
        buf.seek(0, _io.SEEK_END)            # caller left the cursor at EOF
        p = VolumePath("/Volumes/cat/sch/vol/stream.bin", service=service)
        n = p._upload(buf)
        assert n == len(b"payload-bytes")
        assert (fake_volume_root / "stream.bin").read_bytes() == b"payload-bytes"

    def test_upload_non_seekable_readable_stream(
        self, fake_volume_root, service,
    ):
        # A readable but non-seekable stream must be drained via read(),
        # not coerced through bytes(stream) (which wouldn't read it at all).
        class _Pipe:
            def __init__(self, data: bytes) -> None:
                self._data = data
                self._read = False

            def read(self, n: int = -1) -> bytes:
                if self._read:
                    return b""
                self._read = True
                return self._data

        p = VolumePath("/Volumes/cat/sch/vol/pipe.bin", service=service)
        n = p._upload(_Pipe(b"piped"))
        assert n == 5
        assert (fake_volume_root / "pipe.bin").read_bytes() == b"piped"


# ===========================================================================
# Off-cluster: the Files API is still the source of truth
# ===========================================================================


# ===========================================================================
# _mkdir / _remove_file / _remove_dir fast paths
# ===========================================================================


class TestMkdirFastPath:

    def test_mkdir_creates_directory_on_mount(
        self, fake_volume_root, service,
    ):
        p = VolumePath(
            "/Volumes/cat/sch/vol/newdir", service=service,
        )
        p._mkdir(parents=False, exist_ok=False)
        assert (fake_volume_root / "newdir").is_dir()
        service.client.workspace_client.assert_not_called()

    def test_mkdir_parents_true_creates_intermediate_dirs(
        self, fake_volume_root, service,
    ):
        p = VolumePath(
            "/Volumes/cat/sch/vol/deep/nested/path", service=service,
        )
        p._mkdir(parents=True, exist_ok=False)
        assert (
            fake_volume_root / "deep" / "nested" / "path"
        ).is_dir()

    def test_mkdir_exist_ok_silences_already_exists(
        self, fake_volume_root, service,
    ):
        (fake_volume_root / "already").mkdir()
        p = VolumePath(
            "/Volumes/cat/sch/vol/already", service=service,
        )
        # Should NOT raise.
        p._mkdir(parents=False, exist_ok=True)

    def test_mkdir_exist_ok_false_raises_on_collision(
        self, fake_volume_root, service,
    ):
        (fake_volume_root / "already").mkdir()
        p = VolumePath(
            "/Volumes/cat/sch/vol/already", service=service,
        )
        with pytest.raises(FileExistsError):
            p._mkdir(parents=False, exist_ok=False)


class TestRemoveFileFastPath:

    def test_remove_file_unlinks_from_mount(
        self, fake_volume_root, service,
    ):
        target = fake_volume_root / "f.bin"
        target.write_bytes(b"x")
        p = VolumePath(
            "/Volumes/cat/sch/vol/f.bin", service=service,
        )
        from yggdrasil.dataclasses import WaitingConfig
        p._remove_file(missing_ok=False, wait=WaitingConfig.from_(True))
        assert not target.exists()
        service.client.workspace_client.assert_not_called()

    def test_remove_file_missing_ok_swallows_not_found(
        self, fake_volume_root, service,
    ):
        from yggdrasil.dataclasses import WaitingConfig
        p = VolumePath(
            "/Volumes/cat/sch/vol/ghost.bin", service=service,
        )
        # Should NOT raise.
        p._remove_file(missing_ok=True, wait=WaitingConfig.from_(True))

    def test_remove_file_not_missing_ok_raises(
        self, fake_volume_root, service,
    ):
        from yggdrasil.dataclasses import WaitingConfig
        p = VolumePath(
            "/Volumes/cat/sch/vol/ghost.bin", service=service,
        )
        with pytest.raises(FileNotFoundError):
            p._remove_file(
                missing_ok=False, wait=WaitingConfig.from_(True),
            )


class TestRemoveDirFastPath:

    def test_remove_empty_dir_via_rmdir(
        self, fake_volume_root, service,
    ):
        (fake_volume_root / "empty").mkdir()
        p = VolumePath(
            "/Volumes/cat/sch/vol/empty", service=service,
        )
        from yggdrasil.dataclasses import WaitingConfig
        p._remove_dir(
            recursive=False,
            missing_ok=False,
            wait=WaitingConfig.from_(True),
        )
        assert not (fake_volume_root / "empty").exists()
        service.client.workspace_client.assert_not_called()

    def test_remove_dir_recursive_wipes_subtree(
        self, fake_volume_root, service,
    ):
        sub = fake_volume_root / "tree"
        sub.mkdir()
        (sub / "a.bin").write_bytes(b"A")
        (sub / "inner").mkdir()
        (sub / "inner" / "b.bin").write_bytes(b"B")
        p = VolumePath(
            "/Volumes/cat/sch/vol/tree", service=service,
        )
        from yggdrasil.dataclasses import WaitingConfig
        p._remove_dir(
            recursive=True,
            missing_ok=False,
            wait=WaitingConfig.from_(True),
        )
        assert not sub.exists()


# ===========================================================================
# Off-cluster fallback
# ===========================================================================


class TestOSErrorFallback:

    def test_read_oserror_falls_back_to_files_api(
        self, fake_volume_root, monkeypatch, service,
    ):
        import builtins
        real_open = builtins.open

        def bad_open(path, *a, **kw):
            # Match the mount path regardless of OS separator (Windows joins
            # the fake root with ``\``).
            if "/Volumes/cat/sch/vol/explode" in str(path).replace("\\", "/"):
                raise PermissionError("simulated")
            return real_open(path, *a, **kw)

        monkeypatch.setattr(builtins, "open", bad_open)
        files = service.client.workspace_client.return_value.files
        response = MagicMock()
        response.contents = MagicMock()
        response.contents.read.return_value = b"recovered"
        response.last_modified = None
        files.download.return_value = response
        p = VolumePath(
            "/Volumes/cat/sch/vol/explode", service=service,
        )
        mv = p._read_mv(-1, 0)
        assert bytes(mv) == b"recovered"


class TestFilesApiPathStillUsedOffCluster:
    """When the local-mount probe returns False (the off-cluster
    case), every operation must still go through ``files.*`` — the
    fast path is opt-in by mount detection, never silently active."""

    def test_stat_falls_back_to_files_api(self, monkeypatch, service):
        monkeypatch.setattr(
            vp_module, "_local_mount_available", lambda: False,
        )
        # Wire a fake metadata response so ``_stat_uncached`` returns
        # a real shape instead of cascading through every probe.
        files = service.client.workspace_client.return_value.files
        files.get_metadata.return_value = MagicMock(
            content_length=42,
            modification_time=None,
        )
        p = VolumePath(
            "/Volumes/cat/sch/vol/file.bin", service=service,
        )
        stats = p._stat_uncached()
        files.get_metadata.assert_called_once_with(
            "/Volumes/cat/sch/vol/file.bin",
        )
        assert stats.size == 42
