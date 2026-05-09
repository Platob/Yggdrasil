"""Mock-driven tests for :class:`VolumePath`."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from yggdrasil.databricks.fs import VolumePath
from yggdrasil.io.io_stats import IOKind


class NotFound(Exception):
    pass


class InternalError(Exception):
    pass


class PermissionDenied(Exception):
    pass


@pytest.fixture(autouse=True)
def reset_remote_stat_cache():
    from yggdrasil.io.path.remote_path import RemotePath
    RemotePath._STAT_CACHE.clear()
    yield
    RemotePath._STAT_CACHE.clear()


@pytest.fixture
def workspace():
    return MagicMock()


def _file_meta(size: int, mtime_ms: int = 0):
    return SimpleNamespace(
        content_length=size,
        modification_time=mtime_ms,
    )


class TestConstruction:

    def test_legacy_posix_string(self, workspace) -> None:
        p = VolumePath(
            "/Volumes/cat/sch/vol/data.parquet", workspace=workspace,
        )
        assert p.full_path() == "/Volumes/cat/sch/vol/data.parquet"
        assert p.api_path == "/Volumes/cat/sch/vol/data.parquet"

    def test_url_form(self, workspace) -> None:
        p = VolumePath("volumes:///cat/sch/vol/x", workspace=workspace)
        assert p.full_path() == "/Volumes/cat/sch/vol/x"


class TestStat:

    def test_existing_file(self, workspace) -> None:
        workspace.files.get_metadata.return_value = _file_meta(42)
        p = VolumePath("/Volumes/c/s/v/x", workspace=workspace)
        s = p._stat_uncached()
        assert s.kind is IOKind.FILE
        assert s.size == 42

    def test_directory_fallback(self, workspace) -> None:
        workspace.files.get_metadata.side_effect = NotFound()
        workspace.files.get_directory_metadata.return_value = SimpleNamespace()
        p = VolumePath("/Volumes/c/s/v/dir", workspace=workspace)
        assert p._stat_uncached().kind is IOKind.DIRECTORY

    def test_missing(self, workspace) -> None:
        workspace.files.get_metadata.side_effect = NotFound()
        workspace.files.get_directory_metadata.side_effect = NotFound()
        p = VolumePath("/Volumes/c/s/v/x", workspace=workspace)
        assert p._stat_uncached().kind is IOKind.MISSING


class TestRead:

    def test_full_object_read(self, workspace) -> None:
        workspace.files.get_metadata.return_value = _file_meta(5)
        body = SimpleNamespace(read=lambda: b"hello")
        workspace.files.download.return_value = SimpleNamespace(contents=body)

        p = VolumePath("/Volumes/c/s/v/x", workspace=workspace)
        assert p.read_bytes() == b"hello"
        workspace.files.download.assert_called_once_with("/Volumes/c/s/v/x")

    def test_missing_raises(self, workspace) -> None:
        workspace.files.get_metadata.return_value = _file_meta(5)
        workspace.files.download.side_effect = NotFound()
        p = VolumePath("/Volumes/c/s/v/x", workspace=workspace)
        with pytest.raises(FileNotFoundError):
            p.read_bytes()


class TestWrite:

    def test_overwrite(self, workspace) -> None:
        # Initial probe: missing.
        workspace.files.get_metadata.side_effect = NotFound()
        workspace.files.get_directory_metadata.side_effect = NotFound()
        p = VolumePath("/Volumes/c/s/v/x", workspace=workspace)
        p.write_bytes(b"abcdef")
        kwargs = workspace.files.upload.call_args.kwargs
        assert kwargs["file_path"] == "/Volumes/c/s/v/x"
        assert kwargs["overwrite"] is True
        # ``contents`` is a stdlib ``BytesIO`` — read it for the
        # payload we sent.
        assert kwargs["contents"].getvalue() == b"abcdef"

    def test_pwrite_does_rmw(self, workspace) -> None:
        workspace.files.get_metadata.return_value = _file_meta(5)
        body = SimpleNamespace(read=lambda: b"abcde")
        workspace.files.download.return_value = SimpleNamespace(contents=body)
        p = VolumePath("/Volumes/c/s/v/x", workspace=workspace)
        p.pwrite(b"XX", 1)
        sent = workspace.files.upload.call_args.kwargs["contents"].getvalue()
        assert sent == b"aXXde"


class TestMutators:

    def test_unlink(self, workspace) -> None:
        workspace.files.get_metadata.return_value = _file_meta(0)
        p = VolumePath("/Volumes/c/s/v/x", workspace=workspace)
        p.unlink()
        workspace.files.delete.assert_called_once_with("/Volumes/c/s/v/x")

    def test_mkdir(self, workspace) -> None:
        p = VolumePath("/Volumes/c/s/v/folder", workspace=workspace)
        p.mkdir()
        workspace.files.create_directory.assert_called_once_with(
            "/Volumes/c/s/v/folder",
        )


class TestStagingPath:

    def test_default_layout(self, workspace) -> None:
        p = VolumePath.staging_path(
            catalog_name="cat",
            schema_name="sch",
            resource_name="tbl",
            workspace=workspace,
        )
        full = p.full_path()
        assert full.startswith("/Volumes/cat/sch/tmp/.sql/cat/sch/tbl/part-")
        assert full.endswith(".parquet")
        assert p.temporary is True
        assert p.workspace is workspace

    def test_temporary_false(self, workspace) -> None:
        p = VolumePath.staging_path(
            catalog_name="cat",
            schema_name="sch",
            temporary=False,
            workspace=workspace,
        )
        assert p.temporary is False
        assert "/cat/sch/tmp/.sql/cat/sch/default/" in p.full_path()

    def test_unique_per_call(self, workspace) -> None:
        a = VolumePath.staging_path(
            catalog_name="c", schema_name="s", workspace=workspace,
        )
        b = VolumePath.staging_path(
            catalog_name="c", schema_name="s", workspace=workspace,
        )
        assert a.full_path() != b.full_path()

    def test_client_aggregator(self, workspace) -> None:
        client = MagicMock()
        client.workspace_client.return_value = workspace
        p = VolumePath.staging_path(
            catalog_name="c", schema_name="s", client=client,
        )
        assert p.workspace is workspace

    def test_segments_are_sanitized(self, workspace) -> None:
        p = VolumePath.staging_path(
            catalog_name="`cat`",
            schema_name="  sch  ",
            resource_name="a/b",
            workspace=workspace,
        )
        full = p.full_path()
        assert "/cat/sch/tmp/.sql/cat/sch/a_b/" in full


class TestVolumeAutoCreate:
    """``_call_ensuring_parents`` should walk the cheap path first
    (``create_directory`` on the parent) and only blind-create the
    catalog / schema / managed volume when that fails NotFound."""

    def test_only_subdir_missing_skips_volume_create(self, workspace) -> None:
        # Upload fails because parent dir missing; one parent
        # ``create_directory`` is enough — no catalog/schema/volume
        # creates should happen.
        uploads = [NotFound("does not exist"), None]

        def upload(**_kwargs):
            r = uploads.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        workspace.files.upload.side_effect = upload
        p = VolumePath(
            "/Volumes/cat/sch/vol/sub/file.bin", workspace=workspace,
        )
        p.write_bytes(b"payload")

        workspace.files.create_directory.assert_called_once_with(
            "/Volumes/cat/sch/vol/sub",
        )
        workspace.catalogs.create.assert_not_called()
        workspace.schemas.create.assert_not_called()
        workspace.volumes.create.assert_not_called()
        assert workspace.files.upload.call_count == 2

    def test_volume_missing_blind_creates_catalog_schema_volume(
        self, workspace,
    ) -> None:
        # Both upload and parent ``create_directory`` fail NotFound,
        # so the volume must not exist — fall through to blind
        # catalog / schema / managed-volume creates.
        uploads = [NotFound("Volume does not exist"), None]
        create_dirs = [NotFound("does not exist"), None]

        def upload(**_kwargs):
            r = uploads.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        def create_directory(_path):
            r = create_dirs.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        workspace.files.upload.side_effect = upload
        workspace.files.create_directory.side_effect = create_directory

        p = VolumePath(
            "/Volumes/cat/sch/vol/sub/file.bin", workspace=workspace,
        )
        p.write_bytes(b"payload")

        # No probes — straight to creates.
        workspace.catalogs.get.assert_not_called()
        workspace.schemas.get.assert_not_called()
        workspace.volumes.read.assert_not_called()

        workspace.catalogs.create.assert_called_once_with(name="cat")
        workspace.schemas.create.assert_called_once_with(
            name="sch", catalog_name="cat",
        )
        vol_kwargs = workspace.volumes.create.call_args.kwargs
        assert vol_kwargs["catalog_name"] == "cat"
        assert vol_kwargs["schema_name"] == "sch"
        assert vol_kwargs["name"] == "vol"
        vt = vol_kwargs["volume_type"]
        assert getattr(vt, "name", str(vt)).upper() == "MANAGED"

    def test_already_exists_swallowed_by_blind_creates(self, workspace) -> None:
        # Volume create races with another caller — ``AlreadyExists``
        # is treated as success, no retry storm.
        uploads = [NotFound("Volume does not exist"), None]
        create_dirs = [NotFound("does not exist"), None]

        def upload(**_kwargs):
            r = uploads.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        def create_directory(_path):
            r = create_dirs.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        class AlreadyExists(Exception):
            pass

        workspace.files.upload.side_effect = upload
        workspace.files.create_directory.side_effect = create_directory
        workspace.catalogs.create.side_effect = AlreadyExists()
        workspace.schemas.create.side_effect = AlreadyExists()
        workspace.volumes.create.side_effect = AlreadyExists()

        p = VolumePath(
            "/Volumes/cat/sch/vol/sub/file.bin", workspace=workspace,
        )
        p.write_bytes(b"payload")  # should not raise

    def test_propagates_when_not_a_volume_path(self, workspace) -> None:
        # Path too shallow to address a volume — auto-create can't help,
        # so the original error must surface.
        workspace.files.upload.side_effect = NotFound("does not exist")
        p = VolumePath("/Volumes/onlycat", workspace=workspace)
        with pytest.raises(NotFound):
            p.write_bytes(b"x")
        workspace.volumes.create.assert_not_called()


class TestRetryPolicy:

    @pytest.fixture
    def sleeps(self):
        recorded: list[float] = []
        return recorded, recorded.append

    def test_internal_error_retries(self, workspace, sleeps) -> None:
        recorded, spy = sleeps
        attempts = [InternalError(), InternalError(), _file_meta(3)]

        def get_metadata(path):
            r = attempts.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        workspace.files.get_metadata.side_effect = get_metadata
        p = VolumePath(
            "/Volumes/c/s/v/x", workspace=workspace, retry_sleep=spy,
        )
        assert p.size == 3
        assert recorded == [1.0, 2.0]

    def test_permission_retries_once(self, workspace, sleeps) -> None:
        recorded, spy = sleeps
        attempts = [PermissionDenied(), _file_meta(2)]

        def get_metadata(path):
            r = attempts.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        workspace.files.get_metadata.side_effect = get_metadata
        p = VolumePath(
            "/Volumes/c/s/v/x", workspace=workspace, retry_sleep=spy,
        )
        assert p.size == 2
        assert recorded == [1.0]
