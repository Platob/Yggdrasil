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
