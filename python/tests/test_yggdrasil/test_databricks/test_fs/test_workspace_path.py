"""Mock-driven tests for :class:`WorkspacePath`."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from yggdrasil.databricks.fs import WorkspacePath
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


def _file_status(size: int) -> SimpleNamespace:
    return SimpleNamespace(
        object_type=SimpleNamespace(name="FILE"),
        size=size,
        modified_at=0,
    )


def _dir_status() -> SimpleNamespace:
    return SimpleNamespace(
        object_type=SimpleNamespace(name="DIRECTORY"),
        size=0,
        modified_at=0,
    )


class TestConstruction:

    def test_legacy_posix_string(self, workspace) -> None:
        p = WorkspacePath("/Workspace/Users/me/notebook.py", workspace=workspace)
        assert p.full_path() == "/Workspace/Users/me/notebook.py"
        assert p.api_path == "/Workspace/Users/me/notebook.py"

    def test_url_form(self, workspace) -> None:
        p = WorkspacePath("workspace:///Users/me/x", workspace=workspace)
        assert p.full_path() == "/Workspace/Users/me/x"


class TestStat:

    def test_existing_file(self, workspace) -> None:
        workspace.workspace.get_status.return_value = _file_status(42)
        p = WorkspacePath("/Workspace/x", workspace=workspace)
        s = p._stat_uncached()
        assert s.kind is IOKind.FILE
        assert s.size == 42

    def test_directory(self, workspace) -> None:
        workspace.workspace.get_status.return_value = _dir_status()
        p = WorkspacePath("/Workspace/folder", workspace=workspace)
        assert p._stat_uncached().kind is IOKind.DIRECTORY

    def test_missing(self, workspace) -> None:
        workspace.workspace.get_status.side_effect = NotFound()
        p = WorkspacePath("/Workspace/x", workspace=workspace)
        assert p._stat_uncached().kind is IOKind.MISSING


class TestRead:

    def test_read_bytes(self, workspace) -> None:
        workspace.workspace.get_status.return_value = _file_status(5)
        body = SimpleNamespace(read=lambda: b"hello")
        workspace.workspace.download.return_value = SimpleNamespace(contents=body)

        p = WorkspacePath("/Workspace/x", workspace=workspace)
        assert p.read_bytes() == b"hello"

    def test_missing_raises(self, workspace) -> None:
        workspace.workspace.get_status.return_value = _file_status(5)
        workspace.workspace.download.side_effect = NotFound()
        p = WorkspacePath("/Workspace/x", workspace=workspace)
        with pytest.raises(FileNotFoundError):
            p.read_bytes()


class TestWrite:

    def test_overwrite(self, workspace) -> None:
        workspace.workspace.get_status.side_effect = NotFound()
        p = WorkspacePath("/Workspace/x", workspace=workspace)
        p.write_bytes(b"abcdef")
        kwargs = workspace.workspace.upload.call_args.kwargs
        assert kwargs["path"] == "/Workspace/x"
        assert kwargs["overwrite"] is True
        assert kwargs["content"].getvalue() == b"abcdef"


class TestMutators:

    def test_unlink(self, workspace) -> None:
        workspace.workspace.get_status.return_value = _file_status(0)
        p = WorkspacePath("/Workspace/x", workspace=workspace)
        p.unlink()
        workspace.workspace.delete.assert_called_once_with(
            "/Workspace/x", recursive=False,
        )

    def test_remove_dir(self, workspace) -> None:
        workspace.workspace.get_status.return_value = _dir_status()
        p = WorkspacePath("/Workspace/folder", workspace=workspace)
        p.remove(recursive=True)
        workspace.workspace.delete.assert_called_once_with(
            "/Workspace/folder", recursive=True,
        )

    def test_mkdir(self, workspace) -> None:
        p = WorkspacePath("/Workspace/a/b", workspace=workspace)
        p.mkdir()
        workspace.workspace.mkdirs.assert_called_once_with("/Workspace/a/b")


class TestListing:

    def test_iterdir(self, workspace) -> None:
        workspace.workspace.list.return_value = [
            SimpleNamespace(
                path="/Workspace/Users/me/a.py",
                object_type=SimpleNamespace(name="FILE"),
            ),
            SimpleNamespace(
                path="/Workspace/Users/me/sub",
                object_type=SimpleNamespace(name="DIRECTORY"),
            ),
        ]
        p = WorkspacePath("/Workspace/Users/me", workspace=workspace)
        children = list(p.iterdir())
        assert [c.full_path() for c in children] == [
            "/Workspace/Users/me/a.py",
            "/Workspace/Users/me/sub",
        ]


class TestMePlaceholder:
    """``/Workspace/Users/<me>/...`` should resolve ``<me>`` to the
    bound workspace client's ``current_user.me().user_name``."""

    def test_resolves_me_from_current_user(self, workspace) -> None:
        workspace.current_user.me.return_value = SimpleNamespace(
            user_name="alice@example.com",
        )
        # Reset the per-client cache so the resolve fires.
        from yggdrasil.databricks.fs.workspace_path import _USER_NAME_CACHE
        _USER_NAME_CACHE.clear()

        p = WorkspacePath(
            "/Workspace/Users/<me>/scratch/x.txt", workspace=workspace,
        )
        assert p.full_path() == (
            "/Workspace/Users/alice@example.com/scratch/x.txt"
        )
        assert p.api_path == p.full_path()
        # Second access uses the cache — no extra round-trip.
        _ = p.full_path()
        assert workspace.current_user.me.call_count == 1

    def test_unrelated_path_untouched(self, workspace) -> None:
        # No ``<me>`` segment → no SDK call.
        p = WorkspacePath("/Workspace/Shared/x.txt", workspace=workspace)
        assert p.full_path() == "/Workspace/Shared/x.txt"
        workspace.current_user.me.assert_not_called()

    def test_resolution_failure_leaves_placeholder(self, workspace) -> None:
        from yggdrasil.databricks.fs.workspace_path import _USER_NAME_CACHE
        _USER_NAME_CACHE.clear()
        workspace.current_user.me.side_effect = RuntimeError("no perms")
        p = WorkspacePath(
            "/Workspace/Users/<me>/x.txt", workspace=workspace,
        )
        # Returns the original path; no exception.
        assert p.full_path() == "/Workspace/Users/<me>/x.txt"


class TestParentRetry:
    """Upload / mkdirs should auto-mkdir the parent on NotFound and
    swallow ``Folder X is protected`` messages from a protected
    ancestor."""

    def test_upload_creates_parent_on_not_found(self, workspace) -> None:
        attempts = [NotFound("does not exist"), None]

        def upload(**_kwargs):
            r = attempts.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        workspace.workspace.upload.side_effect = upload
        p = WorkspacePath(
            "/Workspace/Users/me/run-1/sub/file.txt", workspace=workspace,
        )
        p.write_bytes(b"abc")
        # Parent directory should have been created.
        workspace.workspace.mkdirs.assert_called_with(
            "/Workspace/Users/me/run-1/sub",
        )
        assert workspace.workspace.upload.call_count == 2

    def test_mkdirs_protected_ancestor_is_non_fatal(self, workspace) -> None:
        workspace.workspace.mkdirs.side_effect = type(
            "BadRequest", (Exception,), {},
        )("Folder Users is protected")
        p = WorkspacePath(
            "/Workspace/Users/me/run-1/listing", workspace=workspace,
        )
        # Should not raise.
        p.mkdir(parents=True, exist_ok=True)


class TestRetryPolicy:

    @pytest.fixture
    def sleeps(self):
        recorded: list[float] = []
        return recorded, recorded.append

    def test_internal_error_retries(self, workspace, sleeps) -> None:
        recorded, spy = sleeps
        attempts = [InternalError(), _file_status(3)]

        def get_status(path):
            r = attempts.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        workspace.workspace.get_status.side_effect = get_status
        p = WorkspacePath("/Workspace/x", workspace=workspace, retry_sleep=spy)
        assert p.size == 3
        assert recorded == [1.0]

    def test_permission_retries_once(self, workspace, sleeps) -> None:
        recorded, spy = sleeps
        workspace.workspace.get_status.side_effect = PermissionDenied()
        p = WorkspacePath("/Workspace/x", workspace=workspace, retry_sleep=spy)
        assert p._stat_uncached().kind is IOKind.MISSING
        # One permission retry.
        assert recorded == [1.0]
        assert workspace.workspace.get_status.call_count == 2
