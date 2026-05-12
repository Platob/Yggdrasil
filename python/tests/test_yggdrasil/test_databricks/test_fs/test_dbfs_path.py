"""Mock-driven behavior tests for :class:`DBFSPath`."""
from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from yggdrasil.databricks.fs import DBFSPath
from yggdrasil.io.io_stats import IOKind


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class NotFound(Exception):
    """Mock for ``databricks.sdk.errors.NotFound``."""


class InternalError(Exception):
    """Mock for ``databricks.sdk.errors.InternalError``."""


class BadRequest(Exception):
    """Mock for ``databricks.sdk.errors.BadRequest``."""


class PermissionDenied(Exception):
    """Mock for ``databricks.sdk.errors.platform.PermissionDenied``."""


class StreamWriter:
    """Mock for the ``dbfs.open(write=True)`` context manager."""

    def __init__(self, sink: list[bytes]) -> None:
        self._sink = sink

    def __enter__(self) -> "StreamWriter":
        return self

    def __exit__(self, *args) -> None:
        return None

    def write(self, data: bytes) -> None:
        self._sink.append(bytes(data))


@pytest.fixture(autouse=True)
def reset_remote_stat_cache():
    from yggdrasil.io.path.remote_path import RemotePath
    RemotePath._STAT_CACHE.clear()
    yield
    RemotePath._STAT_CACHE.clear()


@pytest.fixture
def client():
    """Mock :class:`DatabricksClient` — the single access point.

    Built as a :class:`MagicMock`; ``client.workspace_client()`` returns
    the auto-created child mock (the :func:`workspace` fixture below
    grabs that same child so tests can configure both surfaces from
    either name).
    """
    return MagicMock()


@pytest.fixture
def workspace(client):
    """Shortcut for ``client.workspace_client.return_value``.

    Tests configure the per-method behavior via ``workspace.dbfs.<method>``
    return values / side_effects, then pass ``client=client`` to the
    path constructor — the path reaches the SDK through
    :attr:`DatabricksPath.client`.
    """
    return client.workspace_client.return_value


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:

    def test_legacy_posix_string(self, workspace, client) -> None:
        p = DBFSPath("/dbfs/foo/bar.txt", client=client)
        assert p.full_path() == "/dbfs/foo/bar.txt"
        assert p.api_path == "/foo/bar.txt"

    def test_url_form(self, workspace, client) -> None:
        p = DBFSPath("dbfs:///foo/bar.txt", client=client)
        assert p.full_path() == "/dbfs/foo/bar.txt"

    def test_root(self, workspace, client) -> None:
        p = DBFSPath("/dbfs", client=client)
        assert p.full_path() == "/dbfs"
        assert p.api_path == "/"


# ---------------------------------------------------------------------------
# Stat
# ---------------------------------------------------------------------------


class TestStat:

    def test_existing_file(self, workspace, client) -> None:
        workspace.dbfs.get_status.return_value = SimpleNamespace(
            is_dir=False, file_size=42, modification_time=10_000,
        )
        p = DBFSPath("/dbfs/x", client=client)
        s = p._stat_uncached()
        assert s.kind is IOKind.FILE
        assert s.size == 42
        assert s.mtime == pytest.approx(10.0)
        workspace.dbfs.get_status.assert_called_once_with("/x")

    def test_directory(self, workspace, client) -> None:
        workspace.dbfs.get_status.return_value = SimpleNamespace(
            is_dir=True, file_size=0, modification_time=0,
        )
        p = DBFSPath("/dbfs/folder", client=client)
        assert p._stat_uncached().kind is IOKind.DIRECTORY

    def test_missing(self, workspace, client) -> None:
        workspace.dbfs.get_status.side_effect = NotFound()
        p = DBFSPath("/dbfs/no-such", client=client)
        assert p._stat_uncached().kind is IOKind.MISSING


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


class TestRead:

    def test_full_object_read(self, workspace, client) -> None:
        # Aggressive whole-file read: no precondition ``get_status``
        # probe — the SDK call gets the full chunk budget and the
        # short page tells us EOF. Saves one round trip per cold read.
        workspace.dbfs.read.return_value = SimpleNamespace(
            data=base64.b64encode(b"hello").decode(),
        )
        p = DBFSPath("/dbfs/x", client=client)
        assert p.read_bytes() == b"hello"
        workspace.dbfs.get_status.assert_not_called()
        workspace.dbfs.read.assert_called_once()
        call = workspace.dbfs.read.call_args.kwargs
        assert call["path"] == "/x"
        assert call["offset"] == 0
        # Chunk budget is the SDK's 1 MiB cap; the server returns a
        # short page on EOF, which is how we know we're done.
        assert call["length"] == 1 * 1024 * 1024

    def test_chunked_read(self, workspace, client) -> None:
        # 1.5 MiB file — needs two chunks.
        big_size = int(1.5 * 1024 * 1024)
        workspace.dbfs.get_status.return_value = SimpleNamespace(
            is_dir=False, file_size=big_size, modification_time=0,
        )
        responses = [
            SimpleNamespace(data=base64.b64encode(b"A" * (1024 * 1024)).decode()),
            SimpleNamespace(data=base64.b64encode(b"B" * (big_size - 1024 * 1024)).decode()),
        ]
        workspace.dbfs.read.side_effect = responses
        p = DBFSPath("/dbfs/big", client=client)
        out = p.read_bytes()
        assert out == b"A" * (1024 * 1024) + b"B" * (big_size - 1024 * 1024)
        assert workspace.dbfs.read.call_count == 2

    def test_missing_raises(self, workspace, client) -> None:
        workspace.dbfs.get_status.return_value = SimpleNamespace(
            is_dir=False, file_size=10, modification_time=0,
        )
        workspace.dbfs.read.side_effect = NotFound()
        p = DBFSPath("/dbfs/x", client=client)
        with pytest.raises(FileNotFoundError):
            p.read_bytes()


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


class TestWrite:

    def test_write_bytes(self, workspace, client) -> None:
        sink: list[bytes] = []
        workspace.dbfs.open.return_value = StreamWriter(sink)
        # Stat probe for invalidation - missing initially.
        workspace.dbfs.get_status.side_effect = NotFound()

        p = DBFSPath("/dbfs/out", client=client)
        p.write_bytes(b"abcdef")
        # Streaming open got called with the right path + flags.
        kwargs = workspace.dbfs.open.call_args.kwargs
        assert kwargs == {
            "path": "/out", "read": False, "write": True, "overwrite": True,
        }
        assert b"".join(sink) == b"abcdef"

    def test_pwrite_does_rmw(self, workspace, client) -> None:
        # Existing 5 bytes; pwrite at pos=1 splices in 'XX'.
        workspace.dbfs.get_status.return_value = SimpleNamespace(
            is_dir=False, file_size=5, modification_time=0,
        )
        workspace.dbfs.read.return_value = SimpleNamespace(
            data=base64.b64encode(b"abcde").decode(),
        )
        sink: list[bytes] = []
        workspace.dbfs.open.return_value = StreamWriter(sink)
        p = DBFSPath("/dbfs/x", client=client)
        p.pwrite(b"XX", 1)
        assert b"".join(sink) == b"aXXde"

    def test_truncate_shrinks(self, workspace, client) -> None:
        workspace.dbfs.get_status.return_value = SimpleNamespace(
            is_dir=False, file_size=6, modification_time=0,
        )
        # Honor the ``length`` arg so truncate's range read returns
        # exactly the head it asks for.
        full = b"abcdef"

        def read(*, path, offset, length):
            return SimpleNamespace(
                data=base64.b64encode(full[offset : offset + length]).decode(),
            )

        workspace.dbfs.read.side_effect = read
        sink: list[bytes] = []
        workspace.dbfs.open.return_value = StreamWriter(sink)
        p = DBFSPath("/dbfs/x", client=client)
        p.truncate(3)
        assert b"".join(sink) == b"abc"


# ---------------------------------------------------------------------------
# Mutators
# ---------------------------------------------------------------------------


class TestMutators:

    def test_unlink(self, workspace, client) -> None:
        workspace.dbfs.get_status.return_value = SimpleNamespace(
            is_dir=False, file_size=0, modification_time=0,
        )
        p = DBFSPath("/dbfs/x", client=client)
        p.unlink()
        workspace.dbfs.delete.assert_called_once_with("/x", recursive=False)

    def test_remove_dir(self, workspace, client) -> None:
        workspace.dbfs.get_status.return_value = SimpleNamespace(
            is_dir=True, file_size=0, modification_time=0,
        )
        p = DBFSPath("/dbfs/folder", client=client)
        p.remove(recursive=True)
        workspace.dbfs.delete.assert_called_once_with("/folder", recursive=True)

    def test_mkdir(self, workspace, client) -> None:
        p = DBFSPath("/dbfs/a/b", client=client)
        p.mkdir()
        workspace.dbfs.mkdirs.assert_called_once_with("/a/b")


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


class TestListing:

    def test_iterdir(self, workspace, client) -> None:
        workspace.dbfs.list.return_value = [
            SimpleNamespace(path="/folder/a.parquet", is_dir=False),
            SimpleNamespace(path="/folder/sub", is_dir=True),
        ]
        p = DBFSPath("/dbfs/folder", client=client)
        children = list(p.iterdir())
        assert len(children) == 2
        assert children[0].full_path() == "/dbfs/folder/a.parquet"
        assert children[1].full_path() == "/dbfs/folder/sub"


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


class TestRetryPolicy:

    @pytest.fixture
    def sleeps(self):
        recorded: list[float] = []
        return recorded, recorded.append

    def test_internal_error_retries_then_succeeds(self, workspace, client, sleeps) -> None:
        recorded, spy = sleeps
        attempts = [
            InternalError(),
            BadRequest(),
            SimpleNamespace(is_dir=False, file_size=4, modification_time=0),
        ]

        def get_status(path):
            r = attempts.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        workspace.dbfs.get_status.side_effect = get_status
        p = DBFSPath("/dbfs/x", client=client, retry_sleep=spy)
        assert p.size == 4
        assert recorded == [1.0, 1.0]

    def test_internal_error_gives_up_after_4(self, workspace, client, sleeps) -> None:
        recorded, spy = sleeps
        workspace.dbfs.get_status.side_effect = InternalError()
        p = DBFSPath("/dbfs/x", client=client, retry_sleep=spy)
        # Stat returns MISSING when the SDK keeps failing — the
        # ``except Exception`` swallow is part of the contract for
        # stat probes, but the retry helper still fired its full
        # schedule before that swallow.
        s = p._stat_uncached()
        assert s.kind is IOKind.MISSING
        assert recorded == [1.0, 1.0, 1.0, 1.0]
        assert workspace.dbfs.get_status.call_count == 5

    def test_not_found_does_not_retry(self, workspace, client, sleeps) -> None:
        recorded, spy = sleeps
        workspace.dbfs.get_status.side_effect = NotFound()
        p = DBFSPath("/dbfs/x", client=client, retry_sleep=spy)
        # NotFound isn't classified as transient (default name) and
        # the stat path catches it cleanly.
        assert p._stat_uncached().kind is IOKind.MISSING
        assert recorded == []
        # One call attempted; no retries.
        assert workspace.dbfs.get_status.call_count == 1
