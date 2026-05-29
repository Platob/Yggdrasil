"""Mock-driven tests for :class:`VolumePath`.

:class:`VolumePath` now issues Databricks Files-API traffic over
yggdrasil's own :class:`HTTPSession` (``/api/2.0/fs/files`` /
``/api/2.0/fs/directories``) instead of the SDK's ``workspace.files``
client. To keep this unit suite focused on :class:`VolumePath`'s own
logic — URL building, status → exception translation, pagination,
parent auto-creation — without standing up a real workspace, the
``client`` fixture wires a :class:`_FakeFilesSession` that translates
each Files-API HTTP call back onto the same ``workspace.files``
MagicMock the tests already configure. Real wire retry / stream-resume
lives in the :class:`HTTPSession` tests; :class:`TestTransportResilience`
exercises the volume ↔ session integration against a real session with
a stubbed socket layer.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from yggdrasil.databricks.fs import VolumePath
from yggdrasil.databricks.volume.volumes import Volumes
from yggdrasil.io.io_stats import IOKind
from yggdrasil.url import URL

from tests.test_yggdrasil.test_databricks._files_fake import (
    FakeResp as _FakeResp,
    wire_files_session,
)


class NotFound(Exception):
    pass


class InternalError(Exception):
    pass


class PermissionDenied(Exception):
    pass


@pytest.fixture(autouse=True)
def reset_remote_stat_cache():
    yield


@pytest.fixture(autouse=True)
def reset_volume_credentials_refresher_singletons():
    # ``VolumeCredentialsRefresher`` is a process-wide singleton keyed
    # by ``(volume_id, operation)``. Tests reuse the same volume_id /
    # operation pair across cases — without an explicit reset the
    # singleton from a prior test (along with its cached AWSClient)
    # would survive into the next, masking refresher behavior the new
    # test is trying to assert.
    from yggdrasil.databricks.fs.volume_path import VolumeCredentialsRefresher

    VolumeCredentialsRefresher._INSTANCES.clear()
    yield
    VolumeCredentialsRefresher._INSTANCES.clear()


@pytest.fixture(autouse=True)
def reset_volume_info_cache():
    # ``Volume`` singletons cache ``VolumeInfo`` per (host, cat, sch,
    # vol). Tests share path coordinates across cases, so leaking the
    # cache would short-circuit the SDK call this case is trying to
    # observe.
    from yggdrasil.databricks.volume.volume import Volume

    Volume._INSTANCES.clear()
    yield
    Volume._INSTANCES.clear()


@pytest.fixture
def client():
    # Files-API transport seam — ``VolumePath`` reaches the workspace
    # over ``client.files_session().fetch`` with an Authorization header
    # from ``client.files_authorization()``, against ``client.base_url``.
    # The fake session translates each call back onto ``workspace.files``.
    return wire_files_session(MagicMock())


@pytest.fixture
def workspace(client):
    return client.workspace_client.return_value


@pytest.fixture
def service(client):
    """Mock :class:`Volumes` service whose ``client`` is the fixture
    :func:`client`.

    :class:`VolumePath` reaches the :class:`DatabricksClient` through
    ``self.service.client``; wiring the spec'd mock keeps the
    existing ``client.workspace_client.return_value.<api>``
    configuration working unchanged.
    """
    svc = MagicMock(spec=Volumes)
    svc.client = client
    return svc


def _file_meta(size: int, mtime_ms: int = 0):
    return SimpleNamespace(
        content_length=size,
        modification_time=mtime_ms,
    )


def _op_token(op) -> str:
    """Normalize the *operation* argument the SDK was called with.

    The production code passes either a :class:`VolumeOperation` enum
    (when the SDK exposes one) or the literal string ``"READ_VOLUME"``
    / ``"WRITE_VOLUME"`` (older SDK fallback). Tests compare against
    the wire token; ``.value`` / ``.name`` collapse the enum, and a
    bare string flows through unchanged.
    """
    return getattr(op, "value", None) or getattr(op, "name", None) or str(op)


class TestConstruction:

    def test_legacy_posix_string(self, workspace, client, service) -> None:
        p = VolumePath(
            "/Volumes/cat/sch/vol/data.parquet",
            service=service,
        )
        assert p.full_path() == "/Volumes/cat/sch/vol/data.parquet"
        assert p.api_path == "/Volumes/cat/sch/vol/data.parquet"

    def test_url_form(self, workspace, client, service) -> None:
        p = VolumePath("dbfs+volume:///cat/sch/vol/x", service=service)
        assert p.full_path() == "/Volumes/cat/sch/vol/x"

    def test_client_kwarg_wraps_into_service(self, client) -> None:
        # ``client=`` is the user-facing shortcut for "bind this path
        # to this workspace" — wraps into a fresh ``Volumes(client=…)``
        # so ``self.client`` returns the caller's client verbatim. Was
        # silently dropped before fixing — fell back to
        # ``DatabricksService.current()`` which builds a default
        # ``DatabricksClient`` against the env vars (typically empty)
        # and then explodes inside ``make_config`` on the first SDK call.
        p = VolumePath("/Volumes/cat/sch/vol/x", client=client)
        assert p.client is client

    def test_client_kwarg_propagates_to_children_and_parents(self, client) -> None:
        # Parent / child paths derived via ``_from_url`` must carry the
        # same explicit client through the URL walk.
        p = VolumePath("/Volumes/cat/sch/vol/dir/x", client=client)
        assert p.parent.client is client
        assert (p / "sub").client is client

    def test_distinct_clients_get_distinct_singletons(self) -> None:
        # The singleton cache key folds ``client`` into the slot
        # ``service`` would occupy so two callers passing different
        # ``DatabricksClient`` instances against the same URL don't
        # collide on the cached path.
        c1 = MagicMock()
        c2 = MagicMock()
        p1 = VolumePath("/Volumes/cat/sch/vol/x", client=c1)
        p2 = VolumePath("/Volumes/cat/sch/vol/x", client=c2)
        assert p1 is not p2
        assert p1.client is c1
        assert p2.client is c2


class TestStat:

    def test_existing_file(self, workspace, client, service) -> None:
        # Leaf carries ``.`` — heuristic probes ``get_metadata`` first.
        workspace.files.get_metadata.return_value = _file_meta(42)
        p = VolumePath("/Volumes/c/s/v/x.parquet", service=service)
        s = p._stat_uncached()
        assert s.kind is IOKind.FILE
        assert s.size == 42
        workspace.files.get_directory_metadata.assert_not_called()

    def test_existing_directory_no_extension(self, workspace, client, service) -> None:
        # Bare leaf — heuristic probes ``get_directory_metadata`` first,
        # so the single round trip resolves the directory without
        # touching ``get_metadata``.
        workspace.files.get_directory_metadata.return_value = SimpleNamespace()
        p = VolumePath("/Volumes/c/s/v/dir", service=service)
        assert p._stat_uncached().kind is IOKind.DIRECTORY
        workspace.files.get_metadata.assert_not_called()

    def test_file_fallback_when_leaf_has_no_extension(
        self, workspace, client, service
    ) -> None:
        # Even a bare-leaf path can be a file (extensionless data
        # dumps). When the directory probe NotFounds, fall back to
        # ``get_metadata``.
        workspace.files.get_directory_metadata.side_effect = NotFound()
        workspace.files.get_metadata.return_value = _file_meta(7)
        p = VolumePath("/Volumes/c/s/v/x", service=service)
        s = p._stat_uncached()
        assert s.kind is IOKind.FILE
        assert s.size == 7

    def test_directory_fallback_when_leaf_has_extension(
        self, workspace, client, service
    ) -> None:
        # ``foo.parquet`` looks like a file but could legitimately be a
        # directory; when ``get_metadata`` NotFounds we fall back to
        # ``get_directory_metadata``.
        workspace.files.get_metadata.side_effect = NotFound()
        workspace.files.get_directory_metadata.return_value = SimpleNamespace()
        p = VolumePath("/Volumes/c/s/v/dir.d", service=service)
        assert p._stat_uncached().kind is IOKind.DIRECTORY

    def test_missing(self, workspace, client, service) -> None:
        workspace.files.get_metadata.side_effect = NotFound()
        workspace.files.get_directory_metadata.side_effect = NotFound()
        workspace.files.list_directory_contents.side_effect = NotFound()
        p = VolumePath("/Volumes/c/s/v/x", service=service)
        assert p._stat_uncached().kind is IOKind.MISSING

    def test_implicit_directory_visible_through_listing(
        self,
        workspace,
        client,
        service,
    ) -> None:
        # ``files.upload("/.../parent/file.bin")`` against a parent
        # the caller never explicitly mkdir'd silently materialises
        # the file but leaves ``get_directory_metadata(parent)``
        # returning NotFound. Listing still enumerates the children,
        # so ``_stat`` falls back to ``list_directory_contents`` and
        # reports the parent as ``DIRECTORY``. Without this,
        # ``remove(parent, recursive=True, missing_ok=False)`` would
        # raise against a directory the caller just wrote into.
        workspace.files.get_metadata.side_effect = NotFound()
        workspace.files.get_directory_metadata.side_effect = NotFound()
        workspace.files.list_directory_contents.return_value = iter(
            [
                SimpleNamespace(
                    path="/Volumes/c/s/v/parent/file.bin",
                    is_directory=False,
                ),
            ]
        )
        p = VolumePath("/Volumes/c/s/v/parent", service=service)
        assert p._stat_uncached().kind is IOKind.DIRECTORY


class TestRead:

    def test_full_object_read(self, workspace, client, service) -> None:
        workspace.files.get_metadata.return_value = _file_meta(5)
        body = SimpleNamespace(read=lambda: b"hello")
        workspace.files.download.return_value = SimpleNamespace(contents=body)

        p = VolumePath("/Volumes/c/s/v/x", service=service)
        assert p.read_bytes() == b"hello"
        workspace.files.download.assert_called_once_with("/Volumes/c/s/v/x")

    def test_missing_raises(self, workspace, client, service) -> None:
        workspace.files.get_metadata.return_value = _file_meta(5)
        workspace.files.download.side_effect = NotFound()
        p = VolumePath("/Volumes/c/s/v/x", service=service)
        with pytest.raises(FileNotFoundError):
            p.read_bytes()

    def test_read_bytes_skips_metadata_probe(self, workspace, client, service) -> None:
        # ``DatabricksPath.read_mv(-1, 0)`` short-circuits the base
        # ``Holder.read_mv`` size probe, so a whole-file read is one
        # ``files.download`` round trip — no preceding
        # ``files.get_metadata`` call.
        body = SimpleNamespace(read=lambda: b"hello")
        workspace.files.download.return_value = SimpleNamespace(contents=body)
        p = VolumePath("/Volumes/c/s/v/x", service=service)
        assert p.read_bytes() == b"hello"
        workspace.files.get_metadata.assert_not_called()
        workspace.files.download.assert_called_once()

    def test_parquet_read_arrow_table_one_sdk_call(
        self, workspace, client, service
    ) -> None:
        # The tabular IO ↔ remote path interaction is the headline
        # scenario: ``ParquetFile(VolumePath).read_arrow_table()`` must
        # bottom out in a single ``files.download`` call. Earlier
        # versions issued a ``get_metadata`` probe before the
        # download to short-circuit on empty buffers; that's now
        # gated on the ``size_known`` predicate so a cold remote
        # path skips the probe and falls back to "parse what we
        # got" semantics via the format reader's own EOF errors.
        import io as _io
        import pyarrow as pa
        import pyarrow.parquet as pq
        from yggdrasil.io.primitive.parquet_file import ParquetFile

        sink = _io.BytesIO()
        pq.write_table(
            pa.table({"id": pa.array([1, 2, 3], type=pa.int64())}),
            sink,
        )
        payload = sink.getvalue()

        body = SimpleNamespace(read=lambda: payload)
        workspace.files.download.return_value = SimpleNamespace(
            contents=body,
            content_type="application/octet-stream",
            last_modified="Mon, 01 Jan 2024 00:00:00 GMT",
        )
        p = VolumePath("/Volumes/c/s/v/x.parquet", service=service)
        ParquetFile(holder=p).read_arrow_table()
        workspace.files.get_metadata.assert_not_called()
        assert workspace.files.download.call_count == 1


class TestWrite:

    def test_overwrite(self, workspace, client, service) -> None:
        # Initial probe: missing.
        workspace.files.get_metadata.side_effect = NotFound()
        workspace.files.get_directory_metadata.side_effect = NotFound()
        p = VolumePath("/Volumes/c/s/v/x", service=service)
        p.write_bytes(b"abcdef")
        kwargs = workspace.files.upload.call_args.kwargs
        assert kwargs["file_path"] == "/Volumes/c/s/v/x"
        assert kwargs["overwrite"] is True
        # ``FilesExt.upload`` probes ``contents.seekable()`` — bytes
        # are wrapped in a fresh ``BytesIO`` so the SDK sees a
        # seekable stream rather than a raw bytes object.
        sent = kwargs["contents"]
        assert isinstance(sent, io.BytesIO)
        assert sent.getvalue() == b"abcdef"

    def test_stream_input_routes_through_upload(
        self, workspace, client, service
    ) -> None:
        """A caller-supplied ``BinaryIO`` lands on one ``PUT /files`` upload."""
        import io

        workspace.files.get_metadata.side_effect = NotFound()
        workspace.files.get_directory_metadata.side_effect = NotFound()
        stream = io.BytesIO(b"streamed-payload")
        p = VolumePath("/Volumes/c/s/v/x", service=service)
        p.write_bytes(stream)

        kwargs = workspace.files.upload.call_args.kwargs
        # The whole stream is read into a replayable byte body and PUT
        # once — no chunked RMW loop, no per-attempt rewind dance.
        assert isinstance(kwargs["contents"], io.BytesIO)
        assert kwargs["contents"].getvalue() == b"streamed-payload"
        assert workspace.files.upload.call_count == 1

    def test_stream_input_read_into_replayable_body(
        self, workspace, client, service
    ) -> None:
        """A stream is drained into bytes up front so a transport-layer
        replay PUTs the full body, not an empty tail.

        The seek-on-retry dance the SDK path needed is gone: the body is
        already bytes by the time it reaches the wire, so the
        :class:`HTTPSession`'s own retry/resume replays it verbatim.
        """
        import io

        workspace.files.get_metadata.side_effect = NotFound()
        workspace.files.get_directory_metadata.side_effect = NotFound()

        VolumePath("/Volumes/c/s/v/x", service=service).write_bytes(
            io.BytesIO(b"abcdef"),
        )

        assert workspace.files.upload.call_count == 1
        sent = workspace.files.upload.call_args.kwargs["contents"]
        assert sent.getvalue() == b"abcdef"

    def test_pwrite_does_rmw(self, workspace, client, service) -> None:
        workspace.files.get_metadata.return_value = _file_meta(5)
        body = SimpleNamespace(read=lambda: b"abcde")
        workspace.files.download.return_value = SimpleNamespace(contents=body)
        p = VolumePath("/Volumes/c/s/v/x", service=service)
        p.pwrite(b"XX", 1)
        sent = workspace.files.upload.call_args.kwargs["contents"]
        assert isinstance(sent, io.BytesIO)
        assert sent.getvalue() == b"aXXde"

    def test_open_wb_multi_page_payload_uploads_once(
        self,
        workspace,
        client,
        service,
    ) -> None:
        # ``with vp.open("wb") as fh: fh.write(...)`` must collapse to
        # a single ``files.upload`` even when the payload spans
        # multiple buffered pages — the cursor's release flush and the
        # parent's release flush both fire (``IO._release`` →
        # ``parent.flush()`` plus ``RemotePath._release`` →
        # ``self.flush()``), but the second hop must observe a clean
        # dirty-page set and skip the upload. Without this guarantee
        # every ``open("wb") + write`` round-trips the body twice over
        # the wire.
        workspace.files.get_metadata.side_effect = NotFound()
        workspace.files.get_directory_metadata.side_effect = NotFound()
        workspace.files.download.side_effect = NotFound()
        page_size = 1024
        payload = b"A" * (page_size * 5 + 13)  # 5+ pages, off-boundary tail
        p = VolumePath(
            "/Volumes/c/s/v/x.bin",
            service=service,
            page_size=page_size,
        )
        with p.open("wb") as fh:
            fh.write(payload)
        assert workspace.files.upload.call_count == 1
        sent = workspace.files.upload.call_args.kwargs["contents"]
        assert isinstance(sent, io.BytesIO)
        assert sent.getvalue() == payload

    def test_open_wb_many_small_writes_uploads_once(
        self,
        workspace,
        client,
        service,
    ) -> None:
        # Repeated small writes inside one ``open("wb")`` accumulate in
        # the page cache and commit on close as one upload — no
        # per-write flush, even when the running total crosses a page
        # boundary partway through.
        workspace.files.get_metadata.side_effect = NotFound()
        workspace.files.get_directory_metadata.side_effect = NotFound()
        workspace.files.download.side_effect = NotFound()
        page_size = 1024
        chunk = b"X" * 200
        n_chunks = 50  # 10_000 bytes total, ~10 pages
        p = VolumePath(
            "/Volumes/c/s/v/x.bin",
            service=service,
            page_size=page_size,
        )
        with p.open("wb") as fh:
            for _ in range(n_chunks):
                fh.write(chunk)
        assert workspace.files.upload.call_count == 1
        sent = workspace.files.upload.call_args.kwargs["contents"]
        assert sent.getvalue() == chunk * n_chunks


class TestWriteAll:

    def test_single_upload_no_stat_no_truncate(
        self, workspace, client, service
    ) -> None:
        p = VolumePath("/Volumes/c/s/v/data.bin", service=service)
        p.write_bytes(b"hello-world", overwrite=True)

        workspace.files.upload.assert_called_once()
        kwargs = workspace.files.upload.call_args.kwargs
        assert kwargs["file_path"] == "/Volumes/c/s/v/data.bin"
        assert kwargs["overwrite"] is True
        sent = kwargs["contents"]
        assert isinstance(sent, io.BytesIO)
        assert sent.getvalue() == b"hello-world"
        workspace.files.get_metadata.assert_not_called()
        workspace.files.download.assert_not_called()

    def test_stream_input(self, workspace, client, service) -> None:
        p = VolumePath("/Volumes/c/s/v/data.bin", service=service)
        stream = io.BytesIO(b"streamed")
        p.write_bytes(stream, overwrite=True)

        workspace.files.upload.assert_called_once()
        workspace.files.get_metadata.assert_not_called()

    def test_memoryview_input(self, workspace, client, service) -> None:
        p = VolumePath("/Volumes/c/s/v/data.bin", service=service)
        p.write_bytes(memoryview(b"view"), overwrite=True)

        sent = workspace.files.upload.call_args.kwargs["contents"]
        assert isinstance(sent, io.BytesIO)
        assert sent.getvalue() == b"view"

    def test_pyarrow_buffer_input(self, workspace, client, service) -> None:
        import pyarrow as pa

        buf = pa.BufferOutputStream()
        buf.write(b"arrow-buffer-payload")
        arrow_buf = buf.getvalue()

        p = VolumePath("/Volumes/c/s/v/data.bin", service=service)
        n = p.write_bytes(arrow_buf, overwrite=True)
        assert n == 20
        sent = workspace.files.upload.call_args.kwargs["contents"]
        assert sent.getvalue() == b"arrow-buffer-payload"

    def test_parquet_roundtrip(self, workspace, client, service) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({"id": pa.array([1, 2, 3], type=pa.int64())})
        sink = io.BytesIO()
        pq.write_table(table, sink)
        parquet_bytes = sink.getvalue()

        p = VolumePath("/Volumes/c/s/v/out.parquet", service=service)
        n = p.write_bytes(parquet_bytes, overwrite=True)

        assert n == len(parquet_bytes)
        workspace.files.upload.assert_called_once()
        workspace.files.get_metadata.assert_not_called()
        workspace.files.download.assert_not_called()

        sent = workspace.files.upload.call_args.kwargs["contents"]
        assert isinstance(sent, io.BytesIO)
        roundtrip = pa.BufferReader(sent.getvalue())
        assert pq.read_table(roundtrip).equals(table)

    def test_auto_creates_parents(self, workspace, client, service) -> None:
        uploads = [NotFound("does not exist"), None]

        def upload(**_kwargs):
            r = uploads.pop(0)
            if isinstance(r, Exception):
                raise r

        workspace.files.upload.side_effect = upload
        p = VolumePath("/Volumes/cat/sch/vol/sub/x.parquet", service=service)
        p.write_bytes(b"data", overwrite=True)

        workspace.files.create_directory.assert_called_once_with(
            "/Volumes/cat/sch/vol/sub",
        )
        assert workspace.files.upload.call_count == 2

    def test_overwrite_fewer_calls_than_plain_write(self, workspace, client, service) -> None:
        workspace.files.get_metadata.side_effect = NotFound()
        workspace.files.get_directory_metadata.side_effect = NotFound()

        p1 = VolumePath("/Volumes/c/s/v/a.bin", service=service)
        p1.write_bytes(b"via-write-bytes")
        wb_upload_count = workspace.files.upload.call_count
        wb_meta_count = workspace.files.get_metadata.call_count

        workspace.files.upload.reset_mock()
        workspace.files.get_metadata.reset_mock()
        workspace.files.get_metadata.side_effect = NotFound()

        p2 = VolumePath("/Volumes/c/s/v/b.bin", service=service)
        p2.write_bytes(b"via-overwrite", overwrite=True)
        ow_upload_count = workspace.files.upload.call_count
        ow_meta_count = workspace.files.get_metadata.call_count

        assert ow_upload_count == 1
        assert ow_meta_count == 0
        assert ow_upload_count <= wb_upload_count

    def test_parquetfile_write_arrow_table_uses_write_all(
        self,
        workspace,
        client,
        service,
    ) -> None:
        """ParquetFile(holder=VolumePath).write_arrow_table() routes
        through write_all via _commit_format_payload: 1 upload,
        1 get_metadata (mode guard size check), 0 downloads."""
        import pyarrow as pa
        from yggdrasil.io.primitive.parquet_file import ParquetFile

        table = pa.table(
            {
                "id": pa.array([1, 2, 3], type=pa.int64()),
                "name": pa.array(["a", "b", "c"]),
            }
        )

        workspace.files.download.side_effect = NotFound()
        workspace.files.get_metadata.side_effect = NotFound()
        workspace.files.get_directory_metadata.side_effect = NotFound()
        workspace.files.list_directory_contents.side_effect = NotFound()

        p = VolumePath("/Volumes/c/s/v/out.parquet", service=service)
        pf = ParquetFile(holder=p, owns_holder=False)
        pf.write_arrow_table(table)

        assert workspace.files.upload.call_count == 1
        assert workspace.files.download.call_count == 0

    def test_parquetfile_write_single_upload(
        self,
        workspace,
        client,
        service,
    ) -> None:
        """write_arrow_table via ParquetFile issues exactly 1 upload.

        Before: _commit_format_payload did seek(0) + truncate(0) +
        write_bytes — on VolumePath truncate downloads the file +
        re-uploads, then write_bytes does another upload = 2+ uploads.
        After: write_all skips truncate — 1 upload total.
        """
        import pyarrow as pa
        from yggdrasil.io.primitive.parquet_file import ParquetFile

        table = pa.table({"x": pa.array([10, 20, 30], type=pa.int32())})

        workspace.files.download.side_effect = NotFound()
        workspace.files.get_metadata.side_effect = NotFound()
        workspace.files.get_directory_metadata.side_effect = NotFound()
        workspace.files.list_directory_contents.side_effect = NotFound()

        p = VolumePath("/Volumes/c/s/v/metrics.parquet", service=service)
        pf = ParquetFile(holder=p, owns_holder=False)
        pf.write_arrow_table(table)

        assert workspace.files.upload.call_count == 1
        assert workspace.files.download.call_count == 0


class TestMutators:

    def test_unlink(self, workspace, client, service) -> None:
        workspace.files.get_metadata.return_value = _file_meta(0)
        # Leaf has ``.`` so the heuristic resolves it as a file
        # without spuriously routing through ``get_directory_metadata``.
        p = VolumePath("/Volumes/c/s/v/x.bin", service=service)
        p.unlink()
        workspace.files.delete.assert_called_once_with("/Volumes/c/s/v/x.bin")

    def test_mkdir(self, workspace, client, service) -> None:
        p = VolumePath("/Volumes/c/s/v/folder", service=service)
        p.mkdir()
        workspace.files.create_directory.assert_called_once_with(
            "/Volumes/c/s/v/folder",
        )


class TestListing:

    def test_iterdir_preserves_catalog(self, workspace, client, service) -> None:
        # ``list_directory_contents`` returns canonical
        # ``/Volumes/<cat>/<sch>/<vol>/...`` paths; the catalog
        # segment must round-trip through child construction.
        workspace.files.list_directory_contents.return_value = [
            SimpleNamespace(
                path="/Volumes/trading/sch/vol/folder/a.bin", is_directory=False
            ),
            SimpleNamespace(
                path="/Volumes/trading/sch/vol/folder/sub", is_directory=True
            ),
        ]
        p = VolumePath("/Volumes/trading/sch/vol/folder", service=service)
        children = list(p.iterdir())
        assert [c.full_path() for c in children] == [
            "/Volumes/trading/sch/vol/folder/a.bin",
            "/Volumes/trading/sch/vol/folder/sub",
        ]
        assert all(isinstance(c, VolumePath) for c in children)

    def test_iterdir_does_not_persist_children_as_singletons(
        self,
        workspace,
        client,
        service,
    ) -> None:
        # ``_ls`` builds children with ``singleton_ttl=False`` so an
        # iterdir-style hot loop doesn't pin thousands of short-lived
        # paths in the bounded ``DatabricksPath._INSTANCES`` cache.
        from yggdrasil.databricks.path import DatabricksPath

        DatabricksPath._INSTANCES.clear()
        cache_size_before = len(list(DatabricksPath._INSTANCES.keys()))

        workspace.files.list_directory_contents.return_value = [
            SimpleNamespace(
                path="/Volumes/c/s/v/folder/ephemeral.bin",
                is_directory=False,
            ),
            SimpleNamespace(
                path="/Volumes/c/s/v/folder/also_ephemeral.bin",
                is_directory=False,
            ),
        ]
        p = VolumePath("/Volumes/c/s/v/folder", service=service)
        # Force materialisation so the listing actually runs.
        children = list(p.iterdir())
        assert len(children) == 2

        # No listing child ended up cached. ``p`` itself may or may not
        # have landed depending on construction path, but the children
        # explicitly opt out.
        keys_after = list(DatabricksPath._INSTANCES.keys())
        for k in keys_after:
            assert "ephemeral.bin" not in repr(
                k
            ), f"listing children should not enter _INSTANCES, found {k!r}"

    def test_iterdir_seeds_child_stat(self, workspace, client, service) -> None:
        # ``list_directory_contents`` already carries ``is_directory``
        # + ``file_size`` per entry, so every child's stat cache must
        # land warm. Otherwise, an N-entry iterdir() that asks
        # ``size`` / ``is_file()`` per child floods the Files API
        # with N extra ``get_metadata`` round trips.
        workspace.files.list_directory_contents.return_value = [
            SimpleNamespace(
                path="/Volumes/c/s/v/folder/a.parquet",
                is_directory=False,
                file_size=1024,
                last_modified=None,
            ),
            SimpleNamespace(
                path="/Volumes/c/s/v/folder/sub",
                is_directory=True,
                file_size=0,
                last_modified=None,
            ),
        ]
        p = VolumePath("/Volumes/c/s/v/folder", service=service)
        children = list(p.iterdir())
        # Inspecting every child collapses to a local hit.
        assert children[0].size == 1024
        assert children[0].is_file() is True
        assert children[1].is_dir() is True
        workspace.files.get_metadata.assert_not_called()
        workspace.files.get_directory_metadata.assert_not_called()


class TestVolumeAutoCreate:
    """``_call_ensuring_parents`` should walk the cheap path first
    (``create_directory`` on the parent) and only blind-create the
    catalog / schema / managed volume when that fails NotFound."""

    @pytest.fixture(autouse=True)
    def _missing_stat(self, workspace):
        # These cases write brand-new files; the pre-write stat probe
        # must read MISSING so the flow proceeds to the upload whose
        # parent auto-creation they exercise.
        workspace.files.get_metadata.side_effect = NotFound()
        workspace.files.get_directory_metadata.side_effect = NotFound()

    def test_only_subdir_missing_skips_volume_create(
        self, workspace, client, service
    ) -> None:
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
            "/Volumes/cat/sch/vol/sub/file.bin",
            service=service,
        )
        p.write_bytes(b"payload")

        workspace.files.create_directory.assert_called_once_with(
            "/Volumes/cat/sch/vol/sub",
        )
        workspace.catalogs.create.assert_not_called()
        workspace.schemas.create.assert_not_called()
        workspace.volumes.create.assert_not_called()
        assert workspace.files.upload.call_count == 2

    def test_volume_missing_only_creates_volume(
        self, workspace, client, service
    ) -> None:
        # Common case: catalog + schema already exist, only the volume
        # is missing. The upload's ``NotFound("Volume … does not exist")``
        # message names the volume directly, so the recovery path skips
        # its cheap ``files.create_directory`` probe and goes straight
        # to ``volumes.create``. ``files.create_directory`` is only
        # called once afterwards, to materialise the sub-directory.
        uploads = [NotFound("Volume 'cat.sch.vol' does not exist"), None]

        def upload(**_kwargs):
            r = uploads.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        workspace.files.upload.side_effect = upload

        p = VolumePath(
            "/Volumes/cat/sch/vol/sub/file.bin",
            service=service,
        )
        p.write_bytes(b"payload")

        workspace.catalogs.get.assert_not_called()
        workspace.schemas.get.assert_not_called()
        workspace.volumes.read.assert_not_called()

        # Volume created first; catalog/schema untouched because volume
        # create succeeded. The cheap-path probe was skipped — only the
        # post-volume-create ``create_directory`` call should land.
        workspace.catalogs.create.assert_not_called()
        workspace.schemas.create.assert_not_called()
        workspace.files.create_directory.assert_called_once_with(
            "/Volumes/cat/sch/vol/sub",
        )
        vol_kwargs = workspace.volumes.create.call_args.kwargs
        assert vol_kwargs["catalog_name"] == "cat"
        assert vol_kwargs["schema_name"] == "sch"
        assert vol_kwargs["name"] == "vol"
        vt = vol_kwargs["volume_type"]
        assert getattr(vt, "name", str(vt)).upper() == "MANAGED"

    def test_generic_not_found_still_probes_create_directory(
        self,
        workspace,
        client,
        service,
    ) -> None:
        # When the upload error doesn't name the volume (just a generic
        # ``"does not exist"``), the recovery path still attempts the
        # cheap ``files.create_directory`` probe first — the volume
        # itself may well exist and only the sub-directory is missing.
        # Only if that probe also NotFounds do we fall through to
        # ``volumes.create``.
        uploads = [NotFound("Path does not exist"), None]
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
            "/Volumes/cat/sch/vol/sub/file.bin",
            service=service,
        )
        p.write_bytes(b"payload")

        # Cheap probe attempted first (fails), then volume created,
        # then probe retried (succeeds).
        assert workspace.files.create_directory.call_count == 2
        workspace.volumes.create.assert_called_once()

    def test_schema_missing_creates_schema_then_volume(
        self,
        workspace,
        client,
        service,
    ) -> None:
        # Schema is also missing — first ``volumes.create`` fails
        # NotFound, then ``schemas.create`` succeeds, then volume create
        # is retried. Catalog should not be touched.
        uploads = [NotFound("Volume does not exist"), None]
        volume_creates = [NotFound("Schema does not exist"), None]

        def upload(**_kwargs):
            r = uploads.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        def volumes_create(**_kwargs):
            r = volume_creates.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        workspace.files.upload.side_effect = upload
        workspace.volumes.create.side_effect = volumes_create

        p = VolumePath(
            "/Volumes/cat/sch/vol/sub/file.bin",
            service=service,
        )
        p.write_bytes(b"payload")

        workspace.catalogs.create.assert_not_called()
        workspace.schemas.create.assert_called_once_with(
            name="sch",
            catalog_name="cat",
        )
        assert workspace.volumes.create.call_count == 2

    def test_catalog_missing_creates_full_chain(
        self, workspace, client, service
    ) -> None:
        # Both schema and catalog are missing — volume.create then
        # schema.create both fail NotFound, falling through to catalog
        # → schema → volume creation.
        uploads = [NotFound("Volume does not exist"), None]
        volume_creates = [NotFound("Schema does not exist"), None]
        schema_creates = [NotFound("Catalog does not exist"), None]

        def upload(**_kwargs):
            r = uploads.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        def volumes_create(**_kwargs):
            r = volume_creates.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        def schemas_create(**_kwargs):
            r = schema_creates.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        workspace.files.upload.side_effect = upload
        workspace.volumes.create.side_effect = volumes_create
        workspace.schemas.create.side_effect = schemas_create

        p = VolumePath(
            "/Volumes/cat/sch/vol/sub/file.bin",
            service=service,
        )
        p.write_bytes(b"payload")

        workspace.catalogs.create.assert_called_once_with(name="cat")
        assert workspace.schemas.create.call_count == 2
        assert workspace.volumes.create.call_count == 2

    def test_already_exists_swallowed(self, workspace, client, service) -> None:
        # Volume create races with another caller — ``AlreadyExists``
        # is treated as success, no retry storm.
        uploads = [NotFound("Volume does not exist"), None]

        def upload(**_kwargs):
            r = uploads.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        class AlreadyExists(Exception):
            pass

        workspace.files.upload.side_effect = upload
        workspace.volumes.create.side_effect = AlreadyExists()

        p = VolumePath(
            "/Volumes/cat/sch/vol/sub/file.bin",
            service=service,
        )
        p.write_bytes(b"payload")  # should not raise

        # Volume.create raised AlreadyExists → no schema/catalog touch.
        workspace.catalogs.create.assert_not_called()
        workspace.schemas.create.assert_not_called()

    def test_propagates_when_not_a_volume_path(
        self, workspace, client, service
    ) -> None:
        # Path too shallow to address a volume — auto-create can't help,
        # so the error must surface. (The Files API 404 surfaces as
        # FileNotFoundError now that the transport is plain HTTP.)
        workspace.files.upload.side_effect = NotFound("does not exist")
        p = VolumePath("/Volumes/onlycat", service=service)
        with pytest.raises(FileNotFoundError):
            p.write_bytes(b"x")
        workspace.volumes.create.assert_not_called()


class TestTransportResilience:
    """Reads / writes now flow over yggdrasil's :class:`HTTPSession`, which
    owns the transient-retry + resume-on-disconnect policy the SDK's Files
    client handled poorly. Verify the volume ↔ session integration against
    a *real* session with the socket send stubbed."""

    def test_read_survives_transient_ssl_eof(self, client, service) -> None:
        # A mid-flight ``UNEXPECTED_EOF`` on the first wire send must be
        # retried by the session and the read must still complete — the
        # exact failure the SDK Files client mishandles.
        import datetime as _dt
        import ssl

        from yggdrasil.http_ import HTTPSession
        from yggdrasil.http_.response import HTTPResponse

        host = "https://resilience-probe.databricks.com"
        client.base_url = URL.from_(host)
        client.files_authorization.return_value = "Bearer t"
        session = HTTPSession(base_url=host, verify=False)
        client.files_session.return_value = session

        calls = {"n": 0}

        def fake_send_once(*, request, **_kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ssl.SSLError(
                    "EOF occurred in violation of protocol (_ssl.c:1, "
                    "UNEXPECTED_EOF)"
                )
            return HTTPResponse(
                request=request,
                status_code=200,
                headers={},
                tags={},
                buffer=b"hello",
                received_at=_dt.datetime.now(_dt.timezone.utc),
            )

        session._send_once = fake_send_once

        p = VolumePath("/Volumes/c/s/v/x", service=service)
        assert p.read_bytes() == b"hello"
        # First attempt raised the transient EOF; the session retried and
        # the second attempt delivered the body.
        assert calls["n"] == 2

    def test_server_error_surfaces_after_session_retries(
        self, client, service
    ) -> None:
        # When the workspace keeps returning 503, the session exhausts its
        # retries and the volume op fails loudly rather than silently
        # returning empty data.
        host = "https://error-probe.databricks.com"
        client.base_url = URL.from_(host)
        client.files_authorization.return_value = "Bearer t"

        class _ErrSession:
            def fetch(self, *a, **k):
                return _FakeResp(
                    503,
                    headers={"Content-Type": "application/json"},
                    json_data={"message": "upstream unavailable"},
                )

        client.files_session.return_value = _ErrSession()

        p = VolumePath("/Volumes/c/s/v/x", service=service)
        with pytest.raises(OSError, match="503"):
            p.read_bytes()


# ---------------------------------------------------------------------------
# Native S3 storage fast path — storage_location + temporary_credentials + s3_path
# ---------------------------------------------------------------------------


def _volume_info(
    *,
    catalog: str = "cat",
    schema: str = "sch",
    name: str = "vol",
    volume_id: str = "volume-uuid-0001",
    storage_location: str = "s3://my-bucket/__unitystorage/cat/sch/vol",
):
    return SimpleNamespace(
        catalog_name=catalog,
        schema_name=schema,
        name=name,
        volume_id=volume_id,
        volume_type="MANAGED",
        storage_location=storage_location,
        full_name=f"{catalog}.{schema}.{name}",
        access_point=None,
    )


def _aws_creds_response(
    *,
    access_key_id: str = "AKIA-test",
    secret_access_key: str = "secret-test",
    session_token: str = "session-test",
):
    import datetime as _dt

    return SimpleNamespace(
        aws_temp_credentials=SimpleNamespace(
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
            access_point=None,
        ),
        expiration_time=_dt.datetime(2030, 1, 1, tzinfo=_dt.timezone.utc),
        url=None,
        azure_aad=None,
        azure_user_delegation_sas=None,
        gcp_oauth_token=None,
        r2_temp_credentials=None,
    )


class TestVolumeInfoCaching:

    def test_volume_info_caches_after_first_read(
        self, workspace, client, service
    ) -> None:
        workspace.volumes.read.return_value = _volume_info()
        p = VolumePath("/Volumes/cat/sch/vol/x", service=service)
        first = p.volume_info()
        second = p.volume_info()
        assert first is second
        workspace.volumes.read.assert_called_once_with("cat.sch.vol")

    def test_volume_info_refresh_forces_reread(
        self, workspace, client, service
    ) -> None:
        workspace.volumes.read.side_effect = [_volume_info(), _volume_info()]
        p = VolumePath("/Volumes/cat/sch/vol/x", service=service)
        p.volume_info()
        p.volume_info(refresh=True)
        assert workspace.volumes.read.call_count == 2

    def test_storage_location_resolves_from_volume_info(
        self, workspace, client, service
    ) -> None:
        workspace.volumes.read.return_value = _volume_info(
            storage_location="s3://bkt/__unitystorage/c/s/v",
        )
        p = VolumePath("/Volumes/c/s/v/sub/y.parquet", service=service)
        assert p.storage_location() == "s3://bkt/__unitystorage/c/s/v"

    def test_storage_location_caches_independently_of_volume_info(
        self,
        workspace,
        client,
        service,
    ) -> None:
        workspace.volumes.read.return_value = _volume_info()
        p = VolumePath("/Volumes/cat/sch/vol/x", service=service)
        p.storage_location()
        p.storage_location()
        # One read call drives both — the value is snapshotted onto
        # ``_storage_location`` and the second call returns the cached
        # string without re-touching ``VolumeInfo``.
        workspace.volumes.read.assert_called_once()

    def test_storage_location_missing_raises(self, workspace, client, service) -> None:
        workspace.volumes.read.return_value = _volume_info(storage_location=None)
        p = VolumePath("/Volumes/cat/sch/vol/x", service=service)
        with pytest.raises(ValueError, match="storage_location"):
            p.storage_location()


class TestTemporaryCredentials:

    def test_vends_via_volume_id(self, workspace, client, service) -> None:
        workspace.volumes.read.return_value = _volume_info(volume_id="vid-42")
        gen = (
            workspace.temporary_volume_credentials.generate_temporary_volume_credentials
        )
        gen.return_value = _aws_creds_response()

        p = VolumePath("/Volumes/cat/sch/vol/x", service=service)
        resp = p.temporary_credentials(mode="read")
        assert resp.aws_temp_credentials.access_key_id == "AKIA-test"

        # Call kwargs must include the volume_id from VolumeInfo plus
        # the operation token (READ_VOLUME for read-only modes).
        # ``VolumeOperation`` isn't a stable import across SDK versions
        # (older SDKs don't expose it at all), so compare against the
        # wire token via ``.value`` / ``.name`` / str() instead of the
        # enum identity.
        gen.assert_called_once()
        kwargs = gen.call_args.kwargs
        assert kwargs["volume_id"] == "vid-42"
        assert _op_token(kwargs["operation"]) == "READ_VOLUME"

    def test_write_operation_maps_to_write_volume(
        self, workspace, client, service
    ) -> None:
        workspace.volumes.read.return_value = _volume_info()
        gen = (
            workspace.temporary_volume_credentials.generate_temporary_volume_credentials
        )
        gen.return_value = _aws_creds_response()
        p = VolumePath("/Volumes/cat/sch/vol/x", service=service)
        p.temporary_credentials(mode="overwrite")

        assert _op_token(gen.call_args.kwargs["operation"]) == "WRITE_VOLUME"

    def test_missing_volume_id_raises(self, workspace, client, service) -> None:
        workspace.volumes.read.return_value = _volume_info(volume_id=None)
        p = VolumePath("/Volumes/cat/sch/vol/x", service=service)
        with pytest.raises(ValueError, match="volume_id"):
            p.temporary_credentials()


# ---------------------------------------------------------------------------
# Process-wide singleton refresher
# ---------------------------------------------------------------------------


class TestCredentialsRefresherSingleton:

    def test_same_volume_collapses_to_one_provider(
        self,
        workspace,
        client,
        service,
    ) -> None:
        from yggdrasil.databricks.fs.volume_path import VolumeCredentialsRefresher

        workspace.volumes.read.return_value = _volume_info(volume_id="vid-A")
        gen = (
            workspace.temporary_volume_credentials.generate_temporary_volume_credentials
        )
        gen.return_value = _aws_creds_response()

        # Two distinct VolumePath instances pointing at the same volume.
        p1 = VolumePath("/Volumes/cat/sch/vol/x", service=service)
        p2 = VolumePath("/Volumes/cat/sch/vol/y", service=service)
        r1 = p1.credentials_refresher()
        r2 = p2.credentials_refresher()
        assert isinstance(r1, VolumeCredentialsRefresher)
        assert r1 is r2

    def test_get_credentials_per_mode_hits_right_operation(
        self,
        workspace,
        client,
        service,
    ) -> None:
        workspace.volumes.read.return_value = _volume_info(volume_id="vid-A")
        gen = (
            workspace.temporary_volume_credentials.generate_temporary_volume_credentials
        )
        gen.return_value = _aws_creds_response()

        p = VolumePath("/Volumes/cat/sch/vol/x", service=service)
        r = p.credentials_refresher()
        r.get_credentials(mode="read")
        r.get_credentials(mode="overwrite")
        ops = [_op_token(c.kwargs["operation"]) for c in gen.call_args_list]
        assert "READ_VOLUME" in ops and "WRITE_VOLUME" in ops

    def test_different_volume_id_yields_different_provider(
        self,
        workspace,
        client,
        service,
    ) -> None:
        gen = (
            workspace.temporary_volume_credentials.generate_temporary_volume_credentials
        )
        gen.return_value = _aws_creds_response()

        workspace.volumes.read.return_value = _volume_info(
            name="v1",
            volume_id="vid-A",
        )
        p1 = VolumePath("/Volumes/cat/sch/v1/x", service=service)
        r1 = p1.credentials_refresher()

        workspace.volumes.read.return_value = _volume_info(
            name="v2",
            volume_id="vid-B",
        )
        p2 = VolumePath("/Volumes/cat/sch/v2/x", service=service)
        r2 = p2.credentials_refresher()
        assert r1 is not r2

    def test_aws_client_shared_through_provider(
        self, workspace, client, service
    ) -> None:
        workspace.volumes.read.return_value = _volume_info(volume_id="vid-A")
        gen = (
            workspace.temporary_volume_credentials.generate_temporary_volume_credentials
        )
        gen.return_value = _aws_creds_response()

        p1 = VolumePath("/Volumes/cat/sch/vol/x", service=service)
        p2 = VolumePath("/Volumes/cat/sch/vol/y", service=service)
        # Same volume + mode + region → same AWSClient instance.
        c1 = p1.aws(mode="read", region="us-east-1")
        c2 = p2.aws(mode="read", region="us-east-1")
        assert c1 is c2

    def test_aws_client_cached_per_mode_and_region(
        self, workspace, client, service
    ) -> None:
        workspace.volumes.read.return_value = _volume_info(volume_id="vid-A")
        gen = (
            workspace.temporary_volume_credentials.generate_temporary_volume_credentials
        )
        gen.return_value = _aws_creds_response()

        p = VolumePath("/Volumes/cat/sch/vol/x", service=service)
        r = p.credentials_refresher()
        c_us = r.aws_client(mode="read", region="us-east-1")
        c_eu = r.aws_client(mode="read", region="eu-central-1")
        c_us_write = r.aws_client(mode="overwrite", region="us-east-1")
        c_us_again = r.aws_client(mode="read", region="us-east-1")
        assert c_us is c_us_again
        assert c_us is not c_eu
        assert c_us is not c_us_write

    def test_workspace_rebound_on_repeat_construction(
        self, workspace, client, service
    ) -> None:
        workspace.volumes.read.return_value = _volume_info(volume_id="vid-A")
        gen = (
            workspace.temporary_volume_credentials.generate_temporary_volume_credentials
        )
        gen.return_value = _aws_creds_response()

        # First binding uses ``workspace`` (fixture-provided).
        p1 = VolumePath("/Volumes/cat/sch/vol/x", service=service)
        r1 = p1.credentials_refresher()
        assert r1.workspace is workspace

        # Second binding with a different client must refresh the
        # singleton's ref so subsequent refresh cycles use the new
        # auth context.
        client_b = MagicMock()
        workspace_b = client_b.workspace_client.return_value
        workspace_b.volumes.read.return_value = _volume_info(volume_id="vid-A")
        workspace_b.temporary_volume_credentials.generate_temporary_volume_credentials.return_value = (
            _aws_creds_response()
        )
        service_b = MagicMock(spec=Volumes)
        service_b.client = client_b
        p2 = VolumePath("/Volumes/cat/sch/vol/y", service=service_b)
        r2 = p2.credentials_refresher()
        assert r2 is r1
        assert r2.workspace is workspace_b

    def test_pickle_collapses_to_live_singleton(
        self, workspace, client, service
    ) -> None:
        import pickle

        workspace.volumes.read.return_value = _volume_info(volume_id="vid-A")
        gen = (
            workspace.temporary_volume_credentials.generate_temporary_volume_credentials
        )
        gen.return_value = _aws_creds_response()

        p = VolumePath("/Volumes/cat/sch/vol/x", service=service)
        r = p.credentials_refresher()
        # In-process pickle round-trip must collapse to the same
        # singleton (no duplicate boto session, no duplicate refresh).
        loaded = pickle.loads(pickle.dumps(r))
        assert loaded is r

    def test_get_credentials_returns_canonical_credentials(
        self, workspace, client, service
    ) -> None:
        from yggdrasil.aws.config import AwsCredentials

        workspace.volumes.read.return_value = _volume_info(volume_id="vid-A")
        gen = (
            workspace.temporary_volume_credentials.generate_temporary_volume_credentials
        )
        gen.return_value = _aws_creds_response(
            access_key_id="AKIA-direct",
            secret_access_key="secret-direct",
        )

        p = VolumePath("/Volumes/cat/sch/vol/x", service=service)
        r = p.credentials_refresher()
        out = r.get_credentials(mode="read")
        assert isinstance(out, AwsCredentials)
        assert out.access_key_id == "AKIA-direct"
        assert out.secret_access_key == "secret-direct"


class TestVolumeInfoNotFoundRecovery:

    def test_creates_volume_on_not_found_then_re_reads(
        self, workspace, client, service
    ) -> None:
        # First ``volumes.read`` raises NotFound; the recovery path
        first_call = {"done": False}

        def read(full_name):
            if not first_call["done"]:
                first_call["done"] = True
                raise NotFound("Volume cat.sch.vol does not exist")
            return _volume_info()

        workspace.volumes.read.side_effect = read
        workspace.volumes.create.return_value = SimpleNamespace(
            volume_id="volume-uuid-0001",
        )

        p = VolumePath("/Volumes/cat/sch/vol/x", service=service)
        info = p.volume_info()
        assert info.volume_id == "volume-uuid-0001"
        assert workspace.volumes.read.call_count == 2
        # The create call should have run with the volume coordinates
        # parsed from the path.
        create_kwargs = workspace.volumes.create.call_args.kwargs
        assert create_kwargs["catalog_name"] == "cat"
        assert create_kwargs["schema_name"] == "sch"
        assert create_kwargs["name"] == "vol"

    def test_recovery_also_creates_missing_schema(
        self, workspace, client, service
    ) -> None:
        read_outcomes = [NotFound("missing"), _volume_info(volume_id="vid")]

        def read(full_name):
            outcome = read_outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        create_outcomes = [
            NotFound("schema does not exist"),
            SimpleNamespace(volume_id="vid"),
        ]

        def create(**kwargs):
            outcome = create_outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        workspace.volumes.read.side_effect = read
        workspace.volumes.create.side_effect = create

        p = VolumePath("/Volumes/cat/sch/vol/x", service=service)
        info = p.volume_info()
        assert info.volume_id == "vid"
        workspace.schemas.create.assert_called_once()

    def test_propagates_other_errors_unchanged(
        self, workspace, client, service
    ) -> None:
        # PermissionDenied is deterministic — the recovery path must
        # not swallow it.
        workspace.volumes.read.side_effect = PermissionDenied("nope")
        p = VolumePath("/Volumes/cat/sch/vol/x", service=service)
        with pytest.raises(PermissionDenied):
            p.volume_info()
        workspace.volumes.create.assert_not_called()

    def test_temporary_credentials_inherits_create_on_not_found(
        self,
        workspace,
        client,
        service,
    ) -> None:
        # ``temporary_credentials`` calls ``volume_info`` first, so
        # the create-on-NotFound flow surfaces transparently — the
        # caller gets back the AWS creds without ever seeing a
        # NotFound.
        read_calls = {"n": 0}

        def read(full_name):
            read_calls["n"] += 1
            if read_calls["n"] == 1:
                raise NotFound("missing")
            return _volume_info(volume_id="vid-after-create")

        workspace.volumes.read.side_effect = read
        workspace.volumes.create.return_value = SimpleNamespace(
            volume_id="vid-after-create",
        )
        gen = (
            workspace.temporary_volume_credentials.generate_temporary_volume_credentials
        )
        gen.return_value = _aws_creds_response()

        p = VolumePath("/Volumes/cat/sch/vol/x", service=service)
        resp = p.temporary_credentials(mode="read")
        assert resp.aws_temp_credentials.access_key_id == "AKIA-test"
        workspace.volumes.create.assert_called_once()
        gen.assert_called_once()
        assert gen.call_args.kwargs["volume_id"] == "vid-after-create"


# ---------------------------------------------------------------------------
# storage_path / arrow_filesystem — Path-shaped storage location
# ---------------------------------------------------------------------------


class TestStoragePath:

    def test_returns_s3_path_for_s3_volume(self, workspace, client, service) -> None:
        from yggdrasil.aws.fs.path import S3Path

        workspace.volumes.read.return_value = _volume_info(
            storage_location="s3://my-bucket/__unitystorage/cat/sch/vol",
        )
        gen = (
            workspace.temporary_volume_credentials.generate_temporary_volume_credentials
        )
        gen.return_value = _aws_creds_response()

        p = VolumePath("/Volumes/cat/sch/vol/x.parquet", service=service)
        root = p.storage_path(region="us-east-1")
        assert isinstance(root, S3Path)
        assert root.full_path() == "s3://my-bucket/__unitystorage/cat/sch/vol"

    def test_caches_path_on_instance(self, workspace, client, service) -> None:
        workspace.volumes.read.return_value = _volume_info(
            storage_location="s3://bkt/u/c/s/v",
        )
        gen = (
            workspace.temporary_volume_credentials.generate_temporary_volume_credentials
        )
        gen.return_value = _aws_creds_response()

        p = VolumePath("/Volumes/c/s/v/x", service=service)
        first = p.storage_path()
        second = p.storage_path()
        # Same Path instance — no rebuild on subsequent calls.
        assert first is second

    def test_refresh_drops_instance_cache(self, workspace, client, service) -> None:
        workspace.volumes.read.side_effect = [
            _volume_info(storage_location="s3://bkt/u/c/s/v"),
            _volume_info(storage_location="s3://bkt/u/c/s/v"),
        ]
        gen = (
            workspace.temporary_volume_credentials.generate_temporary_volume_credentials
        )
        gen.return_value = _aws_creds_response()

        p = VolumePath("/Volumes/c/s/v/x", service=service)
        p.storage_path()
        p.storage_path(refresh=True)
        # ``refresh=True`` forces a fresh ``volumes.read``; the
        # rebuilt :class:`S3Path` happens to collapse to the
        # singleton-by-URL instance, but the SDK was hit twice.
        assert workspace.volumes.read.call_count == 2

    def test_unsupported_scheme_raises(self, workspace, client, service) -> None:
        workspace.volumes.read.return_value = _volume_info(
            storage_location="ftp://nope/no",
        )
        p = VolumePath("/Volumes/c/s/v/x", service=service)
        with pytest.raises(ValueError, match="Unknown scheme"):
            p.storage_path()


class TestVolumeArrowFilesystem:

    def test_builds_pyarrow_s3_filesystem(self, workspace, client, service) -> None:
        # The credential snapshot path imports boto3 — skip cleanly
        # when the optional dep is missing instead of letting the
        # install probe hit the network.
        pytest.importorskip("boto3")

        import pyarrow.fs as pafs

        workspace.volumes.read.return_value = _volume_info(
            storage_location="s3://bkt/u/c/s/v",
        )
        gen = (
            workspace.temporary_volume_credentials.generate_temporary_volume_credentials
        )
        gen.return_value = _aws_creds_response()

        p = VolumePath("/Volumes/c/s/v/x", service=service)
        fs = p.arrow_filesystem(region="us-east-1")
        assert isinstance(fs, pafs.S3FileSystem)

    def test_arrow_filesystem_routes_through_s3service(
        self, workspace, client, service
    ) -> None:
        # Spy that ``VolumePath.arrow_filesystem`` actually goes
        # through ``S3Service.arrow_filesystem`` rather than building
        # the pyarrow object directly. That keeps the credential
        # snapshot logic centralized in one place.
        workspace.volumes.read.return_value = _volume_info(
            storage_location="s3://bkt/u/c/s/v",
        )
        gen = (
            workspace.temporary_volume_credentials.generate_temporary_volume_credentials
        )
        gen.return_value = _aws_creds_response()

        p = VolumePath("/Volumes/c/s/v/x", service=service)
        aws_client = p.aws(region="us-east-1")
        from unittest.mock import patch

        sentinel = object()
        with patch.object(
            type(aws_client.s3),
            "arrow_filesystem",
            return_value=sentinel,
        ) as spy:
            out = p.arrow_filesystem(region="us-east-1")
        assert out is sentinel
        spy.assert_called_once()


class TestS3ServiceArrowFilesystem:

    def test_snapshots_botocore_credentials(self) -> None:
        # ``S3Service.arrow_filesystem`` should pull a frozen
        # credentials snapshot from the boto session and hand it to
        # pyarrow's S3FileSystem. We patch out the actual
        # construction so the test doesn't touch the network.
        pytest.importorskip("boto3")

        import pyarrow.fs as pafs
        from yggdrasil.aws import AWSClient
        from yggdrasil.aws.fs.service import S3Service

        client = AWSClient(
            access_key_id="AKIA",
            secret_access_key="secret",
            session_token="tok",
            region="us-east-1",
        )
        service = S3Service(service=service)
        fs = service.arrow_filesystem()
        assert isinstance(fs, pafs.S3FileSystem)

    def test_region_override(self) -> None:
        pytest.importorskip("boto3")

        import pyarrow.fs as pafs
        from yggdrasil.aws import AWSClient
        from yggdrasil.aws.fs.service import S3Service

        client = AWSClient(
            access_key_id="AKIA",
            secret_access_key="secret",
            region="us-east-1",
        )
        service = S3Service(service=service)
        # The override region should land on the pyarrow filesystem;
        # there isn't a public reader on S3FileSystem for the region,
        # but constructing without error is sufficient signal.
        fs = service.arrow_filesystem(region="eu-central-1")
        assert isinstance(fs, pafs.S3FileSystem)


class TestUCNavigation:
    """``catalog_name`` / ``schema_name`` / ``volume_name`` plus the
    lazy ``catalog`` / ``schema`` properties."""

    def test_names_under_volume(self, workspace, client, service) -> None:
        p = VolumePath("/Volumes/cat/sch/vol/sub/x.bin", service=service)
        assert p.catalog_name == "cat"
        assert p.schema_name == "sch"
        assert p.volume_name == "vol"

    def test_names_none_for_volumes_root(self, workspace, client, service) -> None:
        p = VolumePath("/Volumes", service=service)
        assert p.catalog_name is None
        assert p.schema_name is None
        assert p.volume_name is None

    def test_catalog_property_is_cached(self, workspace, client, service) -> None:
        sentinel = object()
        client.catalogs.catalog.return_value = sentinel
        p = VolumePath("/Volumes/cat/sch/vol/x", service=service)
        assert p.catalog is sentinel
        assert p.catalog is sentinel
        client.catalogs.catalog.assert_called_once_with("cat")

    def test_schema_property_is_cached(self, workspace, client, service) -> None:
        sentinel = object()
        client.schemas.schema.return_value = sentinel
        p = VolumePath("/Volumes/cat/sch/vol/x", service=service)
        assert p.schema is sentinel
        assert p.schema is sentinel
        client.schemas.schema.assert_called_once_with(
            catalog_name="cat",
            schema_name="sch",
        )

    def test_catalog_raises_without_uc_prefix(self, workspace, client, service) -> None:
        p = VolumePath("/Volumes", service=service)
        with pytest.raises(ValueError, match="/Volumes/<cat>/<sch>/<vol>"):
            _ = p.catalog

    def test_schema_raises_without_uc_prefix(self, workspace, client, service) -> None:
        p = VolumePath("/Volumes", service=service)
        with pytest.raises(ValueError, match="/Volumes/<cat>/<sch>/<vol>"):
            _ = p.schema
