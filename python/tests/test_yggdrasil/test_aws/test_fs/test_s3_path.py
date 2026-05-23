"""Mock-driven behavior tests for :class:`yggdrasil.aws.fs.path.S3Path`.

Tests inject a :class:`unittest.mock.Mock` shaped like
:class:`S3Service` whose ``boto_client`` attribute is the mock
boto3 S3 client. The Path implementation must call the boto methods
through ``self.service.boto_client`` with the right arguments and
translate responses + errors into the :class:`Holder` contract.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from yggdrasil.aws.fs.path import S3Path
from yggdrasil.aws.fs.service import S3Service
from yggdrasil.io.io_stats import IOKind

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Body:
    """Minimal boto3 ``StreamingBody`` double — read once, close cleanly."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self.closed = False

    def read(self) -> bytes:
        return self._data

    def close(self) -> None:
        self.closed = True


def _client_error(code: str = "NoSuchKey", status: int = 404) -> Exception:
    """A boto-shaped error mimicking ``ClientError`` duck-typed
    ``response`` attribute."""
    exc = Exception(f"{code} {status}")
    exc.response = {  # type: ignore[attr-defined]
        "Error": {"Code": code},
        "ResponseMetadata": {"HTTPStatusCode": status},
    }
    return exc


def _make_paginator(pages: list[dict]) -> Any:
    paginator = MagicMock()
    paginator.paginate = MagicMock(return_value=iter(pages))
    return paginator


@pytest.fixture(autouse=True)
def reset_remote_stat_cache():
    yield


@pytest.fixture
def client():
    """Mock boto3-shaped S3 client.

    Methods used by S3Path that the test sets up per-case:
    ``head_object``, ``list_objects_v2``, ``get_object``,
    ``put_object``, ``delete_object``, ``delete_objects``,
    ``get_paginator``.
    """
    return MagicMock()


@pytest.fixture
def service(client):
    """Mock :class:`S3Service` whose ``boto_client`` is the test
    fixture's boto :func:`client`.

    :class:`S3Path` reaches the boto client through
    ``self.service.boto_client``, so wiring the mock service to
    expose the same underlying mock keeps the per-test
    ``client.head_object.return_value = ...`` setup pattern
    working unchanged.
    """
    svc = MagicMock(spec=S3Service)
    svc.boto_client = client
    return svc


# ---------------------------------------------------------------------------
# URL parsing + scheme normalization
# ---------------------------------------------------------------------------


class TestUrlParsing:

    def test_basic_url(self, client, service) -> None:
        p = S3Path("s3://my-bucket/data/file.parquet", service=service)
        assert p.bucket == "my-bucket"
        assert p.key == "data/file.parquet"
        assert p.full_path() == "s3://my-bucket/data/file.parquet"

    def test_bucket_root(self, client, service) -> None:
        p = S3Path("s3://my-bucket/", service=service)
        assert p.bucket == "my-bucket"
        assert p.key == ""
        assert p.full_path() == "s3://my-bucket/"

    def test_s3a_normalizes_to_s3(self, client, service) -> None:
        p = S3Path("s3a://my-bucket/x", service=service)
        assert p.url.scheme == "s3"
        assert p.full_path().startswith("s3://")

    def test_s3n_normalizes_to_s3(self, client, service) -> None:
        p = S3Path("s3n://my-bucket/x", service=service)
        assert p.url.scheme == "s3"


class TestPredicates:

    def test_remote_path_pins(self, client, service) -> None:
        p = S3Path("s3://b/k", service=service)
        assert p.is_remote_path
        assert not p.is_local_path
        assert not p.is_memory


# ---------------------------------------------------------------------------
# Stat
# ---------------------------------------------------------------------------


class TestStat:

    def test_existing_object(self, client, service) -> None:
        client.head_object.return_value = {
            "ContentLength": 42,
            "LastModified": None,
        }
        p = S3Path("s3://b/k", service=service)
        s = p._stat_uncached()
        assert s.kind is IOKind.FILE
        assert s.size == 42
        client.head_object.assert_called_once_with(Bucket="b", Key="k")

    def test_existing_prefix(self, client, service) -> None:
        client.head_object.side_effect = _client_error()
        client.list_objects_v2.return_value = {"KeyCount": 3}
        p = S3Path("s3://b/folder", service=service)
        s = p._stat_uncached()
        assert s.kind is IOKind.DIRECTORY

    def test_missing_object(self, client, service) -> None:
        client.head_object.side_effect = _client_error()
        client.list_objects_v2.return_value = {"KeyCount": 0}
        p = S3Path("s3://b/no-such", service=service)
        s = p._stat_uncached()
        assert s.kind is IOKind.MISSING

    def test_bucket_root_is_directory(self, client, service) -> None:
        p = S3Path("s3://b/", service=service)
        s = p._stat_uncached()
        assert s.kind is IOKind.DIRECTORY
        # Bucket-root probe must NOT call head_object.
        client.head_object.assert_not_called()


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


class TestReadMv:

    def test_full_object_read(self, client, service) -> None:
        client.head_object.return_value = {"ContentLength": 5, "LastModified": None}
        client.get_object.return_value = {
            "Body": _Body(b"hello"),
            "ContentLength": 5,
        }
        p = S3Path("s3://b/k", service=service)
        assert p.read_bytes() == b"hello"
        # ``read_bytes()`` / ``read_mv(-1, 0)`` issues a single
        # whole-object GET (no ``Range`` header) — the ``HeadObject``
        # probe the base ``Holder.read_mv`` used to run is skipped
        # because S3 surfaces the canonical size on the GET response
        # and we seed the stat cache from there.
        kwargs = client.get_object.call_args.kwargs
        assert kwargs["Bucket"] == "b"
        assert kwargs["Key"] == "k"
        assert "Range" not in kwargs
        client.head_object.assert_not_called()

    def test_range_read(self, client, service) -> None:
        client.head_object.return_value = {"ContentLength": 100, "LastModified": None}
        client.get_object.return_value = {"Body": _Body(b"abc")}
        # Disable the page buffer so the test asserts the narrow
        # ``Range`` shape the user requested instead of the page-aligned
        # ``Range`` :class:`RemotePath` issues when buffersize is set.
        p = S3Path("s3://b/k", service=service, buffersize=None)
        out = p.pread(3, 10)
        assert out == b"abc"
        assert client.get_object.call_args.kwargs["Range"] == "bytes=10-12"

    def test_zero_byte_read_short_circuits(self, client, service) -> None:
        # n=0 is a no-op — the reworked Holder.read_mv normalizes to
        # an empty memoryview without hitting the client.
        client.head_object.return_value = {"ContentLength": 5, "LastModified": None}
        p = S3Path("s3://b/k", service=service)
        assert p.pread(0, 0) == b""
        client.get_object.assert_not_called()

    def test_missing_key_raises_filenotfounderror(self, client, service) -> None:
        client.head_object.return_value = {"ContentLength": 5, "LastModified": None}
        client.get_object.side_effect = _client_error()
        p = S3Path("s3://b/k", service=service)
        with pytest.raises(FileNotFoundError, match="s3://b/k"):
            p.read_bytes()

    def test_invalid_range_returns_empty(self, client, service) -> None:
        client.head_object.return_value = {"ContentLength": 5, "LastModified": None}
        client.get_object.side_effect = _client_error("InvalidRange", 416)
        p = S3Path("s3://b/k", service=service)
        # With size=5 the holder bounds n to remaining; force a bigger
        # range manually via pread to exercise the 416 branch.
        # ``pread`` resolves through ``read_mv``; ask for a window
        # outside bounds — read_mv will raise on bounds, so call the
        # private ``_read_mv`` directly.
        assert bytes(p._read_mv(10, 3)) == b""


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


class TestWrite:

    def test_whole_object_put(self, client, service) -> None:
        client.head_object.side_effect = _client_error()
        client.list_objects_v2.return_value = {"KeyCount": 0}
        p = S3Path("s3://b/k", service=service)
        n = p.write_bytes(b"abcdef")
        assert n == 6
        kwargs = client.put_object.call_args.kwargs
        assert kwargs["Bucket"] == "b"
        assert kwargs["Key"] == "k"
        assert kwargs["Body"] == b"abcdef"

    def test_pwrite_does_rmw(self, client, service) -> None:
        # Existing 5 bytes; positional write at offset 1 splices in 'XX'.
        client.head_object.return_value = {"ContentLength": 5, "LastModified": None}
        client.get_object.return_value = {"Body": _Body(b"abcde")}
        p = S3Path("s3://b/k", service=service)
        p.pwrite(b"XX", 1)
        body = client.put_object.call_args.kwargs["Body"]
        assert body == b"aXXde"

    def test_pwrite_past_eof_zero_pads(self, client, service) -> None:
        client.head_object.return_value = {"ContentLength": 2, "LastModified": None}
        client.get_object.return_value = {"Body": _Body(b"ab")}
        p = S3Path("s3://b/k", service=service)
        p.pwrite(b"X", 5)
        body = client.put_object.call_args.kwargs["Body"]
        assert body == b"ab\x00\x00\x00X"

    def test_truncate_shrinks(self, client, service) -> None:
        client.head_object.return_value = {"ContentLength": 6, "LastModified": None}
        client.get_object.return_value = {"Body": _Body(b"abcdef")}
        p = S3Path("s3://b/k", service=service)
        p.truncate(3)
        body = client.put_object.call_args.kwargs["Body"]
        assert body == b"abc"

    def test_truncate_to_zero_uploads_empty(self, client, service) -> None:
        client.head_object.return_value = {"ContentLength": 6, "LastModified": None}
        client.get_object.return_value = {"Body": _Body(b"abcdef")}
        p = S3Path("s3://b/k", service=service)
        p.truncate(0)
        body = client.put_object.call_args.kwargs["Body"]
        assert body == b""


class TestWriteAll:

    def test_single_put_no_stat_no_get(self, client, service) -> None:
        p = S3Path("s3://b/data.bin", service=service)
        n = p.write_all(b"hello-world")
        assert n == 11
        client.put_object.assert_called_once()
        kwargs = client.put_object.call_args.kwargs
        assert kwargs["Bucket"] == "b"
        assert kwargs["Key"] == "data.bin"
        assert kwargs["Body"] == b"hello-world"
        client.head_object.assert_not_called()
        client.get_object.assert_not_called()

    def test_stream_input(self, client, service) -> None:
        import io

        p = S3Path("s3://b/data.bin", service=service)
        p.write_all(io.BytesIO(b"streamed"))
        assert client.put_object.call_args.kwargs["Body"] == b"streamed"
        client.head_object.assert_not_called()

    def test_memoryview_input(self, client, service) -> None:
        p = S3Path("s3://b/data.bin", service=service)
        p.write_all(memoryview(b"view"))
        assert client.put_object.call_args.kwargs["Body"] == b"view"

    def test_parquet_roundtrip(self, client, service) -> None:
        import io
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({"id": pa.array([1, 2, 3], type=pa.int64())})
        sink = io.BytesIO()
        pq.write_table(table, sink)
        parquet_bytes = sink.getvalue()

        p = S3Path("s3://b/out.parquet", service=service)
        n = p.write_all(parquet_bytes)

        assert n == len(parquet_bytes)
        client.put_object.assert_called_once()
        client.head_object.assert_not_called()
        client.get_object.assert_not_called()

        sent = client.put_object.call_args.kwargs["Body"]
        roundtrip = pa.BufferReader(sent)
        assert pq.read_table(roundtrip).equals(table)

    def test_seeds_stat_cache(self, client, service) -> None:
        p = S3Path("s3://b/k", service=service)
        p.write_all(b"12345")
        assert p.size == 5
        client.head_object.assert_not_called()

    def test_fewer_sdk_calls_than_write_bytes(self, client, service) -> None:
        client.head_object.side_effect = _client_error()
        client.list_objects_v2.return_value = {"KeyCount": 0}

        p1 = S3Path("s3://b/a.bin", service=service)
        p1.write_bytes(b"via-write-bytes")
        wb_put_count = client.put_object.call_count
        wb_head_count = client.head_object.call_count

        client.put_object.reset_mock()
        client.head_object.reset_mock()
        client.head_object.side_effect = _client_error()

        p2 = S3Path("s3://b/b.bin", service=service)
        p2.write_all(b"via-write-all")
        wa_put_count = client.put_object.call_count
        wa_head_count = client.head_object.call_count

        assert wa_put_count == 1
        assert wa_head_count == 0
        assert wa_put_count <= wb_put_count
        assert wa_head_count < wb_head_count


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


class TestListing:

    def test_iterdir_mixes_objects_and_prefixes(self, client, service) -> None:
        pages = [
            {
                "Contents": [
                    {"Key": "data/a.parquet", "Size": 100, "LastModified": None},
                    {"Key": "data/b.parquet", "Size": 200, "LastModified": None},
                ],
                "CommonPrefixes": [{"Prefix": "data/sub/"}],
            }
        ]
        client.get_paginator.return_value = _make_paginator(pages)
        p = S3Path("s3://b/data/", service=service)
        children = list(p.iterdir())
        assert sorted(c.key for c in children) == [
            "data/a.parquet",
            "data/b.parquet",
            "data/sub/",
        ]
        # Non-recursive listing uses Delimiter='/'.
        client.get_paginator.return_value.paginate.assert_called_once()
        kwargs = client.get_paginator.return_value.paginate.call_args.kwargs
        assert kwargs["Delimiter"] == "/"
        assert kwargs["Prefix"] == "data/"

    def test_recursive_drops_delimiter(self, client, service) -> None:
        pages = [{"Contents": [{"Key": "data/x.parquet"}]}]
        client.get_paginator.return_value = _make_paginator(pages)
        p = S3Path("s3://b/data/", service=service)
        list(p.ls(recursive=True))
        kwargs = client.get_paginator.return_value.paginate.call_args.kwargs
        assert "Delimiter" not in kwargs

    def test_iterdir_filters_zero_byte_placeholders(self, client, service) -> None:
        pages = [
            {
                "Contents": [
                    {"Key": "data/", "Size": 0},
                    {"Key": "data/real.parquet", "Size": 5},
                ],
            }
        ]
        client.get_paginator.return_value = _make_paginator(pages)
        p = S3Path("s3://b/data/", service=service)
        names = [c.key for c in p.iterdir()]
        # The "data/" placeholder is suppressed; only the real file
        # comes through.
        assert names == ["data/real.parquet"]


# ---------------------------------------------------------------------------
# Mkdir / remove
# ---------------------------------------------------------------------------


class TestRemove:

    def test_unlink_calls_delete_object(self, client, service) -> None:
        client.head_object.return_value = {"ContentLength": 1, "LastModified": None}
        p = S3Path("s3://b/key", service=service)
        p.unlink()
        client.delete_object.assert_called_once_with(Bucket="b", Key="key")

    def test_remove_dir_paginates_and_batches(self, client, service) -> None:
        pages = [{"Contents": [{"Key": f"d/x{i}"} for i in range(3)]}]
        client.get_paginator.return_value = _make_paginator(pages)
        client.delete_objects.return_value = {}
        p = S3Path("s3://b/d/", service=service)
        # Force the directory branch.
        client.head_object.side_effect = _client_error()
        client.list_objects_v2.return_value = {"KeyCount": 3}
        p.remove(recursive=True)
        kwargs = client.delete_objects.call_args.kwargs
        assert kwargs["Bucket"] == "b"
        assert [o["Key"] for o in kwargs["Delete"]["Objects"]] == [
            "d/x0",
            "d/x1",
            "d/x2",
        ]


class TestMkdir:

    def test_is_no_op(self, client, service) -> None:
        p = S3Path("s3://b/folder", service=service)
        p.mkdir()
        # No S3 call — directory creation is implicit on first child.
        client.put_object.assert_not_called()


# ---------------------------------------------------------------------------
# Path API delegation
# ---------------------------------------------------------------------------


class TestPathApi:

    def test_joinpath_inherits_service(self, client, service) -> None:
        p = S3Path("s3://b/folder/", service=service)
        child = p / "leaf.parquet"
        assert isinstance(child, S3Path)
        assert child.bucket == "b"
        assert child.key == "folder/leaf.parquet"
        # Same service object propagates — and the boto client
        # reached through it is the test fixture's mock.
        assert child._service is service
        assert child.client is client

    def test_with_suffix(self, client, service) -> None:
        p = S3Path("s3://b/data/x.csv", service=service)
        renamed = p.with_suffix(".parquet")
        assert renamed.key == "data/x.parquet"


# ---------------------------------------------------------------------------
# Stat caching (via RemotePath)
# ---------------------------------------------------------------------------


class TestStatCaching:

    def test_repeated_stat_hits_cache(self, client, service) -> None:
        client.head_object.return_value = {"ContentLength": 5, "LastModified": None}
        p = S3Path("s3://b/k", service=service)
        # First stat → real call.
        first = p.stat()
        second = p.stat()
        assert first is second
        assert client.head_object.call_count == 1

    def test_write_seeds_post_write_size(self, client, service) -> None:
        # Initial probe → 5 bytes.
        client.head_object.return_value = {"ContentLength": 5, "LastModified": None}
        client.list_objects_v2.return_value = {"KeyCount": 0}
        client.get_object.return_value = {"Body": _Body(b"abcde")}
        p = S3Path("s3://b/k", service=service)
        assert p.size == 5
        # ``truncate`` re-uploads via ``put_object``, which seeds the
        # stat cache with the post-write size — the next ``size``
        # lookup hits the seeded entry, no second ``HeadObject``.
        p.truncate(2)
        assert p.size == 2
        assert client.head_object.call_count == 1


# ---------------------------------------------------------------------------
# Retry behavior
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    """Transient SDK errors retry up to 4 times with a flat 1 s sleep
    between attempts; permission errors fail fast (no retry);
    deterministic errors (NotFound, etc.) propagate immediately."""

    @pytest.fixture
    def sleeps(self):
        recorded: list[float] = []

        def spy(t: float) -> None:
            recorded.append(t)

        return recorded, spy

    def test_transient_eventually_succeeds(self, client, sleeps, service) -> None:
        recorded, spy = sleeps
        responses = [
            _client_error("InternalError", 500),
            _client_error("BadRequest", 400),
            {"ContentLength": 7, "LastModified": None},
        ]

        def head_object(**kwargs):
            r = responses.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        client.head_object.side_effect = head_object
        p = S3Path("s3://b/k", service=service, retry_sleep=spy)
        assert p.size == 7
        # Two transient retries → two flat 1 s sleeps.
        assert recorded == [1.0, 1.0]
        assert client.head_object.call_count == 3

    def test_transient_gives_up_after_4_retries(self, client, sleeps, service) -> None:
        recorded, spy = sleeps
        client.head_object.side_effect = _client_error("InternalError", 500)
        p = S3Path("s3://b/k", service=service, retry_sleep=spy)
        with pytest.raises(Exception):
            p.stat()
        # 4 retries fired (flat 1 s each) before giving up; 5 head calls total.
        assert recorded == [1.0, 1.0, 1.0, 1.0]
        assert client.head_object.call_count == 5

    def test_permission_fails_fast(self, client, sleeps, service) -> None:
        recorded, spy = sleeps
        client.head_object.side_effect = _client_error("AccessDenied", 403)
        p = S3Path("s3://b/k", service=service, retry_sleep=spy)
        with pytest.raises(Exception):
            p.stat()
        # Permission errors are deterministic; no retry, no sleep.
        assert recorded == []
        assert client.head_object.call_count == 1

    def test_not_found_does_not_retry(self, client, sleeps, service) -> None:
        recorded, spy = sleeps
        client.head_object.side_effect = _client_error("NoSuchKey", 404)
        client.list_objects_v2.return_value = {"KeyCount": 0}
        p = S3Path("s3://b/k", service=service, retry_sleep=spy)
        s = p._stat_uncached()
        assert s.kind is IOKind.MISSING
        # No sleeps — NotFound is deterministic.
        assert recorded == []

    def test_deterministic_errors_propagate(self, client, sleeps, service) -> None:
        recorded, spy = sleeps
        client.put_object.side_effect = ValueError("not retryable")
        p = S3Path("s3://b/k", service=service, retry_sleep=spy)
        # Force an OVERWRITE of an empty object → put_object once.
        client.head_object.side_effect = _client_error()
        client.list_objects_v2.return_value = {"KeyCount": 0}
        with pytest.raises(ValueError, match="not retryable"):
            p.write_bytes(b"x")
        assert recorded == []


# ---------------------------------------------------------------------------
# PyArrow filesystem fast path
# ---------------------------------------------------------------------------


class TestSingletonCaching:
    """:class:`S3Path` inherits :class:`Singleton` — two callers asking
    for the same ``(URL, service)`` share the live instance + warm
    stat cache. Different URLs or different services land on distinct
    instances."""

    def setup_method(self) -> None:
        # Drop any cached instances so each test starts clean.
        S3Path._INSTANCES.clear()

    def test_same_url_and_client_returns_same_instance(self, client, service) -> None:
        a = S3Path("s3://bucket/key.parquet", service=service)
        b = S3Path("s3://bucket/key.parquet", service=service)
        assert a is b

    def test_init_is_idempotent_under_cache_hit(self, client, service) -> None:
        # Mutate after construction; the second constructor call must
        # NOT clobber the live state (preserves bound client + warm
        # stat cache).
        a = S3Path("s3://bucket/k", service=service)
        a._stat_cached = "sentinel-cache"  # type: ignore[assignment]
        b = S3Path("s3://bucket/k", service=service)
        assert a is b
        assert b._stat_cached == "sentinel-cache"

    def test_different_url_returns_different_instance(self, client, service) -> None:
        a = S3Path("s3://bucket/k1", service=service)
        b = S3Path("s3://bucket/k2", service=service)
        assert a is not b

    def test_different_service_returns_different_instance(self) -> None:
        c1 = MagicMock()
        c2 = MagicMock()
        s1 = MagicMock(spec=S3Service)
        s1.boto_client = c1
        s2 = MagicMock(spec=S3Service)
        s2.boto_client = c2
        a = S3Path("s3://bucket/k", service=s1)
        b = S3Path("s3://bucket/k", service=s2)
        assert a is not b
        assert a.client is c1
        assert b.client is c2

    def test_scheme_aliases_collapse_onto_canonical(self, client, service) -> None:
        # ``s3a://`` and ``s3n://`` normalize to ``s3://`` at construction;
        # the singleton key reflects the canonical URL so all three
        # spellings share one instance.
        a = S3Path("s3://bucket/k", service=service)
        b = S3Path("s3a://bucket/k", service=service)
        c = S3Path("s3n://bucket/k", service=service)
        assert a is b is c

    def test_stat_cache_shared_across_constructions(self, client, service) -> None:
        # First construction warms the stat cache via _stat(). Re-
        # constructing returns the same instance, so the second probe
        # rides the cached entry instead of re-issuing head_object.
        client.head_object.return_value = {
            "ContentLength": 42,
            "LastModified": None,
            "ContentType": "application/octet-stream",
        }
        a = S3Path("s3://bucket/k", service=service)
        assert a._stat().size == 42
        assert client.head_object.call_count == 1

        b = S3Path("s3://bucket/k", service=service)
        assert b._stat().size == 42
        # Still only one head_object — the second path *is* the first
        # path, and its stat cache is fresh.
        assert client.head_object.call_count == 1


class TestArrowFilesystem:

    def test_arrow_uri_renders_bucket_key(self, client, service) -> None:
        p = S3Path("s3://my-bucket/path/to/file.parquet", service=service)
        assert p.arrow_uri == "my-bucket/path/to/file.parquet"

    def test_arrow_filesystem_returns_pyarrow_s3fs(self) -> None:
        # Building a real AWSClient pulls in boto3 — skip when the
        # optional dep is missing instead of letting the install probe
        # hit the network.
        pytest.importorskip("boto3")

        import pyarrow.fs as pafs
        from yggdrasil.aws import AWSClient
        from yggdrasil.aws.fs.service import S3Service

        # Build a real AWSClient with static creds and bind an
        # S3Path to its service. ``arrow_filesystem`` should snapshot
        # the boto session and hand back pyarrow's S3FS.
        aws = AWSClient(
            access_key_id="AKIA",
            secret_access_key="secret",
            region="us-east-1",
        )
        # ``S3Service.path`` builds an S3Path bound to this service;
        # the service is stored on ``_service`` and the boto client
        # is reached through ``service.boto_client``.
        real_service = S3Service(client=aws)
        p = real_service.path("s3://b/k")
        fs = p.arrow_filesystem()
        assert isinstance(fs, pafs.S3FileSystem)
