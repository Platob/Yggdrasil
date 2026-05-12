"""Mock-driven behavior tests for :class:`yggdrasil.aws.fs.path.S3Path`.

Tests inject a :class:`unittest.mock.Mock` shaped like the boto3 S3
client. The Path implementation must call the boto methods with the
right arguments and translate responses + errors into the
:class:`Holder` contract.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from yggdrasil.aws.fs.path import S3Path
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
    from yggdrasil.io.path.remote_path import RemotePath
    RemotePath._INSTANCES.clear()
    yield
    RemotePath._INSTANCES.clear()


@pytest.fixture
def client():
    """Mock boto3-shaped S3 client.

    Methods used by S3Path that the test sets up per-case:
    ``head_object``, ``list_objects_v2``, ``get_object``,
    ``put_object``, ``delete_object``, ``delete_objects``,
    ``get_paginator``.
    """
    return MagicMock()


# ---------------------------------------------------------------------------
# URL parsing + scheme normalization
# ---------------------------------------------------------------------------


class TestUrlParsing:

    def test_basic_url(self, client) -> None:
        p = S3Path("s3://my-bucket/data/file.parquet", client=client)
        assert p.bucket == "my-bucket"
        assert p.key == "data/file.parquet"
        assert p.full_path() == "s3://my-bucket/data/file.parquet"

    def test_bucket_root(self, client) -> None:
        p = S3Path("s3://my-bucket/", client=client)
        assert p.bucket == "my-bucket"
        assert p.key == ""
        assert p.full_path() == "s3://my-bucket/"

    def test_s3a_normalizes_to_s3(self, client) -> None:
        p = S3Path("s3a://my-bucket/x", client=client)
        assert p.url.scheme == "s3"
        assert p.full_path().startswith("s3://")

    def test_s3n_normalizes_to_s3(self, client) -> None:
        p = S3Path("s3n://my-bucket/x", client=client)
        assert p.url.scheme == "s3"


class TestSingletonCache:
    """``RemotePath.__new__`` collapses repeated constructions for the
    same URL onto one cached instance.

    The optimization reuses the URL parsed once for the cache key so
    the subclass ``__init__`` doesn't re-parse, and the new pre-allocate
    cache lookup short-circuits the ``Holder.__new__`` dispatch chain
    entirely on warm hits.
    """

    def test_str_form_returns_same_instance(self, client) -> None:
        a = S3Path("s3://bucket/key", client=client)
        b = S3Path("s3://bucket/key", client=client)
        assert a is b

    def test_url_form_returns_same_instance_as_str_form(self, client) -> None:
        from yggdrasil.io.url import URL
        a = S3Path("s3://bucket/key", client=client)
        b = S3Path(url=URL.from_("s3://bucket/key"), client=client)
        assert a is b

    def test_fresh_construction_drops_resolved_url_after_init(self, client) -> None:
        # ``__new__`` stashes the parsed URL onto the new instance so
        # subclass ``__init__`` doesn't re-parse. After init the slot
        # is cleared so the parsed URL isn't kept alive twice.
        p = S3Path("s3://bucket/key", client=client)
        assert p._resolved_url is None


class TestPredicates:

    def test_remote_path_pins(self, client) -> None:
        p = S3Path("s3://b/k", client=client)
        assert p.is_remote_path
        assert not p.is_local_path
        assert not p.is_memory


# ---------------------------------------------------------------------------
# Stat
# ---------------------------------------------------------------------------


class TestStat:

    def test_existing_object(self, client) -> None:
        client.head_object.return_value = {
            "ContentLength": 42,
            "LastModified": None,
        }
        p = S3Path("s3://b/k", client=client)
        s = p._stat_uncached()
        assert s.kind is IOKind.FILE
        assert s.size == 42
        client.head_object.assert_called_once_with(Bucket="b", Key="k")

    def test_existing_prefix(self, client) -> None:
        client.head_object.side_effect = _client_error()
        client.list_objects_v2.return_value = {"KeyCount": 3}
        p = S3Path("s3://b/folder", client=client)
        s = p._stat_uncached()
        assert s.kind is IOKind.DIRECTORY

    def test_missing_object(self, client) -> None:
        client.head_object.side_effect = _client_error()
        client.list_objects_v2.return_value = {"KeyCount": 0}
        p = S3Path("s3://b/no-such", client=client)
        s = p._stat_uncached()
        assert s.kind is IOKind.MISSING

    def test_bucket_root_is_directory(self, client) -> None:
        p = S3Path("s3://b/", client=client)
        s = p._stat_uncached()
        assert s.kind is IOKind.DIRECTORY
        # Bucket-root probe must NOT call head_object.
        client.head_object.assert_not_called()


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


class TestReadMv:

    def test_full_object_read(self, client) -> None:
        client.head_object.return_value = {"ContentLength": 5, "LastModified": None}
        client.get_object.return_value = {"Body": _Body(b"hello")}
        p = S3Path("s3://b/k", client=client)
        assert p.read_bytes() == b"hello"
        # Range header covers [0, size-1] inclusive.
        kwargs = client.get_object.call_args.kwargs
        assert kwargs["Bucket"] == "b"
        assert kwargs["Key"] == "k"
        assert kwargs["Range"] == "bytes=0-4"

    def test_range_read(self, client) -> None:
        client.head_object.return_value = {"ContentLength": 100, "LastModified": None}
        client.get_object.return_value = {"Body": _Body(b"abc")}
        p = S3Path("s3://b/k", client=client)
        out = p.pread(3, 10)
        assert out == b"abc"
        assert client.get_object.call_args.kwargs["Range"] == "bytes=10-12"

    def test_zero_byte_read_short_circuits(self, client) -> None:
        # n=0 is a no-op — the reworked Holder.read_mv normalizes to
        # an empty memoryview without hitting the client.
        client.head_object.return_value = {"ContentLength": 5, "LastModified": None}
        p = S3Path("s3://b/k", client=client)
        assert p.pread(0, 0) == b""
        client.get_object.assert_not_called()

    def test_missing_key_raises_filenotfounderror(self, client) -> None:
        client.head_object.return_value = {"ContentLength": 5, "LastModified": None}
        client.get_object.side_effect = _client_error()
        p = S3Path("s3://b/k", client=client)
        with pytest.raises(FileNotFoundError, match="s3://b/k"):
            p.read_bytes()

    def test_invalid_range_returns_empty(self, client) -> None:
        client.head_object.return_value = {"ContentLength": 5, "LastModified": None}
        client.get_object.side_effect = _client_error("InvalidRange", 416)
        p = S3Path("s3://b/k", client=client)
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

    def test_whole_object_put(self, client) -> None:
        client.head_object.side_effect = _client_error()
        client.list_objects_v2.return_value = {"KeyCount": 0}
        p = S3Path("s3://b/k", client=client)
        n = p.write_bytes(b"abcdef")
        assert n == 6
        kwargs = client.put_object.call_args.kwargs
        assert kwargs["Bucket"] == "b"
        assert kwargs["Key"] == "k"
        assert kwargs["Body"] == b"abcdef"

    def test_pwrite_does_rmw(self, client) -> None:
        # Existing 5 bytes; positional write at offset 1 splices in 'XX'.
        client.head_object.return_value = {"ContentLength": 5, "LastModified": None}
        client.get_object.return_value = {"Body": _Body(b"abcde")}
        p = S3Path("s3://b/k", client=client)
        p.pwrite(b"XX", 1)
        body = client.put_object.call_args.kwargs["Body"]
        assert body == b"aXXde"

    def test_pwrite_past_eof_zero_pads(self, client) -> None:
        client.head_object.return_value = {"ContentLength": 2, "LastModified": None}
        client.get_object.return_value = {"Body": _Body(b"ab")}
        p = S3Path("s3://b/k", client=client)
        p.pwrite(b"X", 5)
        body = client.put_object.call_args.kwargs["Body"]
        assert body == b"ab\x00\x00\x00X"

    def test_truncate_shrinks(self, client) -> None:
        client.head_object.return_value = {"ContentLength": 6, "LastModified": None}
        client.get_object.return_value = {"Body": _Body(b"abcdef")}
        p = S3Path("s3://b/k", client=client)
        p.truncate(3)
        body = client.put_object.call_args.kwargs["Body"]
        assert body == b"abc"

    def test_truncate_to_zero_uploads_empty(self, client) -> None:
        client.head_object.return_value = {"ContentLength": 6, "LastModified": None}
        client.get_object.return_value = {"Body": _Body(b"abcdef")}
        p = S3Path("s3://b/k", client=client)
        p.truncate(0)
        body = client.put_object.call_args.kwargs["Body"]
        assert body == b""


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


class TestListing:

    def test_iterdir_mixes_objects_and_prefixes(self, client) -> None:
        pages = [{
            "Contents": [
                {"Key": "data/a.parquet", "Size": 100, "LastModified": None},
                {"Key": "data/b.parquet", "Size": 200, "LastModified": None},
            ],
            "CommonPrefixes": [{"Prefix": "data/sub/"}],
        }]
        client.get_paginator.return_value = _make_paginator(pages)
        p = S3Path("s3://b/data/", client=client)
        children = list(p.iterdir())
        assert sorted(c.key for c in children) == [
            "data/a.parquet", "data/b.parquet", "data/sub/",
        ]
        # Non-recursive listing uses Delimiter='/'.
        client.get_paginator.return_value.paginate.assert_called_once()
        kwargs = client.get_paginator.return_value.paginate.call_args.kwargs
        assert kwargs["Delimiter"] == "/"
        assert kwargs["Prefix"] == "data/"

    def test_recursive_drops_delimiter(self, client) -> None:
        pages = [{"Contents": [{"Key": "data/x.parquet"}]}]
        client.get_paginator.return_value = _make_paginator(pages)
        p = S3Path("s3://b/data/", client=client)
        list(p.ls(recursive=True))
        kwargs = client.get_paginator.return_value.paginate.call_args.kwargs
        assert "Delimiter" not in kwargs

    def test_iterdir_filters_zero_byte_placeholders(self, client) -> None:
        pages = [{
            "Contents": [
                {"Key": "data/", "Size": 0},
                {"Key": "data/real.parquet", "Size": 5},
            ],
        }]
        client.get_paginator.return_value = _make_paginator(pages)
        p = S3Path("s3://b/data/", client=client)
        names = [c.key for c in p.iterdir()]
        # The "data/" placeholder is suppressed; only the real file
        # comes through.
        assert names == ["data/real.parquet"]


# ---------------------------------------------------------------------------
# Mkdir / remove
# ---------------------------------------------------------------------------


class TestRemove:

    def test_unlink_calls_delete_object(self, client) -> None:
        client.head_object.return_value = {"ContentLength": 1, "LastModified": None}
        p = S3Path("s3://b/key", client=client)
        p.unlink()
        client.delete_object.assert_called_once_with(Bucket="b", Key="key")

    def test_remove_dir_paginates_and_batches(self, client) -> None:
        pages = [{"Contents": [{"Key": f"d/x{i}"} for i in range(3)]}]
        client.get_paginator.return_value = _make_paginator(pages)
        client.delete_objects.return_value = {}
        p = S3Path("s3://b/d/", client=client)
        # Force the directory branch.
        client.head_object.side_effect = _client_error()
        client.list_objects_v2.return_value = {"KeyCount": 3}
        p.remove(recursive=True)
        kwargs = client.delete_objects.call_args.kwargs
        assert kwargs["Bucket"] == "b"
        assert [o["Key"] for o in kwargs["Delete"]["Objects"]] == [
            "d/x0", "d/x1", "d/x2",
        ]


class TestMkdir:

    def test_is_no_op(self, client) -> None:
        p = S3Path("s3://b/folder", client=client)
        p.mkdir()
        # No S3 call — directory creation is implicit on first child.
        client.put_object.assert_not_called()


# ---------------------------------------------------------------------------
# Path API delegation
# ---------------------------------------------------------------------------


class TestPathApi:

    def test_joinpath_inherits_client(self, client) -> None:
        p = S3Path("s3://b/folder/", client=client)
        child = p / "leaf.parquet"
        assert isinstance(child, S3Path)
        assert child.bucket == "b"
        assert child.key == "folder/leaf.parquet"
        # Same client object propagates.
        assert child._client is client

    def test_with_suffix(self, client) -> None:
        p = S3Path("s3://b/data/x.csv", client=client)
        renamed = p.with_suffix(".parquet")
        assert renamed.key == "data/x.parquet"


# ---------------------------------------------------------------------------
# Stat caching (via RemotePath)
# ---------------------------------------------------------------------------


class TestStatCaching:

    def test_repeated_stat_hits_cache(self, client) -> None:
        client.head_object.return_value = {"ContentLength": 5, "LastModified": None}
        p = S3Path("s3://b/k", client=client)
        # First stat → real call.
        first = p.stat()
        second = p.stat()
        assert first is second
        assert client.head_object.call_count == 1

    def test_write_seeds_post_write_size(self, client) -> None:
        # Initial probe → 5 bytes.
        client.head_object.return_value = {"ContentLength": 5, "LastModified": None}
        client.list_objects_v2.return_value = {"KeyCount": 0}
        client.get_object.return_value = {"Body": _Body(b"abcde")}
        p = S3Path("s3://b/k", client=client)
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

    def test_transient_eventually_succeeds(self, client, sleeps) -> None:
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
        p = S3Path("s3://b/k", client=client, retry_sleep=spy)
        assert p.size == 7
        # Two transient retries → two flat 1 s sleeps.
        assert recorded == [1.0, 1.0]
        assert client.head_object.call_count == 3

    def test_transient_gives_up_after_4_retries(self, client, sleeps) -> None:
        recorded, spy = sleeps
        client.head_object.side_effect = _client_error("InternalError", 500)
        p = S3Path("s3://b/k", client=client, retry_sleep=spy)
        with pytest.raises(Exception):
            p.stat()
        # 4 retries fired (flat 1 s each) before giving up; 5 head calls total.
        assert recorded == [1.0, 1.0, 1.0, 1.0]
        assert client.head_object.call_count == 5

    def test_permission_fails_fast(self, client, sleeps) -> None:
        recorded, spy = sleeps
        client.head_object.side_effect = _client_error("AccessDenied", 403)
        p = S3Path("s3://b/k", client=client, retry_sleep=spy)
        with pytest.raises(Exception):
            p.stat()
        # Permission errors are deterministic; no retry, no sleep.
        assert recorded == []
        assert client.head_object.call_count == 1

    def test_not_found_does_not_retry(self, client, sleeps) -> None:
        recorded, spy = sleeps
        client.head_object.side_effect = _client_error("NoSuchKey", 404)
        client.list_objects_v2.return_value = {"KeyCount": 0}
        p = S3Path("s3://b/k", client=client, retry_sleep=spy)
        s = p._stat_uncached()
        assert s.kind is IOKind.MISSING
        # No sleeps — NotFound is deterministic.
        assert recorded == []

    def test_deterministic_errors_propagate(self, client, sleeps) -> None:
        recorded, spy = sleeps
        client.put_object.side_effect = ValueError("not retryable")
        p = S3Path("s3://b/k", client=client, retry_sleep=spy)
        # Force an OVERWRITE of an empty object → put_object once.
        client.head_object.side_effect = _client_error()
        client.list_objects_v2.return_value = {"KeyCount": 0}
        with pytest.raises(ValueError, match="not retryable"):
            p.write_bytes(b"x")
        assert recorded == []


# ---------------------------------------------------------------------------
# PyArrow filesystem fast path
# ---------------------------------------------------------------------------


class TestArrowFilesystem:

    def test_arrow_uri_renders_bucket_key(self, client) -> None:
        p = S3Path("s3://my-bucket/path/to/file.parquet", client=client)
        assert p.arrow_uri == "my-bucket/path/to/file.parquet"

    def test_arrow_filesystem_returns_pyarrow_s3fs(self) -> None:
        import pyarrow.fs as pafs
        from yggdrasil.aws.client import AWSClient
        from yggdrasil.aws.config import AWSConfig
        from yggdrasil.aws.fs.service import S3Service

        # Build a real AWSClient with static creds and bind an
        # S3Path to its boto client. ``arrow_filesystem`` should
        # snapshot the boto session and hand back pyarrow's S3FS.
        cfg = AWSConfig(
            access_key_id="AKIA",
            secret_access_key="secret",
            region="us-east-1",
        )
        aws = AWSClient(config=cfg)
        # Build through S3Service.path so the boto client wiring is
        # the same as production. Pass-through ``service`` lives on
        # ``_client`` thanks to ``S3Service.path``'s constructor.
        service = S3Service(client=aws)
        p = service.path("s3://b/k")
        fs = p.arrow_filesystem()
        assert isinstance(fs, pafs.S3FileSystem)

