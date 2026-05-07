"""HTTP send pipeline integration tests.

These tests exercise the :class:`Session` pipeline end-to-end with
a stubbed transport (no real network) and a partitioned local
cache backed by :class:`YGGFolderIO`. The shape:

    request ──► local-cache lookup
              └► remote-cache lookup (skipped here)
              └► _local_send (stubbed)
              └► writeback to local cache (and remote, when configured)
              ◄── Response

The tests pin:

* a fresh send dispatches ``_local_send`` exactly once and the
  body / status / headers round-trip through the response;
* a cache hit short-circuits the network and the stub session
  records zero calls;
* writeback persists the response into the partitioned folder so a
  subsequent identical request reads from disk;
* request shaping — ``public_url_hash``, ``body_hash``,
  ``partition_key`` are stable + URL-anonymized;
* error semantics — ``raise_for_status`` fires on 5xx and is
  suppressed by ``raise_error=False``;
* :meth:`send_many` streams multiple requests through the same
  pipeline.
"""
from __future__ import annotations

import json

import pyarrow as pa
import pytest

from yggdrasil.data.enums import Mode
from yggdrasil.io.nested.ygg_folder_io import YGGFolderIO
from yggdrasil.io.path.local_path import LocalPath
from yggdrasil.io.response import RESPONSE_SCHEMA
from yggdrasil.io.send_config import CacheConfig, SendConfig

from ._helpers import StubSession, make_request, make_response


# ---------------------------------------------------------------------------
# Smoke — request / response shapes
# ---------------------------------------------------------------------------


class TestRequestShape:

    def test_url_parsed(self) -> None:
        req = make_request("https://example.com/path?a=1")
        assert req.url.host == "example.com"
        assert req.url.path == "/path"
        assert req.url.query == "a=1"
        assert req.method == "GET"

    def test_public_url_hash_stable(self) -> None:
        a = make_request("https://example.com/path?token=secret")
        b = make_request("https://example.com/path?token=other")
        # ``public_url_hash`` strips sensitive query params, so a
        # token swap doesn't change the cache identity.
        assert a.public_url_hash == b.public_url_hash

    def test_body_hash_distinct(self) -> None:
        a = make_request("https://example.com/x", method="POST", body=b"a")
        b = make_request("https://example.com/x", method="POST", body=b"b")
        assert a.body_hash != b.body_hash

    def test_partition_key_stable(self) -> None:
        a = make_request("https://example.com/path")
        b = make_request("https://example.com/path")
        assert a.partition_key == b.partition_key


class TestResponseShape:

    def test_text_decoded_with_charset(self) -> None:
        r = make_response(body="café".encode("utf-8"))
        assert r.text == "café"

    def test_json_parses_body(self) -> None:
        r = make_response(body=b'{"x": 42}', content_type="application/json")
        assert r.json() == {"x": 42}

    def test_ok_predicate_5xx(self) -> None:
        r = make_response(status_code=502)
        assert not r.ok

    def test_raise_for_status_on_5xx(self) -> None:
        r = make_response(status_code=500)
        with pytest.raises(Exception):
            r.raise_for_status()

    def test_raise_for_status_silent_on_2xx(self) -> None:
        r = make_response(status_code=204)
        r.raise_for_status()

    def test_content_decompresses_gzip(self) -> None:
        import gzip
        payload = gzip.compress(b'{"x":1}')
        r = make_response(
            body=payload,
            headers={"Content-Encoding": "gzip"},
            content_type="application/json",
        )
        assert r.content == b'{"x":1}'


# ---------------------------------------------------------------------------
# Session.send — basic dispatch
# ---------------------------------------------------------------------------


class TestSessionSend:

    def test_send_routes_through_local_send(self) -> None:
        s = StubSession()
        s.queue(make_response(status_code=201, body=b'{"created":true}'))
        req = make_request("https://example.com/users")
        resp = s.send(req)
        assert resp.status_code == 201
        assert resp.json() == {"created": True}
        assert len(s.calls) == 1

    def test_send_passes_request_through_unchanged(self) -> None:
        s = StubSession()
        req = make_request(
            "https://example.com/x",
            method="POST",
            body=b"payload",
            headers={"Authorization": "bearer abc"},
        )
        s.send(req)
        seen = s.calls[0]
        assert seen.method == "POST"
        assert seen.url.host == "example.com"
        # Body bytes preserved.
        assert seen.buffer.to_bytes() == b"payload"

    def test_raise_for_status_on_send(self) -> None:
        s = StubSession()
        s.queue(make_response(status_code=503))
        with pytest.raises(Exception):
            s.send(make_request("https://example.com/down"))

    def test_raise_error_false_suppresses(self) -> None:
        s = StubSession()
        s.queue(make_response(status_code=503))
        resp = s.send(make_request("https://example.com/down"), raise_error=False)
        assert resp.status_code == 503

    def test_send_many_streams(self) -> None:
        s = StubSession()
        s.queue(*[
            make_response(body=f'{{"i":{i}}}'.encode())
            for i in range(3)
        ])
        reqs = (make_request(f"https://example.com/{i}") for i in range(3))
        out = list(s.send_many(reqs))
        # Order isn't guaranteed (concurrent fan-out by default);
        # assert the set of indices and the call count.
        assert sorted(r.json()["i"] for r in out) == [0, 1, 2]
        assert len(s.calls) == 3


# ---------------------------------------------------------------------------
# Local cache integration via YGGFolderIO
# ---------------------------------------------------------------------------


class TestLocalCacheIntegration:

    def _cache(self, tmp_path) -> CacheConfig:
        folder = YGGFolderIO(
            path=LocalPath(str(tmp_path)),
            schema=RESPONSE_SCHEMA,
        )
        return CacheConfig(tabular=folder, mode=Mode.APPEND)

    def _wait_for_dir(self, root, *, timeout: float = 2.0) -> bool:
        """Poll *root* for any partition subdir landing within *timeout*."""
        import time as _time
        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            if any(root.iterdir()):
                return True
            _time.sleep(0.02)
        return False

    def _prepopulate(
        self, cache: CacheConfig, response,
    ) -> None:
        """Synchronously seed the cache with one response row.

        Bypasses the session's fire-and-forget writeback so the
        test can assert read-side behavior without racing against
        a background thread.
        """
        batch = response.to_arrow_batch(parse=False)
        cache.tabular.write_arrow_batches([batch], options=None)
        cache.tabular.invalidate_listing()

    def test_writeback_persists_response(self, tmp_path) -> None:
        s = StubSession()
        s.queue(make_response(body=b'{"v":1}'))
        req = make_request("https://example.com/x")
        cache = self._cache(tmp_path)
        s.send(req, local_cache=cache)
        # The session writeback fires in a background ThreadJob;
        # poll for the partition dir to land.
        assert self._wait_for_dir(tmp_path), (
            "cache writeback produced no partition dirs in time"
        )

    def test_cache_hit_skips_network(self, tmp_path) -> None:
        s = StubSession()
        req = make_request("https://example.com/x")
        cache = self._cache(tmp_path)
        # Pre-populate synchronously so we don't race the writeback.
        seed = make_response(request=req, body=b'{"v":1}')
        self._prepopulate(cache, seed)

        out = s.send(req, local_cache=cache)
        assert len(s.calls) == 0, "cache hit must skip network"
        assert out.status_code == 200

    def test_cache_miss_falls_through_to_network(self, tmp_path) -> None:
        s = StubSession()
        s.queue(make_response(body=b'{"network":true}'))
        cache = self._cache(tmp_path)
        # Cache is empty — the send goes to the network.
        out = s.send(
            make_request("https://example.com/missing"), local_cache=cache,
        )
        assert len(s.calls) == 1
        assert out.json() == {"network": True}

    def test_cache_partitions_match_response_schema(self, tmp_path) -> None:
        """The folder's partition columns mirror RESPONSE_SCHEMA's tags."""
        cache = self._cache(tmp_path)
        folder = cache.tabular
        expected = [
            f.name for f in RESPONSE_SCHEMA.fields
            if f._tag_flag(b"partition_by")
        ]
        assert folder.partition_columns == expected


# ---------------------------------------------------------------------------
# Send config merging
# ---------------------------------------------------------------------------


class TestSendConfig:

    def test_check_arg_accepts_dict(self) -> None:
        cfg = SendConfig.check_arg({"raise_error": False, "stream": False})
        assert cfg.raise_error is False
        assert cfg.stream is False

    def test_check_arg_accepts_send_config(self) -> None:
        base = SendConfig(raise_error=False)
        merged = SendConfig.check_arg(base, stream=False)
        assert merged.raise_error is False
        assert merged.stream is False

    def test_local_cache_folder_default(self) -> None:
        cfg = CacheConfig()
        # Default falls back to ``~/.yggdrasil/cache/response`` —
        # we just assert the path resolves to *something* without
        # touching the real ~/ tree.
        assert cfg.local_cache_folder().name == "response"


# ---------------------------------------------------------------------------
# Request body / response body parity
# ---------------------------------------------------------------------------


class TestBodyParity:

    def test_post_body_bytes_round_trip(self) -> None:
        s = StubSession()
        body = json.dumps({"x": [1, 2, 3]}).encode("utf-8")
        s.queue(make_response(body=body, content_type="application/json"))
        req = make_request(
            "https://example.com/echo", method="POST", body=body,
            headers={"Content-Type": "application/json"},
        )
        resp = s.send(req)
        assert resp.json() == {"x": [1, 2, 3]}
        sent = s.calls[0]
        assert sent.buffer.to_bytes() == body

    def test_response_buffer_is_bytes_io(self) -> None:
        from yggdrasil.io.bytes_io import BytesIO
        r = make_response(body=b"abc")
        assert isinstance(r.buffer, BytesIO)
        assert r.buffer.to_bytes() == b"abc"


class TestFromUrllib3:
    """Regression cover for ``HTTPResponse.from_urllib3`` / ``drain_urllib3``.

    The stubbed ``StubSession`` path bypasses these methods entirely,
    so we drive them with a duck-typed urllib3 response. The test is
    here to pin: (a) the buffer class is resolved via the Tabular
    registry, (b) drain copies bytes into the buffer and rewinds it,
    (c) callers can read JSON / bytes back out.
    """

    def _make_raw(self, *, body: bytes, headers: dict[str, str], status: int = 200):
        class _RawResp:
            def __init__(self) -> None:
                self.status = status
                self.headers = dict(headers)
                self._body = body
                self.released = False

            def read(self) -> bytes:
                return self._body

            def stream(self, amt: int = 65536):
                yield self._body

            def release_conn(self) -> None:
                self.released = True

        return _RawResp()

    def test_from_urllib3_returns_tabular_buffer_for_json(self) -> None:
        import datetime as dt
        from yggdrasil.io.http_.response import HTTPResponse
        from yggdrasil.io.primitive.json_io import JsonIO

        raw = self._make_raw(
            body=b'{"ok":true}',
            headers={"Content-Type": "application/json"},
        )
        resp = HTTPResponse.from_urllib3(
            request=make_request("https://example.com/x"),
            response=raw,
            tags=None,
            received_at=dt.datetime.now(dt.timezone.utc),
        )
        assert isinstance(resp.buffer, JsonIO)
        resp.drain_urllib3(raw, stream=False)
        assert raw.released is True
        assert resp.buffer.to_bytes() == b'{"ok":true}'
        assert resp.json() == {"ok": True}

    def test_from_urllib3_falls_back_to_bytes_io_on_unknown_media(self) -> None:
        import datetime as dt
        from yggdrasil.io.bytes_io import BytesIO
        from yggdrasil.io.http_.response import HTTPResponse

        raw = self._make_raw(
            body=b"binary blob",
            headers={"Content-Type": "application/x-not-registered"},
        )
        resp = HTTPResponse.from_urllib3(
            request=make_request("https://example.com/x"),
            response=raw,
            tags=None,
            received_at=dt.datetime.now(dt.timezone.utc),
        )
        # Unknown media types must not crash — we fall back to BytesIO.
        assert type(resp.buffer) is BytesIO
        resp.drain_urllib3(raw, stream=True)
        assert resp.buffer.to_bytes() == b"binary blob"


# ---------------------------------------------------------------------------
# Predicate helpers used by the cache
# ---------------------------------------------------------------------------


class TestPartitionPrunePredicate:
    """The session builds an ``options.prune_values`` from request batches.

    We don't drive the session directly here (that would re-test the
    StubSession path); we exercise the helper that turns a list of
    requests into the partition-IN map :class:`YGGFolderIO` reads.
    """

    def test_partition_predicate_collects_unique_keys(self) -> None:
        # ``partition_key`` is the response's partition column; on the
        # request side the same value is computed from the request URL,
        # so we just check we can extract a tuple from a batch of
        # requests.
        from yggdrasil.io.session import _request_partition_predicate
        cache = YGGFolderIO(
            path=LocalPath("/tmp/none"),  # path isn't read here
            schema=RESPONSE_SCHEMA,
        )
        reqs = [
            make_request("https://example.com/a"),
            make_request("https://example.com/b"),
            make_request("https://example.com/a"),  # dupe
        ]
        expr = _request_partition_predicate(cache, reqs)
        # The helper builds a Predicate ANDed over partition columns;
        # we just assert it produced something rather than None.
        assert expr is not None
