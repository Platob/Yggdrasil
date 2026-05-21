"""HTTP send pipeline integration tests.

These tests exercise the :class:`Session` pipeline end-to-end with
a stubbed transport (no real network) and the URL-mirrored fast-path
local cache (one ``.arrow`` per request ``public_hash`` under the
cache root). The shape:

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
* writeback persists the response under the URL-mirrored tree so a
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
from pathlib import Path

import pyarrow as pa
import pytest

from yggdrasil.data.enums import Mode
from yggdrasil.io.send_config import CacheConfig, SendConfig
from yggdrasil.io.session import (
    _local_fast_path_relative,
    _store_fast_path_arrow_batch,
)

from ._helpers import StubSession, make_request, make_response


# ---------------------------------------------------------------------------
# Smoke — request / response shapes
# ---------------------------------------------------------------------------


class TestRequestShape:

    def test_multi_codec_accept_encoding_passes_through(self) -> None:
        # RFC 7231 §5.3.4 — a real client (browser, urllib3 default,
        # bespoke API client) sends Accept-Encoding as a comma-separated
        # preference list. ``Codec.from_`` only parses single codecs;
        # ``normalize_headers`` previously called it eagerly and raised
        # ``ValueError: Cannot resolve Codec from non-codec MIME type
        # 'gzip, deflate, br, zstd'`` for any such request.
        from yggdrasil.io.request import PreparedRequest
        req = PreparedRequest.prepare(
            "POST",
            "https://example.com/refresh",
            headers={"Accept-Encoding": "gzip, deflate, br, zstd"},
        )
        assert req.headers.get("Accept-Encoding") == "gzip, deflate, br, zstd"

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


class TestRequestsCompat:
    """``requests.Session``-style call shapes route through ``request()``."""

    def test_post_form_data_dict_urlencodes_body(self) -> None:
        s = StubSession()
        s.post("https://example.com/login", data={"user": "alice", "pw": "x y"})
        seen = s.calls[0]
        assert seen.method == "POST"
        assert seen.buffer.to_bytes() == b"user=alice&pw=x+y"
        assert seen.headers.get("Content-Type") == "application/x-www-form-urlencoded"

    def test_post_form_data_sequence_of_tuples(self) -> None:
        # ``requests`` accepts ``[(k, v), (k, v2)]`` for repeated keys —
        # ``urlencode(doseq=True)`` handles both shapes.
        s = StubSession()
        s.post("https://example.com/x", data=[("k", "1"), ("k", "2")])
        assert s.calls[0].buffer.to_bytes() == b"k=1&k=2"

    def test_post_raw_data_bytes_passthrough(self) -> None:
        s = StubSession()
        s.post(
            "https://example.com/raw",
            data=b"raw-bytes",
            headers={"Content-Type": "application/octet-stream"},
        )
        seen = s.calls[0]
        assert seen.buffer.to_bytes() == b"raw-bytes"
        # Caller-supplied Content-Type wins over the form default.
        assert seen.headers.get("Content-Type") == "application/octet-stream"

    def test_post_raw_data_str_encoded_utf8(self) -> None:
        s = StubSession()
        s.post("https://example.com/raw", data="héllo")
        assert s.calls[0].buffer.to_bytes() == "héllo".encode("utf-8")

    def test_data_and_body_conflict_raises(self) -> None:
        s = StubSession()
        with pytest.raises(ValueError, match="body=.*data="):
            s.post("https://example.com/x", body=b"a", data={"b": "c"})

    def test_get_with_cookies_dict(self) -> None:
        s = StubSession()
        s.get("https://example.com/x", cookies={"sid": "abc", "lang": "fr"})
        cookie = s.calls[0].headers.get("Cookie")
        assert "sid=abc" in cookie
        assert "lang=fr" in cookie

    def test_cookies_does_not_override_explicit_header(self) -> None:
        s = StubSession()
        s.get(
            "https://example.com/x",
            headers={"Cookie": "already=set"},
            cookies={"sid": "abc"},
        )
        assert s.calls[0].headers.get("Cookie") == "already=set"

    def test_get_with_timeout_alias_routes_to_wait(self) -> None:
        # ``timeout=`` is the requests spelling of our ``wait=``; both
        # resolve through ``WaitingConfig.from_`` so a numeric arg works.
        from yggdrasil.dataclasses.waiting import WaitingConfig

        s = StubSession()
        s.queue(make_response())
        captured: dict[str, WaitingConfig] = {}

        original_send = s.send

        def spy(req, **kwargs):
            captured["wait"] = WaitingConfig.from_(kwargs.get("wait"))
            return original_send(req, **kwargs)

        s.send = spy  # type: ignore[method-assign]
        s.get("https://example.com/x", timeout=7.5)
        assert captured["wait"].timeout == 7.5

    def test_timeout_and_wait_conflict_raises(self) -> None:
        s = StubSession()
        with pytest.raises(ValueError, match="wait=.*timeout="):
            s.get("https://example.com/x", wait=2.0, timeout=5.0)

    def test_get_with_params_query_string(self) -> None:
        # Sanity: existing ``params=`` already mirrors ``requests`` —
        # pinned here so the compat layer doesn't regress it.
        s = StubSession()
        s.get("https://example.com/search", params={"q": "yggdrasil", "n": "10"})
        seen = s.calls[0]
        assert "q=yggdrasil" in seen.url.query
        assert "n=10" in seen.url.query


# ---------------------------------------------------------------------------
# Local cache integration via the URL-mirrored fast-path tree
# ---------------------------------------------------------------------------


class TestLocalCacheIntegration:

    def _cache(self, tmp_path) -> CacheConfig:
        return CacheConfig(path=str(tmp_path), mode=Mode.APPEND)

    def _wait_for_dir(self, root, *, timeout: float = 2.0) -> bool:
        """Poll *root* for any subdir landing within *timeout*."""
        import time as _time
        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            if any(root.iterdir()):
                return True
            _time.sleep(0.02)
        return False

    def _wait_for_readable(self, cache, *, timeout: float = 3.0) -> bool:
        """Poll until any non-hidden ``.arrow`` file lands in the cache tree.

        Writeback fires on a daemon thread; the file may not be on
        disk immediately after :meth:`Session.send` returns.
        """
        import time as _time

        deadline = _time.monotonic() + timeout
        root = Path(cache.path)
        while _time.monotonic() < deadline:
            try:
                if root.exists() and any(
                    p.is_file() and not p.name.startswith(".")
                    for p in root.rglob("*.arrow")
                ):
                    return True
            except OSError:
                pass
            _time.sleep(0.05)
        return False

    def _prepopulate(
        self, cache: CacheConfig, response,
    ) -> None:
        """Synchronously seed the fast-path file for *response*.

        Bypasses the session's fire-and-forget writeback so the
        test can assert read-side behavior without racing against
        a background thread.
        """
        req = response.request
        rel = _local_fast_path_relative(req.method, req.url, req.public_hash)
        _store_fast_path_arrow_batch(
            cache.path, rel, response.to_arrow_batch(parse=False),
        )

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

    def test_cache_hit_distinct_url_does_not_match(self, tmp_path) -> None:
        """A cached row for /a must not satisfy a request for /b."""
        s = StubSession()
        cache = self._cache(tmp_path)
        seed_req = make_request("https://example.com/a")
        self._prepopulate(cache, make_response(request=seed_req, body=b'{"v":"a"}'))

        s.queue(make_response(body=b'{"v":"b"}'))
        out = s.send(make_request("https://example.com/b"), local_cache=cache)
        assert len(s.calls) == 1, "different URL must miss cache"
        assert out.json() == {"v": "b"}

    def test_cache_hit_5xx_still_raises_when_raise_error(self, tmp_path) -> None:
        """raise_error applies after a cache hit too."""
        s = StubSession()
        cache = self._cache(tmp_path)
        req = make_request("https://example.com/x")
        self._prepopulate(
            cache, make_response(request=req, status_code=500, body=b"boom"),
        )
        with pytest.raises(Exception):
            s.send(req, local_cache=cache, raise_error=True)
        assert len(s.calls) == 0, "cache hit must precede the network"

    def test_received_to_filters_out_stale_row(self, tmp_path) -> None:
        """A row outside [received_from, received_to) is ignored on read."""
        import datetime as dt
        s = StubSession()
        old = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
        cache = CacheConfig(
            path=str(tmp_path),
            mode=Mode.APPEND,
            received_from=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
            received_to=dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc),
        )
        req = make_request("https://example.com/x")
        self._prepopulate(
            cache,
            make_response(request=req, body=b'{"old":true}', received_at=old),
        )

        s.queue(make_response(body=b'{"fresh":true}'))
        out = s.send(req, local_cache=cache)
        assert len(s.calls) == 1, "stale row outside the window must miss"
        assert out.json() == {"fresh": True}

    def test_upsert_mode_skips_lookup_and_refetches(self, tmp_path) -> None:
        """UPSERT bypasses the read, always going to the network."""
        s = StubSession()
        cache = CacheConfig(path=str(tmp_path), mode=Mode.UPSERT)
        req = make_request("https://example.com/x")
        self._prepopulate(
            cache, make_response(request=req, body=b'{"cached":true}'),
        )

        s.queue(make_response(body=b'{"fresh":true}'))
        out = s.send(req, local_cache=cache)
        assert len(s.calls) == 1, "UPSERT must always hit the network"
        assert out.json() == {"fresh": True}

    def test_per_request_cache_override_wins(self, tmp_path) -> None:
        """A request-level cache config overrides the session-level one."""
        s = StubSession()
        s.queue(make_response(body=b'{"network":true}'))
        session_cache = self._cache(tmp_path)
        # Pre-seed the session-level cache with a row that *would* match
        # if the session config were used.
        seed_req = make_request("https://example.com/x")
        self._prepopulate(
            session_cache,
            make_response(request=seed_req, body=b'{"cached":true}'),
        )

        # Per-request config points at a fresh empty folder — that must
        # win, forcing a network fetch.
        empty_dir = tmp_path / "alt"
        empty_dir.mkdir()
        req_cache = CacheConfig(path=str(empty_dir), mode=Mode.APPEND)
        req = make_request("https://example.com/x").copy(
            local_cache_config=req_cache,
        )

        out = s.send(req, local_cache=session_cache)
        assert len(s.calls) == 1, "per-request config must override the session"
        assert out.json() == {"network": True}

    def test_writeback_round_trips_on_second_send(self, tmp_path) -> None:
        """Send → writeback → identical send reads the persisted row."""
        s = StubSession()
        cache = self._cache(tmp_path)
        req = make_request("https://example.com/x")
        # Queue a response bound to *this* request — the lookup keys
        # off the embedded request's hash, so the queued response must
        # carry it for the round-trip read to find it.
        s.queue(make_response(request=req, body=b'{"v":"first"}'))

        first = s.send(req, local_cache=cache)
        assert first.json() == {"v": "first"}
        assert self._wait_for_readable(cache), "writeback never landed on disk"

        # Second send must hit the cache rather than the network.
        second = s.send(req, local_cache=cache)
        assert len(s.calls) == 1, "second send must read from disk"
        assert second.json() == {"v": "first"}


class TestCacheConfigCoercion:
    """``CacheConfig.check_arg`` accepts a few convenience shapes."""

    def test_check_arg_path_sets_local_path(self, tmp_path) -> None:
        from yggdrasil.io.path import LocalPath, Path as YggPath

        cfg = CacheConfig.check_arg(tmp_path)
        assert cfg.is_local is True
        assert cfg.is_remote is False
        # ``path`` is coerced to a live abstract :class:`Path`; equality
        # against the string form still holds via Path.__eq__.
        assert isinstance(cfg.path, YggPath)
        assert isinstance(cfg.path, LocalPath)
        assert cfg.path == str(tmp_path)
        assert cfg.local_cache_enabled is True
        assert str(cfg.local_cache_folder()) == str(tmp_path)

    def test_check_arg_abstract_path_is_held_as_is(self, tmp_path) -> None:
        # Passing a live :class:`Path` (any backend) round-trips through
        # :meth:`check_arg` unchanged — the singleton identity is
        # preserved so a downstream caller reading ``cfg.path`` gets the
        # very same backend handle the call site built.
        from yggdrasil.io.path import LocalPath

        target = LocalPath(str(tmp_path))
        cfg = CacheConfig.check_arg(target)
        assert cfg.path is target
        assert cfg.local_cache_folder() is target

    def test_check_arg_str_path_coerces_to_localpath(self, tmp_path) -> None:
        from yggdrasil.io.path import LocalPath

        cfg = CacheConfig.check_arg(str(tmp_path))
        assert isinstance(cfg.path, LocalPath)
        assert cfg.path == str(tmp_path)

    def test_check_arg_timedelta_sets_window(self) -> None:
        import datetime as dt

        cfg = CacheConfig.check_arg(dt.timedelta(hours=6))
        # ``check_arg`` resolves a timedelta into a concrete window
        # (received_to defaults to now, received_from = now - delta).
        assert cfg.received_from is not None
        assert cfg.received_to is not None
        assert cfg.received_to - cfg.received_from == dt.timedelta(hours=6)

    def test_check_arg_dict_round_trip(self) -> None:
        cfg = CacheConfig.check_arg({"mode": "APPEND"})
        assert cfg.mode is Mode.APPEND

    def test_pickle_round_trip_preserves_path(self, tmp_path) -> None:
        # The dataclass holds a live :class:`Path` but :meth:`__getstate__`
        # projects to the URL string for transport — so the config still
        # crosses Spark / multiprocessing / Power Query worker boundaries
        # without dragging a bound backend handle along, and
        # :meth:`__setstate__` rehydrates via :meth:`Path.from_`.
        import pickle

        from yggdrasil.io.path import LocalPath

        cfg = CacheConfig(path=LocalPath(str(tmp_path)))
        restored = pickle.loads(pickle.dumps(cfg))
        assert isinstance(restored.path, LocalPath)
        assert restored.path == cfg.path


class TestLocalCacheFolderPerHost:
    """Default cache path splits per ``Session.base_url`` host + path.

    Different APIs sharing the same machine should not collide on
    disk. When a session has no ``base_url`` the cache falls back to
    the ``default`` bucket so behavior is well-defined.
    """

    def test_default_when_no_base_url(self) -> None:
        s = StubSession()
        cfg = CacheConfig()
        path = cfg.local_cache_folder(session=s)
        assert path.name == "default"
        assert path.parent.name == "response"

    def test_host_only_base_url(self) -> None:
        from yggdrasil.io.url import URL

        s = StubSession(base_url=URL.from_("https://api.example.com/"))
        path = CacheConfig().local_cache_folder(session=s)
        assert path.name == "api.example.com"
        assert path.parent.name == "response"

    def test_host_plus_path_base_url(self) -> None:
        from yggdrasil.io.url import URL

        s = StubSession(base_url=URL.from_("https://api.example.com/v1/markets/"))
        path = CacheConfig().local_cache_folder(session=s)
        # Trailing / on base_url must not produce an empty leaf.
        parts = tuple(path.parts)
        assert parts[-3:] == ("response", "api.example.com", "v1/markets") or (
            parts[-4:] == ("response", "api.example.com", "v1", "markets")
        )

    def test_explicit_path_overrides_default(self, tmp_path) -> None:
        from yggdrasil.io.url import URL
        s = StubSession(base_url=URL.from_("https://api.example.com/"))
        cache = CacheConfig(path=str(tmp_path), mode=Mode.APPEND)
        # Explicit ``path`` wins — host derivation is for the
        # auto-built default only.
        assert str(cache.local_cache_folder(session=s)) == str(tmp_path)

    def test_distinct_hosts_get_distinct_folders(self) -> None:
        from yggdrasil.io.url import URL
        a = StubSession(base_url=URL.from_("https://a.example.com/"))
        b = StubSession(base_url=URL.from_("https://b.example.com/"))
        pa = CacheConfig().local_cache_folder(session=a)
        pb = CacheConfig().local_cache_folder(session=b)
        assert pa != pb
        assert pa.name == "a.example.com"
        assert pb.name == "b.example.com"


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
        # No session → no host context → falls back to the
        # ``response/default`` bucket so it doesn't collide with
        # any real per-host cache root.
        path = cfg.local_cache_folder()
        assert path.name == "default"
        assert path.parent.name == "response"


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

    def test_response_buffer_is_holder(self) -> None:
        from yggdrasil.io.holder import Holder
        r = make_response(body=b"abc")
        assert isinstance(r.buffer, Holder)
        assert r.buffer.to_bytes() == b"abc"


class TestFromUrllib3:
    """Regression cover for ``HTTPResponse.from_urllib3``.

    The stubbed ``StubSession`` path bypasses this method entirely,
    so we drive it with a duck-typed urllib3 response. The test is
    here to pin: (a) the buffer class is resolved via the Tabular
    registry, (b) the body is drained into the buffer in one pass,
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

    def test_from_urllib3_stamps_media_on_holder_for_json(self) -> None:
        import datetime as dt
        from yggdrasil.io.holder import Holder
        from yggdrasil.io.http_.response import HTTPResponse
        from yggdrasil.io.primitive.json_file import JSONFile

        raw = self._make_raw(
            body=b'{"ok":true}',
            headers={"Content-Type": "application/json"},
        )
        resp = HTTPResponse.from_urllib3(
            request=make_request("https://example.com/x"),
            response=raw,
            tags=None,
            received_at=dt.datetime.now(dt.timezone.utc),
            stream=False,
        )
        # The buffer is a Holder; opening it via response.open() routes
        # through the JSONFile leaf for the stamped media type.
        assert isinstance(resp.buffer, Holder)
        with resp.open(mode="rb") as bio:
            assert isinstance(bio, JSONFile)
        assert raw.released is True
        assert resp.buffer.to_bytes() == b'{"ok":true}'
        assert resp.json() == {"ok": True}

    def test_from_urllib3_falls_back_to_holder_on_unknown_media(self) -> None:
        import datetime as dt
        from yggdrasil.io.holder import Holder
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
            stream=True,
        )
        # Unknown media types must not crash — buffer stays a plain Holder.
        assert isinstance(resp.buffer, Holder)
        assert resp.buffer.to_bytes() == b"binary blob"


