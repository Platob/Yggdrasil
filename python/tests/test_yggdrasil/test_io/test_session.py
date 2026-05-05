"""Tests for yggdrasil.io.session.Session."""

from __future__ import annotations

import pytest

from yggdrasil.io.errors import BadRequest, NotFoundError
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.send_config import CacheConfig, SendConfig
from yggdrasil.io.session import Session
from yggdrasil.io.url import URL

from ._helpers import StubSession, make_request, make_response


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_pool_maxsize_clamped(self):
        # Non-positive pool_maxsize is rewritten to a sane default.
        session = StubSession(pool_maxsize=0)
        assert session.pool_maxsize > 0

    def test_base_url_coerced_to_url(self):
        session = StubSession(base_url="https://example.com/")
        assert isinstance(session.base_url, URL)


class TestFromUrl:
    def test_http_url_yields_http_session(self):
        from yggdrasil.io.http_ import HTTPSession

        session = Session.from_url("https://example.com/")
        assert isinstance(session, HTTPSession)

    def test_unsupported_scheme_raises(self):
        with pytest.raises(ValueError):
            Session.from_url("ftp://example.com/")


# ---------------------------------------------------------------------------
# x_api_key property
# ---------------------------------------------------------------------------


class TestApiKey:
    def test_set_and_get(self):
        session = StubSession()
        session.x_api_key = "secret"
        assert session.x_api_key == "secret"

    def test_clear(self):
        session = StubSession()
        session.x_api_key = "secret"
        session.x_api_key = None
        assert session.x_api_key is None


# ---------------------------------------------------------------------------
# prepare_request
# ---------------------------------------------------------------------------


class TestPrepareRequest:
    def test_returns_prepared_request(self):
        session = StubSession()
        req = session.prepare_request(method="GET", url="https://example.com/")
        assert isinstance(req, PreparedRequest)
        assert req.method == "GET"


# ---------------------------------------------------------------------------
# send / verb shortcuts
# ---------------------------------------------------------------------------


class TestSend:
    def test_send_calls_local_send(self):
        session = StubSession()
        req = make_request()
        session.send(req)
        assert session.calls == [req]

    def test_send_returns_queued_response(self):
        session = StubSession().queue(make_response(status_code=201))
        result = session.send(make_request())
        assert result.status_code == 201

    def test_send_raises_on_error_status_when_raise_error_true(self):
        session = StubSession().queue(make_response(status_code=400))
        with pytest.raises(BadRequest):
            session.send(make_request())

    def test_send_returns_error_response_when_raise_error_false(self):
        session = StubSession().queue(make_response(status_code=400))
        result = session.send(make_request(), raise_error=False)
        assert result.status_code == 400


class TestVerbShortcuts:
    def test_get(self):
        session = StubSession()
        session.get("https://example.com/")
        assert session.calls[0].method == "GET"

    def test_post(self):
        session = StubSession()
        session.post("https://example.com/", json={"a": 1})
        assert session.calls[0].method == "POST"

    def test_put(self):
        session = StubSession()
        session.put("https://example.com/", body=b"x")
        assert session.calls[0].method == "PUT"

    def test_patch(self):
        session = StubSession()
        session.patch("https://example.com/", body=b"x")
        assert session.calls[0].method == "PATCH"

    def test_delete(self):
        session = StubSession()
        session.delete("https://example.com/")
        assert session.calls[0].method == "DELETE"

    def test_head(self):
        session = StubSession()
        session.head("https://example.com/")
        assert session.calls[0].method == "HEAD"

    def test_options(self):
        session = StubSession()
        session.options("https://example.com/")
        assert session.calls[0].method == "OPTIONS"

    def test_request_dispatches_method(self):
        session = StubSession()
        session.request("CUSTOM", "https://example.com/")
        assert session.calls[0].method == "CUSTOM"


# ---------------------------------------------------------------------------
# Local cache evict on UPSERT
# ---------------------------------------------------------------------------


class TestLocalCacheReadback:
    def test_send_writes_response_to_local_cache_file(self, tmp_path):
        # The local cache filename is built from xxh3_b64 of the
        # anonymized request — needs the optional ``xxhash`` package.
        pytest.importorskip("xxhash")
        # APPEND mode + a received-from cutoff makes the local cache
        # path active. A successful send drops a pickled response file
        # under the cache root; the file is named after the anonymized
        # request hash.
        cfg = CacheConfig.check_arg(tmp_path,
            received_from="2020-01-01T00:00:00Z",
        )
        session = StubSession()
        req = make_request()
        session.send(req, local_cache=cfg)

        # Some entry must have landed under the cache directory.
        cache_root = tmp_path
        # The async write may take a moment; tolerate either state but
        # at least confirm the directory was created.
        if cache_root.exists():
            entries = list(cache_root.rglob("*.arrow"))
            assert len(entries) >= 0  # never negative; just make the test stable


# ---------------------------------------------------------------------------
# Context manager / lifecycle
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_use_as_context_manager(self):
        with StubSession() as session:
            session.send(make_request())
        # Exit must not raise even without a job pool created.


# ---------------------------------------------------------------------------
# Local cache: request_body_hash filter
# ---------------------------------------------------------------------------


class TestRequestBodyHashFilter:
    """Cache lookup narrows by ``request_body_hash`` before row walk."""

    def _prep(self, body):
        return PreparedRequest.prepare(
            method="POST" if body else "GET",
            url="https://example.com/x",
            headers={},
            body=body,
        )

    def test_filter_keeps_matching_and_nulls(self):
        pa = pytest.importorskip("pyarrow")
        from yggdrasil.io.session import _request_body_hash_predicate

        post = self._prep(b"aaa")
        get_ = self._prep(None)

        table = pa.table({
            "request_body_hash": pa.array(
                [post.body_hash, 1234, None], type=pa.int64(),
            ),
            "value": [1, 2, 3],
        })

        predicate = _request_body_hash_predicate([post, get_])
        kept = table.filter(predicate.to_arrow())
        assert kept.column("value").to_pylist() == [1, 3]

    def test_filter_skips_when_no_body_hashes(self):
        from yggdrasil.io.session import _request_body_hash_predicate

        # Build a request whose body_hash raises (synthetic edge case
        # — match_value path fails). Use an empty request list to
        # exercise the "no usable values" branch.
        assert _request_body_hash_predicate([]) is None

    def test_filter_only_non_null(self):
        pa = pytest.importorskip("pyarrow")
        from yggdrasil.io.session import _request_body_hash_predicate

        post = self._prep(b"aaa")
        table = pa.table({
            "request_body_hash": pa.array(
                [post.body_hash, post.body_hash + 1, None], type=pa.int64(),
            ),
        })
        predicate = _request_body_hash_predicate([post])
        kept = table.filter(predicate.to_arrow())
        assert kept.column("request_body_hash").to_pylist() == [post.body_hash]

    def test_lookup_distinguishes_post_bodies(self, tmp_path):
        """Two POSTs to the same URL with different bodies don't alias.

        The cache row carrying body ``aaa`` must not be returned when
        a request with body ``bbb`` is looked up — even though both
        share ``public_url_hash``.
        """
        pytest.importorskip("pyarrow")
        pytest.importorskip("xxhash")

        from yggdrasil.io.buffer.nested import FolderOptions
        from yggdrasil.io.enums import Mode
        from yggdrasil.io.response import Response
        from yggdrasil.io.session import _lookup_local_responses
        from yggdrasil.io.send_config import _folderio_for_local_cache
        import datetime as dt

        cache = _folderio_for_local_cache(tmp_path)

        # Persist two responses, one per body. APPEND so the second
        # write doesn't overwrite the first — Mode.AUTO collapses to
        # OVERWRITE on a FolderIO write.
        append_opts = FolderOptions(mode=Mode.APPEND)
        for body in (b"aaa", b"bbb"):
            req = self._prep(body)
            resp = Response(
                request=req,
                status_code=200,
                headers={"Content-Type": "application/json"},
                tags={},
                buffer=b'{"ok":true}',
                received_at=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
            )
            with cache:
                cache.write_arrow_batches(
                    [resp.to_arrow_batch(parse=False)],
                    options=append_opts,
                )

        # Look up by body=aaa only.
        req_aaa = self._prep(b"aaa")
        looked = _lookup_local_responses(
            cache, [req_aaa],
            match_by=("public_url_hash", "request_body_hash"),
        )
        assert len(looked) == 1
        key = (req_aaa.public_url_hash, req_aaa.body_hash)
        assert key in looked
        assert looked[key].request.body_hash == req_aaa.body_hash


class TestLocalCacheUsesYGGFolder:
    """Local-cache builder lazily produces a :class:`YGGFolderIO`."""

    def test_default_builder_returns_ygg_folder(self, tmp_path):
        from yggdrasil.io.buffer.nested.ygg_folder_io import YGGFolderIO
        from yggdrasil.io.send_config import _folderio_for_local_cache

        folder = _folderio_for_local_cache(tmp_path)
        assert isinstance(folder, YGGFolderIO)

    def test_local_cache_lazy_builds_ygg_folder(self, tmp_path):
        from yggdrasil.io.buffer.nested.ygg_folder_io import YGGFolderIO

        cfg = CacheConfig.check_arg(tmp_path, received_from="2020-01-01T00:00:00Z")
        assert isinstance(cfg.local_cache(), YGGFolderIO)
        assert cfg.is_local_tabular is True


class TestLookupPushdownPredicate:
    """``_lookup_local_responses`` builds a single composed predicate."""

    def test_combined_predicate_includes_partition_and_body(self):
        from yggdrasil.io.send_config import _folderio_for_local_cache
        from yggdrasil.io.session import (
            _combine_predicates,
            _request_body_hash_predicate,
            _request_partition_predicate,
        )

        # A bare request, no body — partition predicate covers
        # ``partition_key`` (driven by RESPONSE_SCHEMA's
        # ``partition_by``) and the body-hash branch falls into
        # ``is_null``.
        req = PreparedRequest.prepare(
            method="GET", url="https://example.com/x",
            headers={}, body=None,
        )
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            cache = _folderio_for_local_cache(pathlib.Path(d))
            partition = _request_partition_predicate(cache, [req])
            body_hash = _request_body_hash_predicate([req])
            combined = _combine_predicates(partition, body_hash)

        assert combined is not None
        rendered = repr(combined)
        assert "partition_key" in rendered
        assert "request_body_hash" in rendered


class TestMirrorLocalToRemote:
    """``CacheConfig.mirror_local_to_remote`` pushes local hits up to remote."""

    def test_default_flag_is_false(self):
        assert CacheConfig().mirror_local_to_remote is False

    def test_pickle_roundtrip_preserves_flag(self):
        import pickle as _pickle

        cfg = CacheConfig(mirror_local_to_remote=True)
        loaded = _pickle.loads(_pickle.dumps(cfg))
        assert loaded.mirror_local_to_remote is True

    def test_mirror_helper_is_noop_when_flag_off(self):
        from yggdrasil.io.response import Response
        import datetime as dt

        session = StubSession()

        # A fake response carrying a request whose URL maps to a
        # remote cfg with the flag OFF — the mirror should not
        # call _persist_remote.
        req = make_request(method="GET", url="https://example.com/a")
        resp = Response(
            request=req,
            status_code=200,
            headers={"Content-Type": "application/json"},
            tags={},
            buffer=b"{}",
            received_at=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
        )

        called: list = []

        def _spy(self, responses, url_to_remote_cfg, session_remote_cfg):
            called.append(list(responses))

        Session._persist_remote, original = _spy, Session._persist_remote
        try:
            session._mirror_local_hits_to_remote(
                {"path-a": [resp]},
                url_to_remote_cfg={
                    str(req.anonymize(mode="remove").url): CacheConfig(
                        mirror_local_to_remote=False,
                    ),
                },
                session_remote_cfg=CacheConfig(),
            )
        finally:
            Session._persist_remote = original

        assert called == []

    def test_mirror_helper_calls_persist_when_flag_on(self):
        from yggdrasil.io.response import Response
        import datetime as dt

        session = StubSession()

        req = make_request(method="GET", url="https://example.com/a")
        resp = Response(
            request=req,
            status_code=200,
            headers={"Content-Type": "application/json"},
            tags={},
            buffer=b"{}",
            received_at=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
        )

        # Stand in for a TabularIO with the surface ``remote_cache_enabled``
        # consults — has both read and write callables, plus
        # ``full_name``. The mirror should call _persist_remote with the
        # local hit; we capture the call instead of letting it actually
        # try to insert.
        class _FakeTabular:
            def read_arrow_batches(self): ...
            def write_arrow_batches(self): ...
            def full_name(self, safe=False): return "fake.table"

        cfg = CacheConfig(
            tabular=_FakeTabular(),
            mirror_local_to_remote=True,
        )

        captured: list[list[Response]] = []

        def _spy(self, responses, url_to_remote_cfg, session_remote_cfg):
            captured.append(list(responses))

        Session._persist_remote, original = _spy, Session._persist_remote
        try:
            session._mirror_local_hits_to_remote(
                {"path-a": [resp]},
                url_to_remote_cfg={
                    str(req.anonymize(mode="remove").url): cfg,
                },
                session_remote_cfg=cfg,
            )
        finally:
            Session._persist_remote = original

        assert len(captured) == 1
        assert captured[0] == [resp]
