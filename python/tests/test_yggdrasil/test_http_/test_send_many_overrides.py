"""Tests for send_many / send_many_batches Ellipsis-default config overrides.

Verifies that SendConfig kwargs on :meth:`HTTPSession.send_many` and
:meth:`HTTPSession.send_many_batches` default to ``...`` and only
touch per-request configs when the caller passes an explicit value.
"""
from __future__ import annotations

import pytest

from yggdrasil.http_.send_config import CacheConfig, SendConfig
<<<<<<< HEAD
from yggdrasil.http_.io_session import Session
=======
from yggdrasil.http_.session import Session
>>>>>>> 7d53e95

from ._helpers import StubSession, make_request, make_response


@pytest.fixture(autouse=True)
def _clear_session_cache():
    Session._INSTANCES.clear()
    yield
    Session._INSTANCES.clear()


# ------------------------------------------------------------------ #
# send_many — config stamping
# ------------------------------------------------------------------ #


class TestSendManyOverrides:
    """Ellipsis-defaulted kwargs on send_many."""

    def test_no_overrides_preserves_request_config(self):
        """Calling send_many() with all defaults leaves per-request
        configs untouched."""
        cfg = SendConfig(raise_error=False, cache_only=True)
        req = make_request("https://example.com/a")
        req.send_config = cfg

        s = StubSession()
        s.queue(make_response(request=req))
        list(s.send_many(iter([req])))

        assert req.send_config is cfg

    def test_raise_error_override_applies_to_all(self):
        """An explicit ``raise_error=False`` overrides every request."""
        req_a = make_request("https://example.com/a")
        req_a.send_config = SendConfig(raise_error=True)
        req_b = make_request("https://example.com/b")

        s = StubSession()
        s.queue(
            make_response(request=req_a),
            make_response(request=req_b),
        )
        list(s.send_many(iter([req_a, req_b]), raise_error=False))

        assert req_a.send_config.raise_error is False
        assert req_b.send_config.raise_error is False

    def test_override_preserves_per_request_local_cache(self):
        """Per-request local_cache survives a call-level override of
        other fields — mirrors the merge-back in send()."""
        per_req_cache = CacheConfig(tabular="/tmp/per-request")
        req = make_request("https://example.com/x")
        req.send_config = SendConfig(
            raise_error=True,
            local_cache=per_req_cache,
        )

        s = StubSession()
        s.queue(make_response(request=req))
        list(s.send_many(iter([req]), raise_error=False))

        assert req.send_config.raise_error is False
        assert req.send_config.local_cache is per_req_cache

    def test_override_preserves_per_request_remote_cache(self):
        """Per-request remote_cache survives a call-level override."""
        per_req_cache = CacheConfig(tabular="/tmp/remote")
        req = make_request("https://example.com/x")
        req.send_config = SendConfig(
            raise_error=True,
            remote_cache=per_req_cache,
        )

        s = StubSession()
        s.queue(make_response(request=req))
        list(s.send_many(iter([req]), raise_error=False))

        assert req.send_config.raise_error is False
        assert req.send_config.remote_cache is per_req_cache

    def test_config_param_used_for_unconfigured_requests(self):
        """The ``config`` positional is stamped on requests that have
        no per-request send_config."""
        base = SendConfig(raise_error=False, cache_only=True)
        req = make_request("https://example.com/a")
        assert req.send_config is None

        s = StubSession()
        s.queue(make_response(request=req))
        list(s.send_many(iter([req]), base))

        assert req.send_config.raise_error is False
        assert req.send_config.cache_only is True

    def test_config_plus_override(self):
        """A ``config`` base combined with an explicit override."""
        base = SendConfig(raise_error=True, cache_only=True)
        req = make_request("https://example.com/a")

        s = StubSession()
        s.queue(make_response(request=req))
        list(s.send_many(iter([req]), base, raise_error=False))

        assert req.send_config.raise_error is False
        assert req.send_config.cache_only is True

    def test_cache_only_override(self):
        """``cache_only=True`` propagates to all requests."""
        req = make_request("https://example.com/a")
        req.send_config = SendConfig(cache_only=False)

        s = StubSession()
        s.queue(make_response(request=req))
        list(s.send_many(iter([req]), cache_only=True))

        assert req.send_config.cache_only is True

    def test_multiple_overrides_at_once(self):
        """Several explicit overrides applied together."""
        req = make_request("https://example.com/a")
        req.send_config = SendConfig(raise_error=True, cache_only=False)

        s = StubSession()
        s.queue(make_response(request=req))
        list(s.send_many(
            iter([req]),
            raise_error=False,
            cache_only=True,
        ))

        assert req.send_config.raise_error is False
        assert req.send_config.cache_only is True


# ------------------------------------------------------------------ #
# send_many_batches — config stamping
# ------------------------------------------------------------------ #


class TestSendManyBatchesOverrides:
    """Ellipsis-defaulted kwargs on send_many_batches."""

    def test_no_overrides_preserves_request_config(self):
        cfg = SendConfig(raise_error=False)
        req = make_request("https://example.com/a")
        req.send_config = cfg

        s = StubSession()
        s.queue(make_response(request=req))
        list(s.send_many_batches(iter([req])))

        assert req.send_config is cfg

    def test_raise_error_override_applies(self):
        req = make_request("https://example.com/a")
        req.send_config = SendConfig(raise_error=True)

        s = StubSession()
        s.queue(make_response(request=req))
        list(s.send_many_batches(iter([req]), raise_error=False))

        assert req.send_config.raise_error is False

    def test_override_preserves_per_request_local_cache(self):
        per_req_cache = CacheConfig(tabular="/tmp/per-request")
        req = make_request("https://example.com/x")
        req.send_config = SendConfig(
            raise_error=True,
            local_cache=per_req_cache,
        )

        s = StubSession()
        s.queue(make_response(request=req))
        list(s.send_many_batches(iter([req]), raise_error=False))

        assert req.send_config.raise_error is False
        assert req.send_config.local_cache is per_req_cache

    def test_unconfigured_request_gets_override_config(self):
        req = make_request("https://example.com/a")
        assert req.send_config is None

        s = StubSession()
        s.queue(make_response(request=req))
        list(s.send_many_batches(iter([req]), raise_error=False))

        assert req.send_config.raise_error is False
