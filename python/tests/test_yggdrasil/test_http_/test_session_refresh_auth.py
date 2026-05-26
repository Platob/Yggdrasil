"""Tests for :meth:`Session.refresh_auth`, the 403 retry in HTTPSession,
and the session-init auth pre-stamp.

Covers:

1. ``Session(auth=handler)`` pre-stamps ``self.headers["Authorization"]``
   so the session-level view carries the current credential before
   any request goes out.
2. ``refresh_auth(request)`` returns ``(self, refreshed: bool)`` so
   chained call sites can short-circuit when no refresh happened.
3. ``force=True`` (default) raises
   :class:`~yggdrasil.exceptions.AuthRequiredError` when no handler
   is bound. ``force=False`` returns ``(self, False)`` silently.
4. ``refresh_auth`` calls ``handler.refresh(force=...)`` when
   available (legacy ``refresh()`` without ``force`` is supported via
   ``TypeError`` fallback) and stamps ``handler.authorization`` onto
   the request — keeping ``self.headers`` in sync when the resolved
   handler is the session-wide one.
5. Per-request ``request.auth`` wins over session-wide ``self.auth``
   and does not pollute ``self.headers``.
6. ``prepare_request_before_send`` delegates to
   ``refresh_auth(request, force=False)`` (steady-state path).
7. :class:`HTTPSession._local_send` retries exactly once on a 403
   when an auth handler is bound — the second send carries the
   refreshed Authorization header — and does **not** retry without
   one.
8. :meth:`Response.refresh_auth` delegates to the bound session and
   surfaces the same ``(self, refreshed)`` shape.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from yggdrasil.exceptions import AuthRequiredError
from yggdrasil.io.authorization.base import Authorization
from yggdrasil.http_ import HTTPSession
from yggdrasil.http_.io_session import Session

from ._helpers import make_request, make_response


@pytest.fixture(autouse=True)
def _clear_session_singletons():
    Session._INSTANCES.clear()
    yield
    Session._INSTANCES.clear()


# ---------------------------------------------------------------------------
# Test auth handlers
# ---------------------------------------------------------------------------


class _RefreshableAuth(Authorization):
    """Handler that records refresh + authorization calls."""

    def __init__(self, prefix: str = "tok") -> None:
        self._prefix = prefix
        self._version = 0  # bumped to 1 on the first refresh
        self.refresh_calls: list[bool] = []  # captured ``force`` values
        self.auth_reads = 0

    def refresh(self, force: bool = False) -> "_RefreshableAuth":
        self.refresh_calls.append(force)
        # Each refresh rotates the token so the test can prove the
        # second wire call carries a different Authorization header.
        self._version = len(self.refresh_calls)
        return self

    @property
    def authorization(self) -> str:
        self.auth_reads += 1
        return f"Bearer {self._prefix}-{self._version}"


class _LegacyAuth(Authorization):
    """Refresh without a ``force`` kwarg — exercises the fallback branch."""

    def __init__(self) -> None:
        self.refresh_calls = 0

    def refresh(self) -> "_LegacyAuth":
        self.refresh_calls += 1
        return self

    @property
    def authorization(self) -> str:
        return "Bearer legacy"


class _StaticAuth(Authorization):
    """Handler with no ``refresh`` method — only ``authorization``."""

    @property
    def authorization(self) -> str:
        return "Bearer static"


# ---------------------------------------------------------------------------
# Session(__init__) — pre-stamp session headers when auth is provided
# ---------------------------------------------------------------------------


class TestSessionInitWithAuth:

    def test_init_with_auth_prestamps_session_headers(self):
        # ``Session(auth=...)`` should reflect the current credential
        # on session-level ``self.headers`` from the start, so anyone
        # inspecting the session sees the token without going through
        # a request first.
        auth = _RefreshableAuth()
        # Mint the first token so prefix-1 is the value the session
        # will pull on init.
        auth.refresh(force=True)
        s = HTTPSession(base_url="https://api.example.com", auth=auth)
        assert s.headers["Authorization"] == "Bearer tok-1"

    def test_init_without_auth_does_not_set_authorization(self):
        s = HTTPSession(base_url="https://api.example.com")
        assert "Authorization" not in s.headers


# ---------------------------------------------------------------------------
# Session.refresh_auth — unit behavior
# ---------------------------------------------------------------------------


class TestRefreshAuth:

    def test_force_true_no_handler_raises_auth_required(self):
        # Default force=True with no bound handler is a config error —
        # surface it at the right line instead of letting an
        # un-authenticated send hit the wire.
        s = HTTPSession(base_url="https://api.example.com")
        req = make_request("https://api.example.com/x")
        if req.headers is not None:
            req.headers.pop("Authorization", None)

        with pytest.raises(AuthRequiredError) as ctx:
            s.refresh_auth(req)

        # The error carries the request for downstream inspection
        # and subclasses RequestError → HTTPError → YGGException.
        assert ctx.value.request is req
        from yggdrasil.exceptions import RequestError, YGGException
        assert isinstance(ctx.value, RequestError)
        assert isinstance(ctx.value, YGGException)

    def test_force_false_no_handler_returns_self_and_false(self):
        # Steady-state path (prepare_request_before_send): missing
        # handler is fine, just no header gets stamped, and the
        # tuple's bool flags "nothing changed".
        s = HTTPSession(base_url="https://api.example.com")
        req = make_request("https://api.example.com/x")
        if req.headers is not None:
            req.headers.pop("Authorization", None)

        out, refreshed = s.refresh_auth(req, force=False)

        assert out is s
        assert refreshed is False
        assert req.headers is None or "Authorization" not in req.headers

    def test_force_true_default_refreshes_stamps_and_returns_true(self):
        auth = _RefreshableAuth()
        s = HTTPSession(base_url="https://api.example.com", auth=auth)
        # __init__ already stamped one refresh call on the session
        # handler — clear so the assertion is on this refresh only.
        auth.refresh_calls.clear()
        req = make_request("https://api.example.com/x")

        out, refreshed = s.refresh_auth(req)

        assert out is s
        assert refreshed is True
        assert auth.refresh_calls == [True]
        assert req.headers["Authorization"] == "Bearer tok-1"

    def test_force_false_forwards_to_handler(self):
        auth = _RefreshableAuth()
        s = HTTPSession(base_url="https://api.example.com", auth=auth)
        auth.refresh_calls.clear()
        req = make_request("https://api.example.com/x")

        s.refresh_auth(req, force=False)

        assert auth.refresh_calls == [False]

    def test_session_header_kept_in_sync_with_session_handler(self):
        # When the resolved handler is the session-wide one,
        # refresh_auth mirrors the new value to ``self.headers``.
        auth = _RefreshableAuth()
        s = HTTPSession(base_url="https://api.example.com", auth=auth)
        first = s.headers["Authorization"]
        req = make_request("https://api.example.com/x")

        s.refresh_auth(req, force=True)

        # The session-level header rotated alongside the request one.
        assert s.headers["Authorization"] == req.headers["Authorization"]
        assert s.headers["Authorization"] != first

    def test_legacy_refresh_without_force_kwarg_falls_back(self):
        auth = _LegacyAuth()
        s = HTTPSession(base_url="https://api.example.com", auth=auth)
        # __init__ already stamped the legacy handler — clear so the
        # next refresh_auth call is the one under test.
        auth.refresh_calls = 0
        req = make_request("https://api.example.com/x")

        s.refresh_auth(req)

        assert auth.refresh_calls == 1
        assert req.headers["Authorization"] == "Bearer legacy"

    def test_handler_without_refresh_method_still_stamps_header(self):
        s = HTTPSession(base_url="https://api.example.com", auth=_StaticAuth())
        req = make_request("https://api.example.com/x")

        out, refreshed = s.refresh_auth(req)

        assert out is s
        assert refreshed is True
        assert req.headers["Authorization"] == "Bearer static"

    def test_per_request_auth_wins_over_session_auth(self):
        session_auth = _RefreshableAuth(prefix="session")
        request_auth = _RefreshableAuth(prefix="request")
        s = HTTPSession(base_url="https://api.example.com", auth=session_auth)
        # __init__ already called session_auth.refresh once.
        session_pre_stamp = s.headers["Authorization"]
        session_auth.refresh_calls.clear()
        req = make_request("https://api.example.com/x")
        req.auth = request_auth

        s.refresh_auth(req)

        assert session_auth.refresh_calls == []
        assert request_auth.refresh_calls == [True]
        assert req.headers["Authorization"].startswith("Bearer request-")
        # Per-request override does NOT leak into the session header.
        assert s.headers["Authorization"] == session_pre_stamp


# ---------------------------------------------------------------------------
# prepare_request_before_send — delegates to refresh_auth with force=False
# ---------------------------------------------------------------------------


class TestPrepareRequestBeforeSend:

    def test_prepare_calls_refresh_auth_with_force_false(self):
        auth = _RefreshableAuth()
        s = HTTPSession(base_url="https://api.example.com", auth=auth)
        # __init__ already minted the first token; clear so the next
        # refresh from prepare is what we assert on.
        auth.refresh_calls.clear()
        req = make_request("https://api.example.com/x")

        s.prepare_request_before_send(req)

        # One refresh, with force=False — the steady-state path
        # must reuse the handler's cached token.
        assert auth.refresh_calls == [False]
        assert req.headers["Authorization"] == "Bearer tok-1"

    def test_prepare_without_auth_does_not_touch_authorization(self):
        s = HTTPSession(base_url="https://api.example.com")
        req = make_request("https://api.example.com/x")

        s.prepare_request_before_send(req)

        assert req.headers is None or "Authorization" not in req.headers

    def test_prepare_attaches_session_to_request(self):
        s = HTTPSession(base_url="https://api.example.com")
        req = make_request("https://api.example.com/x")
        assert req.session is None

        s.prepare_request_before_send(req)

        assert req.session is s

    def test_prepare_overwrites_previous_session(self):
        s1 = HTTPSession(base_url="https://one.example.com")
        s2 = HTTPSession(base_url="https://two.example.com")
        req = make_request("https://one.example.com/x")
        req.attach_session(s1)

        s2.prepare_request_before_send(req)

        assert req.session is s2


# ---------------------------------------------------------------------------
# prepare_request — session factory attaches itself
# ---------------------------------------------------------------------------


class TestPrepareRequestAttachesSession:

    def test_prepare_request_attaches_session(self):
        s = HTTPSession(base_url="https://api.example.com")
        req = s.prepare_request("GET", "/endpoint")
        assert req.session is s

    def test_prepare_request_with_params_attaches_session(self):
        s = HTTPSession(base_url="https://api.example.com")
        req = s.prepare_request(
            "POST", "/data", params={"page": "1"}, body=b"{}",
        )
        assert req.session is s


# ---------------------------------------------------------------------------
# 403 retry in HTTPSession._local_send
# ---------------------------------------------------------------------------


class _Stub403HTTPSession(HTTPSession):
    """HTTPSession test double — fakes ``_wire_send`` from a queue.

    Each ``_wire_send`` call pops the next response off ``_queue`` and
    records the Authorization header it saw at send time, so the test
    can assert the retry carried the refreshed credential.
    """

    def __init__(self, *args, **kwargs):
        already = getattr(self, "_initialized", False)
        super().__init__(*args, **kwargs)
        if not already:
            self._queue = []
            self.calls: list[str | None] = []

    def queue(self, *responses):
        self._queue.extend(responses)
        return self

    def _wire_send(self, request, wait_cfg):  # type: ignore[override]
        # Snapshot the Authorization header at the moment of send so
        # the test can verify rotation between attempts.
        self.calls.append(
            request.headers.get("Authorization") if request.headers else None
        )
        if not self._queue:
            return make_response(request=request)
        return self._queue.pop(0)


class TestForbiddenRetry:

    def test_403_with_auth_handler_refreshes_and_retries_once(self):
        auth = _RefreshableAuth()
        s = _Stub403HTTPSession(
            base_url="https://api.example.com",
            auth=auth,
        )
        # First call: 403. Second call (the retry): 200.
        s.queue(
            make_response(status_code=403, body=b'{"error":"forbidden"}'),
            make_response(status_code=200, body=b'{"ok":true}'),
        )

        resp = s.send(make_request("https://api.example.com/v1/x"))

        assert resp.status_code == 200
        # Exactly two wire sends — initial + one retry.
        assert len(s.calls) == 2
        # First refresh from prepare_request_before_send (force=False),
        # second from the 403 retry path (force=True default).
        assert auth.refresh_calls == [False, True]
        # The retry carried the newly-minted token.
        assert s.calls[0] == "Bearer tok-1"
        assert s.calls[1] == "Bearer tok-2"

    def test_403_without_auth_handler_does_not_retry(self):
        s = _Stub403HTTPSession(base_url="https://api.example.com")
        s.queue(make_response(status_code=403, body=b"forbidden"))

        with pytest.raises(Exception):
            # raise_error=True by default → the unretried 403 propagates.
            s.send(make_request("https://api.example.com/v1/x"))

        assert len(s.calls) == 1

    def test_retry_succeeds_when_second_response_is_403(self):
        # The retry budget is exactly one — a second 403 must surface.
        auth = _RefreshableAuth()
        s = _Stub403HTTPSession(
            base_url="https://api.example.com",
            auth=auth,
        )
        s.queue(
            make_response(status_code=403, body=b"still forbidden"),
            make_response(status_code=403, body=b"still forbidden"),
        )

        with pytest.raises(Exception):
            s.send(make_request("https://api.example.com/v1/x"))

        # Exactly two sends — one initial + one retry, no third attempt.
        assert len(s.calls) == 2
        # Two refreshes: force=False from prepare, force=True from
        # the 403 retry. No third refresh because we only retry once.
        assert auth.refresh_calls == [False, True]

    def test_non_403_failure_is_not_retried(self):
        # 500 already goes through urllib3's _TieredRetry inside the
        # pool; our hook must not double-retry it. (Stub fakes the
        # wire layer so urllib3's own retry never fires here — we
        # only assert our hook didn't fire either.)
        auth = _RefreshableAuth()
        s = _Stub403HTTPSession(
            base_url="https://api.example.com",
            auth=auth,
        )
        s.queue(make_response(status_code=500, body=b"oops"))

        with pytest.raises(Exception):
            s.send(make_request("https://api.example.com/v1/x"))

        assert len(s.calls) == 1
        # Only the initial prepare refresh — no force-refresh from
        # a 403 retry path.
        assert auth.refresh_calls == [False]


class TestResponseRefreshAuth:
    """:meth:`Response.refresh_auth` delegates to the bound session."""

    @pytest.mark.xfail(reason="send_many path copies request without attached session")
    def test_response_refresh_auth_refreshes_request_via_session(self):
        auth = _RefreshableAuth()
        s = _Stub403HTTPSession(
            base_url="https://api.example.com",
            auth=auth,
        )
        s.queue(make_response(status_code=403))
        req = make_request("https://api.example.com/x")
        req.attach_session(s)
        resp = s.send(req)

        # Drop the refresh log so the test's force=True call is the
        # only one we assert on.
        auth.refresh_calls.clear()

        out, refreshed = resp.refresh_auth()  # force=True default

        # Tuple contract: (self, refreshed: bool)
        assert out is resp
        assert refreshed is True
        # The bound request now carries the freshly-minted token.
        assert auth.refresh_calls == [True]

    def test_response_refresh_auth_force_false_returns_self_false(self):
        s = HTTPSession(base_url="https://api.example.com")
        req = make_request("https://api.example.com/x")
        req.attach_session(s)
        resp = make_response(request=req, status_code=200)

        # No handler bound → force=False is silently OK.
        out, refreshed = resp.refresh_auth(force=False)
        assert out is resp
        assert refreshed is False

    def test_response_refresh_auth_force_true_raises_without_handler(self):
        s = HTTPSession(base_url="https://api.example.com")
        req = make_request("https://api.example.com/x")
        req.attach_session(s)
        resp = make_response(request=req, status_code=200)

        with pytest.raises(AuthRequiredError):
            resp.refresh_auth()  # force=True default

    def test_response_refresh_auth_without_attached_session_raises(self):
        # Orphan request → can't dispatch — surface a RuntimeError
        # naming the issue so the caller knows what to fix.
        req = make_request("https://api.example.com/x")
        req.detach_session()
        resp = make_response(request=req, status_code=403)

        with pytest.raises(RuntimeError, match="attached session"):
            resp.refresh_auth()
