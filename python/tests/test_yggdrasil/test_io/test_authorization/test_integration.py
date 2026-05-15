"""Integration tests for :class:`Authorization` with
:class:`PreparedRequest` and :class:`Session`.

These tests pin the wire-up between the auth contract and the request /
session machinery — how a handler reaches the outbound ``Authorization``
header, who wins when both the request and the session carry one,
refresh-on-every-send semantics for rotating tokens, type validation
at the property setters, copy / pickle preservation, and the end-to-end
``send`` pipeline through :class:`StubSession`.

The :class:`Authorization` base is exercised via a tiny ``_StaticAuth``
double; no real MSAL / AAD round-trip is issued — those paths live in
``test_msal.py``.
"""
from __future__ import annotations

import pickle

import pytest

from yggdrasil.io.authorization import Authorization
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.session import Session

from .._helpers import StubSession, make_request, make_response


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StaticAuth(Authorization):
    """Fixed-header :class:`Authorization` — the simplest concrete handler."""

    def __init__(self, header: str) -> None:
        self._header = header

    @property
    def authorization(self) -> str:
        return self._header


class _RotatingAuth(Authorization):
    """:class:`Authorization` that emits a new value on every property read.

    Mirrors the live MSAL refresh-on-expiry contract: each ``send`` reads
    the property afresh, so callers always get the current token without
    having to re-bind the handler.
    """

    def __init__(self, prefix: str = "Bearer ") -> None:
        self._prefix = prefix
        self.calls = 0

    @property
    def authorization(self) -> str:
        self.calls += 1
        return f"{self._prefix}tok-{self.calls}"


# ---------------------------------------------------------------------------
# Session singleton hygiene — clear cross-test bleed.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_session_singleton_cache():
    Session._INSTANCES.clear()
    yield
    Session._INSTANCES.clear()


# ---------------------------------------------------------------------------
# PreparedRequest.auth — constructor + property
# ---------------------------------------------------------------------------


class TestPreparedRequestAuth:

    def test_prepare_accepts_auth_handler(self) -> None:
        auth = _StaticAuth("Bearer abc")
        req = PreparedRequest.prepare(
            "GET", "https://example.com/x", auth=auth,
        )
        assert req.auth is auth

    def test_prepare_without_auth_leaves_auth_none(self) -> None:
        req = PreparedRequest.prepare("GET", "https://example.com/x")
        assert req.auth is None

    def test_prepare_does_not_write_header_eagerly(self) -> None:
        # ``auth`` is resolved lazily — ``prepare`` must NOT call the
        # handler's ``authorization`` property, otherwise a single
        # request reuse would freeze the token at construction time.
        auth = _RotatingAuth()
        req = PreparedRequest.prepare(
            "GET", "https://example.com/x", auth=auth,
        )
        assert auth.calls == 0
        assert req.authorization is None  # no header written yet

    def test_auth_setter_stores_handler(self) -> None:
        req = make_request()
        auth = _StaticAuth("Bearer abc")
        req.auth = auth
        assert req.auth is auth

    def test_auth_setter_clears_with_none(self) -> None:
        req = make_request()
        req.auth = _StaticAuth("Bearer abc")
        req.auth = None
        assert req.auth is None

    def test_auth_setter_rejects_string(self) -> None:
        # The setter is the strict door — bare strings have to go through
        # ``authorization`` instead so the call site is explicit about
        # "static value" vs "live handler".
        req = make_request()
        with pytest.raises(TypeError, match="auth must be an Authorization"):
            req.auth = "Bearer literal"  # type: ignore[assignment]

    def test_auth_setter_rejects_arbitrary_object(self) -> None:
        req = make_request()
        with pytest.raises(TypeError, match="auth must be an Authorization"):
            req.auth = object()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# PreparedRequest.authorization — string + handler dual-mode setter
# ---------------------------------------------------------------------------


class TestAuthorizationPropertyDualMode:

    def test_string_setter_writes_header(self) -> None:
        req = make_request()
        req.authorization = "Bearer literal"
        assert req.headers.get("Authorization") == "Bearer literal"
        assert req.authorization == "Bearer literal"
        # ``_auth`` stays None — a static string is not a handler.
        assert req.auth is None

    def test_none_clears_header(self) -> None:
        req = make_request(headers={"Authorization": "Bearer old"})
        req.authorization = None
        assert req.headers.get("Authorization") is None
        assert req.authorization is None

    def test_handler_via_authorization_setter_binds_auth_not_header(self) -> None:
        # When given an :class:`Authorization` instance, the setter must
        # bind it to ``_auth`` and leave the header alone — header
        # materialization happens later in ``prepare_authorization``.
        req = make_request()
        handler = _StaticAuth("Bearer h-value")
        req.authorization = handler
        assert req.auth is handler
        assert req.headers.get("Authorization") is None

    def test_string_setter_after_handler_clears_handler(self) -> None:
        # Switching from a handler to a static string must drop the
        # handler — otherwise the next send would re-overwrite the
        # static value with the handler's output.
        req = make_request()
        req.authorization = _StaticAuth("Bearer h-value")
        req.authorization = "Bearer static"
        assert req.auth is None
        assert req.headers.get("Authorization") == "Bearer static"


# ---------------------------------------------------------------------------
# prepare_authorization — handler → header materialization
# ---------------------------------------------------------------------------


class TestPrepareAuthorization:

    def test_no_handler_is_noop(self) -> None:
        req = make_request()
        result = req.prepare_authorization()
        assert result is req
        assert req.headers.get("Authorization") is None

    def test_writes_header_from_handler(self) -> None:
        req = make_request()
        req.auth = _StaticAuth("Bearer materialized")
        req.prepare_authorization()
        assert req.headers.get("Authorization") == "Bearer materialized"

    def test_overwrites_existing_authorization_header(self) -> None:
        # A pre-existing static header must not block the handler's
        # output — the handler is the source of truth once bound.
        req = make_request(headers={"Authorization": "Bearer stale"})
        req.auth = _StaticAuth("Bearer fresh")
        req.prepare_authorization()
        assert req.headers.get("Authorization") == "Bearer fresh"

    def test_reads_handler_on_every_call(self) -> None:
        # The whole point of binding a handler instead of a static
        # string: each call pulls the current token. A handler whose
        # property rotates must produce different headers per call.
        req = make_request()
        handler = _RotatingAuth()
        req.auth = handler
        req.prepare_authorization()
        first = req.headers.get("Authorization")
        req.prepare_authorization()
        second = req.headers.get("Authorization")
        assert first == "Bearer tok-1"
        assert second == "Bearer tok-2"
        assert handler.calls == 2


# ---------------------------------------------------------------------------
# Copy / clone preserves the bound handler
# ---------------------------------------------------------------------------


class TestCopyPreservesAuth:

    def test_copy_default_preserves_handler(self) -> None:
        auth = _StaticAuth("Bearer kept")
        req = PreparedRequest.prepare(
            "GET", "https://example.com/x", auth=auth,
        )
        clone = req.copy()
        assert clone.auth is auth

    def test_copy_explicit_none_clears_handler(self) -> None:
        auth = _StaticAuth("Bearer kept")
        req = PreparedRequest.prepare(
            "GET", "https://example.com/x", auth=auth,
        )
        clone = req.copy(auth=None)
        assert clone.auth is None
        # Original is untouched — ``copy`` returns a fresh instance.
        assert req.auth is auth

    def test_copy_swaps_handler(self) -> None:
        a = _StaticAuth("Bearer a")
        b = _StaticAuth("Bearer b")
        req = PreparedRequest.prepare(
            "GET", "https://example.com/x", auth=a,
        )
        clone = req.copy(auth=b)
        assert clone.auth is b
        assert req.auth is a


# ---------------------------------------------------------------------------
# Pickle round-trip — the handler is durable state
# ---------------------------------------------------------------------------


class TestPickleAuth:

    def test_request_pickle_preserves_handler(self) -> None:
        # ``_auth`` is not in ``_TRANSIENT_STATE_ATTRS`` — the handler
        # is part of the request's wire contract and must survive a
        # Spark-worker pickle round-trip.
        req = PreparedRequest.prepare(
            "GET", "https://example.com/x", auth=_StaticAuth("Bearer t"),
        )
        clone = pickle.loads(pickle.dumps(req))
        assert clone.auth is not None
        assert clone.auth.authorization == "Bearer t"

    def test_request_pickle_with_no_handler(self) -> None:
        req = PreparedRequest.prepare("GET", "https://example.com/x")
        clone = pickle.loads(pickle.dumps(req))
        assert clone.auth is None


# ---------------------------------------------------------------------------
# Session.auth — property + type validation + pickle
# ---------------------------------------------------------------------------


class TestSessionAuth:

    def test_constructor_accepts_auth_handler(self) -> None:
        auth = _StaticAuth("Bearer session")
        session = StubSession(base_url="https://api.example.com", auth=auth)
        assert session.auth is auth

    def test_constructor_without_auth(self) -> None:
        session = StubSession(base_url="https://api.example.com")
        assert session.auth is None

    def test_auth_setter_stores_handler(self) -> None:
        session = StubSession()
        auth = _StaticAuth("Bearer session")
        session.auth = auth
        assert session.auth is auth

    def test_auth_setter_clears_with_none(self) -> None:
        session = StubSession()
        session.auth = _StaticAuth("Bearer session")
        session.auth = None
        assert session.auth is None

    def test_auth_setter_rejects_non_authorization(self) -> None:
        session = StubSession()
        with pytest.raises(TypeError, match="auth must be an Authorization"):
            session.auth = "Bearer literal"  # type: ignore[assignment]

    def test_pickle_round_trip_preserves_handler(self) -> None:
        # The session-wide handler must travel into Spark workers
        # alongside the rest of the session state.
        session = StubSession(auth=_StaticAuth("Bearer roundtrip"))
        clone = pickle.loads(pickle.dumps(session))
        assert clone.auth is not None
        assert clone.auth.authorization == "Bearer roundtrip"


# ---------------------------------------------------------------------------
# Integration — Session.send pipeline writes the Authorization header
# ---------------------------------------------------------------------------


class TestSendIntegration:

    def test_session_auth_applied_on_send(self) -> None:
        # Session-wide handler reaches the wire as the ``Authorization``
        # header on every outbound request — that's the headline contract
        # of ``prepare_request_before_send``.
        session = StubSession(auth=_StaticAuth("Bearer session-tok"))
        req = make_request()
        session.queue(make_response(request=req))
        session.send(req)
        assert len(session.calls) == 1
        sent = session.calls[0]
        assert sent.headers["Authorization"] == "Bearer session-tok"

    def test_request_auth_wins_over_session_auth(self) -> None:
        # Per-request handler must beat the session-wide fallback —
        # otherwise a caller couldn't override the default credentials
        # for a single call (e.g. service-account vs user-token).
        session = StubSession(auth=_StaticAuth("Bearer session-tok"))
        req = make_request()
        req.auth = _StaticAuth("Bearer request-tok")
        session.queue(make_response(request=req))
        session.send(req)
        assert session.calls[0].headers["Authorization"] == "Bearer request-tok"

    def test_no_handler_leaves_headers_untouched(self) -> None:
        # No session auth, no request auth — the send must not invent
        # an ``Authorization`` header out of nowhere.
        session = StubSession()
        req = make_request()
        session.queue(make_response(request=req))
        session.send(req)
        assert "Authorization" not in session.calls[0].headers

    def test_static_header_survives_when_no_handler_bound(self) -> None:
        # A pre-set static ``Authorization`` header has to pass through
        # untouched — the handler path is the only one that mutates it.
        session = StubSession()
        req = make_request(headers={"Authorization": "Bearer static"})
        session.queue(make_response(request=req))
        session.send(req)
        assert session.calls[0].headers["Authorization"] == "Bearer static"

    def test_session_handler_overrides_static_header(self) -> None:
        # ``prepare_request_before_send`` writes the resolved handler
        # value straight into ``headers["Authorization"]`` — a stale
        # static value on the request must not survive a bound handler.
        session = StubSession(auth=_StaticAuth("Bearer session-tok"))
        req = make_request(headers={"Authorization": "Bearer stale"})
        session.queue(make_response(request=req))
        session.send(req)
        assert session.calls[0].headers["Authorization"] == "Bearer session-tok"

    def test_handler_invoked_per_send_for_rotating_token(self) -> None:
        # Each send hits the handler's ``authorization`` property afresh
        # — refresh-on-expiry handlers (MSAL et al.) rely on this to
        # roll the token between calls.
        handler = _RotatingAuth()
        session = StubSession(auth=handler)

        req1 = make_request()
        session.queue(make_response(request=req1))
        session.send(req1)

        req2 = make_request()
        session.queue(make_response(request=req2))
        session.send(req2)

        assert handler.calls == 2
        assert session.calls[0].headers["Authorization"] == "Bearer tok-1"
        assert session.calls[1].headers["Authorization"] == "Bearer tok-2"

    def test_request_handler_takes_precedence_per_call(self) -> None:
        # The per-request override is decided independently each send —
        # a request without its own handler still falls back to the
        # session-wide one, even when an earlier request overrode it.
        session_handler = _RotatingAuth(prefix="Session ")
        request_handler = _RotatingAuth(prefix="Request ")
        session = StubSession(auth=session_handler)

        req_override = make_request()
        req_override.auth = request_handler
        session.queue(make_response(request=req_override))
        session.send(req_override)

        req_fallback = make_request()
        session.queue(make_response(request=req_fallback))
        session.send(req_fallback)

        assert session.calls[0].headers["Authorization"] == "Request tok-1"
        assert session.calls[1].headers["Authorization"] == "Session tok-1"
        # Only one handler is called per send — no double-resolve.
        assert request_handler.calls == 1
        assert session_handler.calls == 1
