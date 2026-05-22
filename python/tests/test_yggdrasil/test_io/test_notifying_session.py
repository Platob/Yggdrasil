"""Tests for :class:`ErrorNotifyingHTTPSession`.

Exercises the four interesting shapes:

1. **Success path** — 200 OK never fires the notifier.
2. **HTTP failure (>= 400)** — notifier fires with the response,
   pipeline keeps going (no raise) when ``raise_on_failure=False``.
3. **Wire-level failure** — ``_local_send`` raising fires the notifier
   with the exception; a synthetic ``status_code=0`` response comes
   back to the caller.
4. **Strict mode** — ``raise_on_failure=True`` notifies *and* re-raises.

Plus: notifier exceptions are caught so the pipeline never breaks on
a flaky alert path, and the SMTP notifier factory builds a callable
whose body is correctly formatted (no real SMTP server contacted —
the stdlib ``smtplib`` connection is patched out).
"""
from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import pytest

from yggdrasil.http_ import ErrorNotifyingHTTPSession, smtp_email_notifier
from yggdrasil.io.session import Session

from ._helpers import make_request, make_response


# ---------------------------------------------------------------------------
# Singleton-cache hygiene — clear singleton bleed between tests so each
# notifier attachment lands on a fresh instance.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clear_session_singleton_cache():
    Session._INSTANCES.clear()
    yield
    Session._INSTANCES.clear()


# ---------------------------------------------------------------------------
# Helper — a session subclass that returns canned responses without a
# real network. Mirrors StubSession but extends ErrorNotifyingHTTPSession
# so the notifier path runs against the same _send pipeline.
# ---------------------------------------------------------------------------
class _StubNotifyingHTTPSession(ErrorNotifyingHTTPSession):
    """Test double: ``_local_send`` returns from a queued list, or raises."""

    def __init__(self, *args, **kwargs):
        already = getattr(self, "_initialized", False)
        super().__init__(*args, **kwargs)
        if not already:
            self._queue: list = []
            self.calls: list = []

    def queue(self, *responses):
        self._queue.extend(responses)
        return self

    def queue_exception(self, exc):
        self._queue.append(exc)
        return self

    def _local_send(self, request, config):  # type: ignore[override]
        self.calls.append(request)
        if not self._queue:
            return make_response(request=request)
        item = self._queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


# ---------------------------------------------------------------------------
# Notifier callback contract
# ---------------------------------------------------------------------------
class TestNotifierContract:

    def test_success_does_not_notify(self):
        notifier = MagicMock()
        s = _StubNotifyingHTTPSession(
            base_url="https://api.example.com",
            notifier=notifier,
        )
        s.queue(make_response(status_code=200))
        resp = s.send(make_request("https://api.example.com/v1/x"))
        assert resp.status_code == 200
        notifier.assert_not_called()

    def test_http_failure_fires_notifier_with_response(self):
        notifier = MagicMock()
        s = _StubNotifyingHTTPSession(
            base_url="https://api.example.com",
            notifier=notifier,
        )
        s.queue(make_response(status_code=429, body=b'{"error":"rate"}'))
        # Default SendConfig has raise_error=True; the notifying session
        # must catch the resulting failure and surface the response.
        resp = s.send(make_request("https://api.example.com/v1/x"))
        assert resp.status_code == 429
        notifier.assert_called_once()
        args, kwargs = notifier.call_args
        response_arg, exc_arg, session_arg = args
        assert response_arg is resp
        assert exc_arg is None
        assert session_arg is s

    def test_wire_failure_fires_notifier_with_exception(self):
        notifier = MagicMock()
        s = _StubNotifyingHTTPSession(
            base_url="https://api.example.com",
            notifier=notifier,
        )
        boom = ConnectionResetError("upstream died")
        s.queue_exception(boom)
        resp = s.send(make_request("https://api.example.com/v1/x"))
        # Pipeline continued; synthetic response carries status 0 + diag header.
        assert resp.status_code == 0
        assert "x-ygg-error" in resp.headers
        assert "ConnectionResetError" in resp.headers["x-ygg-error"]

        notifier.assert_called_once()
        response_arg, exc_arg, _ = notifier.call_args.args
        assert response_arg is None
        assert exc_arg is boom

    def test_no_notifier_is_silent(self):
        # notifier=None is the no-op shape — behaves like the parent.
        s = _StubNotifyingHTTPSession(base_url="https://api.example.com", notifier=None)
        s.queue(make_response(status_code=500))
        resp = s.send(make_request("https://api.example.com/v1/x"))
        # No notifier means no swallow either — the parent's raise path
        # should kick in. Default SendConfig.raise_error=True, so the
        # response is the failing 500 itself (no synthetic).
        assert resp.status_code == 500

    def test_notifier_exception_does_not_break_pipeline(self):
        # The notifier raises but the pipeline keeps going. ``caplog`` is
        # unreliable here because conftest pins ``logger.propagate=False``
        # on the yggdrasil logger; assert on the observable behavior
        # (pipeline returns the failing response) instead of the log line.
        def bad_notifier(response, exc, session):
            raise RuntimeError("flaky alert backend")

        s = _StubNotifyingHTTPSession(
            base_url="https://api.example.com",
            notifier=bad_notifier,
        )
        s.queue(make_response(status_code=500))
        resp = s.send(make_request("https://api.example.com/v1/x"))
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# raise_on_failure flag
# ---------------------------------------------------------------------------
class TestRaiseOnFailure:

    def test_default_does_not_raise(self):
        s = _StubNotifyingHTTPSession(
            base_url="https://api.example.com",
            notifier=MagicMock(),
        )
        s.queue(make_response(status_code=500))
        resp = s.send(make_request("https://api.example.com/v1/x"))
        assert resp.status_code == 500

    def test_strict_mode_reraises_after_notify(self):
        notifier = MagicMock()
        s = _StubNotifyingHTTPSession(
            base_url="https://api.example.com",
            notifier=notifier,
            raise_on_failure=True,
        )
        s.queue(make_response(status_code=500))
        with pytest.raises(Exception):  # noqa: PT011 — parent raises an HTTPError-like
            s.send(make_request("https://api.example.com/v1/x"))
        notifier.assert_called_once()

    def test_strict_mode_reraises_wire_exception(self):
        notifier = MagicMock()
        s = _StubNotifyingHTTPSession(
            base_url="https://api.example.com",
            notifier=notifier,
            raise_on_failure=True,
        )
        s.queue_exception(ConnectionResetError("nope"))
        with pytest.raises(ConnectionResetError):
            s.send(make_request("https://api.example.com/v1/x"))
        notifier.assert_called_once()


# ---------------------------------------------------------------------------
# Transient state — notifier doesn't leak into the singleton-key /
# pickle payload
# ---------------------------------------------------------------------------
class TestTransientState:

    def test_notifier_in_transient_attrs(self):
        assert "notifier" in ErrorNotifyingHTTPSession._TRANSIENT_STATE_ATTRS
        assert "raise_on_failure" in ErrorNotifyingHTTPSession._TRANSIENT_STATE_ATTRS

    def test_notifier_not_in_pickle_state(self):
        s = ErrorNotifyingHTTPSession(
            base_url="https://api.example.com",
            notifier=MagicMock(),
        )
        state = s.__getstate__()
        assert "notifier" not in state
        assert "raise_on_failure" not in state


# ---------------------------------------------------------------------------
# SMTP notifier factory
# ---------------------------------------------------------------------------
class TestSmtpEmailNotifier:

    def test_requires_recipients(self):
        with pytest.raises(ValueError):
            smtp_email_notifier(
                host="localhost",
                from_addr="ops@example.com",
                to_addrs=[],
            )

    def test_sends_message_via_smtplib(self):
        notifier = smtp_email_notifier(
            host="smtp.example.com",
            port=587,
            from_addr="ops@example.com",
            to_addrs=["alerts@example.com"],
            use_tls=True,
            username="user",
            password="pw",
        )

        # Build a synthetic failing response so the notifier has
        # something to render in the email body.
        req = make_request("https://api.vendor.example.com/v1/orders")
        resp = make_response(request=req, status_code=429, body=b"rate limited")
        session = ErrorNotifyingHTTPSession(base_url="https://api.vendor.example.com")

        with patch("smtplib.SMTP") as smtp_cls:
            instance = smtp_cls.return_value.__enter__.return_value
            notifier(resp, None, session)

        smtp_cls.assert_called_once_with("smtp.example.com", 587, timeout=10.0)
        instance.starttls.assert_called_once()
        instance.login.assert_called_once_with("user", "pw")
        instance.send_message.assert_called_once()
        sent = instance.send_message.call_args.args[0]
        assert sent["From"] == "ops@example.com"
        assert sent["To"] == "alerts@example.com"
        assert "429" in sent["Subject"]
        body = sent.get_content()
        assert "rate limited" in body
        assert "https://api.vendor.example.com/v1/orders" in body

    def test_smtp_failure_re_raises_runtime_error(self):

        notifier = smtp_email_notifier(
            host="smtp.example.com",
            from_addr="ops@example.com",
            to_addrs=["alerts@example.com"],
        )
        req = make_request()
        resp = make_response(request=req, status_code=500)
        session = ErrorNotifyingHTTPSession(base_url="https://api.example.com")

        with patch("smtplib.SMTP", side_effect=socket.timeout("smtp down")):
            with pytest.raises(RuntimeError, match="smtp_email_notifier failed"):
                notifier(resp, None, session)

    def test_notifier_wire_failure_includes_exception(self):
        notifier = smtp_email_notifier(
            host="smtp.example.com",
            from_addr="ops@example.com",
            to_addrs="alerts@example.com",
        )
        session = ErrorNotifyingHTTPSession(base_url="https://api.example.com")

        with patch("smtplib.SMTP") as smtp_cls:
            instance = smtp_cls.return_value.__enter__.return_value
            notifier(None, ConnectionResetError("upstream gone"), session)

        sent = instance.send_message.call_args.args[0]
        body = sent.get_content()
        assert "ConnectionResetError" in body
        assert "upstream gone" in body


# ---------------------------------------------------------------------------
# Smoke: end-to-end notifier + 429 + retry-exhaustion shape
# ---------------------------------------------------------------------------
class TestSmokeEndToEnd:
    """Glue check: 429 → notifier → pipeline keeps running.

    Doesn't exercise the real ``_TieredRetry`` because the stub session
    bypasses the network pool entirely. The retry shape is covered by
    the existing :mod:`test_session_cache_internals` tests; this only
    asserts the notifying wrapper integrates with the same _send
    pipeline.
    """

    def test_429_round_trip_does_not_crash(self):
        events: list = []

        def notifier(response, exc, session):
            events.append((response.status_code if response else None,
                           type(exc).__name__ if exc else None))

        s = _StubNotifyingHTTPSession(
            base_url="https://api.example.com",
            notifier=notifier,
        )
        # Two 429s in a row — second call still completes, both notify.
        s.queue(make_response(status_code=429))
        s.queue(make_response(status_code=429))

        r1 = s.send(make_request("https://api.example.com/v1/a"))
        r2 = s.send(make_request("https://api.example.com/v1/b"))

        assert r1.status_code == 429
        assert r2.status_code == 429
        assert events == [(429, None), (429, None)]
