"""HTTP session that catches request failures and fires a notifier callback.

Why this exists
---------------
:class:`HTTPSession` already ships a ``_TieredRetry`` that absorbs
429 / 5xx / transport blips with a status-aware backoff schedule and
respects server ``Retry-After`` headers. That covers the happy path:
transient failures retry, the response comes back, the pipeline
continues.

What it does *not* cover is the **persistent** failure — a 429 that
outlasts the retry budget, a 500 that never clears, a connect timeout
when the upstream is down. The base session re-raises those (when
``SendConfig.raise_error=True``, the default) which aborts whatever
ingestion job was driving it.

:class:`ErrorNotifyingHTTPSession` wraps :meth:`_send` so that:

1. Persistent failures (final status >= 400, or :meth:`_local_send`
   raising) fire a configurable :attr:`notifier` callback — handy
   for "email me when the vendor API has been down for 30 s",
   Slack pings, PagerDuty triggers, or just structured logging.
2. By default the pipeline **does not break**: the failing response
   is returned to the caller (or a synthetic one is built when the
   wire-level send raised), with ``status_code`` intact, so the
   caller can decide whether to skip / retry / alert.
3. Set :attr:`raise_on_failure=True` for the strict shape — notify
   *and* re-raise.

The notifier signature is ``(response, exc, session) -> None``. One of
``response`` / ``exc`` is always set; never both ``None``. Notifier
exceptions are caught + logged, never propagate — the notifier is a
side channel, not a load-bearing dependency.

Use :func:`smtp_email_notifier` to build a stdlib-only SMTP notifier
without pulling in any extra deps.
"""
from __future__ import annotations

import datetime as dt
import logging
import smtplib
import socket
from email.message import EmailMessage
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Union

from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.send_config import SendConfig

from .response import HTTPResponse
from .session import HTTPSession

if TYPE_CHECKING:  # pragma: no cover
    from yggdrasil.io.response import Response


__all__ = [
    "ErrorNotifyingHTTPSession",
    "Notifier",
    "smtp_email_notifier",
]

LOGGER = logging.getLogger(__name__)


#: Notifier signature. Exactly one of ``response`` / ``exc`` is set on
#: each invocation: response carries the HTTP-level failure (>= 400)
#: that exhausted retries; exc carries the transport-level failure that
#: :meth:`_local_send` could not absorb. ``session`` is the live
#: :class:`HTTPSession` for context (base_url, headers, …).
Notifier = Callable[[Optional["Response"], Optional[BaseException], "HTTPSession"], None]


class ErrorNotifyingHTTPSession(HTTPSession):
    """:class:`HTTPSession` that calls :attr:`notifier` on persistent failure.

    Construct with ``notifier=<callable>`` (or set
    ``session.notifier = ...`` after construction — the attribute is
    transient and not part of the singleton key). When :attr:`notifier`
    is ``None``, the session behaves exactly like its parent.

    ``raise_on_failure`` controls the re-raise behavior:

    - ``False`` (default): notify, then return the failing response to
      the caller. A wire-level exception is converted into a synthetic
      :class:`HTTPResponse` with ``status_code=0`` so the caller never
      sees ``None``.
    - ``True``: notify, then re-raise. Useful when the pipeline really
      should stop and the notifier exists just to alert ops.
    """

    # ``notifier`` / ``raise_on_failure`` are runtime configuration, not
    # identity. List them transient so the singleton-key probe + pickle
    # round-trip skip them; callers re-attach the notifier on the
    # receiving side.
    _TRANSIENT_STATE_ATTRS = HTTPSession._TRANSIENT_STATE_ATTRS | {
        "notifier",
        "raise_on_failure",
    }

    def __init__(
        self,
        *args,
        notifier: Optional[Notifier] = None,
        raise_on_failure: bool = False,
        **kwargs,
    ) -> None:
        # Singleton dispatch: if the parent already initialised this
        # instance for a different (notifier, raise_on_failure) tuple,
        # the new arguments still apply — they're transient overrides.
        already_initialised = getattr(self, "_initialized", False)
        super().__init__(*args, **kwargs)
        # ``super().__init__`` is a no-op when the singleton was already
        # built, so we always (re)attach the notifier slots ourselves.
        if not already_initialised or notifier is not None:
            self.notifier = notifier
        if not already_initialised:
            self.raise_on_failure = raise_on_failure
        elif raise_on_failure is not False:
            self.raise_on_failure = raise_on_failure

    # ------------------------------------------------------------------ #
    # Send pipeline override
    # ------------------------------------------------------------------ #
    def _send(self, request: PreparedRequest, config: SendConfig):  # type: ignore[override]
        """Wrap parent ``_send`` so persistent failures notify instead of crashing.

        Two failure shapes get the same treatment:

        - **Response-level**: parent returned a :class:`Response` with
          ``ok=False`` (status >= 400). Could come from a non-retryable
          4xx that the retry policy let through, or from 429 / 5xx
          exhausting the retry budget.
        - **Wire-level**: ``_local_send`` raised — connect timeout,
          DNS failure, socket reset after retry exhaustion.

        We force ``raise_error=False`` on the config we hand to parent
        so its end-of-pipeline ``raise_for_status`` doesn't bypass our
        notifier; the caller's ``raise_error`` intent is re-applied via
        :attr:`raise_on_failure` below.

        Cache hits are not failures even when ``status >= 400`` — the
        cached row already includes the failure record. The notifier
        only fires once per logical request because the parent's
        cache-hit branch returns before we'd re-check.
        """
        delegated = config.merge(raise_error=False) if config.raise_error else config
        try:
            response = super()._send(request, delegated)
        except Exception as exc:
            self._fire_notifier(response=None, exc=exc)
            if self.raise_on_failure or self._notifier_demands_raise():
                raise
            # Synthetic response — ``status_code=0`` flags "no wire-level reply",
            # ``request`` survives so callers can inspect what they sent. We
            # build it via the same factory the live network path uses so
            # downstream Tabular / cast machinery does not branch.
            return self._build_failure_response(request, exc)

        if not response.ok:
            self._fire_notifier(response=response, exc=None)
            if self.raise_on_failure:
                response.raise_for_status()

        return response

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _fire_notifier(
        self,
        *,
        response: Optional["Response"],
        exc: Optional[BaseException],
    ) -> None:
        notifier = getattr(self, "notifier", None)
        if notifier is None:
            return
        try:
            notifier(response, exc, self)
        except Exception:
            # Notifier failures must not break the data pipeline — they're
            # an alert channel, not a load-bearing path.
            LOGGER.exception("Notifier %r raised; suppressing", notifier)

    def _notifier_demands_raise(self) -> bool:
        # Reserved hook — currently always False. Subclasses may flip it
        # (e.g. to re-raise for certain notifier types) without overriding
        # _send.
        return False

    def _build_failure_response(
        self,
        request: PreparedRequest,
        exc: BaseException,
    ) -> HTTPResponse:
        """Build a synthetic response carrying the wire-level failure.

        ``status_code=0`` is the convention for "no HTTP-level reply
        received". The headers / body are empty; the exception text
        lands in a custom header for diagnostics.
        """
        return HTTPResponse(
            request=request,
            status_code=0,
            headers={"x-ygg-error": f"{type(exc).__name__}: {exc}"},
            tags={},
            buffer=b"",
            received_at=dt.datetime.now(dt.timezone.utc),
            local_cached=False,
            remote_cached=False,
        )


# ---------------------------------------------------------------------------
# SMTP notifier factory — stdlib only
# ---------------------------------------------------------------------------
def smtp_email_notifier(
    *,
    host: str,
    port: int = 25,
    from_addr: str,
    to_addrs: Union[str, Sequence[str]],
    use_tls: bool = False,
    username: Optional[str] = None,
    password: Optional[str] = None,
    subject_prefix: str = "[ygg-http]",
    timeout: float = 10.0,
) -> Notifier:
    """Build a :data:`Notifier` that sends one email per persistent failure.

    Uses stdlib ``smtplib`` + ``email.message`` — no extra deps. The
    callable is closed over the SMTP parameters; one notifier per
    distinct mail config.

    The subject reads ``"{subject_prefix} {method} {host}{path} -> {status}"``;
    the body carries the request method / URL / headers, the response
    status / body excerpt, and the exception (when there was one).

    SMTP failures inside the notifier are caught + logged by the
    enclosing :meth:`_fire_notifier` so a flaky mail server does not
    take the ingestion pipeline down.
    """
    recipients = [to_addrs] if isinstance(to_addrs, str) else list(to_addrs)
    if not recipients:
        raise ValueError("smtp_email_notifier: at least one recipient required.")

    def notifier(response, exc, session) -> None:
        request = getattr(response, "request", None) if response is not None else None

        if response is not None:
            status = response.status_code
            url = str(getattr(request, "url", "?"))
            method = getattr(request, "method", "?")
            # ``Response.content`` is the bytes accessor (decoded through
            # the response codec when present). ``Response.body`` returns
            # the underlying ``Holder`` / ``Memory`` — not subscriptable.
            try:
                body_excerpt = (response.content or b"")[:2048]
            except Exception:
                body_excerpt = b""
        else:
            status = 0
            url = "?"
            method = "?"
            body_excerpt = b""

        subject = f"{subject_prefix} {method} {url} -> {status}"
        body_lines = [
            f"Time:    {dt.datetime.now(dt.timezone.utc).isoformat()}",
            f"Session: {session!r}",
            f"Method:  {method}",
            f"URL:     {url}",
            f"Status:  {status}",
        ]
        if exc is not None:
            body_lines.append(f"Error:   {type(exc).__name__}: {exc}")
        if body_excerpt:
            try:
                excerpt_text = body_excerpt.decode("utf-8", errors="replace")
            except Exception:
                excerpt_text = repr(body_excerpt)
            body_lines.append("")
            body_lines.append("Body excerpt (first 2 KB):")
            body_lines.append(excerpt_text)

        msg = EmailMessage()
        msg["From"] = from_addr
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg.set_content("\n".join(body_lines))

        try:
            with smtplib.SMTP(host, port, timeout=timeout) as smtp:
                if use_tls:
                    smtp.starttls()
                if username is not None and password is not None:
                    smtp.login(username, password)
                smtp.send_message(msg)
        except (OSError, socket.timeout, smtplib.SMTPException) as smtp_exc:
            # Re-raise — the enclosing _fire_notifier catches and logs.
            raise RuntimeError(
                f"smtp_email_notifier failed to dispatch to {host}:{port}: {smtp_exc}"
            ) from smtp_exc

    return notifier
