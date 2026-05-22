"""HTTP transport for yggdrasil — single entry point :class:`HTTPSession`.

Construct an :class:`HTTPSession` (singleton-cached per config) and drive HTTP
through its inherited verb methods; the returned :class:`HTTPResponse` carries
the body, headers, status, and request bound on a single object. The
supporting types (:class:`HTTPRequest`, :class:`HTTPPath`, :class:`Cookies`,
:class:`ErrorNotifyingHTTPSession`) round out the surface for paths, cookie
jars, and failure notifications.

The pool / retry / timeout primitives live in :mod:`yggdrasil.http_._pool`
(stdlib-only, urllib3-shaped) — feature code should not import them directly.

Submodule imports are deferred via PEP 562 ``__getattr__`` so importing
``yggdrasil.http_._pool`` (used by :mod:`yggdrasil.exceptions.http`) does not
pull the full session / response / path chain — that chain transitively reaches
back into ``yggdrasil.exceptions`` and would otherwise close the loop.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "HTTPSession",
    "HTTPResponse",
    "HTTPRequest",
    "HTTPPath",
    "Cookies",
    "ErrorNotifyingHTTPSession",
    "Notifier",
    "smtp_email_notifier",
]

_LAZY_NAMES = {
    "HTTPSession": (".session", "HTTPSession"),
    "HTTPResponse": (".response", "HTTPResponse"),
    "HTTPRequest": (".request", "HTTPRequest"),
    "HTTPPath": (".path", "HTTPPath"),
    "Cookies": (".cookies", "Cookies"),
    "ErrorNotifyingHTTPSession": (".notifying_session", "ErrorNotifyingHTTPSession"),
    "Notifier": (".notifying_session", "Notifier"),
    "smtp_email_notifier": (".notifying_session", "smtp_email_notifier"),
}


def __getattr__(name: str):
    spec = _LAZY_NAMES.get(name)
    if spec is None:
        raise AttributeError(f"module 'yggdrasil.http_' has no attribute {name!r}")
    import importlib
    mod = importlib.import_module(spec[0], __name__)
    value = getattr(mod, spec[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_NAMES})


if TYPE_CHECKING:
    from .session import HTTPSession
    from .response import HTTPResponse
    from .request import HTTPRequest
    from .path import HTTPPath
    from .cookies import Cookies
    from .notifying_session import (
        ErrorNotifyingHTTPSession,
        Notifier,
        smtp_email_notifier,
    )
