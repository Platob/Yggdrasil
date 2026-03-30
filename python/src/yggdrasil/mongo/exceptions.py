from __future__ import annotations


class HttpMongoError(RuntimeError):
    """Raised when the HTTP Mongo gateway returns an error."""


class HttpMongoTransportError(HttpMongoError):
    """Raised when the HTTP transport fails or returns invalid data."""
