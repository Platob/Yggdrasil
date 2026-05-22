"""Tests for the centralised :mod:`yggdrasil.exceptions` surface.

Verifies:

1. ``YGGException`` is the single library-wide root — every exception
   yggdrasil deliberately raises subclasses it.
2. HTTP exceptions live at :mod:`yggdrasil.exceptions` (not
   ``yggdrasil.io.errors``) and double-subclass the transport
   :class:`yggdrasil.http_._pool.exceptions.HTTPError` so transport-level
   ``except`` blocks still catch them.
3. ``make_for_status`` dispatches to the right subclass for each
   status code and the produced exception is catchable as both
   ``YGGException`` and the specific subclass.
4. ``CastError`` keeps its dual-subclass property
   (``YGGException`` + ``pa.ArrowInvalid``).
5. The legacy module path ``yggdrasil.io.errors`` no longer exists —
   callers must use ``yggdrasil.exceptions``.
"""
from __future__ import annotations

import importlib

import pyarrow as pa
import pytest

from yggdrasil.http_._pool import exceptions as _u3

from yggdrasil.exceptions import (
    BadRequest,
    CastError,
    ClientError,
    ForbiddenError,
    HTTPError,
    InternalServerError,
    NotFoundError,
    ServerError,
    TooManyRequests,
    UnauthorizedError,
    YGGException,
    make_for_status,
)

from .test_io._helpers import make_request, make_response


# ---------------------------------------------------------------------------
# Hierarchy invariants
# ---------------------------------------------------------------------------


class TestHierarchy:

    def test_yggexception_is_root(self):
        assert issubclass(CastError, YGGException)
        assert issubclass(HTTPError, YGGException)
        for cls in (
            ClientError, ServerError, BadRequest, UnauthorizedError,
            ForbiddenError, NotFoundError, TooManyRequests,
            InternalServerError,
        ):
            assert issubclass(cls, YGGException), cls.__name__
            assert issubclass(cls, HTTPError), cls.__name__

    def test_http_errors_keep_pool_compatibility(self):
        # Transport-level retry / catch blocks must still match.
        assert issubclass(HTTPError, _u3.HTTPError)
        assert issubclass(NotFoundError, _u3.HTTPError)

    def test_cast_error_keeps_arrow_compatibility(self):
        assert issubclass(CastError, pa.ArrowInvalid)
        assert issubclass(CastError, YGGException)


# ---------------------------------------------------------------------------
# make_for_status — status → subclass dispatch
# ---------------------------------------------------------------------------


class TestMakeForStatus:

    @pytest.mark.parametrize(
        "status,expected_cls",
        [
            (400, BadRequest),
            (401, UnauthorizedError),
            (403, ForbiddenError),
            (404, NotFoundError),
            (429, TooManyRequests),
            (500, InternalServerError),
            (418, ClientError),   # un-mapped 4xx → ClientError fallback
            (599, ServerError),   # un-mapped 5xx → ServerError fallback
        ],
    )
    def test_dispatches_to_specific_subclass(self, status, expected_cls):
        req = make_request("https://api.example.com/x")
        resp = make_response(request=req, status_code=status, body=b"err")
        exc = make_for_status(resp)
        assert isinstance(exc, expected_cls)
        # And catchable as the library-wide root.
        assert isinstance(exc, YGGException)

    def test_returns_none_for_2xx(self):
        req = make_request("https://api.example.com/x")
        resp = make_response(request=req, status_code=200, body=b'{"ok":true}')
        assert make_for_status(resp) is None

    def test_response_bound_attributes_present(self):
        req = make_request("https://api.example.com/missing")
        resp = make_response(request=req, status_code=404, body=b"not found")
        exc = make_for_status(resp)
        assert isinstance(exc, NotFoundError)
        assert exc.response is resp
        assert exc.request is req
        assert exc.status_code == 404


# ---------------------------------------------------------------------------
# Catch shape — both the library-wide and transport-wide branches catch.
# ---------------------------------------------------------------------------


class TestCatchShape:

    def test_except_ygg_exception_catches_http_failure(self):
        req = make_request("https://api.example.com/x")
        resp = make_response(request=req, status_code=403, body=b"nope")
        exc = make_for_status(resp)
        try:
            raise exc  # type: ignore[misc]
        except YGGException as e:
            assert isinstance(e, ForbiddenError)
        else:  # pragma: no cover — branch above must take the catch
            pytest.fail("YGGException did not catch HTTP error")

    def test_except_pool_httperror_still_catches(self):
        req = make_request("https://api.example.com/x")
        resp = make_response(request=req, status_code=500, body=b"oops")
        exc = make_for_status(resp)
        try:
            raise exc  # type: ignore[misc]
        except _u3.HTTPError as e:
            assert isinstance(e, InternalServerError)


# ---------------------------------------------------------------------------
# Legacy path removed — no back-compat shim per project rule.
# ---------------------------------------------------------------------------


def test_legacy_io_errors_module_is_gone():
    with pytest.raises(ImportError):
        importlib.import_module("yggdrasil.io.errors")
