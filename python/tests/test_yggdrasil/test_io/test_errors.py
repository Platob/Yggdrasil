"""Tests for yggdrasil.io.errors."""

from __future__ import annotations

import urllib3.exceptions as _u3

from yggdrasil.io.errors import (
    BadGatewayError,
    BadRequest,
    CacheError,
    ClientError,
    ConnectTimeoutError,
    ForbiddenError,
    GatewayTimeout,
    HTTPError,
    HTTPStatusError,
    InternalServerError,
    NotFoundError,
    RequestError,
    ResponseError,
    ServerError,
    ServiceUnavailable,
    TooManyRequests,
    UnauthorizedError,
    from_urllib3,
    make_for_status,
)

from ._helpers import make_request, make_response


# ---------------------------------------------------------------------------
# make_for_status mapping
# ---------------------------------------------------------------------------


class TestMakeForStatusMapping:
    def test_2xx_returns_none(self):
        resp = make_response(status_code=200)
        assert make_for_status(resp) is None

    def test_3xx_returns_none(self):
        resp = make_response(status_code=302)
        assert make_for_status(resp) is None

    def test_400_to_bad_request(self):
        resp = make_response(status_code=400)
        assert isinstance(make_for_status(resp), BadRequest)

    def test_401_to_unauthorized(self):
        resp = make_response(status_code=401)
        assert isinstance(make_for_status(resp), UnauthorizedError)

    def test_403_to_forbidden(self):
        resp = make_response(status_code=403)
        assert isinstance(make_for_status(resp), ForbiddenError)

    def test_404_to_not_found(self):
        resp = make_response(status_code=404)
        assert isinstance(make_for_status(resp), NotFoundError)

    def test_429_to_too_many_requests(self):
        resp = make_response(status_code=429)
        err = make_for_status(resp)
        assert isinstance(err, TooManyRequests)

    def test_500_to_internal_server_error(self):
        resp = make_response(status_code=500)
        assert isinstance(make_for_status(resp), InternalServerError)

    def test_502_to_bad_gateway(self):
        assert isinstance(make_for_status(make_response(status_code=502)), BadGatewayError)

    def test_503_to_service_unavailable(self):
        assert isinstance(make_for_status(make_response(status_code=503)), ServiceUnavailable)

    def test_504_to_gateway_timeout(self):
        assert isinstance(make_for_status(make_response(status_code=504)), GatewayTimeout)

    def test_unknown_4xx_falls_back_to_client_error(self):
        err = make_for_status(make_response(status_code=418))
        assert isinstance(err, ClientError)
        assert not isinstance(err, NotFoundError)

    def test_unknown_5xx_falls_back_to_server_error(self):
        err = make_for_status(make_response(status_code=599))
        assert isinstance(err, ServerError)


# ---------------------------------------------------------------------------
# Retry-After parsing
# ---------------------------------------------------------------------------


class TestRetryAfterAttribute:
    def test_too_many_requests_with_retry_after(self):
        resp = make_response(status_code=429, headers={"Retry-After": "60"})
        err = make_for_status(resp)
        assert isinstance(err, TooManyRequests)
        assert err.retry_after == 60.0

    def test_too_many_requests_without_retry_after(self):
        err = make_for_status(make_response(status_code=429))
        assert isinstance(err, TooManyRequests)
        assert err.retry_after is None

    def test_service_unavailable_with_retry_after(self):
        resp = make_response(status_code=503, headers={"Retry-After": "30"})
        err = make_for_status(resp)
        assert isinstance(err, ServiceUnavailable)
        assert err.retry_after == 30.0


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


class TestHierarchy:
    def test_inherits_urllib3_root(self):
        err = HTTPError("boom")
        assert isinstance(err, _u3.HTTPError)

    def test_status_error_inherits_response_error(self):
        resp = make_response(status_code=500)
        err = make_for_status(resp)
        assert isinstance(err, ResponseError)
        assert isinstance(err, HTTPStatusError)

    def test_response_attached(self):
        resp = make_response(status_code=400)
        err = make_for_status(resp)
        assert err.response is resp


# ---------------------------------------------------------------------------
# from_urllib3 wrapping
# ---------------------------------------------------------------------------


class TestFromUrllib3:
    def test_request_bound_timeout(self):
        req = make_request()
        wrapped = from_urllib3(_u3.TimeoutError("slow"), request=req)
        assert isinstance(wrapped, RequestError)
        assert wrapped.request is req

    def test_connect_timeout_specific_class(self):
        req = make_request()
        wrapped = from_urllib3(_u3.ConnectTimeoutError("slow connect"), request=req)
        # NewConnectionError-style failures land here; ReadTimeout cannot be
        # cheaply constructed from outside, so only assert the connect path.
        assert isinstance(wrapped, ConnectTimeoutError)

    def test_response_bound_default(self):
        resp = make_response()
        wrapped = from_urllib3(_u3.HTTPError("oops"), response=resp)
        assert isinstance(wrapped, ResponseError)
        assert wrapped.response is resp


# ---------------------------------------------------------------------------
# CacheError
# ---------------------------------------------------------------------------


class TestCacheError:
    def test_carries_cause(self):
        original = RuntimeError("disk full")
        err = CacheError("write failed", cause=original)
        assert err.cause is original

    def test_message_in_args(self):
        err = CacheError("write failed")
        assert err.args[0] == "write failed"
