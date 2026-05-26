"""Pickle-protocol tests for :class:`PreparedRequest` and :class:`Response`.

Mirrors the :class:`Session` pickle contract:

* every non-transient field survives a round-trip,
* the transient ``_session`` back-reference is excluded by
  ``__getstate__`` and reset to ``None`` by ``__setstate__`` so a
  request / response can be re-bound on the worker via
  :meth:`attach_session`,
* :class:`Response` (slotted) walks the MRO so subclasses with their
  own ``__slots__ = ()`` (notably :class:`HTTPResponse`) emit the
  inherited fields instead of an empty payload.
"""
from __future__ import annotations

import datetime as dt
import pickle

import pytest

from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.http_.response import HTTPResponse
from yggdrasil.http_.session import HTTPSession
from yggdrasil.http_.request import PreparedRequest
from yggdrasil.http_.response import Response
<<<<<<< HEAD
from yggdrasil.http_.io_session import Session
=======
from yggdrasil.http_.session import Session
>>>>>>> 7d53e95


@pytest.fixture(autouse=True)
def _clear_session_singleton_cache():
    Session._INSTANCES.clear()
    yield
    Session._INSTANCES.clear()


def _make_request() -> PreparedRequest:
    return PreparedRequest.prepare(
        method="POST",
        url="https://example.com/path?q=1",
        headers={"X-Test": "1", "Content-Type": "application/json"},
        body=b'{"k":"v"}',
        tags={"env": "dev"},
    )


def _make_response(request: PreparedRequest, *, cls: type[Response] = Response) -> Response:
    return cls(
        request=request,
        status_code=201,
        headers={"Content-Type": "application/json"},
        tags={"env": "dev"},
        buffer=BytesIO(b'{"ok":true}', copy=False),
        received_at=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
    )


# ---------------------------------------------------------------------------
# PreparedRequest
# ---------------------------------------------------------------------------


class TestPreparedRequestPickle:

    def test_session_excluded_from_state(self) -> None:
        req = _make_request()
        req.attach_session(HTTPSession(base_url="https://example.com"))
        assert req._session is not None
        state = req.__getstate__()
        assert "_session" not in state

    def test_dict_attrs_preserved(self) -> None:
        req = _make_request()
        state = req.__getstate__()
        # All declared __dict__ fields land in the payload
        for key in (
            "method", "url", "headers", "tags", "buffer",
            "sent_at", "_sender",
        ):
            assert key in state, f"missing {key!r} in pickle state"

    def test_round_trip_preserves_fields(self) -> None:
        req = _make_request()
        req.attach_session(HTTPSession(base_url="https://example.com"))
        clone = pickle.loads(pickle.dumps(req))
        assert clone.method == "POST"
        assert clone.url.host == "example.com"
        assert clone.url.query == "q=1"
        assert clone.headers.get("X-Test") == "1"
        assert clone.tags.get("env") == "dev"
        assert clone.buffer.to_bytes() == b'{"k":"v"}'
        # Transient session reset to None for re-binding on the worker
        assert clone._session is None

    def test_transient_set_includes_session(self) -> None:
        assert "_session" in PreparedRequest._TRANSIENT_STATE_ATTRS


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class TestResponsePickle:

    def test_session_excluded_from_state(self) -> None:
        r = _make_response(_make_request())
        r.attach_session(HTTPSession(base_url="https://example.com"))
        state = r.__getstate__()
        assert "_session" not in state

    def test_slot_attrs_preserved(self) -> None:
        r = _make_response(_make_request())
        state = r.__getstate__()
        for key in (
            "request", "status_code", "headers", "tags",
            "buffer", "received_at", "_receiver",
        ):
            assert key in state, f"missing slot {key!r} in pickle state"

    def test_round_trip_preserves_fields(self) -> None:
        r = _make_response(_make_request())
        r.attach_session(HTTPSession(base_url="https://example.com"))
        clone = pickle.loads(pickle.dumps(r))
        assert clone.status_code == 201
        assert clone.headers.get("Content-Type") == "application/json"
        assert clone.tags.get("env") == "dev"
        assert clone.buffer.to_bytes() == b'{"ok":true}'
        assert clone.request.url.host == "example.com"
        assert clone._session is None

    def test_transient_set_includes_session(self) -> None:
        assert "_session" in Response._TRANSIENT_STATE_ATTRS


# ---------------------------------------------------------------------------
# HTTPResponse — subclass with empty __slots__ used to round-trip empty
# ---------------------------------------------------------------------------


class TestHTTPResponsePickle:

    def test_state_walks_mro_for_inherited_slots(self) -> None:
        # Regression: ``self.__slots__`` on an HTTPResponse instance is
        # ``()``, so the old getstate emitted an empty dict and unpickle
        # raised AttributeError on ``buffer``. The MRO-aware walk picks
        # up the parent's slot fields.
        hr = _make_response(_make_request(), cls=HTTPResponse)
        state = hr.__getstate__()
        assert "buffer" in state
        assert "request" in state
        assert "status_code" in state

    def test_round_trip_preserves_fields(self) -> None:
        hr = _make_response(_make_request(), cls=HTTPResponse)
        clone = pickle.loads(pickle.dumps(hr))
        assert isinstance(clone, HTTPResponse)
        assert clone.status_code == 201
        assert clone.buffer.to_bytes() == b'{"ok":true}'
        assert clone.request.url.host == "example.com"
        assert clone._session is None
