"""`Response.ok` / `error` / `raise_for_status` / `warn_for_status`."""

from __future__ import annotations

import datetime as dt

import pytest

from .._helpers import make_request, make_response


def test_ok_true_for_2xx_and_error_is_none() -> None:
    resp = make_response(status_code=204, body=b"")
    assert resp.ok is True
    assert resp.error() is None


def test_ok_false_for_5xx_and_error_is_set() -> None:
    resp = make_response(status_code=500, body=b"")
    assert resp.ok is False
    assert resp.error() is not None


def test_raise_for_status_raises_on_5xx() -> None:
    resp = make_response(status_code=500, body=b"")
    with pytest.raises(Exception):
        resp.raise_for_status()


def test_warn_for_status_emits_runtime_warning_on_5xx() -> None:
    resp = make_response(status_code=500, body=b"")
    with pytest.warns(RuntimeWarning):
        resp.warn_for_status()


def test_received_at_is_epoch_microseconds() -> None:
    resp = make_response(request=make_request(), received_at=42)
    assert resp.received_at == dt.datetime(1970, 1, 1, 0, 0, 42, tzinfo=dt.timezone.utc)
