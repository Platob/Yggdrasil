# tests/test_pickle_json.py
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timezone

import pytest

from yggdrasil.pickle.json import dump, dumps, load, loads


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mk(d: dict) -> str:
    return json.dumps(d, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# loads() – basic decoding
# ---------------------------------------------------------------------------

def test_loads_str():
    assert loads('{"x": 1}') == {"x": 1}


def test_loads_bytes():
    assert loads(b'{"x": 1}') == {"x": 1}


def test_loads_bytearray():
    assert loads(bytearray(b'{"x": 1}')) == {"x": 1}


def test_loads_memoryview():
    assert loads(memoryview(b'{"x": 1}')) == {"x": 1}


def test_loads_encoding_latin1():
    raw = '{"v": "caf\u00e9"}'.encode("latin-1")
    obj = loads(raw, encoding="latin-1")
    assert obj["v"] == "café"


def test_loads_errors_replace_on_bad_utf8():
    raw = b'{"x":"\xff"}'
    obj = loads(raw, errors="replace")
    assert isinstance(obj["x"], str)


def test_loads_list():
    assert loads("[1,2,3]") == [1, 2, 3]


def test_loads_null():
    assert loads("null") is None


def test_loads_nested():
    obj = loads('{"a":{"b":[1,2]}}')
    assert obj == {"a": {"b": [1, 2]}}


def test_loads_strings_stay_strings():
    # Strings that look like datetimes or nulls are NOT auto-converted
    obj = loads(_mk({"ts": "2026-02-28 12:34:56", "x": "null", "y": ""}))
    assert obj["ts"] == "2026-02-28 12:34:56"
    assert obj["x"] == "null"
    assert obj["y"] == ""


# ---------------------------------------------------------------------------
# dumps() – serialisation
# ---------------------------------------------------------------------------

def test_dumps_returns_bytes():
    assert isinstance(dumps({"a": 1}), bytes)


def test_dumps_compact_default():
    s = dumps({"a": 1, "b": 2}).decode()
    assert ": " not in s
    assert ", " not in s


def test_dumps_indent_overrides_compact():
    s = dumps({"a": 1}, indent=2).decode()
    assert "\n" in s


def test_dumps_sort_keys():
    s = dumps({"b": 2, "a": 1}, sort_keys=True).decode()
    assert s.index('"a"') < s.index('"b"')


def test_dumps_ensure_ascii_escapes_non_ascii():
    s = dumps({"k": "é"}, ensure_ascii=True).decode()
    assert "\\u00e9" in s


def test_dumps_ensure_ascii_false_keeps_unicode():
    s = dumps({"k": "é"}, ensure_ascii=False).decode()
    assert "é" in s


def test_dumps_datetime_isoformat():
    s = dumps({"dt": datetime(2026, 2, 28, 12, 34, 56, tzinfo=timezone.utc)}).decode()
    assert '"dt":"2026-02-28T12:34:56+00:00"' in s


def test_dumps_date_isoformat():
    s = dumps({"d": date(2026, 2, 28)}).decode()
    assert '"d":"2026-02-28"' in s


def test_dumps_time_isoformat():
    s = dumps({"t": time(12, 34, 56, 123456)}).decode()
    assert '"t":"12:34:56.123456"' in s


def test_dumps_bytes_utf8():
    s = dumps({"b": b"hello"}).decode()
    assert '"b":"hello"' in s


def test_dumps_bytes_non_utf8_raises():
    with pytest.raises(TypeError, match="not JSON serializable"):
        dumps({"b": b"\xff\xfe"})


def test_dumps_dataclass():
    @dataclass
    class Point:
        x: int
        y: int

    s = dumps(Point(1, 2)).decode()
    assert '"x":1' in s
    assert '"y":2' in s


def test_dumps_unsupported_type_raises():
    with pytest.raises(TypeError, match="not JSON serializable"):
        dumps({"v": object()})


def test_dumps_custom_default():
    s = dumps({"v": 42}, default=lambda o: str(o)).decode()
    # default is only called for non-standard types; 42 is an int, handled natively
    assert '"v":42' in s


def test_dumps_encoding_latin1():
    b = dumps({"k": "café"}, encoding="latin-1")
    assert b.decode("latin-1") == '{"k":"café"}'


# ---------------------------------------------------------------------------
# load() / dump() – file-like objects
# ---------------------------------------------------------------------------

def test_dump_load_binary_roundtrip():
    buf = io.BytesIO()
    dump({"x": 1, "y": "hello"}, buf)
    buf.seek(0)
    assert load(buf) == {"x": 1, "y": "hello"}


def test_dump_load_text_roundtrip():
    buf = io.StringIO()
    dump({"a": [1, 2, 3]}, buf)
    buf.seek(0)
    assert load(buf) == {"a": [1, 2, 3]}


def test_dump_load_preserves_datetime_as_isostring():
    # dump encodes datetime as ISO string; load returns it as a plain string
    buf = io.BytesIO()
    dump({"dt": datetime(2026, 2, 28, tzinfo=timezone.utc)}, buf)
    buf.seek(0)
    obj = load(buf)
    assert obj["dt"] == "2026-02-28T00:00:00+00:00"


def test_load_encoding_kwarg():
    raw = '{"v": "caf\u00e9"}'.encode("latin-1")
    buf = io.BytesIO(raw)
    obj = load(buf, encoding="latin-1")
    assert obj["v"] == "café"


def test_dump_dataclass_to_file():
    @dataclass
    class Config:
        name: str
        value: int

    buf = io.BytesIO()
    dump(Config("foo", 42), buf)
    buf.seek(0)
    obj = load(buf)
    assert obj == {"name": "foo", "value": 42}
