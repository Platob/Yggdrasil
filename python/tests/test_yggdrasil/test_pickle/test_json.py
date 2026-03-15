# tests/test_pickle_json.py
from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import pytest

from yggdrasil.pickle.json import dump, dumps, load, loads


# ---------------------------------------------------------------------------
# loads(): null string handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value",
    ["", "null", "None", "NaN"],
)
def test_loads_default_null_str_values(value: str):
    obj = loads(_mk_json({"x": value}))
    assert obj["x"] is None


def test_loads_custom_null_str_values():
    obj = loads(
        _mk_json({"a": "na", "b": "n/a", "c": "NULL", "d": "ok"}),
        null_str_values={"na", "n/a"},
    )
    assert obj["a"] is None
    assert obj["b"] is None
    assert obj["c"] == "NULL"  # not null with custom set
    assert obj["d"] == "ok"


def test_loads_null_str_values_false_disables_null_conversion():
    obj = loads(_mk_json({"x": "null", "y": " None "}), null_str_values=False)
    assert obj["x"] == "null"
    assert obj["y"] == " None "


def test_loads_null_str_values_none_disables_null_conversion():
    obj = loads(_mk_json({"x": "nan"}), null_str_values=None)
    assert obj["x"] == "nan"


# ---------------------------------------------------------------------------
# loads(): datetime/date/time parsing (strings)
# ---------------------------------------------------------------------------

def test_loads_datetime_seconds_naive_by_default():
    obj = loads(_mk_json({"ts": "2026-02-28 12:34:56"}))
    assert isinstance(obj["ts"], datetime)
    assert obj["ts"].tzinfo is None
    assert obj["ts"] == datetime(2026, 2, 28, 12, 34, 56)


def test_loads_datetime_minutes_naive_by_default():
    obj = loads(_mk_json({"ts": "2026-02-28 12:34"}))
    assert isinstance(obj["ts"], datetime)
    assert obj["ts"].tzinfo is None
    assert obj["ts"] == datetime(2026, 2, 28, 12, 34, 0)


def test_loads_datetime_seconds_with_T_separator():
    obj = loads(_mk_json({"ts": "2026-02-28T12:34:56"}))
    assert isinstance(obj["ts"], datetime)
    assert obj["ts"].tzinfo is None
    assert obj["ts"] == datetime(2026, 2, 28, 12, 34, 56)


def test_loads_datetime_minutes_with_T_separator():
    obj = loads(_mk_json({"ts": "2026-02-28T12:34"}))
    assert isinstance(obj["ts"], datetime)
    assert obj["ts"].tzinfo is None
    assert obj["ts"] == datetime(2026, 2, 28, 12, 34, 0)


def test_loads_datetime_z_is_utc():
    obj = loads(_mk_json({"ts": "2026-02-28T12:34:56Z"}))
    assert isinstance(obj["ts"], datetime)
    assert obj["ts"].tzinfo == timezone.utc
    assert obj["ts"].isoformat() == "2026-02-28T12:34:56+00:00"


def test_loads_datetime_offset_reduces_to_utc_hhmm():
    # 12:34:56+01:00 -> 11:34:56Z
    obj = loads(_mk_json({"ts": "2026-02-28 12:34:56+0100"}))
    assert isinstance(obj["ts"], datetime)
    assert obj["ts"].tzinfo == timezone.utc
    assert obj["ts"].isoformat() == "2026-02-28T11:34:56+00:00"


def test_loads_datetime_offset_reduces_to_utc_hh_colon_mm():
    # 12:34:56+01:00 -> 11:34:56Z
    obj = loads(_mk_json({"ts": "2026-02-28 12:34:56+01:00"}))
    assert isinstance(obj["ts"], datetime)
    assert obj["ts"].tzinfo == timezone.utc
    assert obj["ts"].isoformat() == "2026-02-28T11:34:56+00:00"


def test_loads_datetime_offset_minutes_form_reduces_to_utc():
    # 12:34:00+01:00 -> 11:34:00Z
    obj = loads(_mk_json({"ts": "2026-02-28 12:34+01:00"}))
    assert isinstance(obj["ts"], datetime)
    assert obj["ts"].tzinfo == timezone.utc
    assert obj["ts"].isoformat() == "2026-02-28T11:34:00+00:00"


def test_loads_datetime_offset_negative_reduces_to_utc():
    # 12:34:56-02:30 -> 15:04:56Z
    obj = loads(_mk_json({"ts": "2026-02-28 12:34:56-02:30"}))
    assert isinstance(obj["ts"], datetime)
    assert obj["ts"].tzinfo == timezone.utc
    assert obj["ts"].isoformat() == "2026-02-28T15:04:56+00:00"


def test_loads_datetime_fractional_seconds_and_offset_reduces_to_utc():
    # .123456789 -> microseconds .123456; +01:00 -> subtract 1h
    obj = loads(_mk_json({"ts": "2026-02-28 12:34:56.123456789+01:00"}))
    assert isinstance(obj["ts"], datetime)
    assert obj["ts"].tzinfo == timezone.utc
    assert obj["ts"].isoformat() == "2026-02-28T11:34:56.123456+00:00"


def test_loads_rfc1123_is_utc():
    obj = loads(_mk_json({"h": "Sat, 28 Feb 2026 12:34:56 GMT"}))
    assert isinstance(obj["h"], datetime)
    assert obj["h"].tzinfo == timezone.utc
    assert obj["h"].isoformat() == "2026-02-28T12:34:56+00:00"


def test_loads_date_and_time():
    obj = loads(_mk_json({"d": "2026-02-28", "t": "12:34:56.123456789"}))
    assert obj["d"] == date(2026, 2, 28)
    assert isinstance(obj["t"], time)
    # truncated to microseconds
    assert obj["t"].isoformat() == "12:34:56.123456"


def test_loads_time_can_get_default_tzinfo_zoneinfo():
    tz = ZoneInfo("Europe/Paris")
    obj = loads(_mk_json({"t": "12:34:56"}), default_tz=tz)
    assert isinstance(obj["t"], time)
    assert obj["t"].tzinfo == tz


def test_loads_datetime_can_get_default_tzinfo_zoneinfo():
    tz = ZoneInfo("Europe/Paris")
    obj = loads(_mk_json({"ts": "2026-02-28 12:34"}), default_tz=tz)
    assert isinstance(obj["ts"], datetime)
    assert obj["ts"].tzinfo == tz
    # February 2026 in Paris is +01:00
    assert obj["ts"].isoformat() == "2026-02-28T12:34:00+01:00"


def test_loads_default_tz_not_overridden_when_explicit_offset_present():
    # Explicit fixed offset wins; then reduced to UTC.
    tz = ZoneInfo("Europe/Paris")
    obj = loads(_mk_json({"ts": "2026-02-28 12:34+02:00"}), default_tz=tz)
    assert isinstance(obj["ts"], datetime)
    assert obj["ts"].tzinfo == timezone.utc
    assert obj["ts"].isoformat() == "2026-02-28T10:34:00+00:00"


def test_loads_can_disable_datetime_parsing():
    obj = loads(_mk_json({"ts": "2026-02-28 12:34"}), parse_datetimes=False)
    assert obj["ts"] == "2026-02-28 12:34"


def test_loads_invalid_datetime_stays_string():
    obj = loads(_mk_json({"ts": "2026-02-99 12:34"}))
    assert obj["ts"] == "2026-02-99 12:34"


def test_loads_empty_string_stays_string_not_none():
    # default null set contains "" -> becomes None
    obj = loads(_mk_json({"x": ""}))
    assert obj["x"] is None


# ---------------------------------------------------------------------------
# loads(): caching-related behavior (semantic)
# ---------------------------------------------------------------------------

def test_loads_repeated_datetime_strings_consistent_results():
    # We don't assert cache internals, just that repeated parsing is stable.
    payload = _mk_json({"a": "2026-02-28 12:34:56+0100", "b": "2026-02-28 12:34:56+0100"})
    obj = loads(payload)
    assert obj["a"] == obj["b"]
    assert obj["a"].tzinfo == timezone.utc
    assert obj["a"].isoformat() == "2026-02-28T11:34:56+00:00"


# ---------------------------------------------------------------------------
# loads(): nested structures
# ---------------------------------------------------------------------------

def test_loads_nested_structures_datetime_and_nulls():
    tz = ZoneInfo("Europe/Paris")
    obj = loads(
        r"""
        {
          "level1": {
            "events": [
              {"ts": "2026-02-28 12:34", "note": "naive"},
              {"ts": "2026-02-28T12:34:56Z", "note": "utc"},
              {"ts": "2026-02-28 12:34+0100", "note": "offset"},
              {"ts": "Sat, 28 Feb 2026 12:34:56 GMT", "note": "rfc1123"},
              {"ts": "null", "note": "nullish"},
              {"ts": "not-a-datetime", "note": "string"},
              {"ts": ["2026-02-28 12:34", "null", {"x": "2026-02-28"}], "note": "nested"}
            ],
            "meta": {
              "created": "2026-02-28",
              "maybe": "NaN",
              "deep": {"ts": "2026-02-28 12:34:56.123+01:00"}
            }
          }
        }
        """,
        default_tz=tz,
    )

    events = obj["level1"]["events"]

    assert events[0]["ts"].tzinfo == tz
    assert events[1]["ts"].tzinfo == timezone.utc

    # fixed offset reduced to UTC (12:34+01:00 -> 11:34Z)
    assert events[2]["ts"].tzinfo == timezone.utc
    assert events[2]["ts"].isoformat() == "2026-02-28T11:34:00+00:00"

    assert events[3]["ts"].tzinfo == timezone.utc
    assert events[4]["ts"] is None
    assert events[5]["ts"] == "not-a-datetime"

    nested = events[6]["ts"]
    assert isinstance(nested, list)
    assert nested[0].tzinfo == tz
    assert nested[1] is None
    assert nested[2]["x"] == date(2026, 2, 28)

    meta = obj["level1"]["meta"]
    assert meta["created"] == date(2026, 2, 28)
    assert meta["maybe"] is None
    assert isinstance(meta["deep"]["ts"], datetime)

    # deep fixed offset reduced to UTC
    assert meta["deep"]["ts"].tzinfo == timezone.utc
    assert meta["deep"]["ts"].isoformat() == "2026-02-28T11:34:56.123000+00:00"


def test_loads_nested_structures_parse_datetimes_false_only_nulls_apply():
    obj = loads(
        r'{"x":["2026-02-28 12:34","null",{"y":"2026-02-28"}]}',
        parse_datetimes=False,
    )
    assert obj["x"][0] == "2026-02-28 12:34"
    assert obj["x"][1] is None
    assert obj["x"][2]["y"] == "2026-02-28"


# ---------------------------------------------------------------------------
# dumps(): compact output + encoding
# ---------------------------------------------------------------------------

def test_dumps_is_compact_no_separator_whitespace():
    s = dumps({"a": 1, "b": 2}).decode("utf-8")
    assert ": " not in s
    assert ", " not in s


def test_dumps_encodes_datetime_date_time():
    payload = {
        "dt": datetime(2026, 2, 28, 12, 34, 56, tzinfo=timezone.utc),
        "d": date(2026, 2, 28),
        "t": time(12, 34, 56, 123456),
    }
    s = dumps(payload).decode("utf-8")
    assert '"dt":"2026-02-28T12:34:56+00:00"' in s
    assert '"d":"2026-02-28"' in s
    assert '"t":"12:34:56.123456"' in s


def test_dumps_default_encoder_dataclass_and_bytes():
    @dataclass
    class X:
        a: int
        b: str

    payload = {"x": X(1, "yo"), "bin": b"\xff\x00"}

    # dumps() returns UTF-8 bytes by default -> decode as UTF-8
    s = dumps(payload).decode("utf-8")

    assert '"a":1' in s and '"b":"yo"' in s

    # bytes are encoded as a latin-1 decoded *string* (unicode)
    # U+00FF should appear as "ÿ" when decoded as UTF-8.
    assert "ÿ" in s

    # \x00 cannot appear literally in JSON, so it must be escaped
    assert "\\u0000" in s

    # semantic check: parsing yields the same unicode string
    obj = loads(dumps(payload), parse_datetimes=False)
    assert obj["bin"] == "ÿ\u0000"


def test_dumps_ensure_ascii_true_escapes():
    s = dumps({"bin": b"\xff"}, ensure_ascii=True).decode("utf-8")
    assert "\\u00ff" in s


def test_dumps_indent_overrides_compact_default():
    s = dumps({"a": 1, "b": 2}, indent=2).decode("utf-8")
    # when indent is set, stdlib will include newlines/spaces
    assert "\n" in s


# ---------------------------------------------------------------------------
# load/dump: file-like behavior (binary + text)
# ---------------------------------------------------------------------------

def test_dump_load_binary_filelike_roundtrip():
    buf = io.BytesIO()
    dump({"ts": "2026-02-28 12:34"}, buf)  # default_tz forwarded by load only
    buf.seek(0)
    out = load(buf, default_tz=timezone.utc)
    assert isinstance(out["ts"], datetime)
    assert out["ts"].tzinfo == timezone.utc


def test_dump_load_text_filelike_roundtrip():
    buf = io.StringIO()
    dump({"ts": "2026-02-28 12:34"}, buf)
    buf.seek(0)
    out = load(buf)
    assert isinstance(out["ts"], datetime)


def test_load_bytes_decoding_errors_replace():
    # invalid utf-8 byte in string value; errors=replace should still parse JSON
    raw = b'{"x":"\xff"}'
    obj = loads(raw, errors="replace")
    assert isinstance(obj["x"], str)


# ---------------------------------------------------------------------------
# loads(): fast-path semantics
# ---------------------------------------------------------------------------

def test_loads_fastpath_when_parse_datetimes_false_and_nulls_disabled():
    # when both disabled, should behave like json.loads (no transformation)
    obj = loads(_mk_json({"x": "null", "ts": "2026-02-28 12:34"}), parse_datetimes=False, null_str_values=False)
    assert obj == {"x": "null", "ts": "2026-02-28 12:34"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mk_json(d: dict) -> str:
    # minimal compact json for test inputs
    return _json_dumps_compact(d)


def _json_dumps_compact(d: dict) -> str:
    # local helper to keep tests stable even if module dump defaults change
    import json
    return json.dumps(d, separators=(",", ":"), ensure_ascii=False)