# tests/test_logical_serialized.py
from __future__ import annotations

import io
import ipaddress
import uuid
import zipfile
from datetime import UTC, date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

# Adjust this import path to your real module location.
import yggdrasil.pickle.ser.logicals as m


def _roundtrip(obj):
    ser = m.LogicalSerialized.from_python_object(obj)
    assert ser is not None, f"Failed to serialize object: {obj!r}"
    return ser, ser.as_python()


def test_metadata_merge_none():
    assert m._metadata_merge(None, None) is None


def test_metadata_merge_combines_both():
    out = m._metadata_merge({b"a": b"1"}, {b"b": b"2"})
    assert out == {b"a": b"1", b"b": b"2"}


def test_metadata_helpers():
    meta = {b"x": b"abc", b"n": b"123"}
    assert m._metadata_bytes(meta, b"x") == b"abc"
    assert m._metadata_text(meta, b"x") == "abc"
    assert m._metadata_int(meta, b"n") == 123
    assert m._metadata_text(meta, b"missing", "dflt") == "dflt"
    assert m._metadata_int(meta, b"missing", 7) == 7


def test_pack_unpack_i32():
    raw = m._pack_i32(-123)
    assert m._unpack_i32(raw, tag_name="X") == -123


def test_pack_unpack_i64():
    raw = m._pack_i64(-1234567890123)
    assert m._unpack_i64(raw, tag_name="X") == -1234567890123


def test_pack_unpack_u64():
    raw = m._pack_u64(1234567890123)
    assert m._unpack_u64(raw, tag_name="X") == 1234567890123


def test_pack_unpack_f64_pair():
    raw = m._pack_f64_pair(1.5, -2.25)
    assert m._unpack_f64_pair(raw, tag_name="X") == (1.5, -2.25)


def test_pack_unpack_i64_triplet():
    raw = m._pack_i64_triplet(-10, 20, 30)
    assert m._unpack_i64_triplet(raw, tag_name="X") == (-10, 20, 30)


def test_unpack_i32_invalid_size():
    with pytest.raises(ValueError):
        m._unpack_i32(b"\x00", tag_name="X")


def test_unpack_i64_invalid_size():
    with pytest.raises(ValueError):
        m._unpack_i64(b"\x00", tag_name="X")


def test_unpack_u64_invalid_size():
    with pytest.raises(ValueError):
        m._unpack_u64(b"\x00", tag_name="X")


def test_unpack_f64_pair_invalid_size():
    with pytest.raises(ValueError):
        m._unpack_f64_pair(b"\x00", tag_name="X")


def test_unpack_i64_triplet_invalid_size():
    with pytest.raises(ValueError):
        m._unpack_i64_triplet(b"\x00", tag_name="X")


def test_unit_to_micros():
    assert m._unit_to_micros(2, m.U_S, tag_name="X") == 2_000_000
    assert m._unit_to_micros(2, m.U_MS, tag_name="X") == 2_000
    assert m._unit_to_micros(2, m.U_US, tag_name="X") == 2
    assert m._unit_to_micros(2, m.U_NS, tag_name="X") == 0


def test_unit_to_micros_invalid():
    with pytest.raises(ValueError):
        m._unit_to_micros(1, "bad", tag_name="X")


def test_datetime_from_epoch_units():
    assert m._datetime_from_epoch(1, m.U_S) == datetime(1970, 1, 1, 0, 0, 1, tzinfo=UTC)
    assert m._datetime_from_epoch(1_000, m.U_MS) == datetime(1970, 1, 1, 0, 0, 1, tzinfo=UTC)
    assert m._datetime_from_epoch(1_000_000, m.U_US) == datetime(1970, 1, 1, 0, 0, 1, tzinfo=UTC)
    assert m._datetime_from_epoch(1_000_000_000, m.U_NS) == datetime(1970, 1, 1, 0, 0, 1, tzinfo=UTC)


def test_timedelta_from_value_units():
    assert m._timedelta_from_value(2, m.U_S) == timedelta(seconds=2)
    assert m._timedelta_from_value(2_000, m.U_MS) == timedelta(seconds=2)
    assert m._timedelta_from_value(2_000_000, m.U_US) == timedelta(seconds=2)
    assert m._timedelta_from_value(2_000_000_000, m.U_NS) == timedelta(seconds=2)


def test_time_from_offset_us():
    t = m._time_from_offset(3_661_234_567, m.U_US)
    assert t == time(1, 1, 1, 234567)


def test_time_from_offset_negative():
    with pytest.raises(ValueError):
        m._time_from_offset(-1, m.U_US)


def test_time_from_offset_over_day():
    with pytest.raises(ValueError):
        m._time_from_offset(m.DAY_MICROS, m.U_US)


def test_tz_to_text_none():
    assert m._tz_to_text(None) is None


def test_tz_to_text_utc():
    assert m._tz_to_text(UTC) == "UTC"


def test_tz_to_text_fixed_offset():
    tz = timezone(timedelta(hours=5, minutes=30))
    assert m._tz_to_text(tz) == "+05:30"


def test_load_tzinfo_none():
    assert m._load_tzinfo(None) is None


def test_load_tzinfo_utc():
    tz = m._load_tzinfo({m.M_TZ: b"UTC"})
    assert tz is UTC


def test_load_tzinfo_fixed_offset():
    tz = m._load_tzinfo({m.M_TZ: b"+02:30"})
    assert tz is not None
    assert tz.utcoffset(None) == timedelta(hours=2, minutes=30)


def test_load_tzinfo_invalid():
    with pytest.raises(ValueError):
        m._load_tzinfo({m.M_TZ: b"definitely/not-a-zone"})


@pytest.mark.skipif(m.ZoneInfo is None, reason="zoneinfo unavailable")
def test_tz_to_text_zoneinfo():
    tz = m.ZoneInfo("Europe/Paris")
    assert m._tz_to_text(tz) == "Europe/Paris"


@pytest.mark.skipif(m.ZoneInfo is None, reason="zoneinfo unavailable")
def test_load_tzinfo_zoneinfo():
    tz = m._load_tzinfo({m.M_TZ: b"Europe/Paris"})
    assert tz is not None
    assert getattr(tz, "key", None) == "Europe/Paris"


def test_current_os_name():
    assert m._current_os_name() in (m.OS_POSIX, m.OS_WINDOWS)


def test_sanitize_path_part():
    assert m._sanitize_path_part("a/b\\c:d") == "a_b_c_d"
    assert m._sanitize_path_part("\x00") == "_"


def test_rebuild_path_same_os():
    raw = "tmp/example"
    p = m._rebuild_path(raw, source_os=m._current_os_name())
    assert isinstance(p, Path)
    assert str(p) == str(Path(raw))


def test_rebuild_path_windows_to_posix_absolute(monkeypatch):
    monkeypatch.setattr(m, "_current_os_name", lambda: m.OS_POSIX)
    p = m._rebuild_path(r"C:\tmp\x", source_os=m.OS_WINDOWS)
    assert p == Path("/c/tmp/x")


def test_rebuild_path_windows_to_posix_relative(monkeypatch):
    monkeypatch.setattr(m, "_current_os_name", lambda: m.OS_POSIX)
    p = m._rebuild_path(r"tmp\x", source_os=m.OS_WINDOWS)
    assert p == Path("tmp/x")


def test_rebuild_path_posix_to_windows_absolute(monkeypatch):
    monkeypatch.setattr(m, "_current_os_name", lambda: m.OS_WINDOWS)
    p = m._rebuild_path("/var/tmp/x", source_os=m.OS_POSIX)
    assert str(p) == str(Path(r"\var\tmp\x"))


def test_rebuild_path_posix_to_windows_relative(monkeypatch):
    monkeypatch.setattr(m, "_current_os_name", lambda: m.OS_WINDOWS)
    p = m._rebuild_path("var/tmp/x", source_os=m.OS_POSIX)
    assert str(p) == str(Path(r"var\tmp\x"))


def test_dir_total_bytes(tmp_path: Path):
    (tmp_path / "a.txt").write_bytes(b"abc")
    (tmp_path / "b.txt").write_bytes(b"12345")
    assert m._dir_total_bytes(tmp_path) == 8


def test_should_exclude_path_exact():
    assert m._should_exclude_path(("src", "__pycache__", "x.pyc")) is True
    assert m._should_exclude_path(("project", ".git", "config")) is True


def test_should_exclude_path_pattern():
    assert m._should_exclude_path(("pkg", "foo.pyc")) is True
    assert m._should_exclude_path(("dist", "bar.dist-info")) is True


def test_should_exclude_path_non_excluded():
    assert m._should_exclude_path(("src", "main.py")) is False


def test_zip_directory_to_bytes_excludes_junk(tmp_path: Path):
    (tmp_path / "keep.txt").write_text("ok")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.pyc").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "also_keep.txt").write_text("ok2")

    data = m._zip_directory_to_bytes(tmp_path)

    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        names = set(zf.namelist())

    assert "keep.txt" in names
    assert "sub/also_keep.txt" in names
    assert "__pycache__/junk.pyc" not in names


def test_extract_zip_bytes_to_temp_dir():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("a.txt", "hello")
        zf.writestr("sub/b.txt", "world")

    out = m._extract_zip_bytes_to_temp_dir(buf.getvalue(), dirname="demo")
    assert out.is_dir()
    assert (out / "a.txt").read_text() == "hello"
    assert (out / "sub" / "b.txt").read_text() == "world"


def test_write_file_bytes_to_temp_path():
    out = m._write_file_bytes_to_temp_path(b"abc", filename="x.txt")
    assert out.is_file()
    assert out.suffix == ".txt"
    assert out.read_bytes() == b"abc"


def test_decimal_roundtrip():
    ser, value = _roundtrip(Decimal("123.4500"))
    assert value == Decimal("123.4500")
    assert ser.tag == m.Tags.DECIMAL
    coefficient, precision, scale = m._unpack_i64_triplet(ser.decode(), tag_name="DECIMAL")
    assert coefficient == 1234500
    assert precision == 7
    assert scale == 4


def test_decimal_negative_roundtrip():
    _, value = _roundtrip(Decimal("-0.125"))
    assert value == Decimal("-0.125")


def test_datetime_roundtrip_naive():
    dt = datetime(2024, 1, 2, 3, 4, 5, 678901)
    _, value = _roundtrip(dt)
    assert value == dt.replace(tzinfo=UTC)


def test_datetime_roundtrip_tzaware():
    tz = timezone(timedelta(hours=2))
    dt = datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=tz)
    _, value = _roundtrip(dt)
    assert value == dt


@pytest.mark.skipif(m.ZoneInfo is None, reason="zoneinfo unavailable")
def test_datetime_roundtrip_zoneinfo():
    tz = m.ZoneInfo("Europe/Paris")
    dt = datetime(2024, 6, 1, 12, 0, 0, tzinfo=tz)
    _, value = _roundtrip(dt)
    assert value == dt


def test_date_roundtrip():
    _, value = _roundtrip(date(2024, 2, 29))
    assert value == date(2024, 2, 29)


def test_time_roundtrip_naive():
    _, value = _roundtrip(time(12, 34, 56, 123456))
    assert value == time(12, 34, 56, 123456)


def test_time_roundtrip_tzaware():
    t = time(12, 34, 56, 123456, tzinfo=timezone(timedelta(hours=-3)))
    _, value = _roundtrip(t)
    assert value == t


def test_timedelta_roundtrip():
    td = timedelta(days=2, seconds=3, microseconds=4)
    _, value = _roundtrip(td)
    assert value == td


def test_timezone_roundtrip_fixed():
    tz = timezone(timedelta(hours=5, minutes=45))
    _, value = _roundtrip(tz)
    assert value.utcoffset(None) == tz.utcoffset(None)


@pytest.mark.skipif(m.ZoneInfo is None, reason="zoneinfo unavailable")
def test_timezone_roundtrip_zoneinfo():
    tz = m.ZoneInfo("Europe/Paris")
    _, value = _roundtrip(tz)
    assert getattr(value, "key", None) == "Europe/Paris"


def test_uuid_roundtrip():
    u = uuid.uuid4()
    _, value = _roundtrip(u)
    assert value == u


def test_complex_roundtrip():
    _, value = _roundtrip(complex(1.25, -9.5))
    assert value == complex(1.25, -9.5)


def test_bytes_roundtrip():
    ser, value = _roundtrip(b"hello")
    assert value == b"hello"
    assert ser.metadata[m.M_KIND] == m.K_BYTES


def test_bytearray_roundtrip():
    ser, value = _roundtrip(bytearray(b"hello"))
    assert value == bytearray(b"hello")
    assert ser.metadata[m.M_KIND] == m.K_BYTEARRAY


def test_memoryview_roundtrip():
    ser, value = _roundtrip(memoryview(b"hello"))
    assert bytes(value) == b"hello"
    assert ser.metadata[m.M_KIND] == m.K_MEMORYVIEW


def test_ipaddress_roundtrip_ipv4_addr():
    obj = ipaddress.IPv4Address("1.2.3.4")
    _, value = _roundtrip(obj)
    assert value == obj


def test_ipaddress_roundtrip_ipv6_addr():
    obj = ipaddress.IPv6Address("2001:db8::1")
    _, value = _roundtrip(obj)
    assert value == obj


def test_ipaddress_roundtrip_ipv4_interface():
    obj = ipaddress.IPv4Interface("10.0.0.1/24")
    _, value = _roundtrip(obj)
    assert value == obj


def test_ipaddress_roundtrip_ipv6_interface():
    obj = ipaddress.IPv6Interface("2001:db8::1/64")
    _, value = _roundtrip(obj)
    assert value == obj


def test_ipaddress_roundtrip_ipv4_network():
    obj = ipaddress.IPv4Network("10.0.0.0/24")
    _, value = _roundtrip(obj)
    assert value == obj


def test_ipaddress_roundtrip_ipv6_network():
    obj = ipaddress.IPv6Network("2001:db8::/64")
    _, value = _roundtrip(obj)
    assert value == obj


def test_pure_windows_path_serializes_as_path_mode():
    obj = PureWindowsPath(r"C:\tmp\x.txt")
    ser = m.LogicalSerialized.from_python_object(obj)
    assert ser is not None
    assert ser.tag == m.Tags.PATH
    assert ser.metadata[m.M_PATH_MODE] == m.PATH_MODE_PATH
    assert ser.metadata[m.M_PATH_OS] == m.OS_WINDOWS
    assert ser.metadata[m.M_KIND] == m.K_PATH_WINDOWS


def test_pure_posix_path_serializes_as_path_mode():
    obj = PurePosixPath("/tmp/x.txt")
    ser = m.LogicalSerialized.from_python_object(obj)
    assert ser is not None
    assert ser.tag == m.Tags.PATH
    assert ser.metadata[m.M_PATH_MODE] == m.PATH_MODE_PATH
    assert ser.metadata[m.M_PATH_OS] == m.OS_POSIX
    assert ser.metadata[m.M_KIND] == m.K_PATH_POSIX


def test_path_roundtrip_plain_missing_path(tmp_path: Path):
    p = tmp_path / "missing.txt"
    ser = m.PathSerialized.from_value(p)
    assert ser.metadata[m.M_PATH_MODE] == m.PATH_MODE_PATH
    out = ser.as_python()
    assert isinstance(out, Path)
    assert str(out) == str(Path(str(p))).replace(str(Path.home()), "~")


def test_path_roundtrip_file_mode(tmp_path: Path):
    p = tmp_path / "data.bin"
    p.write_bytes(b"payload")

    ser = m.PathSerialized.from_value(p)
    assert ser.metadata[m.M_PATH_MODE] == m.PATH_MODE_FILE
    assert ser.metadata[m.M_FILENAME] == b"data.bin"

    out = ser.as_python()
    assert out.is_file()
    assert out.read_bytes() == b"payload"
    assert out.suffix == ".bin"


def test_path_roundtrip_small_dir_mode(tmp_path: Path):
    d = tmp_path / "demo"
    d.mkdir()
    (d / "a.txt").write_text("A")
    (d / "sub").mkdir()
    (d / "sub" / "b.txt").write_text("B")

    ser = m.PathSerialized.from_value(d)
    assert ser.metadata[m.M_PATH_MODE] == m.PATH_MODE_DIR_ZIP
    assert ser.metadata[m.M_DIRNAME] == b"demo"

    out = ser.as_python()
    assert out.is_dir()
    assert (out / "a.txt").read_text() == "A"
    assert (out / "sub" / "b.txt").read_text() == "B"


def test_path_roundtrip_large_dir_falls_back_to_path_mode(tmp_path: Path, monkeypatch):
    d = tmp_path / "bigdir"
    d.mkdir()
    (d / "x.txt").write_text("x")

    monkeypatch.setattr(m, "MAX_INLINE_DIR_BYTES", 1)

    ser = m.PathSerialized.from_value(d)
    assert ser.metadata[m.M_PATH_MODE] == m.PATH_MODE_PATH
    out = ser.as_python()
    assert isinstance(out, Path)
    assert str(out) == str(Path(str(d))).replace(str(Path.home()), "~")


def test_path_dir_mode_skips_excluded_entries(tmp_path: Path):
    d = tmp_path / "demo"
    d.mkdir()
    (d / "keep.txt").write_text("K")
    (d / "__pycache__").mkdir()
    (d / "__pycache__" / "x.pyc").write_bytes(b"x")

    ser = m.PathSerialized.from_value(d)
    assert ser.metadata[m.M_PATH_MODE] == m.PATH_MODE_DIR_ZIP

    out = ser.as_python()
    assert (out / "keep.txt").exists()
    assert not (out / "__pycache__" / "x.pyc").exists()


def test_path_value_invalid_mode():
    ser = m.PathSerialized.build(
        tag=m.Tags.PATH,
        data=b"abc",
        metadata={m.M_PATH_MODE: b"bad"},
        codec=None,
    )
    with pytest.raises(ValueError):
        _ = ser.as_python()


def test_timezone_value_invalid_kind():
    ser = m.TimezoneSerialized.build(
        tag=m.Tags.TIMEZONE,
        data=b"UTC",
        metadata={m.M_KIND: b"bad"},
        codec=None,
    )
    with pytest.raises(ValueError):
        _ = ser.as_python()


def test_ipaddress_value_invalid_kind():
    ser = m.IPAddressSerialized.build(
        tag=m.Tags.IPADDRESS,
        data=b"1.2.3.4",
        metadata={m.M_KIND: b"bad"},
        codec=None,
    )
    with pytest.raises(ValueError):
        _ = ser.as_python()


def test_logical_from_python_object_unknown_type():
    class Weird:
        pass

    assert m.LogicalSerialized.from_python_object(Weird()) is None