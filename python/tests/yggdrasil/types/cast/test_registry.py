import datetime

import pytest

from yggdrasil.types.cast import convert, register


def test_builtin_converters():
    assert convert("1", int) == 1
    assert convert("", int | None) is None
    assert convert("2025-12-10", datetime.date) == datetime.date(2025, 12, 10)
    assert convert("2025-12-10T07:08:09", datetime.datetime) == datetime.datetime(
        2025, 12, 10, 7, 8, 9, tzinfo=datetime.timezone.utc
    )
    assert convert("2025-12-10T07:08:09Z", datetime.datetime) == datetime.datetime(
        2025, 12, 10, 7, 8, 9, tzinfo=datetime.timezone.utc
    )
    assert convert("2025-12-10T07:08:09+02:00", datetime.datetime) == datetime.datetime(
        2025, 12, 10, 7, 8, 9, tzinfo=datetime.timezone(datetime.timedelta(hours=2))
    )
    assert convert("2025-12-10T07:08:09.123456789Z", datetime.datetime) == datetime.datetime(
        2025, 12, 10, 7, 8, 9, 123456, tzinfo=datetime.timezone.utc
    )
    assert convert("2025-12-10T07:08:09.1+02:00", datetime.datetime) == datetime.datetime(
        2025, 12, 10, 7, 8, 9, 100000, tzinfo=datetime.timezone(datetime.timedelta(hours=2))
    )
    assert convert("2025-12-10 07:08:09", datetime.datetime) == datetime.datetime(
        2025, 12, 10, 7, 8, 9, tzinfo=datetime.timezone.utc
    )
    assert convert("2025-12-10", datetime.datetime) == datetime.datetime(
        2025, 12, 10, 0, 0, tzinfo=datetime.timezone.utc
    )
    assert convert("2025-12-10 07:08:09+0200", datetime.datetime) == datetime.datetime(
        2025, 12, 10, 7, 8, 9, tzinfo=datetime.timezone(datetime.timedelta(hours=2))
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    converted_now = convert("now", datetime.datetime)
    assert converted_now.tzinfo == datetime.timezone.utc
    assert abs((converted_now - now).total_seconds()) < 1
    assert convert("07:08:09", datetime.time) == datetime.time(7, 8, 9)
    assert convert("3.14", float) == 3.14
    assert convert("true", bool) is True
    assert convert("FALSE", bool) is False


def test_custom_registration():
    @register(int, str)
    def _int_to_str(value, cast_options, default_value):
        return f"val={value}"

    assert convert(3, str) == "val=3"

    with pytest.raises(TypeError):
        convert(1.2, str)


def test_collection_conversions():
    assert convert(["1", "2"], list[int]) == [1, 2]
    assert convert({"a", "b"}, set[str]) == {"a", "b"}
    assert convert(("1", "2"), tuple[int, ...]) == (1, 2)
    with pytest.raises(TypeError):
        convert(("1", "2"), tuple[int, str, float])

    result = convert({"a": "1", "b": "2"}, dict[str, int])
    assert result == {"a": 1, "b": 2}
    nested = convert({"numbers": ["1", "2"]}, dict[str, list[int]])
    assert nested == {"numbers": [1, 2]}
