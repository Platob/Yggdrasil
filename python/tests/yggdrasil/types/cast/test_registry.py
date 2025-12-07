import dataclasses
import datetime
import enum
from typing import List, Set, Tuple, Dict, Any

import pyarrow as pa
import pytest

from yggdrasil.types.cast.registry import convert, convert_to_python_iterable


def test_builtin_converters():
    assert convert("1", int) == 1
    assert convert("", int | None) is 0
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


class Color(enum.Enum):
    RED = 1
    BLUE = 2


def test_enum_conversions():
    assert convert("RED", Color) is Color.RED
    assert convert("blue", Color) is Color.BLUE
    assert convert(1, Color) is Color.RED
    assert convert("2", Color) is Color.BLUE

    with pytest.raises(TypeError):
        convert("green", Color)


@dataclasses.dataclass
class Widget:
    name: str
    color: Color
    count: int = 1
    tags: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Container:
    widget: Widget
    enabled: bool


def test_dataclass_conversion():
    source = {
        "widget": {"name": "example", "color": "blue", "tags": ["x", "y"]},
        "enabled": "true",
    }

    result = convert(source, Container)

    assert isinstance(result, Container)
    assert result.widget == Widget(name="example", color=Color.BLUE, tags=["x", "y"])
    assert result.widget.count == 1
    assert result.enabled is True

    nested_dataclass = Container(widget=Widget(name="direct", color=Color.RED), enabled=False)
    assert convert(nested_dataclass, Container) is nested_dataclass

    with pytest.raises(TypeError):
        convert({}, Container)


@dataclasses.dataclass
class SimpleConfig:
    number: int
    label: str | None = None


@dataclasses.dataclass
class ConfigWrapper:
    config: SimpleConfig
    active: bool = False


def test_create_dataclass_from_dict_defaults():
    basic_result = convert({"number": "7"}, SimpleConfig)
    assert basic_result == SimpleConfig(number=7, label=None)

    wrapped_result = convert({}, ConfigWrapper)
    assert wrapped_result == ConfigWrapper(config=SimpleConfig(number=0, label=None), active=False)

    nested_override = convert({"config": {"number": "3", "label": "x"}}, ConfigWrapper)
    assert nested_override == ConfigWrapper(config=SimpleConfig(number=3, label="x"), active=False)

def test_non_iterable_raises_type_error():
    value = 42
    target_origin = list
    target_args = (int,)

    with pytest.raises(TypeError) as excinfo:
        convert_to_python_iterable(
            value=value,
            target_hint=List[int],
            target_origin=target_origin,
            target_args=target_args,
            options=None,
        )

    msg = str(excinfo.value)
    # Message shape: "Cannot convert <type> to <target_origin.__name__>"
    assert "int" in msg


def test_string_raises_type_error():
    value = "not allowed"
    target_origin = list
    target_args = (str,)

    with pytest.raises(TypeError) as excinfo:
        convert_to_python_iterable(
            value=value,
            target_hint=List[str],
            target_origin=target_origin,
            target_args=target_args,
            options=None,
        )

    msg = str(excinfo.value)
    assert "str" in msg


def test_bytes_raises_type_error():
    value = b"also not allowed"
    target_origin = list
    target_args = (bytes,)

    with pytest.raises(TypeError) as excinfo:
        convert_to_python_iterable(
            value=value,
            target_hint=List[bytes],
            target_origin=target_origin,
            target_args=target_args,
            options=None,
        )

    msg = str(excinfo.value)
    assert "bytes" in msg


def test_list_to_list_of_ints():
    value = ["1", "2", "3"]
    target_origin = list
    target_args = (int,)

    out = convert_to_python_iterable(
        value=value,
        target_hint=List[int],
        target_origin=target_origin,
        target_args=target_args,
        options=None,
    )

    assert isinstance(out, list)
    # relies on convert() respecting int as target type
    assert out == [1, 2, 3]


def test_list_to_set_of_strings():
    value = [1, 2, 2, 3]
    target_origin = set
    target_args = (str,)

    out = convert_to_python_iterable(
        value=value,
        target_hint=Set[str],
        target_origin=target_origin,
        target_args=target_args,
        options=None,
    )

    assert isinstance(out, set)
    # relies on convert() doing a sane str conversion
    assert out == {"1", "2", "3"}


def test_list_to_tuple_of_floats():
    value = ["1.0", "2.5"]
    target_origin = tuple
    target_args = (float,)

    out = convert_to_python_iterable(
        value=value,
        target_hint=Tuple[float],
        target_origin=target_origin,
        target_args=target_args,
        options=None,
    )

    assert isinstance(out, tuple)
    assert out == (1.0, 2.5)


def test_arrow_array_to_list_of_ints():
    arr = pa.array([1, 2, 3])
    target_origin = list
    target_args = (int,)

    out = convert_to_python_iterable(
        value=arr,
        target_hint=List[int],
        target_origin=target_origin,
        target_args=target_args,
        options=None,
    )

    # pa.Array.to_pylist() -> list of Python scalars, then converted
    assert isinstance(out, list)
    assert out == [1, 2, 3]
    assert all(isinstance(x, int) for x in out)


def test_arrow_array_to_set_of_strings():
    arr = pa.array([1, 2, 2, 3])
    target_origin = set
    target_args = (str,)

    out = convert_to_python_iterable(
        value=arr,
        target_hint=Set[str],
        target_origin=target_origin,
        target_args=target_args,
        options=None,
    )

    assert isinstance(out, set)
    assert out == {"1", "2", "3"}


def test_arrow_table_to_list_of_dicts():
    table = pa.table({"a": [1, 2], "b": ["x", "y"]})
    target_origin = list
    target_args = (dict,)

    out = convert_to_python_iterable(
        value=table,
        target_hint=List[Dict[str, Any]],
        target_origin=target_origin,
        target_args=target_args,
        options=None,
    )

    # pa.Table.to_pylist() -> list[dict]
    assert isinstance(out, list)
    assert out == [
        {"a": 1, "b": "x"},
        {"a": 2, "b": "y"},
    ]
    assert all(isinstance(row, dict) for row in out)


def test_arrow_recordbatch_to_list_of_dicts():
    batch = pa.record_batch({"a": [1, 2], "b": [True, False]})
    target_origin = list
    target_args = (dict,)

    out = convert_to_python_iterable(
        value=batch,
        target_hint=List[Dict[str, Any]],
        target_origin=target_origin,
        target_args=target_args,
        options=None,
    )

    assert isinstance(out, list)
    assert out == [
        {"a": 1, "b": True},
        {"a": 2, "b": False},
    ]
    assert all(isinstance(row, dict) for row in out)


def test_arrow_chunked_array_to_list():
    arr = pa.chunked_array([[1, 2], [3]])
    target_origin = list
    target_args = (int,)

    out = convert_to_python_iterable(
        value=arr,
        target_hint=List[int],
        target_origin=target_origin,
        target_args=target_args,
        options=None,
    )

    assert isinstance(out, list)
    assert out == [1, 2, 3]
    assert all(isinstance(x, int) for x in out)


def test_no_target_args_does_not_crash_and_returns_same_length():
    value = [1, "2", 3.0]
    target_origin = list
    target_args = [Any]  # empty -> element_hint defaults to Any

    out = convert_to_python_iterable(
        value=value,
        target_hint=List[Any],
        target_origin=target_origin,
        target_args=target_args,
        options=None,
    )

    # At minimum, it should not raise and should preserve length/order.
    assert isinstance(out, list)
    assert len(out) == len(value)


@dataclasses.dataclass
class Item:
    a: int
    b: str


def test_list_of_dataclass_instances():
    # incoming raw data that should map naturally onto the dataclass
    value = [
        {"a": 1, "b": "x"},
        {"a": 2, "b": "y"},
    ]

    out = convert(
        value=value,
        target_hint=List[Item],
    )

    # container type
    assert isinstance(out, list)
    assert len(out) == 2

    # element types should be the dataclass
    assert all(isinstance(elem, Item) for elem in out)

    # and values should be mapped correctly
    assert out[0] == Item(a=1, b="x")
    assert out[1] == Item(a=2, b="y")
