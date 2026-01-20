# test_python_defaults.py
import dataclasses
import datetime
import decimal
import typing
import uuid

import pyarrow as pa
import pytest

from yggdrasil.types.python_defaults import (
    default_arrow_array,
    default_arrow_scalar,
    default_python_scalar,
    default_scalar,
)


def assert_arrow_scalar(s: pa.Scalar, expected_py, expected_type: pa.DataType):
    assert isinstance(s, pa.Scalar)
    assert s.type == expected_type
    assert s.as_py() == expected_py


@pytest.mark.parametrize(
    "hint, expected",
    [
        (str, ""),
        (int, 0),
        (float, 0.0),
        (bool, False),
        (bytes, b""),
    ],
)
def test_default_python_scalar_primitives(hint, expected):
    assert default_python_scalar(hint) == expected


def test_default_python_scalar_optional():
    assert default_python_scalar(typing.Optional[int]) is None
    assert default_python_scalar(int | None) is None  # py>=3.10 union syntax


def test_default_python_scalar_specials():
    assert default_python_scalar(datetime.timedelta) == datetime.timedelta(0)
    assert default_python_scalar(decimal.Decimal) == decimal.Decimal(0)
    assert default_python_scalar(uuid.UUID) == uuid.UUID(int=0)

    dt = default_python_scalar(datetime.datetime)
    assert dt == datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)

    d = default_python_scalar(datetime.date)
    assert d == datetime.date(1970, 1, 1)

    t = default_python_scalar(datetime.time)
    assert t == datetime.time(0, 0, 0, tzinfo=datetime.timezone.utc)


def test_default_python_scalar_collections():
    assert default_python_scalar(list[int]) == []
    assert default_python_scalar(set[str]) == set()
    assert default_python_scalar(dict[str, int]) == {}

    assert default_python_scalar(tuple) == tuple()
    assert default_python_scalar(tuple[int, str]) == (0, "")
    assert default_python_scalar(tuple[int, ...]) == tuple()


@dataclasses.dataclass
class _DC:
    a: int
    b: typing.Optional[str]
    c: list[int] = dataclasses.field(default_factory=list)
    _hidden: int = dataclasses.field(default=123, init=False)
    d: int = dataclasses.field(init=False, default=99)


def test_default_python_scalar_dataclass():
    v = default_python_scalar(_DC)
    assert isinstance(v, _DC)
    assert v.a == 0
    assert v.b is None
    assert v.c == []
    assert v._hidden == 123
    assert v.d == 99


def test_default_python_scalar_falls_back_to_constructor():
    class Foo:
        def __init__(self):
            self.x = 1

    v = default_python_scalar(Foo)
    assert isinstance(v, Foo)
    assert v.x == 1


def test_default_python_scalar_constructor_error_raises_typeerror():
    class NeedsArg:
        def __init__(self, x):
            self.x = x

    with pytest.raises(TypeError):
        default_python_scalar(NeedsArg)


def test_default_python_scalar_non_class_goes_through_convert_without_mocking():
    """
    Exercises this branch without mocking:

        origin is None and not inspect.isclass(hint):
            from .cast import convert
            arrow_field: pa.Field = convert(hint, pa.Field)
            arrow_scalar = default_arrow_scalar(dtype=arrow_field.type, nullable=arrow_field.nullable)
            return arrow_scalar.as_py()

    We compute the expected value from the actual convert() result so the test
    stays correct even if your converter chooses nullable=True for Polars dtypes.
    """
    polars = pytest.importorskip("polars")

    # Non-class hint -> should hit the convert() branch
    hint = polars.Int32()

    # Use real converter to determine expected
    from yggdrasil.types.cast import convert

    arrow_field = convert(hint, pa.Field)
    assert isinstance(arrow_field, pa.Field)

    out = default_python_scalar(hint)

    if arrow_field.nullable:
        assert out is None
    else:
        assert out == 0


def test_default_scalar_routes_python_vs_arrow():
    assert default_scalar(int) == 0

    f = pa.field("x", pa.int64(), nullable=False)
    s = default_scalar(f)
    assert_arrow_scalar(s, 0, pa.int64())

    dt = pa.string()
    s2 = default_scalar(dt, nullable=False)
    assert_arrow_scalar(s2, "", pa.string())


def test_default_arrow_scalar_nullable_always_none():
    dt = pa.int32()
    s = default_arrow_scalar(dt, nullable=True)
    assert_arrow_scalar(s, None, dt)


@pytest.mark.parametrize(
    "dt, expected_py",
    [
        (pa.null(), None),
        (pa.bool_(), False),
        (pa.int8(), 0),
        (pa.int16(), 0),
        (pa.int32(), 0),
        (pa.int64(), 0),
        (pa.uint8(), 0),
        (pa.uint16(), 0),
        (pa.uint32(), 0),
        (pa.uint64(), 0),
        (pa.float32(), 0.0),
        (pa.float64(), 0.0),
        (pa.string(), ""),
        (pa.string_view(), ""),
        (pa.large_string(), ""),
        (pa.binary(), b""),
        (pa.binary_view(), b""),
        (pa.large_binary(), b""),
    ],
)
def test_default_arrow_scalar_known_defaults(dt, expected_py):
    s = default_arrow_scalar(dt, nullable=False)
    assert_arrow_scalar(s, expected_py, dt)


def test_default_arrow_scalar_timestamp_date_time_duration():
    # This branch returns pa.scalar(0, type=dtype), but .as_py() is NOT 0 for these dtypes.
    # It becomes a datetime/date/time/timedelta depending on dtype and pyarrow version/build.
    dtypes = [
        pa.timestamp("us"),      # may become datetime.datetime(1970-01-01 ...)
        pa.time32("s"),          # may become datetime.time(00:00:00)
        pa.time64("ns"),         # may become datetime.time(00:00:00)
        pa.date32(),             # may become datetime.date(1970-01-01)
        pa.date64(),             # may become datetime.date(...) or datetime.datetime(...) depending on version
        pa.duration("ms"),       # may become datetime.timedelta(0)
    ]

    for dt in dtypes:
        s = default_arrow_scalar(dt, nullable=False)
        assert s.type == dt

        # Compare to PyArrow’s own canonical scalar(0) for this dtype.
        expected = pa.scalar(0, type=dt)

        # Strong check: same logical scalar value (avoids repr / tz drama)
        assert s.equals(expected)

        # Also check python conversion matches
        assert s.as_py() == expected.as_py()


def test_default_arrow_scalar_decimal():
    dt = pa.decimal128(10, 2)
    s = default_arrow_scalar(dt, nullable=False)
    assert s.type == dt
    assert s.as_py() == decimal.Decimal(0)


def test_default_arrow_scalar_fixed_size_binary():
    dt = pa.binary(4)
    s = default_arrow_scalar(dt, nullable=False)
    assert s.type == dt
    assert s.as_py() == b"\x00" * 4


def test_default_arrow_scalar_struct():
    dt = pa.struct(
        [
            pa.field("a", pa.int32(), nullable=False),
            pa.field("b", pa.string(), nullable=True),
        ]
    )
    s = default_arrow_scalar(dt, nullable=False)
    assert s.type == dt
    assert s["a"].as_py() == 0
    assert s["b"].as_py() is None


def test_default_arrow_scalar_lists_and_map():
    # list / large_list
    for dt in [pa.list_(pa.int32()), pa.large_list(pa.int32())]:
        s = default_arrow_scalar(dt, nullable=False)
        assert s.type == dt
        assert s.as_py() == []

    # list_view (if supported by current pyarrow)
    if hasattr(pa, "list_view"):
        dt = pa.list_view(pa.int32())
        s = default_arrow_scalar(dt, nullable=False)
        assert s.type == dt
        assert s.as_py() == []

    # map: Arrow maps convert to list[tuple[key, value]] in Python, not dict.
    dt = pa.map_(pa.string(), pa.int32())
    s = default_arrow_scalar(dt, nullable=False)
    assert s.type == dt
    assert s.as_py() == []  # empty map -> empty list of pairs



def test_default_arrow_scalar_fixed_size_list():
    # Some pyarrow versions don't expose pa.fixed_size_list, but you can often build
    # a FixedSizeListType via pa.list_(..., list_size=N).
    if hasattr(pa, "fixed_size_list"):
        dt = pa.fixed_size_list(pa.field("item", pa.int8(), nullable=False), 3)
    else:
        # Prefer field form if supported; fall back to value_type form.
        try:
            dt = pa.list_(pa.field("item", pa.int8(), nullable=False), list_size=3)
        except TypeError:
            dt = pa.list_(pa.int8(), list_size=3)

    s = default_arrow_scalar(dt, nullable=False)
    assert s.type == dt

    py = s.as_py()
    assert isinstance(py, list)
    assert len(py) == 3
    assert py == [0, 0, 0]


def test_default_arrow_scalar_unknown_type_raises():
    # Not handled by default_arrow_scalar -> should raise
    dt = pa.dictionary(pa.int8(), pa.string())
    with pytest.raises(TypeError):
        default_arrow_scalar(dt, nullable=False)


def test_default_arrow_array_repeat_size():
    dt = pa.int32()
    arr = default_arrow_array(dt, nullable=False, size=4)
    assert isinstance(arr, pa.Array)
    assert arr.type == dt
    assert arr.to_pylist() == [0, 0, 0, 0]


def test_default_arrow_array_repeat_nullable_size():
    dt = pa.string()
    arr = default_arrow_array(dt, nullable=True, size=3)
    assert isinstance(arr, pa.Array)
    assert arr.type == dt
    assert arr.to_pylist() == [None, None, None]


def test_default_arrow_array_chunked():
    dt = pa.int32()
    out = default_arrow_array(dt, nullable=False, chunks=[2, 1])
    assert isinstance(out, pa.ChunkedArray)
    assert out.type == dt
    assert out.length() == 3
    assert out.to_pylist() == [0, 0, 0]


def test_default_arrow_array_scalar_default_override():
    dt = pa.int32()
    out = default_arrow_array(dt, nullable=False, size=3, scalar_default=pa.scalar(5, type=dt))
    assert isinstance(out, pa.Array)
    assert out.to_pylist() == [5, 5, 5]
