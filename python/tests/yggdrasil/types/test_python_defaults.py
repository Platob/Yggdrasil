import dataclasses
import datetime
import decimal
import uuid
from typing import Optional, Tuple, List, Dict, Set

import pyarrow as pa
import pytest

from yggdrasil.types import python_defaults as pd


# -----------------------------
# Helpers & sample dataclasses
# -----------------------------


@dataclasses.dataclass
class SimpleDataClass:
    a: int
    b: str = "x"
    _internal: int = 1
    c: int = dataclasses.field(default_factory=lambda: 5)
    d: int = dataclasses.field(init=False, default=10)


class CustomNoArg:
    def __init__(self):
        self.value = 42


class CustomNeedsArg:
    def __init__(self, x):
        self.x = x


# -----------------------------
# _is_optional
# -----------------------------


def test_is_optional_true_for_optional_and_pep604_union():
    assert pd._is_optional(Optional[int]) is True
    assert pd._is_optional(int | None) is True  # PEP 604 syntax


def test_is_optional_false_for_plain_type():
    assert pd._is_optional(int) is False
    assert pd._is_optional(List[int]) is False


# -----------------------------
# _default_for_collection
# -----------------------------


def test_default_for_collection_builtin_collections():
    assert pd._default_for_collection(list) == []
    assert pd._default_for_collection(set) == set()
    assert pd._default_for_collection(dict) == {}
    assert pd._default_for_collection(tuple) == tuple()


def test_default_for_collection_custom_collection_subclass():
    class MyCollection(list):
        pass

    result = pd._default_for_collection(MyCollection)
    assert isinstance(result, MyCollection)
    assert len(result) == 0


# -----------------------------
# _default_for_tuple_args
# -----------------------------


def test_default_for_tuple_args_empty_and_variadic():
    assert pd._default_for_tuple_args(()) == ()
    assert pd._default_for_tuple_args((int, ...)) == ()


def test_default_for_tuple_args_concrete_types():
    result = pd._default_for_tuple_args((int, str, bool))
    assert result == (0, "", False)


# -----------------------------
# _default_for_dataclass
# -----------------------------


def test_default_for_dataclass_respects_defaults_and_factories_and_ignores_hidden():
    obj = pd._default_for_dataclass(SimpleDataClass)

    assert isinstance(obj, SimpleDataClass)
    # field with no default -> default_scalar(int) -> 0
    assert obj.a == 0
    # explicit default preserved
    assert obj.b == "x"
    # default_factory used
    assert obj.c == 5
    # init=False fields are not passed in kwargs, should still be default
    assert obj.d == 10
    # fields starting with "_" are ignored
    assert obj._internal == 1


# -----------------------------
# default_python_scalar
# -----------------------------


def test_default_python_scalar_primitives_and_specials():
    assert pd.default_python_scalar(int) == 0
    assert pd.default_python_scalar(float) == 0.0
    assert pd.default_python_scalar(str) == ""
    assert pd.default_python_scalar(bool) is False
    assert pd.default_python_scalar(bytes) == b""

    dt = pd.default_python_scalar(datetime.datetime)
    assert isinstance(dt, datetime.datetime)
    assert dt == datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)

    d = pd.default_python_scalar(datetime.date)
    assert d == datetime.date(1970, 1, 1)

    t = pd.default_python_scalar(datetime.time)
    assert t == datetime.time(0, 0, 0, tzinfo=datetime.timezone.utc)

    td = pd.default_python_scalar(datetime.timedelta)
    assert td == datetime.timedelta(0)

    u = pd.default_python_scalar(uuid.UUID)
    assert u.int == 0

    dec = pd.default_python_scalar(decimal.Decimal)
    assert dec == decimal.Decimal(0)


def test_default_python_scalar_optional_returns_none():
    assert pd.default_python_scalar(Optional[int]) is None
    assert pd.default_python_scalar(int | None) is None


def test_default_python_scalar_collections_and_tuples():
    # bare collection types
    assert pd.default_python_scalar(list) == []
    assert pd.default_python_scalar(set) == set()
    assert pd.default_python_scalar(dict) == {}
    assert pd.default_python_scalar(tuple) == tuple()

    # typing-hinted collections
    assert pd.default_python_scalar(List[int]) == []
    assert pd.default_python_scalar(Set[str]) == set()
    assert pd.default_python_scalar(Dict[str, int]) == {}
    assert pd.default_python_scalar(Tuple[int, str]) == (0, "")


def test_default_python_scalar_dataclass():
    obj = pd.default_python_scalar(SimpleDataClass)
    assert isinstance(obj, SimpleDataClass)
    assert obj.a == 0
    assert obj.b == "x"
    assert obj.c == 5


def test_default_python_scalar_custom_type_no_args():
    obj = pd.default_python_scalar(CustomNoArg)
    assert isinstance(obj, CustomNoArg)
    assert obj.value == 42


def test_default_python_scalar_raises_for_custom_type_with_args():
    with pytest.raises(TypeError):
        pd.default_python_scalar(CustomNeedsArg)


# -----------------------------
# default_arrow_scalar
# -----------------------------


def test_default_arrow_scalar_primitive_non_nullable():
    scalar = pd.default_arrow_scalar(dtype=pa.int32(), nullable=False)
    assert isinstance(scalar, pa.Scalar)
    assert scalar.type == pa.int32()
    assert scalar.as_py() == 0

    scalar_str = pd.default_arrow_scalar(dtype=pa.string(), nullable=False)
    assert scalar_str.type == pa.string()
    assert scalar_str.as_py() == ""


def test_default_arrow_scalar_nullable_wraps_none():
    scalar = pd.default_arrow_scalar(dtype=pa.int64(), nullable=True)
    assert scalar.type == pa.int64()
    assert scalar.is_valid is False
    assert scalar.as_py() is None


def test_default_arrow_scalar_temporal_types():
    ts_type = pa.timestamp("ns")
    ts_scalar = pd.default_arrow_scalar(dtype=ts_type, nullable=False)
    assert ts_scalar.type == ts_type
    # Arrow converts 0 to epoch datetime
    assert ts_scalar.as_py() == datetime.datetime(1970, 1, 1)

    date_type = pa.date32()
    date_scalar = pd.default_arrow_scalar(dtype=date_type, nullable=False)
    assert date_scalar.type == date_type
    assert date_scalar.as_py() == datetime.date(1970, 1, 1)


def test_default_arrow_scalar_decimal_type():
    dec_type = pa.decimal128(10, 0)
    scalar = pd.default_arrow_scalar(dtype=dec_type, nullable=False)
    assert scalar.type == dec_type
    assert scalar.as_py() == decimal.Decimal(0)


def test_default_arrow_scalar_fixed_size_binary():
    dtype = pa.binary(4)
    scalar = pd.default_arrow_scalar(dtype=dtype, nullable=False)
    assert scalar.type == dtype
    assert scalar.as_py() == b"\x00" * 4


def test_default_arrow_scalar_struct_type():
    dtype = pa.struct(
        [
            pa.field("a", pa.int32(), nullable=False),
            pa.field("b", pa.string(), nullable=False),
        ]
    )

    scalar = pd.default_arrow_scalar(dtype=dtype, nullable=False)
    assert scalar.type == dtype
    # as_py returns dict mapping field names to python values
    assert scalar.as_py() == {"a": 0, "b": ""}


def test_default_arrow_scalar_list_and_fixed_size_list():
    list_type = pa.list_(pa.int32())
    list_scalar = pd.default_arrow_scalar(dtype=list_type, nullable=False)
    assert list_scalar.type == list_type
    assert list_scalar.as_py() == []

    fixed_list_type = pa.list_(pa.field("item", pa.int32(), nullable=False), 3)  # FixedSizeListType
    fixed_list_scalar = pd.default_arrow_scalar(
        dtype=fixed_list_type, nullable=False
    )
    assert fixed_list_scalar.type == fixed_list_type
    assert fixed_list_scalar.as_py() == [0, 0, 0]


def test_default_arrow_scalar_map_type():
    map_type = pa.map_(pa.string(), pa.int32())
    scalar = pd.default_arrow_scalar(dtype=map_type, nullable=False)
    assert scalar.type == map_type
    assert scalar.as_py(maps_as_pydicts="strict") == {}


# -----------------------------
# default_arrow_array
# -----------------------------


def test_default_arrow_array_simple_array():
    dtype = pa.int32()
    arr = pd.default_arrow_array(dtype=dtype, nullable=False, size=3)
    assert isinstance(arr, pa.Array)
    assert arr.type == dtype
    assert arr.to_pylist() == [0, 0, 0]


def test_default_arrow_array_nullable_array():
    dtype = pa.int32()
    arr = pd.default_arrow_array(dtype=dtype, nullable=True, size=3)
    assert arr.to_pylist() == [None, None, None]


def test_default_arrow_array_with_chunks():
    dtype = pa.int32()
    chunks = [2, 3]
    carr = pd.default_arrow_array(dtype=dtype, nullable=False, chunks=chunks)

    assert isinstance(carr, pa.ChunkedArray)
    assert carr.num_chunks == 2
    assert carr.type == dtype
    assert carr.chunks[0].to_pylist() == [0, 0]
    assert carr.chunks[1].to_pylist() == [0, 0, 0]


def test_default_arrow_array_with_custom_scalar_default():
    dtype = pa.int32()
    scalar_default = pa.scalar(1, type=dtype)

    arr = pd.default_arrow_array(
        dtype=dtype, nullable=False, size=4, scalar_default=scalar_default
    )
    assert arr.to_pylist() == [1, 1, 1, 1]


# -----------------------------
# default_scalar (top-level)
# -----------------------------


def test_default_scalar_python_type_delegates_to_default_python_scalar():
    assert pd.default_scalar(int) == 0
    assert pd.default_scalar(str) == ""


def test_default_scalar_arrow_dtype_delegates_to_default_arrow_scalar():
    dtype = pa.int64()
    scalar = pd.default_scalar(dtype, nullable=False)
    assert isinstance(scalar, pa.Scalar)
    assert scalar.type == dtype
    assert scalar.as_py() == 0


def test_default_scalar_arrow_nullable_true():
    dtype = pa.int64()
    scalar = pd.default_scalar(dtype, nullable=True)
    assert scalar.as_py() is None
