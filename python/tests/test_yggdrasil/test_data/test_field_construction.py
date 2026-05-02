"""Construction paths for :class:`Field`.

A :class:`Field` has roughly a dozen ways to be built — the factory
function, the dataclass constructor, ``from_pytype`` / ``from_str`` /
``from_dataclass`` / ``from_dict`` / ``from_json`` / ``from_any`` /
``from_arrow_field`` — and they all eventually land on the same
:class:`Field` shape. These tests pin one canonical example per
entry-point so a regression in any constructor shows up close to its
source.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

import pyarrow as pa
import pytest

from yggdrasil.data.data_field import Field, field
from yggdrasil.data.types.nested import StructType
from yggdrasil.data.types.primitive import IntegerType, StringType


# ---------------------------------------------------------------------------
# Fixture types — only used by the from_dataclass / from_pytype tests.
# ---------------------------------------------------------------------------


@dataclass
class TradeRow:
    ts: str
    qty: int


@dataclass
class DataclassWithDefaultedField:
    value: Optional[int] = None
    name: str = "x"


# ---------------------------------------------------------------------------
# Factory + dataclass constructor
# ---------------------------------------------------------------------------


class TestFactoryAndInit:

    def test_field_factory_normalizes_metadata_keys_to_bytes(self) -> None:
        out = field(
            "price",
            pa.int64(),
            nullable=False,
            metadata={"comment": "settlement price"},
        )

        assert isinstance(out, Field)
        assert out.name == "price"
        assert isinstance(out.dtype, IntegerType)
        assert out.nullable is False
        assert out.metadata is not None
        assert out.metadata[b"comment"] == b"settlement price"

    def test_init_normalizes_arrow_dtype_to_yggdrasil_dtype(self) -> None:
        out = Field("value", pa.string(), nullable=True)

        assert out.name == "value"
        assert isinstance(out.dtype, StringType)
        assert out.arrow_type == pa.string()

    def test_default_attribute_round_trip(self) -> None:
        out = Field("value", pa.int64(), default=7)

        assert out.has_default is True
        assert out.default == 7

    def test_copy_overrides_named_attributes_only(self) -> None:
        src = Field(
            "value",
            pa.int64(),
            nullable=True,
            metadata={"comment": "a"},
        )

        out = src.copy(name="value2", nullable=False)

        assert out.name == "value2"
        assert isinstance(out.dtype, IntegerType)
        assert out.nullable is False
        assert out.metadata is not None
        assert out.metadata[b"comment"] == b"a"


# ---------------------------------------------------------------------------
# from_pytype
# ---------------------------------------------------------------------------


class TestFromPytype:

    def test_optional_marks_nullable(self) -> None:
        out = Field.from_pytype(Optional[int], name="qty")

        assert out.name == "qty"
        assert isinstance(out.dtype, IntegerType)
        assert out.nullable is True

    def test_plain_type_uses_dtype_lower_name(self) -> None:
        out = Field.from_pytype(int)

        assert out.name == "int"
        assert isinstance(out.dtype, IntegerType)
        assert out.nullable is False

    def test_string_arg_is_rejected_by_pytype(self) -> None:
        # Field shorthand must go through ``from_str``; from_pytype refuses.
        with pytest.raises(ValueError, match="Unexpected trailing tokens"):
            Field.from_pytype("qty:int32")


# ---------------------------------------------------------------------------
# from_str
# ---------------------------------------------------------------------------


class TestFromStr:

    def test_shorthand_with_nullability_suffix(self) -> None:
        out = Field.from_str("qty!:int64")

        assert out.name == "qty"
        assert isinstance(out.dtype, IntegerType)
        assert out.dtype.byte_size == 8
        assert out.nullable is False

    def test_json_payload(self) -> None:
        out = Field.from_str(
            '{"name":"qty","dtype":{"id":3,"byte_size":4,"signed":true},'
            '"nullable":false}'
        )

        assert out.name == "qty"
        assert isinstance(out.dtype, IntegerType)
        assert out.arrow_type == pa.int32()
        assert out.nullable is False


# ---------------------------------------------------------------------------
# from_dataclass / from_dataclass_field
# ---------------------------------------------------------------------------


class TestFromDataclass:

    def test_dataclass_promotes_to_struct_field(self) -> None:
        out = Field.from_dataclass(TradeRow)

        assert out.name == "TradeRow"
        assert isinstance(out.dtype, StructType)
        assert out.nullable is False
        assert out.arrow_type == pa.struct(
            [
                pa.field("ts", pa.string(), nullable=True),
                pa.field("qty", pa.int32(), nullable=True),
            ]
        )

    def test_dataclass_field_optional_resolves_to_nullable_with_no_default(
        self,
    ) -> None:
        dc_field = dataclasses.fields(DataclassWithDefaultedField)[0]

        out = Field.from_dataclass_field(dc_field, owner=DataclassWithDefaultedField)

        assert out.name == "value"
        assert isinstance(out.dtype, IntegerType)
        assert out.nullable is True
        assert out.default is None


# ---------------------------------------------------------------------------
# from_dict / from_json
# ---------------------------------------------------------------------------


class TestFromDictAndJson:

    def test_from_dict_builds_field_with_metadata(self) -> None:
        out = Field.from_dict(
            {
                "name": "value",
                "dtype": {"id": 11},
                "nullable": True,
                "metadata": {"comment": "hello"},
            }
        )

        assert out.name == "value"
        assert isinstance(out.dtype, StringType)
        assert out.nullable is True
        assert out.metadata is not None
        assert out.metadata[b"comment"] == b"hello"

    def test_from_json_round_trips_through_to_json(self) -> None:
        src = Field(
            "value",
            pa.int64(),
            nullable=False,
            metadata={"comment": "x"},
        )

        out = Field.from_json(src.to_json())

        assert out.name == "value"
        assert isinstance(out.dtype, IntegerType)
        assert out.nullable is False


# ---------------------------------------------------------------------------
# from_any — multi-shape dispatch
# ---------------------------------------------------------------------------


class TestFromAny:

    def test_existing_field_passes_through_identical(self) -> None:
        src = Field("value", pa.int64())
        assert Field.from_any(src) is src

    def test_arrow_field_promotes(self) -> None:
        out = Field.from_any(pa.field("name", pa.string(), nullable=False))

        assert out.name == "name"
        assert isinstance(out.dtype, StringType)
        assert out.nullable is False

    def test_unsupported_object_raises(self) -> None:
        class _Unsupported:
            pass

        with pytest.raises(TypeError, match="Cannot build Field from"):
            Field.from_any(_Unsupported())
