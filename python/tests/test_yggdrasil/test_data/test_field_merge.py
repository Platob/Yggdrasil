from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data.data_field import Field, field
from yggdrasil.io import SaveMode


class TestFieldMergeWith:
    def test_merge_with_same_field_returns_self(self) -> None:
        left = field(
            "price",
            pa.float64(),
            nullable=False,
            metadata={"comment": "settlement price"},
        )

        out = left.merge_with(left)

        assert out is left

    def test_merge_with_combines_metadata_and_nullable_by_default(self) -> None:
        left = field(
            "price",
            pa.int32(),
            nullable=False,
            metadata={"left_only": "a", "shared": "left"},
        )
        right = field(
            "price",
            pa.int32(),
            nullable=True,
            metadata={"right_only": "b", "shared": "right"},
        )

        out = left.merge_with(right)

        assert out is not left
        assert out is not right
        assert out.name == "price"
        assert out.nullable is True
        assert out.metadata == {
            b"left_only": b"a",
            b"shared": b"right",
            b"right_only": b"b",
        }

        # originals unchanged
        assert left.nullable is False
        assert left.metadata == {
            b"left_only": b"a",
            b"shared": b"left",
        }
        assert right.metadata == {
            b"right_only": b"b",
            b"shared": b"right",
        }

    def test_merge_with_merge_metadata_false_drops_metadata_from_result(self) -> None:
        left = field(
            "qty",
            pa.int64(),
            nullable=False,
            metadata={"comment": "lhs"},
        )
        right = field(
            "qty",
            pa.int64(),
            nullable=True,
            metadata={"unit": "mw"},
        )

        out = left.merge_with(right, merge_metadata=False)

        assert out.metadata == {b"comment": b"lhs"}
        assert out.nullable is True

    def test_merge_with_merge_nullable_false_keeps_left_nullable(self) -> None:
        left = field("qty", pa.int64(), nullable=False)
        right = field("qty", pa.int64(), nullable=True)

        out = left.merge_with(right, merge_nullable=False)

        assert out.nullable is False

    def test_merge_with_merge_dtype_false_uses_right_dtype(self) -> None:
        left = field("value", pa.int32(), nullable=False)
        right = field("value", pa.float64(), nullable=False)

        out = left.merge_with(right, merge_dtype=False)

        assert out.dtype.equals(right.dtype)
        assert out.arrow_type == pa.float64()

    def test_merge_with_uses_left_name_when_present(self) -> None:
        left = field("left_name", pa.int32(), nullable=False)
        right = field("right_name", pa.int32(), nullable=False)

        out = left.merge_with(right)

        assert out.name == "left_name"

    def test_merge_with_uses_other_name_when_self_name_empty(self) -> None:
        left = field("", pa.int32(), nullable=False)
        right = field("fallback_name", pa.int32(), nullable=False)

        out = left.merge_with(right)

        assert out.name == "fallback_name"

    @pytest.mark.parametrize("mode", [SaveMode.APPEND, "append"])
    def test_merge_with_accepts_save_mode_enum_and_string(self, mode: SaveMode | str) -> None:
        left = field("value", pa.int32(), nullable=False)
        right = field("value", pa.int64(), nullable=False)

        out = left.merge_with(right, mode=mode, upcast=True)

        assert isinstance(out, Field)
        assert out.name == "value"

    def test_merge_with_inplace_mutates_self(self) -> None:
        left = field(
            "price",
            pa.int32(),
            nullable=False,
            metadata={"comment": "left"},
        )
        right = field(
            "price",
            pa.float64(),
            nullable=True,
            metadata={"unit": "eur"},
        )

        out = left.merge_with(
            right,
            inplace=True,
            merge_dtype=False,
        )

        # current implementation mutates self
        assert left.arrow_type == pa.float64()
        assert left.nullable is True
        assert left.metadata == {
            b"comment": b"left",
            b"unit": b"eur",
        }

        # and returns a copy, not self
        assert out is left
        assert out.equals(left)

    def test_merge_with_inplace_false_does_not_mutate_self(self) -> None:
        left = field(
            "price",
            pa.int32(),
            nullable=False,
            metadata={"comment": "left"},
        )
        right = field(
            "price",
            pa.float64(),
            nullable=True,
            metadata={"unit": "eur"},
        )

        out = left.merge_with(
            right,
            inplace=False,
            merge_dtype=False,
        )

        assert left.arrow_type == pa.int32()
        assert left.nullable is False
        assert left.metadata == {b"comment": b"left"}

        assert out.arrow_type == pa.float64()
        assert out.nullable is True
        assert out.metadata == {
            b"comment": b"left",
            b"unit": b"eur",
        }

    def test_merge_with_preserves_default_metadata_when_merged(self) -> None:
        left = field("x", pa.int32(), nullable=False, default=123)
        right = field("x", pa.int32(), nullable=False, metadata={"comment": "rhs"})

        out = left.merge_with(right)

        assert out.has_default is True
        assert out.default == 123
        assert out.metadata is not None
        assert out.metadata[b"comment"] == b"rhs"