"""``Field.merge_with`` semantics.

The merge contract pins:

* Same field merged with itself is identity (no copy).
* Cross-name merge takes the left name when present, otherwise the
  right name.
* Metadata unions with left winning on conflict, regardless of mode.
* ``mode`` accepts both :class:`Mode` and string aliases.
* ``inplace`` actually mutates ``self`` and returns it; ``inplace=False``
  leaves both inputs untouched.
* ``default`` carries through from the side that has it; ``metadata``
  still merges from the other side.
"""
from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data.data_field import Field, field
from yggdrasil.data.enums import Mode


class TestSelfMerge:

    def test_self_merge_is_identity(self) -> None:
        f = field(
            "price",
            pa.float64(),
            nullable=False,
            metadata={"comment": "settlement price"},
        )
        assert f.merge_with(f) is f


class TestMetadataAndNullability:

    def test_metadata_unions_with_left_wins_on_conflict(self) -> None:
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

        assert out is not left and out is not right
        assert out.name == "price"
        assert out.nullable is False
        assert out.metadata == {
            b"left_only": b"a",
            b"shared": b"left",
            b"right_only": b"b",
        }

        # Originals untouched.
        assert left.metadata == {b"left_only": b"a", b"shared": b"left"}
        assert right.metadata == {b"right_only": b"b", b"shared": b"right"}


class TestNamePropagation:

    def test_left_name_wins_when_present(self) -> None:
        left = field("left_name", pa.int32(), nullable=False)
        right = field("right_name", pa.int32(), nullable=False)

        assert left.merge_with(right).name == "left_name"

    def test_empty_left_name_falls_back_to_right(self) -> None:
        left = field("", pa.int32(), nullable=False)
        right = field("fallback_name", pa.int32(), nullable=False)

        assert left.merge_with(right).name == "fallback_name"


class TestModeKeyword:

    @pytest.mark.parametrize("mode", [Mode.APPEND, "append"])
    def test_accepts_mode_enum_or_string(self, mode) -> None:
        left = field("value", pa.int32(), nullable=False)
        right = field("value", pa.int64(), nullable=False)

        out = left.merge_with(right, mode=mode, upcast=True)

        assert isinstance(out, Field)
        assert out.name == "value"


class TestInplace:

    def test_inplace_true_mutates_self_and_returns_it(self) -> None:
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

        out = left.merge_with(right, inplace=True, merge_dtype=False)

        # left mutated in place.
        assert left.arrow_type == pa.int32()
        assert left.nullable is False
        assert left.metadata == {b"comment": b"left", b"unit": b"eur"}

        # return value points back at self.
        assert out is left
        assert out.equals(left)

    def test_inplace_false_leaves_self_untouched(self) -> None:
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

        out = left.merge_with(right, inplace=False, merge_dtype=False)

        # left is untouched.
        assert left.arrow_type == pa.int32()
        assert left.nullable is False
        assert left.metadata == {b"comment": b"left"}

        # out has the merged metadata.
        assert out.arrow_type == pa.int32()
        assert out.nullable is False
        assert out.metadata == {b"comment": b"left", b"unit": b"eur"}


class TestDefaultPropagation:

    def test_default_from_left_survives_metadata_merge_from_right(self) -> None:
        left = field("x", pa.int32(), nullable=False, default=123)
        right = field("x", pa.int32(), nullable=False, metadata={"comment": "rhs"})

        out = left.merge_with(right)

        assert out.has_default is True
        assert out.default_value == 123
        assert out.metadata is not None
        assert out.metadata[b"comment"] == b"rhs"
