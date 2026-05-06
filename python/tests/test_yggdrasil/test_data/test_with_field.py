"""Behavior tests for :meth:`Field.with_field` and :meth:`Field.with_fields`.

Two responsibilities the tests pin:

* **Mode dispatch** — collisions on a child name resolve via the
  ``mode`` kwarg (AUTO/OVERWRITE → replace, IGNORE → keep existing,
  APPEND → keep both, UPSERT/MERGE → :meth:`merge_with`,
  ERROR_IF_EXISTS → raise).
* **Auto-promotion** — calling ``with_field`` on a non-struct field
  (a primitive, a list, etc.) returns a struct whose first child is
  the previous self (keeps its name + nullability + metadata) and
  whose second child is the incoming one.
"""
from __future__ import annotations

import pytest

from yggdrasil.data.data_field import Field
from yggdrasil.data.enums import Mode
from yggdrasil.data.types.nested.struct import StructType
from yggdrasil.data.types.primitive import (
    Float64Type,
    Int64Type,
    StringType,
)


# ---------------------------------------------------------------------------
# Auto-promotion to struct
# ---------------------------------------------------------------------------


class TestAutoPromote:

    def test_primitive_becomes_struct_inplace(self) -> None:
        prim = Field(name="age", dtype=Int64Type())
        ret = prim.with_field(Field(name="name", dtype=StringType()))
        assert ret is prim  # inplace
        assert isinstance(prim.dtype, StructType)
        assert [f.name for f in prim.children_fields] == ["age", "name"]

    def test_primitive_keeps_top_level_identity(self) -> None:
        prim = Field(
            name="trade", dtype=Float64Type(), nullable=False,
            metadata={b"unit": b"usd"},
        )
        prim.with_field(Field(name="ticker", dtype=StringType()))
        # Top-level field still named ``trade`` and not nullable.
        assert prim.name == "trade"
        assert prim.nullable is False
        # The wrapped previous self is the first child.
        original_child = prim.children_fields[0]
        assert original_child.name == "trade"
        assert isinstance(original_child.dtype, Float64Type)
        assert original_child.metadata == {b"unit": b"usd"}

    def test_promote_returns_copy_when_not_inplace(self) -> None:
        prim = Field(name="x", dtype=Int64Type())
        promoted = prim.with_field(
            Field(name="y", dtype=StringType()), inplace=False,
        )
        assert promoted is not prim
        # Original primitive untouched.
        assert isinstance(prim.dtype, Int64Type)
        assert isinstance(promoted.dtype, StructType)

    def test_string_shorthand(self) -> None:
        # ``with_field("price")`` should accept a bare-string and
        # synthesize a Field via ``Field.from_any``.
        prim = Field(name="seed", dtype=StringType())
        prim.with_field("price")
        assert "price" in [f.name for f in prim.children_fields]


# ---------------------------------------------------------------------------
# Append / replace / ignore / error / merge on an existing struct
# ---------------------------------------------------------------------------


class TestStructCollisions:

    def _struct(self) -> Field:
        s = Field.empty()
        s.with_field(Field(name="id", dtype=Int64Type()))
        s.with_field(Field(name="name", dtype=StringType()))
        return s

    def test_first_write_appends(self) -> None:
        s = self._struct()
        s.with_field(Field(name="price", dtype=Float64Type()))
        assert [f.name for f in s.children_fields] == ["id", "name", "price"]

    def test_default_mode_replaces_collision(self) -> None:
        s = self._struct()
        # Default is AUTO → replace.
        s.with_field(Field(name="id", dtype=StringType()))
        assert isinstance(s.field("id").dtype, StringType)
        # No duplicate child.
        assert [f.name for f in s.children_fields] == ["id", "name"]

    def test_overwrite_mode_replaces(self) -> None:
        s = self._struct()
        s.with_field(
            Field(name="id", dtype=StringType()), mode=Mode.OVERWRITE,
        )
        assert isinstance(s.field("id").dtype, StringType)

    def test_ignore_mode_keeps_existing(self) -> None:
        s = self._struct()
        s.with_field(
            Field(name="id", dtype=StringType()), mode=Mode.IGNORE,
        )
        assert isinstance(s.field("id").dtype, Int64Type)

    def test_error_if_exists_raises(self) -> None:
        s = self._struct()
        with pytest.raises(ValueError, match="already exists"):
            s.with_field(
                Field(name="id", dtype=StringType()),
                mode=Mode.ERROR_IF_EXISTS,
            )

    def test_append_mode_keeps_both(self) -> None:
        s = self._struct()
        # ``Mode.APPEND`` honestly appends — duplicate names allowed
        # at the children level (Arrow Struct supports this).
        s.with_field(Field(name="id", dtype=StringType()), mode=Mode.APPEND)
        names = [f.name for f in s.children_fields]
        assert names.count("id") == 2

    def test_upsert_mode_merges_dtype(self) -> None:
        s = self._struct()
        # Stamp metadata on the incoming child; merge should preserve
        # the existing dtype (no upcast/downcast) but pull in the new
        # metadata.
        s.with_field(
            Field(
                name="id",
                dtype=Int64Type(),
                metadata={b"role": b"identifier"},
            ),
            mode=Mode.UPSERT,
        )
        merged = s.field("id")
        assert merged.metadata.get(b"role") == b"identifier"
        # Still one entry.
        assert [f.name for f in s.children_fields].count("id") == 1


# ---------------------------------------------------------------------------
# String-alias mode resolution
# ---------------------------------------------------------------------------


class TestModeAlias:

    def test_string_alias_resolves(self) -> None:
        s = Field.empty()
        s.with_field("a")
        s.with_field("a", mode="overwrite")
        # No duplicate, AUTO collision behavior under the alias.
        assert len([f for f in s.children_fields if f.name == "a"]) == 1


# ---------------------------------------------------------------------------
# with_fields — bulk
# ---------------------------------------------------------------------------


class TestWithFields:

    def test_appends_multiple(self) -> None:
        s = Field.empty()
        s.with_fields([
            Field(name="a", dtype=Int64Type()),
            Field(name="b", dtype=StringType()),
            Field(name="c", dtype=Float64Type()),
        ])
        assert [f.name for f in s.children_fields] == ["a", "b", "c"]

    def test_promote_then_extend(self) -> None:
        prim = Field(name="root", dtype=Int64Type())
        prim.with_fields([
            Field(name="b", dtype=StringType()),
            Field(name="c", dtype=Float64Type()),
        ])
        names = [f.name for f in prim.children_fields]
        assert names == ["root", "b", "c"]

    def test_mixed_string_and_field_inputs(self) -> None:
        s = Field.empty()
        s.with_fields(["x", Field(name="y", dtype=StringType())])
        assert {f.name for f in s.children_fields} == {"x", "y"}

    def test_mode_applies_per_entry(self) -> None:
        s = Field.empty()
        s.with_field(Field(name="a", dtype=Int64Type()))
        s.with_fields(
            [
                Field(name="a", dtype=StringType()),
                Field(name="b", dtype=Float64Type()),
            ],
            mode=Mode.IGNORE,
        )
        # Existing 'a' kept; new 'b' added.
        assert isinstance(s.field("a").dtype, Int64Type)
        assert isinstance(s.field("b").dtype, Float64Type)


# ---------------------------------------------------------------------------
# Inplace semantics
# ---------------------------------------------------------------------------


class TestInplace:

    def test_default_inplace_true(self) -> None:
        s = Field.empty()
        ret = s.with_field("x")
        assert ret is s

    def test_inplace_false_returns_copy(self) -> None:
        s = Field.empty()
        ret = s.with_field("x", inplace=False)
        assert ret is not s
        # Original unchanged.
        assert len(s.children_fields) == 0
        assert "x" in [f.name for f in ret.children_fields]
