"""``Schema`` — the OrderedDict-shaped, name-keyed schema container.

Schema is intentionally Mutable + Mapping-like: it owns an
:class:`OrderedDict` of fields, supports ``__getitem__`` / ``__setitem__``
/ ``pop`` / ``popitem`` / ``setdefault``, and overloads the four set
operators (``+`` / ``-`` / ``&`` / ``|`` and their in-place variants)
to make schema reconciliation read like set algebra.

Tests grouped by surface:

* construction — factory, init normalization, ``schema()`` helper.
* metadata — ``name`` / ``comment`` / ``description``.
* shape — ``dtype`` / ``arrow_fields`` / ``copy``.
* mutation — append / extend / setitem / delitem / clear /
  setdefault / pop / popitem.
* set operators — add, sub, and, or, plus in-place variants.
* autotag — schema-level + per-field propagation, partition_by /
  cluster_by consumption.
* exporters — to_arrow_schema, to_field, polars / spark flavors.
* constructors — from_field, from_any_fields, from_any, from_,
  _coerce_other, _merge_metadata, from_path.
"""
from __future__ import annotations

from collections import OrderedDict

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from yggdrasil.data.constants import DEFAULT_FIELD_NAME
from yggdrasil.data.data_field import Field
from yggdrasil.data.schema import Schema, schema
from yggdrasil.data.types.nested import StructType
from yggdrasil.data.types.primitive import IntegerType, StringType, TimestampType
from yggdrasil.polars.tests import PolarsTestCase
from yggdrasil.spark.tests import SparkTestCase


# ---------------------------------------------------------------------------
# Helpers — every test that builds a field ends up calling one of these.
# ---------------------------------------------------------------------------


def _int_field(
    name: str,
    *,
    nullable: bool = True,
    metadata: dict | None = None,
    tags: dict | None = None,
    default=...,
) -> Field:
    kwargs: dict = {
        "name": name,
        "dtype": IntegerType(byte_size=8, signed=True),
        "nullable": nullable,
        "metadata": metadata,
        "tags": tags,
    }
    if default is not ...:
        kwargs["default"] = default
    return Field(**kwargs)


def _str_field(
    name: str,
    *,
    nullable: bool = True,
    metadata: dict | None = None,
    tags: dict | None = None,
    default=...,
) -> Field:
    kwargs: dict = {
        "name": name,
        "dtype": StringType(),
        "nullable": nullable,
        "metadata": metadata,
        "tags": tags,
    }
    if default is not ...:
        kwargs["default"] = default
    return Field(**kwargs)


# ---------------------------------------------------------------------------
# Construction — schema() factory + init normalization
# ---------------------------------------------------------------------------


class TestConstruction:

    def test_schema_factory_accepts_iterable_and_varargs(self) -> None:
        s = schema(
            [_int_field("a"), _str_field("b")],
            _int_field("c"),
            metadata={"name": "trade_row", "comment": "schema comment"},
        )

        assert list(s.keys()) == ["a", "b", "c"]
        assert s.name == "trade_row"
        assert s.comment == "schema comment"

    def test_schema_factory_accepts_single_field(self) -> None:
        s = schema(_int_field("a"))

        assert list(s.keys()) == ["a"]
        assert isinstance(s["a"], Field)

    def test_init_with_mapping_rewrites_field_names_to_keys(self) -> None:
        s = Schema(
            inner_fields={
                "qty": _int_field("wrong_name"),
                "book": _str_field("book"),
            },
            metadata=None,
        )

        assert list(s.keys()) == ["qty", "book"]
        assert s["qty"].name == "qty"
        assert s["book"].name == "book"

    def test_ordered_dict_input_preserves_insertion_order(self) -> None:
        inner = OrderedDict(
            [
                ("a", _int_field("a")),
                ("b", _str_field("b")),
                ("c", _int_field("c")),
            ]
        )
        s = Schema(inner_fields=inner)

        assert list(s.keys()) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Metadata — comment / description / name aliasing
# ---------------------------------------------------------------------------


class TestMetadata:

    def test_comment_falls_back_through_description(self) -> None:
        s_comment = schema([_int_field("a")], metadata={"comment": "hello"})
        s_desc = schema([_int_field("a")], metadata={"description": "world"})
        s_none = schema([_int_field("a")])

        assert s_comment.comment == "hello"
        assert s_desc.comment == "world"
        assert s_none.comment is None


# ---------------------------------------------------------------------------
# Shape — dtype / arrow_fields / copy
# ---------------------------------------------------------------------------


class TestShape:

    def test_dtype_is_struct_with_field_order(self) -> None:
        s = schema([_int_field("qty"), _str_field("book")])

        dtype = s.dtype

        assert isinstance(dtype, StructType)
        assert [f.name for f in dtype.fields] == ["qty", "book"]

    def test_arrow_fields_returns_arrow_field_list(self) -> None:
        s = schema([_int_field("qty"), _str_field("book")])

        out = s.arrow_fields

        assert len(out) == 2
        assert all(isinstance(f, pa.Field) for f in out)
        assert [f.name for f in out] == ["qty", "book"]

    def test_copy_deep_copies_fields_and_keeps_metadata(self) -> None:
        s = schema(
            [_int_field("a"), _str_field("b")],
            metadata={"name": "x", "comment": "y"},
        )

        out = s.copy()

        assert out is not s
        assert list(out.keys()) == ["a", "b"]
        assert out.metadata == s.metadata
        assert out["a"] is not s["a"]
        assert out["b"] is not s["b"]

    def test_copy_can_replace_fields_and_metadata(self) -> None:
        s = schema([_int_field("a")], metadata={"name": "old"})

        out = s.copy(
            fields=[_str_field("b")],
            metadata={"name": "new", "comment": "desc"},
        )

        assert list(out.keys()) == ["b"]
        assert out.name == "new"
        assert out.comment == "desc"


# ---------------------------------------------------------------------------
# Mutation — Mapping-shaped surface
# ---------------------------------------------------------------------------


class TestAppendExtend:

    def test_append_returns_new_schema(self) -> None:
        s1 = schema([_int_field("a")])
        s2 = s1.append(_str_field("b"))

        assert list(s1.keys()) == ["a"]
        assert list(s2.keys()) == ["a", "b"]

    def test_extend_returns_new_schema(self) -> None:
        s1 = schema([_int_field("a")])
        s2 = s1.extend([_str_field("b"), _int_field("c")])

        assert list(s1.keys()) == ["a"]
        assert list(s2.keys()) == ["a", "b", "c"]


class TestGetItem:

    def test_by_name(self) -> None:
        s = schema([_int_field("a"), _str_field("b")])

        assert s["a"].name == "a"
        assert s["b"].name == "b"

    def test_by_index(self) -> None:
        s = schema([_int_field("a"), _str_field("b")])

        assert s[0].name == "a"
        assert s[1].name == "b"

    def test_get_returns_default_for_missing(self) -> None:
        s = schema([_int_field("a")])

        assert s.get("a").name == "a"
        assert s.get("missing") is None
        assert s.get("missing", "fallback") == "fallback"

    def test_iter_len_contains(self) -> None:
        s = schema([_int_field("a"), _str_field("b")])

        assert list(iter(s)) == ["a", "b"]
        assert len(s) == 2
        assert "a" in s
        assert "b" in s
        assert "c" not in s
        assert 0 in s
        assert 1 in s
        assert 2 not in s


class TestSetItem:

    def test_setitem_rewrites_field_name_to_key(self) -> None:
        s = schema([_int_field("a")])

        s["b"] = _str_field("wrong")

        assert list(s.keys()) == ["a", "b"]
        assert s["b"].name == "b"

    def test_setitem_rejects_non_string_key(self) -> None:
        s = schema([_int_field("a")])

        with pytest.raises(TypeError):
            s[1] = _int_field("b")  # type: ignore[index]


class TestDelItem:

    def test_delete_by_name_then_index(self) -> None:
        s = schema([_int_field("a"), _str_field("b"), _int_field("c")])

        del s["b"]
        assert list(s.keys()) == ["a", "c"]

        del s[0]
        assert list(s.keys()) == ["c"]


class TestPopAndSetDefault:

    def test_pop_by_name_index_and_default(self) -> None:
        s = schema([_int_field("a"), _str_field("b")])

        assert s.pop("a").name == "a"
        assert list(s.keys()) == ["b"]

        assert s.pop(0).name == "b"
        assert list(s.keys()) == []

        assert s.pop("missing", "fallback") == "fallback"

    def test_pop_missing_raises(self) -> None:
        s = schema([_int_field("a")])

        with pytest.raises(KeyError):
            s.pop("missing")

        with pytest.raises(IndexError):
            s.pop(10)

    def test_setdefault_existing_returns_existing_field(self) -> None:
        s = schema([_int_field("a")])

        out = s.setdefault("a", _str_field("other"))

        assert out.name == "a"
        assert isinstance(out.dtype, IntegerType)
        assert list(s.keys()) == ["a"]

    def test_setdefault_new_inserts_and_rewrites_name(self) -> None:
        s = schema([_int_field("a")])

        out = s.setdefault("b", _str_field("wrong"))

        assert out.name == "b"
        assert list(s.keys()) == ["a", "b"]
        assert s["b"].name == "b"

    def test_setdefault_rejects_non_string_key(self) -> None:
        s = schema([_int_field("a")])

        with pytest.raises(TypeError):
            s.setdefault(1, _int_field("x"))  # type: ignore[arg-type]

    def test_setdefault_requires_default_for_new_key(self) -> None:
        s = schema([_int_field("a")])

        with pytest.raises(ValueError):
            s.setdefault("b")

    def test_popitem_last_and_first(self) -> None:
        s = schema([_int_field("a"), _str_field("b"), _int_field("c")])

        key_last, field_last = s.popitem()
        assert key_last == "c"
        assert field_last.name == "c"

        key_first, field_first = s.popitem(last=False)
        assert key_first == "a"
        assert field_first.name == "a"

        assert list(s.keys()) == ["b"]

    def test_clear(self) -> None:
        s = schema([_int_field("a"), _str_field("b")])

        s.clear()

        assert len(s) == 0
        assert list(s.keys()) == []


# ---------------------------------------------------------------------------
# Set operators — +, -, &, |
# ---------------------------------------------------------------------------


class TestAdd:

    def test_add_unions_with_rhs_overriding_duplicate_names(self) -> None:
        s1 = schema(
            [_int_field("a"), _str_field("b")],
            metadata={"comment": "left", "name": "left_name"},
        )
        s2 = schema(
            [_int_field("b", nullable=False), _int_field("c")],
            metadata={"description": "right_desc", "name": "right_name"},
        )

        out = s1 + s2

        assert list(out.keys()) == ["a", "b", "c"]
        assert out["b"].nullable is False
        assert out.metadata is not None
        assert out.metadata[b"name"] == b"right_name"
        assert out.metadata[b"comment"] == b"left"
        assert out.metadata[b"description"] == b"right_desc"

    def test_radd_with_zero_returns_copy(self) -> None:
        s = schema([_int_field("a")], metadata={"name": "x"})

        out = 0 + s

        assert out is not s
        assert list(out.keys()) == ["a"]
        assert out.metadata == s.metadata

    def test_iadd_mutates_in_place(self) -> None:
        s1 = schema([_int_field("a")], metadata={"comment": "left"})
        s2 = schema([_str_field("b")], metadata={"name": "right"})

        original_id = id(s1)
        s1 += s2

        assert id(s1) == original_id
        assert list(s1.keys()) == ["a", "b"]
        assert s1.metadata is not None
        assert s1.metadata[b"comment"] == b"left"
        assert s1.metadata[b"name"] == b"right"


class TestSub:

    def test_sub_returns_difference(self) -> None:
        s1 = schema(
            [_int_field("a"), _str_field("b"), _int_field("c")],
            metadata={"name": "x"},
        )
        s2 = schema([_str_field("b")])

        out = s1 - s2

        assert list(out.keys()) == ["a", "c"]
        assert out.metadata == s1.metadata

    def test_isub_mutates_in_place(self) -> None:
        s1 = schema([_int_field("a"), _str_field("b"), _int_field("c")])
        s2 = schema([_str_field("b"), _int_field("x")])

        s1 -= s2

        assert list(s1.keys()) == ["a", "c"]


class TestAnd:

    def test_and_intersection_in_left_order(self) -> None:
        s1 = schema(
            [_int_field("a"), _str_field("b"), _int_field("c")],
            metadata={"comment": "left"},
        )
        s2 = schema(
            [_int_field("c"), _int_field("a")],
            metadata={"name": "right"},
        )

        out = s1 & s2

        assert list(out.keys()) == ["a", "c"]
        assert out.metadata is not None
        assert out.metadata[b"comment"] == b"left"
        assert out.metadata[b"name"] == b"right"

    def test_iand_mutates_to_intersection(self) -> None:
        s1 = schema(
            [_int_field("a"), _str_field("b"), _int_field("c")],
            metadata={"comment": "left"},
        )
        s2 = schema(
            [_int_field("c"), _int_field("a")],
            metadata={"name": "right"},
        )

        s1 &= s2

        assert list(s1.keys()) == ["a", "c"]
        assert s1.metadata is not None
        assert s1.metadata[b"comment"] == b"left"
        assert s1.metadata[b"name"] == b"right"


class TestOr:

    def test_or_and_ior_delegate_to_add(self) -> None:
        s1 = schema([_int_field("a")], metadata={"comment": "left"})
        s2 = schema([_str_field("b")], metadata={"name": "right"})

        out = s1 | s2
        assert list(out.keys()) == ["a", "b"]
        assert out.metadata is not None
        assert out.metadata[b"comment"] == b"left"
        assert out.metadata[b"name"] == b"right"

        s1 |= s2
        assert list(s1.keys()) == ["a", "b"]
        assert s1.metadata is not None
        assert s1.metadata[b"comment"] == b"left"
        assert s1.metadata[b"name"] == b"right"


# ---------------------------------------------------------------------------
# Autotag — propagates per-field, lifts partition / cluster metadata
# ---------------------------------------------------------------------------


class TestAutotag:

    def test_returns_new_schema_with_updated_metadata(self) -> None:
        s = schema([_int_field("a")], metadata={"name": DEFAULT_FIELD_NAME})

        out = s.autotag(tags={"semantic_type": "fact"})

        assert list(out.keys()) == ["a"]
        # Metadata is either replaced or kept-by-identity (autotag is
        # allowed to short-circuit when nothing changes); both are
        # acceptable signals that the call ran.
        assert out.metadata is not None
        assert out.metadata != s.metadata or out.metadata is s.metadata

    def test_propagates_dtype_and_field_tags(self) -> None:
        s = schema(
            [
                _int_field("user_id", nullable=False),
                _str_field("email"),
                Field(
                    "created_at",
                    TimestampType(unit="us", tz="UTC"),
                    nullable=False,
                ),
            ],
            metadata={"primary_key": "user_id"},
        )

        out = s.autotag(tags={"layer": "silver"})

        id_tags = out["user_id"].tags or {}
        assert id_tags[b"type_name"] == b"integer"
        assert id_tags[b"signed"] == b"true"
        assert id_tags[b"nullable"] == b"false"
        assert id_tags[b"primary_key"] == b"true"

        email_tags = out["email"].tags or {}
        assert email_tags[b"type_name"] == b"string"

        ts_tags = out["created_at"].tags or {}
        assert ts_tags[b"type_name"] == b"timestamp"
        assert ts_tags[b"unit"] == b"us"
        assert ts_tags[b"tz"] == b"UTC"

        assert out.metadata is not None
        assert out.metadata[b"t:layer"] == b"silver"

    def test_partition_and_cluster_metadata_consumed_into_field_tags(self) -> None:
        s = schema(
            [
                _int_field("trade_date"),
                _int_field("book_id"),
                _int_field("trade_id"),
            ],
            metadata={
                "partition_by": "trade_date",
                "cluster_by": '["book_id", "trade_id"]',
            },
        )

        out = s.autotag()

        assert [f.name for f in out.partition_by] == ["trade_date"]
        assert [f.name for f in out.cluster_by] == ["book_id", "trade_id"]

        assert (out["trade_date"].tags or {})[b"partition_by"] == b"true"
        assert (out["book_id"].tags or {})[b"cluster_by"] == b"true"

        assert out.metadata is not None
        assert b"partition_by" not in out.metadata
        assert b"cluster_by" not in out.metadata


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------


class TestExporters:

    def test_to_arrow_schema_carries_field_order_and_metadata(self) -> None:
        s = schema(
            [_int_field("qty", nullable=False), _str_field("book")],
            metadata={"name": "trade_row"},
        )

        out = s.to_arrow_schema()

        assert isinstance(out, pa.Schema)
        assert out.names == ["qty", "book"]
        assert out.field("qty").nullable is False
        assert out.metadata is not None
        assert out.metadata[b"name"] == b"trade_row"

    def test_to_field_lifts_to_struct_field(self) -> None:
        s = schema(
            [_int_field("qty"), _str_field("book")],
            metadata={"name": "trade_row", "nullable": "t"},
        )

        f = s.to_field()

        assert isinstance(f, Field)
        assert f.name == "trade_row"
        assert isinstance(f.dtype, StructType)
        assert f.nullable is True
        assert [child.name for child in f.dtype.fields] == ["qty", "book"]


class TestPolarsExport(PolarsTestCase):

    def test_to_polars_schema(self) -> None:
        s = schema([_int_field("qty"), _str_field("book")])

        out = s.to_polars_schema()

        self.assertIsInstance(out, self.pl.Schema)
        self.assertEqual(list(out.names()), ["qty", "book"])


class TestSparkExport(SparkTestCase):

    def test_to_spark_schema(self) -> None:
        s = schema([_int_field("qty", nullable=False), _str_field("book")])

        out = s.to_spark_schema()

        self.assertEqual(out.__class__.__name__, "StructType")
        self.assertEqual([f.name for f in out.fields], ["qty", "book"])
        self.assertFalse(out["qty"].nullable)


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


class TestConstructors:

    def test_from_field_unwraps_struct_into_schema(self) -> None:
        f = Field.from_str("trade_row: struct<qty:int64, book:string>")

        s = Schema.from_field(f)

        assert isinstance(s, Schema)
        assert list(s.keys()) == ["qty", "book"]

    def test_from_any_fields_preserves_input_order(self) -> None:
        s = Schema.from_any_fields(
            [_int_field("a"), _str_field("b"), _int_field("c")],
            metadata={"name": "ordered"},
        )

        assert list(s.keys()) == ["a", "b", "c"]
        assert s.name == "ordered"

    def test_from_any_and_from_with_field(self) -> None:
        f = _int_field("qty")

        s1 = Schema.from_any(f)
        s2 = Schema.from_(f)

        assert list(s1.keys()) == ["qty"]
        assert list(s2.keys()) == ["qty"]


# ---------------------------------------------------------------------------
# Internal helpers — _merge_metadata, _coerce_other
# ---------------------------------------------------------------------------


class TestInternalHelpers:

    def test_merge_metadata_right_wins_on_overlap(self) -> None:
        left = {b"a": b"1", b"b": b"2"}
        right = {b"b": b"x", b"c": b"3"}

        assert Schema._merge_metadata(left, right) == {
            b"a": b"1",
            b"b": b"x",
            b"c": b"3",
        }

    def test_coerce_other_lifts_field_to_schema(self) -> None:
        f = _int_field("qty")

        out = Schema._coerce_other(f)

        assert isinstance(out, Schema)
        assert list(out.keys()) == ["qty"]


# ---------------------------------------------------------------------------
# from_path — load schema from a parquet file
# ---------------------------------------------------------------------------


class TestFromPath:

    def test_discovers_field_and_table_metadata(self, tmp_path) -> None:
        arrow_schema = pa.schema(
            [
                pa.field("x", pa.int64(), metadata={b"comment": b"the x"}),
                pa.field("s", pa.string()),
            ],
            metadata={b"author": b"ygg"},
        )
        table = pa.table(
            {"x": [1, 2, 3], "s": ["a", "b", "c"]}, schema=arrow_schema
        )
        path = tmp_path / "data.parquet"
        pq.write_table(table, path)

        out = Schema.from_path(path)

        assert [f.name for f in out.fields] == ["x", "s"]
        assert out["x"].arrow_type == pa.int64()
        assert out["s"].arrow_type == pa.string()
        assert out.metadata == {b"author": b"ygg"}
        assert out["x"].metadata == {b"comment": b"the x"}

    def test_minimal_parquet_round_trip(self, tmp_path) -> None:
        path = tmp_path / "data.parquet"
        pq.write_table(pa.table({"x": [1, 2]}), path)

        assert [f.name for f in Schema.from_path(path).fields] == ["x"]

    def test_field_from_path_returns_struct_field(self, tmp_path) -> None:
        path = tmp_path / "data.parquet"
        pq.write_table(pa.table({"x": [1, 2], "s": ["a", "b"]}), path)

        out = Field.from_path(path)

        assert isinstance(out.dtype, StructType)
        assert [f.name for f in out.dtype.fields] == ["x", "s"]
