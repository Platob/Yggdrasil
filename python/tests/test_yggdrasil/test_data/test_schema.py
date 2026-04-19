from __future__ import annotations

from collections import OrderedDict

import pyarrow as pa
import pytest
from yggdrasil.data.constants import DEFAULT_FIELD_NAME

from yggdrasil.data.data_field import Field
from yggdrasil.data.schema import Schema, schema
from yggdrasil.data.types.nested import StructType
from yggdrasil.data.types.primitive import IntegerType, StringType, TimestampType
from yggdrasil.polars.tests import PolarsTestCase
from yggdrasil.spark.tests import SparkTestCase


def _field_int(
    name: str,
    *,
    nullable: bool = True,
    metadata: dict[bytes | str, bytes | str | object] | None = None,
    tags: dict[bytes | str, bytes | str | object] | None = None,
    default=...,
) -> Field:
    kwargs: dict[str, object] = {
        "name": name,
        "dtype": IntegerType(byte_size=8, signed=True),
        "nullable": nullable,
        "metadata": metadata,
        "tags": tags,
    }
    if default is not ...:
        kwargs["default"] = default
    return Field(**kwargs)


def _field_str(
    name: str,
    *,
    nullable: bool = True,
    metadata: dict[bytes | str, bytes | str | object] | None = None,
    tags: dict[bytes | str, bytes | str | object] | None = None,
    default=...,
) -> Field:
    kwargs: dict[str, object] = {
        "name": name,
        "dtype": StringType(),
        "nullable": nullable,
        "metadata": metadata,
        "tags": tags,
    }
    if default is not ...:
        kwargs["default"] = default
    return Field(**kwargs)


def test_schema_helper_accepts_iterable_and_varargs():
    s = schema(
        [_field_int("a"), _field_str("b")],
        _field_int("c"),
        metadata={"name": "trade_row", "comment": "schema comment"},
    )

    assert list(s.keys()) == ["a", "b", "c"]
    assert s.name == "trade_row"
    assert s.comment == "schema comment"


def test_schema_helper_accepts_single_non_iterable_field():
    s = schema(_field_int("a"))

    assert list(s.keys()) == ["a"]
    assert isinstance(s["a"], Field)


def test_post_init_normalizes_mapping_input_and_rewrites_names():
    s = Schema(
        inner_fields={
            "qty": _field_int("wrong_name"),
            "book": _field_str("book"),
        },
        metadata=None,
    )

    assert list(s.keys()) == ["qty", "book"]
    assert s["qty"].name == "qty"
    assert s["book"].name == "book"


def test_fields_and_children_fields_are_tuples_in_order():
    s = schema([_field_int("a"), _field_str("b")])

    assert isinstance(s.fields, tuple)
    assert isinstance(s.children_fields, tuple)
    assert [f.name for f in s.fields] == ["a", "b"]
    assert [f.name for f in s.children_fields] == ["a", "b"]


def test_name_defaults_to_root_and_can_be_set():
    s = schema([_field_int("a")])

    assert s.name == DEFAULT_FIELD_NAME

    s.name = "orders"

    assert s.name == "orders"
    assert s.metadata is not None
    assert s.metadata[b"name"] == b"orders"


def test_name_setter_falsey_value_falls_back_to_root():
    s = schema([_field_int("a")], metadata={"name": "orders"})

    s.name = ""

    assert s.name == "orders"
    assert s.metadata is not None
    assert s.metadata[b"name"] == b"orders"


def test_nullable_defaults_false_and_can_be_enabled():
    s = schema([_field_int("a")])

    assert s.nullable is False

    s.nullable = True

    assert s.nullable is True
    assert s.metadata is not None
    assert s.metadata[b"nullable"] == b"t"


def test_nullable_setter_does_not_write_false_when_missing_metadata():
    s = schema([_field_int("a")])

    s.nullable = False

    assert s.nullable is False
    assert s.metadata is None or b"nullable" not in s.metadata


def test_comment_reads_comment_then_description():
    s1 = schema([_field_int("a")], metadata={"comment": "hello"})
    s2 = schema([_field_int("a")], metadata={"description": "world"})
    s3 = schema([_field_int("a")])

    assert s1.comment == "hello"
    assert s2.comment == "world"
    assert s3.comment is None


def test_dtype_returns_struct_type():
    s = schema([_field_int("qty"), _field_str("book")])

    dtype = s.dtype

    assert isinstance(dtype, StructType)
    assert [f.name for f in dtype.fields] == ["qty", "book"]


def test_arrow_fields_returns_arrow_fields():
    s = schema([_field_int("qty"), _field_str("book")])

    arrow_fields = s.arrow_fields

    assert len(arrow_fields) == 2
    assert all(isinstance(f, pa.Field) for f in arrow_fields)
    assert [f.name for f in arrow_fields] == ["qty", "book"]


def test_partition_cluster_primary_foreign_key_properties():
    s = schema(
        [
            _field_int("date_id", tags={"partition_by": "true"}),
            _field_int("book_id", tags={"cluster_by": "true", "primary_key": "true"}),
            _field_int("trade_id", tags={"primary_key": "true"}),
            _field_int("counterparty_id", tags={"foreign_key": "dim_counterparty.id"}),
        ]
    )

    assert [f.name for f in s.partition_by] == ["date_id"]
    assert [f.name for f in s.cluster_by] == ["book_id"]
    assert [f.name for f in s.primary_keys] == ["book_id", "trade_id"]
    assert s.primary_key_names == ["book_id", "trade_id"]
    assert [f.name for f in s.foreign_keys] == ["counterparty_id"]
    assert s.foreign_key_names == ["counterparty_id"]
    assert s.foreign_key_refs == {"counterparty_id": "dim_counterparty.id"}


def test_copy_preserves_fields_and_metadata_but_deep_copies_field_objects():
    s1 = schema(
        [_field_int("a"), _field_str("b")],
        metadata={"name": "x", "comment": "y"},
    )

    s2 = s1.copy()

    assert s2 is not s1
    assert list(s2.keys()) == ["a", "b"]
    assert s2.metadata == s1.metadata
    assert s2["a"] is not s1["a"]
    assert s2["b"] is not s1["b"]


def test_copy_can_replace_fields_and_metadata():
    s1 = schema([_field_int("a")], metadata={"name": "old"})
    s2 = s1.copy(
        fields=[_field_str("b")],
        metadata={"name": "new", "comment": "desc"},
    )

    assert list(s2.keys()) == ["b"]
    assert s2.name == "new"
    assert s2.comment == "desc"


def test_append_returns_new_schema():
    s1 = schema([_field_int("a")])
    s2 = s1.append(_field_str("b"))

    assert list(s1.keys()) == ["a"]
    assert list(s2.keys()) == ["a", "b"]


def test_extend_returns_new_schema():
    s1 = schema([_field_int("a")])
    s2 = s1.extend([_field_str("b"), _field_int("c")])

    assert list(s1.keys()) == ["a"]
    assert list(s2.keys()) == ["a", "b", "c"]


def test_getitem_by_name():
    s = schema([_field_int("a"), _field_str("b")])

    assert s["a"].name == "a"
    assert s["b"].name == "b"


def test_getitem_by_index():
    s = schema([_field_int("a"), _field_str("b")])

    assert s[0].name == "a"
    assert s[1].name == "b"


def test_setitem_rewrites_field_name_to_key():
    s = schema([_field_int("a")])

    s["b"] = _field_str("wrong")

    assert list(s.keys()) == ["a", "b"]
    assert s["b"].name == "b"


def test_setitem_rejects_non_string_key():
    s = schema([_field_int("a")])

    with pytest.raises(TypeError):
        s[1] = _field_int("b")  # type: ignore[index]


def test_delitem_by_name_and_index():
    s = schema([_field_int("a"), _field_str("b"), _field_int("c")])

    del s["b"]
    assert list(s.keys()) == ["a", "c"]

    del s[0]
    assert list(s.keys()) == ["c"]


def test_iter_len_contains():
    s = schema([_field_int("a"), _field_str("b")])

    assert list(iter(s)) == ["a", "b"]
    assert len(s) == 2
    assert "a" in s
    assert "b" in s
    assert "c" not in s
    assert 0 in s
    assert 1 in s
    assert 2 not in s


def test_get_returns_default_for_missing():
    s = schema([_field_int("a")])

    assert s.get("a").name == "a"
    assert s.get("missing") is None
    assert s.get("missing", "fallback") == "fallback"


def test_pop_by_name_and_index_and_default():
    s = schema([_field_int("a"), _field_str("b")])

    popped_a = s.pop("a")
    assert popped_a.name == "a"
    assert list(s.keys()) == ["b"]

    popped_b = s.pop(0)
    assert popped_b.name == "b"
    assert list(s.keys()) == []

    assert s.pop("missing", "fallback") == "fallback"


def test_pop_raises_for_missing_without_default():
    s = schema([_field_int("a")])

    with pytest.raises(KeyError):
        s.pop("missing")

    with pytest.raises(IndexError):
        s.pop(10)


def test_setdefault_existing_returns_existing():
    s = schema([_field_int("a")])

    out = s.setdefault("a", _field_str("other"))

    assert out.name == "a"
    assert isinstance(out.dtype, IntegerType)
    assert list(s.keys()) == ["a"]


def test_setdefault_new_inserts_and_rewrites_name():
    s = schema([_field_int("a")])

    out = s.setdefault("b", _field_str("wrong"))

    assert out.name == "b"
    assert list(s.keys()) == ["a", "b"]
    assert s["b"].name == "b"


def test_setdefault_rejects_non_string_key():
    s = schema([_field_int("a")])

    with pytest.raises(TypeError):
        s.setdefault(1, _field_int("x"))  # type: ignore[arg-type]


def test_setdefault_requires_default_for_new_key():
    s = schema([_field_int("a")])

    with pytest.raises(ValueError):
        s.setdefault("b")


def test_popitem_last_and_first():
    s = schema([_field_int("a"), _field_str("b"), _field_int("c")])

    key_last, field_last = s.popitem()
    assert key_last == "c"
    assert field_last.name == "c"

    key_first, field_first = s.popitem(last=False)
    assert key_first == "a"
    assert field_first.name == "a"

    assert list(s.keys()) == ["b"]


def test_clear_removes_all_fields():
    s = schema([_field_int("a"), _field_str("b")])

    s.clear()

    assert len(s) == 0
    assert list(s.keys()) == []


def test_add_merges_and_rhs_overrides_duplicate_names():
    s1 = schema(
        [_field_int("a"), _field_str("b")],
        metadata={"comment": "left", "name": "left_name"},
    )
    s2 = schema(
        [_field_int("b", nullable=False), _field_int("c")],
        metadata={"description": "right_desc", "name": "right_name"},
    )

    out = s1 + s2

    assert list(out.keys()) == ["a", "b", "c"]
    assert out["b"].nullable is False
    assert out.metadata is not None
    assert out.metadata[b"name"] == b"right_name"
    assert out.metadata[b"comment"] == b"left"
    assert out.metadata[b"description"] == b"right_desc"


def test_radd_with_zero_returns_copy():
    s = schema([_field_int("a")], metadata={"name": "x"})

    out = 0 + s

    assert out is not s
    assert list(out.keys()) == ["a"]
    assert out.metadata == s.metadata


def test_iadd_mutates_in_place():
    s1 = schema([_field_int("a")], metadata={"comment": "left"})
    s2 = schema([_field_str("b")], metadata={"name": "right"})

    original_id = id(s1)
    s1 += s2

    assert id(s1) == original_id
    assert list(s1.keys()) == ["a", "b"]
    assert s1.metadata is not None
    assert s1.metadata[b"comment"] == b"left"
    assert s1.metadata[b"name"] == b"right"


def test_sub_returns_difference():
    s1 = schema([_field_int("a"), _field_str("b"), _field_int("c")], metadata={"name": "x"})
    s2 = schema([_field_str("b")])

    out = s1 - s2

    assert list(out.keys()) == ["a", "c"]
    assert out.metadata == s1.metadata


def test_isub_mutates_in_place():
    s1 = schema([_field_int("a"), _field_str("b"), _field_int("c")])
    s2 = schema([_field_str("b"), _field_int("x")])

    s1 -= s2

    assert list(s1.keys()) == ["a", "c"]


def test_and_returns_intersection_in_left_order():
    s1 = schema(
        [_field_int("a"), _field_str("b"), _field_int("c")],
        metadata={"comment": "left"},
    )
    s2 = schema(
        [_field_int("c"), _field_int("a")],
        metadata={"name": "right"},
    )

    out = s1 & s2

    assert list(out.keys()) == ["a", "c"]
    assert out.metadata is not None
    assert out.metadata[b"comment"] == b"left"
    assert out.metadata[b"name"] == b"right"


def test_iand_mutates_to_intersection():
    s1 = schema([_field_int("a"), _field_str("b"), _field_int("c")], metadata={"comment": "left"})
    s2 = schema([_field_int("c"), _field_int("a")], metadata={"name": "right"})

    s1 &= s2

    assert list(s1.keys()) == ["a", "c"]
    assert s1.metadata is not None
    assert s1.metadata[b"comment"] == b"left"
    assert s1.metadata[b"name"] == b"right"


def test_or_and_ior_delegate_to_add():
    s1 = schema([_field_int("a")], metadata={"comment": "left"})
    s2 = schema([_field_str("b")], metadata={"name": "right"})

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


def test_autotag_returns_new_schema_and_updates_metadata_tags():
    s = schema([_field_int("a")], metadata={"name": DEFAULT_FIELD_NAME})
    out = s.autotag(tags={"semantic_type": "fact"})

    assert list(out.keys()) == ["a"]
    assert out.metadata is not None
    assert out.metadata != s.metadata or out.metadata is s.metadata


def test_autotag_propagates_dtype_and_field_tags_per_column():
    s = schema(
        [
            _field_int("user_id", nullable=False),
            _field_str("email"),
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
    assert id_tags[b"kind"] == b"integer"
    assert id_tags[b"signed"] == b"true"
    assert id_tags[b"nullable"] == b"false"
    assert id_tags[b"role"] == b"identifier"
    assert id_tags[b"primary_key"] == b"true"

    email_tags = out["email"].tags or {}
    assert email_tags[b"kind"] == b"string"
    assert email_tags[b"pii"] == b"email"

    ts_tags = out["created_at"].tags or {}
    assert ts_tags[b"kind"] == b"timestamp"
    assert ts_tags[b"unit"] == b"us"
    assert ts_tags[b"tz"] == b"UTC"
    assert ts_tags[b"role"] == b"audit_timestamp"

    assert out.metadata is not None
    assert out.metadata[b"t:layer"] == b"silver"


def test_autotag_pops_partition_by_and_cluster_by_metadata():
    s = schema(
        [
            _field_int("trade_date"),
            _field_int("book_id"),
            _field_int("trade_id"),
        ],
        metadata={
            "partition_by": "trade_date",
            "cluster_by": '["book_id", "trade_id"]',
        },
    )

    out = s.autotag()

    assert [f.name for f in out.partition_by] == ["trade_date"]
    assert [f.name for f in out.cluster_by] == ["book_id", "trade_id"]

    date_tags = out["trade_date"].tags or {}
    assert date_tags[b"partition_by"] == b"true"

    book_tags = out["book_id"].tags or {}
    assert book_tags[b"cluster_by"] == b"true"

    # Keys popped off the schema-level metadata so they don't leak to Arrow/Delta.
    assert out.metadata is not None
    assert b"partition_by" not in out.metadata
    assert b"cluster_by" not in out.metadata


def test_to_arrow_schema():
    s = schema(
        [_field_int("qty", nullable=False), _field_str("book")],
        metadata={"name": "trade_row"},
    )

    out = s.to_arrow_schema()

    assert isinstance(out, pa.Schema)
    assert out.names == ["qty", "book"]
    assert out.field("qty").nullable is False
    assert out.metadata is not None
    assert out.metadata[b"name"] == b"trade_row"


def test_to_field_round_trip_shape():
    s = schema(
        [_field_int("qty"), _field_str("book")],
        metadata={"name": "trade_row", "nullable": "t"},
    )

    f = s.to_field()

    assert isinstance(f, Field)
    assert f.name == "trade_row"
    assert isinstance(f.dtype, StructType)
    assert f.nullable is True
    assert [child.name for child in f.dtype.fields] == ["qty", "book"]


def test_from_field():
    f = Field.from_str("trade_row: struct<qty:int64, book:string>")

    s = Schema.from_field(f)

    assert isinstance(s, Schema)
    assert list(s.keys()) == ["qty", "book"]


def test_from_any_fields_preserves_order():
    s = Schema.from_any_fields(
        [_field_int("a"), _field_str("b"), _field_int("c")],
        metadata={"name": "ordered"},
    )

    assert list(s.keys()) == ["a", "b", "c"]
    assert s.name == "ordered"


def test_from_any_and_from__with_field():
    f = _field_int("qty")

    s1 = Schema.from_any(f)
    s2 = Schema.from_(f)

    assert list(s1.keys()) == ["qty"]
    assert list(s2.keys()) == ["qty"]


def test_merge_metadata_staticmethod():
    left = {b"a": b"1", b"b": b"2"}
    right = {b"b": b"x", b"c": b"3"}

    out = Schema._merge_metadata(left, right)

    assert out == {b"a": b"1", b"b": b"x", b"c": b"3"}


def test_coerce_other_from_field():
    f = _field_int("qty")

    out = Schema._coerce_other(f)

    assert isinstance(out, Schema)
    assert list(out.keys()) == ["qty"]


class TestSchemaPolars(PolarsTestCase):

    def test_to_polars_schema(self):
        s = schema([_field_int("qty"), _field_str("book")])

        out = s.to_polars_schema()

        self.assertIsInstance(out, self.pl.Schema)
        self.assertEqual(list(out.names()), ["qty", "book"])


class TestSchemaSpark(SparkTestCase):

    def test_to_spark_schema(self):
        s = schema([_field_int("qty", nullable=False), _field_str("book")])

        out = s.to_spark_schema()

        self.assertEqual(out.__class__.__name__, "StructType")
        self.assertEqual([f.name for f in out.fields], ["qty", "book"])
        self.assertFalse(out["qty"].nullable)


def test_ordered_dict_input_is_preserved():
    inner = OrderedDict(
        [
            ("a", _field_int("a")),
            ("b", _field_str("b")),
            ("c", _field_int("c")),
        ]
    )

    s = Schema(inner_fields=inner)

    assert list(s.keys()) == ["a", "b", "c"]


def test_schema_from_path_discovers_fields_and_metadata(tmp_path):
    import pyarrow.parquet as pq

    arrow_schema = pa.schema(
        [
            pa.field("x", pa.int64(), metadata={b"comment": b"the x"}),
            pa.field("s", pa.string()),
        ],
        metadata={b"author": b"ygg"},
    )
    table = pa.table({"x": [1, 2, 3], "s": ["a", "b", "c"]}, schema=arrow_schema)
    path = tmp_path / "data.parquet"
    pq.write_table(table, path)

    out = Schema.from_path(path)

    assert [f.name for f in out.fields] == ["x", "s"]
    assert out["x"].arrow_type == pa.int64()
    assert out["s"].arrow_type == pa.string()
    assert out.metadata == {b"author": b"ygg"}
    assert out["x"].metadata == {b"comment": b"the x"}


def test_schema_from_path_accepts_path_io_instance(tmp_path):
    import pyarrow.parquet as pq
    from yggdrasil.io.buffer.local_path_io import LocalPathIO

    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"x": [1, 2]}), path)

    out = Schema.from_path(LocalPathIO.make(path))

    assert [f.name for f in out.fields] == ["x"]


def test_schema_from_path_accepts_path_io_factory(tmp_path):
    import pyarrow.parquet as pq
    from yggdrasil.io.buffer.local_path_io import LocalPathIO

    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"x": [1, 2]}), path)

    out = Schema.from_path(path, path_io=LocalPathIO)

    assert [f.name for f in out.fields] == ["x"]


def test_field_from_path_returns_struct_field(tmp_path):
    import pyarrow.parquet as pq

    path = tmp_path / "data.parquet"
    pq.write_table(
        pa.table({"x": [1, 2], "s": ["a", "b"]}),
        path,
    )

    out = Field.from_path(path)

    assert isinstance(out.dtype, StructType)
    assert [f.name for f in out.dtype.fields] == ["x", "s"]