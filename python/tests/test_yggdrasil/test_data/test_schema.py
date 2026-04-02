"""Unit tests for yggdrasil/data/schema.py"""
from __future__ import annotations

from collections import OrderedDict

import pyarrow as pa
import pytest

from yggdrasil.data.field import Field
from yggdrasil.data.schema import Schema, schema


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _f(name: str, dtype: pa.DataType = pa.int64(), *, nullable: bool = True) -> Field:
    return Field(name=name, arrow_type=dtype, nullable=nullable)


def _simple_schema(*names: str) -> Schema:
    return Schema.from_fields([_f(n) for n in names])


# ============================================================================
# schema() factory
# ============================================================================


class TestSchemaFactory:
    def test_basic(self):
        s = schema([_f("a"), _f("b")])
        assert s.names == ("a", "b")

    def test_with_metadata(self):
        s = schema([_f("x")], metadata={"comment": "test"})
        assert s.metadata[b"comment"] == b"test"

    def test_with_tags(self):
        s = schema([_f("x")], tags={"env": "prod"})
        assert s.metadata[b"t:env"] == b"prod"

    def test_returns_schema_instance(self):
        assert isinstance(schema([_f("x")]), Schema)


# ============================================================================
# Schema.__post_init__ — construction variants
# ============================================================================


class TestSchemaConstruction:
    def test_from_ordered_dict_of_fields(self):
        od = OrderedDict([("a", _f("a")), ("b", _f("b"))])
        s = Schema(inner_fields=od)
        assert s.names == ("a", "b")

    def test_from_list_of_fields(self):
        s = Schema(inner_fields=[_f("x"), _f("y")])
        assert s.names == ("x", "y")

    def test_from_mapping_renames_if_key_differs(self):
        s = Schema(inner_fields={"alias": _f("original")})
        # key wins → field is renamed
        assert "alias" in s
        assert s["alias"].name == "alias"

    def test_from_list_of_arrow_fields(self):
        s = Schema(inner_fields=[pa.field("p", pa.float32()), pa.field("q", pa.string())])
        assert s.names == ("p", "q")

    def test_empty_schema(self):
        s = Schema()
        assert len(s) == 0


# ============================================================================
# Schema.tags property & setter / update_tags
# ============================================================================


class TestSchemaTags:
    def test_tags_none_when_no_metadata(self):
        s = _simple_schema("a")
        assert s.tags is None

    def test_tags_returned_without_prefix(self):
        s = Schema(inner_fields=[_f("x")], metadata={b"t:env": b"prod", b"other": b"v"})
        assert s.tags == {b"env": b"prod"}

    def test_tags_setter_sets_metadata(self):
        s = _simple_schema("a")
        s.tags = {"env": "staging"}
        assert s.metadata[b"t:env"] == b"staging"

    def test_tags_setter_merges_with_existing_metadata(self):
        s = Schema(inner_fields=[_f("x")], metadata={b"comment": b"hi"})
        s.tags = {"env": "prod"}
        assert s.metadata[b"comment"] == b"hi"
        assert s.metadata[b"t:env"] == b"prod"

    def test_update_tags(self):
        s = _simple_schema("a")
        s.update_tags({"role": "dim"})
        assert s.metadata[b"t:role"] == b"dim"

    def test_update_tags_none_is_noop(self):
        s = _simple_schema("a")
        s.update_tags(None)
        assert s.metadata is None


# ============================================================================
# Schema properties
# ============================================================================


class TestSchemaProperties:
    def test_names(self):
        s = _simple_schema("a", "b", "c")
        assert s.names == ("a", "b", "c")

    def test_fields(self):
        s = _simple_schema("x", "y")
        assert all(isinstance(f, Field) for f in s.fields)
        assert [f.name for f in s.fields] == ["x", "y"]

    def test_arrow_fields(self):
        s = _simple_schema("x", "y")
        afs = s.arrow_fields
        assert all(isinstance(f, pa.Field) for f in afs)
        assert [f.name for f in afs] == ["x", "y"]

    def test_partition_by(self):
        s = Schema.from_fields([
            Field("dt", pa.date32(), metadata={b"t:partition_by": b"true"}),
            _f("id"),
        ])
        pb = s.partition_by
        assert len(pb) == 1
        assert pb[0].name == "dt"

    def test_cluster_by(self):
        s = Schema.from_fields([
            _f("id"),
            Field("region", pa.string(), metadata={b"t:cluster_by": b"true"}),
        ])
        cb = s.cluster_by
        assert len(cb) == 1
        assert cb[0].name == "region"

    def test_comment_from_metadata(self):
        s = Schema(inner_fields=[_f("x")], metadata={b"comment": b"my table"})
        assert s.comment == "my table"

    def test_comment_from_description(self):
        s = Schema(inner_fields=[_f("x")], metadata={b"description": b"desc text"})
        assert s.comment == "desc text"

    def test_comment_none_when_missing(self):
        s = _simple_schema("x")
        assert s.comment is None


# ============================================================================
# Schema.copy
# ============================================================================


class TestSchemaCopy:
    def test_copy_is_independent(self):
        s = _simple_schema("a", "b")
        c = s.copy()
        c["c"] = _f("c")
        assert "c" not in s

    def test_copy_preserves_metadata(self):
        s = Schema(inner_fields=[_f("x")], metadata={b"k": b"v"})
        c = s.copy()
        assert c.metadata == {b"k": b"v"}

    def test_copy_with_new_fields(self):
        s = _simple_schema("a", "b")
        c = s.copy(fields=[_f("z")])
        assert c.names == ("z",)

    def test_copy_with_new_metadata(self):
        s = Schema(inner_fields=[_f("x")], metadata={b"old": b"v"})
        c = s.copy(metadata={"new": "w"})
        assert c.metadata == {b"new": b"w"}

    def test_copy_with_tags(self):
        s = _simple_schema("x")
        c = s.copy(tags={"env": "test"})
        assert c.metadata[b"t:env"] == b"test"


# ============================================================================
# Schema.append / extend
# ============================================================================


class TestSchemaAppendExtend:
    def test_append_single(self):
        s = _simple_schema("a")
        s2 = s.append(_f("b"))
        assert s2.names == ("a", "b")
        assert s.names == ("a",)  # original unchanged

    def test_append_multiple(self):
        s = _simple_schema("a")
        s2 = s.append(_f("b"), _f("c"))
        assert s2.names == ("a", "b", "c")

    def test_append_overwrites_existing(self):
        s = _simple_schema("a", "b")
        s2 = s.append(Field("a", pa.string()))
        assert s2["a"].arrow_type == pa.string()

    def test_extend_from_iterable(self):
        s = _simple_schema("a")
        s2 = s.extend([_f("b"), _f("c")])
        assert s2.names == ("a", "b", "c")


# ============================================================================
# Schema MutableMapping interface
# ============================================================================


class TestSchemaMutableMapping:
    def test_getitem(self):
        s = _simple_schema("a", "b")
        assert s["a"].name == "a"

    def test_getitem_missing_raises(self):
        s = _simple_schema("a")
        with pytest.raises(KeyError):
            _ = s["z"]

    def test_setitem_new_field(self):
        s = _simple_schema("a")
        s["b"] = _f("b")
        assert "b" in s

    def test_setitem_renames_to_key(self):
        s = _simple_schema("a")
        s["alias"] = _f("original")
        assert s["alias"].name == "alias"

    def test_setitem_accepts_arrow_field(self):
        s = _simple_schema("a")
        s["z"] = pa.field("z", pa.string())
        assert s["z"].arrow_type == pa.string()

    def test_delitem(self):
        s = _simple_schema("a", "b")
        del s["a"]
        assert "a" not in s
        assert s.names == ("b",)

    def test_delitem_missing_raises(self):
        s = _simple_schema("a")
        with pytest.raises(KeyError):
            del s["z"]

    def test_iter(self):
        s = _simple_schema("a", "b", "c")
        assert list(s) == ["a", "b", "c"]

    def test_len(self):
        s = _simple_schema("a", "b")
        assert len(s) == 2

    def test_contains(self):
        s = _simple_schema("a")
        assert "a" in s
        assert "z" not in s


# ============================================================================
# Schema set operators: +, -, &, |
# ============================================================================


class TestSchemaOperators:
    def setup_method(self):
        self.left = _simple_schema("a", "b", "c")
        self.right = _simple_schema("b", "c", "d")

    def test_add_merges_fields(self):
        result = self.left + self.right
        assert set(result.names) == {"a", "b", "c", "d"}

    def test_add_right_wins_on_overlap(self):
        left = Schema.from_fields([Field("x", pa.int32())])
        right = Schema.from_fields([Field("x", pa.string())])
        result = left + right
        assert result["x"].arrow_type == pa.string()

    def test_add_preserves_order(self):
        result = self.left + self.right
        # left fields first, then new from right
        assert result.names[0] == "a"
        assert "d" in result.names

    def test_radd_zero(self):
        result = 0 + self.left
        assert result.names == self.left.names

    def test_iadd(self):
        s = _simple_schema("a", "b")
        s += _simple_schema("c")
        assert "c" in s

    def test_sub_removes_fields(self):
        result = self.left - self.right
        assert result.names == ("a",)

    def test_isub(self):
        s = _simple_schema("a", "b", "c")
        s -= _simple_schema("b")
        assert s.names == ("a", "c")

    def test_and_keeps_intersection(self):
        result = self.left & self.right
        assert set(result.names) == {"b", "c"}

    def test_iand(self):
        s = _simple_schema("a", "b", "c")
        s &= _simple_schema("b", "c")
        assert s.names == ("b", "c")

    def test_or_same_as_add(self):
        result = self.left | self.right
        assert set(result.names) == set((self.left + self.right).names)

    def test_ror(self):
        result = self.right | self.left
        assert set(result.names) == {"a", "b", "c", "d"}

    def test_ior(self):
        s = _simple_schema("a")
        s |= _simple_schema("b")
        assert "b" in s

    def test_sub_with_arrow_schema(self):
        other = pa.schema([pa.field("b", pa.int64())])
        result = self.left - other
        assert "b" not in result

    def test_and_with_arrow_schema(self):
        other = pa.schema([pa.field("a", pa.int64()), pa.field("c", pa.int64())])
        result = self.left & other
        assert set(result.names) == {"a", "c"}


# ============================================================================
# Schema metadata merge in operators
# ============================================================================


class TestSchemaMetadataMerge:
    def test_add_merges_metadata(self):
        left = Schema(inner_fields=[_f("a")], metadata={b"k1": b"v1"})
        right = Schema(inner_fields=[_f("b")], metadata={b"k2": b"v2"})
        result = left + right
        assert result.metadata[b"k1"] == b"v1"
        assert result.metadata[b"k2"] == b"v2"

    def test_sub_keeps_left_metadata(self):
        left = Schema(inner_fields=[_f("a"), _f("b")], metadata={b"k": b"v"})
        right = _simple_schema("b")
        result = left - right
        assert result.metadata == {b"k": b"v"}


# ============================================================================
# Schema.autotag
# ============================================================================


class TestSchemaAutotag:
    def _tag(self, f: Field, key: str) -> str | None:
        raw = f.metadata or {}
        v = raw.get(b"t:" + key.encode())
        return v.decode() if v else None

    def test_autotag_all_fields(self):
        s = Schema.from_fields([
            Field("trade_id", pa.int64()),
            Field("price", pa.float64()),
            Field("ts", pa.timestamp("us")),
        ])
        tagged = s.autotag()
        assert self._tag(tagged["trade_id"], "role") == "identifier"
        assert self._tag(tagged["price"], "role") == "price"
        assert self._tag(tagged["ts"], "role") == "event_time"

    def test_autotag_with_schema_tags(self):
        s = Schema.from_fields([_f("x")])
        tagged = s.autotag(tags={"layer": "gold"})
        assert tagged.metadata[b"t:layer"] == b"gold"

    def test_autotag_returns_new_schema(self):
        s = Schema.from_fields([_f("x")])
        tagged = s.autotag()
        assert tagged is not s


# ============================================================================
# Schema.from_any / from_fields / from_arrow / to_arrow_schema
# ============================================================================


class TestSchemaConversion:
    def test_from_any_returns_same_schema(self):
        s = _simple_schema("a")
        assert Schema.from_any(s) is s

    def test_from_any_from_arrow_schema(self):
        arrow = pa.schema([pa.field("x", pa.int32()), pa.field("y", pa.string())])
        s = Schema.from_any(arrow)
        assert s.names == ("x", "y")

    def test_from_fields(self):
        s = Schema.from_fields([_f("a"), _f("b")])
        assert s.names == ("a", "b")

    def test_from_fields_with_arrow_fields(self):
        s = Schema.from_fields([pa.field("p", pa.float32()), pa.field("q", pa.bool_())])
        assert s.names == ("p", "q")

    def test_from_arrow(self):
        arrow = pa.schema([pa.field("m", pa.int64(), nullable=False)])
        s = Schema.from_arrow(arrow)
        assert s["m"].nullable is False

    def test_from_arrow_schema(self):
        arrow = pa.schema([pa.field("n", pa.string())])
        s = Schema.from_arrow_schema(arrow)
        assert "n" in s

    def test_from_arrow_preserves_metadata(self):
        arrow = pa.schema([pa.field("x", pa.int32())], metadata={b"k": b"v"})
        s = Schema.from_arrow(arrow)
        assert s.metadata == {b"k": b"v"}

    def test_to_arrow_schema(self):
        s = Schema.from_fields([_f("a"), _f("b", pa.string())])
        arrow = s.to_arrow_schema()
        assert isinstance(arrow, pa.Schema)
        assert arrow.names == ["a", "b"]

    def test_to_arrow_schema_preserves_metadata(self):
        s = Schema(inner_fields=[_f("x")], metadata={b"m": b"val"})
        arrow = s.to_arrow_schema()
        assert arrow.metadata[b"m"] == b"val"

    def test_roundtrip_arrow(self):
        s = Schema.from_fields([_f("x", pa.float32()), _f("y", pa.bool_())])
        s2 = Schema.from_arrow(s.to_arrow_schema())
        assert s2.names == s.names
        assert s2["x"].arrow_type == s["x"].arrow_type


# ============================================================================
# Schema.from_polars / to_polars_schema
# ============================================================================


class TestSchemaPolars:
    @pytest.fixture(autouse=True)
    def _skip_if_no_polars(self):
        pytest.importorskip("polars")

    def test_from_polars_schema(self):
        import polars as pl

        pl_schema = pl.Schema({"a": pl.Int64(), "b": pl.String()})
        s = Schema.from_polars(pl_schema)
        assert s.names == ("a", "b")
        assert s["a"].arrow_type == pa.int64()

    def test_from_polars_dataframe(self):
        import polars as pl

        df = pl.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})
        s = Schema.from_polars(df)
        assert "x" in s
        assert "y" in s

    def test_from_polars_lazyframe(self):
        import polars as pl

        lf = pl.LazyFrame({"x": [1], "y": ["a"]})
        s = Schema.from_polars(lf)
        assert s.names == ("x", "y")

    def test_from_polars_invalid_raises(self):
        with pytest.raises(TypeError):
            Schema.from_polars("not_a_polars_object")

    def test_to_polars_schema(self):
        import polars as pl

        s = Schema.from_fields([_f("a"), _f("b", pa.string())])
        pl_schema = s.to_polars_schema()
        assert isinstance(pl_schema, pl.Schema)
        assert "a" in pl_schema
        assert "b" in pl_schema

    def test_roundtrip_polars(self):
        import polars as pl

        pl_schema = pl.Schema({"x": pl.Float32(), "y": pl.Boolean()})
        s = Schema.from_polars(pl_schema)
        result = s.to_polars_schema()
        assert result["x"] == pl.Float32()
        assert result["y"] == pl.Boolean()


# ============================================================================
# Schema.cast_table / cast_unstructured
# ============================================================================


class TestSchemaCast:
    def test_cast_arrow_table(self):
        s = Schema.from_fields([
            Field("a", pa.int64()),
            Field("b", pa.string()),
        ])
        table = pa.table({"a": pa.array([1, 2, 3], pa.int32()), "b": pa.array(["x", "y", "z"])})
        result = s.cast_table(table)
        assert result.schema.field("a").type == pa.int64()

    def test_cast_unstructured_to_table(self):
        s = Schema.from_fields([
            Field("a", pa.int64()),
            Field("b", pa.string()),
        ])
        raw = {"a": [1, 2], "b": ["x", "y"]}
        result = s.cast_unstructured(raw, as_type=pa.Table)
        assert isinstance(result, pa.Table)
        assert result.num_rows == 2

