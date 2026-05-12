"""Field.alias + Field.select_in_* + CastOptions.match_by behaviour."""
from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data import Field, Schema
from yggdrasil.data.options import CastOptions


def _id_field() -> Field:
    return Field.from_(pa.field("id", pa.int64()))


# ---------------------------------------------------------------------------
# Field.alias / has_alias / set_alias / with_alias
# ---------------------------------------------------------------------------


class TestAlias:
    def test_alias_falls_back_to_name(self) -> None:
        f = _id_field()
        assert f.alias == "id"
        assert f.has_alias is False

    def test_set_alias_records_in_metadata(self) -> None:
        f = _id_field()
        f.set_alias("user_id")
        assert f.alias == "user_id"
        assert f.has_alias is True

    def test_set_alias_same_as_name_is_no_op(self) -> None:
        f = _id_field()
        f.set_alias("id")
        assert f.has_alias is False
        assert f.alias == "id"

    def test_set_alias_promotes_to_name_when_name_missing(self) -> None:
        f = Field.from_(pa.field("", pa.int64()))
        assert f.name == ""
        f.set_alias("user_id")
        assert f.name == "user_id"
        assert f.has_alias is False

    def test_set_alias_clear(self) -> None:
        f = _id_field()
        f.set_alias("user_id")
        f.set_alias(None)
        assert f.has_alias is False
        assert f.alias == "id"

    def test_with_alias_returns_copy(self) -> None:
        f = _id_field()
        g = f.with_alias("user_id")
        assert f.has_alias is False
        assert g.has_alias is True
        assert g.alias == "user_id"

    def test_with_alias_no_op_when_equal_to_name(self) -> None:
        f = _id_field()
        assert f.with_alias("id") is f

    def test_with_alias_promotes_to_name_when_name_missing(self) -> None:
        f = Field.from_(pa.field("", pa.int64()))
        g = f.with_alias("user_id")
        assert g.name == "user_id"
        assert g.has_alias is False


# ---------------------------------------------------------------------------
# Field.position
# ---------------------------------------------------------------------------


class TestPosition:
    def test_default_is_none(self) -> None:
        assert _id_field().position is None

    def test_set_position(self) -> None:
        f = _id_field()
        f.set_position(2)
        assert f.position == 2

    def test_set_position_clear(self) -> None:
        f = _id_field()
        f.set_position(2)
        f.set_position(None)
        assert f.position is None

    def test_with_position_returns_copy(self) -> None:
        f = _id_field()
        g = f.with_position(1)
        assert f.position is None
        assert g.position == 1

    def test_negative_position_rejected(self) -> None:
        with pytest.raises(ValueError):
            _id_field().set_position(-1)

    def test_position_used_as_fallback_in_arrow(self) -> None:
        t = pa.table({"a": [10, 20, 30], "b": ["x", "y", "z"]})
        # Field name doesn't match any column; alias none; position
        # picks column index 1 → "b".
        f = Field.from_(pa.field("c", pa.string())).with_position(1)
        out = f.select_in_arrow_tabular(t)
        assert out.to_pylist() == ["x", "y", "z"]

    def test_position_used_as_fallback_in_field(self) -> None:
        schema = Field.from_(pa.schema([("a", pa.int64()), ("b", pa.string())]))
        f = Field.from_(pa.field("c", pa.string())).with_position(1)
        matched = f.select_in_field(schema)
        assert matched is not None and matched.name == "b"


# ---------------------------------------------------------------------------
# select_in_arrow_tabular
# ---------------------------------------------------------------------------


class TestSelectInArrowTabular:
    def test_select_by_name(self) -> None:
        t = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        arr = _id_field().select_in_arrow_tabular(t)
        assert arr.to_pylist() == [1, 2, 3]

    def test_select_by_alias_when_name_missing(self) -> None:
        t = pa.table({"user_id": [1, 2, 3], "name": ["a", "b", "c"]})
        f = _id_field().with_alias("user_id")
        arr = f.select_in_arrow_tabular(t)
        assert arr.to_pylist() == [1, 2, 3]

    def test_missing_raises_by_default(self) -> None:
        t = pa.table({"name": ["a", "b", "c"]})
        with pytest.raises(KeyError, match="Field 'id' not found"):
            _id_field().select_in_arrow_tabular(t)

    def test_missing_returns_default_when_provided(self) -> None:
        t = pa.table({"name": ["a", "b", "c"]})
        out = _id_field().select_in_arrow_tabular(t, default=None)
        assert out is None

    def test_default_can_be_a_typed_array(self) -> None:
        t = pa.table({"name": ["a", "b", "c"]})
        sentinel = pa.nulls(t.num_rows, type=pa.int64())
        out = _id_field().select_in_arrow_tabular(t, default=sentinel)
        assert out is sentinel

    def test_record_batch_returns_array(self) -> None:
        rb = pa.record_batch({"id": [1, 2], "name": ["a", "b"]})
        arr = _id_field().select_in_arrow_tabular(rb)
        assert isinstance(arr, pa.Array)
        assert arr.to_pylist() == [1, 2]

    def test_cast_string_to_int(self) -> None:
        t = pa.table({"id": ["1", "2", "3"], "name": ["a", "b", "c"]})
        arr = _id_field().select_in_arrow_tabular(t)
        assert arr.to_pylist() == [1, 2, 3]
        assert pa.types.is_int64(arr.type)


# ---------------------------------------------------------------------------
# select_in_arrow / select_in dispatch
# ---------------------------------------------------------------------------


class TestSelectInArrowDispatch:
    def test_array_passes_through_cast(self) -> None:
        arr = pa.array(["1", "2", "3"])
        out = _id_field().select_in_arrow(arr)
        assert out.to_pylist() == [1, 2, 3]

    def test_table_dispatches_to_tabular(self) -> None:
        t = pa.table({"id": [1, 2]})
        out = _id_field().select_in_arrow(t)
        assert out.to_pylist() == [1, 2]

    def test_unsupported_raises(self) -> None:
        with pytest.raises(TypeError, match="select_in_arrow"):
            _id_field().select_in_arrow({"id": 1})


class TestSelectIn:
    def test_arrow_table(self) -> None:
        t = pa.table({"id": [1, 2]})
        assert _id_field().select_in(t).to_pylist() == [1, 2]

    def test_field_returns_matching_child(self) -> None:
        schema = Field.empty()
        # Build a Schema/Field-like with two children.
        schema = Field.from_(pa.schema([("id", pa.int64()), ("name", pa.string())]))
        matched = _id_field().select_in(schema)
        assert matched is not None
        assert matched.name == "id"


# ---------------------------------------------------------------------------
# select_in_polars_*
# ---------------------------------------------------------------------------


class TestSelectInPolars:
    @pytest.fixture(autouse=True)
    def _import_polars(self) -> None:
        pytest.importorskip("polars")

    def test_select_in_polars_frame(self) -> None:
        import polars as pl
        df = pl.DataFrame({"identifier": [1, 2, 3], "name": ["a", "b", "c"]})
        f = _id_field().with_alias("identifier")
        s = f.select_in_polars_frame(df)
        assert s.to_list() == [1, 2, 3]
        assert s.name == "id"

    def test_select_in_polars_lazy_frame_returns_expr(self) -> None:
        import polars as pl
        lf = pl.DataFrame({"id": [1, 2, 3]}).lazy()
        expr = _id_field().select_in_polars_lazy_frame(lf)
        assert isinstance(expr, pl.Expr)
        assert lf.select(expr).collect().get_column("id").to_list() == [1, 2, 3]

    def test_polars_frame_default(self) -> None:
        import polars as pl
        df = pl.DataFrame({"name": ["a", "b", "c"]})
        sentinel = pl.Series("id", [None, None, None], dtype=pl.Int64)
        out = _id_field().select_in_polars_frame(df, default=sentinel)
        assert out is sentinel


# ---------------------------------------------------------------------------
# select_in_field
# ---------------------------------------------------------------------------


class TestSelectInField:
    def test_finds_by_name(self) -> None:
        schema = Field.from_(pa.schema([("id", pa.int64()), ("name", pa.string())]))
        out = _id_field().select_in_field(schema)
        assert out is not None and out.name == "id"

    def test_finds_by_alias(self) -> None:
        schema = Field.from_(
            pa.schema([("user_id", pa.int64()), ("name", pa.string())]),
        )
        f = _id_field().with_alias("user_id")
        out = f.select_in_field(schema)
        assert out is not None and out.name == "user_id"

    def test_missing_raises(self) -> None:
        schema = Field.from_(pa.schema([("name", pa.string())]))
        with pytest.raises(KeyError):
            _id_field().select_in_field(schema)

    def test_missing_returns_default_when_provided(self) -> None:
        schema = Field.from_(pa.schema([("name", pa.string())]))
        f = _id_field()
        out = f.select_in_field(schema, default=f)
        assert out is f

    def test_finds_when_child_alias_matches_self_name(self) -> None:
        # Source's child renames "user_id" -> "id" via its own alias.
        # The lookup field carries name="id" and should match.
        renamed = Field.from_(pa.field("user_id", pa.int64())).with_alias("id")
        schema = Schema([renamed, Field.from_(pa.field("name", pa.string()))])
        out = _id_field().select_in_field(schema)
        assert out is not None and out.name == "user_id"

    def test_finds_when_child_alias_matches_self_alias(self) -> None:
        # Both sides carry alias "id"; neither side's bare name matches.
        renamed = Field.from_(pa.field("user_id", pa.int64())).with_alias("id")
        schema = Schema([renamed, Field.from_(pa.field("name", pa.string()))])
        f = Field.from_(pa.field("identifier", pa.int64())).with_alias("id")
        out = f.select_in_field(schema)
        assert out is not None and out.name == "user_id"

    def test_name_on_name_wins_over_name_on_alias(self) -> None:
        # First child aliases "id" away from its real name; second child
        # has the literal name "id". A name-on-name match must beat the
        # alias match no matter the iteration order.
        aliased = Field.from_(pa.field("user_id", pa.int64())).with_alias("id")
        direct = Field.from_(pa.field("id", pa.int64()))
        schema = Schema([aliased, direct])
        out = _id_field().select_in_field(schema)
        assert out is not None and out.name == "id"

    def test_self_name_wins_over_self_alias(self) -> None:
        # ``self`` has both name and alias; matching by name takes
        # priority over matching by alias.
        schema = Schema([
            Field.from_(pa.field("id", pa.int64())),
            Field.from_(pa.field("user_id", pa.int64())),
        ])
        f = _id_field().with_alias("user_id")
        out = f.select_in_field(schema)
        assert out is not None and out.name == "id"

    def test_position_fallback_when_names_dont_match(self) -> None:
        schema = Schema([
            Field.from_(pa.field("alpha", pa.int64())),
            Field.from_(pa.field("beta", pa.int64())),
        ])
        f = Field.from_(pa.field("gamma", pa.int64())).with_position(1)
        out = f.select_in_field(schema)
        assert out is not None and out.name == "beta"

    def test_position_returns_child_at_index_with_duplicate_names(self) -> None:
        # Two children share a name. Resolving by position must return
        # the child at the requested index, not the first match by
        # name (which is what a name-scan after position lookup would
        # do).
        a = Field.from_(pa.field("dup", pa.int64()))
        b = Field.from_(pa.field("dup", pa.int64())).with_alias("second")
        schema = Schema([a, b])
        f = Field.from_(pa.field("other", pa.int64())).with_position(1)
        out = f.select_in_field(schema)
        assert out is b
        assert out.alias == "second"

    def test_position_out_of_range_falls_through_to_default(self) -> None:
        schema = Schema([Field.from_(pa.field("only", pa.int64()))])
        f = Field.from_(pa.field("missing", pa.int64())).with_position(5)
        assert f.select_in_field(schema, default=None) is None
        with pytest.raises(KeyError):
            f.select_in_field(schema)

    def test_error_message_lists_name_and_alias_pairs(self) -> None:
        renamed = Field.from_(pa.field("user_id", pa.int64())).with_alias("uid")
        schema = Schema([renamed])
        with pytest.raises(KeyError) as info:
            _id_field().select_in_field(schema)
        msg = str(info.value)
        assert "user_id/uid" in msg
        assert "'id'" in msg


# ---------------------------------------------------------------------------
# CastOptions.match_by / match_by_keys
# ---------------------------------------------------------------------------


class TestCastOptionsMatchBy:
    def test_string_keys_coerced_to_fields(self) -> None:
        opts = CastOptions(match_by=["id", "tenant"])
        assert opts.match_by_keys == ["id", "tenant"]
        assert all(isinstance(f, Field) for f in opts.match_by)

    def test_field_keys_passthrough(self) -> None:
        f = _id_field()
        opts = CastOptions(match_by=[f])
        assert opts.match_by_keys == ["id"]
        assert opts.match_by[0] is f

    def test_mixed_keys_normalized(self) -> None:
        f = _id_field().with_alias("user_id")
        opts = CastOptions(match_by=[f, "tenant"])
        assert opts.match_by_keys == ["id", "tenant"]

    def test_match_by_keys_none_when_unset(self) -> None:
        assert CastOptions().match_by_keys is None

    def test_empty_match_by_collapses_to_none(self) -> None:
        assert CastOptions(match_by=[]).match_by is None
        assert CastOptions(match_by=[]).match_by_keys is None
