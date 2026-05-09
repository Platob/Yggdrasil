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

    def test_missing_returns_none_with_raise_error_false(self) -> None:
        t = pa.table({"name": ["a", "b", "c"]})
        out = _id_field().select_in_arrow_tabular(t, raise_error=False)
        assert out is None

    def test_return_default_synthesizes_null_array(self) -> None:
        t = pa.table({"name": ["a", "b", "c"]})
        out = _id_field().select_in_arrow_tabular(t, return_default=True)
        assert out.to_pylist() == [None, None, None]
        assert pa.types.is_int64(out.type)

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
        s = _id_field().select_in_polars_frame(df, return_default=True)
        assert s.to_list() == [None, None, None]


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

    def test_missing_returns_default_self(self) -> None:
        schema = Field.from_(pa.schema([("name", pa.string())]))
        f = _id_field()
        out = f.select_in_field(schema, return_default=True)
        assert out is f


# ---------------------------------------------------------------------------
# CastOptions.match_by / match_by_keys / match_by_fields
# ---------------------------------------------------------------------------


class TestCastOptionsMatchBy:
    def test_match_by_names_resolves_to_keys(self) -> None:
        opts = CastOptions(match_by_names=["id", "tenant"])
        assert opts.match_by_keys == ["id", "tenant"]

    def test_match_by_fields_take_precedence(self) -> None:
        f = _id_field()
        opts = CastOptions(match_by=[f], match_by_names=["other"])
        assert opts.match_by_keys == ["id"]

    def test_match_by_keys_none_when_neither_set(self) -> None:
        assert CastOptions().match_by_keys is None

    def test_match_by_fields_resolves_against_target_schema(self) -> None:
        schema = Schema.from_(
            pa.schema([("id", pa.int64()), ("name", pa.string())]),
        )
        opts = CastOptions(target_field=schema, match_by_names=["id"])
        fields = opts.match_by_fields
        assert fields is not None
        assert [f.name for f in fields] == ["id"]
