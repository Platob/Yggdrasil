import pyarrow as pa
import pytest

pandas = pytest.importorskip("pandas")

from yggdrasil.types.cast.arrow import ArrowCastOptions
from yggdrasil.types.cast.pandas import (
    cast_pandas_series,
    cast_pandas_dataframe,
)


def make_options(**overrides):
    """
    Helper to construct ArrowCastOptions with defaults, then override attributes.
    Assumes ArrowCastOptions.__safe_init__() is default-constructible and its fields are mutable.
    """
    opts = ArrowCastOptions.__safe_init__()
    for k, v in overrides.items():
        setattr(opts, k, v)
    return opts


# ---------------- cast_pandas_series ----------------


def test_cast_pandas_series_roundtrip_without_target_schema():
    s = pandas.Series([1, 2, 3], name="a")
    opts = make_options()  # no target schema/field → should be effectively identity

    result = cast_pandas_series(s, opts)

    # index + name preserved, values unchanged
    assert list(result.index) == list(s.index)
    assert result.name == "a"
    assert result.tolist() == [1, 2, 3]


def test_cast_pandas_series_respects_index_and_name_after_cast():
    idx = [10, 20, 30]
    s = pandas.Series([1.1, 2.2, 3.3], name="col", index=idx)

    # Even if ArrowCastOptions causes some casting internally, we only care
    # here that index and name survive the roundtrip.
    opts = make_options()
    result = cast_pandas_series(s, opts)

    assert list(result.index) == idx
    assert result.name == "col"
    assert result.tolist() == s.tolist()


# ---------------- cast_pandas_dataframe ----------------


def test_cast_pandas_dataframe_roundtrip_with_schema_and_preserve_index():
    df = pandas.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [0.1, 0.2, 0.3],
        },
        index=[10, 11, 12],
    )

    schema = pa.schema(
        [
            pa.field("a", pa.int64()),
            pa.field("b", pa.float64()),
        ]
    )

    opts = make_options(target_schema=schema)

    result = cast_pandas_dataframe(df, opts)

    # Same index back after Arrow roundtrip
    assert list(result.index) == [10, 11, 12]

    # Columns present and in same order as schema
    assert list(result.columns) == ["a", "b"]

    # Values preserved (cast_arrow_table is responsible for actual type casting)
    assert result["a"].tolist() == [1, 2, 3]
    assert result["b"].tolist() == [0.1, 0.2, 0.3]


def test_cast_pandas_dataframe_disallow_extra_columns_drops_them():
    df = pandas.DataFrame(
        {
            "a": [1, 2],
            "b": [0.1, 0.2],
            "extra": ["x", "y"],
        }
    )

    schema = pa.schema(
        [
            pa.field("a", pa.int64()),
            pa.field("b", pa.float64()),
        ]
    )

    # We assume cast_arrow_table respects target_schema and drops "extra".
    opts = make_options(target_schema=schema, allow_add_columns=False)

    result = cast_pandas_dataframe(df, opts)

    assert list(result.columns) == ["a", "b"]
    assert "extra" not in result.columns


def test_cast_pandas_dataframe_allow_extra_columns_kept_and_appended():
    df = pandas.DataFrame(
        {
            "a": [1, 2],
            "b": [0.1, 0.2],
            "extra": ["x", "y"],
        }
    )

    schema = pa.schema(
        [
            pa.field("a", pa.int64()),
            pa.field("b", pa.float64()),
        ]
    )

    # cast_arrow_table should produce only schema columns;
    # wrapper then adds "extra" back when allow_add_columns=True.
    opts = make_options(target_schema=schema, allow_add_columns=True)

    result = cast_pandas_dataframe(df, opts)

    # Schema columns present
    assert "a" in result.columns
    assert "b" in result.columns
    # Extra column preserved by wrapper
    assert "extra" in result.columns
    assert result["extra"].tolist() == ["x", "y"]
    assert result.shape[0] == 2
