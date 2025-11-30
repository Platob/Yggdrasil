import pytest
import pyarrow as pa

from yggdrasil.types.cast.arrow import ArrowCastOptions
from yggdrasil.types.cast.pandas import cast_pandas_dataframe, cast_pandas_series
from yggdrasil.types.cast.polars import cast_polars_dataframe, cast_polars_series


@pytest.fixture
def sample_arrow_schema():
    return pa.schema(
        [
            pa.field("a", pa.int64()),
            pa.field("b", pa.string(), nullable=True),
        ]
    )


def test_cast_polars_series_to_int():
    polars = pytest.importorskip("polars")

    series = polars.Series("numbers", ["1", "2", "3"])
    result = cast_polars_series(series, pa.int64())

    assert result.dtype == polars.Int64
    assert result.to_list() == [1, 2, 3]


def test_cast_polars_series_avoids_arrow_roundtrip(monkeypatch):
    polars = pytest.importorskip("polars")

    def fail(*args, **kwargs):  # pragma: no cover - ensures the path is unused
        raise AssertionError("to_arrow should not be called during Polars casting")

    monkeypatch.setattr(polars.Series, "to_arrow", fail)

    series = polars.Series("numbers", ["1", "2", "3"])
    result = cast_polars_series(series, pa.int64())

    assert result.to_list() == [1, 2, 3]


def test_cast_polars_series_fills_non_nullable_defaults():
    polars = pytest.importorskip("polars")

    series = polars.Series("numbers", [1, None])
    target = pa.field("numbers", pa.int64(), nullable=False)

    result = cast_polars_series(series, target)

    assert result.to_list() == [1, 0]
    assert result.dtype == polars.Int64


def test_cast_polars_dataframe_with_missing_column(sample_arrow_schema):
    polars = pytest.importorskip("polars")

    dataframe = polars.DataFrame({"A": ["10", "20"]})
    options = ArrowCastOptions.safe_init(target_field=sample_arrow_schema)

    result = cast_polars_dataframe(dataframe, options)
    table = result.to_arrow()

    a_field = table.schema.field("a")
    b_field = table.schema.field("b")

    assert a_field.type.equals(pa.int64())
    assert pa.types.is_string(b_field.type) or pa.types.is_large_string(b_field.type)
    assert table.column("a").to_pylist() == [10, 20]
    assert table.column("b").to_pylist() == [None, None]


def test_cast_polars_dataframe_uses_target_schema_option(sample_arrow_schema):
    polars = pytest.importorskip("polars")

    dataframe = polars.DataFrame({"a": ["1", "2"]})
    options = ArrowCastOptions.safe_init(target_field=sample_arrow_schema)

    result = cast_polars_dataframe(dataframe, options)
    table = result.to_arrow()

    b_field = table.schema.field("b")
    assert table.schema.field("a").type.equals(pa.int64())
    assert pa.types.is_string(b_field.type) or pa.types.is_large_string(b_field.type)
    assert table.column("a").to_pylist() == [1, 2]
    assert table.column("b").to_pylist() == [None, None]


def test_cast_polars_dataframe_avoids_arrow_roundtrip(monkeypatch, sample_arrow_schema):
    polars = pytest.importorskip("polars")

    def fail(*args, **kwargs):  # pragma: no cover - ensures the path is unused
        raise AssertionError("to_arrow should not be called during Polars casting")

    monkeypatch.setattr(polars.DataFrame, "to_arrow", fail)

    dataframe = polars.DataFrame({"a": [1]})
    result = cast_polars_dataframe(
        dataframe, ArrowCastOptions.safe_init(target_field=sample_arrow_schema)
    )

    assert result.schema == polars.Schema([("a", polars.Int64), ("b", polars.Utf8)])


def test_cast_pandas_series_to_float():
    pandas = pytest.importorskip("pandas", reason="pandas is optional")

    series = pandas.Series(["1.1", "2.2"], name="vals")
    result = cast_pandas_series(series, pa.float64())

    assert list(result) == [1.1, 2.2]
    assert result.name == "vals"


def test_cast_pandas_dataframe_to_schema(sample_arrow_schema):
    pandas = pytest.importorskip("pandas", reason="pandas is optional")

    dataframe = pandas.DataFrame({"a": ["1", "2"], "B": ["x", "y"]})
    options = ArrowCastOptions.safe_init(target_field=sample_arrow_schema)

    result = cast_pandas_dataframe(dataframe, options)
    casted_table = pa.Table.from_pandas(result, schema=sample_arrow_schema, preserve_index=False)

    assert casted_table.schema.equals(sample_arrow_schema)
    assert casted_table.column("a").to_pylist() == [1, 2]
    assert casted_table.column("b").to_pylist() == ["x", "y"]


def test_cast_pandas_dataframe_uses_target_schema_option(sample_arrow_schema):
    pandas = pytest.importorskip("pandas", reason="pandas is optional")

    dataframe = pandas.DataFrame({"a": ["1", "2"]})
    options = ArrowCastOptions.safe_init(target_field=sample_arrow_schema)

    result = cast_pandas_dataframe(dataframe, options)
    casted_table = pa.Table.from_pandas(result, schema=sample_arrow_schema, preserve_index=False)

    assert casted_table.schema.equals(sample_arrow_schema)
    assert casted_table.column("a").to_pylist() == [1, 2]
    assert casted_table.column("b").to_pylist() == [None, None]


def test_cast_pandas_series_nested_list_non_nullable():
    pandas = pytest.importorskip("pandas", reason="pandas is optional")

    series = pandas.Series([["1"], None], name="nested")
    target = pa.field("nested", pa.list_(pa.int64()), nullable=False)

    result = cast_pandas_series(series, target)

    assert list(result) == [[1], []]
    assert result.name == "nested"


def test_cast_polars_series_non_nullable_nested_list():
    polars = pytest.importorskip("polars")

    series = polars.Series("nested", [[1, 2], None])
    target = pa.field("nested", pa.list_(pa.int64()), nullable=False)

    result = cast_polars_series(series, target)

    assert result.dtype == polars.List(polars.Int64)
    assert result.to_list() == [[1, 2], []]


def test_cast_polars_dataframe_struct_field():
    polars = pytest.importorskip("polars")

    dataframe = polars.DataFrame({"payload": [{"x": 1, "y": "a"}, None]})
    schema = pa.schema(
        [
            pa.field(
                "payload",
                pa.struct(
                    [
                        pa.field("x", pa.int64(), nullable=False),
                        pa.field("y", pa.string(), nullable=True),
                    ]
                ),
                nullable=False,
            )
        ]
    )

    result = cast_polars_dataframe(dataframe, ArrowCastOptions.safe_init(target_field=schema))

    assert result.schema == polars.Schema([("payload", polars.Struct({"x": polars.Int64, "y": polars.Utf8}))])
    assert result.select(polars.col("payload").struct.field("x")).to_series().to_list() == [1, 0]
