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


def test_cast_polars_series_uses_newest_compat(monkeypatch):
    polars = pytest.importorskip("polars")

    captured = []
    original = polars.Series.to_arrow

    def spy(self, *args, **kwargs):
        captured.append(kwargs.get("compat_level"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(polars.Series, "to_arrow", spy)

    series = polars.Series("numbers", ["1", "2", "3"])
    cast_polars_series(series, pa.int64())

    assert captured[-1] == polars.CompatLevel.newest()


def test_cast_polars_dataframe_with_missing_column(sample_arrow_schema):
    polars = pytest.importorskip("polars")

    dataframe = polars.DataFrame({"A": ["10", "20"]})
    options = ArrowCastOptions()

    result = cast_polars_dataframe(dataframe, sample_arrow_schema, options)
    table = result.to_arrow()

    a_field = table.schema.field("a")
    b_field = table.schema.field("b")

    assert a_field.type.equals(pa.int64())
    assert pa.types.is_string(b_field.type) or pa.types.is_large_string(b_field.type)
    assert table.column("a").to_pylist() == [10, 20]
    assert table.column("b").to_pylist() == [None, None]


def test_cast_polars_dataframe_uses_newest_compat(monkeypatch, sample_arrow_schema):
    polars = pytest.importorskip("polars")

    captured = []
    original = polars.DataFrame.to_arrow

    def spy(self, *args, **kwargs):
        captured.append(kwargs.get("compat_level"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(polars.DataFrame, "to_arrow", spy)

    dataframe = polars.DataFrame({"a": [1]})
    cast_polars_dataframe(dataframe, sample_arrow_schema)

    assert captured[-1] == polars.CompatLevel.newest()


def test_cast_pandas_series_to_float():
    pandas = pytest.importorskip("pandas", reason="pandas is optional")

    series = pandas.Series(["1.1", "2.2"], name="vals")
    result = cast_pandas_series(series, pa.float64())

    assert list(result) == [1.1, 2.2]
    assert result.name == "vals"


def test_cast_pandas_dataframe_to_schema(sample_arrow_schema):
    pandas = pytest.importorskip("pandas", reason="pandas is optional")

    dataframe = pandas.DataFrame({"a": ["1", "2"], "B": ["x", "y"]})
    options = ArrowCastOptions()

    result = cast_pandas_dataframe(dataframe, sample_arrow_schema, options)
    casted_table = pa.Table.from_pandas(result, schema=sample_arrow_schema, preserve_index=False)

    assert casted_table.schema.equals(sample_arrow_schema)
    assert casted_table.column("a").to_pylist() == [1, 2]
    assert casted_table.column("b").to_pylist() == ["x", "y"]
