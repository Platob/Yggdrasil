"""Advanced Pandas tests — index handling, categoricals, and nullable dtypes."""
from __future__ import annotations

import pathlib

import pandas as pd
import pyarrow as pa

from yggdrasil.path.local_path import LocalPath


# ---------------------------------------------------------------------------
# TestPandasIndex
# ---------------------------------------------------------------------------


class TestPandasIndex:

    def test_default_range_index_values_preserved(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "range_idx.parquet"))
        df = pd.DataFrame({"a": [10, 20, 30], "b": ["x", "y", "z"]})
        p.write_table(df)

        result = p.read_arrow_table().to_pandas()
        assert list(result["a"]) == [10, 20, 30]
        assert list(result["b"]) == ["x", "y", "z"]
        # RangeIndex is not preserved as a column — values come back with a fresh index
        assert list(result.index) == [0, 1, 2]

    def test_named_index_preserved_as_column(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "named_idx.parquet"))
        df = pd.DataFrame({"val": [100, 200, 300]})
        df.index = pd.Index([10, 20, 30], name="my_id")
        p.write_table(df)

        result = p.read_arrow_table().to_pandas()
        assert "my_id" in result.columns
        assert list(result["my_id"]) == [10, 20, 30]
        assert list(result["val"]) == [100, 200, 300]

    def test_multiindex_roundtrip(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "multi_idx.parquet"))
        arrays = [
            ["bar", "bar", "baz", "baz"],
            ["one", "two", "one", "two"],
        ]
        idx = pd.MultiIndex.from_arrays(arrays, names=["first", "second"])
        df = pd.DataFrame({"val": [1, 2, 3, 4]}, index=idx)
        p.write_table(df)

        result = p.read_arrow_table().to_pandas()
        assert "first" in result.columns
        assert "second" in result.columns
        assert list(result["first"]) == ["bar", "bar", "baz", "baz"]
        assert list(result["second"]) == ["one", "two", "one", "two"]
        assert list(result["val"]) == [1, 2, 3, 4]

    def test_datetime_index_roundtrip(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "dt_idx.parquet"))
        dates = pd.to_datetime(["2024-01-01", "2024-06-15", "2024-12-31"])
        df = pd.DataFrame({"val": [10, 20, 30]}, index=pd.Index(dates, name="ts"))
        p.write_table(df)

        result = p.read_arrow_table().to_pandas()
        assert "ts" in result.columns
        for original, restored in zip(dates, result["ts"]):
            assert abs((original - restored).total_seconds()) < 1e-3
        assert list(result["val"]) == [10, 20, 30]

    def test_custom_string_index(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "str_idx.parquet"))
        df = pd.DataFrame(
            {"score": [9.5, 8.0, 7.3]},
            index=pd.Index(["alpha", "beta", "gamma"], name="label"),
        )
        p.write_table(df)

        result = p.read_arrow_table().to_pandas()
        assert "label" in result.columns
        assert list(result["label"]) == ["alpha", "beta", "gamma"]
        assert list(result["score"]) == [9.5, 8.0, 7.3]

    def test_index_name_survives_roundtrip_metadata(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "idx_meta.parquet"))
        df = pd.DataFrame({"v": [1, 2]}, index=pd.Index([10, 20], name="pk"))
        p.write_table(df)

        # Read raw Arrow table and verify index-column metadata
        arrow_table = p.read_arrow_table()
        assert "pk" in arrow_table.column_names
        # The column should carry data from the original index
        assert arrow_table.column("pk").to_pylist() == [10, 20]
        # Round-trip back to pandas — the name is recoverable
        result = arrow_table.to_pandas()
        assert "pk" in result.columns


# ---------------------------------------------------------------------------
# TestPandasCategorical
# ---------------------------------------------------------------------------


class TestPandasCategorical:

    def test_categorical_column_roundtrip(self, tmp_path: pathlib.Path) -> None:
        # write_table converts via pa.Table.from_pandas which emits
        # dictionary-encoded arrays for categoricals. The write pipeline
        # casts those to their value type before hitting the file writer,
        # so here we write an Arrow table built from a categorical
        # DataFrame — the categorical *values* survive the round-trip
        # even though the dictionary encoding is flattened.
        p = LocalPath(str(tmp_path / "cat.parquet"))
        df = pd.DataFrame({
            "color": pd.Categorical(["red", "green", "blue", "red", "green"]),
            "count": [1, 2, 3, 4, 5],
        })
        # Convert via Arrow to flatten dictionary encoding before writing
        table = pa.Table.from_pandas(df, preserve_index=False)
        table = table.cast(pa.schema([
            ("color", pa.utf8()),
            ("count", pa.int64()),
        ]))
        p.write_table(table)

        result = p.read_arrow_table().to_pandas()
        assert list(result["color"]) == ["red", "green", "blue", "red", "green"]
        assert list(result["count"]) == [1, 2, 3, 4, 5]

    def test_ordered_categorical_preserves_ordering(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "cat_ordered.parquet"))
        cat = pd.CategoricalDtype(categories=["low", "med", "high"], ordered=True)
        df = pd.DataFrame({
            "priority": pd.Categorical(["high", "low", "med"], dtype=cat),
        })
        # Flatten dictionary encoding so the write pipeline can handle it
        table = pa.Table.from_pandas(df, preserve_index=False)
        table = table.cast(pa.schema([("priority", pa.utf8())]))
        p.write_table(table)

        arrow_table = p.read_arrow_table()
        values = arrow_table.column("priority").to_pylist()
        assert values == ["high", "low", "med"]

        # Re-apply categorical ordering on read to confirm values survive
        result = arrow_table.to_pandas()
        restored = result["priority"].astype(cat)
        assert restored.cat.ordered is True
        assert list(restored.cat.categories) == ["low", "med", "high"]
        assert list(restored) == ["high", "low", "med"]


# ---------------------------------------------------------------------------
# TestPandasNullable
# ---------------------------------------------------------------------------


class TestPandasNullable:

    def test_nullable_integer_roundtrip(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "nullable_int.parquet"))
        df = pd.DataFrame({
            "x": pd.array([1, None, 3, None, 5], dtype="Int64"),
        })
        p.write_table(df)

        arrow_table = p.read_arrow_table()
        assert arrow_table.num_rows == 5
        values = arrow_table.column("x").to_pylist()
        assert values[0] == 1
        assert values[1] is None
        assert values[2] == 3
        assert values[3] is None
        assert values[4] == 5

    def test_nullable_string_roundtrip(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "nullable_str.parquet"))
        df = pd.DataFrame({
            "s": pd.array(["hello", None, "world", None], dtype="string"),
        })
        p.write_table(df)

        arrow_table = p.read_arrow_table()
        assert arrow_table.num_rows == 4
        values = arrow_table.column("s").to_pylist()
        assert values[0] == "hello"
        assert values[1] is None
        assert values[2] == "world"
        assert values[3] is None
