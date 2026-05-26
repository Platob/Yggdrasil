"""Pandas read/write integration tests on LocalPath."""
from __future__ import annotations

import pathlib

import pandas as pd
import pyarrow as pa

from yggdrasil.enums import Mode
from yggdrasil.path.local_path import LocalPath


# ---------------------------------------------------------------------------
# TestPandasWrite
# ---------------------------------------------------------------------------


class TestPandasWrite:

    def test_write_pandas_read_back_as_arrow(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "df.parquet"))
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        p.write_table(df)

        result = p.read_arrow_table()
        assert result.num_rows == 3
        assert result.column_names == ["a", "b"]
        assert result.column("a").to_pylist() == [1, 2, 3]
        assert result.column("b").to_pylist() == ["x", "y", "z"]

    def test_write_to_parquet_format(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "out.parquet"))
        df = pd.DataFrame({"x": [10, 20], "y": [0.5, 1.5]})
        p.write_table(df)

        assert p.exists()
        assert p.size > 0
        table = p.read_arrow_table()
        assert table.num_rows == 2
        assert table.column("x").to_pylist() == [10, 20]

    def test_write_to_ipc_format(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "out.ipc"))
        df = pd.DataFrame({"k": [100, 200, 300]})
        p.write_table(df)

        assert p.exists()
        table = p.read_arrow_table()
        assert table.num_rows == 3
        assert table.column("k").to_pylist() == [100, 200, 300]

    def test_write_with_mode_overwrite(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "ow.parquet"))
        df_first = pd.DataFrame({"v": [1, 2, 3]})
        p.write_table(df_first)
        assert p.read_arrow_table().num_rows == 3

        df_second = pd.DataFrame({"v": [10, 20]})
        p.write_table(df_second, mode=Mode.OVERWRITE)
        result = p.read_arrow_table()
        assert result.num_rows == 2
        assert result.column("v").to_pylist() == [10, 20]


# ---------------------------------------------------------------------------
# TestPandasRead
# ---------------------------------------------------------------------------


class TestPandasRead:

    def test_write_arrow_read_as_pandas(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "arrow_src.parquet"))
        table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        p.write_table(table)

        result_df = p.read_arrow_table().to_pandas()
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 3
        assert list(result_df["a"]) == [1, 2, 3]
        assert list(result_df["b"]) == ["x", "y", "z"]

    def test_column_types_preserved(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "types.parquet"))
        table = pa.table({
            "ints": pa.array([1, 2, 3], type=pa.int64()),
            "floats": pa.array([1.1, 2.2, 3.3], type=pa.float64()),
            "strings": pa.array(["a", "b", "c"], type=pa.string()),
            "bools": pa.array([True, False, True], type=pa.bool_()),
        })
        p.write_table(table)

        result_df = p.read_arrow_table().to_pandas()
        assert result_df["ints"].dtype.name.startswith("int")
        assert result_df["floats"].dtype.name.startswith("float")
        assert result_df["bools"].dtype in (bool, "bool", pd.BooleanDtype())

    def test_large_dataframe(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "large.parquet"))
        table = pa.table({
            "id": pa.array(range(10_000), type=pa.int64()),
            "value": pa.array([float(i) * 0.1 for i in range(10_000)], type=pa.float64()),
        })
        p.write_table(table)

        result_df = p.read_arrow_table().to_pandas()
        assert len(result_df) == 10_000
        assert result_df["id"].iloc[0] == 0
        assert result_df["id"].iloc[-1] == 9_999


# ---------------------------------------------------------------------------
# TestPandasRoundTrip
# ---------------------------------------------------------------------------


class TestPandasRoundTrip:

    def test_pandas_parquet_roundtrip(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "rt.parquet"))
        df_in = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["alpha", "beta", "gamma"],
            "score": [9.5, 8.3, 7.1],
        })
        p.write_table(df_in)

        df_out = p.read_arrow_table().to_pandas()
        assert list(df_out["id"]) == [1, 2, 3]
        assert list(df_out["name"]) == ["alpha", "beta", "gamma"]
        assert list(df_out["score"]) == [9.5, 8.3, 7.1]

    def test_pandas_ipc_roundtrip(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "rt.ipc"))
        df_in = pd.DataFrame({
            "x": [10, 20, 30],
            "y": ["foo", "bar", "baz"],
        })
        p.write_table(df_in)

        df_out = p.read_arrow_table().to_pandas()
        assert list(df_out["x"]) == [10, 20, 30]
        assert list(df_out["y"]) == ["foo", "bar", "baz"]

    def test_pandas_datetime_roundtrip(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "dt.parquet"))
        timestamps = pd.to_datetime(["2024-01-01", "2024-06-15", "2024-12-31"])
        df_in = pd.DataFrame({
            "event": ["a", "b", "c"],
            "ts": timestamps,
        })
        p.write_table(df_in)

        df_out = p.read_arrow_table().to_pandas()
        assert list(df_out["event"]) == ["a", "b", "c"]
        # Timestamps should survive the round-trip within microsecond precision
        for original, restored in zip(timestamps, df_out["ts"]):
            assert abs((original - restored).total_seconds()) < 1e-3
