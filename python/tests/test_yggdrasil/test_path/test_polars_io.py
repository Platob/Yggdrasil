"""Polars read/write integration tests on LocalPath via as_media."""
from __future__ import annotations

import pathlib

import polars as pl
import pyarrow as pa

from yggdrasil.enums import Mode
from yggdrasil.path.local_path import LocalPath


# ---------------------------------------------------------------------------
# TestPolarsWrite
# ---------------------------------------------------------------------------


class TestPolarsWrite:

    def test_write_polars_dataframe_read_back_as_arrow(self, tmp_path: pathlib.Path) -> None:
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        p = LocalPath(str(tmp_path / "out.ipc"))
        leaf = p.as_media()
        leaf.write_table(df)

        result = leaf.read_arrow_table()
        assert result.num_rows == 3
        assert result.column("a").to_pylist() == [1, 2, 3]
        assert result.column("b").to_pylist() == ["x", "y", "z"]

    def test_write_polars_lazyframe(self, tmp_path: pathlib.Path) -> None:
        lf = pl.LazyFrame({"c": [10, 20], "d": [1.5, 2.5]})
        p = LocalPath(str(tmp_path / "lazy.ipc"))
        leaf = p.as_media()
        leaf.write_table(lf)

        result = leaf.read_arrow_table()
        assert result.num_rows == 2
        assert result.column("c").to_pylist() == [10, 20]
        assert result.column("d").to_pylist() == [1.5, 2.5]

    def test_write_to_parquet_format(self, tmp_path: pathlib.Path) -> None:
        df = pl.DataFrame({"id": [1, 2, 3], "val": [100, 200, 300]})
        p = LocalPath(str(tmp_path / "data.parquet"))
        leaf = p.as_media()
        leaf.write_table(df)

        result = leaf.read_arrow_table()
        assert result.num_rows == 3
        assert result.column("id").to_pylist() == [1, 2, 3]
        assert result.column("val").to_pylist() == [100, 200, 300]

    def test_write_to_ipc_format(self, tmp_path: pathlib.Path) -> None:
        df = pl.DataFrame({"x": [7, 8], "y": ["foo", "bar"]})
        p = LocalPath(str(tmp_path / "data.ipc"))
        leaf = p.as_media()
        leaf.write_table(df)

        result = leaf.read_arrow_table()
        assert result.num_rows == 2
        assert result.column("x").to_pylist() == [7, 8]
        assert result.column("y").to_pylist() == ["foo", "bar"]

    def test_write_with_mode_overwrite(self, tmp_path: pathlib.Path) -> None:
        p = LocalPath(str(tmp_path / "overwrite.parquet"))
        leaf = p.as_media()

        df1 = pl.DataFrame({"v": [1, 2, 3]})
        leaf.write_table(df1)
        assert leaf.read_arrow_table().num_rows == 3

        df2 = pl.DataFrame({"v": [10, 20]})
        leaf.write_table(df2, mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.num_rows == 2
        assert result.column("v").to_pylist() == [10, 20]


# ---------------------------------------------------------------------------
# TestPolarsRead
# ---------------------------------------------------------------------------


class TestPolarsRead:

    def test_write_arrow_read_as_polars(self, tmp_path: pathlib.Path) -> None:
        table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        p = LocalPath(str(tmp_path / "arrow_src.ipc"))
        leaf = p.as_media()
        leaf.write_table(table)

        result = pl.from_arrow(leaf.read_arrow_table())
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (3, 2)
        assert result["a"].to_list() == [1, 2, 3]
        assert result["b"].to_list() == ["x", "y", "z"]

    def test_read_parquet_as_polars(self, tmp_path: pathlib.Path) -> None:
        table = pa.table({"k": [10, 20, 30], "v": [1.1, 2.2, 3.3]})
        p = LocalPath(str(tmp_path / "read_me.parquet"))
        leaf = p.as_media()
        leaf.write_table(table)

        result = pl.from_arrow(leaf.read_arrow_table())
        assert isinstance(result, pl.DataFrame)
        assert result["k"].to_list() == [10, 20, 30]
        assert result["v"].to_list() == [1.1, 2.2, 3.3]

    def test_column_types_preserved(self, tmp_path: pathlib.Path) -> None:
        df = pl.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        })
        p = LocalPath(str(tmp_path / "types.parquet"))
        leaf = p.as_media()
        leaf.write_table(df)

        result = pl.from_arrow(leaf.read_arrow_table())
        assert result["int_col"].dtype == pl.Int64
        assert result["float_col"].dtype == pl.Float64
        assert result["str_col"].dtype in (pl.Utf8, pl.String)
        assert result["bool_col"].dtype == pl.Boolean

    def test_large_dataframe(self, tmp_path: pathlib.Path) -> None:
        n = 10_000
        df = pl.DataFrame({
            "idx": list(range(n)),
            "value": [float(i) * 0.1 for i in range(n)],
        })
        p = LocalPath(str(tmp_path / "large.parquet"))
        leaf = p.as_media()
        leaf.write_table(df)

        result = pl.from_arrow(leaf.read_arrow_table())
        assert result.shape == (n, 2)
        assert result["idx"].to_list()[0] == 0
        assert result["idx"].to_list()[-1] == n - 1


# ---------------------------------------------------------------------------
# TestPolarsRoundTrip
# ---------------------------------------------------------------------------


class TestPolarsRoundTrip:

    def test_polars_ipc_round_trip(self, tmp_path: pathlib.Path) -> None:
        original = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        p = LocalPath(str(tmp_path / "rt.ipc"))
        leaf = p.as_media()
        leaf.write_table(original)

        restored = pl.from_arrow(leaf.read_arrow_table())
        assert restored.equals(original)

    def test_polars_parquet_round_trip(self, tmp_path: pathlib.Path) -> None:
        original = pl.DataFrame({"x": [10, 20], "y": ["hello", "world"]})
        p = LocalPath(str(tmp_path / "rt.parquet"))
        leaf = p.as_media()
        leaf.write_table(original)

        restored = pl.from_arrow(leaf.read_arrow_table())
        assert restored.equals(original)

    def test_polars_csv_round_trip(self, tmp_path: pathlib.Path) -> None:
        original = pl.DataFrame({"name": ["alice", "bob"], "city": ["paris", "london"]})
        p = LocalPath(str(tmp_path / "rt.csv"))
        leaf = p.as_media()
        leaf.write_table(original)

        restored = pl.from_arrow(leaf.read_arrow_table())
        # CSV round-trip keeps string columns intact
        assert restored["name"].to_list() == ["alice", "bob"]
        assert restored["city"].to_list() == ["paris", "london"]

    def test_nested_list_column_via_parquet(self, tmp_path: pathlib.Path) -> None:
        original = pl.DataFrame({
            "id": [1, 2, 3],
            "tags": [["a", "b"], ["c"], ["d", "e", "f"]],
        })
        p = LocalPath(str(tmp_path / "nested.parquet"))
        leaf = p.as_media()
        leaf.write_table(original)

        restored = pl.from_arrow(leaf.read_arrow_table())
        assert restored["id"].to_list() == [1, 2, 3]
        assert restored["tags"].to_list() == [["a", "b"], ["c"], ["d", "e", "f"]]
