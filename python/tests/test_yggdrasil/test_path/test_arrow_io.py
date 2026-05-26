"""Arrow IPC and Parquet read/write tests on LocalPath."""
from __future__ import annotations

import pathlib

import pyarrow as pa
import pytest

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.enums import Mode
from yggdrasil.path.local_path import LocalPath


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ipc_leaf(tmp_path: pathlib.Path, name: str = "data.ipc"):
    p = LocalPath(str(tmp_path / name), singleton_ttl=False)
    return p.as_media("arrow")


def _parquet_leaf(tmp_path: pathlib.Path, name: str = "data.parquet"):
    p = LocalPath(str(tmp_path / name), singleton_ttl=False)
    return p.as_media("parquet")


def _simple_table(n: int = 3) -> pa.Table:
    return pa.table({"x": list(range(n)), "y": [f"v{i}" for i in range(n)]})


# ---------------------------------------------------------------------------
# TestArrowIPCReadWrite
# ---------------------------------------------------------------------------


class TestArrowIPCReadWrite:

    def test_write_then_read_roundtrip(self, tmp_path: pathlib.Path) -> None:
        leaf = _ipc_leaf(tmp_path)
        table = _simple_table()
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.num_rows == 3
        assert result.column("x").to_pylist() == [0, 1, 2]
        assert result.column("y").to_pylist() == ["v0", "v1", "v2"]

    def test_write_record_batch_then_read_batches(self, tmp_path: pathlib.Path) -> None:
        leaf = _ipc_leaf(tmp_path)
        batch = pa.record_batch({"a": [10, 20, 30]})
        leaf.write_arrow_batches([batch], mode=Mode.OVERWRITE)
        batches = list(leaf.read_arrow_batches())
        total = sum(b.num_rows for b in batches)
        assert total == 3
        all_a = pa.concat_tables(
            [pa.Table.from_batches([b]) for b in batches]
        ).column("a").to_pylist()
        assert all_a == [10, 20, 30]

    def test_write_multiple_batches(self, tmp_path: pathlib.Path) -> None:
        leaf = _ipc_leaf(tmp_path)
        b1 = pa.record_batch({"k": [1, 2]})
        b2 = pa.record_batch({"k": [3, 4]})
        b3 = pa.record_batch({"k": [5]})
        leaf.write_arrow_batches([b1, b2, b3], mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.num_rows == 5
        assert sorted(result.column("k").to_pylist()) == [1, 2, 3, 4, 5]

    def test_overwrite_replaces_data(self, tmp_path: pathlib.Path) -> None:
        leaf = _ipc_leaf(tmp_path)
        leaf.write_arrow_table(_simple_table(5), mode=Mode.OVERWRITE)
        assert leaf.read_arrow_table().num_rows == 5
        leaf.write_arrow_table(pa.table({"x": [99], "y": ["only"]}), mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.num_rows == 1
        assert result.column("x").to_pylist() == [99]

    def test_append_adds_rows(self, tmp_path: pathlib.Path) -> None:
        leaf = _ipc_leaf(tmp_path)
        leaf.write_arrow_table(_simple_table(3), mode=Mode.OVERWRITE)
        leaf.write_arrow_table(_simple_table(3), mode=Mode.APPEND)
        result = leaf.read_arrow_table()
        assert result.num_rows == 6

    def test_read_empty_file_returns_zero_rows(self, tmp_path: pathlib.Path) -> None:
        leaf = _ipc_leaf(tmp_path)
        table = pa.table({"x": pa.array([], type=pa.int64())})
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.num_rows == 0

    def test_large_table_roundtrip(self, tmp_path: pathlib.Path) -> None:
        leaf = _ipc_leaf(tmp_path)
        n = 10_000
        table = pa.table({"id": list(range(n)), "val": [f"row_{i}" for i in range(n)]})
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.num_rows == n
        assert result.column("id").to_pylist()[-1] == n - 1

    def test_schema_preserved(self, tmp_path: pathlib.Path) -> None:
        leaf = _ipc_leaf(tmp_path)
        schema = pa.schema([
            ("i", pa.int32()),
            ("f", pa.float64()),
            ("s", pa.utf8()),
            ("b", pa.bool_()),
        ])
        table = pa.table(
            {"i": pa.array([1], type=pa.int32()),
             "f": pa.array([3.14], type=pa.float64()),
             "s": pa.array(["hello"], type=pa.utf8()),
             "b": pa.array([True], type=pa.bool_())},
            schema=schema,
        )
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        for field in schema:
            assert result.schema.field(field.name).type == field.type


# ---------------------------------------------------------------------------
# TestParquetReadWrite
# ---------------------------------------------------------------------------


class TestParquetReadWrite:

    def test_write_read_roundtrip(self, tmp_path: pathlib.Path) -> None:
        leaf = _parquet_leaf(tmp_path)
        table = _simple_table()
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.num_rows == 3
        assert result.column("x").to_pylist() == [0, 1, 2]

    def test_overwrite_replaces(self, tmp_path: pathlib.Path) -> None:
        leaf = _parquet_leaf(tmp_path)
        leaf.write_arrow_table(pa.table({"v": [1, 2, 3]}), mode=Mode.OVERWRITE)
        leaf.write_arrow_table(pa.table({"v": [42]}), mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.num_rows == 1
        assert result.column("v").to_pylist() == [42]

    def test_column_types_preserved(self, tmp_path: pathlib.Path) -> None:
        leaf = _parquet_leaf(tmp_path)
        table = pa.table({
            "i": pa.array([1, 2], type=pa.int64()),
            "f": pa.array([1.5, 2.5], type=pa.float64()),
            "s": pa.array(["a", "b"], type=pa.utf8()),
            "b": pa.array([True, False], type=pa.bool_()),
            "ts": pa.array([1_000_000, 2_000_000], type=pa.timestamp("us")),
        })
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.schema.field("i").type == pa.int64()
        assert result.schema.field("f").type == pa.float64()
        assert result.schema.field("s").type in (pa.utf8(), pa.large_utf8(), pa.string_view())
        assert result.schema.field("b").type == pa.bool_()
        assert pa.types.is_timestamp(result.schema.field("ts").type)

    def test_nullable_columns(self, tmp_path: pathlib.Path) -> None:
        leaf = _parquet_leaf(tmp_path)
        table = pa.table({
            "a": pa.array([1, None, 3], type=pa.int64()),
            "b": pa.array(["x", None, "z"], type=pa.utf8()),
        })
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.column("a").to_pylist() == [1, None, 3]
        assert result.column("b").to_pylist() == ["x", None, "z"]

    def test_nested_struct_column(self, tmp_path: pathlib.Path) -> None:
        leaf = _parquet_leaf(tmp_path)
        struct_type = pa.struct([("sub_a", pa.int32()), ("sub_b", pa.utf8())])
        arr = pa.array(
            [{"sub_a": 1, "sub_b": "foo"}, {"sub_a": 2, "sub_b": "bar"}],
            type=struct_type,
        )
        table = pa.table({"nested": arr})
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.num_rows == 2
        assert pa.types.is_struct(result.schema.field("nested").type)
        values = result.column("nested").to_pylist()
        assert values[0]["sub_a"] == 1
        assert values[1]["sub_b"] == "bar"

    def test_large_parquet_roundtrip(self, tmp_path: pathlib.Path) -> None:
        leaf = _parquet_leaf(tmp_path)
        n = 10_000
        table = pa.table({"id": list(range(n)), "score": [float(i) for i in range(n)]})
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.num_rows == n


# ---------------------------------------------------------------------------
# TestCastOptions
# ---------------------------------------------------------------------------


class TestCastOptions:

    def test_read_with_column_selection(self, tmp_path: pathlib.Path) -> None:
        leaf = _ipc_leaf(tmp_path)
        table = pa.table({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        target = Schema.from_arrow(pa.schema([("a", pa.int64()), ("c", pa.int64())]))
        result = leaf.read_arrow_table(options=CastOptions(target=target))
        assert set(result.column_names) == {"a", "c"}
        assert result.column("a").to_pylist() == [1, 2]
        assert result.column("c").to_pylist() == [5, 6]

    def test_read_with_predicate_filter(self, tmp_path: pathlib.Path) -> None:
        from yggdrasil.execution.expr import col

        leaf = _ipc_leaf(tmp_path)
        table = pa.table({"x": [1, 2, 3, 4, 5], "y": ["a", "b", "c", "d", "e"]})
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        opts = CastOptions(predicate=(col("x") > 3))
        result = leaf.read_arrow_table(options=opts)
        assert result.column("x").to_pylist() == [4, 5]
        assert result.column("y").to_pylist() == ["d", "e"]

    def test_read_with_mode(self, tmp_path: pathlib.Path) -> None:
        leaf = _ipc_leaf(tmp_path)
        table = pa.table({"v": [10, 20]})
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table(options=CastOptions(mode=Mode.OVERWRITE))
        assert result.num_rows == 2

    def test_target_projects_columns(self, tmp_path: pathlib.Path) -> None:
        leaf = _parquet_leaf(tmp_path)
        table = pa.table({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        target = Schema.from_arrow(pa.schema([("y", pa.int64())]))
        result = leaf.read_arrow_table(options=CastOptions(target=target))
        assert result.column_names == ["y"]
        assert result.column("y").to_pylist() == [3, 4]

    def test_target_schema_filters_on_read(self, tmp_path: pathlib.Path) -> None:
        leaf = _ipc_leaf(tmp_path)
        full_table = pa.table({
            "id": [1, 2, 3],
            "name": ["alice", "bob", "carol"],
            "age": [30, 25, 35],
        })
        leaf.write_arrow_table(full_table, mode=Mode.OVERWRITE)
        subset_schema = Schema.from_arrow(
            pa.schema([("id", pa.int64()), ("name", pa.large_utf8())])
        )
        result = leaf.read_arrow_table(options=CastOptions(target=subset_schema))
        assert set(result.column_names) == {"id", "name"}
        assert result.num_rows == 3


# ---------------------------------------------------------------------------
# TestWriteModes
# ---------------------------------------------------------------------------


class TestWriteModes:

    def test_overwrite_clears_and_replaces(self, tmp_path: pathlib.Path) -> None:
        leaf = _ipc_leaf(tmp_path)
        leaf.write_arrow_table(pa.table({"v": [1, 2, 3, 4, 5]}), mode=Mode.OVERWRITE)
        leaf.write_arrow_table(pa.table({"v": [99]}), mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        assert result.num_rows == 1
        assert result.column("v").to_pylist() == [99]

    def test_append_adds_rows(self, tmp_path: pathlib.Path) -> None:
        leaf = _ipc_leaf(tmp_path)
        t = pa.table({"n": [1, 2]})
        leaf.write_arrow_table(t, mode=Mode.OVERWRITE)
        leaf.write_arrow_table(t, mode=Mode.APPEND)
        result = leaf.read_arrow_table()
        assert result.num_rows == 4
        assert sorted(result.column("n").to_pylist()) == [1, 1, 2, 2]

    def test_ignore_noop_when_exists(self, tmp_path: pathlib.Path) -> None:
        leaf = _ipc_leaf(tmp_path)
        leaf.write_arrow_table(pa.table({"v": [1]}), mode=Mode.OVERWRITE)
        leaf.write_arrow_table(pa.table({"v": [999]}), mode=Mode.IGNORE)
        result = leaf.read_arrow_table()
        assert result.column("v").to_pylist() == [1]

    def test_error_if_exists_raises(self, tmp_path: pathlib.Path) -> None:
        leaf = _ipc_leaf(tmp_path)
        leaf.write_arrow_table(pa.table({"v": [1]}), mode=Mode.OVERWRITE)
        with pytest.raises(FileExistsError):
            leaf.write_arrow_table(pa.table({"v": [2]}), mode=Mode.ERROR_IF_EXISTS)

    def test_auto_default_behavior(self, tmp_path: pathlib.Path) -> None:
        leaf = _ipc_leaf(tmp_path)
        leaf.write_arrow_table(pa.table({"v": [1]}), mode=Mode.AUTO)
        result = leaf.read_arrow_table()
        assert result.num_rows == 1
        # AUTO on second write appends (no match_by set)
        leaf.write_arrow_table(pa.table({"v": [2]}), mode=Mode.AUTO)
        result = leaf.read_arrow_table()
        assert result.num_rows == 2


# ---------------------------------------------------------------------------
# TestSchemaInference
# ---------------------------------------------------------------------------


class TestSchemaInference:

    def test_collect_schema_ipc(self, tmp_path: pathlib.Path) -> None:
        leaf = _ipc_leaf(tmp_path)
        table = pa.table({
            "id": pa.array([1, 2], type=pa.int64()),
            "name": pa.array(["a", "b"], type=pa.utf8()),
        })
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        schema = leaf.collect_schema()
        assert "id" in schema
        assert "name" in schema

    def test_collect_schema_parquet(self, tmp_path: pathlib.Path) -> None:
        leaf = _parquet_leaf(tmp_path)
        table = pa.table({
            "x": pa.array([1.0, 2.0], type=pa.float64()),
            "flag": pa.array([True, False], type=pa.bool_()),
        })
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        schema = leaf.collect_schema()
        assert "x" in schema
        assert "flag" in schema

    def test_schema_metadata_preserved(self, tmp_path: pathlib.Path) -> None:
        leaf = _ipc_leaf(tmp_path)
        arrow_schema = pa.schema([
            pa.field("a", pa.int64(), metadata={b"custom_key": b"custom_val"}),
            pa.field("b", pa.utf8()),
        ])
        table = pa.table(
            {"a": pa.array([1], type=pa.int64()), "b": pa.array(["z"], type=pa.utf8())},
            schema=arrow_schema,
        )
        leaf.write_arrow_table(table, mode=Mode.OVERWRITE)
        result = leaf.read_arrow_table()
        restored_meta = result.schema.field("a").metadata
        assert restored_meta is not None
        assert restored_meta[b"custom_key"] == b"custom_val"
