"""Tests for explicit target schema cast across all tabular IO backends.

Verifies that ``read_arrow_table(target=...)`` and
``read_arrow_table(columns=...)`` correctly project and cast columns
across ArrowIPC, Parquet, and Folder backends.
"""
from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data import field, schema
from yggdrasil.enums.media_type import MediaTypes
from yggdrasil.path.memory import Memory
from yggdrasil.path.folder import Folder, FolderOptions
from yggdrasil.io.primitive.arrow_ipc_file import ArrowIPCFile
from yggdrasil.io.primitive.parquet_file import ParquetFile


def _source_table() -> pa.Table:
    return pa.table({
        "id": pa.array([1, 2, 3], pa.int64()),
        "price": pa.array([1.5, 2.5, 3.5], pa.float64()),
        "name": pa.array(["a", "b", "c"], pa.string()),
    })


def _target_schema() -> "schema":
    return schema(fields=[
        field("id", pa.int64()),
        field("price", pa.float64()),
        field("name", pa.string()),
    ])


class TestArrowIPCTargetCast:

    def _write_and_read(self, target=None, columns=None):
        holder = Memory()
        ipc = ArrowIPCFile(holder=holder, mode="wb")
        ipc.write_arrow_table(_source_table())
        ipc = ArrowIPCFile(holder=holder, mode="rb")
        return ipc.read_arrow_table(target=target, columns=columns)

    def test_target_selects_columns(self):
        target = schema(fields=[field("id", pa.int64()), field("name", pa.string())])
        out = self._write_and_read(target=target)
        assert out.column_names == ["id", "name"]
        assert out.column("id").to_pylist() == [1, 2, 3]
        assert out.column("name").to_pylist() == ["a", "b", "c"]

    def test_target_casts_datatype(self):
        target = schema(fields=[
            field("id", pa.int32()),
            field("price", pa.float32()),
        ])
        out = self._write_and_read(target=target)
        assert out.schema.field("id").type == pa.int32()
        assert out.schema.field("price").type == pa.float32()
        assert out.column("id").to_pylist() == [1, 2, 3]

    def test_columns_kwarg_projects(self):
        out = self._write_and_read(columns=["id"])
        assert out.column_names == ["id"]
        assert out.column("id").to_pylist() == [1, 2, 3]

    def test_target_plus_columns_filters_target(self):
        target = _target_schema()
        out = self._write_and_read(target=target, columns=["id", "price"])
        assert out.column_names == ["id", "price"]


class TestParquetTargetCast:

    def _write_and_read(self, target=None, columns=None):
        holder = Memory()
        pq = ParquetFile(holder=holder, mode="wb")
        pq.write_arrow_table(_source_table())
        pq = ParquetFile(holder=holder, mode="rb")
        return pq.read_arrow_table(target=target, columns=columns)

    def test_target_selects_columns(self):
        target = schema(fields=[field("id", pa.int64()), field("name", pa.string())])
        out = self._write_and_read(target=target)
        assert out.column_names == ["id", "name"]
        assert out.column("id").to_pylist() == [1, 2, 3]

    def test_target_casts_datatype(self):
        target = schema(fields=[
            field("id", pa.int32()),
            field("price", pa.float32()),
        ])
        out = self._write_and_read(target=target)
        assert out.schema.field("id").type == pa.int32()
        assert out.schema.field("price").type == pa.float32()
        assert out.column("id").to_pylist() == [1, 2, 3]

    def test_columns_kwarg_projects(self):
        out = self._write_and_read(columns=["id"])
        assert out.column_names == ["id"]
        assert out.column("id").to_pylist() == [1, 2, 3]

    def test_target_plus_columns_filters_target(self):
        target = _target_schema()
        out = self._write_and_read(target=target, columns=["id", "price"])
        assert out.column_names == ["id", "price"]


class TestFolderPathTargetCast:

    def _write_and_read(self, tmp_path, target=None, columns=None):
        folder = Folder(path=str(tmp_path))
        folder.write_arrow_table(_source_table())
        return folder.read_arrow_table(target=target, columns=columns)

    def test_target_selects_columns(self, tmp_path):
        target = schema(fields=[field("id", pa.int64()), field("name", pa.string())])
        out = self._write_and_read(tmp_path, target=target)
        assert out.column_names == ["id", "name"]
        assert out.column("id").to_pylist() == [1, 2, 3]

    def test_target_casts_datatype(self, tmp_path):
        target = schema(fields=[
            field("id", pa.int32()),
            field("price", pa.float32()),
        ])
        out = self._write_and_read(tmp_path, target=target)
        assert out.schema.field("id").type == pa.int32()
        assert out.schema.field("price").type == pa.float32()
        assert out.column("id").to_pylist() == [1, 2, 3]

    def test_columns_kwarg_projects(self, tmp_path):
        out = self._write_and_read(tmp_path, columns=["id"])
        assert out.column_names == ["id"]
        assert out.column("id").to_pylist() == [1, 2, 3]

    def test_target_plus_columns_filters_target(self, tmp_path):
        target = _target_schema()
        out = self._write_and_read(tmp_path, target=target, columns=["id", "price"])
        assert out.column_names == ["id", "price"]

    def test_target_casts_datatype_partitioned(self, tmp_path):
        src_schema = pa.schema([
            pa.field("pk", pa.int64(), metadata={b"t:partition_by": b"True"}),
            pa.field("id", pa.int64()),
            pa.field("price", pa.float64()),
        ])
        folder = Folder(path=str(tmp_path))
        batch = pa.record_batch(
            [pa.array([1, 2], pa.int64()), pa.array([10, 20], pa.int64()), pa.array([1.5, 2.5], pa.float64())],
            schema=src_schema,
        )
        folder.write_arrow_batches((batch,))

        target = schema(fields=[field("id", pa.int32()), field("price", pa.float32())])
        out = folder.read_arrow_table(target=target, columns=["id"])
        assert out.column_names == ["id"]
        assert out.schema.field("id").type == pa.int32()


class TestBatchLevelTargetCast:
    """Target cast applies per-batch via read_arrow_batches, not just table."""

    def test_arrow_ipc_batches_cast_dtype(self):
        holder = Memory()
        ipc = ArrowIPCFile(holder=holder, mode="wb")
        ipc.write_arrow_table(_source_table())
        ipc = ArrowIPCFile(holder=holder, mode="rb")
        target = schema(fields=[field("id", pa.int32()), field("price", pa.float32())])
        batches = list(ipc.read_arrow_batches(target=target))
        assert len(batches) >= 1
        assert batches[0].schema.field("id").type == pa.int32()
        assert batches[0].schema.field("price").type == pa.float32()

    def test_parquet_batches_cast_dtype(self):
        holder = Memory()
        pq = ParquetFile(holder=holder, mode="wb")
        pq.write_arrow_table(_source_table())
        pq = ParquetFile(holder=holder, mode="rb")
        target = schema(fields=[field("id", pa.int32()), field("price", pa.float32())])
        batches = list(pq.read_arrow_batches(target=target))
        assert len(batches) >= 1
        assert batches[0].schema.field("id").type == pa.int32()
        assert batches[0].schema.field("price").type == pa.float32()

    def test_folder_batches_cast_dtype(self, tmp_path):
        folder = Folder(path=str(tmp_path))
        folder.write_arrow_table(_source_table())
        target = schema(fields=[field("id", pa.int32()), field("name", pa.large_string())])
        batches = list(folder.read_arrow_batches(target=target))
        assert len(batches) >= 1
        assert batches[0].schema.field("id").type == pa.int32()
        assert batches[0].schema.field("name").type == pa.large_string()

    def test_select_by_index(self):
        s = schema(fields=[field("a", pa.int64()), field("b", pa.string()), field("c", pa.float64())])
        out = s.select(0, 2)
        assert out.names == ["a", "c"]

    def test_drop_by_index(self):
        s = schema(fields=[field("a", pa.int64()), field("b", pa.string()), field("c", pa.float64())])
        out = s.drop(1)
        assert out.names == ["a", "c"]


class TestFieldSelectDrop:

    def test_select_by_name(self):
        s = schema(fields=[field("a", pa.int64()), field("b", pa.string()), field("c", pa.float64())])
        out = s.select("a", "c")
        assert out.names == ["a", "c"]

    def test_select_by_list(self):
        s = schema(fields=[field("a", pa.int64()), field("b", pa.string())])
        out = s.select(["a", "b"])
        assert out.names == ["a", "b"]

    def test_select_preserves_types(self):
        s = schema(fields=[field("a", pa.int64()), field("b", pa.string())])
        out = s.select("a")
        assert out.children[0].dtype.to_arrow() == pa.int64()

    def test_select_skips_none(self):
        s = schema(fields=[field("a", pa.int64()), field("b", pa.string())])
        out = s.select("a", None)
        assert out.names == ["a"]

    def test_select_missing_skipped(self):
        s = schema(fields=[field("a", pa.int64()), field("b", pa.string())])
        out = s.select("a", "missing")
        assert out.names == ["a"]

    def test_drop_by_name(self):
        s = schema(fields=[field("a", pa.int64()), field("b", pa.string()), field("c", pa.float64())])
        out = s.drop("b")
        assert out.names == ["a", "c"]

    def test_drop_multiple(self):
        s = schema(fields=[field("a", pa.int64()), field("b", pa.string()), field("c", pa.float64())])
        out = s.drop("a", "c")
        assert out.names == ["b"]

    def test_drop_preserves_types(self):
        s = schema(fields=[field("a", pa.int64()), field("b", pa.string())])
        out = s.drop("a")
        assert out.children[0].dtype.to_arrow() == pa.string()

    def test_drop_none_returns_copy(self):
        s = schema(fields=[field("a", pa.int64()), field("b", pa.string())])
        out = s.drop()
        assert out.names == ["a", "b"]


class TestVariantTargetPassthrough:
    """An ``ObjectType`` (variant) target field is a "keep whatever's here"
    passthrough — it projects a column without forcing a coerce to its
    physical ``large_binary`` stand-in, including when nested inside a
    struct / map / list target."""

    def test_top_level_object_keeps_source_type(self):
        from yggdrasil.data.types.primitive.object import ObjectType
        from yggdrasil.data.options import CastOptions

        src = pa.table({"id": pa.array([1, 2, 3], pa.int64()),
                        "x": pa.array([1.5, 2.5, 3.5], pa.float64())})
        tgt = schema(fields=[field("id", ObjectType())])
        out = CastOptions.check(target=tgt).cast_arrow(src)
        assert out.column_names == ["id"]
        assert out.schema.field("id").type == pa.int64()
        assert out.column("id").to_pylist() == [1, 2, 3]

    def test_struct_child_object_keeps_source_type(self):
        from yggdrasil.data.types.primitive.object import ObjectType
        from yggdrasil.data.types.nested.struct import StructType
        from yggdrasil.data.data_field import Field
        from yggdrasil.data.options import CastOptions

        src = pa.table({"s": pa.array([{"a": 1, "b": 2}, {"a": 3, "b": 4}])})
        tgt = schema(fields=[
            field("s", StructType(fields=(Field(name="a", dtype=ObjectType()),)))
        ])
        out = CastOptions.check(target=tgt).cast_arrow(src)
        # 'a' keeps its int64 type (not large_binary), 'b' is projected away.
        assert out.schema.field("s").type == pa.struct([pa.field("a", pa.int64())])
        assert out.column("s").to_pylist() == [{"a": 1}, {"a": 3}]
