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

    def test_projection_columns_resolves_target_subset(self):
        # The projection is the target's columns (target order) intersected
        # with the file, applied as a zero-copy select.
        from yggdrasil.io.primitive.arrow_ipc_file import ArrowIPCFile
        from yggdrasil.data.options import CastOptions

        names = ["id", "price", "name"]
        opt = CastOptions(target=schema(fields=[field("name", pa.string()),
                                                field("id", pa.int64())]))
        assert ArrowIPCFile._projection_columns(opt, names) == ["name", "id"]
        # No target / full coverage / no file → read everything.
        assert ArrowIPCFile._projection_columns(CastOptions(), names) is None
        full = schema(fields=[field(n, pa.int64()) for n in names])
        assert ArrowIPCFile._projection_columns(CastOptions(target=full), names) is None

    def test_projected_read_stays_a_zero_copy_view(self):
        # A projected read keeps the kept column viewing the source buffer
        # (no decode copy), not just the full read.
        import pyarrow as _pa
        holder = Memory()
        ArrowIPCFile(holder=holder, mode="wb").write_arrow_table(_source_table())
        mv = holder.read_mv(-1, 0)
        base = _pa.py_buffer(mv).address
        end = base + len(mv)
        out = ArrowIPCFile(holder=holder, mode="rb").read_arrow_table(
            target=schema(fields=[field("price", pa.float64())]))
        buf = out.column("price").chunk(0).buffers()[1]
        assert base <= buf.address < end   # still a view into the source

    def test_projection_with_column_absent_from_file_fills_null(self):
        # A target column the file doesn't carry must survive the pushdown —
        # it's just not in included_fields, and the cast fills it with nulls.
        target = schema(fields=[field("id", pa.int64()), field("missing", pa.int64())])
        out = self._write_and_read(target=target)
        assert out.column_names == ["id", "missing"]
        assert out.column("id").to_pylist() == [1, 2, 3]
        assert out.column("missing").to_pylist() == [None, None, None]


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


class TestAutoMergeFillsVariants:
    """``Mode.AUTO`` merge completes autotyping: variant (``ObjectType`` /
    ``NullType``) target slots adopt the matching source field's dtype by name,
    while source-only columns are never pulled in (the field set stays the
    target's). This is what turns a bare ``columns=`` projection into a
    fully-typed target once the source schema is known."""

    def test_auto_fills_object_and_null_without_adding_columns(self):
        from yggdrasil.data.types.primitive.object import ObjectType
        from yggdrasil.data.types.primitive.null import NullType
        from yggdrasil.enums import Mode

        src = field("", schema(fields=[
            field("id", pa.int64()), field("price", pa.float64()),
            field("name", pa.string()),
        ]).dtype)
        tgt = field("", schema(fields=[
            field("id", ObjectType()), field("price", NullType()),
        ]).dtype)
        merged = tgt.merge_with(src, mode=Mode.AUTO)
        # object -> int64, null -> float64; 'name' is NOT added.
        assert merged.names == ["id", "price"]
        assert merged.field(name="id").dtype.to_arrow() == pa.int64()
        assert merged.field(name="price").dtype.to_arrow() == pa.float64()

    def test_auto_prefers_target_no_width_shrink_or_widen(self):
        from yggdrasil.enums import Mode

        # concrete target wins outright — no int32->int64 widen from source.
        src = field("", schema(fields=[field("id", pa.int64())]).dtype)
        tgt = field("", schema(fields=[field("id", pa.int32())]).dtype)
        merged = tgt.merge_with(src, mode=Mode.AUTO)
        assert merged.field(name="id").dtype.to_arrow() == pa.int32()

    def test_default_schema_mode_is_auto_and_cast_fills_via_merged(self):
        from yggdrasil.data.types.primitive.object import ObjectType
        from yggdrasil.data.options import CastOptions
        from yggdrasil.enums import Mode

        assert CastOptions().schema_mode is Mode.AUTO
        src = pa.table({"id": pa.array([1, 2, 3], pa.int64()),
                        "x": pa.array([1.5, 2.5, 3.5], pa.float64())})
        # An object-typed target + bound source: the cast runs against
        # ``merged`` so the projection autotypes to the source dtype.
        opt = CastOptions(
            source=schema(fields=[field("id", pa.int64()), field("x", pa.float64())]),
            target=schema(fields=[field("id", ObjectType())]),
        )
        out = opt.cast_arrow(src)
        assert out.column_names == ["id"]
        assert out.schema.field("id").type == pa.int64()


class TestProjectionWithPredicate:
    """columns= projection + a predicate that filters on a *dropped* column
    must keep that column through the read (for the filter) and still return
    only the projected columns — and an empty result keeps concrete types."""

    def _folder(self, tmp_path):
        from yggdrasil.path.folder import Folder
        sch = pa.schema([
            pa.field("partition_key", pa.int64(), metadata={b"t:partition_by": b"True"}),
            pa.field("request_public_hash", pa.int64()),
        ])
        t = pa.table({"partition_key": pa.array([1, 1, 2], pa.int64()),
                      "request_public_hash": pa.array([111, 222, 333], pa.int64())},
                     schema=sch)
        Folder(path=str(tmp_path / "c")).write_arrow_table(t)
        return Folder(path=str(tmp_path / "c"))

    def test_predicate_on_projected_out_partition_column(self, tmp_path):
        from yggdrasil.execution.expr import col
        out = self._folder(tmp_path).read_arrow_table(
            predicate=col("partition_key").is_in([1]),
            columns=["request_public_hash"],
        )
        assert out.column_names == ["request_public_hash"]
        assert sorted(out.column("request_public_hash").to_pylist()) == [111, 222]
        assert out.schema.field("request_public_hash").type == pa.int64()

    def test_empty_result_keeps_concrete_type_not_objecttype(self, tmp_path):
        from yggdrasil.execution.expr import col
        fp = self._folder(tmp_path)
        fp._schema_cache = ...   # cold cache → fallback resolves via merged
        out = fp.read_arrow_table(
            predicate=col("partition_key").is_in([99]),   # matches nothing
            columns=["request_public_hash"],
        )
        assert out.num_rows == 0
        assert out.column_names == ["request_public_hash"]
        assert out.schema.field("request_public_hash").type == pa.int64()

    def test_direct_parquet_columns_plus_predicate_on_dropped_column(self, tmp_path):
        import pyarrow.parquet as pq
        from yggdrasil.io.primitive.parquet_file import ParquetFile
        from yggdrasil.path.local_path import LocalPath
        from yggdrasil.execution.expr import col

        path = str(tmp_path / "f.parquet")
        pq.write_table(pa.table({"x": [1, 2, 3], "y": [10, 20, 30]}), path)
        out = ParquetFile(holder=LocalPath.from_(path)).read_arrow_table(
            predicate=col("x") == 2, columns=["y"],
        )
        assert out.column_names == ["y"]
        assert out.column("y").to_pylist() == [20]
