from __future__ import annotations

import pytest
import pyarrow as pa
import pyarrow.dataset as ds

from yggdrasil.pickle.ser.pyarrow import (
    ArrowArraySerialized,
    ArrowChunkedArraySerialized,
    ArrowDataTypeSerialized,
    ArrowDatasetSerialized,
    ArrowFieldSerialized,
    ArrowRecordBatchSerialized,
    ArrowScalarSerialized,
    ArrowSchemaSerialized,
    ArrowStreamSerialized,
    ArrowTableSerialized,
    ArrowTensorSerialized,
)
from yggdrasil.pickle.ser.constants import CODEC_NONE
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags


def _sample_table() -> pa.Table:
    return pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int64()),
            "name": pa.array(["a", "b", "c"], type=pa.string()),
            "flag": pa.array([True, False, True], type=pa.bool_()),
        }
    )


def _sample_batch() -> pa.RecordBatch:
    table = _sample_table()
    return table.to_batches()[0]


def _sample_reader() -> pa.RecordBatchReader:
    table = _sample_table()
    return pa.RecordBatchReader.from_batches(table.schema, table.to_batches())


def _assert_tables_equal(left: pa.Table, right: pa.Table) -> None:
    assert left.schema == right.schema
    assert left.equals(right)


def _assert_batches_equal(left: pa.RecordBatch, right: pa.RecordBatch) -> None:
    assert left.schema == right.schema
    assert left.equals(right)


def test_table_roundtrip() -> None:
    obj = _sample_table()

    ser = ArrowTableSerialized.from_value(obj, codec=CODEC_NONE)
    assert isinstance(ser, ArrowTableSerialized)
    assert ser.tag == Tags.ARROW_TABLE
    assert ser.codec == CODEC_NONE

    out = ser.as_python()
    assert isinstance(out, pa.Table)
    _assert_tables_equal(obj, out)


def test_table_roundtrip_via_base_dispatch() -> None:
    obj = _sample_table()

    ser = Serialized.from_python_object(obj, codec=CODEC_NONE)
    assert isinstance(ser, ArrowTableSerialized)

    out = ser.as_python()
    assert isinstance(out, pa.Table)
    _assert_tables_equal(obj, out)


def test_record_batch_roundtrip() -> None:
    obj = _sample_batch()

    ser = ArrowRecordBatchSerialized.from_value(obj, codec=CODEC_NONE)
    assert isinstance(ser, ArrowRecordBatchSerialized)
    assert ser.tag == Tags.ARROW_RECORD_BATCH

    out = ser.as_python()
    assert isinstance(out, pa.RecordBatch)
    _assert_batches_equal(obj, out)


def test_record_batch_roundtrip_via_base_dispatch() -> None:
    obj = _sample_batch()

    ser = Serialized.from_python_object(obj, codec=CODEC_NONE)
    assert isinstance(ser, ArrowRecordBatchSerialized)

    out = ser.as_python()
    assert isinstance(out, pa.RecordBatch)
    _assert_batches_equal(obj, out)


def test_stream_roundtrip() -> None:
    obj = _sample_reader()

    ser = ArrowStreamSerialized.from_value(obj, codec=CODEC_NONE)
    assert isinstance(ser, ArrowStreamSerialized)
    assert ser.tag == Tags.ARROW_STREAM

    out = ser.as_python()
    assert isinstance(out, pa.RecordBatchReader)

    out_table = out.read_all()
    _assert_tables_equal(_sample_table(), out_table)


def test_stream_roundtrip_via_base_dispatch() -> None:
    obj = _sample_reader()

    ser = Serialized.from_python_object(obj, codec=CODEC_NONE)
    assert isinstance(ser, ArrowStreamSerialized)

    out = ser.as_python()
    assert isinstance(out, pa.RecordBatchReader)
    _assert_tables_equal(_sample_table(), out.read_all())


def test_dataset_roundtrip_default_file_encoding() -> None:
    table = _sample_table()
    dataset = ds.dataset(table)

    ser = ArrowDatasetSerialized.from_value(dataset, codec=CODEC_NONE)
    assert isinstance(ser, ArrowDatasetSerialized)
    assert ser.tag == Tags.ARROW_DATASET
    assert ser.metadata is not None
    assert ser.metadata[b"arrow_encoding"] == b"ipc_file"

    out = ser.as_python()
    assert isinstance(out, ds.Dataset)

    out_table = out.to_table()
    _assert_tables_equal(table, out_table)


def test_dataset_roundtrip_stream_encoding() -> None:
    table = _sample_table()
    dataset = ds.dataset(table)

    ser = ArrowDatasetSerialized.from_value(
        dataset,
        codec=CODEC_NONE,
        use_stream=True,
    )
    assert isinstance(ser, ArrowDatasetSerialized)
    assert ser.metadata is not None
    assert ser.metadata[b"arrow_encoding"] == b"ipc_stream"

    out = ser.as_python()
    assert isinstance(out, ds.Dataset)

    out_table = out.to_table()
    _assert_tables_equal(table, out_table)


def test_schema_roundtrip() -> None:
    obj = _sample_table().schema

    ser = ArrowSchemaSerialized.from_value(obj, codec=CODEC_NONE)
    assert isinstance(ser, ArrowSchemaSerialized)
    assert ser.tag == Tags.ARROW_SCHEMA

    out = ser.as_python()
    assert isinstance(out, pa.Schema)
    assert out == obj


def test_field_roundtrip() -> None:
    obj = pa.field("price", pa.decimal128(18, 4), nullable=False)

    ser = ArrowFieldSerialized.from_value(obj, codec=CODEC_NONE)
    assert isinstance(ser, ArrowFieldSerialized)
    assert ser.tag == Tags.ARROW_FIELD

    out = ser.as_python()
    assert isinstance(out, pa.Field)
    assert out == obj


def test_data_type_roundtrip() -> None:
    obj = pa.list_(pa.struct([("x", pa.int32()), ("y", pa.string())]))

    ser = ArrowDataTypeSerialized.from_value(obj, codec=CODEC_NONE)
    assert isinstance(ser, ArrowDataTypeSerialized)
    assert ser.tag == Tags.ARROW_DATA_TYPE

    out = ser.as_python()
    assert isinstance(out, pa.DataType)
    assert out == obj


def test_array_roundtrip() -> None:
    obj = pa.array([1, 2, None, 4], type=pa.int64())

    ser = ArrowArraySerialized.from_value(obj, codec=CODEC_NONE)
    assert isinstance(ser, ArrowArraySerialized)
    assert ser.tag == Tags.ARROW_ARRAY

    out = ser.as_python()
    assert isinstance(out, pa.Array)
    assert out.equals(obj)


def test_chunked_array_roundtrip() -> None:
    obj = pa.chunked_array(
        [
            pa.array([1, 2], type=pa.int64()),
            pa.array([3, None, 5], type=pa.int64()),
        ]
    )

    ser = ArrowChunkedArraySerialized.from_value(obj, codec=CODEC_NONE)
    assert isinstance(ser, ArrowChunkedArraySerialized)
    assert ser.tag == Tags.ARROW_CHUNKED_ARRAY

    out = ser.as_python()
    assert isinstance(out, pa.ChunkedArray)
    assert out.equals(obj)


def test_scalar_roundtrip() -> None:
    obj = pa.scalar("volcano", type=pa.string())

    ser = ArrowScalarSerialized.from_value(obj, codec=CODEC_NONE)
    assert isinstance(ser, ArrowScalarSerialized)
    assert ser.tag == Tags.ARROW_SCALAR

    out = ser.as_python()
    assert isinstance(out, pa.Scalar)
    assert out == obj


@pytest.mark.skipif(not hasattr(pa, "Tensor"), reason="pyarrow Tensor not available")
def test_tensor_roundtrip() -> None:
    import numpy as np

    obj = pa.Tensor.from_numpy(np.arange(12, dtype=np.int64).reshape(3, 4))

    ser = ArrowTensorSerialized.from_value(obj, codec=CODEC_NONE)
    assert isinstance(ser, ArrowTensorSerialized)
    assert ser.tag == Tags.ARROW_TENSOR

    out = ser.as_python()
    assert isinstance(out, pa.Tensor)

    assert out.shape == obj.shape
    assert out.type == obj.type
    assert out.to_numpy().tolist() == obj.to_numpy().tolist()


def test_base_read_from_returns_specialized_arrow_type_for_table() -> None:
    obj = _sample_table()

    ser = ArrowTableSerialized.from_value(obj, codec=CODEC_NONE)
    buf = ser.write_to()

    reread = Serialized.read_from(buf, pos=0)
    assert isinstance(reread, ArrowTableSerialized)
    _assert_tables_equal(obj, reread.as_python())


def test_base_read_from_returns_specialized_arrow_type_for_dataset() -> None:
    obj = ds.dataset(_sample_table())

    ser = ArrowDatasetSerialized.from_value(obj, codec=CODEC_NONE)
    buf = ser.write_to()

    reread = Serialized.read_from(buf, pos=0)
    assert isinstance(reread, ArrowDatasetSerialized)

    out = reread.as_python()
    assert isinstance(out, ds.Dataset)
    _assert_tables_equal(_sample_table(), out.to_table())


def test_metadata_contains_arrow_object_marker() -> None:
    ser = ArrowTableSerialized.from_value(_sample_table(), codec=CODEC_NONE)

    assert ser.metadata is not None
    assert ser.metadata[b"arrow_object"] == b"table"
    assert ser.metadata[b"arrow_encoding"] == b"ipc_file"


def test_dataset_metadata_contains_dataset_markers() -> None:
    ser = ArrowDatasetSerialized.from_value(ds.dataset(_sample_table()), codec=CODEC_NONE)

    assert ser.metadata is not None
    assert ser.metadata[b"arrow_object"] == b"dataset"
    assert ser.metadata[b"arrow_source_type"] == b"dataset"


def test_record_batch_payload_validation_raises_on_wrong_batch_count() -> None:
    table = _sample_table()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_file(sink, table.schema) as writer:
        for batch in table.to_batches():
            writer.write_batch(batch)
            writer.write_batch(batch)

    ser = Serialized.build(
        tag=Tags.ARROW_RECORD_BATCH,
        data=sink.getvalue(),
        metadata={
            b"arrow_object": b"record_batch",
            b"arrow_encoding": b"ipc_file",
        },
        codec=CODEC_NONE,
    )
    assert isinstance(ser, ArrowRecordBatchSerialized)

    with pytest.raises(ValueError, match="exactly 1 batch"):
        _ = ser.as_python()


def test_array_payload_validation_raises_on_multiple_chunks() -> None:
    table = pa.Table.from_arrays(
        [
            pa.chunked_array(
                [
                    pa.array([1, 2], type=pa.int64()),
                    pa.array([3, 4], type=pa.int64()),
                ]
            )
        ],
        names=["__ygg_value__"],
    )

    ser = Serialized.build(
        tag=Tags.ARROW_ARRAY,
        data=_table_to_ipc_bytes_for_test(table),
        metadata={
            b"arrow_object": b"array",
            b"arrow_encoding": b"ipc_file",
        },
        codec=CODEC_NONE,
    )
    assert isinstance(ser, ArrowArraySerialized)

    with pytest.raises(ValueError, match="exactly 1 chunk"):
        _ = ser.as_python()


def test_scalar_payload_validation_raises_on_wrong_length() -> None:
    batch = pa.record_batch(
        [pa.array([1, 2], type=pa.int64())],
        names=["__ygg_value__"],
    )

    ser = Serialized.build(
        tag=Tags.ARROW_SCALAR,
        data=_record_batch_to_ipc_bytes_for_test(batch),
        metadata={
            b"arrow_object": b"scalar",
            b"arrow_encoding": b"ipc_file",
        },
        codec=CODEC_NONE,
    )
    assert isinstance(ser, ArrowScalarSerialized)

    with pytest.raises(ValueError, match="exactly 1 value"):
        _ = ser.as_python()


def _table_to_ipc_bytes_for_test(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _record_batch_to_ipc_bytes_for_test(batch: pa.RecordBatch) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_file(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue().to_pybytes()