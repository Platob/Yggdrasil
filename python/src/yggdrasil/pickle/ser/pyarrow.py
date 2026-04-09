from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Generic, Mapping, TypeVar, cast

from yggdrasil.arrow.lib import pyarrow as pa
import pyarrow.dataset as ds

from yggdrasil.pickle.ser.constants import CODEC_NONE
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags

__all__ = [
    "TArrow",
    "ArrowSerialized",
    "ArrowTableSerialized",
    "ArrowRecordBatchSerialized",
    "ArrowStreamSerialized",
    "ArrowDatasetSerialized",
    "ArrowSchemaSerialized",
    "ArrowFieldSerialized",
    "ArrowDataTypeSerialized",
    "ArrowArraySerialized",
    "ArrowChunkedArraySerialized",
    "ArrowScalarSerialized",
    "ArrowTensorSerialized",
    "_merge_metadata",
    "_schema_file_metadata",
    "_table_to_ipc_file_buffer",
    "_table_from_ipc_file_buffer",
]

TArrow = TypeVar("TArrow", bound=object)

_SENTINEL_FIELD_NAME = "__ygg_value__"


def _merge_metadata(
    base: Mapping[bytes, bytes] | None,
    extra: Mapping[bytes, bytes] | None = None,
) -> dict[bytes, bytes] | None:
    if not base and not extra:
        return None
    out: dict[bytes, bytes] = {}
    if base:
        out.update(base)
    if extra:
        out.update(extra)
    return out


def _schema_file_metadata(
    metadata: Mapping[bytes, bytes] | None,
    *,
    arrow_object: bytes,
    arrow_encoding: bytes = b"ipc_file",
) -> dict[bytes, bytes] | None:
    return _merge_metadata(
        metadata,
        {
            b"arrow_object": arrow_object,
            b"arrow_encoding": arrow_encoding,
        },
    )


def _table_to_ipc_file_buffer(
    table: pa.Table,
    *,
    metadata: Mapping[bytes, bytes] | None = None,
) -> pa.Buffer:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_file(sink, table.schema, metadata=metadata) as writer:
        writer.write_table(table)
    return sink.getvalue()


def _table_from_ipc_file_buffer(buf: pa.Buffer) -> pa.Table:
    reader = pa.ipc.open_file(pa.BufferReader(buf))
    return reader.read_all()


def _record_batch_to_ipc_file_buffer(
    batch: pa.RecordBatch,
    *,
    metadata: Mapping[bytes, bytes] | None = None,
) -> pa.Buffer:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_file(sink, batch.schema, metadata=metadata) as writer:
        writer.write_batch(batch)
    return sink.getvalue()


def _reader_to_ipc_stream_buffer(reader: pa.RecordBatchReader) -> pa.Buffer:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, reader.schema) as writer:
        for batch in reader:
            writer.write_batch(batch)
    return sink.getvalue()


def _schema_to_ipc_file_buffer(
    schema: pa.Schema,
    *,
    metadata: Mapping[bytes, bytes] | None = None,
) -> pa.Buffer:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_file(sink, schema, metadata=metadata):
        pass
    return sink.getvalue()


def _schema_from_ipc_file_buffer(buf: pa.Buffer) -> pa.Schema:
    reader = pa.ipc.open_file(pa.BufferReader(buf))
    return reader.schema


def _field_to_schema(field: pa.Field) -> pa.Schema:
    return pa.schema([field])


def _datatype_to_schema(dtype: pa.DataType) -> pa.Schema:
    return pa.schema([pa.field(_SENTINEL_FIELD_NAME, dtype)])


def _array_to_record_batch(array: pa.Array) -> pa.RecordBatch:
    field = pa.field(_SENTINEL_FIELD_NAME, array.type)
    return pa.record_batch([array], schema=pa.schema([field]))


def _chunked_array_to_table(arr: pa.ChunkedArray) -> pa.Table:
    field = pa.field(_SENTINEL_FIELD_NAME, arr.type)
    return pa.Table.from_arrays([arr], schema=pa.schema([field]))


def _scalar_to_record_batch(scalar: pa.Scalar) -> pa.RecordBatch:
    arr = pa.array([scalar], type=scalar.type)
    field = pa.field(_SENTINEL_FIELD_NAME, scalar.type)
    return pa.record_batch([arr], schema=pa.schema([field]))


def _tensor_to_ipc_buffer(tensor: pa.Tensor) -> pa.Buffer:
    sink = pa.BufferOutputStream()
    pa.ipc.write_tensor(tensor, sink)
    return sink.getvalue()


def _tensor_from_ipc_buffer(buf: pa.Buffer) -> pa.Tensor:
    return pa.ipc.read_tensor(pa.BufferReader(buf))


@dataclass(frozen=True, slots=True)
class ArrowSerialized(Serialized[TArrow], Generic[TArrow]):
    TAG: ClassVar[int]

    @property
    def value(self) -> TArrow:
        raise NotImplementedError

    def as_python(self) -> TArrow:
        return self.value

    def decode_arrow_buffer(self) -> pa.Buffer:
        """
        Return payload as a pyarrow.Buffer suitable for Arrow IPC reads.

        This is only truly zero-copy when:
        - self.codec == CODEC_NONE
        - the underlying Serialized payload can be exposed without materializing
          a new Python bytes object.

        With the current base Serialized API, we can only guarantee the IPC read
        side is Arrow-native; true end-to-end zero-copy may still require a base
        layer hook that exposes the raw payload buffer directly.
        """
        if self.codec == CODEC_NONE:
            return pa.py_buffer(self.to_bytes())

        return pa.py_buffer(self.decode())

    @classmethod
    def from_python_object(
        cls,
        obj: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object] | None:
        if isinstance(obj, pa.Table):
            return ArrowTableSerialized.from_value(obj, metadata=metadata, codec=codec)

        if isinstance(obj, pa.RecordBatch):
            return ArrowRecordBatchSerialized.from_value(
                obj,
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, pa.RecordBatchReader):
            return ArrowStreamSerialized.from_value(
                obj,
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, ds.Dataset):
            return ArrowDatasetSerialized.from_value(
                obj,
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, pa.Schema):
            return ArrowSchemaSerialized.from_value(
                obj,
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, pa.Field):
            return ArrowFieldSerialized.from_value(
                obj,
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, pa.DataType):
            return ArrowDataTypeSerialized.from_value(
                obj,
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, pa.Array):
            return ArrowArraySerialized.from_value(
                obj,
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, pa.ChunkedArray):
            return ArrowChunkedArraySerialized.from_value(
                obj,
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, pa.Scalar):
            return ArrowScalarSerialized.from_value(
                obj,
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, pa.Tensor):
            return ArrowTensorSerialized.from_value(
                obj,
                metadata=metadata,
                codec=codec,
            )

        return None


@dataclass(frozen=True, slots=True)
class ArrowTableSerialized(ArrowSerialized[pa.Table]):
    TAG: ClassVar[int] = Tags.ARROW_TABLE

    @property
    def value(self) -> pa.Table:
        return _table_from_ipc_file_buffer(self.decode_arrow_buffer())

    @classmethod
    def from_value(
        cls,
        table: pa.Table,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        merged = _schema_file_metadata(metadata, arrow_object=b"table")
        buf = _table_to_ipc_file_buffer(table, metadata=merged)
        return cls.build(
            tag=cls.TAG,
            data=buf,
            metadata=merged,
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class ArrowRecordBatchSerialized(ArrowSerialized[pa.RecordBatch]):
    TAG: ClassVar[int] = Tags.ARROW_RECORD_BATCH

    @property
    def value(self) -> pa.RecordBatch:
        reader = pa.ipc.open_file(pa.BufferReader(self.decode_arrow_buffer()))
        num_batches = reader.num_record_batches
        if num_batches != 1:
            raise ValueError(
                f"ARROW_RECORD_BATCH payload must contain exactly 1 batch, got {num_batches}"
            )
        return reader.get_batch(0)

    @classmethod
    def from_value(
        cls,
        batch: pa.RecordBatch,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        merged = _schema_file_metadata(metadata, arrow_object=b"record_batch")
        buf = _record_batch_to_ipc_file_buffer(batch, metadata=merged)
        return cls.build(
            tag=cls.TAG,
            data=buf,
            metadata=merged,
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class ArrowStreamSerialized(ArrowSerialized[pa.RecordBatchReader]):
    TAG: ClassVar[int] = Tags.ARROW_STREAM

    @property
    def value(self) -> pa.RecordBatchReader:
        return cast(
            pa.RecordBatchReader,
            pa.ipc.open_stream(pa.BufferReader(self.decode_arrow_buffer())),
        )

    @classmethod
    def from_value(
        cls,
        reader: pa.RecordBatchReader,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        merged = _merge_metadata(
            metadata,
            {
                b"arrow_object": b"record_batch_reader",
                b"arrow_encoding": b"ipc_stream",
            },
        )
        buf = _reader_to_ipc_stream_buffer(reader)
        return cls.build(
            tag=cls.TAG,
            data=buf,
            metadata=merged,
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class ArrowDatasetSerialized(ArrowSerialized[ds.Dataset]):
    TAG: ClassVar[int] = Tags.ARROW_DATASET

    @property
    def value(self) -> ds.Dataset:
        """
        Reconstruct as an in-memory dataset.

        We intentionally rebuild from an in-memory Arrow-native object rather than
        pretending filesystem fragments still exist.
        """
        encoding = (self.metadata or {}).get(b"arrow_encoding", b"ipc_file")

        if encoding == b"ipc_stream":
            reader = pa.ipc.open_stream(pa.BufferReader(self.decode_arrow_buffer()))
            return ds.dataset(reader)

        if encoding == b"ipc_file":
            table = _table_from_ipc_file_buffer(self.decode_arrow_buffer())
            return ds.dataset(table)

        raise ValueError(
            f"Unsupported ARROW_DATASET encoding: {encoding!r}; expected b'ipc_file' or b'ipc_stream'"
        )

    @classmethod
    def from_value(
        cls,
        dataset: ds.Dataset,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
        use_stream: bool = False,
    ) -> Serialized[object]:
        """
        Persist a dataset as self-contained Arrow IPC and restore it as an
        in-memory dataset on load.

        - use_stream=False: materialize to Table -> IPC file -> ds.dataset(table)
        - use_stream=True:  materialize to RecordBatchReader -> IPC stream -> ds.dataset(reader)

        The stream form preserves streaming semantics better, but note that a
        dataset built from a RecordBatchReader is single-use.
        """
        if use_stream:
            reader = dataset.to_table().to_reader()
            merged = _merge_metadata(
                metadata,
                {
                    b"arrow_object": b"dataset",
                    b"arrow_source_type": b"dataset",
                    b"arrow_encoding": b"ipc_stream",
                },
            )
            buf = _reader_to_ipc_stream_buffer(reader)
        else:
            table = dataset.to_table()
            merged = _merge_metadata(
                metadata,
                {
                    b"arrow_object": b"dataset",
                    b"arrow_source_type": b"dataset",
                    b"arrow_encoding": b"ipc_file",
                },
            )
            buf = _table_to_ipc_file_buffer(table, metadata=merged)

        return cls.build(
            tag=cls.TAG,
            data=buf,
            metadata=merged,
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class ArrowSchemaSerialized(ArrowSerialized[pa.Schema]):
    TAG: ClassVar[int] = Tags.ARROW_SCHEMA

    @property
    def value(self) -> pa.Schema:
        return _schema_from_ipc_file_buffer(self.decode_arrow_buffer())

    @classmethod
    def from_value(
        cls,
        schema: pa.Schema,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        merged = _schema_file_metadata(metadata, arrow_object=b"schema")
        buf = _schema_to_ipc_file_buffer(schema, metadata=merged)
        return cls.build(
            tag=cls.TAG,
            data=buf,
            metadata=merged,
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class ArrowFieldSerialized(ArrowSerialized[pa.Field]):
    TAG: ClassVar[int] = Tags.ARROW_FIELD

    @property
    def value(self) -> pa.Field:
        schema = _schema_from_ipc_file_buffer(self.decode_arrow_buffer())
        if len(schema) != 1:
            raise ValueError(
                f"ARROW_FIELD payload must contain exactly 1 field, got {len(schema)}"
            )
        return schema.field(0)

    @classmethod
    def from_value(
        cls,
        field: pa.Field,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        merged = _schema_file_metadata(metadata, arrow_object=b"field")
        buf = _schema_to_ipc_file_buffer(_field_to_schema(field), metadata=merged)
        return cls.build(
            tag=cls.TAG,
            data=buf,
            metadata=merged,
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class ArrowDataTypeSerialized(ArrowSerialized[pa.DataType]):
    TAG: ClassVar[int] = Tags.ARROW_DATA_TYPE

    @property
    def value(self) -> pa.DataType:
        schema = _schema_from_ipc_file_buffer(self.decode_arrow_buffer())
        if len(schema) != 1:
            raise ValueError(
                f"ARROW_DATA_TYPE payload must contain exactly 1 field, got {len(schema)}"
            )
        return schema.field(0).type

    @classmethod
    def from_value(
        cls,
        dtype: pa.DataType,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        merged = _schema_file_metadata(metadata, arrow_object=b"data_type")
        buf = _schema_to_ipc_file_buffer(_datatype_to_schema(dtype), metadata=merged)
        return cls.build(
            tag=cls.TAG,
            data=buf,
            metadata=merged,
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class ArrowArraySerialized(ArrowSerialized[pa.Array]):
    TAG: ClassVar[int] = Tags.ARROW_ARRAY

    @property
    def value(self) -> pa.Array:
        table = _table_from_ipc_file_buffer(self.decode_arrow_buffer())
        if table.num_columns != 1:
            raise ValueError(
                f"ARROW_ARRAY payload must contain exactly 1 column, got {table.num_columns}"
            )
        col = table.column(0)
        if col.num_chunks != 1:
            raise ValueError(
                f"ARROW_ARRAY payload must contain exactly 1 chunk, got {col.num_chunks}"
            )
        return col.chunk(0)

    @classmethod
    def from_value(
        cls,
        array: pa.Array,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        batch = _array_to_record_batch(array)
        merged = _schema_file_metadata(metadata, arrow_object=b"array")
        buf = _record_batch_to_ipc_file_buffer(batch, metadata=merged)
        return cls.build(
            tag=cls.TAG,
            data=buf,
            metadata=merged,
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class ArrowChunkedArraySerialized(ArrowSerialized[pa.ChunkedArray]):
    TAG: ClassVar[int] = Tags.ARROW_CHUNKED_ARRAY

    @property
    def value(self) -> pa.ChunkedArray:
        table = _table_from_ipc_file_buffer(self.decode_arrow_buffer())
        if table.num_columns != 1:
            raise ValueError(
                f"ARROW_CHUNKED_ARRAY payload must contain exactly 1 column, got {table.num_columns}"
            )
        return table.column(0)

    @classmethod
    def from_value(
        cls,
        arr: pa.ChunkedArray,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        table = _chunked_array_to_table(arr)
        merged = _schema_file_metadata(metadata, arrow_object=b"chunked_array")
        buf = _table_to_ipc_file_buffer(table, metadata=merged)
        return cls.build(
            tag=cls.TAG,
            data=buf,
            metadata=merged,
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class ArrowScalarSerialized(ArrowSerialized[pa.Scalar]):
    TAG: ClassVar[int] = Tags.ARROW_SCALAR

    @property
    def value(self) -> pa.Scalar:
        table = _table_from_ipc_file_buffer(self.decode_arrow_buffer())
        if table.num_columns != 1:
            raise ValueError(
                f"ARROW_SCALAR payload must contain exactly 1 column, got {table.num_columns}"
            )
        col = table.column(0)
        if col.num_chunks != 1:
            raise ValueError(
                f"ARROW_SCALAR payload must contain exactly 1 chunk, got {col.num_chunks}"
            )
        arr = col.chunk(0)
        if len(arr) != 1:
            raise ValueError(
                f"ARROW_SCALAR payload must contain exactly 1 value, got {len(arr)}"
            )
        return arr[0]

    @classmethod
    def from_value(
        cls,
        scalar: pa.Scalar,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        batch = _scalar_to_record_batch(scalar)
        merged = _schema_file_metadata(metadata, arrow_object=b"scalar")
        buf = _record_batch_to_ipc_file_buffer(batch, metadata=merged)
        return cls.build(
            tag=cls.TAG,
            data=buf,
            metadata=merged,
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class ArrowTensorSerialized(ArrowSerialized[pa.Tensor]):
    TAG: ClassVar[int] = Tags.ARROW_TENSOR

    @property
    def value(self) -> pa.Tensor:
        return _tensor_from_ipc_buffer(self.decode_arrow_buffer())

    @classmethod
    def from_value(
        cls,
        tensor: pa.Tensor,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        merged = _merge_metadata(
            metadata,
            {
                b"arrow_object": b"tensor",
                b"arrow_encoding": b"ipc_tensor",
            },
        )
        buf = _tensor_to_ipc_buffer(tensor)
        return cls.build(
            tag=cls.TAG,
            data=buf,
            metadata=merged,
            codec=codec,
        )


for cls in ArrowSerialized.__subclasses__():
    Tags.register_class(cls, tag=cls.TAG)

ArrowTableSerialized = Tags.get_class(Tags.ARROW_TABLE) or ArrowTableSerialized
ArrowRecordBatchSerialized = Tags.get_class(Tags.ARROW_RECORD_BATCH) or ArrowRecordBatchSerialized
ArrowStreamSerialized = Tags.get_class(Tags.ARROW_STREAM) or ArrowStreamSerialized
ArrowDatasetSerialized = Tags.get_class(Tags.ARROW_DATASET) or ArrowDatasetSerialized
ArrowSchemaSerialized = Tags.get_class(Tags.ARROW_SCHEMA) or ArrowSchemaSerialized
ArrowFieldSerialized = Tags.get_class(Tags.ARROW_FIELD) or ArrowFieldSerialized
ArrowDataTypeSerialized = Tags.get_class(Tags.ARROW_DATA_TYPE) or ArrowDataTypeSerialized
ArrowArraySerialized = Tags.get_class(Tags.ARROW_ARRAY) or ArrowArraySerialized
ArrowChunkedArraySerialized = Tags.get_class(Tags.ARROW_CHUNKED_ARRAY) or ArrowChunkedArraySerialized
ArrowScalarSerialized = Tags.get_class(Tags.ARROW_SCALAR) or ArrowScalarSerialized
ArrowTensorSerialized = Tags.get_class(Tags.ARROW_TENSOR) or ArrowTensorSerialized
