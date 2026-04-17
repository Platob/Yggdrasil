import dataclasses
import enum
import logging
from dataclasses import is_dataclass
from typing import Optional, Union, Any

import pyarrow as pa
from yggdrasil.data.cast.options import CastOptions
from yggdrasil.data.cast.registry import register_converter
from yggdrasil.dataclasses.dataclass import dataclass_to_arrow_field
from yggdrasil.pickle.serde import ObjectSerde

from .python_arrow import (
    is_arrow_type_list_like,
    is_arrow_type_string_like,
    is_arrow_type_binary_like,
    merge_arrow_fields,
    merge_arrow_types,
)
from .python_defaults import default_arrow_scalar

__all__ = [
    "ArrowDataType",
    "cast_arrow_array",
    "cast_arrow_tabular",
    "cast_arrow_record_batch_reader",
    "to_spark_arrow_type",
    "to_polars_arrow_type",
    "arrow_type_to_field",
    "is_arrow_type_binary_like",
    "is_arrow_type_string_like",
    "is_arrow_type_list_like",
    "record_batch_to_table",
    "any_to_arrow_scalar",
    "any_to_arrow_field",
    "any_to_arrow_table",
    "any_to_arrow_record_batch",
    "any_to_arrow_schema",
    "arrow_field_to_dict",
    "arrow_type_to_dict",
    "dict_to_arrow_type",
    "dict_to_arrow_field",
    "default_arrow_scalar",
    "merge_arrow_fields",
    "merge_arrow_types",
]

logger = logging.getLogger(__name__)

#: Union alias covering all Arrow DataType variants used throughout this module.
ArrowDataType = Union[
    pa.DataType,
    pa.Decimal128Type,
    pa.TimestampType,
    pa.DictionaryType,
    pa.MapType,
    pa.StructType,
    pa.FixedSizeListType,
]


# ---------------------------------------------------------------------------
# Any-to-Arrow table / record batch
# ---------------------------------------------------------------------------

@register_converter(Any, pa.Table)
def any_to_arrow_table(
    obj: Any,
    options: Optional["CastOptions"] = None,
) -> pa.Table:
    """Convert *any* supported object to a ``pa.Table``, then cast to the target schema.

    Supported input types (detected via namespace inspection):

    * ``pa.Table`` — passed through to :func:`cast_arrow_tabular`.
    * ``pa.RecordBatch`` — wrapped in a single-batch table first.
    * ``pandas.DataFrame`` — via
      :func:`~.pandas.cast.pandas_dataframe_to_arrow_table`.
    * ``pyspark.sql.DataFrame`` — converted via Spark's ``toArrow()`` method.
    * Everything else (Polars, dicts, dataclasses, …) — routed through
      :func:`~.polars.cast.any_to_polars_dataframe` then
      :func:`~.polars.cast.polars_dataframe_to_arrow_table`.

    Args:
        obj: Any object that can be converted to an Arrow table.
        options: Cast options including the target schema.

    Returns:
        A ``pa.Table`` cast to ``options.target_arrow_schema`` (if set).
    """
    if not isinstance(obj, pa.Table):
        if isinstance(obj, pa.RecordBatch):
            obj = pa.Table.from_batches([obj])  # type: ignore
        else:
            namespace = ObjectSerde.full_namespace(obj)

            if namespace.startswith("pandas."):
                from yggdrasil.pandas.cast import pandas_dataframe_to_arrow_table
                obj = pandas_dataframe_to_arrow_table(obj, options)

            if namespace.startswith("pyspark."):
                import pyspark.sql as pyspark_sql
                from yggdrasil.spark.cast import any_to_spark_dataframe

                obj: pyspark_sql.DataFrame = any_to_spark_dataframe(obj, options)
                obj = obj.toArrow()
            else:
                from yggdrasil.polars.cast import any_to_polars_dataframe, polars_dataframe_to_arrow_table
                obj = any_to_polars_dataframe(obj, options)
                obj = polars_dataframe_to_arrow_table(obj, options)

    return cast_arrow_tabular(obj, options)


@register_converter(Any, pa.RecordBatch)
def any_to_arrow_record_batch(
    obj: Any,
    options: Optional[CastOptions] = None,
) -> pa.RecordBatch:
    """Convert *any* supported object to a single ``pa.RecordBatch``.

    Delegates to :func:`any_to_arrow_table` and then extracts the first batch.

    Args:
        obj: Any object convertible to an Arrow table.
        options: Cast options.

    Returns:
        The first ``pa.RecordBatch`` from the resulting table.
    """
    if not isinstance(obj, pa.RecordBatch):
        obj: pa.Table = any_to_arrow_table(obj, options)
        return obj.to_batches()[0]

    return cast_arrow_tabular(obj, options)


# ---------------------------------------------------------------------------
# Scalar casting
# ---------------------------------------------------------------------------

@register_converter(Any, pa.Scalar)
def any_to_arrow_scalar(
    scalar: Any,
    options: Optional[CastOptions] = None,
) -> pa.Scalar:
    """Convert a Python value to an Arrow scalar, then cast to the target field type.

    Conversion rules:

    * ``None`` → type-appropriate default scalar (respects nullability).
    * ``enum.Enum`` → the enum's ``.value`` attribute is used.
    * dataclass → converted to a ``dict`` via ``dataclasses.asdict``, then to a
      struct scalar.  If no target field is set, the field is inferred from the
      dataclass definition via :func:`~.dataclasses.dataclass.dataclass_to_arrow_field`.
    * All other values → ``pa.scalar(value, type=target_field.type)`` with a
      fallback to untyped ``pa.scalar(value)`` on ``ArrowInvalid``.

    Args:
        scalar: Input Python value.
        options: Cast options.  ``options.target_field`` determines the output type.

    Returns:
        An Arrow scalar cast to the target field type.
    """
    if not isinstance(scalar, pa.Scalar):
        options = CastOptions.check(options)
        target_field = options.target_field

        if scalar is None:
            return default_arrow_scalar(
                target_field,
                nullable=True if target_field is None else target_field.nullable,
            )

        if isinstance(scalar, enum.Enum):
            scalar = scalar.value

        if is_dataclass(scalar):
            if not target_field:
                target_field = dataclass_to_arrow_field(scalar)
                options = options.copy(target_field=target_field)
            scalar = dataclasses.asdict(scalar)

        if target_field is None:
            if is_dataclass(scalar):
                scalar = pa.scalar(
                    dataclasses.asdict(scalar),
                    type=dataclass_to_arrow_field(scalar).type,
                )
            else:
                scalar = pa.scalar(scalar)
            return scalar

        try:
            scalar = pa.scalar(scalar, type=target_field.type)
        except pa.ArrowInvalid:
            # Fall back to untyped scalar; cast_arrow_scalar will handle the type.
            scalar = pa.scalar(scalar)

    return cast_arrow_scalar(scalar, options)


@register_converter(pa.Scalar, pa.Scalar)
def cast_arrow_scalar(
    scalar: pa.Scalar,
    options: Optional[CastOptions] = None,
) -> pa.Scalar:
    """Cast an existing Arrow scalar to the type described in *options*.

    Wraps the scalar in a single-element array, delegates to
    :func:`cast_arrow_array`, and returns the first element.  This ensures
    all casting logic is centralised in the array path.

    Args:
        scalar: Arrow scalar to cast.
        options: Cast options.  If ``options.target_field`` is ``None`` the
            scalar is returned unchanged.

    Returns:
        Cast Arrow scalar.
    """
    options = CastOptions.check(options)
    target_field = options.target_field

    if target_field is None:
        return scalar

    arr = pa.array([scalar])
    casted = cast_arrow_array(arr, options)
    return casted[0]


# ---------------------------------------------------------------------------
# Main array cast dispatcher
# ---------------------------------------------------------------------------

@register_converter(pa.Array, pa.Array)
@register_converter(pa.ChunkedArray, pa.ChunkedArray)
def cast_arrow_array(
    array: Union[pa.ChunkedArray, pa.Array],
    options: Optional[CastOptions] = None,
) -> Union[pa.ChunkedArray, pa.Array]:
    options = CastOptions.check(options)
    return options.cast_arrow_array(array)


# ---------------------------------------------------------------------------
# Table / RecordBatch casting
# ---------------------------------------------------------------------------

@register_converter(pa.Table, pa.Table)
@register_converter(pa.RecordBatch, pa.RecordBatch)
def cast_arrow_tabular(
    data: Union[pa.Table, pa.RecordBatch],
    options: Optional[CastOptions] = None,
) -> Union[pa.Table, pa.RecordBatch]:
    options = CastOptions.check(options)
    return options.cast_arrow_tabular(data)


@register_converter(pa.RecordBatchReader, pa.RecordBatchReader)
def cast_arrow_record_batch_reader(
    data: pa.RecordBatchReader,
    options: Optional[CastOptions] = None,
) -> pa.RecordBatchReader:
    """Lazily wrap a ``RecordBatchReader`` so each emitted batch is cast on-the-fly.

    No data is read or materialised until the returned reader is iterated.
    This is the preferred entry point for streaming workloads (e.g. reading
    Parquet files in Arrow IPC streams) where full materialisation is
    undesirable.

    Args:
        data: Source reader.
        options: Cast options including the target schema.  If
            ``options.target_arrow_schema`` is ``None`` the original reader is
            returned as-is.

    Returns:
        A ``pa.RecordBatchReader`` whose schema matches the target and whose
        batches are lazily cast.
    """
    options = CastOptions.check(options)

    if options.target_field is None:
        return data

    def casted_batches(opt=options):
        """Lazily yield cast batches from the upstream reader."""
        for batch in data:
            yield cast_arrow_tabular(batch, opt)

    return pa.RecordBatchReader.from_batches(arrow_schema, casted_batches())  # type: ignore


# ---------------------------------------------------------------------------
# Type normalisation helpers
# ---------------------------------------------------------------------------

def to_spark_arrow_type(dtype: ArrowDataType) -> ArrowDataType:
    """Normalise an Arrow ``DataType`` to a Spark-compatible equivalent.

    Spark does not support several Arrow type variants.  This function
    recursively replaces them:

    * ``large_string`` / ``large_binary`` → ``string`` / ``binary``
    * ``large_list<T>`` → ``list<T>`` (value type also normalised)
    * ``dictionary<index, value>`` → ``value`` type (Spark resolves categories)
    * ``ExtensionType`` → ``storage_type`` (unwrap custom extensions)
    * ``struct`` → ``struct`` with each child field normalised
    * ``map`` → ``map`` with key and item types normalised

    Args:
        dtype: Arrow data type to normalise.

    Returns:
        A Spark-compatible Arrow data type.
    """
    if is_arrow_type_string_like(dtype):
        return pa.string()
    if is_arrow_type_binary_like(dtype):
        return pa.binary()
    if is_arrow_type_list_like(dtype):
        return pa.list_(to_spark_arrow_type(dtype.value_type))
    if pa.types.is_dictionary(dtype):
        return to_spark_arrow_type(dtype.value_type)
    if isinstance(dtype, pa.ExtensionType):
        return to_spark_arrow_type(dtype.storage_type)
    if pa.types.is_struct(dtype):
        new_fields = [
            pa.field(
                f.name,
                to_spark_arrow_type(f.type),
                nullable=f.nullable,
                metadata=f.metadata,
            )
            for f in dtype
        ]
        return pa.struct(new_fields)
    if pa.types.is_map(dtype):
        key_field = dtype.key_field
        item_field = dtype.item_field
        new_key = pa.field(
            key_field.name,
            to_spark_arrow_type(key_field.type),
            nullable=key_field.nullable,
            metadata=key_field.metadata,
        )
        new_item = pa.field(
            item_field.name,
            to_spark_arrow_type(item_field.type),
            nullable=item_field.nullable,
            metadata=item_field.metadata,
        )
        return pa.map_(new_key, new_item)

    return dtype


def to_polars_arrow_type(dtype: ArrowDataType) -> ArrowDataType:
    """Normalise an Arrow ``DataType`` to a Polars-compatible equivalent.

    Extends :func:`to_spark_arrow_type` with one additional rule required by
    Polars: ``map<k,v>`` is not a first-class type in Polars' data model and
    must be represented as ``list<struct<key: K, value: V>>``.

    Transformation summary:

    * All Spark normalisation rules apply first.
    * ``map<K, V>`` → ``list<struct<key: K, value: V>>``
    * ``struct`` children are recursively normalised.
    * ``list`` element types are recursively normalised.

    Args:
        dtype: Arrow data type to normalise.

    Returns:
        A Polars-compatible Arrow data type.
    """
    # Apply Spark normalisation first (handles large_*, dictionary, extensions).
    dtype = to_spark_arrow_type(dtype)

    if pa.types.is_map(dtype):
        key_field = dtype.key_field
        item_field = dtype.item_field

        key_type = to_polars_arrow_type(key_field.type)
        value_type = to_polars_arrow_type(item_field.type)

        # Represent as list<struct<key, value>> — Polars' canonical map form.
        struct_type = pa.field(
            "entries",
            pa.struct(
                [
                    pa.field(
                        key_field.name,
                        key_type,
                        nullable=key_field.nullable,
                        metadata=key_field.metadata,
                    ),
                    pa.field(
                        item_field.name,
                        value_type,
                        nullable=item_field.nullable,
                        metadata=item_field.metadata,
                    ),
                ]
            ),
            nullable=True,
        )
        return pa.list_(struct_type)

    if pa.types.is_struct(dtype):
        new_fields = [
            pa.field(
                f.name,
                to_polars_arrow_type(f.type),
                nullable=f.nullable,
                metadata=f.metadata,
            )
            for f in dtype
        ]
        return pa.struct(new_fields)

    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return pa.list_(to_polars_arrow_type(dtype.value_type))

    return dtype


# ---------------------------------------------------------------------------
# Cross-container casting helpers
# ---------------------------------------------------------------------------

@register_converter(pa.Table, pa.RecordBatch)
def table_to_record_batch(
    data: pa.Table,
    options: Optional[CastOptions] = None,
) -> pa.RecordBatch:
    """Cast a ``Table`` and return a single ``RecordBatch``.

    All ``ChunkedArray`` columns are merged via
    :meth:`~pyarrow.ChunkedArray.combine_chunks` before constructing the batch.
    For empty tables an empty batch with the correct schema is returned.

    Args:
        data: Source table.
        options: Cast options.

    Returns:
        A single ``pa.RecordBatch``.
    """
    casted: pa.Table = cast_arrow_tabular(data, options)

    if casted.num_rows == 0:
        arrays = [pa.array([], type=f.type) for f in casted.schema]
        return pa.RecordBatch.from_arrays(arrays, schema=casted.schema)  # type: ignore

    arrays = [chunked_array.combine_chunks() for chunked_array in casted.columns]
    return pa.RecordBatch.from_arrays(arrays, schema=casted.schema)  # type: ignore


@register_converter(pa.RecordBatch, pa.Table)
def record_batch_to_table(
    data: pa.RecordBatch,
    options: Optional[CastOptions] = None,
) -> pa.Table:
    """Cast a ``RecordBatch`` and wrap the result as a single-batch ``Table``.

    Args:
        data: Source record batch.
        options: Cast options.

    Returns:
        A ``pa.Table`` containing one batch.
    """
    casted = cast_arrow_tabular(data, options)
    return pa.Table.from_batches(batches=[casted], schema=casted.schema)  # type: ignore


@register_converter(pa.Table, pa.RecordBatchReader)
def table_to_record_batch_reader(
    data: pa.Table,
    options: Optional[CastOptions] = None,
) -> pa.RecordBatchReader:
    """Cast a ``Table`` and expose the result as a ``RecordBatchReader``.

    Args:
        data: Source table.
        options: Cast options.

    Returns:
        A ``pa.RecordBatchReader`` iterating over the cast table's batches.
    """
    casted = cast_arrow_tabular(data, options)
    return pa.RecordBatchReader.from_batches(casted.schema, casted.to_batches())  # type: ignore


@register_converter(pa.RecordBatchReader, pa.Table)
def record_batch_reader_to_table(
    data: pa.RecordBatchReader,
    options: Optional[CastOptions] = None,
) -> pa.Table:
    """Cast each batch in a ``RecordBatchReader`` and collect into a ``Table``.

    Args:
        data: Source reader.  All batches are materialised.
        options: Cast options.

    Returns:
        A ``pa.Table`` containing all cast batches.
    """
    casted_reader: pa.RecordBatchReader = cast_arrow_record_batch_reader(data, options)
    return pa.Table.from_batches(batches=list(casted_reader), schema=casted_reader.schema)  # type: ignore


@register_converter(pa.RecordBatch, pa.RecordBatchReader)
def record_batch_to_record_batch_reader(
    data: pa.RecordBatch,
    options: Optional[CastOptions] = None,
) -> pa.RecordBatchReader:
    """Cast a ``RecordBatch`` and wrap it in a single-batch ``RecordBatchReader``.

    Args:
        data: Source record batch.
        options: Cast options.

    Returns:
        A ``pa.RecordBatchReader`` with one batch.
    """
    casted = cast_arrow_tabular(data, options)
    return pa.RecordBatchReader.from_batches(schema=casted.schema, batches=[casted])  # type: ignore


@register_converter(pa.RecordBatchReader, pa.RecordBatch)
def record_batch_reader_to_record_batch(
    data: pa.RecordBatchReader,
    options: Optional[CastOptions] = None,
) -> pa.RecordBatch:
    """Materialise a ``RecordBatchReader`` into a single ``RecordBatch``.

    .. warning::
        All batches are read into memory before merging.  Use
        :func:`cast_arrow_record_batch_reader` for streaming workloads.

    Args:
        data: Source reader.
        options: Cast options.

    Returns:
        A single merged ``pa.RecordBatch``.
    """
    table = record_batch_reader_to_table(data, options)
    return table_to_record_batch(table, options)


# ---------------------------------------------------------------------------
# Field / Schema converters
# ---------------------------------------------------------------------------

@register_converter(pa.Array, pa.Field)
@register_converter(pa.ChunkedArray, pa.Field)
def arrow_array_to_field(
    array: Union[pa.Array, pa.ChunkedArray],
    options: Optional[CastOptions] = None,
) -> pa.Field:
    """Derive a ``pa.Field`` descriptor from an Arrow array.

    The field name is taken from ``options.source_field.to_arrow_field().name`` (or
    ``"root"`` as fallback).  Nullability is set to ``True`` if the array type
    is ``null`` or the array contains any nulls.

    Args:
        array: Array to introspect.
        options: Cast options.

    Returns:
        A ``pa.Field`` describing the array's type and nullability.
    """
    options = CastOptions.check(options=options)
    name = options.source_field.to_arrow_field().name if options.source_field.to_arrow_field() else "root"
    metadata = options.source_field.to_arrow_field().metadata if options.source_field.to_arrow_field() else None

    arrow_field = pa.field(
        name,
        array.type,
        nullable=array.type == pa.null() or array.null_count > 0,
        metadata=metadata,
    )
    return arrow_field


@register_converter(pa.DataType, pa.Field)
def arrow_type_to_field(
    arrow_type: ArrowDataType,
    options: Optional[CastOptions] = None,
) -> pa.Field:
    """Wrap an Arrow ``DataType`` into a ``pa.Field``.

    Args:
        arrow_type: Arrow type to wrap.
        options: Cast options.  Used to derive the field name and nullability.

    Returns:
        A ``pa.Field`` with the given type.
    """
    options = CastOptions.check(options=options)
    name = options.source_field.to_arrow_field().name if options.source_field.to_arrow_field() else "root"
    nullable = (
        options.source_field.to_arrow_field().nullable if options.source_field.to_arrow_field() else True
    )
    metadata = (
        options.source_field.to_arrow_field().metadata if options.source_field.to_arrow_field() else None
    )

    arrow_field = pa.field(name, arrow_type, nullable=nullable, metadata=metadata)
    return arrow_field


@register_converter(Any, pa.Field)
def any_to_arrow_field(
    obj: Any,
    options: Optional[CastOptions] = None,
) -> pa.Field:
    if not isinstance(obj, pa.Field):
        from yggdrasil.data.data_field import Field
        return Field.from_any(obj).to_arrow_field()
    return obj


@register_converter(Any, pa.Schema)
def any_to_arrow_schema(
    obj: Any,
    options: Optional[CastOptions] = None,
) -> pa.Schema:
    """Derive a ``pa.Schema`` from *any* supported object.

    Converts *obj* to an Arrow field via :func:`any_to_arrow_field`, then
    expands the field to a schema via :func:`arrow_field_to_schema`.

    For objects that are already ``pa.Schema``, delegates to
    :func:`arrow_schema_to_schema` (which can optionally merge with a target).

    Args:
        obj: Object to introspect.
        options: Cast options.

    Returns:
        Arrow schema description of the object.
    """
    if not isinstance(obj, pa.Schema):
        from yggdrasil.data.schema import Schema
        return Schema.from_any(obj).to_arrow_schema()
    return obj


# ---------------------------------------------------------------------------
# Dict serialization helpers
# ---------------------------------------------------------------------------


@register_converter(pa.Field, dict)
def arrow_field_to_dict(field: pa.Field, options=None) -> dict[str, Any]:
    """Convert a pyarrow.Field to a JSON-serializable dict."""
    return {
        "name": field.name,
        "type": arrow_type_to_dict(field.type),
        "nullable": field.nullable,
        "metadata": (
            {k.decode(): v.decode() for k, v in field.metadata.items()}
            if field.metadata
            else None
        ),
    }

@register_converter(dict, pa.Field)
def dict_to_arrow_field(d: dict[str, Any], options=None) -> pa.Field:
    """Reconstruct a pyarrow.Field from a dict produced by field_to_dict."""
    metadata = d.get("metadata")
    if metadata:
        metadata = {k.encode(): v.encode() for k, v in metadata.items()}

    return pa.field(
        name=d["name"],
        type=dict_to_arrow_type(d["type"]),
        nullable=d.get("nullable", True),
        metadata=metadata,
    )


# ── Type serialization ────────────────────────────────────────────────────────
@register_converter(pa.DataType, dict)
def arrow_type_to_dict(t: ArrowDataType, options=None) -> dict[str, Any]:
    """Recursively serialize a pyarrow DataType."""
    from yggdrasil.data.types.base import DataType

    return DataType.from_arrow_type(t).to_dict()


@register_converter(dict, pa.DataType)
def dict_to_arrow_type(d: dict[str, Any], options=None) -> ArrowDataType:
    """Recursively deserialize a pyarrow DataType from a dict."""
    from yggdrasil.data.types.base import DataType

    return DataType.from_dict(d).to_arrow()
