from typing import Optional, Tuple

import pyarrow as pa

from .arrow_cast import (
    cast_arrow_array,
    cast_arrow_tabular,
    cast_arrow_record_batch_reader,
)
from .cast_options import CastOptions
from .registry import register_converter
from ..python_defaults import default_arrow_scalar
from ...libs.polarslib import (
    polars, require_polars,
    arrow_type_to_polars_type, polars_type_to_arrow_type,
    arrow_field_to_polars_field, polars_field_to_arrow_field
)

__all__ = [
    "cast_polars_array",
    "cast_polars_dataframe",
    "arrow_type_to_polars_type",
    "polars_type_to_arrow_type",
    "arrow_field_to_polars_field",
    "polars_field_to_arrow_field",
    "polars_series_to_arrow_array",
    "polars_strptime",
    "polars_dataframe_to_arrow_table",
    "arrow_array_to_polars_series",
    "arrow_table_to_polars_dataframe",
    "polars_dataframe_to_record_batch_reader",
    "record_batch_reader_to_polars_dataframe",
]

# ---------------------------------------------------------------------------
# Polars type aliases + decorator wrapper (safe when Polars is missing)
# ---------------------------------------------------------------------------

if polars is not None:
    require_polars()

    PolarsSeries = polars.Series
    PolarsDataFrame = polars.DataFrame

    def polars_converter(*args, **kwargs):
        return register_converter(*args, **kwargs)
else:
    # Dummy types so annotations/decorators don't explode without Polars
    class _PolarsDummy:  # pragma: no cover - only used when Polars not installed
        pass

    PolarsSeries = _PolarsDummy
    PolarsDataFrame = _PolarsDummy

    def polars_converter(*_args, **_kwargs):  # pragma: no cover - no-op decorator
        def _decorator(func):
            return func

        return _decorator


# ---------------------------------------------------------------------------
# Core casting: Polars <-> Arrow types
# ---------------------------------------------------------------------------
@polars_converter(PolarsSeries, PolarsSeries)
def cast_polars_array(
    series: "polars.Series",
    options: Optional[CastOptions] = None,
) -> "polars.Series":
    """
    Cast a Polars Series to a target Arrow type using Polars casting rules.

    `options` is normalized via ArrowCastOptions.check_arg and its
    `target_field` is used as the Arrow target.

    - target_field can be a pa.DataType, pa.Field, or pa.Schema (schema → first field).
    - If a Field is provided, we also respect its nullability by filling nulls
      when nullable=False (using default_from_arrow_hint).
    """
    options = CastOptions.check_arg(options)
    target_field = options.target_field

    if not target_field:
        return series

    source_field = options.source_field or pa.field(series.name, polars_type_to_arrow_type(series.dtype))
    target_polars_field = options.get_target_polars_field()

    # strict=True => fail on lossy casts
    if isinstance(target_polars_field.dtype, polars.Datetime):
        if isinstance(series.dtype, polars.Utf8):
            casted = polars_strptime(series, options=options)
        else:
            casted = series.cast(
                target_polars_field.dtype,
                strict=options.safe
            )
    elif isinstance(target_polars_field.dtype, polars.List):
        # For Structs, we need to cast each subfield individually to respect nullability
        casted = cast_to_list_array(series, options=options)
    elif isinstance(target_polars_field.dtype, polars.Struct):
        # For Structs, we need to cast each subfield individually to respect nullability
        casted = cast_to_struct_array(series, options=options)
    else:
        casted = series.cast(target_polars_field.dtype, strict=options.safe)

    # If Arrow says "non-nullable", fill nulls with a default value
    if source_field.nullable and not target_field.nullable:
        dv = default_arrow_scalar(target_field.type, nullable=target_field.nullable).as_py()
        casted = casted.fill_null(dv)

    # Preserve original series name
    return casted.alias(series.name)


def cast_to_list_array(
    series: "polars.Series",
    options: Optional["CastOptions"] = None,
) -> "polars.Series":
    """
    Cast a Polars List Series to a target Arrow List type using *our own*
    cast_polars_series logic for the inner elements.

    Steps:
    - explode list elements
    - cast inner values via cast_polars_series
    - group back into lists
    - restore null-list rows as None
    """
    options = CastOptions.check_arg(options)
    from .arrow_cast import cast_to_list_array

    arrow_array = polars_series_to_arrow_array(series)
    casted = cast_to_list_array(arrow_array, options)

    return arrow_array_to_polars_series(casted, options)


def cast_to_struct_array(
    series: "polars.Series",
    options: Optional[CastOptions] = None,
) -> "polars.Series":
    """
    Cast a Polars Struct Series to a target Arrow Struct type using Polars casting rules.

    Each subfield is cast individually.
    """
    options = CastOptions.check_arg(options)
    target_arrow_field = options.target_field

    if not target_arrow_field:
        return series

    source_polars_type = series.dtype

    if not isinstance(source_polars_type, polars.Struct):
        raise ValueError(f"Cannot make struct polars series from {source_polars_type}")

    source_polars_fields = source_polars_type.fields
    source_arrow_fields = [polars_field_to_arrow_field(f) for f in source_polars_fields]

    target_arrow_fields: list[pa.Field] = list(target_arrow_field.type)
    target_polars_fields = [arrow_field_to_polars_field(f)for f in target_arrow_fields]

    name_to_index = {f.name: idx for idx, f in enumerate(source_polars_fields)}
    if not options.strict_match_names:
        name_to_index.update({
            f.name.casefold(): idx for idx, f in enumerate(source_polars_fields)
        })

    children = []

    for target_index, child_target_polars_field in enumerate(target_polars_fields):
        child_target_arrow_field: pa.Field = target_arrow_fields[target_index]

        find_name = child_target_polars_field.name if options.strict_match_names else child_target_polars_field.name.casefold()
        source_index = name_to_index.get(find_name)

        if source_index is None:
            if not options.add_missing_columns:
                raise ValueError(f"Missing column {child_target_arrow_field!r} from {target_arrow_fields}")

            dv = default_arrow_scalar(dtype=child_target_arrow_field.type, nullable=child_target_arrow_field.nullable)
            casted_child = polars.lit(value=dv.as_py(), dtype=child_target_polars_field.dtype)
        else:
            child_source_arrow_field: pa.Field = source_arrow_fields[source_index]
            child_source_polars_field: polars.Field = source_polars_fields[source_index]

            casted_child = cast_polars_array(
                series.struct.field(child_source_polars_field.name),
                options=options.copy(
                    source_field=child_source_arrow_field,
                    target_field=child_target_arrow_field
                )
            )

        children.append(casted_child.alias(child_target_polars_field.name))

    return polars.struct(*children).alias(target_arrow_field.name)


def polars_strptime(
    series: "polars.Series",
    options: Optional[CastOptions] = None,
) -> "polars.Series":
    """
    Helper to parse strings to datetime in Polars using optional patterns.
    """
    options = CastOptions.check_arg(options)
    polars_field = options.get_target_polars_field()

    if polars_field is None:
        polars_field = polars.Field(series.name, polars.Datetime("us", "UTC"))

    patterns = options.datetime_patterns or []

    if not patterns:
        # No patterns provided; use default parsing
        return (
            series.str
            .strptime(polars_field.dtype, strict=options.safe)
            .alias(polars_field.name)
        )

    # Try each pattern in sequence until one works
    last_error = None
    for pattern in patterns:
        try:
            return (
                series
                .str.strptime(
                    polars_field.dtype,
                    format=pattern,
                    strict=True,
                    ambiguous="earliest"
                )
            )
        except Exception as e:
            last_error = e

    # If none worked, raise the last error
    raise last_error


@polars_converter(PolarsDataFrame, PolarsDataFrame)
def cast_polars_dataframe(
    data: "polars.DataFrame",
    options: Optional[CastOptions] = None,
) -> "polars.DataFrame":
    """
    Cast a Polars DataFrame to a target Arrow schema using Arrow casting rules.

    Uses:
    - name / case-insensitive / positional matching for columns
    - add_missing_columns to synthesize columns with defaults
    - allow_add_columns to keep or drop extra source columns
    """
    options = CastOptions.check_arg(options)
    target_arrow_schema = options.target_arrow_schema

    if target_arrow_schema is None:
        return data

    sub_source_polars_fields = [
        polars.Field(name, d)
        for name, d in data.schema.items()
    ]
    sub_source_arrow_fields = [
        polars_field_to_arrow_field(f)
        for f in sub_source_polars_fields
    ]
    szb_target_polars_fields = [
        arrow_field_to_polars_field(f)
        for f in target_arrow_schema
    ]

    source_name_to_index = {
        field.name: idx for idx, field in enumerate(sub_source_polars_fields)
    }

    if not options.strict_match_names:
        source_name_to_index.update({
            field.name.casefold(): idx for idx, field in enumerate(sub_source_polars_fields)
        })

    columns: list[Tuple[polars.Field, polars.Series]] = []
    found_column_names = set()

    for sub_target_index, sub_target_field in enumerate(szb_target_polars_fields):
        sub_target_field: polars.Field = sub_target_field
        target_arrow_field = target_arrow_schema.field(sub_target_index)
        source_index = source_name_to_index.get(sub_target_field.name)

        if source_index is None:
            if not options.add_missing_columns:
                raise pa.ArrowInvalid(f"Missing column '{sub_target_field.name}' in source polars dataframe {sub_source_polars_fields}")

            dv = default_arrow_scalar(
                target_arrow_field.type,
                nullable=target_arrow_field.nullable
            )
            series = polars.repeat(
                value=dv.as_py(),
                n=data.shape[0],
                dtype=sub_target_field.dtype
            )
        else:
            source_field = sub_source_arrow_fields[source_index]
            found_column_names.add(source_field.name)
            source_series = data[:, source_index]
            series = cast_polars_array(
                source_series,
                options=options.copy(
                    source_field=source_field,
                    target_field=target_arrow_field
                )
            )

        columns.append((sub_target_field, series.alias(sub_target_field.name)))

    # Start with only the casted schema columns
    result = data.select(c for _, c in columns)

    # If we allow extra columns, horizontally concat them back
    if options.allow_add_columns:
        extra_cols = [
            name for name in data.columns if name not in found_column_names
        ]

        if extra_cols:
            extra_df = data.select(extra_cols)
            result = polars.concat([result, extra_df], how="horizontal")

    return result


# ---------------------------------------------------------------------------
# Polars <-> Arrow conversion helpers
# ---------------------------------------------------------------------------
@polars_converter(PolarsSeries, pa.Array)
def polars_series_to_arrow_array(
    series: "polars.Series",
    options: Optional[CastOptions] = None,
) -> pa.Array:
    """
    Convert a Polars Series to a pyarrow.Array.

    - If cast_options has a target_field, the Series is first cast in Polars
      using `cast_polars_series`, then converted to Arrow.
    - Otherwise, we just call `series.to_arrow()`.
    """
    options = CastOptions.check_arg(options)

    if options.target_field is not None:
        series = cast_polars_array(series, options)

    arr = series.to_arrow()

    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()

    return arr


@polars_converter(PolarsDataFrame, pa.Table)
def polars_dataframe_to_arrow_table(
    data: "polars.DataFrame",
    options: Optional[CastOptions] = None,
) -> pa.Table:
    """
    Convert a Polars DataFrame to a pyarrow.Table.

    - If cast_options.target_schema is set, we apply `cast_polars_dataframe`
      first, then call `.to_arrow()`.
    - Otherwise, we directly call `data.to_arrow()`.
    """
    options = CastOptions.check_arg(options)
    target_field = options.target_field

    if target_field is not None:
        data = cast_polars_dataframe(data, options)

    table = data.to_arrow()

    # If you want Arrow-side casting too, keep this; otherwise it’s redundant.
    if target_field is not None:
        table = cast_arrow_tabular(table, options)

    return table


@polars_converter(pa.Array, PolarsSeries)
@polars_converter(pa.ChunkedArray, PolarsSeries)
def arrow_array_to_polars_series(
    arr: pa.Array,
    options: Optional[CastOptions] = None,
) -> "polars.Series":
    """
    Convert a pyarrow.Array (or ChunkedArray) to a Polars Series.

    - If cast_options.target_field is set, we first apply `cast_arrow_array`
      and then build the Series.
    """
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()

    options = CastOptions.check_arg(options)

    if options.target_field is not None:
        arr = cast_arrow_array(arr, options)

    series = polars.from_arrow(arr)
    assert isinstance(series, polars.Series)
    return series


@polars_converter(pa.Table, PolarsDataFrame)
def arrow_table_to_polars_dataframe(
    table: pa.Table,
    options: Optional[CastOptions] = None,
) -> "polars.DataFrame":
    """
    Convert a pyarrow.Table to a Polars DataFrame.

    - If cast_options.target_schema is set, we first apply `cast_arrow_table`
      then call `polars.from_arrow`.
    """
    options = CastOptions.check_arg(options)

    if options.target_arrow_schema is not None:
        table = cast_arrow_tabular(table, options)

    return polars.from_arrow(table)


# ---------------------------------------------------------------------------
# RecordBatchReader <-> Polars DataFrame
# ---------------------------------------------------------------------------


@polars_converter(PolarsDataFrame, pa.RecordBatchReader)
def polars_dataframe_to_record_batch_reader(
    dataframe: "polars.DataFrame",
    cast_options: Optional[CastOptions] = None,
) -> pa.RecordBatchReader:
    """
    Convert a Polars DataFrame to a pyarrow.RecordBatchReader.

    - If cast_options.target_schema is set, we apply `cast_polars_dataframe`
      first, then convert to Arrow and wrap as a RecordBatchReader.
    """
    table = polars_dataframe_to_arrow_table(dataframe, cast_options)
    batches = table.to_batches()
    return pa.RecordBatchReader.from_batches(table.schema, batches)


@polars_converter(pa.RecordBatchReader, PolarsDataFrame)
def record_batch_reader_to_polars_dataframe(
    reader: pa.RecordBatchReader,
    cast_options: Optional[CastOptions] = None,
) -> "polars.DataFrame":
    """
    Convert a pyarrow.RecordBatchReader to a Polars DataFrame.

    - If cast_options.target_schema is set, we first apply
      `cast_arrow_record_batch_reader` and then collect to a Table and Polars DF.
    """
    opts = CastOptions.check_arg(cast_options)

    if opts.target_arrow_schema is not None:
        reader = cast_arrow_record_batch_reader(reader, opts)

    batches = list(reader)
    if not batches:
        # empty reader -> empty DataFrame
        empty_table = pa.Table.from_arrays([], names=[])
        return polars.from_arrow(empty_table)

    table = pa.Table.from_batches(batches)
    # opts already applied above if needed; no need to double-cast
    return arrow_table_to_polars_dataframe(table, None)
