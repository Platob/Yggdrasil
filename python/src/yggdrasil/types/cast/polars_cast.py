from dataclasses import replace
from typing import Optional

import pyarrow as pa

from .arrow_cast import (
    ArrowCastOptions,
    default_arrow_array,
    cast_arrow_array,
    cast_arrow_table,
    cast_arrow_record_batch_reader,
)
from .registry import register_converter
from ..python_defaults import default_from_arrow_hint
from ...libs.polarslib import (
    polars, require_polars,
    arrow_type_to_polars_type, polars_type_to_arrow_type,
    arrow_field_to_polars_field, polars_field_to_arrow_field
)

__all__ = [
    "cast_polars_series",
    "cast_polars_dataframe",
    "arrow_type_to_polars_type",
    "polars_type_to_arrow_type",
    "arrow_field_to_polars_field",
    "polars_field_to_arrow_field",
    "polars_series_to_arrow_array",
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
def cast_polars_series(
    series: "polars.Series",
    options: Optional[ArrowCastOptions] = None,
) -> "polars.Series":
    """
    Cast a Polars Series to a target Arrow type using Polars casting rules.

    `options` is normalized via ArrowCastOptions.check_arg and its
    `target_field` is used as the Arrow target.

    - target_field can be a pa.DataType, pa.Field, or pa.Schema (schema → first field).
    - If a Field is provided, we also respect its nullability by filling nulls
      when nullable=False (using default_from_arrow_hint).
    """
    options = ArrowCastOptions.check_arg(options)
    target_field = options.target_field

    if target_field is None:
        return series

    # Normalize schema -> first field for a Series cast
    if isinstance(target_field, pa.Schema):
        if len(target_field) != 1:
            raise ValueError("cast_polars_series: Schema target must have exactly one field")
        target_field = target_field[0]

    if not isinstance(target_field, pa.Field):
        # Treat DataType as a single anonymous field
        target_field = pa.field(series.name or "value", target_field, nullable=True)

    target_dtype = target_field.type
    nullable = target_field.nullable

    # Convert Arrow dtype -> Polars dtype
    polars_dtype = arrow_type_to_polars_type(target_dtype, options)

    # strict=True => fail on lossy casts
    casted = series.cast(polars_dtype, strict=options.safe)

    # If Arrow says "non-nullable", fill nulls with a default value
    if not nullable:
        dv = default_from_arrow_hint(target_dtype).as_py()
        casted = casted.fill_null(dv)

    # Preserve original series name
    return casted.alias(series.name)


@polars_converter(PolarsDataFrame, PolarsDataFrame)
def cast_polars_dataframe(
    data: "polars.DataFrame",
    options: Optional[ArrowCastOptions] = None,
) -> "polars.DataFrame":
    """
    Cast a Polars DataFrame to a target Arrow schema using Arrow casting rules.

    Uses:
    - name / case-insensitive / positional matching for columns
    - add_missing_columns to synthesize columns with defaults
    - allow_add_columns to keep or drop extra source columns
    """
    options = ArrowCastOptions.check_arg(options)
    arrow_schema = options.target_schema

    if arrow_schema is None:
        return data

    exact_name_to_index = {name: idx for idx, name in enumerate(data.columns)}
    folded_name_to_index = {
        str(name).casefold(): idx for idx, name in enumerate(data.columns)
    }

    columns: list["polars.Series"] = []

    for field in arrow_schema:
        # 1. Exact name match
        if field.name in exact_name_to_index:
            idx = exact_name_to_index[field.name]
            series = data[:, idx]

        # 2. Case-insensitive name match
        elif not options.strict_match_names and field.name.casefold() in folded_name_to_index:
            idx = folded_name_to_index[field.name.casefold()]
            series = data[:, idx]

        # 3. Positional fallback: reuse next column if allowed
        elif not options.strict_match_names and data.width > len(columns):
            series = data[:, len(columns)]

        # 4. Add missing columns if configured
        elif options.add_missing_columns:
            default_arr = default_arrow_array(field, len(data))
            series = (
                polars.from_arrow(
                    pa.Table.from_arrays([default_arr], names=[field.name])
                )
                .select(field.name)
                .to_series()
            )

        else:
            raise pa.ArrowInvalid(
                f"Missing column {field.name} while casting Polars DataFrame"
            )

        # Override target_field for this column
        col_options = replace(options, target_field=field)
        casted = cast_polars_series(series, col_options)
        columns.append(casted.alias(field.name))

    # Start with only the casted schema columns
    result = polars.DataFrame(columns)

    # If we allow extra columns, horizontally concat them back
    if options.allow_add_columns:
        schema_names = {field.name for field in arrow_schema}
        extra_cols = [name for name in data.columns if name not in schema_names]
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
    cast_options: Optional[ArrowCastOptions] = None,
) -> pa.Array:
    """
    Convert a Polars Series to a pyarrow.Array.

    - If cast_options has a target_field, the Series is first cast in Polars
      using `cast_polars_series`, then converted to Arrow.
    - Otherwise, we just call `series.to_arrow()`.
    """
    opts = ArrowCastOptions.check_arg(cast_options)

    if opts.target_field is not None:
        series = cast_polars_series(series, opts)

    arr = series.to_arrow()
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    return arr


@polars_converter(PolarsDataFrame, pa.Table)
def polars_dataframe_to_arrow_table(
    data: "polars.DataFrame",
    cast_options: Optional[ArrowCastOptions] = None,
) -> pa.Table:
    """
    Convert a Polars DataFrame to a pyarrow.Table.

    - If cast_options.target_schema is set, we apply `cast_polars_dataframe`
      first, then call `.to_arrow()`.
    - Otherwise, we directly call `data.to_arrow()`.
    """
    opts = ArrowCastOptions.check_arg(cast_options)

    if opts.target_schema is not None:
        data = cast_polars_dataframe(data, opts)

    table = data.to_arrow()

    # If you want Arrow-side casting too, keep this; otherwise it’s redundant.
    if opts.target_schema is not None:
        table = cast_arrow_table(table, opts)

    return table


@polars_converter(pa.Array, PolarsSeries)
@polars_converter(pa.ChunkedArray, PolarsSeries)
def arrow_array_to_polars_series(
    arr: pa.Array,
    cast_options: Optional[ArrowCastOptions] = None,
) -> "polars.Series":
    """
    Convert a pyarrow.Array (or ChunkedArray) to a Polars Series.

    - If cast_options.target_field is set, we first apply `cast_arrow_array`
      and then build the Series.
    """
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()

    opts = ArrowCastOptions.check_arg(cast_options)

    if opts.target_field is not None:
        arr = cast_arrow_array(arr, opts)

    series = polars.from_arrow(arr)
    assert isinstance(series, polars.Series)
    return series


@polars_converter(pa.Table, PolarsDataFrame)
def arrow_table_to_polars_dataframe(
    table: pa.Table,
    cast_options: Optional[ArrowCastOptions] = None,
) -> "polars.DataFrame":
    """
    Convert a pyarrow.Table to a Polars DataFrame.

    - If cast_options.target_schema is set, we first apply `cast_arrow_table`
      then call `polars.from_arrow`.
    """
    opts = ArrowCastOptions.check_arg(cast_options)

    if opts.target_schema is not None:
        table = cast_arrow_table(table, opts)

    df = polars.from_arrow(table)
    assert isinstance(df, polars.DataFrame)
    return df


# ---------------------------------------------------------------------------
# RecordBatchReader <-> Polars DataFrame
# ---------------------------------------------------------------------------


@polars_converter(PolarsDataFrame, pa.RecordBatchReader)
def polars_dataframe_to_record_batch_reader(
    dataframe: "polars.DataFrame",
    cast_options: Optional[ArrowCastOptions] = None,
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
    cast_options: Optional[ArrowCastOptions] = None,
) -> "polars.DataFrame":
    """
    Convert a pyarrow.RecordBatchReader to a Polars DataFrame.

    - If cast_options.target_schema is set, we first apply
      `cast_arrow_record_batch_reader` and then collect to a Table and Polars DF.
    """
    opts = ArrowCastOptions.check_arg(cast_options)

    if opts.target_schema is not None:
        reader = cast_arrow_record_batch_reader(reader, opts)

    batches = list(reader)
    if not batches:
        # empty reader -> empty DataFrame
        empty_table = pa.Table.from_arrays([], names=[])
        return polars.from_arrow(empty_table)

    table = pa.Table.from_batches(batches)
    # opts already applied above if needed; no need to double-cast
    return arrow_table_to_polars_dataframe(table, None)
