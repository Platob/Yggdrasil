from typing import Optional, Union

import pyarrow as pa

from ...libs.polarslib import polars, require_polars
from .arrow import ArrowCastOptions, _default_array, _default_python_value

__all__ = [
    "cast_polars_series",
    "cast_polars_dataframe",
]


@require_polars
def _arrow_type_to_polars_dtype(arrow_type: pa.DataType) -> "polars.datatypes.DataType":
    """Derive a Polars dtype from an Arrow dtype."""

    placeholder = pa.Table.from_arrays(
        [pa.array([], type=arrow_type)], names=["__yggdrasil_tmp__"]
    )
    return polars.from_arrow(placeholder).schema["__yggdrasil_tmp__"]


def cast_polars_series(
    series: "polars.Series",
    arrow_type: Union[pa.DataType, pa.Field],
    options: Optional[ArrowCastOptions] = None,
) -> "polars.Series":
    """Cast a Polars Series to a target Arrow type using Polars casting rules."""

    options = options or ArrowCastOptions()

    if isinstance(arrow_type, pa.Field):
        target_dtype = arrow_type.type
        nullable = arrow_type.nullable
    else:
        target_dtype = arrow_type
        nullable = True

    polars_dtype = _arrow_type_to_polars_dtype(target_dtype)
    casted = series.cast(polars_dtype, strict=options.safe)

    if not nullable:
        default_value = _default_python_value(target_dtype)
        casted = casted.fill_null(default_value)

    # Preserve the original series name where possible.
    return casted.alias(series.name)


@require_polars
def cast_polars_dataframe(
    dataframe: "polars.DataFrame",
    options: Optional[ArrowCastOptions] = None,
) -> "polars.DataFrame":
    """Cast a Polars DataFrame to a target Arrow schema using Arrow casting rules."""

    options = options or ArrowCastOptions()
    arrow_schema = options.target_schema

    if arrow_schema is None:
        raise pa.ArrowInvalid("Target schema is required for casting Polars DataFrame")

    if isinstance(arrow_schema, pa.Field):
        arrow_schema = pa.schema([arrow_schema])

    exact_name_to_index = {name: idx for idx, name in enumerate(dataframe.columns)}
    folded_name_to_index = {name.casefold(): idx for idx, name in enumerate(dataframe.columns)}

    columns = []
    for field in arrow_schema:
        if field.name in exact_name_to_index:
            series = dataframe[:, exact_name_to_index[field.name]]
        elif not options.strict_match_names and field.name.casefold() in folded_name_to_index:
            series = dataframe[:, folded_name_to_index[field.name.casefold()]]
        elif not options.strict_match_names and dataframe.width > len(columns):
            series = dataframe[:, len(columns)]
        elif options.add_missing_columns:
            default_arr = _default_array(field, len(dataframe))
            series = polars.from_arrow(
                pa.Table.from_arrays([default_arr], names=[field.name])
            ).select(field.name).to_series()
        else:
            raise pa.ArrowInvalid(f"Missing column {field.name} while casting Polars DataFrame")

        casted = cast_polars_series(series, field, options)
        columns.append(casted.alias(field.name))

    if not options.allow_add_columns and dataframe.width > len(columns):
        dataframe = dataframe.select(dataframe.columns[: len(columns)])

    return polars.DataFrame(columns)
