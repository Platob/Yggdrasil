"""Polars cast helpers for :class:`StructType` targets.

The polars implementation is expression-first — series and tabular
casts both lower to a tree of :class:`pl.Expr` operations, which lets
the cast fuse into the LazyFrame plan when the source is lazy.

Three source shapes lower to expressions:

* :func:`cast_polars_struct_expr` — struct → struct (per-child rebuild
  via ``pl.struct(...)``, null preserved with ``pl.when(is_null)``).
* :func:`cast_polars_map_expr` — list-of-key/value structs → struct,
  one ``list.eval(when key == name)`` lookup per target child.
* :func:`cast_polars_list_expr` — list → struct by positional index
  via ``list.get(i, null_on_oob=True)``.

Series wrappers (:func:`cast_polars_struct_series`,
:func:`cast_polars_list_series`) build a single-column DataFrame and
``.select(expr).to_series()`` — same lowering, eager driver.

:func:`cast_polars_tabular` rebuilds a DataFrame/LazyFrame against
the merged schema with one expression per target column.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from yggdrasil.data.types.id import DataTypeId
from yggdrasil.lazy_imports import polars_module
from yggdrasil.lazy_imports import polars_module

if TYPE_CHECKING:
    import polars
    from yggdrasil.data.options import CastOptions
    from yggdrasil.data.data_field import Field
    from .array import ArrayType
    from .map import MapType
    from .struct import StructType


__all__ = [
    "cast_polars_struct_expr",
    "cast_polars_map_expr",
    "cast_polars_list_expr",
    "cast_polars_struct_series",
    "cast_polars_list_series",
    "cast_polars_tabular",
]


def cast_polars_struct_expr(
    expr: Any,
    options: "CastOptions",
) -> Any:
    pl = polars_module()

    if not options.need_cast(expr):
        return expr

    if options.source.dtype.type_id != DataTypeId.STRUCT:
        raise TypeError(f"Cannot cast {options.source} to {options.target}")

    source_field: "Field" = options.source
    target_type: "StructType" = options.target.dtype

    fields: list[Any] = []

    for i, target_child in enumerate(target_type.children):
        source_child = source_field.field(name=target_child.name, index=i, raise_error=False)

        if source_child is None:
            child_expr = target_child.default_polars_expr(alias=target_child.name)
        else:
            child_expr = target_child.cast_polars_expr(
                expr.struct.field(source_child.name),
                options=options.copy(
                    source=source_child,
                    target=target_child,
                ),
            ).alias(target_child.name)

        fields.append(child_expr)

    struct_expr = pl.struct(fields)

    return pl.when(expr.is_null()).then(pl.lit(None)).otherwise(struct_expr)


def cast_polars_map_expr(
    expr: Any,
    options: "CastOptions",
) -> Any:
    pl = polars_module()

    if not options.need_cast(expr):
        return expr

    if options.source.dtype.type_id != DataTypeId.MAP:
        raise TypeError(f"Cannot cast {options.source} to {options.target}")

    source_field: "Field" = options.source
    source_type: "MapType" = source_field.dtype
    target_type: "StructType" = options.target.dtype

    fields: list[Any] = []

    for target_child in target_type.children:
        matched_values = expr.list.eval(
            pl.when(
                pl.element().struct.field(source_type.key_field.name) == pl.lit(target_child.name)
            )
            .then(pl.element().struct.field(source_type.value_field.name))
            .otherwise(None)
        )

        extracted = matched_values.list.drop_nulls().list.first()

        casted = target_child.cast_polars_expr(
            extracted,
            options=options.copy(
                source=source_type.value_field,
                target=target_child,
            ),
        ).alias(target_child.name)

        fields.append(casted)

    struct_expr = pl.struct(fields)
    casted = pl.when(expr.is_null()).then(pl.lit(None)).otherwise(struct_expr)
    return options.polars_alias(casted)


def cast_polars_list_expr(
    expr: Any,
    options: "CastOptions",
) -> Any:
    pl = polars_module()

    if not options.need_cast(expr):
        return expr

    if options.source.dtype.type_id != DataTypeId.ARRAY:
        raise TypeError(f"Cannot cast {options.source} to {options.target}")

    source_field: "Field" = options.source
    source_type: "ArrayType" = source_field.dtype
    target_type: "StructType" = options.target.dtype

    fields: list[Any] = []

    for i, target_child in enumerate(target_type.children):
        extracted = expr.list.get(i, null_on_oob=True)

        casted = target_child.cast_polars_expr(
            extracted,
            options=options.copy(
                source=source_type.item_field,
                target=target_child,
            ),
        ).alias(target_child.name)

        fields.append(casted)

    struct_expr = pl.struct(fields)
    return pl.when(expr.is_null()).then(pl.lit(None)).otherwise(struct_expr)


def cast_polars_struct_series(
    series: "polars.Series",
    options: "CastOptions",
) -> "polars.Series":
    if not options.need_cast(series):
        return series

    pl = polars_module()
    expr = cast_polars_struct_expr(pl.col(series.name), options).alias(options.target.name)
    return pl.DataFrame({series.name: series}).select(expr).to_series()


def cast_polars_list_series(
    series: "polars.Series",
    options: "CastOptions",
) -> "polars.Series":
    if not options.need_cast(series):
        return series

    pl = polars_module()
    expr = cast_polars_list_expr(pl.col(series.name), options).alias(options.target.name)
    return pl.DataFrame({series.name: series}).select(expr).to_series()


def cast_polars_tabular(
    data: "polars.DataFrame | polars.LazyFrame",
    options: "CastOptions",
) -> "polars.DataFrame | polars.LazyFrame":
    pl = polars_module()

    if not isinstance(data, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError(f"Unsupported tabular type: {type(data)!r}")

    if not options.need_cast(data, check_names=True):
        return data

    source_schema = options.source
    target_schema = options.merged_schema

    # Engine-level fast bypass — when the source polars schema already
    # equals the target engine schema, the per-column rebuild produces
    # the same frame back. Field-level ``need_cast`` may flag a
    # difference for metadata / subclass dtypes that don't surface in
    # ``pl.DataFrame.schema``; this short-circuits those cases. Lazy
    # frames go through ``collect_schema`` so we don't materialize.
    target_pl_schema = target_schema.to_polars_schema()
    source_pl_schema = (
        data.collect_schema() if isinstance(data, pl.LazyFrame) else data.schema
    )
    if source_pl_schema == target_pl_schema:
        return data

    exprs: list[Any] = []

    for i, target_field in enumerate(target_schema.children):
        source_field = source_schema.field_by(
            name=target_field.name,
            index=i,
            raise_error=False,
        )

        if source_field is None:
            expr = target_field.default_polars_expr(alias=target_field.name)
        else:
            expr = target_field.cast_polars_expr(
                pl.col(source_field.name),
                options=options.copy(
                    source=source_field,
                    target=target_field,
                ),
            )

        exprs.append(expr)

    return data.select(exprs)
