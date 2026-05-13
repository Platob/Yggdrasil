""":class:`StructType` — yggdrasil's named-children nested type.

Carries a tuple of :class:`Field` children and dispatches casts
across pyarrow / polars / pandas / spark by delegating to per-engine
helper modules:

* :mod:`.struct_arrow` — Arrow-side casts (struct/map/list/tabular).
* :mod:`.struct_polars` — Polars expression / series / tabular casts.
* :mod:`.struct_pandas` — Pandas object-dtype casts.
* :mod:`.struct_spark` — Spark Column expression casts.

The class methods (``_cast_arrow_array``, ``_cast_polars_series``,
…) are the dispatch surface — they pick the right helper based on
the source's :class:`DataTypeId`. The helpers themselves live in
the engine modules, imported at the bottom of this file to avoid
circular imports (each engine module needs :class:`StructType`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping

import pyarrow as pa
from yggdrasil.data.enums import Mode
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.nested import NestedType
from yggdrasil.data.types.nested._cast_json import (
    cast_arrow_json_string_array,
    cast_polars_json_string_expr,
    cast_spark_json_string_column,
    is_json_string_source,
)
from yggdrasil.lazy_imports import polars_module, spark_sql_module
from yggdrasil.environ.importlib import cached_from_import
from yggdrasil.lazy_imports import field_class

if TYPE_CHECKING:
    import pandas as pd
    import polars
    import pyspark.sql as psql
    import pyspark.sql.types as pst
    from yggdrasil.data.options import CastOptions
    from yggdrasil.data.data_field import Field


__all__ = [
    "NestedType",
    "StructType",
]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, repr=False)
class StructType(NestedType):
    fields: tuple["Field"] = field(default_factory=tuple)

    def __bool__(self):
        return bool(self.fields)

    def __post_init__(self):
        object.__setattr__(self, "fields", tuple(self.fields))

    def equals(
        self,
        other: "DataType",
        check_names: bool = True,
        check_dtypes: bool = True,
        check_metadata: bool = True,
    ) -> bool:
        if not isinstance(other, StructType):
            return False
        if len(self.fields) != len(other.fields):
            return False
        for i, (f1, f2) in enumerate(zip(self.fields, other.fields)):
            if not f1.name:
                f1.with_name(f2.name, inplace=True)
            elif not f2.name:
                f2.with_name(f1.name, inplace=True)

            if not f1.equals(f2, check_names=check_names, check_dtypes=check_dtypes, check_metadata=check_metadata):
                return False
        return True

    def pretty_format(self, indent: int = 2, level: int = 0) -> str:
        pad = " " * (indent * level)

        if not self.fields:
            return f"{pad}struct<>"

        body = ",\n".join(
            child.pretty_format(indent=indent, level=level + 1) for child in self.fields
        )
        return f"{pad}struct<\n{body}\n{pad}>"

    def default_pyobj(self, nullable: bool) -> Any:
        return None if nullable else {
            f.name: f.default_value
            for f in self.fields
        }

    @property
    def children(self) -> tuple["Field"]:
        return self.fields

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.STRUCT

    def _merge_with_same_id(
        self,
        other: "StructType",
        mode: "Mode" = Mode.AUTO,
        downcast: bool = False,
        upcast: bool = False,
    ) -> "StructType":
        if not isinstance(other, StructType):
            raise TypeError(
                f"Cannot merge {self.__class__.__name__} with {other.__class__.__name__}"
            )

        if mode is Mode.IGNORE:
            return self

        merged_fields: list[Field] = []
        missing_fields: list[Field] = []
        seen = set()

        if not self.fields:
            source_type, target_type = self, other
        elif not other.fields:
            source_type, target_type = other, self
        else:
            source_type, target_type = (self, other) if mode is Mode.OVERWRITE else (other, self)

        for i, f in enumerate(target_type.fields):
            found = source_type.field_by(name=f.name, index=i, raise_error=False)
            seen.add(f.name)
            if found is None:
                if mode is Mode.APPEND and not f.dtype.type_id.is_any_or_null:
                    missing_fields.append(f)
                elif mode is Mode.UPSERT:
                    merged_fields.append(f)
            else:
                seen.add(found.name)
                mf = f.merge_with(found, mode=mode, downcast=downcast, upcast=upcast)
                merged_fields.append(mf)

        if mode is Mode.APPEND:
            for f in source_type.fields:
                if f.name not in seen and not f.dtype.type_id.is_any_or_null:
                    missing_fields.append(f)

        merged_fields.extend(missing_fields)

        return self.__class__(fields=tuple(merged_fields))

    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        return pa.types.is_struct(dtype)

    @classmethod
    def from_arrow_type(cls, dtype: pa.DataType) -> "StructType":
        if not pa.types.is_struct(dtype):
            raise TypeError(f"Unsupported Arrow data type: {dtype!r}")
        _f = field_class()
        return cls(fields=[_f.from_arrow_field(f) for f in dtype])

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        pl = polars_module()
        return isinstance(dtype, pl.Struct)

    @classmethod
    def from_polars_type(cls, dtype: "polars.Struct") -> "StructType":
        return cls(fields=[
            cached_from_import("yggdrasil.data.data_field", "Field").from_polars(f)
            for f in dtype.fields
        ])

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        spark = spark_sql_module()
        return isinstance(dtype, spark.types.StructType)

    @classmethod
    def from_spark_type(cls, dtype: "pst.StructType") -> "StructType":
        # Each child is already a ``pst.StructField`` — go through
        # ``from_spark_field`` directly so we skip the isinstance fan
        # ``from_spark`` would walk per child (DataFrame /
        # StructField / DataType / Column branches), and the cached
        # ``from_spark_type`` lookup keeps repeated child dtypes fast.
        Field = cached_from_import("yggdrasil.data.data_field", "Field")
        return cls(fields=[Field.from_spark_field(f) for f in dtype.fields])

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        return cls._matches_dict(value, DataTypeId.STRUCT)

    @classmethod
    def empty(cls) -> "StructType":
        return cls(fields=[])

    @classmethod
    def from_fields(cls, values: Iterable["Field"]):
        if isinstance(values, Mapping):
            return cls(fields=tuple(values.values()))
        return cls(fields=tuple(values))

    @classmethod
    def from_dict(cls, value: dict[str, Any], default: Any = ...) -> "StructType":
        fields = value.get("fields", [])

        try:
            return cls(fields=[field_class().from_dict(f) for f in fields])
        except (TypeError, ValueError):
            if default is ...:
                raise
            return default

    def to_arrow(self) -> pa.DataType:
        return pa.struct([f.to_arrow_field() for f in self.fields])

    def to_polars(self) -> "polars.DataType":
        pl = polars_module()
        return pl.Struct([f.to_polars_field() for f in self.fields])

    def to_spark(self) -> Any:
        spark = spark_sql_module()
        return spark.types.StructType([f.to_pyspark_field() for f in self.fields])

    def as_spark(self) -> "StructType":
        # Recurse via the field-level :meth:`Field.as_spark` so each
        # child's metadata + nullability survive alongside its
        # Spark-flavored dtype.
        spark_fields = tuple(f.as_spark() for f in self.fields)
        if all(a is b for a, b in zip(spark_fields, self.fields)):
            return self
        return StructType(fields=spark_fields)

    def as_polars(self) -> "StructType":
        polars_fields = tuple(f.as_polars() for f in self.fields)
        if all(a is b for a, b in zip(polars_fields, self.fields)):
            return self
        return StructType(fields=polars_fields)

    def to_spark_name(self) -> str:
        fields_ddl = ", ".join(
            # Double any embedded backticks so Databricks/Spark parses the
            # backtick-quoted identifier correctly.
            f"`{f.name.replace('`', '``')}`: {f.dtype.to_spark_name()}"
            for f in self.fields
        )
        return f"STRUCT<{fields_ddl}>"

    def to_dict(self) -> dict[str, Any]:
        base = super(StructType, self).to_dict()
        base["fields"] = [f.to_dict() for f in self.fields]
        return base

    def with_fields(
        self,
        fields: list[Field],
        safe: bool = False,
        inplace: bool = True,
    ) -> "StructType":
        if not safe:
            _f = field_class()
            fields = [_f.from_any(_) for _ in fields]

        if inplace:
            object.__setattr__(self, "fields", tuple(fields))
            return self
        return self.__class__(fields=tuple(fields))

    def _convert_pyobj(self, value: Any, safe: bool = False) -> dict | None:
        if isinstance(value, (bytes, bytearray, memoryview)):
            try:
                value = bytes(value).decode("utf-8")
            except UnicodeDecodeError:
                if safe:
                    raise ValueError(
                        f"Cannot decode bytes as UTF-8 for {type(self).__name__}: "
                        f"{value!r}"
                    )
                return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                if safe:
                    raise ValueError(
                        f"Cannot parse struct from empty string for "
                        f"{type(self).__name__}."
                    )
                return None
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                if safe:
                    raise ValueError(
                        f"Cannot parse struct from {value!r} for "
                        f"{type(self).__name__}."
                    )
                return None
            value = decoded

        if isinstance(value, dict):
            source = value
        elif isinstance(value, (list, tuple)):
            if len(value) != len(self.fields):
                if safe:
                    raise ValueError(
                        f"Positional struct input has {len(value)} items but "
                        f"{type(self).__name__} expects {len(self.fields)}."
                    )
                # Best-effort: truncate or pad with None.
                value = list(value)
                if len(value) < len(self.fields):
                    value = value + [None] * (len(self.fields) - len(value))
                else:
                    value = value[: len(self.fields)]
            source = {f.name: v for f, v in zip(self.fields, value)}
        elif hasattr(value, "asDict"):
            source = value.asDict(recursive=True)
        elif hasattr(value, "__dict__"):
            source = dict(value.__dict__)
        else:
            if safe:
                raise ValueError(
                    f"Cannot convert {type(value).__name__} to struct "
                    f"for {type(self).__name__}: {value!r}."
                )
            return None

        out: dict[str, Any] = {}
        for child in self.fields:
            raw = source.get(child.name)
            out[child.name] = child.dtype.convert_pyobj(
                raw, nullable=child.nullable, safe=safe
            )
        return out

    # ==================================================================
    # Engine dispatch — by source DataTypeId
    # ==================================================================

    def _cast_arrow_array(
        self,
        array: pa.Array,
        options: "CastOptions",
    ) -> pa.StructArray | pa.ChunkedArray:
        # Engine-level bypass — when the array's arrow type already
        # matches the target's projection (including per-child
        # nullability — ``pa.DataType.__eq__`` compares field
        # nullability inside ``pa.struct``), every downstream branch
        # would either rebuild the same buffers or short-circuit
        # anyway. Skip the ``check_source`` Field-from-arrow peek
        # (which builds a fresh Field tree from the struct array) and
        # the per-child rebuild by returning ``array`` directly. Mirror
        # :meth:`ArrayType._cast_arrow_array` / :meth:`MapType._cast_arrow_array`
        # so struct casts pay the same MATCH floor as their nested peers.
        if array.type == self.to_arrow():
            return array

        options = options.check_source(array).check_target(self)

        source_type_id = options.source.dtype.type_id

        if source_type_id == DataTypeId.NULL or array.null_count == len(array):
            return options.target.default_arrow_array(
                size=len(array),
                memory_pool=options.arrow_memory_pool,
            )

        elif is_json_string_source(source_type_id):
            return cast_arrow_json_string_array(array, options=options)

        elif source_type_id == DataTypeId.STRUCT:
            return cast_arrow_struct_array(array, options)

        elif source_type_id == DataTypeId.MAP:
            return cast_arrow_map_array(array, options)

        elif source_type_id == DataTypeId.ARRAY:
            return cast_arrow_list_array(array, options)

        else:
            raise pa.ArrowInvalid(
                f"Cannot cast {options.source} to {options.target}"
            )

    def _cast_arrow_tabular(
        self,
        table: pa.Table | pa.RecordBatch,
        options: "CastOptions",
    ):
        return cast_arrow_tabular(table, options.check_source(table).check_target(self))

    def _cast_arrow_batch_iterator(
        self,
        batches: Iterable[pa.RecordBatch],
        options: "CastOptions",
    ) -> Iterator[pa.RecordBatch]:
        """Cast a stream of :class:`pa.RecordBatch` against this struct.

        Per-batch goes through :meth:`_cast_arrow_tabular`; when
        ``options.byte_size`` is set, the output stream is repacked to
        approximately that many bytes per batch. See
        :func:`struct_arrow.cast_arrow_batch_iterator` for the full
        algorithm.
        """
        return cast_arrow_batch_iterator(batches, options.check_target(self))

    def _cast_polars_series(
        self,
        series: "polars.Series",
        options: "CastOptions",
    ) -> "polars.Series":
        # Engine-level bypass mirroring :meth:`_cast_arrow_array` — when
        # the series' polars dtype already matches the target's
        # projection, the per-child rebuild produces the same series
        # back. Skip the ``check_source`` peek + DataFrame/select hop
        # that the expression path takes downstream.
        if series.dtype == self.to_polars():
            return series

        pl = polars_module()
        options = options.check_source(series).check_target(self)

        if options.source.dtype.type_id == DataTypeId.NULL or series.null_count() == len(series):
            return options.target.default_polars_series(size=len(series))

        expr = self._cast_polars_expr(
            pl.col(series.name),
            options=options,
        ).alias(options.target.name)
        return pl.DataFrame({series.name: series}).select(expr).to_series()

    def _cast_polars_expr(
        self,
        expr: Any,
        options: "CastOptions",
    ) -> Any:
        options = options.check_target(self)

        source_type_id = options.source.dtype.type_id

        if source_type_id == DataTypeId.NULL:
            return options.target.default_polars_expr(alias=options.target.name)

        elif is_json_string_source(source_type_id):
            return cast_polars_json_string_expr(expr, options)

        elif source_type_id == DataTypeId.STRUCT:
            return cast_polars_struct_expr(expr, options)

        elif source_type_id == DataTypeId.MAP:
            return cast_polars_map_expr(expr, options)

        elif source_type_id == DataTypeId.ARRAY:
            return cast_polars_list_expr(expr, options)

        else:
            raise TypeError(
                f"Cannot cast {options.source} to {options.target}"
            )

    def _cast_polars_tabular(
        self,
        table: "polars.DataFrame | polars.LazyFrame",
        options: "CastOptions",
    ):
        return cast_polars_tabular(
            table,
            options.check_source(table).check_target(self),
        )

    def _cast_pandas_series(
        self,
        series: "pd.Series",
        options: "CastOptions",
    ) -> "pd.Series":
        options = options.check_source(series).check_target(self)

        source_type_id = options.source.dtype.type_id

        if source_type_id == DataTypeId.NULL or series.isna().all():
            return options.target.default_pandas_series(size=len(series))

        elif is_json_string_source(source_type_id):
            from .array import _cast_pandas_via_arrow

            return _cast_pandas_via_arrow(series, options, cast_arrow_json_string_array)

        elif source_type_id == DataTypeId.STRUCT:
            return cast_pandas_struct_series(series, options)

        elif source_type_id == DataTypeId.ARRAY:
            return cast_pandas_list_series(series, options)

        else:
            raise TypeError(
                f"Cannot cast {options.source} to {options.target}"
            )

    def _cast_pandas_tabular(
        self,
        frame: "pd.DataFrame",
        options: "CastOptions",
    ) -> "pd.DataFrame":
        return cast_pandas_tabular(frame, options.check_source(frame).check_target(self))

    def _cast_spark_column(
        self,
        column: Any,
        options: "CastOptions",
    ) -> Any:
        options = options.check_source(column).check_target(self)

        source_type_id = options.source.dtype.type_id

        if source_type_id == DataTypeId.NULL:
            return options.target.default_spark_column()

        elif is_json_string_source(source_type_id):
            return cast_spark_json_string_column(column, options)

        elif source_type_id == DataTypeId.STRUCT:
            return cast_spark_struct_column(column, options)

        elif source_type_id == DataTypeId.MAP:
            return cast_spark_map_column(column, options)

        elif source_type_id == DataTypeId.ARRAY:
            return cast_spark_list_column(column, options)

        else:
            raise TypeError(
                f"Cannot cast {options.source} to {options.target}"
            )

    def _cast_spark_tabular(
        self,
        frame: "psql.DataFrame",
        options: "CastOptions",
    ) -> "psql.DataFrame":
        return cast_spark_tabular(frame, options.check_source(frame).check_target(self))


# ---------------------------------------------------------------------------
# Engine helper imports — placed at the bottom so `StructType` is fully
# defined before the engine modules (which import it for type hints
# and isinstance checks) are loaded.
# ---------------------------------------------------------------------------

from .struct_arrow import (  # noqa: E402
    cast_arrow_struct_array,
    cast_arrow_map_array,
    cast_arrow_list_array,
    cast_arrow_tabular,
    cast_arrow_batch_iterator,
)
from .struct_polars import (  # noqa: E402
    cast_polars_struct_expr,
    cast_polars_map_expr,
    cast_polars_list_expr,
    cast_polars_tabular,
)
from .struct_pandas import (  # noqa: E402
    cast_pandas_struct_series,
    cast_pandas_list_series,
    cast_pandas_tabular,
)
from .struct_spark import (  # noqa: E402
    cast_spark_struct_column,
    cast_spark_map_column,
    cast_spark_list_column,
    cast_spark_tabular,
)
