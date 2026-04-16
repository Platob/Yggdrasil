"""Casting options for Arrow- and engine-aware conversions."""
from dataclasses import dataclass
from typing import Any, Optional, Union, TypeVar, TYPE_CHECKING

import pyarrow as pa

from yggdrasil.data.data_field import Field
from yggdrasil.data.types.support import get_polars, get_spark_sql

if TYPE_CHECKING:
    import polars
    import pyspark.sql as ps

__all__ = [
    "CastOptions",
    "CastOptionsArg",
]

S = TypeVar("S", bound=Any)
CastOptionsArg = Union[
    "CastOptions",
    dict,
    pa.DataType,
    pa.Field,
    pa.Schema,
    "Any",
    "Any",
    None,
]


@dataclass(frozen=True, slots=True)
class CastOptions:
    source_field: Optional[Field] = None
    target_field: Optional[Field] = None
    safe: bool = False
    add_missing_fields: bool = True
    arrow_memory_pool: Optional[pa.MemoryPool] = None
    strict_match_names: bool = False
    add_missing_columns: bool = True
    allow_add_columns: bool = False
    datetime_formats: tuple[str, ...] = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    )

    @property
    def source_schema(self):
        return self.source_field.to_schema() if self.source_field is not None else None

    @property
    def target_schema(self):
        return self.target_field.to_schema() if self.target_field is not None else None

    def __post_init__(self):
        if self.source_field is not None:
            object.__setattr__(
                self, "source_field",
                Field.from_(self.source_field)
            )

        if self.target_field is not None:
            object.__setattr__(
                self, "target_field",
                Field.from_(self.target_field)
            )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def check(
        cls,
        options: CastOptionsArg = None,
        source: Any = None,
        source_field: Optional[Field] = None,
        target: Any = None,
        target_field: Optional[Field] = None,
        **kwargs,
    ) -> "CastOptions":
        if source is not None and source_field is None:
            source_field = Field.from_(source)

        if target is not None and target_field is None:
            target_field = Field.from_(target)

        if isinstance(options, CastOptions):
            if kwargs or source_field is not None or target_field is not None:
                return options.copy(
                    source_field=source_field,
                    target_field=target_field,
                    **kwargs
                )
            return options

        if isinstance(options, dict):
            kwargs = {**options, **kwargs}

        return cls(
            source_field=source_field,
            target_field=target_field,
            **kwargs
        )

    def copy(
        self,
        source_field: Field | None = None,
        target_field: Field | None = None,
        safe: bool | None = None,
        add_missing_columns: bool | None = None,
        strict_match_names: bool | None = None,
        add_missing_fields: bool | None = None,
        datetime_formats: tuple[str, ...] | None = None,
        arrow_memory_pool: pa.MemoryPool | None = None,
        allow_add_columns: bool | None = None,
    ) -> "CastOptions":
        return CastOptions(
            safe=self.safe if safe is None else safe,
            source_field=self.source_field if source_field is None else Field.from_(source_field),
            target_field=self.target_field if target_field is None else Field.from_(target_field),
            add_missing_fields=self.add_missing_fields if add_missing_fields is None else add_missing_fields,
            datetime_formats=self.datetime_formats if datetime_formats is None else datetime_formats,
            arrow_memory_pool=self.arrow_memory_pool if arrow_memory_pool is None else arrow_memory_pool,
            strict_match_names=self.strict_match_names if strict_match_names is None else strict_match_names,
            add_missing_columns=self.add_missing_columns if add_missing_columns is None else add_missing_columns,
            allow_add_columns=self.allow_add_columns if allow_add_columns is None else allow_add_columns,
        )

    def check_source(self, obj: Any) -> "CastOptions":
        if self.source_field is None and obj is not None:
            return self.with_source(obj, inplace=True)
        return self

    def with_source(self, obj: Any, inplace: bool = False) -> "CastOptions":
        f = Field.from_(obj)

        if inplace:
            object.__setattr__(self, "source_field", f)
            return self
        return self.copy(source_field=f)

    def check_target(self, obj: Any) -> "CastOptions":
        if self.target_field is None and obj is not None:
            return self.with_target(obj, inplace=True)
        return self

    def with_target(self, obj: Any, inplace: bool = False) -> "CastOptions":
        f = Field.from_(obj)

        if inplace:
            object.__setattr__(self, "target_field", f)
            return self
        return self.copy(target_field=f)

    def cast_arrow(
        self,
        obj: Union[
            pa.Array, pa.ChunkedArray, pa.Table, pa.RecordBatch,
            pa.DataType, pa.Field, pa.Schema,
        ]
    ):
        if isinstance(obj, (pa.Array, pa.ChunkedArray)):
            return self.cast_arrow_array(obj)
        elif isinstance(obj, (pa.Table, pa.RecordBatch)):
            return self.cast_arrow_tabular(obj)
        else:
            raise TypeError(f"Cannot cast {type(obj)} to arrow")

    def cast_arrow_array(
        self,
        array: pa.Array | pa.ChunkedArray
    ):
        if self.target_field is None:
            return array
        return self.target_field.cast_arrow_array(array, options=self)

    def cast_arrow_tabular(
        self,
        table: pa.Table | pa.RecordBatch
    ):
        if self.target_field is None:
            return table
        return self.target_field.cast_arrow_tabular(table, options=self)

    def cast_polars(
        self,
        obj: "polars.DataFrame | polars.LazyFrame | polars.Series | polars.Expr"
    ):
        if self.target_field is None:
            return obj

        pl = get_polars()

        if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
            return self.cast_polars_tabular(obj)
        elif isinstance(obj, (pl.Series, pl.Expr)):
            return self.cast_polars_series(obj)
        else:
            raise TypeError(f"Cannot cast {type(obj)} to polars")

    def cast_polars_tabular(
        self,
        table: "polars.DataFrame | polars.LazyFrame"
    ):
        if self.target_field is None:
            return table
        return self.target_field.cast_polars_tabular(table, options=self)

    def cast_polars_series(
        self,
        table: "polars.Series | polars.Expr"
    ):
        if self.target_field is None:
            return table
        return self.target_field.cast_polars_series(table, options=self)

    def cast_spark(
        self,
        obj: "ps.DataFrame | ps.Column"
    ):
        if self.target_field is None:
            return obj

        spark = get_spark_sql()

        if isinstance(obj, spark.DataFrame):
            return self.cast_spark_tabular(obj)
        elif isinstance(obj, spark.Column):
            return self.cast_spark_column(obj)
        else:
            raise TypeError(f"Cannot cast {type(obj)} to spark")

    def cast_spark_column(
        self,
        col: "ps.Column"
    ):
        if self.target_field is None:
            return col
        return self.target_field.cast_spark_column(col, options=self)

    def cast_spark_tabular(
        self,
        table: "ps.DataFrame"
    ):
        if self.target_field is None:
            return table
        return self.target_field.cast_spark_tabular(table, options=self)

    def fill_arrow_nulls(
        self,
        obj: Union[
            pa.Array, pa.ChunkedArray, pa.Table, pa.RecordBatch,
        ]
    ):
        if self.target_field is None:
            return obj
        return self.target_field.fill_arrow_array_nulls(obj)

    def need_cast(
        self,
        check_names: bool = True,
        check_dtypes: bool = True,
        check_metadata: bool = True
    ):
        if self.target_field is None:
            return False

        if self.source_field is None:
            return True

        return not self.source_field.dtype.equals(
            self.target_field.dtype,
            check_names=check_names, check_dtypes=check_dtypes,
            check_metadata=check_metadata
        )

    def polars_alias(self, series: "polars.Series | polars.Expr"):
        if self.target_field is None:
            return series

        target_name = self.target_field.name

        if target_name:
            if hasattr(series, "name"):
                return series.alias(target_name) if target_name != series.name else series
            else:
                return series.alias(target_name)
        return series

    def spark_alias(self, column: "ps.Column") -> "ps.Column":
        if self.target_field is None:
            return column

        target_name = self.target_field.name

        if target_name:
            if hasattr(column, "name"):
                return column.alias(target_name) if target_name != column.name else column
            else:
                return column.alias(target_name)
        return column