from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from yggdrasil.data.enums import Mode

from ..base import DataType
from ..id import DataTypeId
from yggdrasil.lazy_imports import polars_module, spark_sql_module

if TYPE_CHECKING:
    import polars
    import pyspark.sql.types as pst
    from ...cast.options import CastOptions
    from ...data_field import Field


__all__ = ["ObjectType"]


@dataclass(frozen=True)
class ObjectType(DataType):
    """Variant type — opaque Python object with no fixed schema.

    The catch-all for values that don't map cleanly to any structured or
    primitive type. Cast operations against ObjectType are no-ops: the
    input passes through untouched, which is what you want for a variant
    column carrying heterogeneous data.

    Engine mapping:

    * Polars: ``pl.Object``
    * pandas: ``object`` dtype
    * Arrow:  ``large_binary()`` (physical stand-in; Arrow has no variant type)
    * Spark:  ``BinaryType()`` (physical stand-in)
    """

    # Does not inherit PrimitiveType because it has no ``byte_size``
    # concept and doesn't participate in the primitive merge matrix.

    @classmethod
    def class_type_id(cls) -> DataTypeId:
        return DataTypeId.OBJECT

    def pretty_format(self, indent: int = 2, level: int = 0) -> str:
        pad = " " * (indent * level)
        return f"{pad}object"

    @property
    def children(self) -> list["Field"]:
        return []

    # ------------------------------------------------------------------
    # Engine probes / constructors
    # ------------------------------------------------------------------

    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        return False

    @classmethod
    def from_arrow_type(cls, dtype: pa.DataType) -> "ObjectType":
        raise TypeError(
            f"Cannot infer ObjectType from Arrow type {dtype!r}. "
            "Arrow has no native object type. Create ObjectType() explicitly, "
            "or use DataType.from_arrow_type() for standard Arrow types."
        )

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        pl = polars_module()
        return dtype == pl.Object

    @classmethod
    def from_polars_type(cls, dtype: "polars.DataType") -> "ObjectType":
        if not cls.handles_polars_type(dtype):
            raise TypeError(
                f"Expected Polars Object dtype, got {dtype!r}. "
                "Use DataType.from_polars_type() for standard Polars types."
            )
        return cls()

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        return False

    @classmethod
    def from_spark_type(cls, dtype: "pst.DataType") -> "ObjectType":
        raise TypeError(
            f"Cannot infer ObjectType from Spark type {dtype!r}. "
            "Spark has no native object type. Create ObjectType() explicitly."
        )

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        type_id = value.get("id")
        if type_id == int(DataTypeId.OBJECT):
            return True
        name = str(value.get("name", "")).upper()
        return name == "OBJECT"

    @classmethod
    def from_dict(cls, value: dict[str, Any], default: Any = ...) -> "ObjectType":
        try:
            return cls()
        except Exception as e:
            if default is ...:
                raise e
            return default

    # ------------------------------------------------------------------
    # Exporters
    # ------------------------------------------------------------------

    def _default_pyhint(self) -> Any:
        # Variant column — ``object`` is the closest Python annotation
        # for "anything goes". Original parsed hints (user classes,
        # ``Any``) live on the ``_pyhint_cache`` stamp.
        return object

    def to_arrow(self) -> pa.DataType:
        return pa.large_binary()

    def to_polars(self) -> "polars.DataType":
        pl = polars_module()
        return pl.Object

    def to_spark(self) -> Any:
        spark = spark_sql_module()
        return spark.types.BinaryType()

    def to_spark_name(self) -> str:
        return "BINARY"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.type_id.value,
            "name": self.type_id.name,
        }

    # ------------------------------------------------------------------
    # Merge / cast — both are identity
    # ------------------------------------------------------------------

    def _merge_with_same_id(
        self,
        other: "DataType",
        mode: Mode | None = None,
        downcast: bool = False,
        upcast: bool = False,
    ) -> "ObjectType":
        return self

    # Variant target: never touch the values. The base class already
    # short-circuits on ``self.type_id == OBJECT`` in every ``cast_*``
    # entrypoint, so these overrides are strictly defensive — they
    # protect against a subclass bypassing that guard (tabular paths
    # inside a struct walk, for example).

    def _cast_arrow_array(self, array: pa.Array, options: "CastOptions") -> pa.Array:
        return array

    def _cast_chunked_array(
        self, array: pa.ChunkedArray, options: "CastOptions"
    ) -> pa.ChunkedArray:
        return array

    def _cast_arrow_tabular(
        self,
        table: pa.Table | pa.RecordBatch,
        options: "CastOptions",
    ):
        return table

    def _cast_polars_series(self, series: "polars.Series", options: "CastOptions"):
        return series

    def _cast_polars_expr(self, expr: Any, options: "CastOptions"):
        return expr

    def _cast_polars_tabular(
        self,
        table: "polars.DataFrame | polars.LazyFrame",
        options: "CastOptions",
    ):
        return table

    def _cast_pandas_series(self, series: Any, options: "CastOptions"):
        return series

    def _cast_pandas_tabular(
        self,
        data: "pd.DataFrame",
        options: "CastOptions",
    ):
        return data

    def _cast_spark_column(self, column: Any, options: "CastOptions"):
        return column

    def _cast_spark_tabular(
        self,
        data: "ps.DataFrame",
        options: "CastOptions",
    ):
        return data

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    def default_pyobj(self, nullable: bool) -> Any:
        if nullable:
            return None
        raise NotImplementedError(
            "ObjectType.default_pyobj(nullable=False) is not supported. "
            "Object types represent opaque Python objects with no universal "
            "non-null default. Use nullable=True or convert to a more "
            "specific type first."
        )

    def default_arrow_scalar(self, nullable: bool = True) -> pa.Scalar:
        if nullable:
            return pa.scalar(None, type=pa.large_binary())
        raise NotImplementedError(
            "ObjectType.default_arrow_scalar(nullable=False) is not supported. "
            "Convert to a more specific type first."
        )
