from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from yggdrasil.data.types.id import DataTypeId
from yggdrasil.io import SaveMode
from ..base import DataType
from ..support import get_polars, get_spark_sql

if TYPE_CHECKING:
    import polars
    import pyspark.sql as ps
    import pyspark.sql.types as pst
    from yggdrasil.data.cast.options import CastOptions
    from yggdrasil.data.data_field import Field


__all__ = ["ObjectType"]


@dataclass(frozen=True)
class ObjectType(DataType):
    """Variant type — represents an opaque Python object with no specific schema.

    This is the catch-all type for values that don't map cleanly to any
    structured or primitive type.  Think of it as "keep whatever you have,
    don't touch it."

    ObjectType intentionally bypasses dtype casts: when it's the target type,
    cast operations return the input unchanged.  This makes it safe to use as
    a variant column in schemas where you want to preserve heterogeneous data
    without forcing a conversion.

    Engine mapping:
    - Polars: ``pl.Object``
    - pandas: ``object`` dtype
    - Arrow: ``large_binary()`` (physical stand-in; no native variant type)
    - Spark: ``BinaryType()`` (physical stand-in)
    """

    # ------------------------------------------------------------------
    # DataType protocol
    # ------------------------------------------------------------------
    @property
    def type_id(self) -> DataTypeId:
        return DataTypeId.OBJECT

    @property
    def children_fields(self) -> list[Field]:
        return []

    # ------------------------------------------------------------------
    # Arrow
    # ------------------------------------------------------------------
    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        # Arrow has no native object/variant type.  ObjectType can never be
        # inferred from an Arrow type alone — it must be created explicitly.
        return False

    @classmethod
    def from_arrow_type(cls, dtype: pa.DataType) -> ObjectType:
        raise TypeError(
            f"Cannot infer ObjectType from Arrow type {dtype!r}. "
            "Arrow has no native object type. Create ObjectType() explicitly, "
            "or use DataType.from_arrow_type() for standard Arrow types."
        )

    def to_arrow(self) -> pa.DataType:
        # large_binary is the most general physical representation for
        # opaque Python objects (pickled bytes, arbitrary blobs, etc.).
        return pa.large_binary()

    # ------------------------------------------------------------------
    # Polars
    # ------------------------------------------------------------------
    @classmethod
    def handles_polars_type(cls, dtype: polars.DataType) -> bool:
        pl = get_polars()
        return dtype == pl.Object

    @classmethod
    def from_polars_type(cls, dtype: polars.DataType) -> ObjectType:
        if not cls.handles_polars_type(dtype):
            raise TypeError(
                f"Expected Polars Object dtype, got {dtype!r}. "
                "Use DataType.from_polars_type() for standard Polars types."
            )
        return cls()

    def to_polars(self) -> polars.DataType:
        pl = get_polars()
        return pl.Object

    # ------------------------------------------------------------------
    # Spark
    # ------------------------------------------------------------------
    @classmethod
    def handles_spark_type(cls, dtype: pst.DataType) -> bool:
        # Spark has no native object/variant type.
        return False

    @classmethod
    def from_spark_type(cls, dtype: pst.DataType) -> ObjectType:
        raise TypeError(
            f"Cannot infer ObjectType from Spark type {dtype!r}. "
            "Spark has no native object type. Create ObjectType() explicitly."
        )

    def to_spark(self) -> Any:
        # Binary is the safest physical stand-in for opaque objects in Spark.
        spark = get_spark_sql()
        return spark.types.BinaryType()

    # ------------------------------------------------------------------
    # Databricks DDL
    # ------------------------------------------------------------------
    def to_databricks_ddl(self) -> str:
        return "BINARY"

    # ------------------------------------------------------------------
    # Dict round-trip
    # ------------------------------------------------------------------
    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        type_id = value.get("id")
        if type_id == int(DataTypeId.OBJECT):
            return True
        name = str(value.get("name", "")).upper()
        return name == "OBJECT"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ObjectType:
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": int(DataTypeId.OBJECT),
            "name": DataTypeId.OBJECT.name,
        }

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    def _merge_with_same_id(
        self,
        other: DataType,
        mode: SaveMode | None = None,
        downcast: bool = False,
        upcast: bool = False,
    ) -> ObjectType:
        # Two ObjectTypes are interchangeable — keep self.
        return self

    # ------------------------------------------------------------------
    # Cast bypass — ObjectType is a variant, so skip dtype casts entirely
    # ------------------------------------------------------------------
    def _cast_arrow_array(
        self,
        array: pa.Array,
        options: CastOptions,
    ) -> pa.Array:
        # Variant type: return the array as-is, no conversion.
        return array

    def _cast_chunked_array(
        self,
        array: pa.ChunkedArray,
        options: CastOptions,
    ) -> pa.ChunkedArray:
        return array

    def _cast_polars_series(
        self,
        series: polars.Series,
        options: CastOptions,
    ):
        return series

    def _cast_polars_expr(
        self,
        expr: polars.Expr,
        options: CastOptions,
    ):
        return expr

    def _cast_pandas_series(
        self,
        series: Any,
        options: CastOptions,
    ):
        return series

    def _cast_spark_column(
        self,
        column: ps.Column,
        options: CastOptions,
    ):
        return column

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

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return "ObjectType()"

    def __str__(self) -> str:
        return "object"
