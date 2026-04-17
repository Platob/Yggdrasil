from typing import TYPE_CHECKING, ClassVar, Any

import pyarrow as pa

if TYPE_CHECKING:
    import polars
    from .base import DataType
    import pyspark.sql.types as pst

__all__ = [
    "DataTypes"
]


class DataTypes:
    arrow_cache: ClassVar[dict[pa.DataType, "DataType"]] = {}
    polars_cache: ClassVar[dict["polars.DataType", "DataType"]] = {}
    spark_cache: ClassVar[dict["pst.DataType", "DataType"]] = {}

    @classmethod
    def register(cls, dtype: "DataType") -> None:
        cls.arrow_register(dtype)
        cls.polars_register(dtype)
        cls.spark_register(dtype)

    @classmethod
    def arrow_register(cls, dtype: "DataType") -> None:
        cls.arrow_cache[dtype.to_arrow()] = dtype

    @classmethod
    def arrow_get(cls, dtype: pa.DataType) -> type[Any] | None:
        return cls.arrow_cache.get(dtype)

    @classmethod
    def polars_register(cls, dtype: "DataType") -> None:
        cls.polars_cache[dtype.to_polars()] = dtype

    @classmethod
    def polars_get(cls, dtype: "polars.DataType") -> type[Any] | None:
        return cls.polars_cache.get(dtype)

    @classmethod
    def spark_get(cls, dtype: "pst.DataType") -> type[Any] | None:
        return cls.spark_cache.get(dtype)

    @classmethod
    def spark_register(cls, dtype: "DataType") -> None:
        cls.spark_cache[dtype.to_spark()] = dtype