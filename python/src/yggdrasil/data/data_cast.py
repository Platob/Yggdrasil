from __future__ import annotations

import dataclasses
from typing import ClassVar, TYPE_CHECKING, Any, Union, Optional, List, Dict, Tuple

# Make Polars the primary dependency
import polars as pl

# Conditionally import pyarrow for interoperability
try:
    import pyarrow as pa
    import pyarrow.compute as pc
    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False
    pa = Any  # type: ignore
    pc = Any  # type: ignore

# Define type aliases
DataType = pl.DataType
SeriesLike = Union[pl.Series, "pa.Array", "pa.ChunkedArray"] if HAS_ARROW else pl.Series


class DataUtility:
    """Utility class for data type operations."""

    @staticmethod
    def ensure_polars_type(dtype: DataType | str) -> DataType:
        """Convert string representation to Polars DataType if needed."""
        if isinstance(dtype, str):
            # Convert string to polars data type
            try:
                return getattr(pl, dtype)
            except AttributeError:
                raise ValueError(f"Invalid Polars data type: {dtype}")
        return dtype

    @staticmethod
    def create_series(name: str, dtype: DataType, nullable: bool = True) -> pl.Series:
        """Create an empty series with the given name and type."""
        return pl.Series(name=name, values=[], dtype=dtype)

    @staticmethod
    def get_nested_fields(dtype: DataType) -> List[Tuple[str, DataType]]:
        """Get nested fields for complex types like Struct."""
        if dtype == pl.Struct:
            # For struct types, return field definitions
            # This is a placeholder - actual implementation would depend on the struct definition
            return []
        elif dtype in (pl.List, pl.Array):
            # For list types, we'd need the inner type
            # This is a placeholder
            return []

        # Not a nested type
        return []

    @staticmethod
    def can_convert_types(
        source_dtype: DataType,
        target_dtype: DataType,
        safe: bool = False,
    ) -> bool:
        """Check if source type can be cast to target type."""
        # Same type is always convertible
        if source_dtype == target_dtype:
            return True

        # Safe numeric conversions
        numeric_types = (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                         pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                         pl.Float32, pl.Float64)

        # Type categories for safe conversions
        int_types = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
        float_types = (pl.Float32, pl.Float64)
        string_types = (pl.Utf8, pl.String)

        if safe:
            # Integer to integer (same signedness or widening)
            if source_dtype in int_types and target_dtype in int_types:
                s_bits = int(''.join(filter(str.isdigit, str(source_dtype))))
                t_bits = int(''.join(filter(str.isdigit, str(target_dtype))))
                s_signed = 'UInt' not in str(source_dtype)
                t_signed = 'UInt' not in str(target_dtype)

                # Same signedness and target has more bits
                if s_signed == t_signed and t_bits >= s_bits:
                    return True
                # Source unsigned, target signed and has more bits
                if not s_signed and t_signed and t_bits > s_bits:
                    return True
                return False

            # Float to float (widening)
            if source_dtype in float_types and target_dtype in float_types:
                return str(target_dtype).endswith('64') or str(source_dtype).endswith('32')

            # String types
            if source_dtype in string_types and target_dtype in string_types:
                return True

            # Any numeric to float64 is safe
            if source_dtype in numeric_types and target_dtype == pl.Float64:
                return True

            return False

        # In non-safe mode, most conversions are allowed with potential data loss
        return True


@dataclasses.dataclass(frozen=True)
class DataCaster:
    """Cast data between different types."""

    source_dtype: DataType
    target_dtype: DataType
    source_name: str = "value"
    target_name: str = "value"

    def cast_series(
        self,
        series: SeriesLike,
        target_dtype: Optional[DataType] = None,
        safe: Optional[bool] = None,
    ) -> pl.Series:
        """Cast a series to the target data type."""
        target = target_dtype or self.target_dtype

        # Handle Arrow arrays by converting to Polars first
        if HAS_ARROW and isinstance(series, (pa.Array, pa.ChunkedArray)):
            series = pl.Series.from_arrow(series)

        # Ensure we have a Polars Series
        if not isinstance(series, pl.Series):
            raise TypeError(f"Expected pl.Series or arrow array, got {type(series)}")

        # Keep original name if series has one, otherwise use target name
        name = series.name if series.name else self.target_name

        # Perform the cast
        return series.cast(target)

    def cast_scalar(
        self,
        value: Any,
        target_dtype: Optional[DataType] = None,
        safe: Optional[bool] = None,
    ) -> Any:
        """Cast a scalar value to the target data type."""
        target = target_dtype or self.target_dtype

        # Create a temporary series, cast it, and extract the scalar
        temp_series = pl.Series(self.source_name, [value], dtype=self.source_dtype)
        cast_series = self.cast_series(temp_series, target_dtype=target, safe=safe)

        # Return the scalar value
        return cast_series[0]


@dataclasses.dataclass
class DataCastRegistry:
    """Registry for data casters."""

    cache: Dict[Tuple[str, str], DataCaster] = dataclasses.field(default_factory=dict)
    _instance: ClassVar[DataCastRegistry | None] = None

    @staticmethod
    def inner_key(source_dtype: DataType, target_dtype: DataType) -> Tuple[str, str]:
        """Create a cache key for a type conversion."""
        return (str(source_dtype), str(target_dtype))

    def register(self, caster: DataCaster) -> DataCaster:
        """Register a caster in the registry."""
        key = self.inner_key(caster.source_dtype, caster.target_dtype)
        self.cache[key] = caster
        return caster

    def get(
        self,
        source_dtype: DataType,
        target_dtype: DataType,
    ) -> Optional[DataCaster]:
        """Get an existing caster from the registry."""
        key = self.inner_key(source_dtype, target_dtype)
        return self.cache.get(key)

    def get_or_build(
        self,
        source_dtype: DataType,
        target_dtype: DataType,
        source_name: str = "value",
        target_name: str = "value",
        safe: bool = False,
    ) -> DataCaster:
        """Get an existing caster or create a new one."""
        if not DataUtility.can_convert_types(source_dtype, target_dtype, safe=safe):
            safe_str = " safely" if safe else ""
            raise ValueError(f"Cannot convert {source_dtype} to {target_dtype}{safe_str}")

        cached = self.get(source_dtype, target_dtype)
        if cached is not None:
            return cached

        caster = DataCaster(
            source_dtype=source_dtype,
            target_dtype=target_dtype,
            source_name=source_name,
            target_name=target_name,
        )
        return self.register(caster)

    @classmethod
    def instance(cls) -> DataCastRegistry:
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# Create the global registry instance
DATA_CAST_REGISTRY = DataCastRegistry.instance()


__all__ = [
    "DataCaster",
    "DataCastRegistry",
    "DataUtility",
    "DATA_CAST_REGISTRY",
    "DataType",
    "SeriesLike",
    "HAS_ARROW",
]