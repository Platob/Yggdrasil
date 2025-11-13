from __future__ import annotations

import dataclasses
from typing import ClassVar, TYPE_CHECKING, Any, Union, Optional, List, Dict, Tuple

# Make Polars the primary dependency
import polars as pl

# Import logging
from ..logging import get_logger

# Create module-level logger
logger = get_logger(__name__)

# Conditionally import pyarrow for interoperability
try:
    import pyarrow as pa
    import pyarrow.compute as pc
    HAS_ARROW = True
    logger.info("PyArrow is available, enabling Arrow interoperability")
except ImportError:
    HAS_ARROW = False
    logger.warning("PyArrow not available, Arrow interoperability disabled")
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
            # String lookup is not implemented yet
            raise ValueError(f"String lookup not implemented yet. Received: {dtype}")
        return dtype

    @staticmethod
    def create_series(name: str, dtype: DataType, nullable: bool = True) -> pl.Series:
        """Create an empty series with the given name and type."""
        return pl.Series(name=name, values=[], dtype=dtype)

    @staticmethod
    def is_nested_type(dtype: DataType) -> bool:
        """Check if the data type is a nested type (Struct, List, etc.)."""
        nested_types = {pl.Struct, pl.List, pl.Array}
        # Check if it's a struct type
        if hasattr(dtype, "fields"):
            return True
        # Handle List types
        if isinstance(dtype, type) and dtype in nested_types:
            return True
        # For parametrized list types
        if hasattr(dtype, "dtype") and hasattr(dtype, "__class__") and dtype.__class__.__name__ in ("List", "Array"):
            return True
        # For dictionary/map types
        if hasattr(dtype, "key_type") and hasattr(dtype, "value_type"):
            return True
        return False

    @staticmethod
    def get_inner_type(dtype: DataType) -> Optional[DataType]:
        """Extract the inner type from a list or array type."""
        if hasattr(dtype, "dtype"):  # For parametrized list types
            return dtype.dtype
        return None

    @staticmethod
    def get_nested_fields(dtype: DataType) -> List[Tuple[str, DataType]]:
        """Get nested fields for complex types like Struct, List, Array, and Map.

        Returns:
            List of (name, type) tuples representing the nested fields.
        """
        # Handle Struct types with explicit fields
        if hasattr(dtype, "fields"):
            return [(field.name, field.dtype) for field in dtype.fields]

        # Handle List types (returns inner type with "_inner" name)
        inner_type = DataUtility.get_inner_type(dtype)
        if inner_type is not None:
            return [("_inner", inner_type)]

        # Handle Map/Dictionary types
        if hasattr(dtype, "key_type") and hasattr(dtype, "value_type"):
            return [
                ("_key", dtype.key_type),
                ("_value", dtype.value_type)
            ]

        # Not a nested type or unrecognized nested type
        return []

    @staticmethod
    def can_convert_types(
        source_dtype: DataType,
        target_dtype: DataType,
        safe: bool = False,
        check_names: bool = False,
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

        # Handle nested types
        source_nested = DataUtility.is_nested_type(source_dtype)
        target_nested = DataUtility.is_nested_type(target_dtype)

        if source_nested and target_nested:
            # Both are nested types, compare their structures
            source_fields = DataUtility.get_nested_fields(source_dtype)
            target_fields = DataUtility.get_nested_fields(target_dtype)

            # Basic structure check: same number of fields
            if len(source_fields) != len(target_fields):
                return False

            # Check if fields can be converted
            for (s_name, s_type), (t_name, t_type) in zip(source_fields, target_fields):
                # Check field names if required
                if check_names and s_name != t_name and not s_name.startswith("_") and not t_name.startswith("_"):
                    return False

                # Check if field types are compatible
                if not DataUtility.can_convert_types(s_type, t_type, safe=safe, check_names=check_names):
                    return False

            # All fields passed the checks
            return True

        # One is nested, one is not
        if source_nested != target_nested:
            return False

        # Handle primitive type conversions in safe mode
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

            # Boolean type conversions
            if source_dtype == pl.Boolean and target_dtype in {pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                                             pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}:
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

        logger.debug(
            f"Casting series from {getattr(series, 'dtype', 'unknown')} to {target}"
        )

        # Handle Arrow arrays by converting to Polars first
        if HAS_ARROW and isinstance(series, (pa.Array, pa.ChunkedArray)):
            logger.debug("Converting Arrow array to Polars Series")
            series = pl.Series.from_arrow(series)

        # Ensure we have a Polars Series
        if not isinstance(series, pl.Series):
            logger.error(f"Expected pl.Series or arrow array, got {type(series)}")
            raise TypeError(f"Expected pl.Series or arrow array, got {type(series)}")

        # Keep original name if series has one, otherwise use target name
        name = series.name if series.name else self.target_name

        # Perform the cast
        logger.debug(f"Performing cast to {target} for series '{name}'")
        return series.cast(target)

    def cast_scalar(
        self,
        value: Any,
        target_dtype: Optional[DataType] = None,
        safe: Optional[bool] = None,
    ) -> Any:
        """Cast a scalar value to the target data type."""
        target = target_dtype or self.target_dtype

        logger.debug(
            f"Casting scalar value {value} from {self.source_dtype} to {target}"
        )

        # Create a temporary series, cast it, and extract the scalar
        temp_series = pl.Series(self.source_name, [value], dtype=self.source_dtype)
        cast_series = self.cast_series(temp_series, target_dtype=target, safe=safe)

        # Return the scalar value
        result = cast_series[0]
        logger.debug(f"Cast result: {result}")
        return result


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
        logger.debug(f"Looking up caster for {source_dtype} -> {target_dtype}")

        if not DataUtility.can_convert_types(source_dtype, target_dtype, safe=safe):
            safe_str = " safely" if safe else ""
            error_msg = f"Cannot convert {source_dtype} to {target_dtype}{safe_str}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        cached = self.get(source_dtype, target_dtype)
        if cached is not None:
            logger.debug(f"Using cached caster for {source_dtype} -> {target_dtype}")
            return cached

        logger.info(f"Creating new caster for {source_dtype} -> {target_dtype}")
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
            logger.debug("Creating singleton DataCastRegistry instance")
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