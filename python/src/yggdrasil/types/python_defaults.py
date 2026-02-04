"""Default value helpers for Python and Arrow types."""

import dataclasses
import datetime
import decimal
import inspect
import types
import uuid
from collections.abc import Collection, Mapping, MutableMapping, MutableSequence, MutableSet
from typing import Any, Tuple, Union, get_args, get_origin, Optional, List

import pyarrow as pa

__all__ = [
    "default_scalar",
    "default_python_scalar",
    "default_arrow_scalar",
    "default_arrow_array"
]

DEFAULT_MAPS_INITIALIZED = False

_NONE_TYPE = type(None)
_ARROW_DEFAULTS = {}
_POLARS_DEFAULTS = {}
_PRIMITIVE_DEFAULTS = {}
_SPECIAL_DEFAULTS = {}


def ensure_default_maps_initialized():
    global DEFAULT_MAPS_INITIALIZED
    global _PRIMITIVE_DEFAULTS
    global _SPECIAL_DEFAULTS
    global _ARROW_DEFAULTS
    global _POLARS_DEFAULTS

    if not DEFAULT_MAPS_INITIALIZED:
        _PRIMITIVE_DEFAULTS = {
            str: "",
            int: 0,
            float: 0.0,
            bool: False,
            bytes: b"",
        }

        _SPECIAL_DEFAULTS = {
            datetime.datetime: lambda: datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc),
            datetime.date: lambda: datetime.date(1970, 1, 1),
            datetime.time: lambda: datetime.time(0, 0, 0, tzinfo=datetime.timezone.utc),
            datetime.timedelta: lambda: datetime.timedelta(0),
            uuid.UUID: lambda: uuid.UUID(int=0),
            decimal.Decimal: lambda: decimal.Decimal(0),
        }

        _ARROW_DEFAULTS = {
            pa.null(): pa.scalar(None, type=pa.null()),

            pa.bool_(): pa.scalar(False, type=pa.bool_()),

            pa.int8(): pa.scalar(0, type=pa.int8()),
            pa.int16(): pa.scalar(0, type=pa.int16()),
            pa.int32(): pa.scalar(0, type=pa.int32()),
            pa.int64(): pa.scalar(0, type=pa.int64()),

            pa.uint8(): pa.scalar(0, type=pa.uint8()),
            pa.uint16(): pa.scalar(0, type=pa.uint16()),
            pa.uint32(): pa.scalar(0, type=pa.uint32()),
            pa.uint64(): pa.scalar(0, type=pa.uint64()),

            # pa.float16(): pa.scalar(0.0, type=pa.float16()),
            pa.float32(): pa.scalar(0.0, type=pa.float32()),
            pa.float64(): pa.scalar(0.0, type=pa.float64()),

            pa.string(): pa.scalar("", type=pa.string()),
            pa.string_view(): pa.scalar("", type=pa.string_view()),
            pa.large_string(): pa.scalar("", type=pa.large_string()),

            pa.binary(): pa.scalar(b"", type=pa.binary()),
            pa.binary_view(): pa.scalar(b"", type=pa.binary_view()),
            pa.large_binary(): pa.scalar(b"", type=pa.large_binary()),
        }

        try:
            import polars

            _POLARS_DEFAULTS = {
                polars.Null(): None,
                polars.Boolean(): False,

                polars.Binary(): b"",

                polars.Utf8(): "",

                polars.Int8(): 0,
                polars.Int16(): 0,
                polars.Int32(): 0,
                polars.Int64(): 0,

                polars.UInt8(): 0,
                polars.UInt16(): 0,
                polars.UInt32(): 0,
                polars.UInt64(): 0,

                polars.Float32(): 0.0,
                polars.Float64(): 0.0,
            }
        except ImportError:
            pass

        DEFAULT_MAPS_INITIALIZED = True

def _is_optional(hint) -> bool:
    """Return True when the type hint is Optional.

    Args:
        hint: Type hint to inspect.

    Returns:
        True if Optional.
    """
    origin = get_origin(hint)

    if origin in (Union, types.UnionType):
        return _NONE_TYPE in get_args(hint)

    return False


def _default_for_collection(origin):
    """Return default values for collection-like origins.

    Args:
        origin: Collection origin type.

    Returns:
        Default collection instance or None.
    """
    if origin in (list, MutableSequence):
        return []

    if origin in (set, MutableSet):
        return set()

    if origin in (dict, MutableMapping, Mapping):
        return {}

    if origin in (tuple, Tuple):
        return tuple()

    if origin and issubclass(origin, Collection):
        return origin()

    return None


def _default_for_tuple_args(args):
    """Return a default tuple based on element hints.

    Args:
        args: Tuple element type hints.

    Returns:
        Default tuple instance.
    """
    if not args:
        return tuple()

    if len(args) == 2 and args[1] is Ellipsis:
        return tuple()

    return tuple(default_scalar(arg) for arg in args)


def _default_for_dataclass(hint):
    """Return a default instance for a dataclass type.

    Args:
        hint: Dataclass type.

    Returns:
        Dataclass instance with default values.
    """
    kwargs = {}

    for field in dataclasses.fields(hint):
        if not field.init or field.name.startswith("_"):
            continue

        if field.default is not dataclasses.MISSING:
            value = field.default
        elif field.default_factory is not dataclasses.MISSING:  # type: ignore[attr-defined]
            value = field.default_factory()  # type: ignore[misc]
        else:
            value = default_scalar(field.type)

        kwargs[field.name] = value

    return hint(**kwargs)


def default_arrow_scalar(
    dtype: Union[pa.DataType, pa.ListType, pa.MapType, pa.StructType, pa.FixedSizeListType],
    nullable: bool
):
    """Return a default scalar for a given Arrow type.

    Args:
        dtype: Arrow data type.
        nullable: Whether the scalar should be nullable.

    Returns:
        Arrow scalar default.
    """
    ensure_default_maps_initialized()

    if nullable:
        return pa.scalar(None, type=dtype)

    existing = _ARROW_DEFAULTS.get(dtype)

    if existing is not None:
        return existing

    if (
        pa.types.is_timestamp(dtype)
        or pa.types.is_time(dtype)
        or pa.types.is_duration(dtype)
        or pa.types.is_date(dtype)
    ):
        return pa.scalar(0, type=dtype)
    elif pa.types.is_decimal(dtype):
        return pa.scalar(decimal.Decimal(0), type=dtype)
    elif pa.types.is_fixed_size_binary(dtype):
        return pa.scalar(b"\x00" * dtype.byte_width, type=dtype)
    elif pa.types.is_struct(dtype):
        fields = [
            (field.name, default_arrow_scalar(dtype=field.type, nullable=field.nullable))
            for field in dtype
        ]
        return pa.scalar(fields, type=dtype)
    elif (
        pa.types.is_list(dtype)
        or pa.types.is_large_list(dtype)
        or pa.types.is_list_view(dtype)
    ):
        return pa.scalar([], type=dtype)
    elif pa.types.is_fixed_size_list(dtype):
        value_field: pa.Field = dtype.value_field

        return pa.scalar(
            [default_arrow_scalar(value_field.type, value_field.nullable)] * dtype.list_size,
            type=dtype
        )
    elif pa.types.is_map(dtype):
        return pa.scalar({}, type=dtype)
    else:
        raise TypeError(f"Cannot determine default value for Arrow type {dtype!r}")


def default_arrow_array(
    dtype: Union[pa.DataType, pa.ListType, pa.MapType, pa.StructType],
    nullable: bool,
    size: int = 0,
    memory_pool: Optional[pa.MemoryPool] = None,
    chunks: Optional[List[int]] = None,
    scalar_default: Optional[pa.Scalar] = None,
) -> Union[pa.Array, pa.ChunkedArray]:
    """Return a default Arrow array or chunked array for a given type.

    Args:
        dtype: Arrow data type.
        nullable: Whether values are nullable.
        size: Number of elements.
        memory_pool: Optional Arrow memory pool.
        chunks: Optional chunk sizes.
        scalar_default: Optional scalar default override.

    Returns:
        Arrow array or chunked array.
    """
    if scalar_default is None:
        scalar_default = default_arrow_scalar(dtype=dtype, nullable=nullable)

    # ✅ PyArrow compatibility: repeat(size=0) can throw "Must pass at least one array"
    if size == 0 and (chunks is None):
        return pa.array([], type=dtype)

    if chunks is not None:
        if len(chunks) == 0:
            # also avoid "must pass at least one array" from chunked_array([])
            return pa.chunked_array([], type=dtype)

        return pa.chunked_array(
            [
                pa.repeat(
                    value=scalar_default,
                    size=chunk_size,
                    memory_pool=memory_pool
                ) if chunk_size > 0 else pa.array([], type=dtype)
                for chunk_size in chunks
            ],
            type=dtype,
        )

    return pa.repeat(
        value=scalar_default,
        size=size,
        memory_pool=memory_pool
    )


def default_python_scalar(hint: Any):
    """Return a default Python value for the given type hint.

    Args:
        hint: Type hint to generate defaults for.

    Returns:
        Default Python value.
    """
    if _is_optional(hint):
        return None

    ensure_default_maps_initialized()

    if hint in _PRIMITIVE_DEFAULTS:
        return _PRIMITIVE_DEFAULTS[hint]

    if hint in _SPECIAL_DEFAULTS:
        return _SPECIAL_DEFAULTS[hint]()

    origin = get_origin(hint)

    if origin is None and not inspect.isclass(hint):
        from .cast import convert

        arrow_field: pa.Field = convert(hint, pa.Field)
        arrow_scalar = default_arrow_scalar(dtype=arrow_field.type, nullable=arrow_field.nullable)

        return arrow_scalar.as_py()

    if hint in (list, set, dict, tuple):
        origin = hint

    if origin:
        if origin in (tuple, Tuple):
            return _default_for_tuple_args(get_args(hint))

        collection_default = _default_for_collection(origin)
        if collection_default is not None:
            return collection_default

    if dataclasses.is_dataclass(hint):
        return _default_for_dataclass(hint)

    try:
        return hint()
    except Exception as exc:
        raise TypeError(f"Cannot determine default value for {hint!r}") from exc


def default_scalar(
    hint: Union[
        type,
        pa.DataType, pa.Field,
    ],
    nullable: Optional[bool] = None
):
    """Return a default scalar value for Python or Arrow type hints.

    Args:
        hint: Python type or Arrow type/field.
        nullable: Override nullability for Arrow types.

    Returns:
        Default scalar value.
    """
    if isinstance(hint, pa.Field):
        nullable = hint.nullable if nullable is None else nullable
        return default_arrow_scalar(dtype=hint.type, nullable=nullable)
    if isinstance(hint, pa.DataType):
        return default_arrow_scalar(dtype=hint, nullable=nullable)
    return default_python_scalar(hint)
