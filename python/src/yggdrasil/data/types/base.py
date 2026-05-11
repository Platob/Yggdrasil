from __future__ import annotations

import datetime as dt
import decimal
import enum
import json
import logging
import types
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping, MutableMapping, Sequence, Set as AbstractSet
from dataclasses import fields, is_dataclass, dataclass
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Union,
    get_args,
    get_origin,
    AnyStr,
)

import pyarrow as pa
import pyarrow.compute as pc

from yggdrasil.data.constants import DEFAULT_FIELD_NAME
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.parser import (
    DataTypeMetadata,
    ParsedDataType,
    parse_data_type,
)
from yggdrasil.data.enums.mode import Mode
from yggdrasil.lazy_imports import field_class, polars_module, pandas_module
from yggdrasil.pickle.serde import ObjectSerde
from .support import get_pandas, get_polars, get_spark_sql
from ..base_meta import BaseChildrenFields
from ..data_utils import get_cast_options_class

if TYPE_CHECKING:
    import pandas as pd
    import polars
    import pyspark.sql as ps
    import pyspark.sql.types as pst
    from yggdrasil.data.options import CastOptions
    from yggdrasil.data.data_field import Field


__all__ = [
    "DataTypeId",
    "DataType",
    "PrimitiveType",
    "NullType",
    "BinaryType",
    "BJsonType",
    "SJsonType",
    "StringType",
    "BooleanType",
    "NumericType",
    "IntegerType",
    "Int8Type",
    "Int16Type",
    "Int32Type",
    "Int64Type",
    "UInt8Type",
    "UInt16Type",
    "UInt32Type",
    "UInt64Type",
    "FloatingPointType",
    "Float8Type",
    "Float16Type",
    "Float32Type",
    "Float64Type",
    "DecimalType",
    "TemporalType",
    "DateType",
    "TimeType",
    "TimestampType",
    "DurationType",
    "DictionaryType",
    "EnumType",
    "StrEnumType",
    "IntEnumType",
    "NestedType",
    "ArrayType",
    "MapType",
    "StructType",
]


# ======================================================================
# Module-level constants & private helpers
# ======================================================================

_NONE_TYPE = type(None)

# from_any dispatch table — module prefix → classmethod name. Order
# matters: first prefix match wins. ``yggdrasil`` is last because
# several non-yggdrasil objects carry a ``.dtype`` attribute that we'd
# otherwise shadow.
_FROM_ANY_NS_DISPATCH: tuple[tuple[str, str], ...] = (
    ("pyarrow", "from_arrow"),
    ("polars", "from_polars"),
    ("pandas", "from_pandas"),
    ("pyspark", "from_spark"),
    ("yggdrasil", "from_yggdrasil"),
)


def _safe_issubclass(obj: object, class_or_tuple: object) -> bool:
    if obj is class_or_tuple:
        return True
    if not isinstance(obj, type):
        return False
    # Parameterised generics (``tuple[int, ...]``, ``list[int]``,
    # PEP 585) override ``__class__`` to ``type`` so ``isinstance(g,
    # type)`` returns True, but ``issubclass`` then raises ``TypeError:
    # arg 1 must be a class`` because the generic itself isn't a real
    # class. Catch and treat as "not a subclass" — the caller's normal
    # path is to fall through to ``get_origin`` / ``get_args``.
    try:
        return issubclass(obj, class_or_tuple)  # type: ignore[arg-type]
    except TypeError:
        return False


def _strip_annotated(hint: object) -> object:
    while get_origin(hint) is Annotated:
        args = get_args(hint)
        hint = args[0] if args else Any
    return hint


def _unwrap_newtype(hint: object) -> object:
    while hasattr(hint, "__supertype__"):
        hint = hint.__supertype__
    return hint


def _is_typed_dict_type(hint: object) -> bool:
    return (
        isinstance(hint, type)
        and issubclass(hint, dict)
        and hasattr(hint, "__annotations__")
        and hasattr(hint, "__total__")
    )


def _is_namedtuple_type(hint: object) -> bool:
    return (
        isinstance(hint, type)
        and issubclass(hint, tuple)
        and hasattr(hint, "_fields")
        and hasattr(hint, "__annotations__")
    )


def _literal_values_to_hint(values: tuple[object, ...]) -> object:
    if not values:
        return Any

    types_seen: list[type] = []
    for value in values:
        current = _NONE_TYPE if value is None else type(value)
        if current not in types_seen:
            types_seen.append(current)

    if len(types_seen) == 1:
        return types_seen[0]

    non_null = [t for t in types_seen if t is not _NONE_TYPE]
    if len(non_null) == 1 and len(types_seen) == 2 and _NONE_TYPE in types_seen:
        return Optional[non_null[0]]

    return Union[tuple(types_seen)]


DATA_TYPE_CLASSES: dict[int, type["DataType"]] = {}


# ---------------------------------------------------------------------
# Specialized fixed-width integer / float type-id tables.
#
# Mirrors :data:`yggdrasil.data.types.primitive.numeric._SPECIALIZED_INTEGER_TYPES`
# but keyed by ``DataTypeId`` so ``from_parsed`` can recover ``(byte_size,
# signed)`` from a specialized id without re-scanning the canonical
# alias name. Generic ``INTEGER`` / ``FLOAT`` ids fall through to the
# byte_size hint already on the parsed metadata.
# ---------------------------------------------------------------------

_INT_TYPE_ID_TO_PARAMS: dict[DataTypeId, tuple[int, bool]] = {
    DataTypeId.INT8: (1, True),
    DataTypeId.INT16: (2, True),
    DataTypeId.INT32: (4, True),
    DataTypeId.INT64: (8, True),
    DataTypeId.UINT8: (1, False),
    DataTypeId.UINT16: (2, False),
    DataTypeId.UINT32: (4, False),
    DataTypeId.UINT64: (8, False),
}

_FLOAT_TYPE_ID_TO_SIZE: dict[DataTypeId, int] = {
    DataTypeId.FLOAT8: 1,
    DataTypeId.FLOAT16: 2,
    DataTypeId.FLOAT32: 4,
    DataTypeId.FLOAT64: 8,
}


@dataclass(frozen=True, repr=False)
class DataType(BaseChildrenFields, ABC):
    _singleton_instance: ClassVar["DataType | None"] = None

    # ==================================================================
    # Dunder / identity
    # ==================================================================

    def __str__(self):
        return self.pretty_format()

    def __repr__(self):
        return self.pretty_format()

    def __call__(self, *args, **kwargs):
        if not args and not kwargs:
            return self
        raise ValueError(f"Cannot call {self.__class__} with args or kwargs")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not cls.__subclasses__():
            try:
                type_id = cls.class_type_id()
                if type_id is not None:
                    DATA_TYPE_CLASSES[type_id.value] = cls
            except TypeError:
                pass

    def equals(
        self,
        other: "DataType",
        check_names: bool = True,
        check_dtypes: bool = True,
        check_metadata: bool = True,
    ) -> bool:
        return self == other

    @classmethod
    def class_type_id(cls) -> DataTypeId: ...

    @property
    def type_id(self) -> DataTypeId:
        return self.class_type_id()

    @classmethod
    def instance(cls) -> "DataType":
        if cls is DataType:
            raise TypeError("DataType.instance() must be called on a subclass")

        inst = cls._singleton_instance
        if inst is None:
            inst = cls()
            cls._singleton_instance = inst
        return inst

    # ==================================================================
    # Lifting — dtype → Field / StructType wrappers
    # ==================================================================

    def to_struct(self, name: str | None = None) -> "StructType":
        if self.type_id == DataTypeId.STRUCT:
            return self

        from .nested import StructType

        return StructType(fields=[self.to_field(name=name)])

    def to_field(
        self,
        name: str = DEFAULT_FIELD_NAME,
        nullable: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> "Field":
        return field_class()(
            name=name or DEFAULT_FIELD_NAME,
            dtype=self,
            nullable=nullable,
            metadata=metadata,
        )

    # ==================================================================
    # Merge — schema reconciliation
    # ==================================================================

    def merge_with(
        self,
        other: "DataType",
        mode: Mode | None = None,
        downcast: bool = False,
        upcast: bool = False,
    ):
        mode = Mode.from_(mode, Mode.UPSERT)

        if mode is Mode.IGNORE:
            return self

        if self.type_id == other.type_id:
            return self._merge_with_same_id(
                other=other,
                mode=mode,
                downcast=downcast,
                upcast=upcast,
            )

        return self._merge_with_different_id(
            other=other,
            mode=mode,
            downcast=downcast,
            upcast=upcast,
        )

    @abstractmethod
    def _merge_with_same_id(
        self,
        other: "DataType",
        mode: "Mode" = Mode.AUTO,
        downcast: bool = False,
        upcast: bool = False,
    ):
        raise NotImplementedError

    def _merge_with_different_id(
        self,
        other: "DataType",
        mode: "Mode" = Mode.AUTO,
        downcast: bool = False,
        upcast: bool = False,
    ):
        if mode in (Mode.APPEND, Mode.UPSERT, Mode.AUTO):
            if self.type_id.is_any_or_null:
                return other
            elif other.type_id.is_any_or_null:
                return self

        if downcast == upcast:
            return self

        if downcast:
            return self if self.type_id < other.type_id else other
        else:
            return self if self.type_id > other.type_id else other

    @abstractmethod
    def pretty_format(self, indent: int = 2, level: int = 0) -> str:
        """Pretty-print this dtype with one element per line.

        ``indent`` is the per-level step in spaces. ``level`` is the
        current depth — the line is prefixed with ``indent * level``
        spaces. Flat dtypes render as a single line; nested dtypes
        (struct / list / map) override this to lay each child out on its
        own line at ``level + 1``.
        """
        ...

    # ==================================================================
    # Classmethod plumbing — subclass dispatch helpers
    # ==================================================================

    @classmethod
    def _any_subclass_handles(cls, dtype: Any, handler: str, label: str) -> bool:
        subclasses = cls.__subclasses__()
        if not subclasses:
            raise TypeError(f"Unsupported {label} data type: {dtype!r}")
        return any(getattr(sub, handler)(dtype) for sub in subclasses)

    @staticmethod
    def _target_nullable(options: "CastOptions") -> bool:
        return (
            options.target_field.nullable if options.target_field is not None else True
        )

    @staticmethod
    def _matches_dict(
        value: dict[str, Any], type_id: DataTypeId, *aliases: str
    ) -> bool:
        if value.get("id") == int(type_id):
            return True
        name = str(value.get("name", "")).upper()
        return name == type_id.name or name in aliases

    # ==================================================================
    # Engine handlers — `handles_<engine>_type` subclass probes
    # ==================================================================

    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        return cls._any_subclass_handles(dtype, "handles_arrow_type", "Arrow")

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        return cls._any_subclass_handles(dtype, "handles_polars_type", "Polars")

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        return cls._any_subclass_handles(dtype, "handles_spark_type", "Spark")

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        for subclass in cls.__subclasses__():
            if subclass.handles_dict(value):
                return True

            if subclass.__subclasses__():
                for sc in subclass.__subclasses__():
                    if sc.handles_dict(value):
                        return True
        return False

    # ==================================================================
    # Exporters — dict / arrow / polars / spark / databricks DDL
    # ==================================================================

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.type_id.value,
            "name": self.type_id.name,
        }

    def to_json(self, to_bytes: bool = False) -> AnyStr:
        dumped = json.dumps(self.to_dict())
        if to_bytes:
            return dumped.encode("utf-8")
        return dumped

    @abstractmethod
    def to_arrow(self) -> pa.DataType:
        raise NotImplementedError

    @abstractmethod
    def to_polars(self) -> "polars.DataType":
        raise NotImplementedError

    @abstractmethod
    def to_spark(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def to_spark_name(self) -> str:
        raise NotImplementedError

    def as_polars(self) -> "DataType":
        """Return a Polars-flavored :class:`DataType` for this type.

        Same shape as :meth:`as_spark` — stays on the yggdrasil side
        of the boundary and returns a :class:`DataType` whose
        :meth:`to_polars` lands on a dtype Polars natively
        represents. Defaults to ``self``; subclasses Polars can't
        store at their declared width / precision override:

        * ``Float8Type`` and ``Float16Type`` widen to ``Float32Type``
          (Polars has no sub-32-bit floats);
        * ``TimestampType`` / ``DurationType`` with second-precision
          (``unit="s"``) widen to ``unit="ms"`` (Polars supports
          ``ms`` / ``us`` / ``ns`` only);
        * nested types (``ArrayType`` / ``MapType`` / ``StructType``)
          recurse via ``as_polars`` on their child fields.

        :class:`Field` and :class:`Schema` expose a matching
        ``as_polars`` that delegates to ``self.dtype.as_polars`` and
        re-wraps so callers chain through Field-shaped APIs without
        dropping back to a plain :class:`DataType`.
        """
        return self

    def as_spark(self) -> "DataType":
        """Return a Spark-flavored :class:`DataType` for this type.

        ``as_spark`` lives on the yggdrasil side of the boundary: it
        returns a :class:`DataType` that maps cleanly to a Spark dtype
        (i.e. one ``self.to_spark()`` would round-trip without a
        widening-time surprise). For types Spark already represents
        natively (signed ints, ``Float32`` / ``Float64``, ``Date``,
        ``String`` / ``Binary`` / ``Boolean``, decimal, naive / UTC
        timestamps), the default is to return ``self`` unchanged.

        Subclasses Spark cannot represent natively
        (``IntegerType`` with ``signed=False``, ``Float16Type``,
        ``DurationType``, ``TimeType``, non-UTC ``TimestampType``)
        override this to return the closest Spark-compatible
        yggdrasil dtype — usually a widened integer, a ``StringType``,
        or a naive timestamp. Nested types (``ArrayType`` /
        ``MapType`` / ``StructType``) recurse via ``as_spark`` on
        their child fields so the whole tree comes back
        Spark-compatible in one call.
        """
        return self

    # ==================================================================
    # Autotag — Databricks-friendly shape tags
    # ==================================================================

    def autotag(self) -> dict[bytes, bytes]:
        """Return a dict of Databricks-friendly tags derived from this type.

        These are *auto*-tags: they describe shape, not intent. The base
        output is a single ``kind`` key — a lowercase form of ``type_id``
        (``"integer"``, ``"string"``, ``"timestamp"``, ``"array"``, ...) that
        tag-based Unity Catalog policies can match on. Subclasses extend this
        with dtype-specific detail (``unit``, ``tz``, ``precision`` /
        ``scale``, ``signed``, ``srid``, ...) — always via ``super().autotag()``
        so the ``kind`` key stays present.

        Keys are bare (no ``t:`` prefix) — prefixing is handled by
        ``BaseMetadata.update_tags`` when these land on a Field.
        """
        return {b"type_name": self.type_id.name.lower().encode("utf-8")}

    # ==================================================================
    # Scalar conversion — Python / Arrow value coercion
    # ==================================================================

    def convert_pyobj(self, value: Any, nullable: bool, safe: bool = False):
        if value is None:
            return self.default_pyobj(nullable=nullable)
        return self._convert_pyobj(value, safe=safe)

    def _convert_pyobj(self, value: Any, safe: bool = False):
        return value

    def convert_arrow_scalar(
        self, value: pa.Scalar, nullable: bool, safe: bool = False
    ) -> pa.Scalar:
        return value.cast(self.to_arrow(), safe=safe)

    # ==================================================================
    # Constructors — generic dispatch
    # ==================================================================

    @classmethod
    def from_(cls, value: Any) -> "DataType":
        return cls.from_any(value)

    @classmethod
    def from_any(cls, value: Any) -> "DataType":
        if isinstance(value, DataType):
            return value

        if isinstance(value, ParsedDataType):
            return cls.from_parsed(value)

        if isinstance(value, type) and issubclass(value, DataType):
            return value.instance()

        ns, _ = ObjectSerde.module_and_name(value)

        for prefix, method in _FROM_ANY_NS_DISPATCH:
            if ns.startswith(prefix):
                return getattr(cls, method)(value)

        if isinstance(value, Mapping):
            return cls.from_dict(value)

        if isinstance(value, str):
            return cls.from_str(value)

        if isinstance(value, type):
            return cls.from_pytype(value)

        if isinstance(value, (int, DataTypeId)):
            return cls.from_dict({"id": int(value)})

        if isinstance(value, pa.DataType):
            return cls.from_arrow_type(value)

        if is_dataclass(value):
            return cls.from_dataclass(value)

        if value is None:
            return NullType()

        raise ValueError(
            f"Cannot convert value of type {type(value).__name__} to DataType: {value!r}"
        )

    # ==================================================================
    # Constructors — strings, parsed tokens, dicts
    # ==================================================================

    @classmethod
    def from_str(cls, value: str) -> "DataType":
        token = str(value).strip()
        if not token:
            raise ValueError("Data type string cannot be empty")

        if token.startswith("{") and token.endswith("}"):
            payload = json.loads(token)
            if not isinstance(payload, dict):
                raise ValueError("Data type JSON string must decode to an object")
            return cls.from_dict(payload)

        return cls.from_parsed(parse_data_type(token))

    @classmethod
    def from_parsed(cls, parsed: ParsedDataType) -> "DataType":
        from .enums import DictionaryType, EnumType, IntEnumType, StrEnumType
        from .nested import ArrayType, MapType, StructType
        from .primitive import (
            BinaryType,
            BJsonType,
            BooleanType,
            DateType,
            DecimalType,
            DurationType,
            FloatingPointType,
            IntegerType,
            NullType,
            PrimitiveType,
            SJsonType,
            StringType,
            TimeType,
            TimestampType,
        )

        field_cls = field_class()
        meta: DataTypeMetadata = parsed.metadata

        if parsed.type_id == DataTypeId.NULL:
            return NullType()

        if parsed.type_id == DataTypeId.BOOL:
            return BooleanType()

        if parsed.type_id.is_integer:
            byte_size, signed = _INT_TYPE_ID_TO_PARAMS.get(
                parsed.type_id, (parsed.byte_size or 8, True)
            )
            return IntegerType(byte_size=byte_size, signed=signed)

        if parsed.type_id.is_floating_point:
            byte_size = _FLOAT_TYPE_ID_TO_SIZE.get(
                parsed.type_id, parsed.byte_size or 8
            )
            return FloatingPointType(byte_size=byte_size)

        if parsed.type_id == DataTypeId.DECIMAL:
            precision = meta.precision if meta.precision is not None else 38
            scale = meta.scale if meta.scale is not None else 18
            return DecimalType(
                byte_size=parsed.byte_size, precision=precision, scale=scale
            )

        if parsed.type_id == DataTypeId.DATE or parsed.name in {"dt.date", "date"}:
            return DateType()

        if parsed.type_id == DataTypeId.TIME or parsed.name in {"dt.time", "time"}:
            return TimeType(unit=meta.unit or "us")

        if parsed.type_id == DataTypeId.TIMESTAMP or parsed.name in {
            "dt.datetime",
            "datetime",
        }:
            tz = meta.timezone
            if tz in {"ntz", "without_time_zone"}:
                tz = None
            elif tz in {"ltz", "with_time_zone"}:
                tz = "UTC"
            else:
                tz = tz or "UTC"
            return TimestampType(unit=meta.unit or "us", tz=tz)

        if parsed.type_id == DataTypeId.DURATION or parsed.name in {
            "dt.timedelta",
            "timedelta",
        }:
            return DurationType(unit=meta.unit or "us")

        if parsed.type_id == DataTypeId.BINARY:
            return BinaryType()

        if parsed.type_id == DataTypeId.STRING:
            return StringType()

        if parsed.type_id == DataTypeId.ARRAY:
            child = parsed.item or ParsedDataType(type_id=DataTypeId.OBJECT)
            return ArrayType.from_item(field_cls.from_parsed(child))

        if parsed.type_id == DataTypeId.MAP:
            key_child = parsed.key or ParsedDataType(
                type_id=DataTypeId.STRING, name="key"
            )
            value_child = parsed.value or ParsedDataType(
                type_id=DataTypeId.STRING, name="value"
            )

            key_dtype = field_cls.from_parsed(key_child, name="key")
            value_dtype = field_cls.from_parsed(value_child, name="value")

            return MapType.from_key_value(
                key_field=key_dtype,
                value_field=value_dtype,
                keys_sorted=bool(meta.sorted),
            )

        if parsed.type_id == DataTypeId.STRUCT:
            return StructType(
                fields=[
                    field_cls.from_parsed(child, name=child.name or f"f{i}")
                    for i, child in enumerate(parsed.children)
                ]
            )

        if parsed.type_id == DataTypeId.ENUM:
            if meta.literals:
                literals = tuple(meta.literals)
                if all(isinstance(v, str) for v in literals):
                    return StrEnumType(categories=literals)
                if all(
                    isinstance(v, int) and not isinstance(v, bool)
                    for v in literals
                ):
                    return IntEnumType(categories=literals)
                value_hint = _literal_values_to_hint(literals)
                value_type = cls.from_pytype(value_hint)
                if not isinstance(value_type, PrimitiveType):
                    value_type = StringType()
                return EnumType(value_type=value_type, categories=literals)
            return EnumType()

        if parsed.type_id == DataTypeId.STR_ENUM:
            literals = tuple(meta.literals or ())
            return StrEnumType(categories=literals)

        if parsed.type_id == DataTypeId.INT_ENUM:
            literals = tuple(meta.literals or ())
            return IntEnumType(categories=literals)

        if parsed.type_id == DataTypeId.UNION:
            if not parsed.variants:
                return StringType()

            dtypes = [cls.from_parsed(child) for child in parsed.variants]
            first = dtypes[0]
            if all(
                type(dtype) is type(first) and dtype.to_dict() == first.to_dict()
                for dtype in dtypes[1:]
            ):
                return first

            return StringType()

        if parsed.type_id == DataTypeId.DICTIONARY:
            value_dtype: PrimitiveType | None = None
            if parsed.value_type is not None:
                value_resolved = cls.from_parsed(parsed.value_type)
                if isinstance(value_resolved, PrimitiveType):
                    value_dtype = value_resolved
            return DictionaryType(
                value_type=value_dtype if value_dtype is not None else StringType(),
                categories=(),
                ordered=bool(meta.ordered),
            )

        if parsed.type_id == DataTypeId.SJSON:
            return SJsonType()

        if parsed.type_id == DataTypeId.BJSON:
            return BJsonType(byte_size=parsed.byte_size)

        if parsed.type_id == DataTypeId.OBJECT:
            from .primitive import ObjectType

            return ObjectType()

        raise TypeError(f"Unsupported parsed data type: {parsed!r}")

    @classmethod
    def from_json(cls, obj: AnyStr, default: Any = ...) -> "DataType":
        try:
            if isinstance(obj, (str, bytes)):
                obj = json.loads(obj)

            return cls.from_dict(obj)
        except Exception as e:
            if default is ...:
                raise ValueError(f"Cannot parse JSON string: {obj!r}") from e
            return default

    @classmethod
    def from_dict(cls, value: dict[str, Any], default: Any = ...) -> "DataType":
        type_id = value.get("id") or value.get(b"id")

        if type_id is not None:
            target_class = DATA_TYPE_CLASSES.get(type_id)
            if target_class is not None:
                return target_class.from_dict(value)

            from .primitive import PrimitiveType
            from .nested import NestedType

            target_class = DATA_TYPE_CLASSES.get(type_id)
            if target_class is not None:
                return target_class.from_dict(value)

            raise ValueError(f"Unsupported data type ID: {type_id}")

        if "id" in value.keys() or b"id" in value.keys():
            if cls.__subclasses__():
                for subclass in cls.__subclasses__():
                    if subclass.__subclasses__():
                        for sc in subclass.__subclasses__():
                            if sc.handles_dict(value):
                                return sc.from_dict(value)

                    if subclass.handles_dict(value):
                        return subclass.from_dict(value)

        if default is ...:
            raise ValueError(f"Cannot find a valid {cls.__name__} in: {value}")

        return default

    # ==================================================================
    # Constructors — Python types, dataclasses
    # ==================================================================

    @classmethod
    def from_pytype(cls, hint: Any) -> "DataType":
        if isinstance(hint, DataType):
            return hint

        if isinstance(hint, ParsedDataType):
            return cls.from_parsed(hint)

        if isinstance(hint, str):
            return cls.from_str(hint)

        if hint is None or hint is _NONE_TYPE:
            return NullType()

        hint = _unwrap_newtype(_strip_annotated(hint))

        if hint is Any:
            return StringType()

        if hint is object:
            from .primitive import ObjectType

            return ObjectType()

        if is_dataclass(hint):
            return cls.from_dataclass(hint)

        origin = get_origin(hint)
        args = get_args(hint)

        if _safe_issubclass(hint, DataType):
            return hint.instance()

        if origin is Literal:
            return cls.from_pytype(_literal_values_to_hint(args))

        if origin in (Union, types.UnionType):
            non_null_args = tuple(arg for arg in args if arg is not _NONE_TYPE)

            if not non_null_args:
                return NullType()

            if len(non_null_args) == 1:
                return cls.from_pytype(non_null_args[0])

            dtypes = [cls.from_pytype(arg) for arg in non_null_args]
            first = dtypes[0]
            if all(
                type(dtype) is type(first) and dtype.to_dict() == first.to_dict()
                for dtype in dtypes[1:]
            ):
                return first

            return StringType()

        # Scalar builtins — exact-identity checks run before the
        # issubclass fallbacks below so we don't spuriously treat
        # ``bool`` as ``int`` (bool <: int in Python).
        if hint is bool:
            return BooleanType()

        if hint is int:
            return IntegerType(byte_size=8, signed=True)

        if hint is float:
            return FloatingPointType(byte_size=8)

        if hint is str:
            return StringType()

        if hint in (bytes, bytearray, memoryview):
            return BinaryType()

        if hint is decimal.Decimal:
            return DecimalType(precision=38, scale=18)

        if hint is uuid.UUID:
            return StringType(byte_size=36)

        if hint is dt.datetime:
            return TimestampType(unit="us", tz=None)

        if hint is dt.date:
            return DateType()

        if hint is dt.time:
            return TimeType(unit="us")

        if hint is dt.timedelta:
            return DurationType(unit="us")

        if _safe_issubclass(hint, enum.Enum):
            return EnumType.from_pyenum(hint)

        # Scalar subclass fallbacks — for user-defined types that
        # inherit from builtins (e.g. `class Mass(float): ...`).
        if _safe_issubclass(hint, bool):
            return BooleanType()

        if _safe_issubclass(hint, int):
            return IntegerType(byte_size=8, signed=True)

        if _safe_issubclass(hint, float):
            return FloatingPointType(byte_size=8)

        if _safe_issubclass(hint, str):
            return StringType()

        if _safe_issubclass(hint, (bytes, bytearray, memoryview)):
            return BinaryType()

        if _safe_issubclass(hint, decimal.Decimal):
            return DecimalType(precision=38, scale=18)

        if _safe_issubclass(hint, uuid.UUID):
            return StringType()

        if _safe_issubclass(hint, dt.datetime):
            return TimestampType(unit="us", tz=None)

        if _safe_issubclass(hint, dt.date):
            return DateType()

        if _safe_issubclass(hint, dt.time):
            return TimeType(unit="us")

        if _safe_issubclass(hint, dt.timedelta):
            return DurationType(unit="us")

        # Container generics — list / set / frozenset / Sequence / Set
        if origin in (list, Sequence, set, frozenset, AbstractSet):
            item_hint = args[0] if args else Any
            return ArrayType(
                item_field=cls.from_pytype(item_hint).to_field(
                    name="item",
                    nullable=True,
                )
            )

        if hint in (list, set, frozenset):
            return ArrayType(
                item_field=cls.from_pytype(Any).to_field(
                    name="item",
                    nullable=True,
                )
            )

        # Tuple — two flavors: ``tuple[T, ...]`` is a homogeneous array,
        # ``tuple[T1, T2, T3]`` is a fixed-arity struct with ``_0/_1/_2``.
        if origin is tuple:
            field_cls = field_class()

            if not args:
                return ArrayType(
                    item_field=cls.from_pytype(Any).to_field(
                        name="item",
                        nullable=True,
                    )
                )

            if len(args) == 2 and args[1] is Ellipsis:
                return ArrayType(
                    item_field=cls.from_pytype(args[0]).to_field(
                        name="item",
                        nullable=True,
                    )
                )

            return StructType(
                fields=[
                    field_cls.from_(arg).with_name(f"_{i}")
                    for i, arg in enumerate(args)
                ]
            )

        if hint is tuple:
            return ArrayType(
                item_field=cls.from_pytype(Any).to_field(
                    name="item",
                    nullable=True,
                )
            )

        # Dict / OrderedDict / Mapping generics
        if origin is dict:
            key_hint, value_hint = args if len(args) == 2 else (Any, Any)
            return MapType.from_key_value(
                key_field=cls.from_pytype(key_hint),
                value_field=cls.from_pytype(value_hint),
            )

        if origin is OrderedDict:
            key_hint, value_hint = args if len(args) == 2 else (Any, Any)
            return MapType.from_key_value(
                key_field=cls.from_pytype(key_hint),
                value_field=cls.from_pytype(value_hint),
                keys_sorted=True,
            )

        if origin in (Mapping, MutableMapping):
            key_hint, value_hint = args if len(args) == 2 else (Any, Any)
            return MapType.from_key_value(
                key_field=cls.from_pytype(key_hint),
                value_field=cls.from_pytype(value_hint),
            )

        if hint is dict:
            return MapType.from_key_value(
                key_field=cls.from_pytype(Any),
                value_field=cls.from_pytype(Any),
            )

        if hint is OrderedDict:
            return MapType.from_key_value(
                key_field=cls.from_pytype(Any),
                value_field=cls.from_pytype(Any),
                keys_sorted=True,
            )

        if _safe_issubclass(hint, OrderedDict):
            return MapType.from_key_value(
                key_field=cls.from_pytype(Any),
                value_field=cls.from_pytype(Any),
                keys_sorted=True,
            )

        if _safe_issubclass(hint, Mapping):
            return MapType.from_key_value(
                key_field=cls.from_pytype(Any),
                value_field=cls.from_pytype(Any),
            )

        # TypedDict / NamedTuple / annotated class → StructType
        if _is_typed_dict_type(hint):
            field_cls = field_class()
            annotations = getattr(hint, "__annotations__", {})
            required_keys = getattr(hint, "__required_keys__", frozenset())
            optional_keys = getattr(hint, "__optional_keys__", frozenset())

            inner_fields = []
            for name, field_hint in annotations.items():
                nullable = True
                if required_keys:
                    nullable = name not in required_keys
                elif optional_keys:
                    nullable = name in optional_keys

                inner_fields.append(
                    field_cls(
                        name=name,
                        dtype=cls.from_pytype(field_hint),
                        nullable=nullable,
                    )
                )

            return StructType(fields=inner_fields)

        if _is_namedtuple_type(hint):
            field_cls = field_class()
            annotations = getattr(hint, "__annotations__", {})
            return StructType(
                fields=[
                    field_cls(
                        name=name,
                        dtype=cls.from_pytype(field_hint),
                        nullable=True,
                    )
                    for name, field_hint in annotations.items()
                ]
            )

        if isinstance(hint, type) and getattr(hint, "__annotations__", None):
            field_cls = field_class()
            return StructType(
                fields=[
                    field_cls(
                        name=name,
                        dtype=cls.from_pytype(field_hint),
                        nullable=True,
                    )
                    for name, field_hint in hint.__annotations__.items()
                ]
            )

        raise TypeError(f"Unsupported Python type hint: {hint!r}")

    @classmethod
    def from_dataclass(cls, hint: Any) -> "StructType":
        if not is_dataclass(hint):
            raise TypeError(f"Unsupported dataclass input: {hint!r}")

        field_cls = field_class()
        inner_fields = []

        for f in fields(hint):
            if not (f.init or (not f.init and not f.name.startswith("_"))):
                continue

            inner_fields.append(
                field_cls(
                    name=f.name,
                    dtype=cls.from_any(f.type),
                    nullable=True,
                ).with_metadata(default=f.default)
            )

        return StructType(fields=inner_fields)

    # ==================================================================
    # Constructors — arrow
    # ==================================================================

    @classmethod
    def from_arrow(
        cls,
        value: Union[
            pa.Field,
            pa.DataType,
            pa.Array,
            pa.ChunkedArray,
            pa.Table,
            pa.RecordBatch,
            pa.Scalar,
            pa.Schema,
        ],
    ) -> "DataType":
        if isinstance(value, pa.DataType):
            return cls.from_arrow_type(value)
        if isinstance(value, pa.Field):
            return cls.from_arrow_field(value)
        if isinstance(value, (pa.Array, pa.ChunkedArray, pa.Scalar)):
            return cls.from_arrow_type(value.type)
        if isinstance(value, (pa.Table, pa.RecordBatch)):
            return cls.from_arrow_schema(value.schema)
        if isinstance(value, pa.Schema):
            return cls.from_arrow_schema(value)

        if hasattr(value, "type"):
            value = value.type
        elif hasattr(value, "schema"):
            value = value.schema
        elif hasattr(value, "arrow_schema"):
            value = value.arrow_schema
        elif hasattr(value, "arrow_field"):
            value = value.arrow_field
        else:
            raise TypeError(f"Unsupported arrow input: {value!r}")

        if callable(value):
            value = value()

        return cls.from_arrow(value)

    @classmethod
    def from_arrow_field(cls, value: pa.Field) -> "DataType":
        return cls.from_arrow_type(value.type)

    @classmethod
    def from_arrow_schema(cls, value: pa.Schema) -> "StructType":
        from ..data_field import Field

        return StructType(fields=[Field.from_arrow_field(f) for f in value])

    @classmethod
    def from_arrow_type(cls, value: pa.DataType) -> "DataType":
        for subclass in cls.__subclasses__():
            if subclass.handles_arrow_type(value):
                return subclass.from_arrow_type(value)
        raise TypeError(f"Unsupported Arrow data type: {value!r}")

    # ==================================================================
    # Constructors — polars
    # ==================================================================

    @classmethod
    def from_polars(cls, value: Any) -> "DataType":
        pl = get_polars()

        if isinstance(value, type) and issubclass(value, pl.DataType):
            return cls.from_polars_type(value())
        elif isinstance(value, pl.DataType):
            return cls.from_polars_type(value)
        elif isinstance(value, pl.Expr):
            return cls.from_polars_type(value.dtype)
        elif isinstance(value, pl.Series):
            return cls.from_polars_type(value.dtype)
        elif isinstance(value, pl.DataFrame):
            return cls.from_polars_schema(value.schema)
        elif isinstance(value, pl.LazyFrame):
            return cls.from_polars_schema(value.collect_schema())
        elif isinstance(value, pl.Field):
            return cls.from_polars_field(value)
        elif isinstance(value, pl.Schema):
            return cls.from_polars_schema(value)

        raise TypeError(f"Unsupported Polars data type: {value!r}")

    @classmethod
    def from_polars_field(cls, field: "polars.Field") -> "DataType":
        return cls.from_polars_type(field.dtype)

    @classmethod
    def from_polars_schema(cls, schema: "polars.Schema") -> "StructType":
        pl = get_polars()
        from ..data_field import Field

        return StructType(
            fields=[
                Field.from_polars(pl.Field(name=k, dtype=d)) for k, d in schema.items()
            ]
        )

    @classmethod
    def from_polars_type(cls, dtype: "polars.DataType") -> "DataType":
        for subclass in cls.__subclasses__():
            if subclass.handles_polars_type(dtype):
                return subclass.from_polars_type(dtype)

        from .primitive import StringType

        pl = get_polars()
        if dtype == pl.Categorical():
            return StringType()

        raise TypeError(f"Unsupported Polars data type: {dtype!r}")

    # ==================================================================
    # Constructors — pandas
    # ==================================================================

    @classmethod
    def from_pandas(cls, obj: Any):
        pd = get_pandas()
        import numpy as np

        if isinstance(obj, pd.Series):
            return cls.from_arrow_type(pa.array(obj, from_pandas=True).type)

        if isinstance(obj, pd.DataFrame):
            return cls.from_arrow_schema(pa.Table.from_pandas(obj).schema)

        if isinstance(obj, pd.Timestamp):
            return TimestampType(
                unit="ns",
                tz=str(obj.tz) if obj.tz is not None else None,
            )

        if isinstance(obj, pd.Timedelta):
            return DurationType(unit="ns")

        if obj is pd.NA:
            return NullType()

        dtype = None
        try:
            dtype = pd.api.types.pandas_dtype(obj)
        except Exception:
            pass

        if dtype is not None:
            name = str(dtype)

            integer_map = {
                "Int8": (1, True),
                "Int16": (2, True),
                "Int32": (4, True),
                "Int64": (8, True),
                "UInt8": (1, False),
                "UInt16": (2, False),
                "UInt32": (4, False),
                "UInt64": (8, False),
            }
            if name in integer_map:
                byte_size, signed = integer_map[name]
                return IntegerType(byte_size=byte_size, signed=signed)

            float_map = {
                "Float32": 4,
                "Float64": 8,
            }
            if name in float_map:
                return FloatingPointType(byte_size=float_map[name])

            # pandas extension dtypes (StringDtype, BooleanDtype) are not
            # numpy dtypes; check them before the np.dtype branch. Pandas 3.0
            # made StringDtype the default for Python-string columns.
            if isinstance(dtype, pd.StringDtype):
                return StringType()
            if isinstance(dtype, pd.BooleanDtype):
                return BooleanType()

            if isinstance(dtype, np.dtype):
                if dtype.kind == "i":
                    return IntegerType(byte_size=dtype.itemsize, signed=True)
                if dtype.kind == "u":
                    return IntegerType(byte_size=dtype.itemsize, signed=False)
                if dtype.kind == "f":
                    return FloatingPointType(byte_size=dtype.itemsize)
                if dtype.kind == "b":
                    return BooleanType()
                if dtype.kind == "M":
                    return TimestampType(unit="ns", tz=None)
                if dtype.kind == "m":
                    return DurationType(unit="ns")
                if dtype.kind in {"U", "S", "O"}:
                    return StringType()

        raise TypeError(f"Unsupported Pandas data type: {obj!r}")

    # ==================================================================
    # Constructors — spark
    # ==================================================================

    @classmethod
    def from_spark(cls, value: Any) -> "DataType":
        sp = get_spark_sql()

        if isinstance(value, type) and issubclass(value, sp.types.DataType):
            return cls.from_spark_type(value())
        if isinstance(value, sp.types.DataType):
            return cls.from_spark_type(value)
        if isinstance(value, sp.DataFrame):
            return cls.from_spark_type(value.schema)
        if isinstance(value, sp.types.StructField):
            return cls.from_spark_type(value.dataType)
        if isinstance(value, sp.Column):
            try:
                return cls.from_str(value._jc.expr().dataType().typeName())
            except Exception:
                pass

        raise TypeError(f"Unsupported Spark data type: {value!r}")

    @classmethod
    def from_spark_type(cls, value: "pst.DataType") -> "DataType":
        for subclass in cls.__subclasses__():
            if subclass.handles_spark_type(value):
                return subclass.from_spark_type(value)
        raise ValueError(f"Unknown DataType payload: {value!r}")

    # ==================================================================
    # Constructors — yggdrasil (Field / Schema / dtype holders)
    # ==================================================================

    @classmethod
    def from_yggdrasil(cls, value: Any, *, cls_name: str | None = None) -> "DataType":
        if hasattr(value, "dtype"):
            value = value.dtype

            if isinstance(value, DataType):
                return value

        for attr in ("collect_schema", "data_schema"):
            if hasattr(value, attr):
                from yggdrasil.data.schema import Schema

                data_schema = getattr(value, attr)

                if callable(data_schema):
                    data_schema = data_schema()

                if isinstance(data_schema, Schema):
                    return data_schema.dtype

        for attr in ("collect_data_field", "data_field"):
            if hasattr(value, attr):
                from yggdrasil.data.data_field import Field

                data_field = getattr(value, attr)

                if callable(data_field):
                    data_field = data_field()

                if isinstance(data_field, Field):
                    return data_field.dtype

        if hasattr(value, "to_arrow_field"):
            return cls.from_arrow_field(value.to_arrow_field())

        raise TypeError(
            f"Unsupported Yggdrasil data type for {cls_name or 'DataType'}: {value!r}"
        )

    # ==================================================================
    # Cast — public arrow dispatchers
    # ==================================================================

    def cast_arrow_array(
        self,
        array: pa.Array | pa.ChunkedArray,
        options: "CastOptions | None" = None,
        **more_options,
    ) -> pa.Array | pa.ChunkedArray:
        opts = get_cast_options_class().check(options, **more_options)

        if isinstance(array, pa.ChunkedArray):
            return self._cast_chunked_array(array, opts)
        return self._cast_arrow_array(array, opts)

    def cast_arrow_tabular(
        self,
        table: pa.Table | pa.RecordBatch,
        options: "CastOptions | None" = None,
        **more_options,
    ):
        opts = get_cast_options_class().check(options, **more_options)
        return self._cast_arrow_tabular(table, opts)

    def cast_arrow_batch_iterator(
        self,
        batches: "Iterable[pa.RecordBatch]",
        options: "CastOptions | None" = None,
        **more_options,
    ) -> "Iterator[pa.RecordBatch]":
        """Cast a stream of :class:`pa.RecordBatch` against this dtype.

        Non-struct dtypes promote to a single-column struct via
        :meth:`to_struct` and reuse the struct's iterator helper.
        """
        opts = get_cast_options_class().check(options, **more_options)
        return self._cast_arrow_batch_iterator(batches, opts)

    # ==================================================================
    # Cast — arrow internals
    # ==================================================================

    def _cast_arrow_array(
        self,
        array: pa.Array,
        options: "CastOptions",
    ) -> pa.Array:
        if options.need_cast(array, self):
            target_type = self.to_arrow()

            try:
                casted = pc.cast(
                    array,
                    target_type=target_type,
                    safe=options.safe,
                    memory_pool=options.arrow_memory_pool,
                )
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                # Polars fallback — for casts pyarrow rejects (e.g. string →
                # decimal, nested-struct coercions). Round-trip through
                # polars, then re-cast to pin down the exact target type.
                pl = get_polars()
                array = (
                    pl.from_arrow(array)
                    .cast(dtype=self.to_polars(), strict=options.safe)
                    .to_arrow()
                )
                casted = pc.cast(
                    array,
                    target_type=target_type,
                    safe=options.safe,
                    memory_pool=options.arrow_memory_pool,
                )
            return casted
        return array

    def _cast_chunked_array(
        self,
        array: pa.ChunkedArray,
        options: "CastOptions",
    ) -> pa.ChunkedArray:
        if options.need_cast(array, self):
            chunks = [self._cast_arrow_array(chunk, options) for chunk in array.chunks]
            # Identity short-circuit: if every per-chunk cast returned
            # the same array object it was handed (subclass override
            # decided no work was needed for that chunk), skip the
            # ``pa.chunked_array`` rebuild and hand back the original.
            # Saves an O(num_chunks) constructor pass on partial-cast
            # paths where the per-chunk dispatch already knows the
            # types align (e.g. some subclasses short-circuit on a
            # source-flag check that ``need_cast`` can't see at the
            # ChunkedArray level).
            if chunks and all(
                c is orig for c, orig in zip(chunks, array.chunks)
            ):
                return array
            chunk_type = chunks[0].type if chunks else self.to_arrow()
            return pa.chunked_array(chunks, type=chunk_type)
        return array

    def _cast_arrow_tabular(
        self,
        table: pa.Table | pa.RecordBatch,
        options: "CastOptions",
    ):
        if self.type_id == DataTypeId.STRUCT:
            raise NotImplementedError(
                "Need struct implementation for cast_arrow_tabular"
            )
        return self.to_struct()._cast_arrow_tabular(table, options)

    def _cast_arrow_batch_iterator(
        self,
        batches: "Iterable[pa.RecordBatch]",
        options: "CastOptions",
    ) -> "Iterator[pa.RecordBatch]":
        if self.type_id == DataTypeId.STRUCT:
            raise NotImplementedError(
                "Need struct implementation for cast_arrow_batch_iterator"
            )
        return self.to_struct()._cast_arrow_batch_iterator(batches, options)

    # ==================================================================
    # Cast — public polars dispatchers
    # ==================================================================

    def cast_polars_series(
        self,
        series: "polars.Series | polars.Expr",
        options: "CastOptions | None" = None,
        **more_options,
    ) -> "polars.Series | polars.Expr":
        opts = get_cast_options_class().check(options, **more_options)

        pl = polars_module()
        if isinstance(series, pl.Expr):
            return self._cast_polars_expr(series, opts)
        return self._cast_polars_series(series, opts)

    def cast_polars_expr(
        self,
        series: "polars.Series | polars.Expr",
        options: "CastOptions | None" = None,
        **more_options,
    ) -> "polars.Series | polars.Expr":
        """Expr-shape passthrough to :meth:`cast_polars_series`.

        :meth:`cast_polars_series` already dispatches by isinstance — this
        method exists so callers that know they hold an Expr can name it.
        """
        return self.cast_polars_series(series, options, **more_options)

    def cast_polars_tabular(
        self,
        table: "polars.DataFrame | polars.LazyFrame",
        options: "CastOptions | None" = None,
        **more_options,
    ):
        opts = get_cast_options_class().check(options, **more_options)
        return self._cast_polars_tabular(table, opts)

    # ==================================================================
    # Cast — polars internals
    # ==================================================================

    def _cast_polars_series(
        self,
        series: "polars.Series",
        options: "CastOptions",
    ):
        if options.need_cast(series, self):
            try:
                casted = series.cast(dtype=self.to_polars(), strict=options.safe)
            except Exception:
                # Polars→Arrow→Polars round-trip — used when the direct
                # polars cast rejects (e.g. timezone-lossy timestamp
                # casts). Arrow's cast engine has a wider acceptance set.
                pl = polars_module()
                arrow = self._cast_arrow_array(
                    series.to_arrow(compat_level=pl.CompatLevel.newest()),
                    options,
                )
                casted = pl.Series(
                    name=series.name, values=arrow, dtype=self.to_polars()
                )

            return casted

        return series

    def _cast_polars_expr(
        self,
        expr: "polars.Expr",
        options: "CastOptions",
    ):
        return expr.cast(dtype=self.to_polars(), strict=options.safe)

    def _cast_polars_tabular(
        self,
        table: "polars.DataFrame | polars.LazyFrame",
        options: "CastOptions",
    ):
        if self.type_id == DataTypeId.STRUCT:
            raise NotImplementedError(
                "Need struct implementation for cast_polars_tabular"
            )

        return self.to_struct()._cast_polars_tabular(table, options)

    # ==================================================================
    # Cast — public pandas dispatchers
    # ==================================================================

    def cast_pandas_series(
        self,
        series: "pd.Series",
        options: "CastOptions | None" = None,
        **more_options,
    ):
        opts = get_cast_options_class().check(options, **more_options)
        return self._cast_pandas_series(series, opts)

    def cast_pandas_tabular(
        self,
        data: "pd.DataFrame",
        options: "CastOptions | None" = None,
        **more_options,
    ):
        opts = get_cast_options_class().check(options, **more_options)
        return self._cast_pandas_tabular(data, opts)

    # ==================================================================
    # Cast — pandas internals
    # ==================================================================

    def _cast_pandas_series(
        self,
        series: "pd.Series",
        options: "CastOptions",
    ):
        if options.need_cast(series, self):
            pd = pandas_module()
            try:
                arrow = pa.Array.from_pandas(series)
            except Exception:
                arrow = pa.array(series.tolist(), from_pandas=True)

            casted = self._cast_arrow_array(arrow, options).to_pandas(types_mapper=None)

            if not isinstance(casted, pd.Series):
                casted = pd.Series(casted, index=series.index, name=series.name)
            else:
                casted.index = series.index
                casted.name = series.name

            return casted
        return series

    def _cast_pandas_tabular(
        self,
        data: "pd.DataFrame",
        options: "CastOptions",
    ):
        if self.type_id == DataTypeId.STRUCT:
            preserve_index = bool(data.index.name)
            arrow_table = pa.Table.from_pandas(
                data,
                preserve_index=preserve_index,
            )
            casted = self._cast_arrow_tabular(arrow_table, options).to_pandas()

            if preserve_index:
                casted = casted.set_index(data.index.name)

            return casted

        return self.to_struct()._cast_pandas_tabular(data, options)

    # ==================================================================
    # Cast — public spark dispatchers
    # ==================================================================

    def cast_spark_column(
        self,
        column: "ps.Column",
        options: "CastOptions | None" = None,
        **more_options,
    ):
        opts = get_cast_options_class().check(options, **more_options)
        return self._cast_spark_column(column, opts)

    def cast_spark_tabular(
        self,
        data: "ps.DataFrame",
        options: "CastOptions | None" = None,
        **more_options,
    ):
        opts = get_cast_options_class().check(options, **more_options)
        return self._cast_spark_tabular(data, opts)

    # ==================================================================
    # Cast — spark internals
    # ==================================================================

    def _cast_spark_column(
        self,
        column: "ps.Column",
        options: "CastOptions",
    ):
        if options.need_cast(column, self):
            return column.cast(self.to_spark())
        return column

    def _cast_spark_tabular(
        self,
        data: "ps.DataFrame",
        options: "CastOptions",
    ):
        if self.type_id == DataTypeId.STRUCT:
            raise NotImplementedError(
                "Need struct implementation for cast_polars_tabular"
            )

        return self.to_struct()._cast_spark_tabular(data, options)

    # ==================================================================
    # Fill — per-engine null-fill primitives
    # ==================================================================

    def fill_arrow_array_nulls(
        self,
        array: pa.Array | pa.ChunkedArray,
        *,
        nullable: bool = False,
        default_scalar: pa.Scalar | None = None,
    ) -> pa.Array | pa.ChunkedArray:
        if nullable:
            return array

        if array.null_count == 0:
            return array

        if default_scalar is None:
            default_scalar = self.default_arrow_scalar(nullable=nullable)

        if default_scalar is None:
            return array

        if isinstance(array, pa.ChunkedArray):
            chunks = [
                self.fill_arrow_array_nulls(
                    chunk,
                    nullable=nullable,
                    default_scalar=default_scalar,
                )
                for chunk in array.chunks
            ]
            return pa.chunked_array(chunks, type=array.type)

        return pc.fill_null(array, default_scalar)

    def fill_polars_array_nulls(
        self,
        series: "polars.Series | polars.Expr",
        *,
        nullable: bool = False,
        default_scalar: Any = None,
    ) -> "polars.Series | polars.Expr":
        if nullable:
            return series

        if default_scalar is None:
            default_scalar = self.default_polars_scalar(nullable=nullable)

        if default_scalar is None:
            return series

        pl = polars_module()
        if isinstance(series, pl.Series):
            if series.null_count() == 0:
                return series

        return series.fill_null(default_scalar)

    def fill_pandas_series_nulls(
        self,
        series: "pd.Series",
        *,
        nullable: bool = False,
        default_scalar: Any = None,
    ) -> "pd.Series":
        if nullable:
            return series

        if not series.isna().any():
            return series

        if default_scalar is None:
            default_scalar = self.default_pandas_scalar(nullable=nullable)

        if default_scalar is None:
            return series

        return series.fillna(default_scalar)

    def fill_spark_column_nulls(
        self,
        column: "ps.Column",
        *,
        nullable: bool = False,
        default_scalar: Any = None,
    ) -> "ps.Column":
        spark = get_spark_sql()
        F = spark.functions

        if default_scalar is None:
            default_scalar = self.default_spark_scalar(nullable=nullable)

        if default_scalar is None or self.type_id.is_nested:
            return column

        # ``F.lit`` introspects the value via ``_get_object_id`` and a
        # ``pyarrow.Scalar`` doesn't carry that — silently drops to a
        # no-op fill. Unwrap to its Python value first so ``F.lit``
        # picks the matching Spark literal type.
        if isinstance(default_scalar, pa.Scalar):
            default_scalar = default_scalar.as_py()

        try:
            defaults = F.lit(default_scalar)
        except Exception as e:
            logging.warning(e)
            return column

        return F.when(column.isNull(), defaults).otherwise(column).cast(self.to_spark())

    # ==================================================================
    # Defaults — scalar + column / array factories
    # ==================================================================
    #
    # ``default_<engine>_scalar`` returns a single value; the
    # ``default_<engine>_series`` / ``default_<engine>_column`` /
    # ``default_arrow_array`` methods lift that scalar into a sized
    # container for zero-row or N-row materialization.

    def default_pyobj(self, nullable: bool) -> Any:
        if nullable:
            return None
        raise NotImplementedError(
            f"{type(self).__name__}.default_pyobj(nullable=False) is not implemented"
        )

    def default_arrow_scalar(self, nullable: bool = True) -> pa.Scalar:
        return pa.scalar(self.default_pyobj(nullable=nullable), type=self.to_arrow())

    def default_polars_scalar(self, nullable: bool = True) -> Any:
        return self.default_pyobj(nullable=nullable)

    def default_pandas_scalar(self, nullable: bool = True) -> Any:
        return self.default_pyobj(nullable=nullable)

    def default_spark_scalar(self, nullable: bool = True) -> Any:
        return self.default_pyobj(nullable=nullable)

    def default_arrow_array(
        self,
        nullable: bool,
        size: int = 0,
        memory_pool: Optional[pa.MemoryPool] = None,
        chunks: Optional[list[int]] = None,
        default_scalar: Optional[pa.Scalar] = None,
    ) -> Union[pa.Array, pa.ChunkedArray]:
        if default_scalar is None:
            default_scalar = self.default_arrow_scalar(nullable=nullable)

        arrow_type = self.to_arrow()

        if size == 0 and chunks is None:
            return pa.array([], type=arrow_type)

        if chunks is not None:
            if len(chunks) == 0:
                return pa.chunked_array([], type=arrow_type)

            return pa.chunked_array(
                [
                    (
                        pa.repeat(
                            value=default_scalar,
                            size=chunk_size,
                            memory_pool=memory_pool,
                        )
                        if chunk_size > 0
                        else pa.array([], type=arrow_type)
                    )
                    for chunk_size in chunks
                ],
                type=arrow_type,
            )

        return pa.repeat(
            value=default_scalar,
            size=size,
            memory_pool=memory_pool,
        )

    def default_polars_expr(
        self,
        value: Any = None,
        *,
        alias: str | None = None,
        nullable: bool = True,
    ):
        pl = get_polars()
        value = (
            self.default_polars_scalar(nullable=nullable) if value is None else value
        )
        s = pl.lit(value, dtype=self.to_polars())

        if alias and alias != DEFAULT_FIELD_NAME:
            s = s.alias(alias)
        return s

    def default_polars_series(
        self,
        value: Any = None,
        *,
        nullable: bool = True,
        size: int = 0,
        name: str = "",
    ):
        pl = get_polars()
        value = (
            self.default_polars_scalar(nullable=nullable) if value is None else value
        )
        return pl.Series(
            name=name or DEFAULT_FIELD_NAME,
            values=[value] * size,
            dtype=self.to_polars(),
        )

    def default_pandas_series(
        self,
        value: Any = None,
        *,
        nullable: bool = True,
        size: int = 0,
        name: str | None = None,
        index: Any = None,
    ):
        pd = get_pandas()
        value = (
            self.default_pandas_scalar(nullable=nullable) if value is None else value
        )
        return pd.Series(
            [value] * size,
            index=index,
            name=name,
            dtype=None,
        )

    def default_spark_column(
        self,
        value: Any = None,
        *,
        nullable: bool = True,
    ):
        spark = get_spark_sql()

        # TODO: Spark should handle nested types
        if self.type_id.is_nested:
            return spark.functions.lit(None).cast(self.to_spark())

        value = self.default_spark_scalar(nullable=nullable) if value is None else value
        return spark.functions.lit(value).cast(self.to_spark())


# ======================================================================
# Subclass re-exports — resolve late so subclass registry is populated
# ======================================================================

from .nested import ArrayType, MapType, NestedType, StructType
from .primitive import (
    BinaryType,
    BJsonType,
    BooleanType,
    DateType,
    DecimalType,
    DurationType,
    Float8Type,
    Float16Type,
    Float32Type,
    Float64Type,
    FloatingPointType,
    Int8Type,
    Int16Type,
    Int32Type,
    Int64Type,
    IntegerType,
    NullType,
    NumericType,
    PrimitiveType,
    SJsonType,
    StringType,
    TemporalType,
    TimeType,
    TimestampType,
    UInt8Type,
    UInt16Type,
    UInt32Type,
    UInt64Type,
)
from .enums import DictionaryType, EnumType, IntEnumType, StrEnumType
