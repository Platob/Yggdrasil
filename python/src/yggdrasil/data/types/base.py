from __future__ import annotations

import datetime as dt
import decimal
import enum
import json
import types
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping, MutableMapping, Sequence, Set as AbstractSet
from dataclasses import fields, is_dataclass
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Literal,
    Optional,
    Union,
    get_args,
    get_origin,
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
from yggdrasil.environ.importlib import cached_from_import
from yggdrasil.io import SaveMode
from yggdrasil.pickle.serde import ObjectSerde
from .support import get_pandas, get_polars, get_spark_sql
from ..base_meta import BaseChildrenFields
from ..data_utils import get_cast_options_class

if TYPE_CHECKING:
    import pandas as pd
    import polars
    import pyspark.sql as ps
    import pyspark.sql.types as pst
    from yggdrasil.data.cast.options import CastOptions
    from yggdrasil.data.data_field import Field


__all__ = [
    "DataTypeId",
    "DataType",
    "PrimitiveType",
    "NullType",
    "BinaryType",
    "StringType",
    "BooleanType",
    "NumericType",
    "IntegerType",
    "FloatingPointType",
    "DecimalType",
    "TemporalType",
    "DateType",
    "TimeType",
    "TimestampType",
    "DurationType",
    "NestedType",
    "ArrayType",
    "MapType",
    "StructType",
]


_NONE_TYPE = type(None)

_FROM_ANY_NS_DISPATCH: tuple[tuple[str, str], ...] = (
    ("pyarrow", "from_arrow"),
    ("polars", "from_polars"),
    ("pandas", "from_pandas"),
    ("pyspark", "from_spark"),
    ("yggdrasil", "from_yggdrasil")
)




def _safe_issubclass(obj: object, class_or_tuple: object) -> bool:
    return (
        isinstance(obj, type)
        and issubclass(obj, class_or_tuple)
        or obj is class_or_tuple
    )


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


class DataType(BaseChildrenFields, ABC):
    _singleton_instance: ClassVar["DataType | None"] = None

    def __str__(self):
        return self.to_arrow().__str__()

    def __call__(self, *args, **kwargs):
        if not args and not kwargs:
            return self
        raise ValueError(f"Cannot call {self.__class__} with args or kwargs")

    def equals(
        self,
        other: "DataType",
        check_names: bool = True,
        check_dtypes: bool = True,
        check_metadata: bool = True,
    ) -> bool:
        return self == other

    @property
    @abstractmethod
    def type_id(self) -> DataTypeId: ...

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
        return self.get_data_field_class()(
            name=name or DEFAULT_FIELD_NAME,
            dtype=self,
            nullable=nullable,
            metadata=metadata,
        )

    def merge_with(
        self,
        other: "DataType",
        mode: SaveMode | None = None,
        downcast: bool = False,
        upcast: bool = False,
    ):
        if self.type_id == DataTypeId.NULL:
            return other
        elif other.type_id == DataTypeId.NULL:
            return self
        elif mode == SaveMode.OVERWRITE:
            return other

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
        mode: SaveMode | None = None,
        downcast: bool = False,
        upcast: bool = False,
    ):
        raise NotImplementedError

    def _merge_with_different_id(
        self,
        other: "DataType",
        mode: SaveMode | None = None,
        downcast: bool = False,
        upcast: bool = False,
    ):
        if downcast == upcast:
            return self

        if downcast:
            return self if self.type_id < other.type_id else other
        else:
            return self if self.type_id > other.type_id else other

    @classmethod
    def get_data_field_class(cls) -> type["Field"]:
        return cached_from_import("yggdrasil.data.data_field", "Field")

    @classmethod
    def instance(cls) -> "DataType":
        if cls is DataType:
            raise TypeError("DataType.instance() must be called on a subclass")

        inst = cls._singleton_instance
        if inst is None:
            inst = cls()
            cls._singleton_instance = inst
        return inst

    @classmethod
    def _any_subclass_handles(cls, dtype: Any, handler: str, label: str) -> bool:
        subclasses = cls.__subclasses__()
        if not subclasses:
            raise TypeError(f"Unsupported {label} data type: {dtype!r}")
        return any(getattr(sub, handler)(dtype) for sub in subclasses)

    @staticmethod
    def _target_nullable(options: "CastOptions") -> bool:
        return options.target_field.nullable if options.target_field is not None else True

    @staticmethod
    def _matches_dict(value: dict[str, Any], type_id: DataTypeId, *aliases: str) -> bool:
        if value.get("id") == int(type_id):
            return True
        name = str(value.get("name", "")).upper()
        return name == type_id.name or name in aliases

    @classmethod
    def handles_arrow_type(cls, dtype: pa.DataType) -> bool:
        return cls._any_subclass_handles(dtype, "handles_arrow_type", "Arrow")

    @abstractmethod
    def to_arrow(self) -> pa.DataType:
        raise NotImplementedError

    @classmethod
    def handles_polars_type(cls, dtype: "polars.DataType") -> bool:
        return cls._any_subclass_handles(dtype, "handles_polars_type", "Polars")

    @abstractmethod
    def to_polars(self) -> "polars.DataType":
        raise NotImplementedError

    @classmethod
    def handles_spark_type(cls, dtype: "pst.DataType") -> bool:
        return cls._any_subclass_handles(dtype, "handles_spark_type", "Spark")

    @abstractmethod
    def to_spark(self) -> Any:
        raise NotImplementedError

    def to_polars_flavor(self) -> "polars.DataType":
        """Return the Polars-native counterpart for this object.

        On ``DataType`` the counterpart is a Polars dtype; on ``Field`` /
        ``Schema`` it's a ``pl.Field`` / ``pl.Schema``. The method name is
        uniform across the three classes so callers can dispatch on whatever
        Yggdrasil object they hold.
        """
        return self.to_polars()

    def to_spark_flavor(self) -> "pst.DataType":
        """Return the Spark-native counterpart for this object.

        See :meth:`to_polars_flavor` for the shared contract.
        """
        return self.to_spark()

    @abstractmethod
    def to_databricks_ddl(self) -> str:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.type_id.value,
            "name": self.type_id.name,
        }

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
        return {b"kind": self.type_id.name.lower().encode("utf-8")}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.to_arrow()})"

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
        value = self.default_spark_scalar(nullable=nullable) if value is None else value
        return spark.functions.lit(value).cast(self.to_spark())

    @classmethod
    def from_parsed(cls, parsed: ParsedDataType) -> "DataType":
        from .nested import ArrayType, MapType, StructType
        from .primitive import (
            BinaryType,
            BooleanType,
            DateType,
            DecimalType,
            DurationType,
            FloatingPointType,
            IntegerType,
            NullType,
            StringType,
            TimeType,
            TimestampType,
        )

        field_cls = cls.get_data_field_class()
        meta: DataTypeMetadata = parsed.metadata

        if parsed.type_id == DataTypeId.NULL:
            return NullType()

        if parsed.type_id == DataTypeId.BOOL:
            return BooleanType()

        if parsed.type_id == DataTypeId.INTEGER:
            return IntegerType(byte_size=parsed.byte_size or 8, signed=True)

        if parsed.type_id == DataTypeId.FLOAT:
            return FloatingPointType(byte_size=parsed.byte_size or 8)

        if parsed.type_id == DataTypeId.DECIMAL:
            precision = meta.precision if meta.precision is not None else 38
            scale = meta.scale if meta.scale is not None else 18
            return DecimalType(
                byte_size=parsed.byte_size, precision=precision, scale=scale
            )

        if parsed.type_id == DataTypeId.DATE:
            return DateType()

        if parsed.type_id == DataTypeId.TIME:
            return TimeType(unit=meta.unit or "us")

        if parsed.type_id == DataTypeId.TIMESTAMP:
            tz = meta.timezone
            if tz in {"ntz", "without_time_zone"}:
                tz = None
            elif tz in {"ltz", "with_time_zone"}:
                tz = "UTC"
            return TimestampType(unit=meta.unit or "us", tz=tz)

        if parsed.type_id == DataTypeId.DURATION:
            return DurationType(unit=meta.unit or "us")

        if parsed.type_id == DataTypeId.BINARY:
            return BinaryType()

        if parsed.type_id == DataTypeId.STRING:
            return StringType()

        if parsed.type_id == DataTypeId.ARRAY:
            child = parsed.item or ParsedDataType(type_id=DataTypeId.OBJECT)
            return ArrayType.from_item_field(field_cls.from_parsed(child), safe=False)

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
                    field_cls(
                        name=child.name or f"f{i}",
                        dtype=cls.from_parsed(child),
                        nullable=child.nullable is not False,
                    )
                    for i, child in enumerate(parsed.children)
                ]
            )

        if parsed.type_id == DataTypeId.ENUM:
            if meta.literals:
                return cls.from_pytype(_literal_values_to_hint(meta.literals))
            return StringType()

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
            if parsed.value_type is not None:
                return cls.from_parsed(parsed.value_type)
            return StringType()

        if parsed.type_id == DataTypeId.JSON:
            return StringType()

        if parsed.type_id == DataTypeId.OBJECT:
            # Unknown forward-ref names (e.g. dataclass annotations like
            # "dt.datetime" under `from __future__ import annotations`) come
            # through as OBJECT with the original token kept in `name`.
            # String is the safer round-trip target; reserve ObjectType for
            # the explicit `object` / `any` / `variant` aliases that arrive
            # without a carried name.
            if parsed.name:
                return StringType()
            from .primitive import ObjectType

            return ObjectType()

        raise TypeError(f"Unsupported parsed data type: {parsed!r}")

    def cast_arrow_array(
        self,
        array: pa.Array | pa.ChunkedArray,
        options: "CastOptions | None" = None,
        **more_options,
    ) -> pa.Array | pa.ChunkedArray:
        # Object target is a variant — never touch the values.
        if self.type_id == DataTypeId.OBJECT:
            return array

        opts = get_cast_options_class().check(options, **more_options)
        opts = opts.check_source(array).check_target(self)

        if not opts.need_cast(
            check_names=True, check_dtypes=True, check_metadata=False
        ):
            return array

        # Null-typed source carries no values; reinterpret as typed-null
        # instead of routing through pc.cast / empty-string nullifying.
        src_field = opts.source_field
        if src_field is not None and src_field.dtype.type_id == DataTypeId.NULL:
            length = array.length() if isinstance(array, pa.ChunkedArray) else len(array)
            typed_nulls = pa.nulls(length, type=self.to_arrow())
            if isinstance(array, pa.ChunkedArray):
                typed_nulls = pa.chunked_array([typed_nulls], type=self.to_arrow())
            return opts.fill_arrow_nulls(typed_nulls)

        # Source-driven outgoing cast: categorical/ISO types can translate
        # themselves into numeric / temporal / etc. representations using
        # domain knowledge that the generic pc.cast can't recover.  The hook
        # returns None when the source has no special handling for this
        # target, and we fall through to the standard target-side pipeline.
        if src_field is not None:
            try:
                forwarded = src_field.dtype._outgoing_cast_arrow_array(array, self, opts)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as e:
                raise ValueError(
                    f"Failed casting from {opts.source_field!r} to {opts.target_field!r}: {e}"
                ) from e
            if forwarded is not None:
                return opts.fill_arrow_nulls(forwarded)

        if isinstance(array, pa.ChunkedArray):
            return self._cast_chunked_array(array, opts)
        return self._cast_arrow_array(array, opts)

    @staticmethod
    def _nullify_empty_arrow_strings(
        array: pa.Array | pa.ChunkedArray,
    ) -> pa.Array | pa.ChunkedArray:
        dtype = array.type
        if (
            pa.types.is_string(dtype)
            or pa.types.is_large_string(dtype)
            or pa.types.is_string_view(dtype)
        ):
            empty = pa.scalar("", type=dtype)
        elif (
            pa.types.is_binary(dtype)
            or pa.types.is_large_binary(dtype)
            or pa.types.is_binary_view(dtype)
        ):
            empty = pa.scalar(b"", type=dtype)
        else:
            return array

        null_scalar = pa.scalar(None, type=dtype)

        if isinstance(array, pa.ChunkedArray):
            chunks = [
                pc.if_else(pc.equal(chunk, empty), null_scalar, chunk)
                for chunk in array.chunks
            ]
            return pa.chunked_array(chunks, type=dtype)

        return pc.if_else(pc.equal(array, empty), null_scalar, array)

    def _outgoing_cast_arrow_array(
        self,
        array: pa.Array | pa.ChunkedArray,
        target: "DataType",
        options: "CastOptions",
    ) -> pa.Array | pa.ChunkedArray | None:
        """Optional hook: cast *from* this type to *target* using source-specific knowledge.

        Override on categorical source types (ISO codes, enums, dictionaries)
        to translate into numeric / temporal / etc. representations that the
        generic ``pc.cast`` pipeline can't recover.  Return ``None`` to defer
        to the standard target-side pipeline.
        """
        return None

    def _cast_arrow_array(
        self,
        array: pa.Array,
        options: "CastOptions",
    ) -> pa.Array:
        options = options.check_source(array)

        array = self._nullify_empty_arrow_strings(array)
        target_type = self.to_arrow()

        try:
            casted = pc.cast(
                array,
                target_type=target_type,
                safe=options.safe,
                memory_pool=options.arrow_memory_pool,
            )
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
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

        return options.fill_arrow_nulls(casted)

    def _cast_chunked_array(
        self,
        array: pa.ChunkedArray,
        options: "CastOptions",
    ) -> pa.ChunkedArray:
        chunks = [self._cast_arrow_array(chunk, options) for chunk in array.chunks]
        chunk_type = chunks[0].type if chunks else self.to_arrow()
        return pa.chunked_array(chunks, type=chunk_type)

    def cast_polars_series(
        self,
        series: "polars.Series | polars.Expr",
        options: "CastOptions | None" = None,
        **more_options,
    ) -> "polars.Series | polars.Expr":
        # Object target is a variant — never touch the values.
        if self.type_id == DataTypeId.OBJECT:
            return series

        opts = (
            get_cast_options_class()
            .check(options, **more_options)
            .check_source(series)
            .check_target(self)
        )

        if not opts.need_cast(
            check_names=True, check_dtypes=True, check_metadata=False
        ):
            return series

        pl = get_polars()

        # Null-typed source: reinterpret with target dtype, skip empty-string
        # nullify and value-level cast.
        src_field = opts.source_field
        if src_field is not None and src_field.dtype.type_id == DataTypeId.NULL:
            casted = series.cast(self.to_polars(), strict=False)
            return self.fill_polars_array_nulls(
                casted, nullable=self._target_nullable(opts)
            )

        series = self._nullify_empty_polars_strings(series)

        if isinstance(series, pl.Expr):
            return self._cast_polars_expr(series, opts)
        return self._cast_polars_series(series, opts)

    @staticmethod
    def _nullify_empty_polars_strings(
        series: "polars.Series | polars.Expr",
    ) -> "polars.Series | polars.Expr":
        pl = get_polars()

        if not isinstance(series, pl.Series):
            return series

        if series.dtype != pl.String and series.dtype != pl.Binary:
            return series

        arr = DataType._nullify_empty_arrow_strings(series.to_arrow())
        return pl.Series(name=series.name, values=arr, dtype=series.dtype)

    def cast_polars_expr(
        self,
        series: "polars.Series | polars.Expr",
        options: "CastOptions | None" = None,
        **more_options,
    ) -> "polars.Series | polars.Expr":
        return self.cast_polars_series(series, options, **more_options)

    def _cast_polars_series(
        self,
        series: "polars.Series",
        options: "CastOptions",
    ):
        try:
            casted = series.cast(dtype=self.to_polars(), strict=options.safe)
        except Exception:
            pl = get_polars()
            arrow = self._cast_arrow_array(
                series.to_arrow(compat_level=pl.CompatLevel.newest()),
                options,
            )
            casted = pl.Series(name=series.name, values=arrow, dtype=self.to_polars())

        return self.fill_polars_array_nulls(casted, nullable=self._target_nullable(options))

    def _cast_polars_expr(
        self,
        expr: "polars.Expr",
        options: "CastOptions",
    ):
        casted = expr.cast(dtype=self.to_polars(), strict=options.safe)
        return self.fill_polars_array_nulls(casted, nullable=self._target_nullable(options))

    def cast_polars_tabular(
        self,
        table: "polars.DataFrame | polars.LazyFrame",
        options: "CastOptions | None" = None,
        **more_options,
    ):
        # Object target is a variant — never touch the values.
        if self.type_id == DataTypeId.OBJECT:
            return table

        opts = (
            get_cast_options_class()
            .check(options, **more_options)
            .check_source(table)
            .check_target(self)
        )

        if not opts.need_cast(
            check_names=True, check_dtypes=True, check_metadata=False
        ):
            return table

        return self._cast_polars_tabular(table, opts)

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

    def cast_pandas_series(
        self,
        series: "pd.Series",
        options: "CastOptions | None" = None,
        **more_options,
    ):
        # Object target is a variant — never touch the values.
        if self.type_id == DataTypeId.OBJECT:
            return series

        opts = (
            get_cast_options_class()
            .check(options, **more_options)
            .check_source(series)
            .check_target(self)
        )

        if not opts.need_cast(
            check_names=True, check_dtypes=True, check_metadata=False
        ):
            return series

        # Null-typed source: build a typed null series directly, skip Arrow.
        src_field = opts.source_field
        if src_field is not None and src_field.dtype.type_id == DataTypeId.NULL:
            pd = get_pandas()
            nullable = self._target_nullable(opts)
            value = self.default_pandas_scalar(nullable=nullable) if not nullable else None
            out = pd.Series(
                [value] * len(series),
                index=series.index,
                name=series.name,
            )
            return self.fill_pandas_series_nulls(out, nullable=nullable)

        return self._cast_pandas_series(series, opts)

    def _cast_pandas_series(
        self,
        series: "pd.Series",
        options: "CastOptions",
    ):
        pd = get_pandas()
        options.check_source(series)

        try:
            arrow = pa.Array.from_pandas(series)
        except Exception:
            arrow = pa.array(series.tolist(), from_pandas=True)

        casted = self._cast_arrow_array(arrow, options)
        out = casted.to_pandas(types_mapper=None)

        if not isinstance(out, pd.Series):
            out = pd.Series(out, index=series.index, name=series.name)
        else:
            out.index = series.index
            out.name = series.name

        return self.fill_pandas_series_nulls(out, nullable=self._target_nullable(options))

    def cast_pandas_tabular(
        self,
        data: "pd.DataFrame",
        options: "CastOptions | None" = None,
        **more_options,
    ):
        # Object target is a variant — never touch the values.
        if self.type_id == DataTypeId.OBJECT:
            return data

        opts = (
            get_cast_options_class()
            .check(options, **more_options)
            .check_source(data)
            .check_target(self)
        )

        if not opts.need_cast(
            check_names=True, check_dtypes=True, check_metadata=False
        ):
            return data

        return self.to_struct()._cast_pandas_tabular(data, opts)

    def _cast_pandas_tabular(
        self,
        data: "pd.DataFrame",
        options: "CastOptions",
    ):
        if self.type_id == DataTypeId.STRUCT:
            raise NotImplementedError(
                "Need struct implementation for cast_pandas_tabular"
            )
        return self.to_struct()._cast_pandas_tabular(data, options)

    def cast_arrow_tabular(
        self,
        table: pa.Table | pa.RecordBatch,
        options: "CastOptions | None" = None,
        **more_options,
    ):
        # Object target is a variant — never touch the values.
        if self.type_id == DataTypeId.OBJECT:
            return table

        opts = (
            get_cast_options_class()
            .check(options, **more_options)
            .check_source(table)
            .check_target(self)
        )

        if not opts.need_cast(
            check_names=True, check_dtypes=True, check_metadata=False
        ):
            return table

        return self._cast_arrow_tabular(table, opts)

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

    def cast_spark_column(
        self,
        column: "ps.Column",
        options: "CastOptions | None" = None,
        **more_options,
    ):
        # Object target is a variant — never touch the values.
        if self.type_id == DataTypeId.OBJECT:
            return column

        opts = (
            get_cast_options_class()
            .check(options, **more_options)
            .check_target(self)
        )

        if not opts.need_cast(
            check_names=True,
            check_dtypes=True,
            check_metadata=False,
        ):
            return column

        # Null-typed source: a plain .cast() is already a metadata-only op on
        # a NullType column; skip the empty-string nullify pass.
        src_field = opts.source_field
        if src_field is not None and src_field.dtype.type_id == DataTypeId.NULL:
            casted = column.cast(self.to_spark())
            return self.fill_spark_column_nulls(
                casted, nullable=self._target_nullable(opts)
            )

        return self._cast_spark_column(column, opts)

    def _cast_spark_column(
        self,
        column: "ps.Column",
        options: "CastOptions",
    ):
        options.check_source(column)

        column = self._nullify_empty_spark_strings(column, options)

        casted = column.cast(self.to_spark())
        return self.fill_spark_column_nulls(casted, nullable=self._target_nullable(options))

    @staticmethod
    def _nullify_empty_spark_strings(
        column: "ps.Column",
        options: "CastOptions",
    ) -> "ps.Column":
        source_field = options.source_field
        if source_field is None:
            return column

        source_type_id = source_field.dtype.type_id
        if source_type_id not in (DataTypeId.STRING, DataTypeId.BINARY):
            return column

        spark = get_spark_sql()
        F = spark.functions
        empty = F.lit("") if source_type_id == DataTypeId.STRING else F.lit(b"")
        return F.when(column == empty, F.lit(None)).otherwise(column)

    def cast_spark_tabular(
        self,
        data: "ps.DataFrame",
        options: "CastOptions | None" = None,
        **more_options,
    ):
        # Object target is a variant — never touch the values.
        if self.type_id == DataTypeId.OBJECT:
            return data

        opts = (
            get_cast_options_class()
            .check(options, **more_options)
            .check_source(data)
            .check_target(self)
        )

        if not opts.need_cast(
            check_names=True, check_dtypes=True, check_metadata=False
        ):
            return data

        return self._cast_spark_tabular(data, opts)

    def _cast_spark_tabular(
        self,
        data: "ps.DataFrame",
        options: "CastOptions",
    ):
        if self.type_id == DataTypeId.STRUCT:
            raise NotImplementedError(
                "Need struct implementation for cast_spark_tabular"
            )
        return self.to_struct()._cast_spark_tabular(data, options)

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
            members = list(hint)
            if not members:
                return StringType()
            return cls.from_pytype(type(members[0].value))

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

        if origin is tuple:
            field_cls = cls.get_data_field_class()

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

        if _is_typed_dict_type(hint):
            field_cls = cls.get_data_field_class()
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
            field_cls = cls.get_data_field_class()
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
            field_cls = cls.get_data_field_class()
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

        field_cls = cls.get_data_field_class()
        inner_fields = []

        for f in fields(hint):
            if not (f.init or (not f.init and not f.name.startswith("_"))):
                continue

            inner_fields.append(
                field_cls(
                    name=f.name,
                    dtype=cls.from_any(f.type),
                    nullable=True,
                )
            )

        return StructType(fields=inner_fields)

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

        raise TypeError(f"Unsupported Spark data type: {value!r}")

    @classmethod
    def from_spark_type(cls, value: "pst.DataType") -> "DataType":
        for subclass in cls.__subclasses__():
            if subclass.handles_spark_type(value):
                return subclass.from_spark_type(value)
        raise ValueError(f"Unknown DataType payload: {value!r}")

    @classmethod
    def from_yggdrasil(
        cls,
        value: Any,
        *,
        cls_name: str | None = None
    ) -> "DataType":
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

    @classmethod
    def handles_dict(cls, value: dict[str, Any]) -> bool:
        for subclass in cls.__subclasses__():
            if subclass.handles_dict(value):
                return True
        return False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DataType":
        if not value:
            raise ValueError(
                f"Cannot build {cls.__name__} from empty dictionary {value!r}"
            )

        if "id" in value.keys() or b"id" in value.keys():
            if cls.__subclasses__():
                for subclass in cls.__subclasses__():
                    if subclass.handles_dict(value):
                        return subclass.from_dict(value)
            raise ValueError(f"Unknown DataType payload: {value!r}")

        for key in ("dtype", "type_text", "type_json", "type"):
            dtype_payload = value.get(key)
            if dtype_payload:
                break

        if dtype_payload is None:
            raise ValueError(
                f"Cannot find a valid 'dtype' key in the provided dictionary: {value}"
            )

        base = cls.from_any(dtype_payload)

        if base.type_id == DataTypeId.STRUCT:
            inner_fields = value.get("fields", base.children_fields)
            return base.to_struct().with_fields(inner_fields)
        elif base.type_id == DataTypeId.ARRAY:
            element_type = value.get("elementType")
            if element_type is not None:
                base.item_field.with_dtype(element_type, inplace=True)

            contains_null = value.get("containsNull")
            if contains_null is not None:
                base.item_field.with_nullable(contains_null, inplace=True)
        elif base.type_id == DataTypeId.MAP:
            key_type = value.get("keyType")
            if key_type is not None:
                base.key_field.with_dtype(key_type, inplace=True).with_name(
                    "key", inplace=True
                )

            value_type = value.get("valueType")
            if value_type is not None:
                base.value_field.with_dtype(value_type, inplace=True).with_name(
                    "value", inplace=True
                )

            value_contains_null = value.get("valueContainsNull")
            if value_contains_null is not None:
                base.value_field.with_nullable(value_contains_null, inplace=True)

        return base

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

        if default_scalar is None or default_scalar == {}:
            return column

        return (
            F.when(column.isNull(), F.lit(default_scalar))
            .otherwise(column)
            .cast(self.to_spark())
        )


from .nested import ArrayType, MapType, NestedType, StructType
from .primitive import (
    BinaryType,
    BooleanType,
    DateType,
    DecimalType,
    DurationType,
    FloatingPointType,
    IntegerType,
    NullType,
    NumericType,
    PrimitiveType,
    StringType,
    TemporalType,
    TimeType,
    TimestampType,
)
