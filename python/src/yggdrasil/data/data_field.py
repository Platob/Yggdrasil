from __future__ import annotations

import dataclasses
import types
from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass
from typing import TYPE_CHECKING, Any, Annotated, Optional, Union, get_args, get_origin

import pyarrow as pa

import yggdrasil.pickle.json as json_module
from yggdrasil.data.base_meta import (
    BaseChildrenFields,
    BaseMetadata,
    _merge_metadata_and_tags,
    _normalize_metadata,
    _to_bytes,
)
from yggdrasil.data.constants import DEFAULT_VALUE_KEY, DEFAULT_FIELD_NAME
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.parser import ParsedDataType
from yggdrasil.io import SaveMode
from yggdrasil.pickle.serde import ObjectSerde
from .cast.registry import register_converter
from .data_utils import get_cast_options_class
from .types.base import DataType
from .types.nested import StructType
from .types.support import get_pandas, get_polars, get_spark_sql

if TYPE_CHECKING:
    import pandas as pd
    import polars
    import pyspark.sql as ps
    import pyspark.sql.types as pst
    from yggdrasil.data.cast.options import CastOptions
    from yggdrasil.data.schema import Schema
    from databricks.sdk.service.catalog import ColumnInfo as CatalogColumnInfo
    from databricks.sdk.service.sql import ColumnInfo as SQLColumnInfo


__all__ = [
    "Field",
    "field",
    "_normalize_metadata",
    "_to_bytes",
    "_merge_metadata_and_tags",
]

_TYPE_JSON_METADATA_KEY = b"to_json"
_NONE_TYPE = type(None)


def _attach_type_json_metadata(
    arrow_type: pa.DataType,
    metadata: dict[bytes, bytes] | None,
) -> dict[bytes, bytes]:
    out = dict(metadata or {})

    dtype = DataType.from_arrow_type(arrow_type)
    if isinstance(dtype, type) and issubclass(dtype, DataType):
        dtype = dtype()

    out[_TYPE_JSON_METADATA_KEY] = json_module.dumps(
        dtype.to_dict(),
        to_bytes=True,
        safe=False,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return out


def _strip_internal_metadata(
    metadata: dict[bytes, bytes] | None,
) -> dict[bytes, bytes] | None:
    if not metadata:
        return None

    out = {key: value for key, value in metadata.items() if key != _TYPE_JSON_METADATA_KEY}
    return out or None


def _safe_issubclass(obj: object, class_or_tuple: object) -> bool:
    return isinstance(obj, type) and issubclass(obj, class_or_tuple)


def _strip_annotated(hint: object) -> object:
    while get_origin(hint) is Annotated:
        args = get_args(hint)
        hint = args[0] if args else Any
    return hint


def _unwrap_newtype(hint: object) -> object:
    while hasattr(hint, "__supertype__"):
        hint = hint.__supertype__
    return hint


def _unwrap_nullable_hint(hint: Any) -> tuple[Any, bool]:
    if isinstance(hint, str):
        parsed = ParsedDataType.parse(hint)
        if parsed.type_id == DataTypeId.NULL:
            return _NONE_TYPE, True
        return hint, bool(parsed.nullable)

    hint = _unwrap_newtype(_strip_annotated(hint))
    origin = get_origin(hint)
    args = get_args(hint)

    if origin in (Union, types.UnionType):
        non_null_args = tuple(arg for arg in args if arg is not _NONE_TYPE)
        has_null = len(non_null_args) != len(args)

        if not non_null_args:
            return _NONE_TYPE, True

        if len(non_null_args) == 1:
            return _unwrap_newtype(_strip_annotated(non_null_args[0])), has_null

        return hint, has_null

    return hint, False


def _is_typed_dict_type(hint: object) -> bool:
    return (
        isinstance(hint, type)
        and issubclass(hint, dict)
        and hasattr(hint, "__annotations__")
        and hasattr(hint, "__total__")
    )


def _default_name(value: Any) -> str:
    if isinstance(value, type):
        return getattr(value, "__name__", DEFAULT_FIELD_NAME)
    return getattr(type(value), "__name__", DEFAULT_FIELD_NAME)


def _strip_matching_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"', "`", "´"}:
        return text[1:-1]
    return text


def _split_field_shorthand(value: str) -> tuple[str, str]:
    text = value.strip()
    quote: str | None = None
    depth_paren = 0
    depth_brack = 0
    depth_angle = 0

    for index, char in enumerate(text):
        if quote is not None:
            if char == quote:
                quote = None
            continue

        if char in {"'", '"', "`", "´"}:
            quote = char
            continue

        if char == "(":
            depth_paren += 1
        elif char == ")":
            depth_paren -= 1
        elif char == "[":
            depth_brack += 1
        elif char == "]":
            depth_brack -= 1
        elif char == "<":
            depth_angle += 1
        elif char == ">":
            depth_angle -= 1
        elif char == ":" and depth_paren == 0 and depth_brack == 0 and depth_angle == 0:
            left = text[:index].strip()
            right = text[index + 1 :].strip()
            if not left or not right:
                break
            return left, right

    return DEFAULT_FIELD_NAME, text


def _parse_field_name_token(value: str) -> tuple[str, bool | None]:
    token = _strip_matching_quotes(value.strip())

    if token.endswith("?"):
        return token[:-1].strip(), True
    if token.endswith("!"):
        return token[:-1].strip(), False
    return token, None


def field(
    name: str,
    dtype: DataType | type[DataType] | pa.DataType | None = None,
    *,
    arrow_type: pa.DataType | None = None,
    nullable: bool = True,
    metadata: dict[bytes | str, bytes | str | object] | None = None,
    tags: dict[bytes | str, bytes | str | object] | None = None,
    default: Any = None,
) -> "Field":
    return Field(
        name=name,
        dtype=DataType.from_any(arrow_type if dtype is None else dtype),
        nullable=nullable,
        metadata=_normalize_metadata(metadata, tags=tags, default_value=default),
    )


@dataclass(frozen=True, slots=True, init=False)
class Field(BaseMetadata, BaseChildrenFields):
    name: str
    dtype: DataType
    nullable: bool = True
    metadata: dict[bytes, bytes] | None = None

    def __init__(
        self,
        name: str,
        dtype: DataType | type[DataType] | pa.DataType,
        nullable: bool = True,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
        default: Any = None,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dtype", DataType.from_any(dtype))
        object.__setattr__(self, "nullable", bool(nullable))
        object.__setattr__(self, "metadata", _normalize_metadata(metadata, tags=tags, default_value=default))

    def __str__(self) -> str:
        suffix = " not null" if not self.nullable else ""
        if self.has_default:
            return f"{self.name!r}: {self.dtype!r}{suffix}"
        return f"{self.name!r}: {self.dtype!r}{suffix}"

    def __repr__(self) -> str:
        suffix = " not null" if not self.nullable else ""
        if self.has_default:
            return f"Field({self.name!r}: {self.dtype!r}{suffix})"
        return f"Field({self.name!r}: {self.dtype!r}{suffix})"

    def equals(
        self,
        other: Any,
        check_names: bool = True,
        check_dtypes: bool = True,
        check_metadata: bool = True,
    ) -> bool:
        """Structural equality check with configurable scope.

        Mirrors :meth:`DataType.equals`. Coerces *other* to a ``Field`` so
        that callers can pass a ``pa.Field`` / dict / etc. without manual
        conversion. Returns ``False`` on coercion failure instead of raising.

        - ``check_names``: compare this field's name and recurse into child
          field names for nested types.
        - ``check_dtypes``: recurse into the dtype and compare ``nullable``
          (both are structural, schema-defining attributes).
        - ``check_metadata``: compare this field's metadata and recurse.
        """
        if other is None:
            return False
        if not isinstance(other, Field):
            try:
                other = Field.from_any(other)
            except Exception:
                return False

        if check_names and self.name != other.name:
            return False

        if check_dtypes and self.nullable != other.nullable:
            return False

        if check_metadata and self.metadata != other.metadata:
            return False

        return self.dtype.equals(
            other.dtype,
            check_names=check_names,
            check_dtypes=check_dtypes,
            check_metadata=check_metadata,
        )
        
    @property
    def has_default(self) -> bool:
        return self.metadata.get(DEFAULT_VALUE_KEY) is not None if self.metadata else False

    @property
    def default(self):
        if self.has_default:
            return self.default_arrow_scalar.as_py()
        return None

    @property
    def default_arrow_scalar(self) -> pa.Scalar | None:
        if self.metadata is not None:
            default = self.metadata.get(DEFAULT_VALUE_KEY)

            if default is None:
                return self.dtype.default_arrow_scalar(nullable=self.nullable)

            try:
                default = json_module.loads(default, safe=False)
            except Exception as e:
                raise ValueError(f"Could not parse default value {default!r} for {self!r}: {e}") from e

            return self.dtype.convert_arrow_scalar(pa.scalar(default), nullable=self.nullable, safe=False)
        return None

    @property
    def children_fields(self) -> list["Field"]:
        return self.dtype.children_fields

    def _empty_tags(self) -> dict[bytes, bytes]:
        return {}

    @property
    def arrow_type(self) -> pa.DataType:
        return self.dtype.to_arrow()

    def merge_with(
        self,
        other: "Field",
        *,
        inplace: bool = False,
        mode: SaveMode | None = None,
        downcast: bool = False,
        upcast: bool = False,
        merge_dtype: bool = True,
        merge_nullable: bool = True,
        merge_metadata: bool = True
    ):
        if mode is not None:
            mode = SaveMode.parse(mode)

        other = self.from_any(other)
        if self == other:
            return self

        if self.name == DEFAULT_FIELD_NAME:
            name = other.name
        elif other.name == DEFAULT_FIELD_NAME:
            name = self.name
        else:
            name = self.name or other.name
        nullable = self.nullable or other.nullable if merge_nullable else self.nullable
        metadata = self.metadata if merge_metadata else None
        if merge_metadata:
            metadata = {
                **(self.metadata if self.metadata is not None else {}),
                **(other.metadata if other.metadata is not None else {}),
            }

        if merge_dtype:
            dtype = self.dtype.merge_with(
                other.dtype, mode=mode, downcast=downcast, upcast=upcast
            )
        else:
            dtype = other.dtype

        if inplace:
            object.__setattr__(self, "dtype", dtype)
            object.__setattr__(self, "nullable", bool(nullable))
            object.__setattr__(self, "metadata", metadata)
            return self

        return self.copy(
            name=name or DEFAULT_FIELD_NAME,
            dtype=dtype,
            nullable=nullable,
            metadata=metadata,
        )

    def with_name(self, name: str, inplace: bool = True) -> "Field":
        if name == self.name:
            return self

        if inplace:
            object.__setattr__(self, "name", name)
            return self
        return self.copy(name=name)

    def with_dtype(
        self,
        dtype: DataType | type[DataType] | pa.DataType,
        inplace: bool = True
    ) -> "Field":
        dtype = DataType.from_any(dtype)

        if inplace:
            object.__setattr__(self, "dtype", dtype)
            return self
        return self.copy(dtype=dtype)

    def with_nullable(self, nullable: bool, inplace: bool = True) -> "Field":
        if inplace:
            object.__setattr__(self, "nullable", bool(nullable))
            return self
        return self.copy(nullable=bool(nullable))

    def with_metadata(
        self,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
        inplace: bool = True,
    ):
        if metadata or tags:
            normalized = _normalize_metadata(metadata, tags=tags)
            if inplace:
                object.__setattr__(self, "metadata", normalized)
                return self
            return self.copy(metadata=normalized)
        return self

    def with_default(self, default: Any = None) -> "Field":
        metadata = dict(self.metadata) if self.metadata is not None else {}

        if default is None:
            metadata.pop(DEFAULT_VALUE_KEY, None)
        else:
            metadata[DEFAULT_VALUE_KEY] = json_module.dumps(
                default,
                safe=False,
                to_bytes=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )

        object.__setattr__(self, "metadata", metadata or None)
        return self

    @property
    def partition_by(self) -> bool:
        return self._tag_flag(b"partition_by")

    @partition_by.setter
    def partition_by(self, value: bool) -> None:
        if value:
            self._set_tag_value(b"partition_by", True)

    @property
    def cluster_by(self) -> bool:
        return self._tag_flag(b"cluster_by")

    @cluster_by.setter
    def cluster_by(self, value: bool) -> None:
        if value:
            self._set_tag_value(b"cluster_by", True)

    @property
    def primary_key(self) -> bool:
        return self._tag_flag(b"primary_key")

    @primary_key.setter
    def primary_key(self, value: bool) -> None:
        if value:
            self._set_tag_value(b"primary_key", True)

    def with_primary_key(self, value: bool = True, inplace: bool = True) -> "Field":
        if value:
            if inplace:
                self._set_tag_value(b"primary_key", True)
                return self
            else:
                return self.copy(tags={b"primary_key": True})
        return self

    @property
    def foreign_key(self) -> str | None:
        return self._tag_text(b"foreign_key")

    def copy(
        self,
        *,
        name: str | None = None,
        dtype: DataType | type[DataType] | pa.DataType | None = None,
        nullable: bool | None = None,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
        default: Any = None,
    ) -> "Field":
        keep_default = self.default if default is None else default
        return Field(
            name=self.name if name is None else name,
            dtype=self.dtype if dtype is None else DataType.from_any(dtype),
            nullable=self.nullable if nullable is None else bool(nullable),
            metadata=(
                (dict(self.metadata) if self.metadata is not None else None)
                if metadata is None and tags is None
                else _normalize_metadata(metadata, tags=tags)
            ),
            default=keep_default,
        )

    def autotag(self) -> "Field":
        return self

    @classmethod
    def from_any(cls, obj: Any) -> "Field":
        if isinstance(obj, cls):
            return obj

        ns, _ = ObjectSerde.module_and_name(obj)

        if ns.startswith("yggdrasil"):
            if isinstance(obj, DataType):
                return cls(
                    name=DEFAULT_FIELD_NAME,
                    dtype=obj,
                    nullable=True,
                )

            from .schema import Schema

            if isinstance(obj, Schema):
                return obj.to_field()
        elif ns.startswith("pyarrow"):
            return cls.from_arrow(obj)
        elif ns.startswith("polars"):
            return cls.from_polars(obj)
        elif ns.startswith("pandas"):
            return cls.from_pandas(obj)
        elif ns.startswith("pyspark"):
            return cls.from_spark(obj)
        elif ns.startswith("databricks"):
            return cls.from_databricks(obj)

        if isinstance(obj, type):
            return cls.from_pytype(obj)

        if is_dataclass(obj):
            return cls.from_dataclass(obj)

        if isinstance(obj, str):
            return cls.from_str(obj)

        if isinstance(obj, Mapping):
            return cls.from_dict(obj)

        if hasattr(obj, "type_text"):
            return cls.from_databricks(obj)

        if hasattr(obj, "value"):
            return cls.from_any(obj.value)

        raise TypeError(f"Cannot build Field from {type(obj).__name__}")

    @classmethod
    def from_(cls, obj: Any) -> "Field":
        return cls.from_any(obj)

    @classmethod
    def from_list(cls, value: list) -> "Field":
        if not value:
            raise ValueError("Cannot build Field from empty list")

        try:
            table = pa.table(value)
            return cls.from_arrow(table)
        except Exception:
            pass

        for item in range(min(len(value), 50)):
            try:
                return cls.from_any(value[item])
            except Exception:
                pass

        raise ValueError("Cannot build Field from list with no valid items")

    @classmethod
    def from_pytype(
        cls,
        hint: Any,
        *,
        name: str | None = None,
        nullable: bool | None = None,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
        default: Any = None,
    ) -> "Field":
        if isinstance(hint, str):
            parsed = ParsedDataType.parse(hint)
            resolved_name = name or parsed.name or _default_name(hint)
            resolved_nullable = parsed.nullable if nullable is None else bool(nullable)

            if resolved_nullable is None:
                resolved_nullable = False if parsed.type_id == DataTypeId.NULL else True

            return cls(
                name=resolved_name,
                dtype=DataType.from_parsed(parsed),
                nullable=resolved_nullable,
                metadata=_normalize_metadata(metadata, tags=tags),
                default=default,
            )

        base_hint, inferred_nullable = _unwrap_nullable_hint(hint)
        resolved_nullable = inferred_nullable if nullable is None else bool(nullable)

        return cls(
            name=name or _default_name(base_hint),
            dtype=DataType.from_pytype(base_hint),
            nullable=resolved_nullable,
            metadata=_normalize_metadata(metadata, tags=tags),
            default=default,
        )

    @classmethod
    def from_dataclass(
        cls,
        hint: Any,
        name: str | None = None,
        nullable: bool | None = None,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
        default: Any = None,
    ) -> "Field":
        if not is_dataclass(hint):
            raise TypeError(f"Unsupported dataclass input: {hint!r}")
        elif not isinstance(hint, type):
            hint = hint.__class__

        dtype = StructType.from_dataclass(hint)

        return cls(
            name=name or hint.__name__,
            dtype=dtype,
            nullable=False if nullable is None else bool(nullable),
            metadata=_normalize_metadata(metadata, tags=tags),
            default=None,
        )

    @classmethod
    def from_dataclass_field(
        cls,
        value: dataclasses.Field,
        *,
        owner: type | None = None,
    ) -> "Field":
        default = None
        if value.default is not None:
            default = value.default
        elif value.default_factory is not None:  # type: ignore[attr-defined]
            try:
                default = value.default_factory()  # type: ignore[misc]
            except Exception:
                default = None

        resolved_hint = value.type

        if owner is not None:
            try:
                import typing as _typing

                resolved = _typing.get_type_hints(owner, include_extras=True)
                resolved_hint = resolved.get(value.name, value.type)
            except Exception:
                resolved_hint = value.type

        if isinstance(resolved_hint, str):
            parsed = ParsedDataType.parse(resolved_hint)
            inferred_nullable = bool(parsed.nullable)
            if default is None:
                inferred_nullable = True

            return cls(
                name=value.name,
                dtype=DataType.from_parsed(parsed),
                nullable=inferred_nullable,
                metadata=None,
                default=default,
            )

        base_hint, inferred_nullable = _unwrap_nullable_hint(resolved_hint)

        if default is None:
            inferred_nullable = True

        return cls(
            name=value.name,
            dtype=DataType.from_pytype(base_hint),
            nullable=inferred_nullable,
            metadata=None,
            default=default,
        )

    @classmethod
    def from_str(cls, value: str) -> "Field":
        text = str(value).strip()
        if not text:
            raise ValueError("Field string cannot be empty")

        if text.startswith("{") and text.endswith("}"):
            payload = json_module.loads(text)
            if not isinstance(payload, Mapping):
                raise ValueError("Field JSON string must decode to an object")
            return cls.from_dict(payload)

        name_text, type_text = _split_field_shorthand(text)
        parsed = ParsedDataType.parse(type_text)
        parsed_name, name_nullable = _parse_field_name_token(name_text)

        nullable = parsed.nullable
        if name_nullable is not None:
            nullable = name_nullable
        if nullable is None:
            nullable = False if parsed.type_id == DataTypeId.NULL else True

        return cls(
            name=parsed_name,
            dtype=DataType.from_parsed(parsed),
            nullable=nullable,
            metadata=None,
            default=None,
        )

    @classmethod
    def from_arrow(
        cls,
        value: pa.Field | pa.Schema | pa.DataType | Any,
    ) -> "Field":
        if not isinstance(value, pa.Field):
            if isinstance(value, pa.Field):
                return cls.from_arrow_field(value)
            if isinstance(value, pa.Schema):
                return cls.from_arrow_schema(value)
            if isinstance(value, pa.DataType):
                return cls.from_arrow_field(
                    pa.field(DEFAULT_FIELD_NAME, value, nullable=True, metadata=None)
                )
            if isinstance(value, (pa.Array, pa.ChunkedArray)):
                return cls.from_arrow_field(
                    pa.field(DEFAULT_FIELD_NAME, value.type, nullable=True, metadata=None)
                )

            if hasattr(value, "schema"):
                value = value.schema
            elif hasattr(value, "arrow_schema"):
                value = value.arrow_schema
            elif hasattr(value, "arrow_type"):
                value = value.arrow_type

            if callable(value):
                value = value()

            if isinstance(value, pa.Field):
                return cls.from_arrow_field(value)
            if isinstance(value, pa.Schema):
                return cls.from_arrow_schema(value)
            if isinstance(value, pa.DataType):
                return cls.from_arrow_field(
                    pa.field(DEFAULT_FIELD_NAME, value, nullable=True, metadata=None)
                )
            raise TypeError(f"Cannot build Field from {type(value).__name__}")

        return cls(
            name=value.name or DEFAULT_FIELD_NAME,
            dtype=DataType.from_arrow_type(value.type),
            nullable=value.nullable,
            metadata=_strip_internal_metadata(value.metadata),
        )

    @classmethod
    def from_arrow_schema(cls, value: pa.Schema):
        name = DEFAULT_FIELD_NAME
        if value.metadata:
            name = value.metadata.get(b"name", DEFAULT_FIELD_NAME.encode()).decode("utf-8")

        return cls(
            name=name,
            dtype=DataType.from_arrow_schema(value),
            nullable=False,
            metadata=_strip_internal_metadata(value.metadata),
        )

    @classmethod
    def from_arrow_field(
        cls,
        value: pa.Field,
    ) -> "Field":
        return cls(
            name=value.name or DEFAULT_FIELD_NAME,
            dtype=DataType.from_arrow_type(value.type),
            nullable=value.nullable,
            metadata=_strip_internal_metadata(value.metadata),
        )

    @classmethod
    def from_path(
        cls,
        path: Any,
        *,
        media: Any = None,
        path_io: Any = None,
    ) -> "Field":
        from .schema import Schema

        return Schema.from_path(
            path, media=media, path_io=path_io
        ).to_field()

    def to_arrow(self):
        return self.to_arrow_field()

    def to_arrow_type(self):
        return self.arrow_type

    def to_arrow_field(self) -> pa.Field:
        metadata = self.metadata
        if metadata is None:
            metadata = _attach_type_json_metadata(self.arrow_type, None)
        else:
            metadata = _attach_type_json_metadata(self.arrow_type, metadata)

        return pa.field(
            name=self.name,
            type=self.arrow_type,
            nullable=self.nullable,
            metadata=metadata,
        )

    @classmethod
    def from_pandas(
        cls,
        obj: Any = None,
    ) -> "Field":
        pd = get_pandas()

        if isinstance(obj, pd.DataFrame):
            return cls(
                name=DEFAULT_FIELD_NAME,
                dtype=DataType.from_pandas(obj),
                nullable=False,
                metadata=None,
            )

        if isinstance(obj, pd.Series):
            nullable = bool(obj.isna().any())
            return cls(
                name=obj.name or DEFAULT_FIELD_NAME,
                dtype=DataType.from_pandas(obj),
                nullable=nullable,
                metadata=None,
            )

        if isinstance(obj, pd.Index):
            nullable = bool(obj.hasnans) if hasattr(obj, "hasnans") else False
            return cls(
                name=obj.name or DEFAULT_FIELD_NAME,
                dtype=DataType.from_pandas(obj),
                nullable=nullable,
                metadata=None,
            )

        return cls(
            name=_default_name(obj),
            dtype=DataType.from_pandas(obj),
            nullable=obj is pd.NA,
            metadata=None,
        )

    @classmethod
    def from_databricks(cls, value: "CatalogColumnInfo | SQLColumnInfo") -> "Field":
        dtype_payload = None
        metadata = {}

        for key in ("type_json", "type_text", "type_name", "type"):
            if hasattr(value, key):
                dtype_payload = getattr(value, key)
                if dtype_payload:
                    break

        parsed = DataType.from_any(dtype_payload)

        if hasattr(value, "nullable"):
            nullable = True if value.nullable is None else bool(value.nullable)
        else:
            nullable = True

        if hasattr(value, "comment") and value.comment:
            metadata[b"comment"] = value.comment.encode("utf-8")

        return cls(
            name=value.name,
            dtype=parsed,
            nullable=nullable,
            metadata=metadata,
        )

    def to_databricks_ddl(
        self,
        *,
        put_name: bool = True,
        put_not_null: bool = True,
        put_comment: bool = True,
    ) -> str:
        from yggdrasil.databricks.sql.sql_utils import escape_sql_string, quote_ident

        name_str = f"{quote_ident(self.name)} " if put_name else ""
        nullable_str = " NOT NULL" if put_not_null and not self.nullable else ""

        comment_str = ""
        if put_comment and self.metadata and b"comment" in self.metadata:
            comment = (self.metadata[b"comment"] or b"").decode("utf-8")
            comment_str = f" COMMENT '{escape_sql_string(comment)}'"

        if not pa.types.is_nested(self.arrow_type):
            dtype = DataType.from_arrow_type(self.arrow_type)
            if isinstance(dtype, type) and issubclass(dtype, DataType):
                dtype = dtype()
            sql_type = dtype.to_databricks_ddl()
            return f"{name_str}{sql_type}{nullable_str}{comment_str}"

        if pa.types.is_struct(self.arrow_type):
            struct_body = ", ".join(
                Field.from_arrow(child).to_databricks_ddl(put_comment=False)
                for child in self.arrow_type
            )
            return f"{name_str}STRUCT<{struct_body}>{nullable_str}{comment_str}"

        if pa.types.is_map(self.arrow_type):
            map_type: pa.MapType = self.arrow_type
            key_type = Field.from_arrow(map_type.key_field).to_databricks_ddl(
                put_name=False,
                put_comment=False,
                put_not_null=False,
            )
            val_type = Field.from_arrow(map_type.item_field).to_databricks_ddl(
                put_name=False,
                put_comment=False,
                put_not_null=False,
            )
            return f"{name_str}MAP<{key_type}, {val_type}>{nullable_str}{comment_str}"

        if pa.types.is_list(self.arrow_type) or pa.types.is_large_list(self.arrow_type):
            list_type: pa.ListType = self.arrow_type
            elem_type = Field.from_arrow(list_type.value_field).to_databricks_ddl(
                put_name=False,
                put_comment=False,
                put_not_null=False,
            )
            return f"{name_str}ARRAY<{elem_type}>{nullable_str}{comment_str}"

        raise TypeError(f"Cannot make Databricks DDL from nested type: {self.arrow_type}")

    @classmethod
    def from_parsed(
        cls,
        parsed: ParsedDataType,
        *,
        name: str | None = None,
        nullable: bool | None = None,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
        default: Any = None,
    ) -> "Field":
        if not isinstance(parsed, ParsedDataType):
            raise TypeError(
                f"Field.from_parsed expects ParsedDataType; got {type(parsed).__name__}"
            )

        resolved_name = name or parsed.name or DEFAULT_FIELD_NAME
        resolved_nullable = parsed.nullable if nullable is None else bool(nullable)

        if resolved_nullable is None:
            resolved_nullable = False if parsed.type_id == DataTypeId.NULL else True

        return cls(
            name=resolved_name,
            dtype=DataType.from_parsed(parsed),
            nullable=resolved_nullable,
            metadata=_normalize_metadata(metadata, tags=tags),
            default=default,
        )

    def to_dict(self) -> dict[str, Any]:
        dtype = self.dtype.to_dict()

        out = dict(
            name=self.name,
            dtype=dtype,
            nullable=self.nullable,
            metadata=self.metadata,
        )

        return out

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Field":
        if not value:
            raise ValueError(
                f"Cannot build {cls.__name__} from empty dictionary {value!r}"
            )

        for key in ("dtype", "type_json", "type_text", "type_name", "type"):
            dtype_payload = value.get(key)
            if dtype_payload:
                break

        if dtype_payload is None:
            raise ValueError(
                f"Cannot find a valid 'dtype' key in the provided dictionary: {value}"
            )

        metadata = _normalize_metadata(value.get("metadata"), tags=None)
        parsed = DataType.from_any(dtype_payload)
        nullable = bool(value.get("nullable", True))

        return cls(
            name=value["name"].strip(),
            dtype=parsed,
            nullable=nullable,
            metadata=metadata,
            default=None,
        )

    def to_json(
        self,
        to_bytes: bool = False
    ) -> str:
        payload = self.to_dict()

        return json_module.dumps(
            payload,
            safe=False,
            to_bytes=to_bytes,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, value: Any) -> "Field":
        if isinstance(value, (bytes, bytearray, memoryview)):
            value = bytes(value).decode("utf-8")

        if not isinstance(value, str):
            raise TypeError(
                f"Field.from_json expects str or bytes-like input; got {type(value).__name__}"
            )

        loaded = json_module.loads(value)

        if isinstance(loaded, Mapping):
            return cls.from_dict(loaded)
        elif isinstance(loaded, str):
            return cls.from_str(loaded)
        else:
            raise TypeError(f"Cannot build Field from {type(loaded).__name__}")

    @classmethod
    def from_polars(
        cls,
        obj: Any = None,
    ) -> "Field":
        pl = get_polars()

        if isinstance(obj, pl.Field):
            return cls.from_polars_field(obj)
        if isinstance(obj, pl.Schema):
            return cls.from_polars_schema(obj)
        if isinstance(obj, pl.DataFrame):
            return cls.from_polars_schema(obj.schema)
        if isinstance(obj, pl.LazyFrame):
            schema = obj.collect_schema()
            if isinstance(schema, pl.Schema):
                return cls.from_polars_schema(schema)
        if isinstance(obj, pl.Series):
            return cls.from_polars_series(obj)

        dtype = DataType.from_polars(obj)

        return cls(
            name=DEFAULT_FIELD_NAME,
            dtype=dtype,
            nullable=True,
            metadata={},
        )

    @classmethod
    def from_polars_series(cls, series: "polars.Series") -> "Field":
        dtype = DataType.from_polars_type(series.dtype)
        nullable = bool(series.null_count() > 0)
        return cls(
            name=series.name,
            dtype=dtype,
            nullable=nullable,
            metadata={},
        )

    @classmethod
    def from_polars_field(cls, value: "polars.Field") -> "Field":
        return cls(
            name=value.name,
            dtype=DataType.from_polars_type(value.dtype),
            nullable=True,
            metadata={},
        )

    @classmethod
    def from_polars_schema(cls, value: "polars.Schema") -> "Field":
        return cls(
            name=DEFAULT_FIELD_NAME,
            dtype=DataType.from_polars_schema(value),
            nullable=False,
        )

    def to_polars_field(self) -> "polars.Field":
        pl = get_polars()
        built = pl.Field(self.name, self.dtype.to_polars())
        try:
            built.nullable = self.nullable
        except AttributeError:
            pass
        try:
            built.metadata = self.metadata
        except AttributeError:
            pass
        return built

    @classmethod
    def from_spark(
        cls,
        obj: Any = None,
    ) -> "Field":
        _psql = get_spark_sql()

        if not isinstance(obj, _psql.types.StructField):
            if isinstance(obj, _psql.DataFrame):
                obj = _psql.types.StructField(
                    DEFAULT_FIELD_NAME,
                    obj.schema,
                    nullable=False,
                    metadata=None,
                )
            elif isinstance(obj, _psql.types.StructType):
                obj = _psql.types.StructField(
                    DEFAULT_FIELD_NAME,
                    obj,
                    nullable=True,
                    metadata=None,
                )
            else:
                raise TypeError(f"Cannot build {cls.__name__} from {type(obj).__name__}")

        return cls.from_spark_field(obj)

    @classmethod
    def from_spark_field(cls, value: "pst.StructField") -> "Field":
        return cls(
            name=value.name,
            dtype=DataType.from_spark(value.dataType),
            nullable=value.nullable,
            metadata=_normalize_metadata(value.metadata, tags=None),
        )

    def to_pyspark_field(self) -> "pst.StructField":
        import pyspark.sql as pyspark_sql

        metadata = (
            {
                (
                    key.decode("utf-8")
                    if isinstance(key, bytes)
                    else str(key)
                ): (
                    value.decode("utf-8")
                    if isinstance(value, bytes)
                    else str(value)
                )
                for key, value in self.metadata.items()
                if key != b"to_json"
            }
            if self.metadata
            else {}
        )
        return pyspark_sql.types.StructField(
            self.name,
            self.dtype.to_spark(),
            self.nullable,
            metadata=metadata,
        )

    def to_schema(self) -> "Schema":
        from .schema import Schema

        base = self.to_struct()
        metadata = base.metadata.copy() if base.metadata else {}

        if base.name != DEFAULT_FIELD_NAME:
            metadata[b"name"] = base.name.encode("utf-8")
        return Schema.from_any_fields(base.children_fields, metadata=metadata)

    def to_struct(self):
        dtype = self.dtype.to_struct(name=self.name)
        return Field(self.name, dtype, self.nullable, self.metadata)

    def cast_arrow_array(
        self,
        array: pa.Array | pa.ChunkedArray,
        options: "CastOptions | None" = None,
        **more,
    ):
        # Object target is a variant — never touch the values.
        if self.dtype.type_id == DataTypeId.OBJECT:
            return array
        options = get_cast_options_class().check(options=options, **more)
        casted = self.dtype.cast_arrow_array(array, options=options.with_target(self))
        filled = self.fill_arrow_array_nulls(casted, default_scalar=self.default_arrow_scalar)
        return filled

    def cast_arrow_tabular(
        self,
        table: pa.Table | pa.RecordBatch,
        options: "CastOptions | None" = None,
        **more,
    ):
        if self.dtype.type_id == DataTypeId.OBJECT:
            return table
        options = get_cast_options_class().check(options=options, **more)
        return self.to_struct().dtype.cast_arrow_tabular(table, options=options.with_target(self))

    def cast_polars_series(
        self,
        series: "polars.Series",
        options: "CastOptions | None" = None,
        **more,
    ):
        if self.dtype.type_id == DataTypeId.OBJECT:
            return series
        options = get_cast_options_class().check(options=options, **more)
        casted = self.dtype.cast_polars_series(series, options=options.with_target(self))
        filled = self.fill_polars_array_nulls(casted, default_scalar=self.default_arrow_scalar)
        return filled.alias(self.name) if self.name and self.name != DEFAULT_FIELD_NAME else filled

    def cast_polars_expr(
        self,
        series: "polars.Expr",
        options: "CastOptions | None" = None,
        **more,
    ):
        if self.dtype.type_id == DataTypeId.OBJECT:
            return series
        options = get_cast_options_class().check(options=options, **more)
        casted = self.dtype.cast_polars_expr(series, options=options.with_target(self))
        filled = self.fill_polars_array_nulls(casted, default_scalar=self.default_arrow_scalar)
        return filled.alias(self.name) if self.name and self.name != DEFAULT_FIELD_NAME else filled

    def cast_polars_tabular(
        self,
        data: "polars.DataFrame | polars.LazyFrame",
        options: "CastOptions | None" = None,
        **more,
    ):
        if self.dtype.type_id == DataTypeId.OBJECT:
            return data
        options = get_cast_options_class().check(options=options, **more)
        return self.to_struct().dtype.cast_polars_tabular(data, options=options.with_target(self))

    def cast_pandas_series(
        self,
        series: "pd.Series",
        options: "CastOptions | None" = None,
        **more,
    ):
        if self.dtype.type_id == DataTypeId.OBJECT:
            return series
        options = get_cast_options_class().check(options=options, **more)
        casted = self.dtype.cast_pandas_series(series, options=options.with_target(self))
        filled = self.fill_pandas_series_nulls(casted, default_scalar=self.default_arrow_scalar)
        if self.name and self.name != DEFAULT_FIELD_NAME:
            filled.name = self.name
        return filled

    def cast_pandas_tabular(
        self,
        data: "pd.DataFrame",
        options: "CastOptions | None" = None,
        **more,
    ):
        if self.dtype.type_id == DataTypeId.OBJECT:
            return data
        options = get_cast_options_class().check(options=options, **more)
        return self.dtype.cast_pandas_tabular(data, options=options.with_target(self))

    def cast_spark_column(
        self,
        column: "ps.Column",
        options: "CastOptions | None" = None,
        **more,
    ):
        if self.dtype.type_id == DataTypeId.OBJECT:
            return column
        options = get_cast_options_class().check(options=options, **more)
        options = options.with_target(self).check_source(column)
        casted = self.dtype.cast_spark_column(column, options=options)
        filled = self.fill_spark_column_nulls(casted, default_scalar=self.default_arrow_scalar)
        return filled.alias(self.name) if self.name and self.name != DEFAULT_FIELD_NAME else filled

    def cast_spark_tabular(
        self,
        data: "ps.DataFrame",
        options: "CastOptions | None" = None,
        **more,
    ):
        if self.dtype.type_id == DataTypeId.OBJECT:
            return data
        options = get_cast_options_class().check(options=options, **more)
        return self.dtype.cast_spark_tabular(data, options=options.with_target(self))

    def fill_arrow_nulls(
        self,
        obj: pa.Array | pa.ChunkedArray,
        *,
        default_scalar: pa.Scalar | None = None,
    ):
        if isinstance(obj, (pa.Array, pa.ChunkedArray)):
            return self.fill_arrow_array_nulls(
                obj,
                default_scalar=default_scalar,
            )
        raise TypeError(f"Cannot fill nulls in {type(obj).__name__}")

    def fill_arrow_array_nulls(
        self,
        array: pa.Array | pa.ChunkedArray,
        *,
        default_scalar: pa.Scalar | None = None,
    ):
        if default_scalar is None and self.has_default:
            default_scalar = self.default_arrow_scalar

        return self.dtype.fill_arrow_array_nulls(
            array,
            nullable=self.nullable,
            default_scalar=default_scalar,
        )

    def default_arrow_array(
        self,
        size: int = 0,
        memory_pool: Optional[pa.MemoryPool] = None,
        chunks: Optional[list[int]] = None,
        default_scalar: Optional[pa.Scalar] = None,
    ) -> pa.Array | pa.ChunkedArray:
        if default_scalar is None and self.has_default:
            default_scalar = self.default_arrow_scalar

        return self.dtype.default_arrow_array(
            nullable=self.nullable,
            size=size,
            memory_pool=memory_pool,
            chunks=chunks,
            default_scalar=default_scalar,
        )

    def fill_polars_array_nulls(
        self,
        series: "polars.Series | polars.Expr",
        *,
        default_scalar: pa.Scalar | None = None,
    ):
        if default_scalar is None and self.has_default:
            default_scalar = self.default_arrow_scalar

        return self.dtype.fill_polars_array_nulls(
            series,
            nullable=self.nullable,
            default_scalar=default_scalar,
        )

    def fill_pandas_series_nulls(
        self,
        series: "pd.Series",
        *,
        default_scalar: pa.Scalar | None = None,
    ):
        if default_scalar is None and self.has_default:
            default_scalar = self.default_arrow_scalar

        return self.dtype.fill_pandas_series_nulls(
            series,
            nullable=self.nullable,
            default_scalar=default_scalar,
        )

    def fill_spark_column_nulls(
        self,
        column: "ps.Column",
        *,
        default_scalar: pa.Scalar | None = None,
    ) -> "ps.Column":
        if default_scalar is None and self.has_default:
            default_scalar = self.default_arrow_scalar

        return self.dtype.fill_spark_column_nulls(
            column,
            nullable=self.nullable,
            default_scalar=default_scalar,
        )

    def default_polars_series(
        self,
        *,
        size: int = 0,
        name: str | None = None,
    ):
        return self.dtype.default_polars_series(
            value=self.default,
            nullable=self.nullable,
            size=size,
            name=self.name if name is None else name,
        )

    def default_polars_expr(self, alias: str | None = None):
        return self.dtype.default_polars_expr(value=self.default, nullable=self.nullable, alias=alias or self.name)

    def default_pandas_series(
        self,
        *,
        size: int = 0,
        index: Any = None,
        name: str | None = None,
    ):
        return self.dtype.default_pandas_series(
            value=self.default,
            nullable=self.nullable,
            size=size,
            index=index,
            name=self.name if name is None else name,
        )

    def default_spark_column(self, alias: str | None = None):
        s = self.dtype.default_spark_column(value=self.default, nullable=self.nullable)
        return s.alias(alias) if alias else s.alias(self.name)


@register_converter(Any, Field)
def any_to_field(obj: Any, _: Any):
    return Field.from_any(obj)