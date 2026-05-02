from __future__ import annotations

import dataclasses
import itertools
import types
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, is_dataclass
from typing import TYPE_CHECKING, Any, Annotated, Optional, Union, get_args, get_origin, Iterator, Generator, AnyStr

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
from yggdrasil.io.enums import Mode
from yggdrasil.lazy_imports import path_class, schema_class, pandas_module
from yggdrasil.pickle.serde import ObjectSerde
from .cast.registry import register_converter
from .data_utils import get_cast_options_class, safe_constraint_name
from .types import NullType, ObjectType
from .types.base import DataType
from .types.nested import StructType
from .types.support import get_pandas, get_polars, get_spark_sql

if TYPE_CHECKING:
    import pandas as pd
    import polars
    import pyspark.sql as ps
    import pyspark.sql.types as pst
    from yggdrasil.data.options import CastOptions
    from yggdrasil.data.schema import Schema


__all__ = [
    "Field",
    "field",
    "_normalize_metadata",
    "_to_bytes",
    "_merge_metadata_and_tags",
]


# ======================================================================
# Module-level constants & private helpers
# ======================================================================

_TYPE_JSON_METADATA_KEY = b"type_json"
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


# ======================================================================
# Public factory — `field(...)` shorthand constructor
# ======================================================================

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


@dataclass(frozen=True, slots=True, init=False, repr=False)
class Field(BaseMetadata, BaseChildrenFields):
    name: str
    dtype: DataType
    nullable: bool = True
    metadata: dict[bytes, bytes] | None = None

    def __repr__(self):
        return self.pretty_format()

    def __str__(self):
        return self.pretty_format()

    @classmethod
    def make_default_field(
        cls,
        name: str = "",
        dtype: DataType = ObjectType(),
        nullable: bool = True,
        metadata: dict[bytes, bytes] | None = None,
        tags: dict[bytes, bytes] | None = None,
        default: Any = None,
    ):
        return cls(
            name=name,
            dtype=dtype,
            nullable=nullable,
            metadata=_normalize_metadata(metadata, tags=tags, default_value=default),
        )

    def pretty_format(self, indent: int = 2, level: int = 0) -> str:
        """Pretty-print this field with the header on one line and the dtype below.

        Layout is uniform across flat and nested dtypes::

            'name'[ not null][ {metadata}]
              <dtype block at level + 1>

        ``indent`` is the per-level step in spaces; ``level`` is the
        current depth. The header carries the name, the ``not null``
        marker, and the metadata repr; the dtype renders on the next
        line(s) at ``level + 1`` so its body always sits one step in
        from the field name.

        Examples::

            >>> print(field("id", "int64", nullable=False).pretty_format())
            'id' not null
              int64

            >>> print(field("user", StructType.from_fields([
            ...     field("id", "int64"),
            ...     field("email", "string"),
            ... ])).pretty_format())
            'user'
              struct
                'id'
                  int64,
                'email'
                  string
              >
        """
        pad = " " * (indent * level)

        header = f"{pad}field: {self.name!r}"
        suffix = ""
        if not self.nullable:
            suffix += " not null"

        comment = self.comment
        if comment:
            suffix += f" {comment!r}"

        if self.type_id.is_nested:
            dtype_str = self.dtype.pretty_format(indent=indent, level=level + 1)
            return f"{header}{suffix}\n{dtype_str}"
        else:
            dtype_str = self.dtype.pretty_format()
            return f"{header} {dtype_str}{suffix}"

    # ==================================================================
    # Dunder / identity
    # ==================================================================

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

    def equals(
        self,
        other: Any,
        check_names: bool = True,
        check_dtypes: bool = True,
        check_nullable: bool = True,
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

        if not self.name:
            self.with_name(other.name, inplace=True)
        elif not other.name:
            other.with_name(self.name, inplace=True)

        if check_names and self.name != other.name:
            return False

        if check_nullable and self.nullable != other.nullable:
            return False

        if check_metadata and self.metadata != other.metadata:
            return False

        if check_dtypes and not self.dtype.equals(
            other.dtype,
            check_metadata=check_metadata,
        ):
            return False

        return True

    # ==================================================================
    # Properties — dtype projection, defaults, children
    # ==================================================================

    @property
    def has_default(self) -> bool:
        return self.metadata.get(DEFAULT_VALUE_KEY) is not None if self.metadata else False

    @property
    def default(self):
        if self.metadata is not None:
            default = self.metadata.get(DEFAULT_VALUE_KEY)

            if default is None:
                return self.dtype.default_pyobj(nullable=self.nullable)

            try:
                default = json_module.loads(default, safe=False)
            except Exception as e:
                raise ValueError(f"Could not parse default value {default!r} for {self!r}: {e}") from e

            return self.dtype.convert_pyobj(default, nullable=self.nullable, safe=False)
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
    def type_id(self) -> DataTypeId:
        return self.dtype.type_id

    @property
    def children_fields(self) -> list["Field"]:
        return self.dtype.children_fields

    @property
    def arrow_type(self) -> pa.DataType:
        return self.dtype.to_arrow()

    def _empty_tags(self) -> dict[bytes, bytes]:
        return {}

    # ==================================================================
    # Tag flags — partition_by / cluster_by / primary_key / foreign_key
    # ==================================================================

    def _with_tag_flag(self, key: bytes, value: bool, inplace: bool) -> "Field":
        if inplace:
            if value:
                self._set_tag_value(key, True)
            else:
                self._unset_tag_value(key)
            return self
        else:
            return self.copy()._with_tag_flag(key, value, inplace=True)

    def with_partition_by(self, value: bool = True, inplace: bool = True) -> "Field":
        return self._with_tag_flag(b"partition_by", value, inplace)

    def with_cluster_by(self, value: bool = True, inplace: bool = True) -> "Field":
        return self._with_tag_flag(b"cluster_by", value, inplace)

    def with_primary_key(self, value: bool = True, inplace: bool = False) -> "Field":
        return self._with_tag_flag(b"primary_key", value, inplace)

    def with_foreign_key(self, value: bool = True, inplace: bool = False) -> "Field":
        return self._with_tag_flag(b"foreign_key", value, inplace)

    def with_constraint_key(self, value: bool = True, inplace: bool = False) -> "Field":
        return self._with_tag_flag(b"constraint_key", value, inplace)

    def with_sorted(self, value: bool = True, inplace: bool = False) -> "Field":
        return self._with_tag_flag(b"sorted", value, inplace)

    # ==================================================================
    # Builders — `with_*` mutators, `copy`, `merge_with`, `autotag`
    # ==================================================================

    def with_name(self, name: str, inplace: bool = False) -> "Field":
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
        if dtype == self.dtype:
            return self

        dtype = DataType.from_any(dtype)

        if inplace:
            object.__setattr__(self, "dtype", dtype)
            return self
        return self.copy(dtype=dtype)

    def with_nullable(self, nullable: bool, inplace: bool = True) -> "Field":
        if nullable == self.nullable:
            return self

        if inplace:
            object.__setattr__(self, "nullable", bool(nullable))
            return self
        return self.copy(nullable=bool(nullable))

    def with_metadata(
        self,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
        default: Any = None,
        inplace: bool = True,
    ):
        if metadata or tags or default is not None:
            normalized = _normalize_metadata(
                metadata,
                tags=tags,
                default_value=default
            )
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

    def copy(
        self,
        *,
        name: str | None = None,
        dtype: DataType | type[DataType] | pa.DataType | None = None,
        nullable: bool | None = None,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
    ) -> "Field":
        return Field(
            name=self.name if name is None else name,
            dtype=self.dtype if dtype is None else DataType.from_any(dtype),
            nullable=self.nullable if nullable is None else bool(nullable),
            metadata=(
                (dict(self.metadata) if self.metadata is not None else None)
                if metadata is None and tags is None
                else _normalize_metadata(metadata, tags=tags)
            )
        )

    def merge_with(
        self,
        other: "Field",
        *,
        inplace: bool = False,
        mode: Mode | None = None,
        downcast: bool = False,
        upcast: bool = False,
        merge_dtype: bool = True,
        merge_nullable: bool = True,
        merge_metadata: bool = True
    ):
        other = self.from_any(other)
        if self == other:
            return self

        mode = Mode.from_(mode, default=Mode.AUTO)

        if mode is Mode.AUTO:
            name = self.name or other.name
            nullable = not (not self.nullable or not other.nullable)
            metadata = {
                **(other.metadata or {}),
                **(self.metadata or {}),
            }

            dtype = self.dtype.merge_with(
                other.dtype, mode=mode, downcast=downcast, upcast=upcast
            )

            if inplace:
                object.__setattr__(self, "name", name)
                object.__setattr__(self, "dtype", dtype)
                object.__setattr__(self, "nullable", bool(nullable))
                object.__setattr__(self, "metadata", metadata)
                return self

            return self.copy(
                name=name,
                dtype=dtype,
                nullable=nullable,
                metadata=metadata,
            )
        elif mode is Mode.IGNORE:
            return self
        elif mode is Mode.OVERWRITE:
            if inplace:
                object.__setattr__(self, "dtype", other.dtype)
                object.__setattr__(self, "nullable", other.nullable)
                object.__setattr__(self, "metadata", other.metadata)
                return self
            return self.copy(
                name=other.name or self.name,
                dtype=other.dtype,
                nullable=other.nullable,
                metadata=other.metadata,
            )

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
            dtype = self.dtype

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

    def autotag(self) -> "Field":
        """Stamp this field with tags derived from its dtype and name.

        Writes Databricks-friendly auto-tags in place:

        - Everything from :meth:`DataType.autotag` (``kind`` plus dtype
          detail like ``unit`` / ``tz`` / ``precision`` / ``scale`` /
          ``signed`` / ``iso`` / ``srid``).
        - ``nullable`` for data-quality policies.
        - Name-based heuristics for governance: ``role=identifier`` for
          ``*_id`` / ``*_uuid``, ``role=audit_timestamp`` for ``created_at``
          patterns, plus ``pii`` / ``sensitive`` stamps for columns that
          obviously carry personal or credential data.

        Returns self for fluent chaining. Idempotent when the dtype and name
        are unchanged.
        """
        tags: dict[bytes, bytes] = dict(self.dtype.autotag())
        if not self.nullable:
            tags[b"nullable"] = b"false"

        self.update_tags(tags)
        return self

    # ==================================================================
    # Peek — sample an iterable / list for a one-shot field inference
    # ==================================================================

    @classmethod
    def peek_from(cls, obj: Any) -> tuple[Any, "Field"]:
        if isinstance(obj, (Iterator, Generator)):
            first = next(obj, None)

            if first is None:
                return None, cls(name="", dtype=NullType())

            obj = itertools.chain((first,), obj)

            return obj, cls.from_(first)
        elif isinstance(obj, list):
            if not obj:
                return obj, cls(name="", dtype=NullType())
            return obj, cls.from_(obj[0])
        else:
            return obj, cls.from_(obj)

    # ==================================================================
    # Constructors — generic dispatch entry points
    # ==================================================================

    @classmethod
    def make_constraint_field(
        cls,
        fields: Iterable["Field"],
        name: str = "",
        prefix: str = "",
        default: Any = ...,
        name_limit: int = 256
    ) -> "Field | None":
        if not fields:
            if default is ...:
                raise ValueError(
                    f"No fields specified as primary key for struct type in {fields!r}"
                )
            return default

        fields = [cls.from_(_) for _ in fields]
        name = name or safe_constraint_name(
            [_.name for _ in fields],
            prefix=prefix,
            limit=name_limit
        )

        if len(fields) == 1:
            keep = fields[0].with_name(name)
        else:
            keep = cls(
                name=name,
                dtype=StructType.from_fields(fields),
                nullable=False
            )

        return (
            keep
            .with_name(name=name)
            .with_constraint_key(True)
        )

    @classmethod
    def from_any(
        cls,
        obj: Any,
        *,
        name: str | None = None,
        metadata: dict | None = None
    )-> "Field":
        if isinstance(obj, cls):
            return obj

        if isinstance(obj, DataType):
            return cls(
                name=name or DEFAULT_FIELD_NAME,
                dtype=obj,
                metadata=metadata
            )

        ns, _ = ObjectSerde.module_and_name(obj)

        if ns.startswith("yggdrasil"):
            if isinstance(obj, schema_class()):
                return obj.to_field()
        elif ns.startswith("pyarrow"):
            return cls.from_arrow(obj)
        elif ns.startswith("polars"):
            return cls.from_polars(obj)
        elif ns.startswith("pandas"):
            return cls.from_pandas(obj)
        elif ns.startswith("pyspark"):
            return cls.from_spark(obj)

        pc = path_class()
        if pc.is_pathish(obj):
            try:
                return pc.from_(obj).as_media().collect_schema().to_field()
            except Exception:
                pass

        if isinstance(obj, type):
            return cls.from_pytype(obj)

        if hasattr(obj, "collect_schema"):
            return cls.from_any(obj.collect_schema())

        if callable(obj):
            return cls.from_any(obj())

        if is_dataclass(obj):
            return cls.from_dataclass(obj)

        if isinstance(obj, str):
            return cls.from_str(obj)

        if isinstance(obj, Mapping):
            return cls.from_dict(obj)

        if hasattr(obj, "value"):
            return cls.from_any(obj.value)

        if isinstance(obj, (list, tuple)):
            return cls.from_list(obj)

        raise TypeError(f"Cannot build Field from {type(obj).__name__}")

    @classmethod
    def from_(
        cls,
        obj: Any,
        *,
        name: str | None = None
    ) -> "Field":
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

    # ------------------------------------------------------------------
    # Constructors — Python types, dataclasses, strings, dicts, JSON
    # ------------------------------------------------------------------

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
        default_value: Any = None,
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
            metadata=_normalize_metadata(metadata, tags=tags, default_value=default_value),
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

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], default: Any = ...) -> "Field":
        try:
            if not value:
                raise ValueError(
                    f"Cannot build {cls.__name__} from empty dictionary {value!r}"
                )

            name = value.get("name") or ""
            nullable = bool(value.get("nullable", True))
            dtype = (
                value.get(b"dtype")
                or value.get("dtype")
                or value.get(_TYPE_JSON_METADATA_KEY)
                or value.get(_TYPE_JSON_METADATA_KEY.decode())
            )
            if dtype:
                dtype = DataType.from_json(dtype)

            if dtype is None:
                for key in ("type_text", "type_json", "type"):
                    found = value.get(key)
                    if found is not None:
                        try:
                            dtype = DataType.from_any(found)
                            break
                        except Exception:
                            pass

                if dtype is None:
                    raise ValueError(
                        f"Cannot build {cls.__name__} from dictionary without type: {value!r}"
                    )

            metadata = _normalize_metadata(value.get("metadata", {}), tags=None)

            return cls(
                name=name,
                dtype=dtype,
                nullable=nullable,
                metadata=metadata,
            )
        except Exception as e:
            if default is ...:
                raise ValueError(f"Cannot build {cls.__name__} from dictionary: {e}") from e
            return default

    @classmethod
    def from_json(cls, value: Any) -> "Field":
        if isinstance(value, (bytes, bytearray, memoryview)):
            value = bytes(value).decode("utf-8")

        if not isinstance(value, str):
            if isinstance(value, Mapping):
                return cls.from_dict(value)
            elif isinstance(value, (list, tuple)):
                return cls.from_list(value)
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
    def from_path(cls, path: Any) -> "Field":
        path = path_class().from_(path)
        return path.as_media().collect_schema().to_field()

    # ------------------------------------------------------------------
    # Constructors — arrow
    # ------------------------------------------------------------------

    @classmethod
    def from_arrow(
        cls,
        value: pa.Field | pa.Schema | pa.DataType | Any,
        from_metadata: bool = True,
    ) -> "Field":
        if not isinstance(value, pa.Field):
            if isinstance(value, pa.Field):
                return cls.from_arrow_field(value, from_metadata=from_metadata)
            if isinstance(value, pa.Schema):
                return cls.from_arrow_schema(value, from_metadata=from_metadata)
            if isinstance(value, pa.DataType):
                return cls.from_arrow_field(
                    pa.field(DEFAULT_FIELD_NAME, value, nullable=True, metadata=None),
                    from_metadata=from_metadata
                )
            if isinstance(value, (pa.Array, pa.ChunkedArray)):
                nullable = value.null_count > 0 or len(value) == 0

                return cls.from_arrow_field(
                    pa.field(
                        DEFAULT_FIELD_NAME, value.type,
                        nullable=nullable,
                        metadata=None
                    ),
                    from_metadata=from_metadata
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
                return cls.from_arrow_field(value, from_metadata=from_metadata)
            if isinstance(value, pa.Schema):
                return cls.from_arrow_schema(value, from_metadata=from_metadata)
            if isinstance(value, pa.DataType):
                return cls.from_arrow_field(
                    pa.field(DEFAULT_FIELD_NAME, value, nullable=True, metadata=None),
                    from_metadata=from_metadata
                )
            raise TypeError(f"Cannot build Field from {type(value).__name__}")

        return cls(
            name=value.name or DEFAULT_FIELD_NAME,
            dtype=DataType.from_arrow_type(value.type),
            nullable=value.nullable,
            metadata=_strip_internal_metadata(value.metadata),
        )

    @classmethod
    def from_arrow_schema(cls, value: pa.Schema, from_metadata: bool = True):
        if from_metadata and value.metadata:
            found = value.metadata.get(_TYPE_JSON_METADATA_KEY)
            if found:
                return cls.from_json(found)

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
    def from_arrow_field(cls, value: pa.Field, from_metadata: bool = True) -> "Field":
        if from_metadata and value.metadata:
            dtype = value.metadata.get(_TYPE_JSON_METADATA_KEY, None)
            if dtype is None:
                dtype = DataType.from_arrow_type(value.type)
            else:
                dtype = DataType.from_json(dtype)
        else:
            dtype = DataType.from_arrow_type(value.type)

        return cls(
            name=value.name or DEFAULT_FIELD_NAME,
            dtype=dtype,
            nullable=value.nullable,
            metadata=_strip_internal_metadata(value.metadata),
        )

    # ------------------------------------------------------------------
    # Constructors — pandas
    # ------------------------------------------------------------------

    @classmethod
    def from_pandas(cls, obj: Any = None) -> "Field":
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

    # ------------------------------------------------------------------
    # Constructors — polars
    # ------------------------------------------------------------------

    @classmethod
    def from_polars(cls, obj: Any = None) -> "Field":
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
        try:
            nullable = getattr(value, "nullable", True)
        except Exception:
            nullable = True

        try:
            metadata = getattr(value, "metadata", None)
            if metadata is not None:
                metadata = _normalize_metadata(metadata, tags=None)
        except Exception:
            metadata = None

        return cls(
            name=value.name,
            dtype=DataType.from_polars_type(value.dtype),
            nullable=nullable,
            metadata=metadata,
        )

    @classmethod
    def from_polars_schema(cls, value: "polars.Schema") -> "Field":
        return cls(
            name=DEFAULT_FIELD_NAME,
            dtype=DataType.from_polars_schema(value),
            nullable=False,
        )

    # ------------------------------------------------------------------
    # Constructors — spark
    # ------------------------------------------------------------------

    @classmethod
    def from_spark(cls, obj: Any = None, from_metadata: bool = True) -> "Field":
        _psql = get_spark_sql()

        if not isinstance(obj, _psql.types.StructField):
            if isinstance(obj, _psql.DataFrame):
                obj = _psql.types.StructField(
                    DEFAULT_FIELD_NAME,
                    obj.schema,
                    nullable=False,
                    metadata=None,
                )
            elif isinstance(obj, _psql.types.DataType):
                obj = _psql.types.StructField(
                    DEFAULT_FIELD_NAME,
                    obj,
                    nullable=True,
                    metadata=None,
                )
            else:
                raise TypeError(f"Cannot build {cls.__name__} from {type(obj).__name__}")

        return cls.from_spark_field(obj, from_metadata=from_metadata)

    @classmethod
    def from_spark_field(cls, value: "pst.StructField", from_metadata: bool = True) -> "Field":
        if from_metadata and value.metadata:
            dtype = value.metadata.get(_TYPE_JSON_METADATA_KEY, None)
            if dtype is None:
                dtype = DataType.from_spark(value.dataType)
            else:
                dtype = cls.from_json(dtype)
        else:
            dtype = DataType.from_spark(value.dataType)

        return cls(
            name=value.name,
            dtype=dtype,
            nullable=value.nullable,
            metadata=_normalize_metadata(value.metadata, tags=None),
        )

    # ==================================================================
    # Exporters — dict / JSON / arrow / polars / spark / databricks DDL
    # ==================================================================

    def to_dict(self) -> dict[str, Any]:
        dtype = self.dtype.to_dict()

        out = dict(
            name=self.name,
            dtype=dtype,
            nullable=self.nullable,
        )

        if self.metadata:
            out["metadata"] = {
                k.decode(): v.decode()
                for k, v in self.metadata.items()
            }

        return out

    def to_json(self, to_bytes: bool = False) -> AnyStr:
        payload = self.to_dict()

        return json_module.dumps(
            payload,
            safe=False,
            to_bytes=to_bytes,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def to_arrow(self):
        return self.to_arrow_field()

    def to_arrow_type(self):
        return self.arrow_type

    def to_arrow_field(self, dump_json: bool = True) -> pa.Field:
        metadata = self.metadata.copy() if self.metadata else {}

        if dump_json:
            metadata[_TYPE_JSON_METADATA_KEY] = self.dtype.to_json(to_bytes=True)

        return pa.field(
            name=self.name,
            type=self.arrow_type,
            nullable=self.nullable,
            metadata=metadata,
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

    def to_polars_flavor(self) -> "polars.Field":
        """Polars-native counterpart for this field — a ``pl.Field``."""
        return self.to_polars_field()

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

    def to_spark_flavor(self) -> "pst.StructField":
        """Spark-native counterpart for this field — a ``StructField``."""
        return self.to_pyspark_field()

    def to_schema(
        self,
        metadata: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
    ) -> "Schema":
        from .schema import Schema

        base = self.to_struct()
        final_metadata = base.metadata.copy() if base.metadata else {}
        new_metadata = _normalize_metadata(metadata, tags=tags)
        if new_metadata:
            final_metadata.update(new_metadata)

        if base.name != DEFAULT_FIELD_NAME:
            final_metadata[b"name"] = base.name.encode("utf-8")
        return Schema.from_any_fields(base.children_fields, metadata=final_metadata)

    def to_struct(self):
        dtype = self.dtype.to_struct(name=self.name)
        return Field(self.name, dtype, self.nullable, self.metadata)

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

    # ==================================================================
    # Cast — top-level dispatch (`cast` / engine-level `cast_*`)
    # ==================================================================
    #
    # Three granularities, from coarsest to finest:
    #
    # 1. :meth:`cast` — "cast whatever this is". Inspects the module of
    #    *obj* via :meth:`ObjectSerde.module_and_name`, routes to the
    #    engine dispatcher. Also recurses into plain iterators /
    #    iterables as a lazy generator.
    #
    # 2. :meth:`cast_arrow` / :meth:`cast_polars` / :meth:`cast_pandas`
    #    / :meth:`cast_spark` — "I know this is pyarrow/polars/pandas/
    #    spark; figure out the shape". Runs an isinstance walk within
    #    the engine's own types, routes to the narrow method.
    #
    # 3. :meth:`cast_arrow_tabular`, :meth:`cast_polars_series`, ... —
    #    the narrow methods below. They do the actual cast work, then
    #    delegate the post-cast null-fill + alias pass to the matching
    #    :meth:`finalize_*` method. ``self.dtype.type_id == OBJECT`` is
    #    the variant-column passthrough: a variant column must never
    #    be cast.

    def cast(
        self,
        obj: Any,
        options: "CastOptions | None" = None,
        **more: Any,
    ) -> Any:
        """Cast *obj* to this field using its native engine.

        Routing is by module prefix via :meth:`ObjectSerde.module_and_name`:

        * ``pyarrow.*`` → :meth:`cast_arrow`
        * ``polars.*``  → :meth:`cast_polars`
        * ``pandas.*``  → :meth:`cast_pandas`
        * ``pyspark.*`` → :meth:`cast_spark`
        * iterator / iterable → recurse per element (lazy generator)
        * everything else → :class:`TypeError`

        ``self.dtype.type_id == OBJECT`` is handled by the narrow
        methods — they pass *obj* through unchanged because a variant
        column must never be cast. No redundant guard here.
        """
        ns, _name = ObjectSerde.module_and_name(obj)

        if ns.startswith("pyarrow"):
            return self.cast_arrow(obj, options=options, **more)
        if ns.startswith("polars"):
            return self.cast_polars(obj, options=options, **more)
        if ns.startswith("pandas"):
            return self.cast_pandas(obj, options=options, **more)
        if ns.startswith("pyspark"):
            return self.cast_spark(obj, options=options, **more)

        # Iterator / iterable fallback — preserve laziness. An iterator
        # that yields pa.RecordBatch items passes through as a
        # generator, each batch cast on demand. str/bytes excluded —
        # they're iterable but never tabular.
        if isinstance(obj, Iterator):
            return (self.cast(item, options=options, **more) for item in obj)
        if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
            return (self.cast(item, options=options, **more) for item in obj)

        raise TypeError(
            f"Field.cast: unsupported input {type(obj).__name__!r} "
            f"(module={ns!r}). Expected pyarrow/polars/pandas/pyspark "
            "tabular / array / series / column, or an iterable of such."
        )

    def cast_arrow(
        self,
        obj: Any,
        options: "CastOptions | None" = None,
        **more: Any,
    ) -> Any:
        """Cast any pyarrow object — dispatch by shape.

        Table/RecordBatch → :meth:`cast_arrow_tabular`,
        Array/ChunkedArray → :meth:`cast_arrow_array`.
        """
        if isinstance(obj, (pa.Table, pa.RecordBatch)):
            return self.cast_arrow_tabular(obj, options=options, **more)
        if isinstance(obj, (pa.Array, pa.ChunkedArray)):
            return self.cast_arrow_array(obj, options=options, **more)
        raise TypeError(
            f"Field.cast_arrow: expected pa.Table / pa.RecordBatch / pa.Array / "
            f"pa.ChunkedArray, got {type(obj).__name__}"
        )

    def cast_polars(
        self,
        obj: Any,
        options: "CastOptions | None" = None,
        **more: Any,
    ) -> Any:
        """Cast any polars object — dispatch by shape.

        DataFrame/LazyFrame → :meth:`cast_polars_tabular`,
        Series → :meth:`cast_polars_series`,
        Expr → :meth:`cast_polars_expr`.
        """
        pl = get_polars()
        # Tabular first — a DataFrame is never a Series, and the pl
        # lazy import dominates dispatch latency. Doing the isinstance
        # walk here beats paying for it at every call site.
        if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
            return self.cast_polars_tabular(obj, options=options, **more)
        if isinstance(obj, pl.Series):
            return self.cast_polars_series(obj, options=options, **more)
        if isinstance(obj, pl.Expr):
            return self.cast_polars_expr(obj, options=options, **more)
        raise TypeError(
            f"Field.cast_polars: expected pl.DataFrame / LazyFrame / Series / "
            f"Expr, got {type(obj).__name__}"
        )

    def cast_pandas(
        self,
        obj: Any,
        options: "CastOptions | None" = None,
        **more: Any,
    ) -> Any:
        """Cast any pandas object — dispatch by shape.

        DataFrame → :meth:`cast_pandas_tabular`,
        Series → :meth:`cast_pandas_series`.

        Index isn't handled here: indices aren't data payload in the
        DataIO sense, and ``pa.Table.from_pandas`` carries them via
        ``preserve_index`` at the caller's discretion.
        """
        pd = get_pandas()
        if isinstance(obj, pd.DataFrame):
            return self.cast_pandas_tabular(obj, options=options, **more)
        if isinstance(obj, pd.Series):
            return self.cast_pandas_series(obj, options=options, **more)
        raise TypeError(
            f"Field.cast_pandas: expected pd.DataFrame / pd.Series, "
            f"got {type(obj).__name__}"
        )

    def cast_spark(
        self,
        obj: Any,
        options: "CastOptions | None" = None,
        **more: Any,
    ) -> Any:
        """Cast any spark object — dispatch by shape.

        DataFrame → :meth:`cast_spark_tabular`,
        Column → :meth:`cast_spark_column`.
        """
        sql = get_spark_sql()
        if isinstance(obj, sql.DataFrame):
            return self.cast_spark_tabular(obj, options=options, **more)
        if isinstance(obj, sql.Column):
            return self.cast_spark_column(obj, options=options, **more)
        raise TypeError(
            f"Field.cast_spark: expected pyspark.sql.DataFrame / Column, "
            f"got {type(obj).__name__}"
        )

    # ==================================================================
    # Cast — narrow methods (cast body + `finalize_*` tail)
    # ==================================================================
    #
    # Each narrow method follows the same three-step pattern:
    #
    #     1. OBJECT passthrough — variant columns never cast.
    #     2. Resolve CastOptions (merges kwargs, binds target).
    #     3. Delegate the cast body to self.dtype.cast_<engine>_<shape>.
    #     4. Hand the result to self.finalize_<engine>_<shape> for the
    #        null-fill + alias tail.
    #
    # Having the finalize call here — instead of inlining fill + alias
    # — means there's one source of truth for the post-cast cleanup
    # shape. Changing the null-fill semantics or the rename policy
    # touches finalize only; cast body stays put.

    def cast_arrow_array(
        self,
        array: pa.Array | pa.ChunkedArray,
        options: "CastOptions | None" = None,
        default_scalar: pa.Scalar | None = None,
        **more,
    ):
        # Object target is a variant — never touch the values.
        if self.dtype.type_id == DataTypeId.OBJECT:
            return array
        options = get_cast_options_class().check(options=options, **more)
        casted = self.dtype.cast_arrow_array(array, options=options.with_target(self))
        return self.finalize_arrow_array(casted, default_scalar=default_scalar)

    def cast_arrow_tabular(
        self,
        table: pa.Table | pa.RecordBatch,
        options: "CastOptions | None" = None,
        **more,
    ):
        if self.dtype.type_id == DataTypeId.OBJECT:
            return table
        options = get_cast_options_class().check(options=options, **more)
        casted = self.to_struct().dtype.cast_arrow_tabular(
            table, options=options.with_target(self)
        )
        # Tabular finalize is identity — per-column finalize already
        # ran inside the struct walk. Kept for shape symmetry.
        return self.finalize_arrow(casted)

    def cast_arrow_batch_iterator(
        self,
        batches: "Iterable[pa.RecordBatch]",
        options: "CastOptions | None" = None,
        **more,
    ) -> "Iterator[pa.RecordBatch]":
        """Cast a stream of :class:`pa.RecordBatch` against this field.

        Object targets passthrough (variant). Otherwise the dtype's
        struct view owns the per-batch tabular cast and ``byte_size``
        rechunk — same shape contract as :meth:`cast_arrow_tabular`,
        just lazy.
        """
        if self.dtype.type_id == DataTypeId.OBJECT:
            return iter(batches)
        options = get_cast_options_class().check(options=options, **more)
        return self.to_struct().dtype.cast_arrow_batch_iterator(
            batches, options=options.with_target(self)
        )

    def cast_polars_series(
        self,
        series: "polars.Series",
        options: "CastOptions | None" = None,
        default_scalar: Any = None,
        **more,
    ):
        if self.dtype.type_id == DataTypeId.OBJECT:
            return series
        options = get_cast_options_class().check(options=options, **more)
        casted = self.dtype.cast_polars_series(series, options=options.with_target(self))
        return self.finalize_polars_series(casted, default_scalar=default_scalar)

    def cast_polars_expr(
        self,
        series: "polars.Expr",
        options: "CastOptions | None" = None,
        default_scalar: Any = None,
        **more,
    ):
        if self.dtype.type_id == DataTypeId.OBJECT:
            return series
        options = get_cast_options_class().check(options=options, **more)
        casted = self.dtype.cast_polars_expr(series, options=options.with_target(self))
        return self.finalize_polars_expr(casted, default_scalar=default_scalar)

    def cast_polars_tabular(
        self,
        data: "polars.DataFrame | polars.LazyFrame",
        options: "CastOptions | None" = None,
        **more,
    ):
        if self.dtype.type_id == DataTypeId.OBJECT:
            return data
        options = get_cast_options_class().check(options=options, **more)
        casted = self.to_struct().dtype.cast_polars_tabular(
            data, options=options.with_target(self)
        )
        return self.finalize_polars(casted)

    def cast_pandas_series(
        self,
        series: "pd.Series",
        options: "CastOptions | None" = None,
        default_scalar: Any = None,
        **more,
    ):
        if self.dtype.type_id == DataTypeId.OBJECT:
            return series
        options = get_cast_options_class().check(options=options, **more)
        casted = self.dtype.cast_pandas_series(series, options=options.with_target(self))
        return self.finalize_pandas_series(casted, default_scalar=default_scalar)

    def cast_pandas_tabular(
        self,
        data: "pd.DataFrame",
        options: "CastOptions | None" = None,
        **more,
    ):
        if self.dtype.type_id == DataTypeId.OBJECT:
            return data
        options = get_cast_options_class().check(options=options, **more)
        casted = self.dtype.cast_pandas_tabular(data, options=options.with_target(self))
        return self.finalize_pandas(casted)

    def cast_spark_column(
        self,
        column: "ps.Column",
        options: "CastOptions | None" = None,
        default_scalar: Any = None,
        **more,
    ):
        if self.dtype.type_id == DataTypeId.OBJECT:
            return column
        options = get_cast_options_class().check(options=options, **more)
        options = options.with_target(self).check_source(column)
        casted = self.dtype.cast_spark_column(column, options=options)
        return self.finalize_spark_column(casted, default_scalar=default_scalar)

    def cast_spark_tabular(
        self,
        data: "ps.DataFrame",
        options: "CastOptions | None" = None,
        **more,
    ):
        if self.dtype.type_id == DataTypeId.OBJECT:
            return data
        options = get_cast_options_class().check(options=options, **more)
        casted = self.dtype.cast_spark_tabular(data, options=options.with_target(self))
        return self.finalize_spark(casted)

    # ==================================================================
    # Fill — null-only dispatch (no cast), mirrors the cast dispatcher
    # ==================================================================
    #
    # Same three granularities as the cast side. Useful when the caller
    # already has cast data and only needs the null-fill pass —
    # typically you don't call these directly because the ``cast_*``
    # methods already fill inline via ``finalize_*``, but they're the
    # right entry point when source and target dtypes already agree.

    def fill_nulls(self, obj: Any, *, default_scalar: Any = None) -> Any:
        """Fill nulls in *obj* using the native engine — engine + shape detection.

        Routes the same way :meth:`cast` does. See
        :meth:`fill_arrow` / :meth:`fill_polars` / :meth:`fill_pandas`
        / :meth:`fill_spark` for the per-engine behaviour.
        """
        ns, _name = ObjectSerde.module_and_name(obj)

        if ns.startswith("pyarrow"):
            return self.fill_arrow(obj, default_scalar=default_scalar)
        if ns.startswith("polars"):
            return self.fill_polars(obj, default_scalar=default_scalar)
        if ns.startswith("pandas"):
            return self.fill_pandas(obj, default_scalar=default_scalar)
        if ns.startswith("pyspark"):
            return self.fill_spark(obj, default_scalar=default_scalar)

        if isinstance(obj, Iterator):
            return (
                self.fill_nulls(item, default_scalar=default_scalar) for item in obj
            )
        if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
            return (
                self.fill_nulls(item, default_scalar=default_scalar) for item in obj
            )

        raise TypeError(
            f"Field.fill_nulls: unsupported input {type(obj).__name__!r} "
            f"(module={ns!r})"
        )

    def fill_arrow(self, obj: Any, *, default_scalar: Any = None) -> Any:
        """Fill nulls in any pyarrow object.

        Arrays go through :meth:`fill_arrow_array_nulls` directly.
        Tables / RecordBatches re-use the tabular cast path with
        ``self`` as the target — a no-op cast that still runs the
        per-column null-fill via the struct walk.
        """
        if isinstance(obj, (pa.Array, pa.ChunkedArray)):
            return self.fill_arrow_array_nulls(obj, default_scalar=default_scalar)
        if isinstance(obj, (pa.Table, pa.RecordBatch)):
            return obj
        raise TypeError(
            f"Field.fill_arrow: expected pa.Array/ChunkedArray/Table/"
            f"RecordBatch, got {type(obj).__name__}"
        )

    def fill_polars(self, obj: Any, *, default_scalar: Any = None) -> Any:
        """Fill nulls in any polars object.

        Series / Expr go through :meth:`fill_polars_array_nulls` —
        which handles both shapes uniformly (Expr is the lazy
        counterpart of Series; the fill operator grafts onto each
        identically). DataFrame / LazyFrame route through
        :meth:`cast_polars_tabular` as a self-targeted cast.
        """
        pl = get_polars()
        if isinstance(obj, (pl.Series, pl.Expr)):
            return self.fill_polars_array_nulls(obj, default_scalar=default_scalar)
        if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
            return self.cast_polars_tabular(obj)
        raise TypeError(
            f"Field.fill_polars: expected pl.DataFrame/LazyFrame/Series/Expr, "
            f"got {type(obj).__name__}"
        )

    def fill_pandas(self, obj: Any, *, default_scalar: Any = None) -> Any:
        """Fill nulls in any pandas object."""
        pd = get_pandas()
        if isinstance(obj, pd.Series):
            return self.fill_pandas_series_nulls(obj, default_scalar=default_scalar)
        if isinstance(obj, pd.DataFrame):
            return self.cast_pandas_tabular(obj)
        raise TypeError(
            f"Field.fill_pandas: expected pd.DataFrame/pd.Series, "
            f"got {type(obj).__name__}"
        )

    def fill_spark(self, obj: Any, *, default_scalar: Any = None) -> Any:
        """Fill nulls in any spark object."""
        sql = get_spark_sql()
        if isinstance(obj, sql.Column):
            return self.fill_spark_column_nulls(obj, default_scalar=default_scalar)
        if isinstance(obj, sql.DataFrame):
            return self.cast_spark_tabular(obj)
        raise TypeError(
            f"Field.fill_spark: expected pyspark.sql.DataFrame/Column, "
            f"got {type(obj).__name__}"
        )

    # ------------------------------------------------------------------
    # Fill — narrow per-engine null-fill primitives
    # ------------------------------------------------------------------

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

    def fill_polars_array_nulls(
        self,
        series: "polars.Series | polars.Expr",
        *,
        default_scalar: Any = None,
    ):
        if default_scalar is None and self.has_default:
            default_scalar = self.default

        return self.dtype.fill_polars_array_nulls(
            series,
            nullable=self.nullable,
            default_scalar=default_scalar,
        )

    def fill_pandas_series_nulls(
        self,
        series: "pd.Series",
        *,
        default_scalar: Any = None,
    ):
        if default_scalar is None and self.has_default:
            default_scalar = self.default

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
        if default_scalar is None:
            default_scalar = self.default

        return self.dtype.fill_spark_column_nulls(
            column,
            nullable=self.nullable,
            default_scalar=default_scalar,
        )

    # ==================================================================
    # Default value factories — zero-row / size-N default arrays
    # ==================================================================

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
        return self.dtype.default_polars_expr(
            value=self.default, nullable=self.nullable,
            alias=alias or self.name
        )

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

    # ==================================================================
    # Rename / alias helpers
    # ==================================================================
    #
    # polars and spark expose an ``.alias(name)`` method on their
    # column-like types. These helpers centralize the "rename only if
    # the target name differs from the current name, and only if the
    # target has a non-default name" logic so callers that want a
    # rename-only pass (skipping the full cast) don't have to
    # reimplement the guard every time.

    def polars_alias(self, obj: Any) -> Any:
        """Rename a polars Series / Expr to match this field's name.

        No-op when the target name matches the current name, or when
        this field only has the sentinel name. Calling defensively
        is free — zero-cost on the no-rename path.
        """
        if not self.name or self.name == DEFAULT_FIELD_NAME:
            return obj
        current = getattr(obj, "name", None)
        if current == self.name:
            return obj
        alias = getattr(obj, "alias", None)
        if alias is None:
            # Neither Series nor Expr — nothing to rename against.
            return obj
        return alias(self.name)

    def spark_alias(self, obj: Any) -> Any:
        """Rename a Spark Column to match this field's name.

        Spark DataFrames aren't handled — renaming a DataFrame
        requires a projection with named columns, which isn't a
        single-method operation. Column is the rename target here.
        """
        if not self.name or self.name == DEFAULT_FIELD_NAME:
            return obj
        alias = getattr(obj, "alias", None)
        if alias is None:
            return obj
        return alias(self.name)

    def pandas_alias(self, obj: Any) -> Any:
        """Rename a pandas Series to match this field's name.

        Pandas has no ``.alias()`` — rename is ``series.name = ...``,
        which mutates. This helper returns the series so it chains
        like :meth:`polars_alias` / :meth:`spark_alias`. DataFrames
        aren't handled (column rename is a projection, not a
        single-method op).
        """
        if not self.name or self.name == DEFAULT_FIELD_NAME:
            return obj
        if not hasattr(obj, "name"):
            return obj
        if obj.name == self.name:
            return obj
        obj.name = self.name
        return obj

    # ==================================================================
    # Finalize — post-cast fill + alias; the tail of every `cast_*`
    # ==================================================================
    #
    # The ``cast_*`` methods delegate their post-cast cleanup here, so
    # finalize is both the tail of a cast pipeline AND a standalone
    # entry point when source and target dtypes already agree (no
    # cast needed) but the caller still wants the null-fill + rename
    # pass. :class:`CastOptions` delegates its ``finalize_*_cast``
    # methods here.
    #
    # Tabular finalize is identity: per-column fill + alias already
    # ran inside the struct walk during the tabular cast. The tabular
    # method is kept for shape symmetry so dispatchers can call
    # ``finalize_<engine>`` uniformly regardless of input shape.

    def finalize_arrow_array(
        self,
        array: pa.Array | pa.ChunkedArray,
        *,
        default_scalar: pa.Scalar | None = None,
    ):
        """Fill nulls on a pyarrow Array / ChunkedArray.

        No alias step: pa.Array / ChunkedArray don't carry a name.
        Tabular naming lives in the pa.Field that wraps the array in
        a Table/RecordBatch, which :meth:`cast_arrow_tabular` handles
        through the struct walk.
        """
        return self.fill_arrow_array_nulls(array, default_scalar=default_scalar)

    def finalize_arrow(
        self,
        obj: Any,
        *,
        default_scalar: pa.Scalar | None = None,
    ) -> Any:
        """Finalize any pyarrow object — dispatch by shape.

        Array/ChunkedArray → fill.
        Table/RecordBatch → identity.
        """
        if isinstance(obj, (pa.Array, pa.ChunkedArray)):
            return self.finalize_arrow_array(obj, default_scalar=default_scalar)
        if isinstance(obj, (pa.Table, pa.RecordBatch)):
            return obj
        raise TypeError(
            f"Field.finalize_arrow: expected pa.Array/ChunkedArray/Table/"
            f"RecordBatch, got {type(obj).__name__}"
        )

    def finalize_polars_series(
        self,
        series: "polars.Series",
        *,
        default_scalar: Any = None,
    ):
        """Fill nulls, alias a polars Series to the target name."""
        filled = self.fill_polars_array_nulls(series, default_scalar=default_scalar)
        return self.polars_alias(filled)

    def finalize_polars_expr(
        self,
        expr: "polars.Expr",
        *,
        default_scalar: Any = None,
    ):
        """Fill nulls, alias a polars Expr to the target name.

        Same as :meth:`finalize_polars_series` — polars Series and Expr
        share the fill + alias primitives, so the finalize shape is
        identical. Separate method for call-site clarity.
        """
        filled = self.fill_polars_array_nulls(expr, default_scalar=default_scalar)
        return self.polars_alias(filled)

    def finalize_polars(
        self,
        obj: "polars.Series | polars.Expr | polars.DataFrame | polars.LazyFrame",
        *,
        default_scalar: Any = None,
    ):
        """Finalize any polars object — dispatch by shape.

        Series/Expr → fill + alias.
        DataFrame/LazyFrame → identity (tabular cast already finalized
        per-column via the struct walk).
        """
        pl = get_polars()
        if isinstance(obj, (pl.Series, pl.Expr)):
            filled = self.fill_polars_array_nulls(obj, default_scalar=default_scalar)
            return self.polars_alias(filled)
        if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
            return obj
        raise TypeError(
            f"Field.finalize_polars: expected pl.Series/Expr/DataFrame/LazyFrame, "
            f"got {type(obj).__name__}"
        )

    def finalize_pandas_series(
        self,
        series: "pd.Series",
        *,
        default_scalar: Any = None,
    ):
        """Fill nulls, rename a pandas Series to the target name."""
        filled = self.fill_pandas_series_nulls(series, default_scalar=default_scalar)
        return self.pandas_alias(filled)

    def finalize_pandas(
        self,
        obj: Any,
        *,
        default_scalar: Any = None,
    ) -> Any:
        """Finalize any pandas object — dispatch by shape.

        Series → fill + rename.
        DataFrame → identity.
        """
        pd = pandas_module()
        if isinstance(obj, pd.Series):
            return self.finalize_pandas_series(obj, default_scalar=default_scalar)
        if isinstance(obj, pd.DataFrame):
            return obj
        raise TypeError(
            f"Field.finalize_pandas: expected pd.DataFrame/pd.Series, "
            f"got {type(obj).__name__}"
        )

    def finalize_spark_column(
        self,
        column: "ps.Column",
        *,
        default_scalar: Any = None,
    ) -> "ps.Column":
        """Fill nulls, alias a Spark Column to the target name."""
        filled = self.fill_spark_column_nulls(column, default_scalar=default_scalar)
        return self.spark_alias(filled)

    def finalize_spark(
        self,
        obj: Any,
        *,
        default_scalar: Any = None,
    ) -> Any:
        """Finalize any spark object — dispatch by shape.

        Column → fill + alias.
        DataFrame → identity (tabular cast already finalized).
        """
        sql = get_spark_sql()
        if isinstance(obj, sql.Column):
            return self.finalize_spark_column(obj, default_scalar=default_scalar)
        if isinstance(obj, sql.DataFrame):
            return obj
        raise TypeError(
            f"Field.finalize_spark: expected pyspark.sql.Column/DataFrame, "
            f"got {type(obj).__name__}"
        )

    def finalize(
        self,
        obj: Any,
        *,
        default_scalar: Any = None,
    ) -> Any:
        """Finalize *obj* using its native engine — module-prefix dispatch.

        Mirrors :meth:`cast` / :meth:`fill_nulls` routing.
        """
        ns, _name = ObjectSerde.module_and_name(obj)

        if ns.startswith("pyarrow"):
            return self.finalize_arrow(obj, default_scalar=default_scalar)
        if ns.startswith("polars"):
            return self.finalize_polars(obj, default_scalar=default_scalar)
        if ns.startswith("pandas"):
            return self.finalize_pandas(obj, default_scalar=default_scalar)
        if ns.startswith("pyspark"):
            return self.finalize_spark(obj, default_scalar=default_scalar)

        if isinstance(obj, Iterator):
            return (self.finalize(item, default_scalar=default_scalar) for item in obj)
        if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
            return (self.finalize(item, default_scalar=default_scalar) for item in obj)

        raise TypeError(
            f"Field.finalize: unsupported input {type(obj).__name__!r} "
            f"(module={ns!r})"
        )


# ======================================================================
# Cast registry — `Any → Field` converter
# ======================================================================

@register_converter(Any, Field)
def any_to_field(obj: Any, _: Any):
    return Field.from_any(obj)