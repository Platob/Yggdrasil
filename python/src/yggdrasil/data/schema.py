from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterable, Iterator, MutableMapping
from dataclasses import dataclass, field as dc_field, field
from typing import TYPE_CHECKING, Any, AnyStr, Mapping, overload

import pyarrow as pa
from yggdrasil.data.cast.registry import register_converter
from yggdrasil.data.constants import DEFAULT_FIELD_NAME
from yggdrasil.io import SaveMode

from .base_meta import BaseMetadata, _normalize_metadata, _to_bytes, BaseChildrenFields
from .data_field import Field
from .types.nested import StructType
from .types.support import get_polars, get_spark_sql

if TYPE_CHECKING:
    import polars
    import pyspark.sql.types as pst


__all__ = [
    "Schema",
    "schema",
]


def schema(
    fields: Iterable[Field | pa.Field | str],
    *other: Field | pa.Field,
    metadata: dict[bytes | str, bytes | str | object] | None = None,
    tags: dict[bytes | str, bytes | str | object] | None = None,
) -> "Schema":
    if fields is None:
        fields = []
    elif isinstance(fields, Field):
        fields = [fields]
    elif isinstance(fields, Schema):
        if not metadata:
            metadata = fields.metadata

        fields = fields.children_fields
    elif not isinstance(fields, (list, set, tuple)):
        fields = [fields]

    if other:
        fields = list(fields)
        fields.extend(other)

    return Schema.from_any_fields(
        fields,
        metadata=metadata,
        tags=tags
    )


@dataclass
class Schema(BaseMetadata, BaseChildrenFields, MutableMapping[str, Field]):
    inner_fields: OrderedDict[str, Field] = dc_field(default_factory=OrderedDict)
    metadata: dict[bytes, bytes] | None = field(default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.inner_fields, OrderedDict):
            normalized_fields: OrderedDict[str, Field] = OrderedDict()

            if isinstance(self.inner_fields, Mapping):
                for key, value in self.inner_fields.items():
                    f = Field.from_any(value)
                    if f.name != key:
                        f = f.copy(name=key)
                    normalized_fields[key] = f
            else:
                for value in self.inner_fields:
                    f = Field.from_any(value)
                    normalized_fields[f.name] = f

            self.inner_fields = normalized_fields
    
    def equals(
        self,
        other: Any,
        check_names: bool = True,
        check_dtypes: bool = True,
        check_metadata: bool = True,
    ) -> bool:
        """Structural equality check with configurable scope.

        Mirrors :meth:`DataType.equals` / :meth:`Field.equals`. Coerces
        *other* into a ``Schema`` so callers can pass a ``pa.Schema``,
        mapping, etc. Returns ``False`` on coercion failure instead of
        raising.

        - ``check_names``: compare field names (a reordered schema is
          still equal when names match). When ``False``, fields are
          compared positionally and names are ignored at every depth.
        - ``check_dtypes``: recurse into field dtypes and compare
          per-field ``nullable``.
        - ``check_metadata``: compare schema metadata and recurse.
        """
        if other is None:
            return False
        if not isinstance(other, Schema):
            try:
                other = self.from_any(other)
            except Exception:
                return False

        if check_metadata and self.metadata != other.metadata:
            return False

        self_fields = self.children_fields
        other_fields = other.children_fields

        if len(self_fields) != len(other_fields):
            return False

        if not check_names:
            return all(
                s.equals(
                    o,
                    check_names=check_names,
                    check_dtypes=check_dtypes,
                    check_metadata=check_metadata,
                )
                for s, o in zip(self_fields, other_fields)
            )

        seen: set[str] = set()
        for self_field in self_fields:
            other_field = other.field_by(
                name=self_field.name, raise_error=False
            )
            if other_field is None:
                return False
            if not self_field.equals(
                other_field,
                check_names=check_names,
                check_dtypes=check_dtypes,
                check_metadata=check_metadata,
            ):
                return False
            seen.add(self_field.name)

        for other_field in other_fields:
            if other_field.name not in seen:
                return False

        return True
    
    def _empty_tags(self) -> None:
        return None

    @staticmethod
    def _default_name(value: Any) -> str:
        if isinstance(value, type):
            return getattr(value, "__name__", DEFAULT_FIELD_NAME)
        return getattr(type(value), "__name__", DEFAULT_FIELD_NAME)

    @property
    def children_fields(self) -> tuple[Field, ...]:
        return tuple(self.inner_fields.values())

    @property
    def fields(self) -> tuple[Field, ...]:
        return tuple(self.inner_fields.values())

    @property
    def name(self) -> str:
        if not self.metadata:
            return DEFAULT_FIELD_NAME
        return self.metadata.get(b"name", DEFAULT_FIELD_NAME.encode()).decode("utf-8")

    @name.setter
    def name(self, value: str) -> None:
        if value and value != DEFAULT_FIELD_NAME:
            if not self.metadata:
                self.metadata = {}
            self.metadata[b"name"] = _to_bytes(value)

    @property
    def dtype(self) -> StructType:
        return StructType(fields=self.fields)

    @property
    def nullable(self) -> bool:
        if not self.metadata:
            return False
        return self.metadata.get(b"nullable", b"f").startswith(b"t")

    @nullable.setter
    def nullable(self, value: bool) -> None:
        if value:
            if not self.metadata:
                self.metadata = {}
            self.metadata[b"nullable"] = b"t" if value else b"f"

    @property
    def arrow_fields(self):
        return [_.to_arrow_field() for _ in self.fields]

    @property
    def partition_by(self):
        return [_ for _ in self.inner_fields.values() if _.partition_by]

    @property
    def cluster_by(self):
        return [_ for _ in self.inner_fields.values() if _.cluster_by]

    @property
    def primary_keys(self) -> list[Field]:
        return [f for f in self.inner_fields.values() if f.primary_key]

    @property
    def primary_key_names(self) -> list[str]:
        return [f.name for f in self.inner_fields.values() if f.primary_key]

    @property
    def foreign_keys(self) -> list[Field]:
        return [f for f in self.inner_fields.values() if f.foreign_key]

    @property
    def foreign_key_names(self) -> list[str]:
        return [f.name for f in self.inner_fields.values() if f.foreign_key]

    @property
    def foreign_key_refs(self) -> dict[str, str]:
        """Map each foreign-key column to its ``foreign_key`` tag value."""
        return {
            f.name: f.foreign_key
            for f in self.inner_fields.values()
            if f.foreign_key
        }

    @property
    def comment(self):
        if not self.metadata:
            return None

        c = self.metadata.get(b"comment") or self.metadata.get(b"description")
        if not c:
            return None
        return c.decode("utf-8")

    @staticmethod
    def _merge_metadata(
        left: dict[bytes, bytes] | None,
        right: dict[bytes, bytes] | None,
    ) -> dict[bytes, bytes] | None:
        if not left and not right:
            return None
        return {**(left or {}), **(right or {})} or None

    @staticmethod
    def _coerce_other(other: Any) -> "Schema":
        if isinstance(other, Schema):
            return other
        return Schema.from_any(other)

    def copy(
        self,
        *,
        fields: Iterable[Field | pa.Field] | None = None,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
    ) -> "Schema":
        return Schema(
            inner_fields=(
                OrderedDict((f.name, Field.from_any(f)) for f in fields)
                if fields is not None
                else OrderedDict(
                    (name, f.copy()) for name, f in self.inner_fields.items()
                )
            ),
            metadata=(
                _normalize_metadata(metadata, tags=tags)
                if (metadata is not None or tags is not None)
                else (dict(self.metadata) if self.metadata else None)
            ),
        )

    def append(self, *more_fields: Field | pa.Field) -> "Schema":
        out = self.copy()
        for value in more_fields:
            f = Field.from_any(value)
            out[f.name] = f
        return out

    def extend(self, fields: Iterable[Field | pa.Field]) -> "Schema":
        out = self.copy()
        for value in fields:
            f = Field.from_any(value)
            out[f.name] = f
        return out

    @overload
    def __getitem__(self, key: int) -> Field: ...
    @overload
    def __getitem__(self, key: str) -> Field: ...

    def __getitem__(self, key: int | str) -> Field:
        resolved = self.field(key)
        return self.inner_fields[resolved.name]

    def __setitem__(self, key: str, value: Field | pa.Field) -> None:
        if not isinstance(key, str):
            raise TypeError(
                f"Schema assignment key must be str; got {type(key).__name__}"
            )

        f = Field.from_any(value)
        if f.name != key:
            f = f.copy(name=key)
        self.inner_fields[key] = f

    def __delitem__(self, key: int | str) -> None:
        resolved = self.field(key)
        del self.inner_fields[resolved.name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.inner_fields)

    def __len__(self) -> int:
        return len(self.inner_fields)

    def __contains__(self, key: object) -> bool:
        return self.field(key, raise_error=False) is not None

    @overload
    def get(self, key: int, default: None = None) -> Field | None: ...
    @overload
    def get(self, key: str, default: None = None) -> Field | None: ...
    @overload
    def get(self, key: int | str, default: Any = None) -> Any: ...

    def get(self, key: int | str, default: Any = None) -> Any:
        resolved = self.field(key, raise_error=False)
        if resolved is None:
            return default
        return resolved

    @overload
    def pop(self, key: int) -> Field: ...
    @overload
    def pop(self, key: str) -> Field: ...
    @overload
    def pop(self, key: int | str, default: Any = None) -> Field | Any: ...

    def pop(self, key: int | str, default: Any = ...):
        resolved = self.field(key, raise_error=False)
        if resolved is None:
            if default is ...:
                if isinstance(key, int):
                    raise IndexError(key)
                raise KeyError(key)
            return default
        return self.inner_fields.pop(resolved.name)

    def setdefault(self, key: str, default: Field | pa.Field | None = None) -> Field:
        if not isinstance(key, str):
            raise TypeError(
                f"Schema.setdefault key must be str; got {type(key).__name__}"
            )

        resolved = self.field(key, raise_error=False)
        if resolved is not None:
            return resolved

        if default is None:
            raise ValueError("Schema.setdefault requires a Field default for new keys")

        f = Field.from_any(default)
        if f.name != key:
            f = f.copy(name=key)

        self.inner_fields[key] = f
        return f

    def keys(self):
        return self.inner_fields.keys()

    def popitem(self, last: bool = True) -> tuple[str, Field]:
        return self.inner_fields.popitem(last=last)

    def clear(self) -> None:
        self.inner_fields.clear()

    def __add__(self, other: Any) -> "Schema":
        other = self._coerce_other(other)

        merged_fields = OrderedDict(
            (name, f.copy()) for name, f in self.inner_fields.items()
        )
        for name, f in other.inner_fields.items():
            merged_fields[name] = f.copy()

        return Schema(
            inner_fields=merged_fields,
            metadata=self._merge_metadata(self.metadata, other.metadata),
        )

    def __radd__(self, other: Any) -> "Schema":
        if other == 0:
            return self.copy()
        other = self._coerce_other(other)
        return other.__add__(self)

    def __iadd__(self, other: Any) -> "Schema":
        other = self._coerce_other(other)

        for name, f in other.inner_fields.items():
            self.inner_fields[name] = f.copy()

        self.metadata = self._merge_metadata(self.metadata, other.metadata)
        return self

    def __sub__(self, other: Any) -> "Schema":
        other = self._coerce_other(other)
        remove_names = set(other.inner_fields)

        return Schema(
            inner_fields=OrderedDict(
                (name, f.copy())
                for name, f in self.inner_fields.items()
                if name not in remove_names
            ),
            metadata=dict(self.metadata) if self.metadata else None,
        )

    def __isub__(self, other: Any) -> "Schema":
        other = self._coerce_other(other)

        for name in tuple(other.inner_fields):
            self.inner_fields.pop(name, None)

        return self

    def __and__(self, other: Any) -> "Schema":
        other = self._coerce_other(other)
        keep_names = set(other.inner_fields)

        return Schema(
            inner_fields=OrderedDict(
                (name, f.copy())
                for name, f in self.inner_fields.items()
                if name in keep_names
            ),
            metadata=self._merge_metadata(self.metadata, other.metadata),
        )

    def __iand__(self, other: Any) -> "Schema":
        other = self._coerce_other(other)
        keep_names = set(other.inner_fields)

        self.inner_fields = OrderedDict(
            (name, f)
            for name, f in self.inner_fields.items()
            if name in keep_names
        )
        self.metadata = self._merge_metadata(self.metadata, other.metadata)
        return self

    def __or__(self, other: Any) -> "Schema":
        return self.__add__(other)

    def __ror__(self, other: Any) -> "Schema":
        other = self._coerce_other(other)
        return other.__add__(self)

    def __ior__(self, other: Any) -> "Schema":
        return self.__iadd__(other)

    def autotag(
        self,
        tags: dict[AnyStr, AnyStr] | None = None,
    ):
        primary_key_names = self._pop_field_name_list(b"primary_key")
        partition_by_names = self._pop_field_name_list(b"partition_by")
        cluster_by_names = self._pop_field_name_list(b"cluster_by")

        inner_fields = OrderedDict()
        for name, f in self.inner_fields.items():
            if primary_key_names and name in primary_key_names:
                f.with_primary_key(True, inplace=True)
            if partition_by_names and name in partition_by_names:
                f.with_partition_by(True, inplace=True)
            if cluster_by_names and name in cluster_by_names:
                f.with_cluster_by(True, inplace=True)
            inner_fields[name] = f.autotag()

        self.update_tags(tags)

        return Schema(
            inner_fields=inner_fields,
            metadata=self.metadata,
        )

    def _pop_field_name_list(self, key: bytes) -> set[str]:
        """Pop a ``key`` -> field-name-list entry off ``self.metadata``.

        Accepts a JSON array (``'["a","b"]'``) or a dot-separated string
        (``"a.b"``). Returns an empty set when the key is missing or empty —
        and removes the key either way so it does not leak through to the
        Arrow/Delta schema.
        """
        if not self.metadata:
            return set()
        raw = self.metadata.pop(key, None)
        if not raw:
            return set()
        if raw.startswith(b"[") and raw.endswith(b"]"):
            return set(json.loads(raw))
        return set(raw.decode().split("."))

    def to_arrow_schema(self) -> pa.Schema:
        return pa.schema(
            self.arrow_fields,
            metadata=self.metadata,
        )

    def to_polars_schema(self) -> "polars.Schema":
        pl = get_polars()

        return pl.Schema(
            [
                (f.name, f.dtype.to_polars())
                for f in self.inner_fields.values()
            ]
        )

    def to_spark_schema(self) -> "pst.StructType":
        pyspark_sql = get_spark_sql()

        return pyspark_sql.types.StructType(
            [f.to_pyspark_field() for f in self.inner_fields.values()]
        )

    def to_polars_flavor(self) -> "polars.Schema":
        """Polars-native counterpart for this schema — a ``pl.Schema``."""
        return self.to_polars_schema()

    def to_spark_flavor(self) -> "pst.StructType":
        """Spark-native counterpart for this schema — a ``StructType``."""
        return self.to_spark_schema()

    @classmethod
    def from_(cls, obj: Any):
        if isinstance(obj, Schema):
            return obj
        return cls.from_field(Field.from_any(obj))

    @classmethod
    def from_any(cls, obj: Any):
        if isinstance(obj, Schema):
            return obj
        return cls.from_field(Field.from_any(obj))

    @classmethod
    def from_field(cls, f: Field) -> "Schema":
        return f.to_schema()

    @classmethod
    def from_any_fields(
        cls,
        fields: Iterable[Field | Any],
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
    ):
        inner = OrderedDict()
        for f in fields:
            f = Field.from_any(f)
            inner[f.name] = f

        return cls(
            inner_fields=inner,
            metadata=_normalize_metadata(metadata, tags),
        )

    @classmethod
    def from_arrow(cls, obj):
        return Field.from_arrow(obj).to_schema()

    @classmethod
    def from_path(
        cls,
        path: Any,
        *,
        media: Any = None,
        path_io: Any = None,
    ) -> "Schema":
        from yggdrasil.io.buffer.path_io import PathIO

        if isinstance(path, PathIO):
            resolved = path
        elif isinstance(path_io, PathIO):
            resolved = path_io
        else:
            factory = path_io
            if factory is None:
                from yggdrasil.io.buffer.local_path_io import LocalPathIO
                factory = LocalPathIO
            resolved = factory.make(path, media=media)
        return resolved.collect_schema()

    def to_field(self) -> Field:
        return Field(
            name=self.name,
            dtype=self.dtype,
            nullable=self.nullable,
            metadata=self.metadata,
        )

    def merge_with(
        self,
        other: "Schema",
        inplace: bool = False,
        mode: SaveMode | None = None,
        downcast: bool = False,
        upcast: bool = False,
        merge_dtype: bool = True,
        merge_nullable: bool = True,
        merge_metadata: bool = True
    ):
        return (
            self.to_field()
            .merge_with(
                self.from_(other).to_field(),
                inplace=inplace,
                mode=mode,
                downcast=downcast,
                upcast=upcast,
                merge_dtype=merge_dtype,
                merge_nullable=merge_nullable,
                merge_metadata=merge_metadata
            )
            .to_schema()
        )


@register_converter(Any, Schema)
def any_to_schema(obj: Any, _: Any):
    return Schema.from_any(obj)
