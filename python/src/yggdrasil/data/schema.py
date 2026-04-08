from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Iterator, MutableMapping
from dataclasses import dataclass, field as dc_field
from typing import TYPE_CHECKING, Any, AnyStr, Mapping, overload

import pyarrow as pa

from .field import Field, _normalize_metadata, _to_bytes

if TYPE_CHECKING:
    import polars as pl
    import pyspark.sql.types as pst


__all__ = [
    "Schema",
    "schema",
]


def schema(
    fields: Iterable[Field | pa.Field],
    *,
    metadata: dict[bytes | str, bytes | str | object] | None = None,
    tags: dict[bytes | str, bytes | str | object] | None = None,
) -> "Schema":
    return Schema.from_fields(
        fields,
        metadata=_normalize_metadata(metadata, tags=tags),
    )


@dataclass
class Schema(MutableMapping[str, Field]):
    inner_fields: OrderedDict[str, Field] = dc_field(default_factory=OrderedDict)
    metadata: dict[bytes, bytes] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.inner_fields, OrderedDict):
            normalized_fields: OrderedDict[str, Field] = OrderedDict()

            if isinstance(self.inner_fields, Mapping):
                for key, value in self.inner_fields.items():
                    field = Field.from_any(value)
                    if field.name != key:
                        field = field.copy(name=key)
                    normalized_fields[key] = field
            else:
                for value in self.inner_fields:
                    field = Field.from_any(value)
                    normalized_fields[field.name] = field

            self.inner_fields = normalized_fields

    @property
    def tags(self):
        if not self.metadata:
            return None
        return {k[2:]: v for k, v in self.metadata.items() if k.startswith(b"t:")}

    @tags.setter
    def tags(self, value: dict[AnyStr, AnyStr] | None):
        if value:
            if self.metadata is None:
                self.metadata = {}
            self.metadata.update(
                {
                    b"t:" + _to_bytes(k): _to_bytes(v)
                    for k, v in value.items()
                    if k and v
                }
            )

    def update_tags(self, value: dict[AnyStr, AnyStr] | None):
        if value:
            if self.metadata is None:
                self.metadata = {}
            self.metadata.update(
                {
                    b"t:" + _to_bytes(k): _to_bytes(v)
                    for k, v in value.items()
                    if k and v
                }
            )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.inner_fields.keys())

    @property
    def fields(self) -> tuple[Field, ...]:
        return tuple(self.inner_fields.values())

    def field_at(self, index: int, raise_error: bool = True) -> Field | None:
        if -len(self.inner_fields) <= index < len(self.inner_fields):
            return next(iter(self.inner_fields.values())) if len(self.inner_fields) == 1 and index in (0, -1) else self.fields[index]
        if raise_error:
            raise IndexError(index)
        return None

    def name_at(self, index: int, raise_error: bool = True) -> str | None:
        if -len(self.inner_fields) <= index < len(self.inner_fields):
            return self.names[index]
        if raise_error:
            raise IndexError(index)
        return None

    def index(self, key: str, *, case_sensitive: bool = False) -> int:
        resolved = self._resolve_name(key, case_sensitive=case_sensitive)
        return self.names.index(resolved)

    def _find_case_insensitive_name(self, key: str) -> str | None:
        key_lower = key.lower()
        matches = [name for name in self.inner_fields if name.lower() == key_lower]

        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]

        # Ambiguous on purpose. Exact match is already tried before this path.
        raise KeyError(
            f"Ambiguous field name {key!r}; case-insensitive matches: {matches}"
        )

    def _resolve_name(
        self,
        key: str,
        *,
        case_sensitive: bool = False,
        raise_error: bool = True,
    ) -> str | None:
        if key in self.inner_fields:
            return key

        if not case_sensitive:
            match = self._find_case_insensitive_name(key)
            if match is not None:
                return match

        if raise_error:
            raise KeyError(key)
        return None

    def _resolve_key(
        self,
        key: int | str,
        *,
        case_sensitive: bool = False,
        raise_error: bool = True,
    ) -> str | None:
        if isinstance(key, int):
            return self.name_at(key, raise_error=raise_error)

        if isinstance(key, str):
            return self._resolve_name(
                key,
                case_sensitive=case_sensitive,
                raise_error=raise_error,
            )

        if raise_error:
            raise TypeError(
                f"Schema keys must be int or str; got {type(key).__name__}"
            )
        return None

    @property
    def arrow_fields(self):
        return [_.to_arrow_field() for _ in self.inner_fields.values()]

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
                OrderedDict((field.name, Field.from_any(field)) for field in fields)
                if fields is not None
                else OrderedDict(
                    (name, field.copy()) for name, field in self.inner_fields.items()
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
            field = Field.from_any(value)
            out[field.name] = field
        return out

    def extend(self, fields: Iterable[Field | pa.Field]) -> "Schema":
        out = self.copy()
        for value in fields:
            field = Field.from_any(value)
            out[field.name] = field
        return out

    @overload
    def __getitem__(self, key: int) -> Field: ...
    @overload
    def __getitem__(self, key: str) -> Field: ...

    def __getitem__(self, key: int | str) -> Field:
        resolved = self._resolve_key(key)
        return self.inner_fields[resolved]

    def __setitem__(self, key: str, value: Field | pa.Field) -> None:
        if not isinstance(key, str):
            raise TypeError(
                f"Schema assignment key must be str; got {type(key).__name__}"
            )

        field = Field.from_any(value)
        if field.name != key:
            field = field.copy(name=key)
        self.inner_fields[key] = field

    def __delitem__(self, key: int | str) -> None:
        resolved = self._resolve_key(key)
        del self.inner_fields[resolved]

    def __iter__(self) -> Iterator[str]:
        return iter(self.inner_fields)

    def __len__(self) -> int:
        return len(self.inner_fields)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, int):
            return -len(self.inner_fields) <= key < len(self.inner_fields)

        if isinstance(key, str):
            return self._resolve_name(key, raise_error=False) is not None

        return False

    @overload
    def get(self, key: int, default: None = None) -> Field | None: ...
    @overload
    def get(self, key: str, default: None = None) -> Field | None: ...
    @overload
    def get(self, key: int | str, default: Any = None) -> Any: ...

    def get(self, key: int | str, default: Any = None) -> Any:
        resolved = self._resolve_key(key, raise_error=False)
        if resolved is None:
            return default
        return self.inner_fields[resolved]

    @overload
    def pop(self, key: int) -> Field: ...
    @overload
    def pop(self, key: str) -> Field: ...
    @overload
    def pop(self, key: int | str, default: Any) -> Field | Any: ...

    def pop(self, key: int | str, default: Any = ...):
        resolved = self._resolve_key(key, raise_error=False)
        if resolved is None:
            if default is ...:
                if isinstance(key, int):
                    raise IndexError(key)
                raise KeyError(key)
            return default
        return self.inner_fields.pop(resolved)

    def setdefault(self, key: str, default: Field | pa.Field | None = None) -> Field:
        if not isinstance(key, str):
            raise TypeError(
                f"Schema.setdefault key must be str; got {type(key).__name__}"
            )

        resolved = self._resolve_name(key, raise_error=False)
        if resolved is not None:
            return self.inner_fields[resolved]

        if default is None:
            raise ValueError("Schema.setdefault requires a Field default for new keys")

        field = Field.from_any(default)
        if field.name != key:
            field = field.copy(name=key)

        self.inner_fields[key] = field
        return field

    def popitem(self, last: bool = True) -> tuple[str, Field]:
        return self.inner_fields.popitem(last=last)

    def clear(self) -> None:
        self.inner_fields.clear()

    def __add__(self, other: Any) -> "Schema":
        other = self._coerce_other(other)

        merged_fields = OrderedDict(
            (name, field.copy()) for name, field in self.inner_fields.items()
        )
        for name, field in other.inner_fields.items():
            merged_fields[name] = field.copy()

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

        for name, field in other.inner_fields.items():
            self.inner_fields[name] = field.copy()

        self.metadata = self._merge_metadata(self.metadata, other.metadata)
        return self

    def __sub__(self, other: Any) -> "Schema":
        other = self._coerce_other(other)
        remove_names = set(other.inner_fields)

        return Schema(
            inner_fields=OrderedDict(
                (name, field.copy())
                for name, field in self.inner_fields.items()
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
                (name, field.copy())
                for name, field in self.inner_fields.items()
                if name in keep_names
            ),
            metadata=self._merge_metadata(self.metadata, other.metadata),
        )

    def __iand__(self, other: Any) -> "Schema":
        other = self._coerce_other(other)
        keep_names = set(other.inner_fields)

        self.inner_fields = OrderedDict(
            (name, field)
            for name, field in self.inner_fields.items()
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
        f = OrderedDict()
        for name, field in self.inner_fields.items():
            f[name] = field.autotag()

        self.update_tags(tags)

        return Schema(
            inner_fields=f,
            metadata=self.metadata,
        )

    @classmethod
    def from_any(cls, obj: Any) -> "Schema":
        if isinstance(obj, cls):
            return obj

        from yggdrasil.arrow.cast import any_to_arrow_schema
        return cls.from_arrow(any_to_arrow_schema(obj))

    @classmethod
    def from_fields(
        cls,
        fields: Iterable[Field | pa.Field],
        *,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
    ) -> "Schema":
        inner_fields: OrderedDict[str, Field] = OrderedDict()

        for value in fields:
            field = Field.from_any(value)
            inner_fields[field.name] = field

        return cls(
            inner_fields=inner_fields,
            metadata=_normalize_metadata(metadata, tags=tags),
        )

    @classmethod
    def from_arrow(cls, value: pa.Schema) -> "Schema":
        if not isinstance(value, pa.Schema):
            from yggdrasil.arrow.cast import any_to_arrow_schema
            value = any_to_arrow_schema(value)

        inner_fields: OrderedDict[str, Field] = OrderedDict()
        for arrow_field in value:
            field = Field.from_arrow(arrow_field)
            inner_fields[field.name] = field

        return cls(
            inner_fields=inner_fields,
            metadata=dict(value.metadata) if value.metadata else None,
        )

    @classmethod
    def from_arrow_schema(cls, value: pa.Schema) -> "Schema":
        return cls.from_arrow(value)

    def to_arrow_schema(self) -> pa.Schema:
        return pa.schema(
            fields=[field.to_arrow_field() for field in self.inner_fields.values()],
            metadata=self.metadata,
        )

    @classmethod
    def from_polars(
        cls,
        obj: "pl.Schema | pl.DataFrame | pl.LazyFrame",
        *,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
    ) -> "Schema":
        from yggdrasil.polars.lib import polars as pl

        if isinstance(obj, pl.Schema):
            fields = [
                Field.from_polars(name=name, dtype=dtype) for name, dtype in obj.items()
            ]
            return cls.from_fields(fields, metadata=metadata)

        if isinstance(obj, pl.DataFrame):
            fields = [
                Field.from_polars(name=name, dtype=dtype)
                for name, dtype in obj.schema.items()
            ]
            return cls.from_fields(fields, metadata=metadata)

        if isinstance(obj, pl.LazyFrame):
            fields = [
                Field.from_polars(name=name, dtype=dtype)
                for name, dtype in obj.collect_schema().items()
            ]
            return cls.from_fields(fields, metadata=metadata)

        raise TypeError(
            "Expected a polars.Schema, polars.DataFrame, or polars.LazyFrame; "
            f"got {type(obj).__name__}"
        )

    def to_polars_schema(self) -> "pl.Schema":
        from yggdrasil.polars.lib import polars as pl

        return pl.Schema(
            {
                field.name: field.to_polars_field().dtype
                for field in self.inner_fields.values()
            }
        )

    @classmethod
    def from_pyspark(
        cls,
        obj: "pst.StructType | Any",
        *,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
    ) -> "Schema":
        from yggdrasil.spark.cast import spark_schema_to_arrow_schema
        import pyspark.sql as pyspark_sql

        if isinstance(obj, pyspark_sql.types.StructType):
            arrow_schema = spark_schema_to_arrow_schema(obj)
            merged_metadata = (
                _normalize_metadata(metadata, tags=tags)
                if (metadata is not None or tags is not None)
                else (dict(arrow_schema.metadata) if arrow_schema.metadata else None)
            )
            return cls.from_fields(arrow_schema, metadata=merged_metadata)

        return cls.from_any(obj).copy(metadata=metadata, tags=tags)

    def to_spark_schema(self) -> "pst.StructType":
        from yggdrasil.spark.cast import arrow_schema_to_spark_schema
        return arrow_schema_to_spark_schema(self.to_arrow_schema())

    def cast_options(self, *, safe: bool = True):
        from yggdrasil.data.cast import CastOptions
        return CastOptions(
            target_field=self.to_arrow_schema(),
            safe=safe,
        )

    def cast_table(self, obj: Any, *, safe: bool = True) -> Any:
        return self.cast_options(
            safe=safe
        ).cast_table(obj)

    def cast_unstructured(
        self,
        obj: Any,
        *,
        as_type: type = pa.Table,
        safe: bool = True,
    ) -> Any:
        from yggdrasil.data.cast import CastOptions, convert
        options = CastOptions(target_field=self.to_arrow_schema(), safe=safe)
        return convert(obj, target_hint=as_type, options=options)