"""Schema — a :class:`Field` whose dtype is a :class:`StructType`.

A schema IS a struct field. The class subclasses :class:`Field`,
inheriting all of its storage (``name``, ``dtype``, ``nullable``,
``metadata``) and behaviour (``equals``, ``with_*``, ``merge_with``,
``copy``, ``autotag``, tag flags, engine-level coercions, …). On top
of that it adds a mapping surface (key/index access, ``append`` /
``extend``, set operators) and engine-level schema export
(``to_arrow_schema`` / ``to_polars_schema`` / ``to_spark_schema``).
``inner_fields`` survives as a derived ``OrderedDict`` view of
``self.dtype.fields`` so existing call sites keep working.
"""
from __future__ import annotations

import dataclasses
import json
from collections import OrderedDict
from collections.abc import Iterable, Iterator, MutableMapping
from typing import TYPE_CHECKING, Any, AnyStr, Mapping, overload

import pyarrow as pa

from yggdrasil.data.cast.registry import register_converter
from yggdrasil.data.constants import DEFAULT_FIELD_NAME
from .base_meta import _normalize_metadata
from .data_field import Field, field
from .types.nested import StructType
from .types.support import get_polars, get_spark_sql
from ..lazy_imports import path_class

if TYPE_CHECKING:
    import polars
    import pyspark.sql.types as pst


__all__ = [
    "Schema",
    "schema",
    "Field",
    "field"
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
        tags=tags,
    )


def _normalize_inner_fields(
    value: Any,
) -> list[Field]:
    """Coerce *value* into an ordered list of :class:`Field` children.

    Accepts ``None`` (empty), a single ``Field``, a mapping
    (``{name: field}`` — names are forced to match), an
    ``OrderedDict``, or any other iterable of field-like inputs.
    """
    if value is None:
        return []
    if isinstance(value, Field):
        return [value]
    if isinstance(value, Mapping):
        out: list[Field] = []
        for key, raw in value.items():
            f = Field.from_any(raw)
            if f.name != key:
                f = f.copy(name=key)
            out.append(f)
        return out
    return [Field.from_any(_) for _ in value]


def _peel_name_nullable(metadata: Any) -> tuple[Any, str | None, bool | None]:
    """Pop ``b"name"`` / ``b"nullable"`` out of *metadata* if present.

    Schema stores its ``name`` / ``nullable`` flag as first-class
    :class:`Field` attributes, but callers commonly pass them via the
    metadata dict (legacy shape, or convenience). Strip the keys when
    they exist so they don't appear twice; return *metadata* unchanged
    (same object identity) when neither key is present.
    """
    if not isinstance(metadata, Mapping):
        return metadata, None, None
    if (
        b"name" not in metadata and "name" not in metadata
        and b"nullable" not in metadata and "nullable" not in metadata
    ):
        return metadata, None, None

    cleaned = dict(metadata)
    raw_name = cleaned.pop(b"name", None)
    if raw_name is None:
        raw_name = cleaned.pop("name", None)
    name: str | None = None
    if raw_name is not None:
        name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)

    raw_nullable = cleaned.pop(b"nullable", None)
    if raw_nullable is None:
        raw_nullable = cleaned.pop("nullable", None)
    nullable: bool | None = None
    if raw_nullable is not None:
        if isinstance(raw_nullable, bytes):
            nullable = raw_nullable.startswith(b"t") or raw_nullable.startswith(b"T")
        elif isinstance(raw_nullable, bool):
            nullable = raw_nullable
        else:
            nullable = str(raw_nullable).lower().startswith("t")

    return cleaned, name, nullable


@dataclasses.dataclass(repr=False, eq=False, frozen=True, init=False)
class Schema(Field, MutableMapping[str, Field]):
    """A :class:`Field` whose ``dtype`` is a :class:`StructType`.

    Schema is *just* a struct field with the children-shaped
    convenience surface (mapping / set ops / engine-flavoured schema
    export) bolted on. The strict ``dtype: StructType`` annotation
    makes that contract explicit so callers can rely on it without
    runtime ``isinstance`` checks.
    """

    # Strict shape — Schema's dtype is always a StructType.
    dtype: StructType

    def __init__(
        self,
        inner_fields: Iterable[Field | pa.Field] | Mapping | Field | None = None,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        *,
        name: str | None = None,
        nullable: bool | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
        parent: "Field | None" = None,
    ) -> None:
        children = _normalize_inner_fields(inner_fields)
        meta, embedded_name, embedded_nullable = _peel_name_nullable(metadata)
        if name is None:
            name = embedded_name if embedded_name is not None else DEFAULT_FIELD_NAME
        if nullable is None:
            nullable = bool(embedded_nullable) if embedded_nullable is not None else False

        Field.__init__(
            self,
            name=name,
            dtype=StructType(fields=tuple(children)),
            nullable=bool(nullable),
            metadata=meta,
            tags=tags,
            parent=parent,
        )

    # ------------------------------------------------------------------
    # Repr — header line + indented per-field block
    # ------------------------------------------------------------------

    def __repr__(self):
        body = "".join(
            f"\n{f.pretty_format(level=1)}"
            for f in self.children_fields
        )
        comment = self.comment
        return f"Schema: {self.name!r} {comment!r}{body}"

    def __bool__(self):
        return bool(self.dtype.fields)

    # ------------------------------------------------------------------
    # Children surface — derived from ``self.dtype.fields``
    # ------------------------------------------------------------------

    @property
    def children_fields(self) -> list[Field]:
        return list(self.dtype.fields)

    @property
    def fields(self) -> list[Field]:
        """Children excluding constraint-only fields."""
        return [f for f in self.dtype.fields if not f.constraint_key]

    @property
    def inner_fields(self) -> "OrderedDict[str, Field]":
        """Compat view of the children as an ordered ``{name: field}`` map."""
        return OrderedDict((f.name, f) for f in self.dtype.fields)

    @property
    def arrow_fields(self) -> list[pa.Field]:
        return [f.to_arrow_field() for f in self.fields]

    @property
    def primary_keys(self) -> list[Field]:
        return [f for f in self.fields if f.primary_key]

    @property
    def foreign_keys(self) -> list[Field]:
        return [f for f in self.fields if f.foreign_key]

    @property
    def constraints(self) -> list[Field]:
        return [f for f in self.dtype.fields if f.constraint_key]

    @property
    def comment(self):
        if not self.metadata:
            return None
        c = self.metadata.get(b"comment") or self.metadata.get(b"description")
        return c.decode("utf-8") if c else None

    # ------------------------------------------------------------------
    # Storage mutation — backed by ``self.dtype.fields`` (a tuple)
    # ------------------------------------------------------------------

    def _set_dtype_fields(self, fields: Iterable[Field]) -> None:
        """Replace the underlying StructType's fields tuple in place.

        Mutating ``self.dtype.fields`` doesn't go through any of
        Field's ``with_*`` mutators, so the cached arrow / polars /
        spark projections (and the field-name lookup map) would
        otherwise miss the change. Invalidate them explicitly and
        re-adopt the new children so they bubble future mutations
        back through ``self.parent``.
        """
        object.__setattr__(self.dtype, "fields", tuple(fields))
        self._invalidate_cache()
        self._adopt_children()

    @classmethod
    def empty(
        cls,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
    ) -> "Schema":
        return cls(inner_fields=(), metadata=metadata, tags=tags)

    def is_empty(self) -> bool:
        return len(self.dtype.fields) == 0

    @classmethod
    def peek_from(cls, obj: Any) -> tuple[Any, "Schema"]:
        obj, dfield = Field.peek_from(obj)
        return obj, cls.from_field(dfield)

    def equals(
        self,
        other: Any,
        check_names: bool = True,
        check_dtypes: bool = True,
        check_nullable: bool = True,
        check_metadata: bool = True,
    ) -> bool:
        """Order-independent structural equality across fields.

        :class:`Field.equals` compares the underlying StructType
        positionally; Schema treats children as a name-keyed set
        when ``check_names`` is ``True`` so a reordered schema is
        still equal.
        """
        if other is None:
            return False
        if not isinstance(other, Schema):
            try:
                other = Schema.from_any(other)
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
                    check_nullable=check_nullable,
                    check_metadata=check_metadata,
                )
                for s, o in zip(self_fields, other_fields)
            )

        seen: set[str] = set()
        for self_field in self_fields:
            other_field = other.field_by(name=self_field.name, raise_error=False)
            if other_field is None:
                return False
            if not self_field.equals(
                other_field,
                check_names=check_names,
                check_dtypes=check_dtypes,
                check_nullable=check_nullable,
                check_metadata=check_metadata,
            ):
                return False
            seen.add(self_field.name)

        for other_field in other_fields:
            if other_field.name not in seen:
                return False

        return True

    # ------------------------------------------------------------------
    # Mapping surface
    # ------------------------------------------------------------------

    def copy(
        self,
        *,
        fields: Iterable[Field | pa.Field] | None = None,
        dtype: Any = None,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
        name: str | None = None,
        nullable: bool | None = None,
    ) -> "Schema":
        # ``dtype`` is accepted so :meth:`Field.merge_with` (which
        # threads a merged :class:`StructType` through ``copy``) works
        # for Schema. Only StructType is supported — other dtypes don't
        # round-trip through Schema's children-shaped surface.
        if dtype is not None:
            if fields is not None:
                raise TypeError("Schema.copy: pass either fields= or dtype=, not both")
            if not isinstance(dtype, StructType):
                raise TypeError(
                    f"Schema.copy(dtype=...) only accepts StructType; "
                    f"got {type(dtype).__name__}."
                )
            fields = list(dtype.fields)

        if fields is not None:
            children = _normalize_inner_fields(fields)
        else:
            children = [f.copy() for f in self.dtype.fields]

        if metadata is not None or tags is not None:
            meta = _normalize_metadata(metadata, tags=tags)
        else:
            meta = dict(self.metadata) if self.metadata else None

        # Honour legacy name/nullable embedded in the metadata dict
        # so callers passing ``metadata={"name": ...}`` still update
        # the schema's name.
        meta, embedded_name, embedded_nullable = _peel_name_nullable(meta)

        if name is None:
            name = embedded_name if embedded_name is not None else self.name
        if nullable is None:
            nullable = embedded_nullable if embedded_nullable is not None else self.nullable

        return Schema(
            inner_fields=children,
            metadata=meta,
            name=name,
            nullable=nullable,
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
        return self.field(key)

    def __setitem__(self, key: str, value: Field | pa.Field) -> None:
        if not isinstance(key, str):
            raise TypeError(
                f"Schema assignment key must be str; got {type(key).__name__}"
            )

        f = Field.from_any(value)
        if f.name != key:
            f = f.copy(name=key)

        new_fields: list[Field] = []
        replaced = False
        for existing in self.dtype.fields:
            if existing.name == key:
                new_fields.append(f)
                replaced = True
            else:
                new_fields.append(existing)
        if not replaced:
            new_fields.append(f)
        self._set_dtype_fields(new_fields)

    def __delitem__(self, key: int | str) -> None:
        resolved = self.field(key)
        self._set_dtype_fields(
            f for f in self.dtype.fields if f.name != resolved.name
        )

    def __iter__(self) -> Iterator[str]:
        return iter(f.name for f in self.dtype.fields)

    def __len__(self) -> int:
        return len(self.dtype.fields)

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
        self._set_dtype_fields(
            f for f in self.dtype.fields if f.name != resolved.name
        )
        return resolved

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
        self._set_dtype_fields(list(self.dtype.fields) + [f])
        return f

    def keys(self):
        return [f.name for f in self.dtype.fields]

    def popitem(self, last: bool = True) -> tuple[str, Field]:
        fields = list(self.dtype.fields)
        if not fields:
            raise KeyError("popitem(): schema is empty")
        idx = -1 if last else 0
        f = fields.pop(idx)
        self._set_dtype_fields(fields)
        return f.name, f

    def clear(self) -> None:
        self._set_dtype_fields(())

    # ------------------------------------------------------------------
    # Set operators — return new Schemas; mirror dict / set semantics
    # ------------------------------------------------------------------

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

    @staticmethod
    def _rhs_wins_name(left: "Schema", right: "Schema") -> str:
        """RHS-wins conflict resolution mirroring metadata merge semantics."""
        if right.name and right.name != DEFAULT_FIELD_NAME:
            return right.name
        return left.name

    def __add__(self, other: Any) -> "Schema":
        other = self._coerce_other(other)
        merged: "OrderedDict[str, Field]" = OrderedDict(
            (f.name, f.copy()) for f in self.dtype.fields
        )
        for f in other.dtype.fields:
            merged[f.name] = f.copy()
        return Schema(
            inner_fields=merged,
            metadata=self._merge_metadata(self.metadata, other.metadata),
            name=self._rhs_wins_name(self, other),
            nullable=self.nullable or other.nullable,
        )

    def __radd__(self, other: Any) -> "Schema":
        if other == 0:
            return self.copy()
        other = self._coerce_other(other)
        return other.__add__(self)

    def __iadd__(self, other: Any) -> "Schema":
        other = self._coerce_other(other)
        merged = list(self.dtype.fields)
        seen = {f.name: i for i, f in enumerate(merged)}
        for f in other.dtype.fields:
            if f.name in seen:
                merged[seen[f.name]] = f.copy()
            else:
                seen[f.name] = len(merged)
                merged.append(f.copy())
        self._set_dtype_fields(merged)
        object.__setattr__(self, "metadata", self._merge_metadata(self.metadata, other.metadata))
        if other.name and other.name != DEFAULT_FIELD_NAME:
            object.__setattr__(self, "name", other.name)
        return self

    def __sub__(self, other: Any) -> "Schema":
        other = self._coerce_other(other)
        remove_names = {f.name for f in other.dtype.fields}
        return Schema(
            inner_fields=[f.copy() for f in self.dtype.fields if f.name not in remove_names],
            metadata=dict(self.metadata) if self.metadata else None,
            name=self.name,
            nullable=self.nullable,
        )

    def __isub__(self, other: Any) -> "Schema":
        other = self._coerce_other(other)
        remove_names = {f.name for f in other.dtype.fields}
        self._set_dtype_fields(
            f for f in self.dtype.fields if f.name not in remove_names
        )
        return self

    def __and__(self, other: Any) -> "Schema":
        other = self._coerce_other(other)
        keep_names = {f.name for f in other.dtype.fields}
        return Schema(
            inner_fields=[f.copy() for f in self.dtype.fields if f.name in keep_names],
            metadata=self._merge_metadata(self.metadata, other.metadata),
            name=self._rhs_wins_name(self, other),
            nullable=self.nullable or other.nullable,
        )

    def __iand__(self, other: Any) -> "Schema":
        other = self._coerce_other(other)
        keep_names = {f.name for f in other.dtype.fields}
        self._set_dtype_fields(
            f for f in self.dtype.fields if f.name in keep_names
        )
        object.__setattr__(self, "metadata", self._merge_metadata(self.metadata, other.metadata))
        if other.name and other.name != DEFAULT_FIELD_NAME:
            object.__setattr__(self, "name", other.name)
        return self

    def __or__(self, other: Any) -> "Schema":
        return self.__add__(other)

    def __ror__(self, other: Any) -> "Schema":
        other = self._coerce_other(other)
        return other.__add__(self)

    def __ior__(self, other: Any) -> "Schema":
        return self.__iadd__(other)

    # ------------------------------------------------------------------
    # Tagging
    # ------------------------------------------------------------------

    def autotag(
        self,
        tags: dict[AnyStr, AnyStr] | None = None,
    ) -> "Schema":
        primary_key_names = self._pop_field_name_list(b"primary_key")
        partition_by_names = self._pop_field_name_list(b"partition_by")
        cluster_by_names = self._pop_field_name_list(b"cluster_by")

        new_fields: list[Field] = []
        for f in self.dtype.fields:
            if primary_key_names and f.name in primary_key_names:
                f.with_primary_key(True, inplace=True)
            if partition_by_names and f.name in partition_by_names:
                f.with_partition_by(True, inplace=True)
            if cluster_by_names and f.name in cluster_by_names:
                f.with_cluster_by(True, inplace=True)
            new_fields.append(f.autotag())

        self.update_tags(tags)
        return Schema(
            inner_fields=new_fields,
            metadata=self.metadata,
            name=self.name,
            nullable=self.nullable,
        )

    def _pop_field_name_list(self, key: bytes) -> set[str]:
        """Pop a ``key`` -> field-name-list entry off ``self.metadata``.

        Accepts a JSON array (``'["a","b"]'``) or a dot-separated string
        (``"a.b"``). Returns an empty set when the key is missing or empty
        — and removes the key either way so it does not leak through to
        the engine schemas.
        """
        if not self.metadata:
            return set()
        raw = self.metadata.pop(key, None)
        if not raw:
            return set()
        if raw.startswith(b"[") and raw.endswith(b"]"):
            return set(json.loads(raw))
        return set(raw.decode().split("."))

    # ------------------------------------------------------------------
    # Engine schema export — Field provides ``to_arrow_schema`` /
    # ``to_polars_schema`` / ``to_spark_schema`` for any struct-shaped
    # field; the only thing Schema adds is the constraint-aware
    # filtering (children with ``constraint_key`` set don't make it
    # into the engine schema since they're a yggdrasil concept that
    # has no arrow/polars/spark equivalent).
    # ------------------------------------------------------------------

    def to_arrow_schema(self) -> pa.Schema:
        if self._arrow_schema is not None:
            return self._arrow_schema
        meta = dict(self.metadata) if self.metadata else {}
        if self.name and self.name != DEFAULT_FIELD_NAME:
            meta.setdefault(b"name", self.name.encode("utf-8"))
        if self.nullable:
            meta.setdefault(b"nullable", b"true")

        arrow_fields = [f.to_arrow_field() for f in self.fields]
        if not arrow_fields:
            return pa.schema([], metadata=meta or None)

        built = pa.schema(arrow_fields, metadata=meta or None)
        object.__setattr__(self, "_arrow_schema", built)
        return built

    def to_polars_schema(self) -> "polars.Schema":
        if self._polars_schema is not None:
            return self._polars_schema
        pl = get_polars()

        if not self.fields:
            return pl.Schema([])

        built = pl.Schema(
            [(f.name, f.dtype.to_polars()) for f in self.fields]
        )
        object.__setattr__(self, "_polars_schema", built)
        return built

    def to_spark_schema(self) -> "pst.StructType":
        if self._spark_schema is not None:
            return self._spark_schema

        pyspark_sql = get_spark_sql()

        if not self.fields:
            return pyspark_sql.types.StructType([])

        built = pyspark_sql.types.StructType(
            [f.to_pyspark_field() for f in self.fields]
        )

        object.__setattr__(self, "_spark_schema", built)
        return built

    def to_polars_flavor(self) -> "polars.Schema":
        return self.to_polars_schema()

    def to_spark_flavor(self) -> "pst.StructType":
        return self.to_spark_schema()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

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
        if isinstance(f, Schema):
            return f
        struct = f.to_struct()
        return cls(
            inner_fields=struct.children_fields,
            metadata=struct.metadata,
            name=struct.name if struct.name != DEFAULT_FIELD_NAME else None,
            nullable=struct.nullable,
        )

    @classmethod
    def from_any_fields(
        cls,
        fields: Iterable[Field | Any],
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
    ) -> "Schema":
        return cls(
            inner_fields=[Field.from_any(f) for f in fields],
            metadata=_normalize_metadata(metadata, tags),
        )

    @classmethod
    def from_arrow(cls, obj):
        return cls.from_field(Field.from_arrow(obj))

    @classmethod
    def from_spark(cls, obj: Any, from_metadata: bool = True) -> "Schema":
        # Field.from_spark routes through ``cls(name=..., dtype=...,
        # nullable=..., metadata=...)`` — Schema's init takes
        # ``inner_fields``, not ``dtype``, so the inherited path raises.
        # Lift the Field result instead.
        return cls.from_field(Field.from_spark(obj, from_metadata=from_metadata))

    @classmethod
    def from_path(cls, path: Any) -> "Schema":
        return path_class().from_(path).as_media().collect_schema()

    @classmethod
    def from_dict(cls, d: dict[str, Any], default: Any = ...) -> "Schema":
        f = Field.from_dict(d, default=None)
        if f is None:
            if default is ...:
                raise ValueError("Schema.from_dict requires a Field default for new keys")
            return default
        return cls.from_field(f)

    def to_dict(self, dump_parent: bool = False):
        return self.to_field().to_dict(dump_parent=dump_parent)

    def to_field(self) -> Field:
        """Return this schema as a plain :class:`Field`.

        Used by callers that need the Field shape without the
        Schema-specific surface (set ops, mapping). The returned
        Field shares this schema's dtype and metadata.
        """
        return Field(
            name=self.name,
            dtype=self.dtype,
            nullable=self.nullable,
            metadata=self.metadata,
        )

    # ------------------------------------------------------------------
    # Field-shaped overrides — return Schema, not Field
    # ------------------------------------------------------------------
    #
    # ``with_name`` is inherited from :class:`Field` — the inplace
    # branch returns ``self`` (already a Schema) and the copy branch
    # routes through :meth:`Schema.copy`. Only ``with_metadata`` needs
    # a Schema-specific override because it carries an ``update`` flag
    # that merges instead of replacing.

    def with_metadata(
        self,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
        inplace: bool = False,
        update: bool = True,
    ) -> "Schema":
        new_metadata = _normalize_metadata(metadata, tags)

        if not new_metadata:
            return self if (update or inplace) else self.copy(metadata=None)

        if inplace:
            if update and self.metadata:
                self.metadata.update(new_metadata)
            else:
                object.__setattr__(self, "metadata", new_metadata)
            return self

        if update and self.metadata:
            new_metadata = {**self.metadata, **(new_metadata or {})}

        return self.copy(metadata=new_metadata)


@register_converter(Any, Schema)
def any_to_schema(obj: Any, _: Any):
    return Schema.from_any(obj)
