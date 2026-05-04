"""Schema — a :class:`Field` whose dtype is a :class:`StructType`.

Schema is just a thin type-hinting subclass of :class:`Field`: every
schema-shaped operation (mapping surface, set operators, engine schema
export, autotag propagation, struct-aware ``equals``) lives on
:class:`Field` itself. The subclass is kept around so existing call
sites that type-hint ``Schema`` (and the ``schema(...)`` factory) keep
their meaning, but no behaviour is unique to it — ``Schema`` is just a
``Field`` with the schema-style ``__init__`` signature.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any, Mapping

import pyarrow as pa

from yggdrasil.data.cast.registry import register_converter
from yggdrasil.data.constants import DEFAULT_FIELD_NAME
from .data_field import (
    Field,
    field,
    _normalize_inner_fields,
    _peel_name_nullable,
)
from .types.nested import StructType


__all__ = [
    "Schema",
    "schema",
    "Field",
    "field",
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


@dataclasses.dataclass(repr=False, eq=False, frozen=True, init=False)
class Schema(Field):
    """A :class:`Field` whose ``dtype`` is a :class:`StructType`.

    Schema is *just* a struct field — every method that mattered on the
    old standalone class is now a regular :class:`Field` method that
    works for any struct dtype. This subclass keeps the schema-style
    ``__init__(inner_fields=...)`` signature so existing call sites
    don't need to convert, plus a header-style ``__repr__``; everything
    else is inherited.
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
        # Field-shaped escape hatch — accept ``dtype=`` so that
        # generic ``cls(name=..., dtype=..., ...)`` factories
        # inherited from :class:`Field` (``from_dataclass``,
        # ``from_pandas``, ``from_arrow_field``, …) also build a
        # ``Schema`` without each one knowing about the
        # ``inner_fields`` shim.
        dtype: Any = None,
        default: Any = None,
    ) -> None:
        if dtype is not None:
            if inner_fields is not None:
                raise TypeError(
                    "Schema(): pass either inner_fields= or dtype=, not both"
                )
            # Downgrade to Field's call shape — happens when a
            # generic factory (e.g. ``Schema.from_dataclass``) hands
            # us a fully-built dtype. Lift non-struct dtypes into a
            # single-child struct so the schema-shape contract holds.
            from .types.base import DataType
            resolved = DataType.from_any(dtype)
            if not isinstance(resolved, StructType):
                wrapped_name = name or DEFAULT_FIELD_NAME
                resolved = StructType(
                    fields=(
                        Field(
                            name=wrapped_name,
                            dtype=resolved,
                            nullable=True if nullable is None else bool(nullable),
                        ),
                    )
                )
            Field.__init__(
                self,
                name=name or DEFAULT_FIELD_NAME,
                dtype=resolved,
                nullable=False if nullable is None else bool(nullable),
                metadata=metadata,
                tags=tags,
                default=default,
                parent=parent,
            )
            return

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

    def __repr__(self):
        body = "".join(
            f"\n{f.pretty_format(level=1)}"
            for f in self.children_fields
        )
        comment = self.comment
        return f"Schema: {self.name!r} {comment!r}{body}"

    def to_field(self) -> Field:
        """Return this schema as a plain :class:`Field`.

        Kept for callers that explicitly want the non-Schema concrete
        type — the returned Field shares this schema's dtype and
        metadata. Now that Schema *is* a Field, most call sites can
        just use ``self`` directly.
        """
        return Field(
            name=self.name,
            dtype=self.dtype,
            nullable=self.nullable,
            metadata=self.metadata,
        )


@register_converter(Any, Schema)
def any_to_schema(obj: Any, _: Any):
    if isinstance(obj, Schema):
        return obj
    return Schema.from_field(Field.from_any(obj))
