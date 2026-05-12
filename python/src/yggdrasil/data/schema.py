"""StructField — a :class:`Field` whose dtype is a :class:`StructType`.

``StructField`` (historically ``Schema``) is just a thin type-hinting
subclass of :class:`Field`: every schema-shaped operation (mapping
surface, set operators, engine schema export, autotag propagation,
struct-aware ``equals``) lives on :class:`Field` itself. The subclass
is kept around so existing call sites that type-hint ``Schema`` (and
the ``schema(...)`` factory) keep their meaning, but no behaviour is
unique to it — ``StructField`` is just a ``Field`` with the
schema-style ``__init__`` signature.

``Schema`` remains as an alias for ``StructField`` so the existing
public surface keeps working unchanged.
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
    "StructField",
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
) -> "StructField":
    if fields is None:
        fields = []
    elif isinstance(fields, Field):
        fields = [fields]
    elif isinstance(fields, StructField):
        if not metadata:
            metadata = fields.metadata
        fields = fields.children
    elif not isinstance(fields, (list, set, tuple)):
        fields = [fields]

    if other:
        fields = list(fields)
        fields.extend(other)

    return StructField.from_any_fields(
        fields,
        metadata=metadata,
        tags=tags,
    )


@dataclasses.dataclass(repr=False, eq=False, frozen=True, init=False)
class StructField(Field):
    """A :class:`Field` whose ``dtype`` is a :class:`StructType`.

    Historically named ``Schema`` (still exported as that alias).
    StructField is *just* a struct field — every method that mattered
    on the old standalone class is now a regular :class:`Field` method
    that works for any struct dtype. This subclass keeps the
    schema-style ``__init__(inner_fields=...)`` signature so existing
    call sites don't need to convert, plus a header-style ``__repr__``;
    everything else is inherited.
    """

    # Strict shape — Schema's dtype is always a StructType.
    dtype: StructType

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """``StructField(inner_fields=..., metadata=...)`` — schema-style init.

        Wide ``*args / **kwargs`` signature so the same constructor
        can absorb the two call shapes that reach it:

        1. **Schema-style** — ``StructField(inner_fields=[...],
           metadata=..., name=..., nullable=...)`` — the historical
           ``Schema(...)`` surface.
        2. **Field-style re-init** — when ``Field.__new__`` redirects
           a struct ``Field(name=..., dtype=struct_t, ...)`` call to
           :class:`StructField`, Python follows up by calling
           ``StructField.__init__`` on the already-initialized
           instance with the original *Field*-positional args. The
           re-init guard short-circuits before any signature parsing,
           which avoids a positional-vs-keyword clash on ``metadata``.
        """
        # Re-init guard — slots already set by ``Field.__new__``.
        try:
            if self.dtype is not None:
                return
        except AttributeError:
            pass

        # Parse the schema-style call shape.
        inner_fields = kwargs.pop("inner_fields", ...)
        metadata = kwargs.pop("metadata", None)
        if args:
            if inner_fields is ...:
                inner_fields = args[0]
                args = args[1:]
            if args and metadata is None:
                metadata = args[0]
                args = args[1:]
            if args:
                raise TypeError(
                    f"StructField() takes at most 2 positional arguments "
                    f"(inner_fields, metadata); got {len(args) + 2} extra."
                )
        if inner_fields is ...:
            inner_fields = None
        name = kwargs.pop("name", None)
        nullable = kwargs.pop("nullable", None)
        tags = kwargs.pop("tags", None)
        parent = kwargs.pop("parent", None)
        dtype = kwargs.pop("dtype", None)
        default = kwargs.pop("default", None)
        if kwargs:
            raise TypeError(
                f"StructField() got unexpected keyword arguments: "
                f"{sorted(kwargs)}."
            )

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
            for f in self.children
        )
        comment = self.comment
        return f"StructField: {self.name!r} {comment!r}{body}"

    def to_field(self) -> Field:
        """Return this struct field as a :class:`Field` — i.e. ``self``.

        StructField IS a Field, so the cast is identity. Kept as a
        no-op alias for callers that want to type-narrow.
        """
        return self

    def as_spark(self) -> "StructField":
        """Return a :class:`StructField` whose dtype is Spark-compatible.

        Like :meth:`Field.as_spark`, but the result is wrapped back
        into a :class:`StructField` so callers chain through
        schema-shaped APIs without dropping to a plain :class:`Field`.
        When every child is already Spark-compatible the same instance
        is returned. Use :meth:`to_spark_schema` when you need an
        actual ``pyspark.sql.types.StructType``.
        """
        return self._rewrap_with_dtype(self.dtype.as_spark())

    def as_polars(self) -> "StructField":
        """Return a :class:`StructField` whose dtype is Polars-compatible.

        Same shape as :meth:`as_spark`, just delegating to
        :meth:`DataType.as_polars` on the inner dtype.
        """
        return self._rewrap_with_dtype(self.dtype.as_polars())

    def _rewrap_with_dtype(self, dtype) -> "StructField":
        if dtype is self.dtype:
            return self
        return StructField(
            inner_fields=tuple(dtype.fields),
            metadata=self.metadata,
            name=self.name,
            nullable=self.nullable,
        )


# Historical alias — every existing call site that imports ``Schema``
# from ``yggdrasil.data.schema`` keeps working; new code should reach
# for the descriptive ``StructField`` name.
Schema = StructField


@register_converter(Any, StructField)
def any_to_struct_field(obj: Any, _: Any):
    if isinstance(obj, StructField):
        return obj
    return StructField.from_field(Field.from_any(obj))
