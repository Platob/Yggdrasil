"""StructField — a :class:`Field` whose dtype is a :class:`StructType`.

``StructField`` is a thin type-hinting subclass of :class:`Field`:
every schema-shaped operation lives on :class:`Field` itself. The
subclass exists so the constructor can take a list of children
directly (``StructField([f1, f2])``) instead of the dtype-by-dtype
shape Field requires.

``Schema`` is exported as a historical alias for ``StructField``.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

import pyarrow as pa

from yggdrasil.data.cast.registry import register_converter
from yggdrasil.data.constants import DEFAULT_FIELD_NAME
from .data_field import Field, field, _normalize_inner_fields
from .types.nested import StructType


__all__ = [
    "StructField",
    "Schema",
    "schema",
    "Field",
    "field",
]


def schema(
    fields: Iterable[Field | pa.Field | str] | Field,
    *other: Field | pa.Field,
    name: str = DEFAULT_FIELD_NAME,
    nullable: bool = False,
    metadata: dict[bytes | str, bytes | str | object] | None = None,
    tags: dict[bytes | str, bytes | str | object] | None = None,
) -> "StructField":
    if fields is None:
        fields = ()
    elif isinstance(fields, StructField):
        if not metadata:
            metadata = fields.metadata
        if name == DEFAULT_FIELD_NAME:
            name = fields.name
        fields = fields.children
    elif isinstance(fields, Field):
        fields = (fields,)
    elif not isinstance(fields, (list, set, tuple)):
        fields = (fields,)

    if other:
        fields = (*fields, *other)

    return StructField(
        fields,
        name=name,
        nullable=nullable,
        metadata=metadata,
        tags=tags,
    )


@dataclasses.dataclass(repr=False, eq=False, frozen=True, init=False)
class StructField(Field):
    """A :class:`Field` whose ``dtype`` is a :class:`StructType`.

    ``StructField([f1, f2, ...])`` builds a struct field from its
    children directly — sugar for the equivalent
    ``Field(name=..., dtype=StructType(fields=(...,)))`` chain that
    :class:`Field`'s constructor accepts. Every schema-shaped method
    (mapping surface, set operators, engine schema export, autotag,
    struct-aware ``equals``) is inherited from :class:`Field`.
    """

    # Strict shape — StructField's dtype is always a StructType.
    dtype: StructType

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """``StructField(fields, name=..., metadata=..., ...)``.

        ``fields`` is the children list — :class:`Field` instances,
        :class:`pa.Field` instances, or anything :meth:`Field.from_any`
        accepts. ``dtype=<StructType>`` is accepted as an alternative
        for callers (and the ``Field.__new__`` redirect) that already
        have a built struct dtype in hand; pass one or the other, not
        both.

        The wide ``*args / **kwargs`` shape is here because
        ``Field.__new__`` redirects struct-shaped ``Field(...)`` calls
        (positional *or* keyword) to this class and Python then
        re-enters ``__init__`` on the already-stamped instance with
        the original Field arguments. The re-init guard below
        absorbs that pass before any signature parsing runs.
        """
        # Re-init guard for the Field.__new__ struct redirect.
        try:
            if self.dtype is not None:
                return
        except AttributeError:
            pass

        fields: Any = ...
        if args:
            fields = args[0]
            args = args[1:]
        if "fields" in kwargs:
            if fields is not ...:
                raise TypeError(
                    "StructField(): got multiple values for argument 'fields'"
                )
            fields = kwargs.pop("fields")
        if fields is ...:
            fields = None

        dtype = kwargs.pop("dtype", None)
        name = kwargs.pop("name", DEFAULT_FIELD_NAME)
        nullable = kwargs.pop("nullable", False)
        metadata = kwargs.pop("metadata", None)
        tags = kwargs.pop("tags", None)
        parent = kwargs.pop("parent", None)

        if args or kwargs:
            raise TypeError(
                f"StructField() got unexpected arguments: "
                f"args={args!r}, kwargs={sorted(kwargs)!r}"
            )

        if dtype is not None:
            if fields:
                raise TypeError(
                    "StructField(): pass either fields=... or dtype=..., not both"
                )
            resolved = dtype
        else:
            children = _normalize_inner_fields(fields)
            resolved = StructType(fields=tuple(children))

        Field.__init__(
            self,
            name=name,
            dtype=resolved,
            nullable=bool(nullable),
            metadata=metadata,
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
        """Identity — StructField IS a Field. Kept for type-narrowing callers."""
        return self

    def as_spark(self) -> "StructField":
        """Return a :class:`StructField` whose dtype is Spark-compatible."""
        return self._rewrap_with_dtype(self.dtype.as_spark())

    def as_polars(self) -> "StructField":
        """Return a :class:`StructField` whose dtype is Polars-compatible."""
        return self._rewrap_with_dtype(self.dtype.as_polars())

    def _rewrap_with_dtype(self, dtype: StructType) -> "StructField":
        if dtype is self.dtype:
            return self
        return StructField(
            tuple(dtype.fields),
            name=self.name,
            nullable=self.nullable,
            metadata=self.metadata,
        )


# Historical alias — call sites that imported ``Schema`` keep working.
Schema = StructField


@register_converter(Any, StructField)
def any_to_struct_field(obj: Any, _: Any):
    if isinstance(obj, StructField):
        return obj
    return StructField.from_field(Field.from_any(obj))
