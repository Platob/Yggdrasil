"""Pickle serializers for :class:`~yggdrasil.data.Field` and :class:`~yggdrasil.data.Schema`.

Wire layout
-----------
``FieldSerialized``  (tag ``YGG_FIELD = 303``)
    payload = Arrow IPC file bytes that hold a single-field ``pa.Schema``.
    The field's name, Arrow type, nullability and all metadata are round-tripped
    through Arrow's own IPC format.

``SchemaSerialized``  (tag ``YGG_SCHEMA = 304``)
    payload = Arrow IPC file bytes that hold a ``pa.Schema``.
    All fields (with their metadata) and the schema-level metadata are
    preserved via Arrow IPC.

Both serializers store a ``ygg_object`` key in the wire Header metadata for
debugging purposes (it is never needed for reconstruction — the tag suffices).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Mapping

import pyarrow as pa

from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags
from yggdrasil.pickle.ser.pyarrow import (
    _merge_metadata,
    _schema_to_ipc_file_buffer,
    _schema_from_ipc_file_buffer,
)

if TYPE_CHECKING:
    from yggdrasil.data.field import Field
    from yggdrasil.data.schema import Schema

__all__ = [
    "DataSerialized",
    "FieldSerialized",
    "SchemaSerialized",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _decode_to_arrow_buffer(ser: Serialized) -> pa.Buffer:
    """Return the raw payload as a ``pa.Buffer`` (decompresses if needed)."""
    from yggdrasil.pickle.ser.constants import CODEC_NONE

    if ser.codec == CODEC_NONE:
        return pa.py_buffer(ser.to_bytes())
    return pa.py_buffer(ser.decode())


# ---------------------------------------------------------------------------
# base dispatch hub
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DataSerialized(Serialized[object]):
    """Base / dispatch hub for ``yggdrasil.data`` serializers.

    Not registered as a concrete tag — only the concrete subclasses are.
    """

    TAG: ClassVar[int] = -1  # intentionally invalid; never registered

    def as_python(self) -> object:
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def from_python_object(
        cls,
        obj: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> "Serialized[object] | None":
        from yggdrasil.data.field import Field
        from yggdrasil.data.schema import Schema

        if isinstance(obj, Field):
            return FieldSerialized.from_value(obj, metadata=metadata, codec=codec)

        if isinstance(obj, Schema):
            return SchemaSerialized.from_value(obj, metadata=metadata, codec=codec)

        return None


# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FieldSerialized(DataSerialized):
    """Serialize a :class:`~yggdrasil.data.Field` as Arrow IPC (single-field schema)."""

    TAG: ClassVar[int] = Tags.YGG_FIELD

    @property
    def value(self) -> "Field":
        from yggdrasil.data.field import Field

        arrow_schema = _schema_from_ipc_file_buffer(_decode_to_arrow_buffer(self))

        if len(arrow_schema) != 1:
            raise ValueError(
                f"YGG_FIELD payload must contain exactly 1 Arrow field, "
                f"got {len(arrow_schema)}"
            )

        return Field.from_arrow(arrow_schema.field(0))

    def as_python(self) -> "Field":
        return self.value

    @classmethod
    def from_value(
        cls,
        field: "Field",
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> "FieldSerialized":
        arrow_schema = pa.schema([field.to_arrow_field()])
        wire_metadata = _merge_metadata(metadata, {b"ygg_object": b"field"})
        buf = _schema_to_ipc_file_buffer(arrow_schema)
        return cls.build(  # type: ignore[return-value]
            tag=cls.TAG,
            data=buf,
            metadata=wire_metadata,
            codec=codec,
        )

    @classmethod
    def from_python_object(
        cls,
        obj: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> "Serialized[object] | None":
        from yggdrasil.data.field import Field

        if isinstance(obj, Field):
            return cls.from_value(obj, metadata=metadata, codec=codec)
        return None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SchemaSerialized(DataSerialized):
    """Serialize a :class:`~yggdrasil.data.Schema` as Arrow IPC."""

    TAG: ClassVar[int] = Tags.YGG_SCHEMA

    @property
    def value(self) -> "Schema":
        from yggdrasil.data.schema import Schema

        arrow_schema = _schema_from_ipc_file_buffer(_decode_to_arrow_buffer(self))
        return Schema.from_arrow(arrow_schema)

    def as_python(self) -> "Schema":
        return self.value

    @classmethod
    def from_value(
        cls,
        schema: "Schema",
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> "SchemaSerialized":
        arrow_schema = schema.to_arrow_schema()
        wire_metadata = _merge_metadata(metadata, {b"ygg_object": b"schema"})
        buf = _schema_to_ipc_file_buffer(arrow_schema)
        return cls.build(  # type: ignore[return-value]
            tag=cls.TAG,
            data=buf,
            metadata=wire_metadata,
            codec=codec,
        )

    @classmethod
    def from_python_object(
        cls,
        obj: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> "Serialized[object] | None":
        from yggdrasil.data.schema import Schema

        if isinstance(obj, Schema):
            return cls.from_value(obj, metadata=metadata, codec=codec)
        return None


# ---------------------------------------------------------------------------
# registration — run at import time so tag dispatch works immediately
# ---------------------------------------------------------------------------

for _cls in (FieldSerialized, SchemaSerialized):
    Tags.register_class(_cls, tag=_cls.TAG)

