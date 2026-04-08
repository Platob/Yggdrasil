"""Pickle serializers for :class:`~yggdrasil.io.PreparedRequest` and
:class:`~yggdrasil.io.response.Response`.

Wire layout
-----------
``PreparedRequestSerialized``  (tag ``PREPARED_REQUEST = 222``)
    payload = Arrow IPC file bytes for a single-row RecordBatch produced by
    ``request.to_arrow_batch(parse=False)``.  All request fields — method, URL
    parts, promoted headers, remaining headers, tags, body bytes, body hash and
    sent_at timestamp — are round-tripped through Arrow's own IPC format.
    Reconstruction uses ``PreparedRequest.from_arrow(batch)``.

``ResponseSerialized``  (tag ``RESPONSE = 223``)
    payload = Arrow IPC file bytes for a single-row RecordBatch produced by
    ``response.to_arrow_batch(parse=False)``.  All response fields including
    the embedded request are round-tripped.  Reconstruction uses
    ``Response.from_arrow_tabular(batch)``.

Both store a ``ygg_object`` hint in the wire Header for debugging; the tag
alone is sufficient for dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Mapping

import pyarrow as pa

from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags
from yggdrasil.pickle.ser.pyarrow import (
    _merge_metadata,
    _record_batch_to_ipc_file_buffer,
    _table_from_ipc_file_buffer,
)
from yggdrasil.pickle.ser.constants import CODEC_NONE

if TYPE_CHECKING:
    from yggdrasil.io.request import PreparedRequest
    from yggdrasil.io.response import Response

__all__ = [
    "HttpSerialized",
    "PreparedRequestSerialized",
    "ResponseSerialized",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _decode_to_arrow_buffer(ser: Serialized) -> pa.Buffer:
    """Return the raw payload as a ``pa.Buffer`` (decompress if needed)."""
    if ser.codec == CODEC_NONE:
        return pa.py_buffer(ser.to_bytes())
    return pa.py_buffer(ser.decode())


def _batch_from_payload(ser: Serialized) -> pa.RecordBatch:
    table = _table_from_ipc_file_buffer(_decode_to_arrow_buffer(ser))
    if table.num_rows == 0:
        raise ValueError("HTTP IPC payload contained no rows")
    return table.to_batches()[0]


# ---------------------------------------------------------------------------
# base dispatch hub
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class HttpSerialized(Serialized[object]):
    """Base / dispatch hub for ``yggdrasil.io`` HTTP serializers.

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
        from yggdrasil.io.request import PreparedRequest
        from yggdrasil.io.response import Response

        if isinstance(obj, Response):
            # Response must come first — it IS-A PreparedRequest container
            return ResponseSerialized.from_value(obj, metadata=metadata, codec=codec)

        if isinstance(obj, PreparedRequest):
            return PreparedRequestSerialized.from_value(obj, metadata=metadata, codec=codec)

        return None


# ---------------------------------------------------------------------------
# PreparedRequest
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PreparedRequestSerialized(HttpSerialized):
    """Serialize a :class:`~yggdrasil.io.PreparedRequest` as Arrow IPC."""

    TAG: ClassVar[int] = Tags.PREPARED_REQUEST

    @property
    def value(self) -> "PreparedRequest":
        from yggdrasil.io.request import PreparedRequest

        batch = _batch_from_payload(self)
        return next(PreparedRequest.from_arrow(batch, normalize=False))

    def as_python(self) -> "PreparedRequest":
        return self.value

    @classmethod
    def from_value(
        cls,
        request: "PreparedRequest",
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> "PreparedRequestSerialized":
        batch = request.to_arrow_batch(parse=False)
        wire_metadata = _merge_metadata(
            metadata,
            {
                b"ygg_object": b"prepared_request",
                b"method": request.method.encode(),
                b"url": request.url.to_string().encode(),
            },
        )
        buf = _record_batch_to_ipc_file_buffer(batch, metadata=wire_metadata)
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
        from yggdrasil.io.request import PreparedRequest

        if isinstance(obj, PreparedRequest):
            return cls.from_value(obj, metadata=metadata, codec=codec)
        return None


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ResponseSerialized(HttpSerialized):
    """Serialize a :class:`~yggdrasil.io.response.Response` as Arrow IPC."""

    TAG: ClassVar[int] = Tags.RESPONSE

    @property
    def value(self) -> "Response":
        from yggdrasil.io.response import Response

        batch = _batch_from_payload(self)
        return next(Response.from_arrow_tabular(batch))

    def as_python(self) -> "Response":
        return self.value

    @classmethod
    def from_value(
        cls,
        response: "Response",
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> "ResponseSerialized":
        batch = response.to_arrow_batch(parse=False)
        wire_metadata = _merge_metadata(
            metadata,
            {
                b"ygg_object": b"response",
                b"status_code": str(response.status_code).encode(),
                b"url": response.request.url.to_string().encode(),
            },
        )
        buf = _record_batch_to_ipc_file_buffer(batch, metadata=wire_metadata)
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
        from yggdrasil.io.response import Response

        if isinstance(obj, Response):
            return cls.from_value(obj, metadata=metadata, codec=codec)
        return None


# ---------------------------------------------------------------------------
# registration — run at import time so tag dispatch works immediately
# ---------------------------------------------------------------------------

for _cls in (PreparedRequestSerialized, ResponseSerialized):
    Tags.register_class(_cls, tag=_cls.TAG)

