"""Back-compat shim — DV codec lives at :mod:`yggdrasil.io.delta.deletion_vector`.

Re-exports the public API plus a few underscore-prefixed helpers
the old test suite reaches for (``_decode_payload``, ``_MAGIC_SIMPLE``,
``_MAGIC_ROARING_64``). New code should import from the canonical
module directly.
"""

from __future__ import annotations

from yggdrasil.io.delta.deletion_vector import (  # noqa: F401
    DeletionVector,
    DeletionVectorDescriptor,
    _decode_payload,
    _MAGIC_ROARING_64,
    _MAGIC_SIMPLE,
    decode_deletion_vector,
    encode_inline_deletion_vector,
    mask_batch_with_dv,
    write_uuid_deletion_vector,
)

__all__ = [
    "DeletionVector",
    "DeletionVectorDescriptor",
    "decode_deletion_vector",
    "encode_inline_deletion_vector",
    "mask_batch_with_dv",
    "write_uuid_deletion_vector",
]
