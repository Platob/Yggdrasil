# yggdrasil/pyutils/dynamic_buffer/__init__.py
"""
Spill-to-disk byte buffer with compression and media-type detection.

Public API
----------

.. code-block:: text

    dynamic_buffer/
    ├── _types.py       BytesLike alias
    ├── _config.py      BufferConfig, DEFAULT_CONFIG
    ├── _codec.py       Codec enum  (magic-byte compression detection)
    ├── _media_type.py  MediaType   (magic-byte / text MIME inference)
    └── _buffer.py      BytesIO     (spill-to-disk buffer)

All four public symbols are re-exported from this package so existing
``from yggdrasil.pyutils.dynamic_buffer import BytesIO`` imports keep working
without modification.

Typical usage::

    from yggdrasil.pyutils.dynamic_buffer import BytesIO, BufferConfig

    buf = BytesIO()
    buf.write(payload)
    buf.seek(0)
    print(buf.media_type)   # e.g. MediaType('application/vnd.apache.parquet')
    print(buf.compression)  # e.g. None  or  <Codec.ZSTD: 'zstd'>
    buf.close()
"""

from __future__ import annotations

from .buffer import BytesIO
from .config import BufferConfig, DEFAULT_CONFIG
from .enums.codec import Codec
from .enums.media_type import MediaType
from .types import BytesLike

__all__ = [
    "BytesLike",
    "BufferConfig",
    "DEFAULT_CONFIG",
    "Codec",
    "MediaType",
    "BytesIO",
]