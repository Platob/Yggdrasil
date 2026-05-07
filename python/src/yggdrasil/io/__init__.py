from __future__ import annotations

# URL and IOStats are eager — they don't pull anything that re-enters
# ``yggdrasil.data``. Holder/Memory/MemoryStream are deferred via PEP
# 562 ``__getattr__`` because :class:`Holder` inherits
# :class:`yggdrasil.io.tabular.base.Tabular`, which imports
# :class:`yggdrasil.data.schema.Schema`. Schema in turn pulls
# ``data_field`` — and when ``yggdrasil.data`` is the entry point of the
# import (e.g. via ``data/types/base.py`` → ``data.enums.mime_type`` →
# ``yggdrasil.io``), ``data_field`` is still mid-flight, so eagerly
# loading Holder here closes a circular import.
#
# Buffer (BytesIO / BufferLike) is lazy for the same reason: pulling it
# would trigger the buffer/primitive/tabular chain.
from .url import URL
from .io_stats import IOStats


_LAZY_DATA_NAMES = {"Holder", "Memory", "MemoryStream"}
_LAZY_BUFFER_NAMES = {"BytesIO", "BufferLike"}


def __getattr__(name: str):
    if name in _LAZY_DATA_NAMES:
        if name == "Holder":
            from .holder import Holder as value
        elif name == "Memory":
            from .memory import Memory as value
        else:
            from .memory_stream import MemoryStream as value
        globals()[name] = value
        return value
    if name in _LAZY_BUFFER_NAMES:
        from . import bytes_io as _buffer

        value = getattr(_buffer, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(
        set(globals())
        | _LAZY_DATA_NAMES
        | _LAZY_BUFFER_NAMES
        | {"URL", "IOStats"}
    )
