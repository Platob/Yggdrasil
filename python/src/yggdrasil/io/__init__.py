from __future__ import annotations

from .holder import Holder
from .memory import Memory
from .url import URL


# Lazy re-exports — a top-level ``from .buffer import BytesIO`` here would
# trigger the buffer/primitive/tabular chain (which reaches back into the
# ``yggdrasil.data`` layer via ``arrow.cast`` → ``data.options`` →
# ``data.schema``). A submodule access like ``from yggdrasil.io.enums.mode
# import Mode`` from inside ``yggdrasil.data.types.base`` runs *this*
# ``__init__`` first, so eager imports here form a cycle. PEP 562
# ``__getattr__`` defers ``BytesIO`` / ``BufferLike`` to first attribute
# access on the package — by then ``yggdrasil.data`` is fully loaded.

_LAZY_BUFFER_NAMES = {"BytesIO", "BufferLike"}


def __getattr__(name: str):
    if name in _LAZY_BUFFER_NAMES:
        from . import buffer as _buffer

        value = getattr(_buffer, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(
        set(globals())
        | _LAZY_BUFFER_NAMES
        | {"URL", "Holder", "Memory"}
    )
