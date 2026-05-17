"""Convenience alias :class:`BytesIO` over :class:`IO`.

After the Holder ↔ IO merge, :class:`IO` itself inherits from
:class:`typing.BinaryIO` and exposes the POSIX ``.mode`` string, so
external libraries that type-check against the stdlib file-like
interface (pandas, pyarrow, zipfile, …) accept it directly.
:class:`BytesIO` survives as a one-line, ergonomic shorthand for the
``IO[bytes, O]`` parameterisation — call sites that want the
"this is a binary buffer" intent in the type name keep working
without going through ``IO`` and threading the generic.
"""

from __future__ import annotations

from typing import TypeVar

from yggdrasil.data.options import CastOptions
from yggdrasil.io.base import IO

__all__ = ["BytesIO"]


O = TypeVar("O", bound=CastOptions)


class BytesIO(IO[bytes, O]):
    """Convenience alias — :class:`IO` parameterised on :class:`bytes`.

    Every construction shape inherited from :class:`IO`
    (``BytesIO(b"..")``, ``BytesIO(path=...)``, ``BytesIO(holder=...)``,
    ``BytesIO(media_type=...)``) routes through :meth:`IO.__new__`'s
    format dispatch, so format-specific calls still land on the
    registered leaf (:class:`ParquetFile`, :class:`CSVFile`, …) automatically.
    """
