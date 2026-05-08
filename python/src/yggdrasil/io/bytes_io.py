"""Stdlib :class:`typing.BinaryIO`-compatible facade over :class:`IO`.

:class:`BytesIO` is a thin shim that pairs the rich :class:`IO`
substrate (holder + cursor + format helpers) with the
:class:`typing.BinaryIO` protocol, so external libraries that
type-check against the stdlib file-like interface (pandas,
pyarrow, zipfile, …) continue to accept Yggdrasil byte buffers.

The only divergence from :class:`IO` is the :attr:`mode` property,
which returns the POSIX mode string (``"rb"`` / ``"wb+"`` / …)
instead of the typed :class:`Mode` enum — pandas and friends
test ``"b" in handle.mode`` to dispatch binary vs text reads.
The typed mode is still available via ``self._mode``.
"""

from __future__ import annotations

from typing import BinaryIO, TypeVar

from yggdrasil.data.options import CastOptions
from yggdrasil.io.base import IO

__all__ = ["BytesIO"]


O = TypeVar("O", bound=CastOptions)


class BytesIO(IO[bytes, O], BinaryIO):
    """:class:`IO` parameterised on :class:`bytes` with stdlib parity.

    Construction is unchanged from :class:`IO` — every shape
    (``BytesIO(b"..")``, ``BytesIO(path=...)``, ``BytesIO(holder=...)``,
    ``BytesIO(media_type=...)``) routes through :meth:`IO.__new__`'s
    format dispatch, so format-specific calls land on the registered
    leaf (:class:`ParquetIO`, :class:`CsvIO`, …) automatically.
    """

    @property
    def mode(self) -> str:
        """POSIX mode string — stdlib :class:`IO[bytes]` parity.

        pandas / pyarrow / zipfile inspect ``.mode`` for substrings
        like ``"b"`` to dispatch binary vs text reads, so this surface
        returns the os-mode form (``"rb+"`` / ``"wb+"`` / ``"ab+"`` /
        ``"xb+"``) instead of the :class:`Mode` enum. The typed mode
        is still available via ``self._mode``.
        """
        return self._mode.os_mode
