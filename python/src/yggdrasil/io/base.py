"""Generic typed I/O over a managed :class:`Holder`.

:class:`IO[T, O]` is the thin base that pairs a :class:`Tabular`
contract with a :class:`Holder` substrate and a :class:`Disposable`
lifecycle. It owns the holder slots and the acquire/release plumbing
that every byte-, text-, or block-shaped handle needs; it does NOT
declare a typed read/write surface — the *T* parameter is a marker
for the concrete chunk type each subclass exposes
(``IO[bytes, O]`` for :class:`BytesIO`, hypothetically
``IO[str, O]`` for a text-shaped handle).

The split lets format-specific leaves stay focused: a leaf inherits
``IO[bytes, O]`` once and gets holder ownership, lifecycle cascade,
and the Tabular registry hook for free, then layers its own
chunk-typed protocol on top.
"""

from __future__ import annotations

import io as _stdlib_io
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from yggdrasil.data.enums.mode import Mode, ModeLike
from yggdrasil.data.options import CastOptions
from yggdrasil.disposable import Disposable
from yggdrasil.io.tabular import Tabular

if TYPE_CHECKING:
    from yggdrasil.io.holder import Holder


__all__ = ["IO", "T", "O"]


T = TypeVar("T")
O = TypeVar("O", bound=CastOptions)


class IO(Tabular[O], Disposable, Generic[T, O]):
    """Tabular handle bound to a managed :class:`Holder`.

    Holds the two pieces every byte- or chunk-shaped handle needs:

    * :attr:`_holder` — the byte substrate the handle reads/writes.
    * :attr:`_owns_holder` — whether closing the handle should close
      the holder (``True`` when the handle constructed the holder
      itself; ``False`` when it borrows one from a longer-lived
      caller).

    Lifecycle: :meth:`_acquire` acquires the holder iff owned, and
    :meth:`_release` closes it on the way out. Subclasses chain
    ``super()._acquire()`` / ``super()._release()`` and add their
    own bytes-, text-, or block-level setup around the call.

    *T* is the chunk type the concrete handle exposes (``bytes`` for
    :class:`BytesIO`); it has no effect at this layer beyond
    parametrising the generic for downstream type checkers. *O* is
    the :class:`CastOptions` subtype carried by :class:`Tabular`.
    """

    __slots__ = ("_holder", "_owns_holder", "_pos", "_mode")

    def __new__(
        cls,
        data: Any = None,
        *,
        holder: "Holder | None" = None,
        owns_holder: bool = False,
        mode: ModeLike = "rb+",
        path: Any = None,
        binary: Any = None,
        url: Any = None,
        **kwargs: Any,
    ):
        """Allocate the instance and resolve a holder via :class:`Holder`.

        The holder argument is the load-bearing one: when supplied,
        the instance borrows it (``owns_holder`` controls teardown).
        When ``holder`` is ``None``, the holder-shaped kwargs
        (``data`` / ``path`` / ``binary`` / ``url``) are forwarded
        to :class:`Holder`, whose own ``__new__`` scheme-dispatches
        to the right concrete subclass (:class:`Memory`,
        :class:`LocalPath`, :class:`S3Path`, …). The new holder is
        stamped onto the instance and the instance owns it.

        Inputs :class:`Holder` doesn't recognize (file-like objects,
        backend-specific shapes) are left for the subclass
        ``__init__`` to drain — :meth:`_holder` stays ``None`` until
        the subclass populates it.
        """
        instance = super().__new__(cls)
        instance._holder = holder
        instance._owns_holder = bool(owns_holder)
        if holder is None:
            from yggdrasil.io.holder import Holder as _Holder
            try:
                instance._holder = _Holder(
                    data=data, path=path, binary=binary, url=url,
                )
                instance._owns_holder = True
            except TypeError:
                # Subclass may have richer drain logic (e.g. file-like
                # objects in :class:`BytesIO.from_`). Leave the slots
                # at their initial values for the subclass to finish.
                pass
        return instance

    def __init__(
        self,
        data: Any = None,
        *,
        holder: "Holder | None" = None,
        owns_holder: bool = False,
        mode: ModeLike = "rb+",
        path: Any = None,
        binary: Any = None,
        url: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Tabular / Disposable bookkeeping.

        ``_holder`` and ``_owns_holder`` are populated by
        :meth:`__new__`; this hook only chains the ``Tabular`` /
        ``Disposable`` super-init. The holder-shaped kwargs are
        accepted purely so subclass calls like
        ``BytesIO(path="x.csv")`` reach :meth:`__new__` and get
        absorbed before they reach :class:`Tabular.__init__`.

        ``mode`` follows stdlib :func:`open` semantics. Subclasses
        consume it for read/write predicates, EOF positioning, and
        truncate-on-open behavior.
        """
        super().__init__(**kwargs)
        self._pos: int = 0
        self._mode: Mode = Mode.from_(mode)

    # ==================================================================
    # Identity
    # ==================================================================

    @property
    def holder(self) -> "Holder":
        """The bound :class:`Holder`."""
        return self._holder

    @property
    def owns_holder(self) -> bool:
        """Whether closing self also closes the holder."""
        return self._owns_holder

    # ==================================================================
    # Disposable lifecycle — acquire / release the holder iff owned
    # ==================================================================

    def _acquire(self) -> None:
        """Acquire the holder when owned. Subclasses chain super first."""
        if self._owns_holder:
            self._holder.acquire()

    def _release(self) -> None:
        """Close the holder when owned. Subclasses chain super last."""
        if self._owns_holder:
            try:
                self._holder.close()
            except Exception:
                pass

    # ==================================================================
    # Cursor — position tracking shared by every chunk-shaped child
    # ==================================================================

    @property
    def size(self) -> int:
        """Live size of the bound holder.

        Subclasses that interpose a buffered scratch (e.g.
        :class:`BytesIO` while open) override to read from the
        active surface instead of the durable holder.
        """
        return self._holder.size

    def tell(self) -> int:
        """Current cursor position."""
        return self._pos

    def seek(self, offset: int, whence: int = _stdlib_io.SEEK_SET) -> int:
        """Seek to *offset* relative to *whence*.

        Mirrors :meth:`io.IOBase.seek` with two ergonomic deviations
        that match the rest of the codebase:

        * ``seek(-1, SEEK_SET)`` is a "go to end" sentinel — pairs
          with ``read(-1)`` / "read all". Any other negative
          ``SEEK_SET`` offset raises :class:`ValueError`.
        * ``SEEK_CUR`` / ``SEEK_END`` with a negative offset that
          would land before byte 0 clamps to 0 instead of raising.
        """
        offset = int(offset)
        size = self.size
        if whence == _stdlib_io.SEEK_SET:
            if offset == -1:
                self._pos = size
            elif offset < 0:
                raise ValueError(
                    f"Negative SEEK_SET offset {offset!r} is invalid; "
                    f"use SEEK_END to count from the end."
                )
            else:
                self._pos = offset
        elif whence == _stdlib_io.SEEK_CUR:
            self._pos = max(0, self._pos + offset)
        elif whence == _stdlib_io.SEEK_END:
            self._pos = max(0, size + offset)
        else:
            raise ValueError(f"Invalid whence: {whence!r}")
        return self._pos

    def seekable(self) -> bool:
        return True

    # ==================================================================
    # Mode predicates — stdlib open() semantics
    # ==================================================================

    @property
    def mode(self) -> Mode:
        """Normalized :class:`Mode` for this handle.

        Stored as an enum so predicates like :meth:`readable`,
        :meth:`writable`, :meth:`appendable` route through one
        canonical token instead of re-parsing strings at every
        call site. The original POSIX form is recoverable via
        ``self.mode.os_mode``.
        """
        return self._mode

    def readable(self) -> bool:
        return self._mode.readable

    def writable(self) -> bool:
        return self._mode.writable

    def appendable(self) -> bool:
        """True when writes append at EOF — :data:`Mode.APPEND` only."""
        return self._mode.appendable
