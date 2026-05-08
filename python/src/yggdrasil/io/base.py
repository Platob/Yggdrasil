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

from typing import TYPE_CHECKING, Any, Generic, TypeVar

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

    __slots__ = ("_holder", "_owns_holder")

    def __init__(
        self,
        *,
        holder: "Holder",
        owns_holder: bool = False,
        **kwargs: Any,
    ) -> None:
        """Bind *holder* and record ownership.

        ``owns_holder=True`` transfers close-ownership of *holder*
        to this handle — :meth:`_release` will close it. ``False``
        (the default) borrows the holder; the caller stays
        responsible for its lifecycle.
        """
        super().__init__(**kwargs)
        self._holder: "Holder" = holder
        self._owns_holder: bool = bool(owns_holder)

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
