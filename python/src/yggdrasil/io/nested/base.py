"""Folder-shaped :class:`Tabular` whose batches come from its children.

A :class:`NestedIO` IS a :class:`Tabular` over a folder. Two abstract
hooks — :meth:`iter_children` (read) and :meth:`make_child` (write)
— and the :class:`Tabular` batch contract derives from them. That's
the entire abstraction at this layer. Concrete folder formats
(:class:`FolderIO`, :class:`DeltaIO`, :class:`ZipIO`, …) layer
staging, ignore-rules, mode resolution, and parallelism on top.

Children, not buffers
---------------------

A child is a fully formed :class:`Tabular` — typically another
:class:`Tabular` for files (Parquet, CSV, IPC, …) or another
:class:`NestedIO` for sub-folders. Each child carries a
``parent`` back-pointer to the :class:`NestedIO` that yielded it,
so consumers can walk back up the tree.

Read derivation chains :meth:`iter_children` into a single Arrow
batch stream. Write derivation mints one fresh child via
:meth:`make_child` and drains the incoming batches into it. Concerns
that aren't "yield children / mint a child" — save modes, clearing,
staging-and-rename, schema-merge parallelism, upsert-via-rewrite —
live in subclasses.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Iterable, Iterator, TypeVar

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.io.tabular import Tabular
from yggdrasil.lazy_imports import path_class

if TYPE_CHECKING:
    from yggdrasil.io.path import Path


__all__ = ["NestedIO"]


O = TypeVar("O", bound=CastOptions)


class NestedIO(Tabular[O]):
    """Folder-shaped :class:`Tabular`. Children, not buffers.

    Subclasses implement two hooks:

    - :meth:`iter_children` — yield direct children for reading.
    - :meth:`make_child`    — mint a fresh child for writing.

    The :class:`Tabular` contract (``_read_arrow_batches`` /
    ``_write_arrow_batches``) is derived from those.
    """

    __slots__ = ("path",)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        data: Any = None,
        *,
        path: Any = None,
        parent: "Tabular | None" = None,
        **kwargs: Any,
    ) -> None:
        """Bind to a folder path. No I/O.

        ``data`` and ``path`` accept the same shape; ``path`` wins
        when both are supplied. ``parent`` records this IO's
        position in a folder-of-folders tree (set by the enclosing
        :class:`NestedIO` when it yields this one) and rides through
        ``Tabular.__init__``; ``None`` for the top-level handle.
        """
        super().__init__(parent=parent, **kwargs)

        raw = path if path is not None else data
        if raw is None:
            raise ValueError(
                f"{type(self).__name__} requires a path; got None. "
                "Pass path=... or a path-ish positional."
            )

        P = path_class()
        self.path: "Path" = raw if isinstance(raw, P) else P.from_(raw)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(path={self.path!r})"

    # ==================================================================
    # Children — the two abstract hooks
    # ==================================================================

    @abstractmethod
    def iter_children(self, options: O) -> "Iterator[Tabular]":
        """Yield this folder's direct children.

        Each yielded child is a fully formed :class:`Tabular`
        (typically another :class:`Tabular` leaf for a file, or
        another :class:`NestedIO` for a sub-folder). Implementations
        stamp ``child.parent = self`` on the way out so consumers
        can walk back up the tree.

        Children are returned closed; the read derivation opens
        them inside a ``with`` block before pulling batches.
        """

    @abstractmethod
    def make_child(self, *, media_type: Any = None) -> "Tabular":
        """Mint a fresh child IO for a write target.

        Returns a closed :class:`Tabular` bound under :attr:`path`.
        The writer opens it inside a ``with`` block, drains batches
        in, and the on-close commit fires the durable write. The
        returned child has ``parent = self``.

        Subclasses choose the child name (``part-{N}.parquet``,
        a staging filename, a partition path) and the format —
        ``media_type`` is the optional caller hint when the format
        isn't determined by the folder layout.
        """

    # ==================================================================
    # Tabular hooks — derived from children
    # ==================================================================

    def _read_arrow_batches(self, options: O) -> Iterator[pa.RecordBatch]:
        """Chain :meth:`iter_children` into one Arrow batch stream.

        Sub-folder children recurse through their own
        :meth:`_read_arrow_batches`; leaf children are read in turn
        through :meth:`Tabular.read_arrow_batches`. Each child is
        opened and closed in its own ``with`` block — the parent
        never holds more than one open child at a time.
        """
        for child in self.iter_children(options):
            with child:
                yield from child.read_arrow_batches(options=options)

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: O,
    ) -> None:
        """Mint one child and drain *batches* into it.

        Empty batch streams write nothing — no child is minted, no
        empty file lands in the folder. Concrete subclasses with
        save-mode semantics (clear-then-write, append, upsert) wrap
        or override this hook; the base contract is "append one
        child's worth of batches."
        """
        batch_iter = iter(batches)
        first = next(batch_iter, None)
        if first is None:
            return

        child = self.make_child()
        with child:
            child.write_arrow_batches(_chain_first(first, batch_iter), options=options)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chain_first(
    first: pa.RecordBatch,
    rest: "Iterator[pa.RecordBatch]",
) -> "Iterator[pa.RecordBatch]":
    """Yield *first*, then every batch from *rest*.

    Used by :meth:`NestedIO._write_arrow_batches` to put back the
    batch we peeked off the iterator to test emptiness without
    materializing the rest.
    """
    yield first
    yield from rest