"""Generic folder of tabular files.

A :class:`FolderIO` is a :class:`Tabular` over a directory whose
entries are children of a single declared :attr:`child_class`.
Reading walks the directory and yields one child per entry; writing
mints a fresh child file under the folder. Sub-directories show up
as :class:`FolderIO` instances of the same class, so reads recurse
transparently into trees.

The whole class is two attributes (:attr:`child_class`,
:attr:`child_extension`) and two methods (:meth:`iter_children`,
:meth:`make_child`). The :class:`Tabular` batch hooks chain those
into batch streams. Concerns that aren't "yield children / mint a
child" — partition routing, staging-and-rename, schema sidecars,
save modes, upsert — live in subclasses.

What "private" means
--------------------

Entries whose name starts with ``.`` are skipped during enumeration.
That covers ``.schema`` sidecars, ``.ygg/`` directories, ``.tmp``
fragments, and the rest of the dot-prefixed metadata convention
without the class needing to enumerate them.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Iterator

import pyarrow as pa
from yggdrasil.data.options import CastOptions
from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.io.tabular import Tabular
from yggdrasil.lazy_imports import path_class

if TYPE_CHECKING:
    from yggdrasil.io.path import Path


__all__ = ["FolderIO"]


class FolderIO(Tabular[CastOptions]):
    """Directory of tabular files. Skip private (dot-prefixed) names.

    Subclasses pick the format by overriding :attr:`child_class` and
    :attr:`child_extension`. The child class must implement the
    :class:`Tabular` batch hooks against bytes (Parquet, CSV, IPC
    leaves do this); the default :class:`BytesIO` itself raises when
    asked for batches, so a plain :class:`FolderIO` reads and writes
    bytes only — useful as a base for typed subclasses.
    """

    __slots__ = ("path",)

    #: Concrete child class minted for both reading and writing.
    #: Subclasses override (e.g. :class:`ParquetFolderIO` sets this
    #: to :class:`ParquetIO`). Plain :class:`BytesIO` reads / writes
    #: bytes only — calling the row-oriented surface raises
    #: :class:`NotImplementedError` at the leaf level.
    child_class: ClassVar[type[BytesIO]] = BytesIO

    #: Extension stamped onto minted child filenames. Empty string
    #: means no extension. Subclasses override (e.g. ``"parquet"``).
    #: Read-side enumeration ignores this — every non-private file
    #: is yielded regardless of extension.
    child_extension: ClassVar[str] = ""

    # ==================================================================
    # Construction
    # ==================================================================

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
        when both are supplied. ``parent`` rides through to
        :class:`Tabular`'s slot — set by an enclosing folder when
        it yields this one as a child, ``None`` for the top-level
        handle.
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
    # Children — read
    # ==================================================================

    def iter_children(self, options: CastOptions) -> "Iterator[Tabular]":
        """Yield every non-private direct entry of :attr:`path`.

        Sub-directories come back as a fresh :class:`FolderIO` of
        the same concrete class, so a tree of folders flattens into
        one batch stream when read through
        :meth:`Tabular.read_arrow_batches`. File entries come back
        as :attr:`child_class` instances bound to the entry's path.
        Each child gets ``parent = self`` stamped via
        :meth:`adopt_child` on its way out.

        A missing folder yields nothing — no error, no surprise. A
        stat failure mid-listing on a remote backend (race with a
        delete) silently skips the entry rather than aborting the
        whole walk.
        """
        del options
        if not self.path.exists():
            return

        for entry in self.path.iterdir():
            if entry.name.startswith("."):
                continue

            try:
                is_dir = entry.is_dir()
            except Exception:
                continue

            if is_dir:
                child: "Tabular" = type(self)(path=entry)
            else:
                child = self.child_class(holder=entry)

            yield self.adopt_child(child)

    # ==================================================================
    # Children — write
    # ==================================================================

    def make_child(self, *, media_type: Any = None) -> "Tabular":
        """Mint a fresh child file under :attr:`path`.

        Filename shape: ``part-{epoch_ms}-{seed}{ext}``. The
        millisecond timestamp gives lexical-time ordering for
        free; the 8-byte seed makes within-millisecond collisions
        effectively impossible without coordinating writers. The
        parent directory is auto-created if missing.

        Returns a closed :attr:`child_class` instance bound to the
        new path, with ``parent = self``. The caller opens it
        inside a ``with`` block to actually write bytes.
        """
        del media_type  # subclasses with format dispatch use this
        self.path.mkdir(parents=True, exist_ok=True)

        ext = self.child_extension
        suffix = f".{ext}" if ext else ""
        epoch_ms = int(time.time() * 1000)
        seed = os.urandom(8).hex()
        name = f"part-{epoch_ms}-{seed}{suffix}"

        child_path = self.path / name
        child = self.child_class(holder=child_path)
        return self.adopt_child(child)

    # ==================================================================
    # Tabular hooks — derived from children
    # ==================================================================

    def _read_arrow_batches(
        self, options: CastOptions,
    ) -> Iterator[pa.RecordBatch]:
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
        options: CastOptions,
    ) -> None:
        """Mint one child and drain *batches* into it.

        Empty batch streams write nothing — no child is minted, no
        empty file lands in the folder. Subclasses with save-mode
        semantics (clear-then-write, append, upsert) wrap or
        override this hook; the base contract is "append one
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

    Used by :meth:`FolderIO._write_arrow_batches` to put back the
    batch we peeked off the iterator to test emptiness without
    materializing the rest.
    """
    yield first
    yield from rest