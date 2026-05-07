"""Filesystem folder of tabular files.

:class:`FolderIO` is a :class:`Tabular` over a directory whose
entries are tabular files (parquet, csv, arrow IPC, ndjson, …) and /
or sub-directories. The class has no byte buffer of its own — its
state is the bound :attr:`path` plus the children walk.

Reads
-----

:meth:`iter_children` walks :attr:`path` and yields one child per
non-private entry:

* Files resolve through :class:`MediaType.from_` (extension first,
  magic-byte fallback) to a :class:`Tabular` leaf, or to a generic
  :class:`BytesIO` if the resolution fails.
* Directories come back as a fresh :class:`FolderIO` of the same
  concrete class, so a tree of folders flattens transparently into
  one batch stream.

Writes
------

:meth:`make_child` mints ``part-{epoch_ms}-{seed}.{ext}`` under
:attr:`path` and returns a closed :class:`Tabular` leaf bound to
the new path. The default writer extension is configurable on
:class:`FolderOptions` via ``child_extension``; a call site that
wants parquet writes supplies ``FolderOptions(child_extension="parquet")``.

What "private" means
--------------------

Entries whose name starts with ``.`` are skipped. That covers
``.schema`` sidecars, ``.ygg/`` directories, ``.tmp`` fragments —
the dot-prefixed metadata convention without the class needing to
enumerate them.
"""

from __future__ import annotations

import dataclasses
import os
import time
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Iterator

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.data.enums.media_type import MediaType
from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.io.tabular.base import Tabular

if TYPE_CHECKING:
    from yggdrasil.io.path import Path


__all__ = ["FolderIO", "FolderOptions"]


@dataclasses.dataclass(frozen=True, slots=True)
class FolderOptions(CastOptions):
    """:class:`CastOptions` extended with folder-write knobs."""

    #: Extension stamped onto minted child filenames. Resolves into
    #: a tabular leaf via :class:`MediaType` — ``"parquet"`` /
    #: ``"csv"`` / ``"arrow"`` / ``"ndjson"`` are the obvious choices.
    child_extension: str = "parquet"


class FolderIO(Tabular[FolderOptions]):
    """:class:`Tabular` over a directory of tabular files."""

    mime_type: ClassVar[MimeTypes] = MimeTypes.FOLDER

    __slots__ = ("path",)

    @classmethod
    def options_class(cls):
        return FolderOptions

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
        when both are supplied. ``parent`` rides through to the
        :class:`Tabular` slot — set by the enclosing folder when
        it yields this one as a child.
        """
        super().__init__(parent=parent, **kwargs)

        raw = path if path is not None else data
        if raw is None:
            raise ValueError(
                f"{type(self).__name__} requires a path; got None. "
                "Pass path=... or a path-ish positional."
            )

        from yggdrasil.io.path.path import Path as _Path
        self.path: "Path" = raw if isinstance(raw, _Path) else _Path.from_(raw)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(path={self.path!r})"

    # ==================================================================
    # Context-manager protocol — folder leaves are stateless w.r.t.
    # open/close. Provide a no-op ``with`` block so call sites that
    # do ``with cache:`` (e.g. the session lookup helper) work
    # against either a BytesIO (real Disposable) or a folder.
    # ==================================================================

    def __enter__(self) -> "FolderIO":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    # ==================================================================
    # Children — read
    # ==================================================================

    def iter_children(self) -> "Iterator[Tabular]":
        """Yield every non-private direct entry of :attr:`path`.

        Sub-directories come back as a fresh :class:`FolderIO`. File
        entries route through :class:`MediaType.from_` (extension
        first, magic-byte fallback) to a registered :class:`Tabular`
        leaf — :class:`ParquetIO` for ``.parquet``,
        :class:`ArrowIPCIO` for ``.arrow``, etc. Files that don't
        resolve fall back to a plain :class:`BytesIO`, which is
        useful for the children-surface walk but raises on the
        Tabular hooks (so they're transparently skipped by
        :meth:`_read_arrow_batches`).

        A missing folder yields nothing — no error. A stat failure
        mid-listing (race with a delete) silently skips the entry
        rather than aborting the whole walk.
        """
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
                yield self.adopt_child(type(self)(path=entry))
                continue

            child = self._leaf_for(entry)
            if child is None:
                continue
            yield self.adopt_child(child)

    def _leaf_for(self, entry: "Path") -> "Tabular | None":
        """Resolve a file entry to a :class:`Tabular` leaf.

        Returns ``None`` when the entry doesn't have a registered
        media type — the caller skips it. This is the contract
        :meth:`_read_arrow_batches` relies on to ignore non-tabular
        siblings without forcing the user to clean the directory.
        """
        # Side-effect import: ensures every primitive leaf (parquet /
        # csv / arrow / ndjson / json / xlsx) has registered itself
        # in the Tabular registry, so ``class_for_media_type`` can
        # actually find them.
        import yggdrasil.io.primitive  # noqa: F401

        try:
            mt = MediaType.from_(entry.url, default=None)
        except Exception:
            mt = None
        if mt is None:
            return None
        try:
            cls = Tabular.class_for_media_type(mt, default=None)
        except Exception:
            cls = None
        if cls is None:
            return None
        return cls(holder=entry, owns_holder=False)

    # ==================================================================
    # Children — write
    # ==================================================================

    def make_child(
        self, *, options: FolderOptions | None = None,
    ) -> "Tabular":
        """Mint a fresh tabular leaf bound to a fresh path under :attr:`path`.

        Filename shape: ``part-{epoch_ms}-{seed}.{ext}`` where ``ext``
        comes from ``options.child_extension``. The millisecond
        timestamp gives lexical-time ordering; the 8-byte seed makes
        within-millisecond collisions effectively impossible.

        Returns a closed leaf. Caller opens it inside a ``with``
        block to write bytes.
        """
        opts = options or FolderOptions()
        self.path.mkdir(parents=True, exist_ok=True)

        suffix = f".{opts.child_extension}" if opts.child_extension else ""
        epoch_ms = int(time.time() * 1000)
        seed = os.urandom(8).hex()
        name = f"part-{epoch_ms}-{seed}{suffix}"

        child_path = self.path / name
        leaf = self._leaf_for(child_path)
        if leaf is None:
            # Fall back to a raw BytesIO over the path; subclasses
            # with non-tabular extensions still get a working write.
            leaf = BytesIO(holder=child_path, owns_holder=False)
        return self.adopt_child(leaf)

    # ==================================================================
    # Tabular hooks — derived from children
    # ==================================================================

    def _read_arrow_batches(
        self, options: FolderOptions,
    ) -> Iterator[pa.RecordBatch]:
        """Chain :meth:`iter_children` into one Arrow batch stream.

        Sub-folders recurse through their own
        :meth:`_read_arrow_batches`; leaf children read in turn.
        """
        for child in self.iter_children():
            yield from child._read_arrow_batches(child.options_class()())

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: FolderOptions,
    ) -> None:
        """Mint one child per rechunked group and drain into it.

        OVERWRITE wipes the folder of tabular siblings before
        writing. APPEND just adds a new part file. IGNORE is a
        no-op when the folder is non-empty.

        ``options.byte_size`` (and ``options.row_size``) control the
        on-disk part shape: when either is set, the incoming batch
        stream is run through
        :func:`rechunk_arrow_batches_by_byte_size` and **one part file
        is minted per rechunked batch** — turning a stream of small
        record batches into a small number of ~``byte_size``-sized
        files instead of either one big file or one tiny file per
        batch. With both unset the legacy "one part file per write
        call" shape is preserved.
        """
        action = self._resolve_action(options.mode)

        if action is Mode.IGNORE and self._has_tabular_children():
            return
        if action is Mode.ERROR_IF_EXISTS and self._has_tabular_children():
            raise FileExistsError(
                f"{type(self).__name__} already contains tabular files; "
                f"refusing to write under mode={options.mode!r}."
            )
        if action is Mode.OVERWRITE:
            self._clear_tabular_children()

        byte_size = getattr(options, "byte_size", None) or 0
        row_size = getattr(options, "row_size", None) or 0

        if byte_size > 0 or row_size > 0:
            from yggdrasil.arrow.cast import rechunk_arrow_batches_by_byte_size

            rechunked = rechunk_arrow_batches_by_byte_size(
                batches,
                byte_size=byte_size or None,
                row_size=row_size or None,
            )
            for batch in rechunked:
                if batch.num_rows == 0:
                    continue
                child = self.make_child(options=options)
                child.write_arrow_batches(
                    [batch], options=child.options_class()(),
                )
            return

        batch_iter = iter(batches)
        first = next(batch_iter, None)
        if first is None:
            return

        child = self.make_child(options=options)
        child.write_arrow_batches(
            _chain_first(first, batch_iter),
            options=child.options_class()(),
        )

    def _has_tabular_children(self) -> bool:
        for _ in self.iter_children():
            return True
        return False

    def _clear_tabular_children(self) -> None:
        if not self.path.exists():
            return
        for entry in self.path.iterdir():
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    continue
            except Exception:
                continue
            try:
                entry.unlink(missing_ok=True)
            except Exception:
                pass

    # ==================================================================
    # Compaction — bin-pack small parts towards a target byte size
    # ==================================================================

    #: Default ``±`` tolerance band around *byte_size* for the
    #: "already close enough to the target" check. A 25 % cushion is
    #: wide enough that a Parquet file written from a slightly
    #: smaller-than-target Arrow table doesn't get rewritten the next
    #: pass (saving the read+write round-trip), and tight enough that
    #: a 50 %-of-target file still gets folded into a peer.
    OPTIMIZE_TOLERANCE: "ClassVar[float]" = 0.25

    def optimize(
        self,
        byte_size: "int | None" = None,
        *,
        target_extension: str = "parquet",
        tolerance: float = OPTIMIZE_TOLERANCE,
        **kwargs: Any,
    ) -> int:
        """Compact small part files into ``byte_size``-shaped bundles.

        Walks the tree under :attr:`path` and at every directory that
        holds part files, groups them by combined size and rewrites
        each group as a single fresh part. Two flavors of the pass:

        - ``byte_size=None`` (the default and the shape the local-cache
          compaction loop in :class:`Session` calls with) — collapses
          every directory with more than one part into a single file.
        - ``byte_size=N`` — first-fit-decreasing bin pack into bins of
          capacity ``N`` bytes. Parts whose size is within
          ``±tolerance`` of *N* (or already larger) are skipped: they
          are already "close enough" and rewriting them would just
          burn IO.

        Returns the number of new part files created. Idempotent: a
        second call on a tree that's already at target leaves
        nothing to do and returns ``0``.
        """
        if not self.path.exists():
            return 0
        return self._optimize_walk(
            self.path,
            byte_size=byte_size,
            target_extension=target_extension,
            tolerance=tolerance,
        )

    def _optimize_walk(
        self,
        directory: "Path",
        *,
        byte_size: "int | None",
        target_extension: str,
        tolerance: float,
    ) -> int:
        """Recurse into *directory*, compacting part files at each level.

        Sub-directories recurse first so the inner leaves drop down to
        a single big file before the outer level looks at its own
        children — keeps the bin-pack input stable when a folder
        carries both nested and flat content (rare, but the cache's
        Hive layout flattens this naturally so the cost is zero
        there).
        """
        try:
            entries = list(directory.iterdir())
        except FileNotFoundError:
            return 0

        subdirs: "list[Path]" = []
        files: "list[Path]" = []
        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    subdirs.append(entry)
                else:
                    files.append(entry)
            except Exception:
                continue

        compacted = 0
        for sub in subdirs:
            compacted += self._optimize_walk(
                sub,
                byte_size=byte_size,
                target_extension=target_extension,
                tolerance=tolerance,
            )

        parts = [f for f in files if f.name.startswith("part-")]
        compacted += self._compact_parts(
            directory,
            parts,
            byte_size=byte_size,
            target_extension=target_extension,
            tolerance=tolerance,
        )
        return compacted

    def _compact_parts(
        self,
        directory: "Path",
        parts: "list[Path]",
        *,
        byte_size: "int | None",
        target_extension: str,
        tolerance: float,
    ) -> int:
        """Group *parts* by size and rewrite each group as one file."""
        if len(parts) < 2:
            return 0

        groups: "list[list[Path]]"
        if byte_size is None:
            # No size knob — keep the legacy "everything into one"
            # shape the cache compaction loop expects.
            groups = [list(parts)]
        else:
            sized: "list[tuple[int, Path]]" = []
            for p in parts:
                try:
                    size = int(p.size)
                except Exception:
                    continue
                # Already at target (within ``±tolerance``) or already
                # larger — leave it alone. Splitting an oversized part
                # is a different operation and out of scope here.
                if size >= byte_size * (1.0 - tolerance):
                    continue
                sized.append((size, p))

            if len(sized) < 2:
                return 0

            sized.sort(key=lambda t: t[0], reverse=True)
            groups = []
            bin_sizes: "list[int]" = []
            for size, path in sized:
                placed = False
                for idx, current in enumerate(bin_sizes):
                    if current + size <= byte_size:
                        groups[idx].append(path)
                        bin_sizes[idx] = current + size
                        placed = True
                        break
                if not placed:
                    groups.append([path])
                    bin_sizes.append(size)

        compacted = 0
        leaf_folder = FolderIO(path=directory)
        write_options = FolderOptions(
            mode=Mode.APPEND, child_extension=target_extension,
        )
        for group in groups:
            if len(group) < 2:
                continue
            tables: "list[pa.Table]" = []
            for f in group:
                leaf = leaf_folder._leaf_for(f)
                if leaf is None:
                    continue
                try:
                    tables.append(leaf.read_arrow_table())
                except Exception:
                    # Unreadable part — leave it on disk; another
                    # writer might still be flushing it.
                    tables = []
                    break
            if not tables:
                continue

            try:
                merged = pa.concat_tables(tables, promote_options="default")
            except TypeError:
                # pyarrow < 14 had no ``promote_options`` kwarg.
                merged = pa.concat_tables(tables, promote=True)

            # Write the merged table first; only after the new part
            # is on disk do we drop the originals. A failed write
            # leaves the source files intact.
            leaf_folder.write_arrow_table(merged, options=write_options)
            for f in group:
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass
            compacted += 1
        return compacted

    def _resolve_action(self, mode: Mode) -> Mode:
        if mode is Mode.AUTO or mode is Mode.OVERWRITE or mode is Mode.TRUNCATE:
            return Mode.OVERWRITE
        if mode is Mode.APPEND:
            return Mode.APPEND
        if mode is Mode.IGNORE:
            return Mode.IGNORE
        if mode is Mode.ERROR_IF_EXISTS:
            return Mode.ERROR_IF_EXISTS
        if mode is Mode.UPSERT or mode is Mode.MERGE:
            return Mode.APPEND
        return Mode.OVERWRITE


def _chain_first(
    first: pa.RecordBatch,
    rest: "Iterator[pa.RecordBatch]",
) -> "Iterator[pa.RecordBatch]":
    yield first
    yield from rest
