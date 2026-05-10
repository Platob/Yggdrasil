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
:class:`FolderOptions` via ``child_extension``; the default is
``"arrow"`` (Arrow IPC) — single-pass column-oriented encoding,
no row-group footer to rewrite on append, and a write-side that
matches the in-memory batch shape almost 1:1, so it's the
cheapest format to land a stream of small batches into. Callers
that want parquet supply ``FolderOptions(child_extension="parquet")``.

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
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Iterator

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.data.enums.media_type import MediaType, MediaTypes
from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.io.tabular.base import Tabular

if TYPE_CHECKING:
    from yggdrasil.io.path import Path


__all__ = ["FolderIO", "FolderOptions"]


@dataclasses.dataclass(frozen=True, slots=True)
class FolderOptions(CastOptions):
    """:class:`CastOptions` extended with folder-write knobs."""

    #: Media type of newly minted child files. Drives both the
    #: filename extension and the :class:`Tabular` leaf class
    #: (``ParquetIO`` / ``ArrowIPCIO`` / ``CsvIO`` / …) the folder
    #: dispatches to. Defaults to Arrow IPC — matches the in-memory
    #: batch shape, no row-group footer to rewrite, cheapest format
    #: to land a stream of small batches into. Pass
    #: ``MediaTypes.PARQUET`` (or any registered :class:`MediaType`)
    #: to override; a bare string (``"parquet"``) / extension
    #: (``"csv"``) / mime value (``"application/json"``) is coerced
    #: through :meth:`MediaType.from_`.
    child_media_type: MediaType = MediaTypes.ARROW_IPC

    def __post_init__(self) -> None:
        CastOptions.__post_init__(self)
        coerced = MediaType.from_(self.child_media_type, default=None)
        if coerced is None:
            raise ValueError(
                f"FolderOptions.child_media_type must coerce to a MediaType; "
                f"got {self.child_media_type!r}. Pass one of "
                f"MediaTypes.ARROW_IPC / .PARQUET / a registered extension "
                f"string, or a MediaType instance."
            )
        if coerced is not self.child_media_type:
            object.__setattr__(self, "child_media_type", coerced)


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
        tabular_parent: "Tabular | None" = None,
        static_values: "Mapping[str, Any] | None" = None,
        **kwargs: Any,
    ) -> None:
        """Bind to a folder path. No I/O.

        ``data`` and ``path`` accept the same shape; ``path`` wins
        when both are supplied. ``tabular_parent`` rides through to
        the :class:`Tabular` slot — set by the enclosing folder when
        it yields this one as a child. ``static_values`` rides
        through too: an aggregator (e.g. :class:`YGGFolderIO`)
        minting a per-partition leaf seeds the kv here so every
        descendant inherits the partition constants via the
        :attr:`Tabular.static_values` parent chain — no extra
        per-batch stamping needed to assert the column equality.
        """
        super().__init__(
            tabular_parent=tabular_parent,
            static_values=static_values,
            **kwargs,
        )

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
        is ``options.child_media_type.full_extension``. The
        millisecond timestamp gives lexical-time ordering; a 2-byte
        seed (~65k-value space) breaks ties between writes that land
        in the same millisecond.

        The :class:`Tabular` leaf is dispatched directly from the
        media type via :meth:`Tabular.class_for_media_type`, so the
        write path doesn't go through the path-extension reverse-
        lookup. A media type with no registered leaf falls back to
        a raw :class:`BytesIO` so non-tabular extensions still get a
        working write.

        Returns a closed leaf. Caller opens it inside a ``with``
        block to write bytes.
        """
        opts = options or FolderOptions()
        self.path.mkdir(parents=True, exist_ok=True)

        ext = opts.child_media_type.full_extension
        suffix = f".{ext}" if ext else ""
        epoch_ms = int(time.time() * 1000)
        seed = os.urandom(2).hex()
        name = f"part-{epoch_ms}-{seed}{suffix}"

        child_path = self.path / name
        # Side-effect import: ensures every primitive leaf has
        # registered itself in the Tabular registry before lookup.
        import yggdrasil.io.primitive  # noqa: F401
        cls = Tabular.class_for_media_type(opts.child_media_type, default=None)
        if cls is None:
            leaf: "Tabular" = BytesIO(holder=child_path, owns_holder=False)
        else:
            leaf = cls(holder=child_path, owns_holder=False)
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

        Self + per-child predicate pruning runs through
        :meth:`Tabular._should_prune_by_predicate`: when
        ``options.predicate`` is provably false against the bound
        :attr:`static_values` (own seed + inherited from
        :attr:`tabular_parent`), the whole read is skipped without
        opening the directory; per-child the same check skips
        sub-folders / leaf files whose static surface decides the
        predicate negatively. Children without a static surface fall
        through unchanged (undecidable → read), so a vanilla folder
        without partition KV behaves exactly as before.
        """
        if self._should_prune_by_predicate(options):
            return
        for child in self.iter_children():
            if child._should_prune_by_predicate(options):
                continue
            yield from child._read_arrow_batches(child.options_class()())

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: FolderOptions,
    ) -> None:
        """Mint one child per rechunked group and drain into it.

        Mode dispatch:

        - **OVERWRITE / TRUNCATE** — drop every tabular sibling first,
          then write incoming as fresh part(s).
        - **AUTO / APPEND** (the default for tabular folders) — just
          add a new part file; existing parts are untouched.
        - **UPSERT / MERGE** — only meaningful with
          ``options.match_by``; see below.
        - **IGNORE** — no-op when the folder already holds tabular
          parts; otherwise behaves as APPEND.
        - **ERROR_IF_EXISTS** — raises when the folder is non-empty.

        Merge semantics (``options.match_by`` set):

        - **APPEND** — drop incoming rows whose key tuple already
          exists on disk; write only the survivors into a new part.
          Existing parts are not rewritten.
        - **UPSERT / MERGE** — collect incoming key tuples, walk
          existing parts and keep only rows whose key is *not* in
          that set, then write the survivors plus all incoming as
          fresh part(s). Old parts are dropped at the end so a
          failed write leaves them in place.

        ``options.byte_size`` / ``options.row_size`` route the actual
        bytes-to-disk write through
        :func:`rechunk_arrow_batches`, so the
        bin-packing applies regardless of which mode picked the
        rows. Setting both knobs unset keeps the legacy
        "one part file per write call" shape.
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
            self._write_parts(batches, options)
            return

        match_by = list(options.match_by_keys or ())
        is_upsert = options.mode in (Mode.UPSERT, Mode.MERGE)

        if match_by and self._has_tabular_children():
            if is_upsert:
                self._merge_upsert(batches, match_by, options)
            else:
                self._merge_append(batches, match_by, options)
            return

        # Plain APPEND (or empty folder): mint a fresh part and drain.
        self._write_parts(batches, options)

    def _write_parts(
        self,
        batches: Iterable[pa.RecordBatch],
        options: FolderOptions,
    ) -> None:
        """Mint one or more part files and drain *batches* into them.

        Honors ``options.byte_size`` / ``options.row_size`` for
        per-part rechunking; with neither set, drains the whole
        stream into a single part.
        """
        byte_size = getattr(options, "byte_size", None) or 0
        row_size = getattr(options, "row_size", None) or 0

        if byte_size > 0 or row_size > 0:
            from yggdrasil.arrow.cast import rechunk_arrow_batches

            rechunked = rechunk_arrow_batches(
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

    # ==================================================================
    # Merge helpers — used when options.match_by is set
    # ==================================================================

    def _merge_append(
        self,
        batches: Iterable[pa.RecordBatch],
        match_by: "list[str]",
        options: FolderOptions,
    ) -> None:
        """APPEND with key-aware dedup.

        Rows whose ``match_by`` tuple already exists on disk are
        dropped from the incoming stream; survivors land in a new
        part file. Existing parts are not rewritten.
        """
        existing = self._collect_existing_keys(match_by)
        survivors = self._filter_batches_drop_keys(batches, match_by, existing)
        self._write_parts(survivors, options)

    def _merge_upsert(
        self,
        batches: Iterable[pa.RecordBatch],
        match_by: "list[str]",
        options: FolderOptions,
    ) -> None:
        """UPSERT / MERGE with key-aware rewrite.

        Drains incoming into memory once to capture the set of
        incoming keys, walks existing parts, drops every row whose
        key matches an incoming key, then writes the (filtered
        existing + incoming) stream into fresh parts and unlinks the
        old ones.
        """
        incoming = list(batches)
        if not incoming:
            return

        incoming_keys = self._collect_keys_from_batches(incoming, match_by)
        survivors_existing = self._iter_existing_filtered(match_by, incoming_keys)

        # Snapshot old part files before we touch anything new — we
        # only delete them after the rewrite has succeeded.
        old_files = self._tabular_files()

        merged_iter = _chain_iter(survivors_existing, iter(incoming))
        self._write_parts(merged_iter, options)

        for f in old_files:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass

    def _tabular_files(self) -> "list[Path]":
        if not self.path.exists():
            return []
        out: "list[Path]" = []
        for entry in self.path.iterdir():
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    continue
            except Exception:
                continue
            out.append(entry)
        return out

    def _collect_existing_keys(
        self, match_by: "list[str]",
    ) -> "set[tuple]":
        keys: "set[tuple]" = set()
        for child in self.iter_children():
            if isinstance(child, FolderIO):
                continue
            try:
                for batch in child._read_arrow_batches(child.options_class()()):
                    self._extend_keys_from_batch(keys, batch, match_by)
            except Exception:
                continue
        return keys

    @staticmethod
    def _collect_keys_from_batches(
        batches: "Iterable[pa.RecordBatch]", match_by: "list[str]",
    ) -> "set[tuple]":
        keys: "set[tuple]" = set()
        for batch in batches:
            FolderIO._extend_keys_from_batch(keys, batch, match_by)
        return keys

    @staticmethod
    def _extend_keys_from_batch(
        keys: "set[tuple]",
        batch: pa.RecordBatch,
        match_by: "list[str]",
    ) -> None:
        if not all(c in batch.schema.names for c in match_by):
            return
        cols = [batch.column(c).to_pylist() for c in match_by]
        for row in zip(*cols):
            keys.add(row)

    def _filter_batches_drop_keys(
        self,
        batches: "Iterable[pa.RecordBatch]",
        match_by: "list[str]",
        drop_keys: "set[tuple]",
    ) -> "Iterator[pa.RecordBatch]":
        if not drop_keys:
            yield from batches
            return
        for batch in batches:
            yield from self._batch_filter_drop(batch, match_by, drop_keys)

    @staticmethod
    def _batch_filter_drop(
        batch: pa.RecordBatch,
        match_by: "list[str]",
        drop_keys: "set[tuple]",
    ) -> "Iterator[pa.RecordBatch]":
        if batch.num_rows == 0:
            return
        if not all(c in batch.schema.names for c in match_by):
            yield batch
            return
        cols = [batch.column(c).to_pylist() for c in match_by]
        mask = [row not in drop_keys for row in zip(*cols)]
        if all(mask):
            yield batch
            return
        if not any(mask):
            return
        keep_idx = [i for i, m in enumerate(mask) if m]
        table = pa.Table.from_batches([batch]).take(keep_idx).combine_chunks()
        for inner in table.to_batches():
            if inner.num_rows > 0:
                yield inner

    def _iter_existing_filtered(
        self,
        match_by: "list[str]",
        drop_keys: "set[tuple]",
    ) -> "Iterator[pa.RecordBatch]":
        """Walk existing leaves, yielding only rows whose key isn't in *drop_keys*."""
        for child in self.iter_children():
            if isinstance(child, FolderIO):
                continue
            try:
                stream = child._read_arrow_batches(child.options_class()())
            except Exception:
                continue
            yield from self._filter_batches_drop_keys(stream, match_by, drop_keys)

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
    # Row-level delete
    # ==================================================================

    def _delete(self, predicate: Any, options: FolderOptions) -> int:
        """Walk children, filter each leaf in isolation, rewrite survivors.

        Streams leaf-by-leaf so a single match in one part file doesn't
        trigger a folder-wide rewrite — only the leaves that actually
        hold matched rows are rewritten. Sub-folders recurse. Files
        the predicate fully drains are unlinked outright; leaves with
        a mix of survivors and matches are rewritten as a fresh part
        and the original is unlinked once the new file is on disk.

        Per-batch filtering goes through
        :meth:`Predicate.filter_arrow_batches`, so the row work runs
        in pyarrow's C++ kernels — no Python row iteration.
        """
        if not self.path.exists():
            return 0
        not_pred = ~predicate
        deleted = 0
        for child in self.iter_children():
            if isinstance(child, FolderIO):
                deleted += child._delete(predicate, child.options_class()())
                continue
            deleted += self._delete_leaf(child, not_pred, options)
        return deleted

    def _delete_leaf(
        self,
        child: "Tabular",
        not_pred: Any,
        options: FolderOptions,
    ) -> int:
        """Filter rows in *child*; rewrite as a fresh part or unlink it."""
        survivors: "list[pa.RecordBatch]" = []
        kept_rows = 0
        total_rows = 0

        def _counted() -> "Iterator[pa.RecordBatch]":
            nonlocal total_rows
            for b in child._read_arrow_batches(child.options_class()()):
                total_rows += b.num_rows
                yield b

        try:
            for kept in not_pred.filter_arrow_batches(_counted()):
                kept_rows += kept.num_rows
                survivors.append(kept)
        except Exception:
            return 0

        deleted = total_rows - kept_rows
        if deleted == 0:
            return 0

        leaf_path = getattr(child, "_holder", None)
        if survivors:
            # Mixed: write survivors first, then drop the original. A
            # failed rewrite leaves the original intact.
            self._write_parts(iter(survivors), options)
        if leaf_path is not None:
            try:
                leaf_path.unlink(missing_ok=True)
            except Exception:
                pass
        return deleted

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
        target_media_type: "MediaType | str | Any" = MediaTypes.ARROW_IPC,
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

        ``target_media_type`` (a :class:`MediaType` or anything
        :meth:`MediaType.from_` accepts) selects the format the
        rewritten parts are encoded in. Defaults to Arrow IPC.

        Returns the number of new part files created. Idempotent: a
        second call on a tree that's already at target leaves
        nothing to do and returns ``0``.
        """
        if not self.path.exists():
            return 0
        media = MediaType.from_(target_media_type, default=MediaTypes.ARROW_IPC)
        return self._optimize_walk(
            self.path,
            byte_size=byte_size,
            target_media_type=media,
            tolerance=tolerance,
        )

    def _optimize_walk(
        self,
        directory: "Path",
        *,
        byte_size: "int | None",
        target_media_type: MediaType,
        tolerance: float,
    ) -> int:
        """Recurse into *directory*, compacting part files at each level."""
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
                target_media_type=target_media_type,
                tolerance=tolerance,
            )

        parts = [f for f in files if f.name.startswith("part-")]
        compacted += self._compact_parts(
            directory,
            parts,
            byte_size=byte_size,
            target_media_type=target_media_type,
            tolerance=tolerance,
        )
        return compacted

    def _compact_parts(
        self,
        directory: "Path",
        parts: "list[Path]",
        *,
        byte_size: "int | None",
        target_media_type: MediaType,
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
            mode=Mode.APPEND, child_media_type=target_media_type,
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
        # AUTO maps to APPEND for tabular folders: each write adds a
        # fresh part file alongside the existing ones, the way the
        # response cache and any "drop another batch into the
        # partition" workflow expects. OVERWRITE / TRUNCATE stay
        # destructive and are reserved for the explicit
        # "rewrite the whole folder" call.
        if mode is Mode.OVERWRITE or mode is Mode.TRUNCATE:
            return Mode.OVERWRITE
        if mode is Mode.AUTO or mode is Mode.APPEND:
            return Mode.APPEND
        if mode is Mode.IGNORE:
            return Mode.IGNORE
        if mode is Mode.ERROR_IF_EXISTS:
            return Mode.ERROR_IF_EXISTS
        if mode is Mode.UPSERT or mode is Mode.MERGE:
            return Mode.APPEND
        return Mode.APPEND


def _chain_first(
    first: pa.RecordBatch,
    rest: "Iterator[pa.RecordBatch]",
) -> "Iterator[pa.RecordBatch]":
    yield first
    yield from rest


def _chain_iter(*iters: "Iterable[pa.RecordBatch]") -> "Iterator[pa.RecordBatch]":
    for it in iters:
        yield from it
