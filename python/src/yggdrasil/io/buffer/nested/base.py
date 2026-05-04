"""Folder-oriented :class:`TabularIO` aggregations whose data is the
union of their *children*.

Where :class:`PrimitiveIO` is the multi-inheritance fold of
:class:`BytesIO + TabularIO` for single-buffer leaves (Parquet, IPC,
CSV, ...), :class:`NestedIO` is the parent-side abstraction for
folder-shaped sources whose data is composed of multiple autonomous
sub-IOs.

What changed
------------

The previous version exposed a :class:`Fragment` indirection over its
child files: ``iter_fragments()`` yielded location descriptors with an
``io`` field stapled on. That layered an extra metadata object between
the parent and the bytes that did very little work — every consumer
went straight to ``frag.io`` anyway, and the parent had to keep two
things in sync (the descriptor and the IO it wrapped).

The current shape drops fragments. A :class:`NestedIO` is just a
container of *children*, where every child is itself a fully formed
:class:`TabularIO` (or pure :class:`BytesIO` for opaque files): an
autonomous root IO, openable on its own, readable on its own, with
its own path. Each child carries a ``parent`` back-pointer to the
:class:`NestedIO` that yielded it so callers can walk back up the
tree. The parent's read/write derivations chain children directly.

The contract
------------

A subclass implements:

- :meth:`options_class`     — :class:`NestedOptions` subtype it
                              consumes. Default :class:`NestedOptions`.
- :meth:`iter_children`     — yield direct children. Each yielded
                              child is a :class:`TabularIO`
                              (typically :class:`PrimitiveIO` for
                              files or another :class:`NestedIO` for
                              sub-folders) or a :class:`BytesIO` for
                              opaque blobs. The base setter stamps
                              ``child.parent = self`` on the way out.
- :meth:`make_child`        — mint a fresh child IO bound under
                              :attr:`path` for a given name and
                              media type. Used by the writer.

Storage model
-------------

A :class:`NestedIO` holds a single :class:`Path` (the folder root).
There is no buffer, no codec, no spill — the unit of storage is the
folder, and compression is the per-child format's responsibility.

Save mode semantics
-------------------

- :attr:`Mode.OVERWRITE` — clear non-ignored children, then write
  fresh. The folder itself is never deleted (would race with
  concurrent readers); only its contents.
- :attr:`Mode.APPEND`    — mint a new child alongside existing
  ones. Always native at the base.
- :attr:`Mode.UPSERT`    — generic read-existing/merge/overwrite
  helper. Subclasses with cheaper merge semantics override.
"""

from __future__ import annotations

import dataclasses
import re
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Iterable,
    Iterator,
    TypeVar,
)

import pyarrow as pa

from yggdrasil.arrow.cast import any_to_arrow_table
from yggdrasil.data.schema import Schema
from yggdrasil.disposable import Disposable
from yggdrasil.io.enums import MimeType, Mode
from yggdrasil.io.fs import Path
from yggdrasil.data.options import CastOptions
from yggdrasil.io.buffer.base import TabularIO
from yggdrasil.lazy_imports import path_class

if TYPE_CHECKING:
    from yggdrasil.io.buffer.bytes_io import BytesIO


__all__ = ["NestedIO", "NestedOptions"]


# Time-sortable staging layout: ``<prefix>-<start>-<end>-<seed>(.ext)*``.
# ``_is_ignored_path`` uses this to skip in-flight staging files so
# parallel readers never see half-finalized writes. Mirror of
# ``yggdrasil.io.fs.path._STAGING_TMP_RE``.
_STAGING_TMP_RE: "re.Pattern[str]" = re.compile(
    r"-(\d+)-(\d+)-[0-9a-f]+(?:\.[^/]+)?$"
)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class NestedOptions(CastOptions):
    """Cast options extended with folder-write knobs.

    Inherits everything from :class:`CastOptions` (mode, row_size,
    byte_size, schema-cast hooks, ``recursive``,
    ``children_predicate``).  ``NestedOptions`` adds only knobs the
    folder writer needs — sub-IO discovery is filtered uniformly
    via :attr:`CastOptions.children_predicate` plus the shared
    :func:`yggdrasil.io.buffer.base.matches_children_predicate`
    helper.

    :param child_media_type: the :class:`MediaType` to mint child
        files as on write. ``None`` (default) means "infer from the
        folder's child convention" — the concrete subclass decides
        (e.g. :class:`FolderIO` falls back to a class-level default).
    :param child_row_size: row count per child file on write. ``0``
        or ``None`` means "one child file per write call" — the
        whole batch iterator is drained into a single staging
        file. Positive values cause the writer to roll over to a
        new staging file every ``child_row_size`` rows.
    :param child_byte_size: same as ``child_row_size`` but in
        approximate bytes. Mutually exclusive with
        ``child_row_size`` (row threshold wins if both set).
    """

    child_media_type: Any = None
    child_row_size: int = 0
    child_byte_size: int = 0


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

O = TypeVar("O", bound=NestedOptions)


# ---------------------------------------------------------------------------
# NestedIO
# ---------------------------------------------------------------------------


class NestedIO(TabularIO[O], ABC):
    """Folder-oriented :class:`TabularIO`. Children, not buffers.

    Pairs with :class:`PrimitiveIO` as the two storage flavors of
    :class:`TabularIO`. Holds a single :class:`Path` (the folder
    root); reads enumerate :meth:`iter_children`, writes mint child
    IOs via :meth:`make_child`.

    Each child yielded by :meth:`iter_children` is itself a complete
    autonomous IO (a :class:`PrimitiveIO` for a file, another
    :class:`NestedIO` for a sub-folder, or a :class:`BytesIO` for an
    opaque blob). Children carry ``parent`` set to ``self`` so a
    consumer can walk back up the tree.
    """

    # Used by :meth:`TabularIO.__new__` to short-circuit dispatch on
    # already-resolved leaves. Concrete subclasses (FolderIO,
    # DeltaIO, ...) flip this to True so a direct ``FolderIO(...)``
    # call bypasses the registry lookup.
    _FINAL_TABULAR_IO: ClassVar[bool] = False

    # ------------------------------------------------------------------
    # Registry hook
    # ------------------------------------------------------------------

    @classmethod
    def default_mime_type(cls) -> "MimeType | None":
        """Don't claim any mime type at the abstract layer."""
        return None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __new__(
        cls,
        data: Any = None,
        *args: Any,
        **kwargs: Any,
    ):
        return TabularIO.__new__(cls, data, *args, **kwargs)

    def __init__(
        self,
        data: Any = None,
        *,
        path: Any = None,
        media_type: Any = None,
        parent: "NestedIO | None" = None,
        auto_open: bool = False,
        concurrent: bool = False,
        lock_wait: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialize from a folder path.

        ``data`` and ``path`` accept the same shape; ``path`` wins
        when both are supplied. ``parent`` records this IO's
        position in a folder-of-folders tree (set by the enclosing
        :class:`NestedIO` when it yields this one); ``None`` for
        the top-level handle.

        ``auto_open`` defaults to ``False`` — a pure ``path=``
        construction returns a closed IO. The lifecycle opens on
        first ``with`` entry or explicit :meth:`open`. This matches
        the no-surprises rule callers rely on: building a handle
        should not start probing storage.

        ``concurrent=True`` enables cross-process serialisation: an
        ``.rw.lock`` sidecar against the folder root is acquired on
        :meth:`_acquire` and released on :meth:`_release`. The lock
        is conservative — exclusive across reads and writes — because
        a folder doesn't carry a single mode the way :class:`BytesIO`
        does. Callers that need finer granularity should split read
        and write blocks with their own
        ``self.path.lock(read=...)`` / ``write=...)`` calls.

        ``lock_wait`` follows :class:`WaitingConfig` conventions
        (``None`` = wait forever, ``N`` = ``N`` seconds with backoff,
        ``WaitingConfig(...)`` for full control). Raises
        :class:`TimeoutError` once the deadline elapses.
        """
        # Common TabularIO state (cache slots, _media_type, spill
        # placeholders) — NestedIO subclasses don't use _spill_path
        # but the consistent default is harmless.
        TabularIO.__init__(self, media_type=media_type, concurrent=concurrent)
        self.parent: "NestedIO | None" = parent

        raw = path if path is not None else data
        if raw is None:
            raise ValueError(
                f"{type(self).__name__} requires a path; got None. "
                "Pass path=... or a path-ish positional."
            )
        if isinstance(raw, Path):
            self.path = raw
        else:
            self.path = path_class().from_(raw)

        # Concurrency: lock the folder root for the IO's lifetime when
        # ``concurrent=True``. Inherited from :class:`TabularIO`, but
        # the actual lock object lives here because it's keyed off
        # :attr:`path`. ``lock_wait`` is a :class:`WaitingConfig`
        # argument — see ``Path.lock``.
        self._lock_wait: Any = lock_wait
        self._path_lock = None

        if auto_open:
            Disposable.open(self)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _acquire(self) -> None:
        if not self.path.opened:
            self.path.open()
        if self.concurrent and self._path_lock is None:
            try:
                lock = self.path.lock(
                    read=True, write=True,
                    wait=self._lock_wait,
                )
            except Exception:
                lock = None
            if lock is not None:
                try:
                    lock.acquire()
                except TimeoutError:
                    raise
                except Exception:
                    lock = None
            self._path_lock = lock

    def _release(self) -> None:
        self.unpersist()
        lock = self._path_lock
        self._path_lock = None
        if lock is not None:
            try:
                lock.release()
            except Exception:
                pass

    # ``cached`` / ``persist`` / ``unpersist`` come from
    # :class:`TabularIO` — they drive the shared ``_persisted_data``
    # slot identically for every NestedIO subclass.

    # ------------------------------------------------------------------
    # Options
    # ------------------------------------------------------------------

    @classmethod
    def options_class(cls) -> type[O]:
        return NestedOptions  # type: ignore[return-value]

    # ==================================================================
    # Children surface — primary read API
    # ==================================================================

    @abstractmethod
    def _iter_children(
        self,
        options: O,
    ) -> "Iterator[TabularIO | BytesIO]":
        """Yield this folder's direct children, each as an autonomous IO.

        Implementations enumerate the folder (directory listing,
        log replay for Delta, manifest scan for Iceberg) and build
        one IO per data unit:

        - File leaves are yielded as :class:`PrimitiveIO` (or
          :class:`BytesIO` for non-tabular blobs the caller may
          want to inspect).
        - Sub-folders are yielded as :class:`NestedIO` instances —
          the read/write derivations descend into them transparently.

        Each yielded child has its ``parent`` attribute set to
        ``self`` so consumers can walk back up via the parent
        chain. Children are returned closed; the caller (or the
        derived ``_read_arrow_batches`` loop) opens them inside a
        ``with`` block.

        Public callers should use :meth:`iter_children` (inherited
        from :class:`TabularIO`) which runs :meth:`check_options`
        first.
        """

    # ==================================================================
    # Child IO factory — primary write API
    # ==================================================================

    @abstractmethod
    def make_child(
        self,
        name: str,
        *,
        media_type: Any = None,
    ) -> "BytesIO":
        """Mint a fresh tabular leaf for a write target.

        Returns a closed (un-acquired) :class:`BytesIO` subclass
        (concrete format leaf — ParquetIO, CsvIO, ZipEntryIO, …)
        bound to ``self.path / name``. The writer opens it inside
        the write loop and closes on success — the bound-path
        write-back fires on close. The returned child has
        ``parent = self``.

        :param name: child filename (no path separators), already
            including the format extension. Subclasses that use
            staging accept the staging name from
            :meth:`Path.make_staging` here and rename on success.
        :param media_type: media type for the child. ``None`` lets
            the subclass infer from extension or class default.
        """

    def _attach(self, child: Any) -> Any:
        """Stamp ``child.parent = self`` and return the child.

        Used by subclasses inside ``iter_children`` / ``make_child``
        to keep parent linkage consistent without scattering the
        same line across every implementation.
        """
        try:
            child.parent = self
        except (AttributeError, TypeError):
            # Slotted subclass without a parent slot — best effort,
            # callers that need the link must implement the slot.
            pass
        return child

    # ==================================================================
    # Mode resolution — folder-flavored counterpart of PrimitiveIO's
    # ==================================================================

    def is_empty(self) -> bool:
        """True when the folder has no non-ignored children."""
        try:
            return next(iter(self._iter_children(self._default_options())), None) is None
        except FileNotFoundError:
            return True

    def _default_options(self) -> O:
        """Build a default options instance for internal-only enumeration."""
        return self.options_class()()

    def _resolve_save_mode(self, mode: Any) -> Mode:
        """Resolve any :class:`Mode` to one a folder writer can branch on."""
        m = Mode.from_(mode, default=Mode.AUTO)

        if m in (Mode.AUTO, Mode.OVERWRITE, Mode.TRUNCATE):
            return Mode.OVERWRITE

        if m is Mode.APPEND:
            return Mode.APPEND

        if m is Mode.IGNORE:
            return Mode.IGNORE if not self.is_empty() else Mode.OVERWRITE

        if m is Mode.ERROR_IF_EXISTS:
            if not self.is_empty():
                raise FileExistsError(
                    f"{type(self).__name__} write with "
                    f"Mode.ERROR_IF_EXISTS but folder is non-empty. "
                    f"Path: {self.path!r}"
                )
            return Mode.OVERWRITE

        return m

    # ==================================================================
    # Folder mutators — used by the write path
    # ==================================================================

    def _clear_children(self) -> None:
        """Remove every non-ignored child of the folder.

        Used by OVERWRITE. Does not remove the folder itself —
        concurrent readers may hold the directory handle.
        """
        if not self.path.exists():
            return

        for child in self.path.iterdir():
            if self._is_ignored_path(child):
                continue
            try:
                child.remove(recursive=True, allow_not_found=True)
            except Exception:
                # Best-effort: if a single child can't be removed
                # (race with another writer, permissions), the
                # subsequent write will surface the failure.
                pass

    def _is_ignored_path(self, child: Path) -> bool:
        """Return True for paths that should be hidden from enumeration.

        Default rules:

        - Hide dot-prefixed entries (``.schema``, ``.ygg/``, …).
        - Hide in-flight staging files
          (``tmp-<seed>-<start>-<end>.<ext>``) so a parallel reader
          doesn't pick up a half-written file mid-stage. The
          finalize step renames staging into its final
          ``part-NNNN.<ext>`` shape; only finalized children are
          ever exposed.

        Subclasses (DeltaIO, IcebergIO) override to also hide their
        own metadata directories (``_delta_log/``, ``metadata/``).
        """
        name = child.name
        if name.startswith("."):
            return True
        if name.startswith("tmp-") and _STAGING_TMP_RE.search(name):
            return True
        return False

    # ==================================================================
    # Read derivation — chain children directly
    # ==================================================================

    def _read_arrow_batches(self, options: O) -> Iterator[pa.RecordBatch]:
        """Chain :meth:`iter_children` into a single Arrow batch stream.

        Sub-folder children (other :class:`NestedIO`) recurse
        through their own ``_read_arrow_batches``. Leaf children
        are read in turn through :meth:`TabularIO.read_arrow_batches`.
        Pure :class:`BytesIO` children with no tabular surface are
        skipped — the caller can pull bytes from them via
        :meth:`iter_children` directly.

        The cache short-circuit lives on :meth:`TabularIO.read_arrow_batches`
        — it delegates to ``self._persisted_data`` before this hook runs.
        """
        for child in self._iter_children(options):
            yield from self._read_child_batches(child, options)

    def _read_child_batches(
        self,
        child: Any,
        options: O,
    ) -> Iterator[pa.RecordBatch]:
        """Drain one child's batches.

        Pulled out so subclasses (PartitionedFolderIO, DeltaIO) can
        wrap each child's batch stream (e.g. to inject partition
        columns or apply a deletion vector) without re-implementing
        the dispatch on child type.
        """
        if isinstance(child, NestedIO):
            with child:
                yield from child._read_arrow_batches(options)
            return

        if isinstance(child, TabularIO):
            with child:
                yield from child.read_arrow_batches(options=options)
            return

        # BytesIO without a tabular surface: not readable as arrow.
        # Skip — the caller can still reach it via iter_children.

    def _collect_schema(self, options: O) -> Schema:
        """Merge per-child schemas into a single folder schema."""
        merged: Schema | None = None

        for child in self._iter_children(options):
            schema: Schema | None = None
            if isinstance(child, NestedIO):
                with child:
                    schema = child._collect_schema(options)
            elif isinstance(child, TabularIO):
                with child:
                    try:
                        schema = child.collect_schema(options=options)
                    except Exception:
                        schema = None

            if schema is None:
                continue

            if merged is None:
                merged = schema
            else:
                merged = merged.merge_with(schema, inplace=True)

        return merged if merged is not None else Schema.empty()

    # ==================================================================
    # Write derivation — child minting + dispatch on save mode
    # ==================================================================

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: O,
    ) -> None:
        """Dispatch a batch iterable into one or more child files.

        Resolves save mode first (so OVERWRITE can clear before any
        bytes hit disk), then streams batches into staging children
        per :attr:`NestedOptions.child_row_size` /
        ``child_byte_size``. On a successful drain, staging files
        are renamed to their final names.
        """
        mode = self._resolve_save_mode(options.mode)

        if mode is Mode.IGNORE:
            return

        if mode is Mode.UPSERT:
            self._arrow_upsert_via_rewrite(batches, options)
            return

        if mode is Mode.OVERWRITE:
            self._clear_children()

        self._drain_into_children(batches, options)

    def _drain_into_children(
        self,
        batches: Iterable[pa.RecordBatch],
        options: O,
    ) -> None:
        """Drain *batches* into one or more child files."""
        batch_iter = iter(batches)
        first = next(batch_iter, None)
        if first is None:
            return

        media_type = options.child_media_type or self._default_child_media_type()
        row_threshold = options.child_row_size or 0
        byte_threshold = options.child_byte_size or 0

        if row_threshold <= 0 and byte_threshold <= 0:
            self._write_one_child(
                _chain_first(first, batch_iter),
                media_type=media_type,
                options=options,
            )
            return

        for chunk in _split_batches(
            _chain_first(first, batch_iter),
            row_threshold=row_threshold,
            byte_threshold=byte_threshold,
        ):
            self._write_one_child(
                chunk,
                media_type=media_type,
                options=options,
            )

    def _write_one_child(
        self,
        batches: Iterable[pa.RecordBatch],
        *,
        media_type: Any,
        options: O,
    ) -> None:
        """Mint one staging child, drain *batches* into it, finalize."""
        staging_path = self._make_staging_path(media_type)
        staging_name = staging_path.name

        child = self.make_child(staging_name, media_type=media_type)
        try:
            with child:
                child.write_arrow_batches(batches, options=options)
        except Exception:
            try:
                staging_path.remove(allow_not_found=True)
            except Exception:
                pass
            raise

        self._finalize_child(staging_path, media_type=media_type)

    # ------------------------------------------------------------------
    # Staging / finalization hooks — subclasses override the bits they
    # care about.
    # ------------------------------------------------------------------

    def _make_staging_path(self, media_type: Any) -> Path:
        return self.path.make_staging(media_type=media_type)

    def _finalize_child(self, staging_path: Path, *, media_type: Any) -> None:
        """Promote a staging file to its final name."""
        final_name = self._next_child_name(media_type=media_type)
        final_path = self.path / final_name
        staging_path.rename(final_path)

    def _next_child_name(self, *, media_type: Any) -> str:
        """Compute the next ``part-{N}.{ext}`` name."""
        ext = self._extension_for(media_type)
        prefix = "part-"
        max_idx = -1

        if self.path.exists():
            for child in self.path.iterdir():
                name = child.name
                if not name.startswith(prefix):
                    continue
                stem = name[len(prefix):]
                idx_str = stem.split(".", 1)[0]
                if not idx_str.isdigit():
                    continue
                idx = int(idx_str)
                if idx > max_idx:
                    max_idx = idx

        next_idx = max_idx + 1
        suffix = f".{ext}" if ext else ""
        return f"{prefix}{next_idx:05d}{suffix}"

    def _extension_for(self, media_type: Any) -> str:
        return Path._staging_extension(media_type)

    def _default_child_media_type(self) -> Any:
        """Media type for new children when not specified.

        Default ``None`` lets :meth:`make_child` infer from the
        child name (which is what staging gives us — extension
        baked into the staging name). Subclasses with a fixed
        format (Delta = parquet, IcebergV1 = avro) override.
        """
        return None

    # ==================================================================
    # Append / upsert via rewrite — folder-flavored
    # ==================================================================

    def _arrow_upsert_via_rewrite(self, batches: Any, options: O) -> None:
        """Read existing, merge, OVERWRITE."""
        match_by = options.match_by_names
        if not match_by:
            raise ValueError(
                f"{type(self).__name__} UPSERT requires "
                "options.match_by_names to be a non-empty sequence "
                "of column names. For 'replace everything,' use "
                "Mode.OVERWRITE instead."
            )

        existing_table = self._read_arrow_table(options.copy(read_seek=0))
        incoming_table = any_to_arrow_table(batches, options)

        merged = self.merge_upsert_tables(
            existing_table, incoming_table,
            match_by=match_by,
            update_column_names=options.update_column_names,
        )

        overwrite_options = options.copy(mode=Mode.OVERWRITE)
        row_size = getattr(overwrite_options, "row_size", None) or None
        self._write_arrow_batches(
            merged.to_batches(max_chunksize=row_size),
            overwrite_options,
        )


# ---------------------------------------------------------------------------
# Internal stream helpers
# ---------------------------------------------------------------------------


def _chain_first(
    first: pa.RecordBatch,
    rest: Iterator[pa.RecordBatch],
) -> Iterator[pa.RecordBatch]:
    """Yield *first*, then every batch from *rest*."""
    yield first
    yield from rest


def _split_batches(
    batches: Iterator[pa.RecordBatch],
    *,
    row_threshold: int,
    byte_threshold: int,
) -> Iterator[Iterator[pa.RecordBatch]]:
    """Split a batch iterator into chunks by row / byte threshold.

    ``row_threshold`` wins when both are set. With a row threshold,
    incoming batches are sliced so each emitted chunk holds exactly
    ``row_threshold`` rows (the final chunk may be shorter). Without
    a row threshold, byte accounting accumulates whole batches.
    """

    def _size_bytes(batch: pa.RecordBatch) -> int:
        try:
            return int(batch.nbytes)
        except Exception:
            return 0

    if row_threshold > 0:
        pending: list[pa.RecordBatch] = []
        pending_rows = 0
        for batch in batches:
            offset = 0
            remaining = batch.num_rows
            while remaining > 0:
                take = min(remaining, row_threshold - pending_rows)
                slice_ = batch.slice(offset, take)
                pending.append(slice_)
                pending_rows += take
                offset += take
                remaining -= take
                if pending_rows >= row_threshold:
                    yield iter(pending)
                    pending = []
                    pending_rows = 0
        if pending:
            yield iter(pending)
        return

    pending = []
    nbytes = 0

    for batch in batches:
        pending.append(batch)
        if byte_threshold > 0:
            nbytes += _size_bytes(batch)

        if 0 < byte_threshold <= nbytes:
            yield iter(pending)
            pending = []
            nbytes = 0

    if pending:
        yield iter(pending)
