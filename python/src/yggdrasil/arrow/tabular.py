"""In-memory :class:`Tabular` holding Arrow record batches.

Hot path is fully in-memory: reads yield held batches as-is and
writes mutate the held batch list in place subject to
``options.mode`` (AUTO / OVERWRITE / TRUNCATE → replace, APPEND →
append, IGNORE → no-op when non-empty). Use this when you want a
:class:`Tabular` over Arrow data you already have on the driver
and don't want the IPC serialization round-trip.

Auto-spill via :class:`ArrowIPCFile`
------------------------------------

Same shape as :class:`yggdrasil.io.buffer.bytes_io.BytesIO`: when the
in-memory footprint crosses ``spill_bytes`` (default 128 MiB), the
holder spills the current in-memory tail to a fresh **part file**
inside a per-holder spill *folder* under ``tempfile.gettempdir()``.
Spills are append-only — each consolidation writes only the new
tail, so an ingest-heavy workload pays O(tail) per spill instead of
O(total). Each part is mmap'd separately, so reads remain zero-copy
from the OS page cache and the live state is just a list of
:class:`pa.Table` chunks (concat'd on demand inside the C++ runtime).

Layout::

    {spill_dir}/                             # tmp-{start}-{end}-{seed}/
        part-000000-{seed}.arrow             # Arrow IPC file (one per spill)
        part-000001-{seed}.arrow
        ...

The folder name carries the same ``tmp-{start}-{end}-{seed}``
prefix the existing janitor convention expects, so any cross-
process sweeper that reaps stale spill state finds the folder by
name. Cleanup is one :func:`shutil.rmtree` — no per-file unlink
loops, no half-deleted state on partial failure.

Writes go through
:class:`yggdrasil.io.primitive.arrow_ipc_file.ArrowIPCFile` over a
:class:`yggdrasil.io.path.local_path.LocalPath`, so the spill picks
up the same OSFile streaming, codec knob, and legacy-format toggle
the format leaf already manages. Spill compression defaults to
``None`` because the spill is throwaway local cache, where the
codec overhead would hurt re-read latency without buying anything
we'd keep. Override per-instance via ``spill_compression=`` when
on-disk size matters more than read-back speed.

Skip-when-cached: if a consolidate is requested but the in-memory
tail is empty, the call short-circuits with no I/O — the spill
state is already on disk and there's nothing new to flush.

Flip the spill threshold off with ``spill_bytes=0`` (or ``None``).
Pass an explicit ``spill_path=`` to use a caller-owned folder;
the caller's folder is left intact on :meth:`unpersist` /
:meth:`_release` (mirrors the BytesIO "external spill path"
branch — we still mint our own part files under it).

What we ingest
--------------

:meth:`_ingest` accepts the shapes a real caller actually has on
hand without forcing a manual conversion to Arrow:

- :class:`pyarrow.Table` / :class:`pyarrow.RecordBatch` /
  :class:`pyarrow.RecordBatchReader` / :class:`pyarrow.ChunkedArray`
- another :class:`Tabular` (drained as an Arrow batch stream)
- polars :class:`DataFrame` / :class:`LazyFrame` (LazyFrame
  collects on ingest — the holder is in-memory by design)
- pandas :class:`DataFrame`
- pyspark :class:`DataFrame` (driver-side via ``toArrow`` on
  Spark 4+, ``toPandas`` otherwise)
- ``list[dict]`` rows / ``dict[str, list]`` columns
- any iterable yielding the above
- multiple positional sources: ``ArrowTabular(t1, t2, t3)`` is
  equivalent to ingesting each in order.

That keeps the most common conversion glue on this side of the
API instead of every caller writing the same five-line ``isinstance``
ladder.
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Any, ClassVar, Iterable, Iterator, Optional, Union

import pyarrow as pa
from yggdrasil.data import StructField, Schema

from yggdrasil.data.options import CastOptions
from yggdrasil.io import IOStats
from yggdrasil.io.tabular import O
from yggdrasil.io.tabular.base import Tabular
from yggdrasil.enums import MimeType, Mode
from yggdrasil.pickle.serde import ObjectSerde

logger = logging.getLogger(__name__)


__all__ = ["ArrowTabular"]


# Anything we know how to ingest into the internal batch list.
# The annotation lists the canonical Arrow shapes plus ``Any`` so
# typecheckers don't fight users handing in a polars / pandas /
# spark frame — :meth:`_ingest` recognises those at runtime via
# module-name sniffing.
ArrowSource = Union[
    pa.RecordBatch,
    pa.Table,
    pa.RecordBatchReader,
    "Tabular",
    Iterable[Union[pa.RecordBatch, pa.Table]],
    Any,
    None,
]


# Default spill threshold — matches :class:`yggdrasil.io.memory.Memory`'s
# spill convention so the two layers behave identically on big payloads.
# Override per-instance via ``spill_bytes=`` or per-call via
# ``ArrowTabular.spill_bytes = N``.
_DEFAULT_SPILL_BYTES = 128 * 1024 * 1024
_DEFAULT_SPILL_TTL = 86400


def _write_spill_part(
    path: str, table: pa.Table, compression: "str | None",
) -> None:
    """Write *table* to *path* via :class:`ArrowIPCFile` over a local file.

    Routes through the format leaf so each spill part picks up the
    unified IPC write path (OSFile streaming on local holders, the
    codec / legacy-format options) instead of re-implementing the
    same :func:`pa.ipc.new_file` sequence inline.
    """
    from yggdrasil.io.path.local_path import LocalPath
    from yggdrasil.io.primitive.arrow_ipc_file import (
        ArrowIPCFile,
        ArrowIPCOptions,
    )

    holder = LocalPath(path)
    sink = ArrowIPCFile(holder=holder, owns_holder=True)
    try:
        sink.write_arrow_table(
            table,
            ArrowIPCOptions(mode=Mode.OVERWRITE, compression=compression),
        )
    finally:
        try:
            sink.close()
        except Exception:
            pass


def _mint_spill_dir(ttl_seconds: int) -> pathlib.Path:
    """Mint and create a fresh spill folder under ``tempfile.gettempdir``.

    Folder name follows the same time-sortable ``tmp-{start}-{end}-{seed}``
    convention :func:`yggdrasil.io.base._mint_spill_path` uses for spill
    files, so any cross-process janitor that reaps stale spill state finds
    the folder by name. The directory is created (``mkdir(parents=True,
    exist_ok=True)``) — caller writes part files inside.
    """
    from yggdrasil.io.base import _mint_spill_path

    # Reuse the existing path-minting helper for the time-sortable
    # ``tmp-...`` stem, then drop the ``.{ext}`` suffix to land a folder
    # name. Keeps the janitor's file-vs-folder agnostic.
    minted = _mint_spill_path("dir", ttl_seconds)
    folder = minted.with_suffix("")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


class ArrowTabular(Tabular[CastOptions]):
    """In-memory Arrow batch holder with auto-spill to local IPC.

    Schema is tracked separately so an empty buffer still answers
    :meth:`collect_schema` correctly when one was supplied at
    construction (or carried over from a write that was later
    overwritten).

    State machine
    -------------

    Three states the holder can be in at any time:

    1. **All in-memory.** ``_spilled_tables`` empty; ``_batches``
       holds every record batch. The default for small payloads.
    2. **Spilled, empty tail.** ``_spilled_tables`` holds one or
       more mmap-backed tables (one per spill part); ``_batches``
       is empty. Reads stream from the parts in order.
    3. **Spilled, non-empty tail.** Both sides populated. Reads
       concat the parts with the in-memory tail. Crossing the
       threshold again writes the tail as a *new* part file —
       previous parts stay untouched (append-only spill).
    """

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    @classmethod
    def default_media_type(cls) -> Optional[MimeType]:
        # In-memory containers don't claim a wire format; returning
        # None keeps them out of the media-type registry so they never
        # win factory dispatch by accident.
        return None

    @classmethod
    def from_arrow_batches(
        cls,
        batches: Iterable[pa.RecordBatch],
        *,
        schema: StructField = None,
        **kwargs
    ) -> "ArrowTabular":
        return cls(
            batches,
            schema=schema,
            **kwargs
        )

    def __init__(
        self,
        data: ArrowSource = None,
        *more: ArrowSource,
        schema: Optional[StructField] = None,
        spill_bytes: Optional[int] = _DEFAULT_SPILL_BYTES,
        spill_ttl: int = _DEFAULT_SPILL_TTL,
        spill_path: "Any | None" = None,
        spill_compression: "str | None" = None,
        **kwargs: Any,
    ) -> None:
        # ``**kwargs`` forwards :class:`Tabular`-shared init args
        # (``static_values``, ``media_type``, …) without listing them
        # explicitly here.
        super().__init__(**kwargs)
        self._batches: list[pa.RecordBatch] = []
        self._schema: Optional[pa.Schema] = schema.to_arrow_schema() if isinstance(schema, StructField) else schema if isinstance(schema, pa.Schema) else None
        self._schema_cache: Optional[StructField] = None if schema is None else StructField.from_arrow(schema)

        # Spill state. ``_spill_bytes_threshold == 0`` (or None) keeps
        # the holder permanently in-memory. ``_spill_ttl`` matches the
        # BytesIO convention so the cross-process janitor reaps stale
        # spill state using the same window.
        self._spill_bytes_threshold: int = int(spill_bytes or 0)
        self._spill_ttl: int = int(spill_ttl)
        self._spill_compression: "str | None" = spill_compression
        self._in_memory_bytes: int = 0

        # Append-only spill state. Each spill writes a new part file
        # under ``_spill_dir`` and appends its mmap + table to the
        # parallel lists; we hold the mmaps on the instance so the
        # OS-page-cache pages don't get unmapped while still
        # referenced by ``_spilled_tables`` buffers.
        self._spill_dir: "pathlib.Path | None" = None
        self._owns_spill_dir: bool = True
        self._spill_parts: list[pathlib.Path] = []
        self._spilled_tables: list[pa.Table] = []
        self._spill_mmaps: list[pa.MemoryMappedFile] = []
        # Monotonic part counter so a single holder that re-spills many
        # times produces lexically-sortable ``part-NNNNNN-...`` names.
        self._spill_part_seq: int = 0

        # Materialized :class:`pa.Table` view of (spilled + in-memory) —
        # populated lazily by :meth:`_read_arrow_table`, invalidated on
        # every append / spill / write. Repeated table reads against an
        # untouched holder skip the concat.
        self._table_cache: "pa.Table | None" = None

        if spill_path is not None:
            # Caller-supplied spill folder — we mint our own part files
            # inside it but don't rmtree the folder on close (mirrors
            # the BytesIO "external spill path" branch).
            spill_dir_path = pathlib.Path(str(spill_path))
            spill_dir_path.mkdir(parents=True, exist_ok=True)
            self._spill_dir = spill_dir_path
            self._owns_spill_dir = False

        if data is not None:
            self._ingest(data)
        for src in more:
            if src is not None:
                self._ingest(src)

        # Tabular leaves are stateless w.r.t. Disposable — there is
        # no separate acquire phase to wait for, so just leave the
        # instance live as soon as construction returns.

    def __repr__(self) -> str:
        spill = ""
        if self._spilled_tables and self._spill_dir is not None:
            spill = (
                f", spill_dir={self._spill_dir!s}, "
                f"parts={len(self._spill_parts)}"
            )
        return (
            f"ArrowTabular(num_batches={self._total_batches()}, "
            f"num_rows={self.num_rows}"
            f"{spill})"
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def _collect_schema(self, options: O) -> Schema:
        if options.target:
            return options.target

        if self._schema_cache is None:
            for batch in self.batches:
                self._schema_cache = StructField.from_arrow_schema(batch.schema)
                return self._schema_cache

            return StructField.empty()
        return self._schema_cache

    @property
    def batches(self) -> list[pa.RecordBatch]:
        """Defensive copy of every held batch — spilled + in-memory."""
        out: list[pa.RecordBatch] = []
        for tbl in self._spilled_tables:
            out.extend(tbl.to_batches())
        out.extend(self._batches)
        return out

    def is_empty(self) -> bool:
        return not self._batches and not self._spilled_tables

    @property
    def num_rows(self) -> int:
        n = sum(b.num_rows for b in self._batches)
        n += sum(t.num_rows for t in self._spilled_tables)
        return n

    def __len__(self) -> int:
        return self.num_rows

    def _count(self, options=None) -> int:
        if options is None:
            return self.num_rows
        return sum(
            options.cast_arrow_batch(b).num_rows for b in self._batches
        )

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        for tbl in self._spilled_tables:
            yield from tbl.to_batches()
        yield from self._batches

    @property
    def spilled(self) -> bool:
        """Whether any data is currently mmap-backed by an IPC file."""
        return bool(self._spilled_tables)

    @property
    def spill_dir(self) -> "pathlib.Path | None":
        """Folder under which spill part files are minted, or ``None``."""
        return self._spill_dir

    @property
    def spill_parts(self) -> "list[pathlib.Path]":
        """Defensive copy of the spill part file list (oldest first)."""
        return list(self._spill_parts)

    @property
    def spill_bytes(self) -> int:
        """Current spill threshold in bytes (0 disables auto-spill)."""
        return self._spill_bytes_threshold

    @spill_bytes.setter
    def spill_bytes(self, value: int) -> None:
        self._spill_bytes_threshold = int(value or 0)
        # New threshold may already be exceeded; let the next ingest
        # pick that up via _maybe_spill.
        self._maybe_spill()

    # ------------------------------------------------------------------
    # Tabular contract — cache & persist
    # ------------------------------------------------------------------

    @property
    def cached(self) -> bool:
        # Always materialised — that's the whole point of this class.
        return True

    def unpersist(self) -> None:
        """Drop in-memory + spilled state and remove the owned spill folder."""
        self._batches.clear()
        self._in_memory_bytes = 0
        self._table_cache = None
        self._drop_spilled_tables()
        self._cleanup_owned_spill_dir()

    def persist(
        self,
        engine: str = "auto",
        *,
        data: Any = None,
    ) -> "Tabular":
        # ``persist`` on a memory IO either no-ops (already cached) or
        # replaces the internal data. The engine arg is ignored — the
        # holder is always Arrow-backed.
        if data is not None:
            self.unpersist()
            self._ingest(data)
        return self

    def _release(self) -> None:
        """Sweep mmaps + spill folder on disposal.

        :class:`Disposable` calls this from ``close()`` — mmaps are
        closed first so the OS releases the page mappings before the
        folder is removed. Owned-folder-only: a caller-supplied
        ``spill_path`` is left intact (mirrors the BytesIO external-
        spill convention).
        """
        super()._release()
        self._batches.clear()
        self._in_memory_bytes = 0
        self._table_cache = None
        self._drop_spilled_tables()
        self._cleanup_owned_spill_dir()

    # ------------------------------------------------------------------
    # Tabular contract — read / write hooks
    # ------------------------------------------------------------------

    def stat(self):
        return IOStats(

        )

    def _read_arrow_batches(self, options: CastOptions) -> Iterator[pa.RecordBatch]:
        # Spilled parts first — those batches are zero-copy views into
        # the per-part mmap. The in-memory tail follows in append order.
        for tbl in self._spilled_tables:
            for batch in tbl.to_batches():
                yield options.cast_arrow_batch(batch)
        for batch in self._batches:
            yield options.cast_arrow_batch(batch)

    def _read_arrow_table(self, options: CastOptions) -> pa.Table:
        """Materialize a :class:`pa.Table` from spilled + in-memory state.

        Overrides the base ``list(_read_arrow_batches) →
        Table.from_batches`` loop with:

        * One :func:`pa.concat_tables` over (every spill part + the
          in-memory tail), skipped entirely on the single-source case.
        * A single table-level :meth:`CastOptions.cast_arrow_tabular`
          instead of one cast per batch.
        * A cached :class:`pa.Table` of the concat result; cache hits
          when no append / write has mutated state since the last read,
          and on the no-target / no-rechunk path the cached table is
          returned by reference (zero-copy).
        """
        table = self._materialize_table()
        if table is None:
            return super()._read_arrow_table(options)

        target = getattr(options, "target", None)
        if (
            target is None
            and not getattr(options, "row_size", None)
            and not getattr(options, "byte_size", None)
        ):
            # No cast or rechunk requested — hand back the cached table
            # by reference (pyarrow tables are immutable; safe to share).
            return table
        return options.cast_arrow_table(table)

    def _materialize_table(self) -> "pa.Table | None":
        """Concatenate spilled parts + in-memory tail into one Table.

        Cached on :attr:`_table_cache`; invalidated by every mutation.
        Returns ``None`` when the holder is empty.
        """
        cached = self._table_cache
        if cached is not None:
            return cached

        sources: list[pa.Table] = list(self._spilled_tables)
        if self._batches:
            sources.append(pa.Table.from_batches(self._batches))
        if not sources:
            return None

        if len(sources) == 1:
            table = sources[0]
        else:
            table = pa.concat_tables(sources, promote_options="default")
        self._table_cache = table
        return table

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: CastOptions,
    ) -> None:
        action = self._resolve_save_mode(options.mode)
        if action is Mode.IGNORE:
            return
        if action is Mode.OVERWRITE:
            self._batches.clear()
            self._in_memory_bytes = 0
            self._table_cache = None
            self._drop_spilled_tables()
            self._cleanup_owned_spill_dir()
        elif action is not Mode.APPEND:
            raise NotImplementedError(
                f"{type(self).__name__}._write_arrow_batches handles "
                f"OVERWRITE / APPEND / IGNORE; got {action!r}."
            )
        for batch in batches:
            self._append_batch(batch)
        # Single spill check after the bulk write so threshold-crossing
        # within a single write batches into one round-trip rather than
        # one-spill-per-record-batch.
        self._maybe_spill()

    def _union(self, other: "Tabular", *, mode: "Mode" = ...) -> "ArrowTabular":
        for batch in other.read_arrow_batches():
            self._append_batch(batch)
        self._maybe_spill()
        return self

    # ------------------------------------------------------------------
    # Spill machinery
    # ------------------------------------------------------------------

    def _append_batch(self, batch: pa.RecordBatch) -> None:
        """Append *batch* to the in-memory tail and update the byte counter."""
        self._batches.append(batch)
        if self._schema is None:
            self._schema = batch.schema
        # ``nbytes`` is the contiguous buffer size — matches what we
        # spend on the IPC file plus a small framing overhead. Close
        # enough for the threshold check.
        self._in_memory_bytes += batch.nbytes
        # The cached :class:`pa.Table` covers (spilled + in-memory);
        # any append to the in-memory tail invalidates it.
        self._table_cache = None

    def _maybe_spill(self) -> None:
        """Spill consolidated state to IPC + mmap if threshold crossed.

        Called after every bulk ingest / write. No-op when the
        threshold is disabled (``spill_bytes == 0``) or the in-
        memory tail hasn't crossed it yet.
        """
        threshold = self._spill_bytes_threshold
        if not threshold or self._in_memory_bytes < threshold:
            return
        if not self._batches:
            return
        try:
            self._consolidate_spill()
        except Exception:
            logger.exception(
                "ArrowTabular: spill to local IPC failed; staying "
                "in-memory. The next write attempt will retry.",
            )

    def _consolidate_spill(self) -> None:
        """Append the in-memory tail to a new IPC part file.

        Append-only design: every consolidation writes only the *new*
        batches as a single fresh ``part-NNNNNN-{seed}.arrow`` file
        inside :attr:`_spill_dir`. Previously-spilled parts stay
        untouched on disk and in :attr:`_spilled_tables` — so an
        ingest-heavy workload pays O(in-memory-tail) per spill, not
        O(total-data).

        Skip-when-cached: when the in-memory tail is empty the call
        short-circuits with no I/O — the spill state is already on
        disk and there's nothing new to flush. A best-effort
        ``LOGGER.debug`` line marks the skip so the log trail
        explains why the threshold-crossing didn't fire a write.

        Failure semantics: a half-written part file is unlinked
        before re-raising; previously-spilled state stays intact.

        The write side routes through
        :class:`yggdrasil.io.primitive.arrow_ipc_file.ArrowIPCFile`
        over a :class:`LocalPath`, so each part picks up the same
        OSFile streaming, codec knob, and legacy-format toggle the
        format leaf already manages. The read side opens a fresh
        :func:`pyarrow.memory_map` per part so the table's buffers
        can outlive any context the writer would close — the
        instance retains the per-part mmap and table for the lifetime
        of the spilled state.
        """
        if not self._batches:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "ArrowTabular: spill skipped — in-memory tail "
                    "empty (already cached on disk)",
                )
            return

        merged = pa.Table.from_batches(self._batches)

        if self._spill_dir is None:
            self._spill_dir = _mint_spill_dir(self._spill_ttl)
            self._owns_spill_dir = True

        part_index = self._spill_part_seq
        seed = os.urandom(4).hex()
        part_path = self._spill_dir / f"part-{part_index:06d}-{seed}.arrow"

        try:
            _write_spill_part(
                str(part_path), merged, self._spill_compression,
            )
        except Exception:
            # Half-written part — unlink before re-raising; the
            # previously-spilled parts and the in-memory tail stay
            # intact so the caller can retry.
            try:
                part_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise

        # Re-attach via memory_map for zero-copy reads. Both the mmap
        # and the table need to live on the instance so the OS pages
        # don't get unmapped while batches still reference them.
        part_mmap = pa.memory_map(str(part_path), "r")
        part_table = pa.ipc.open_file(part_mmap).read_all()

        self._spill_parts.append(part_path)
        self._spilled_tables.append(part_table)
        self._spill_mmaps.append(part_mmap)
        self._spill_part_seq += 1

        self._batches.clear()
        self._in_memory_bytes = 0
        self._table_cache = None

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "ArrowTabular spilled %d rows to %r (part %d)",
                part_table.num_rows, str(part_path), part_index,
            )

    def _drop_spilled_tables(self) -> None:
        """Close every per-part mmap and clear the spilled-table refs.

        Cheap to call repeatedly; no-op when nothing has been spilled.
        Files on disk are *not* removed — that's
        :meth:`_cleanup_owned_spill_dir`'s job.
        """
        self._spilled_tables.clear()
        mmaps = self._spill_mmaps
        self._spill_mmaps = []
        for mm in mmaps:
            try:
                mm.close()
            except Exception:
                pass

    def _cleanup_owned_spill_dir(self) -> None:
        """Remove the spill folder tree iff we minted it ourselves.

        Single :func:`shutil.rmtree` over the whole folder — one
        syscall sequence instead of N per-file ``unlink`` round-trips,
        and the OS handles partial-cleanup atomicity. Caller-supplied
        folders are left intact (their part files inside the folder
        too — the caller owns the directory and decides when to sweep).
        """
        import shutil

        directory = self._spill_dir
        owned = self._owns_spill_dir
        # Always reset the part / mmap tracking; the on-disk state is
        # decoupled from the in-memory refs the moment we tear it down.
        self._spill_parts = []
        self._spill_part_seq = 0
        if not owned:
            return
        self._spill_dir = None
        self._owns_spill_dir = True
        if directory is None:
            return
        try:
            shutil.rmtree(directory, ignore_errors=False)
        except FileNotFoundError:
            pass
        except Exception:
            logger.debug(
                "ArrowTabular: failed to rmtree spill folder %r — "
                "leaving it for the cross-process janitor.",
                str(directory),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _total_batches(self) -> int:
        """Best-effort count of held record batches (spilled + in-memory).

        The spilled side's batch count comes from column-0's chunk list
        per part — :class:`pyarrow.Table` doesn't expose a public
        ``num_batches``, but every column shares the same chunking, so
        column-0 answers the question without walking the whole table.
        """
        n = len(self._batches)
        for tbl in self._spilled_tables:
            if tbl.num_columns:
                n += len(tbl.column(0).chunks)
        return n

    def _resolve_save_mode(self, mode: Any) -> Mode:
        m = Mode.from_(mode, default=Mode.AUTO)
        if m in (Mode.AUTO, Mode.OVERWRITE, Mode.TRUNCATE):
            return Mode.OVERWRITE
        if m is Mode.IGNORE:
            return Mode.IGNORE if not self.is_empty() else Mode.OVERWRITE
        if m is Mode.ERROR_IF_EXISTS:
            if not self.is_empty():
                raise FileExistsError(
                    f"{type(self).__name__} write with Mode.ERROR_IF_EXISTS "
                    f"but buffer is non-empty ({self.num_rows} row(s))."
                )
            return Mode.OVERWRITE
        if m is Mode.APPEND:
            return Mode.APPEND
        raise ValueError(
            f"{type(self).__name__} does not support Mode.{m.name}; "
            f"valid: AUTO, OVERWRITE, TRUNCATE, APPEND, IGNORE, ERROR_IF_EXISTS."
        )

    def _ingest(self, source: ArrowSource) -> None:
        if source is None:
            return

        # Arrow-native shapes first — these are the hot paths and skip
        # the module-name sniff entirely.
        if isinstance(source, pa.RecordBatch):
            self._append_batch(source)
            self._maybe_spill()
            return
        if isinstance(source, pa.Table):
            if self._schema is None:
                self._schema = source.schema
            for batch in source.to_batches():
                self._append_batch(batch)
            self._maybe_spill()
            return
        if isinstance(source, pa.RecordBatchReader):
            # ``read_all`` decodes the whole reader inside the C++
            # runtime — no per-batch Python hop, and the resulting
            # table shares chunking with the upstream batches.
            self._ingest(source.read_all())
            return
        if isinstance(source, pa.ChunkedArray):
            raise TypeError(
                f"ArrowTabular can't ingest a bare pa.ChunkedArray "
                f"({source.type!r}); wrap it in a pa.Table with a "
                "column name first: pa.table({'col': chunked})."
            )

        # Another Tabular — drain it as an Arrow batch stream. Covers
        # ParquetFile, ArrowIPCFile, LazyTabular, another ArrowTabular,
        # etc. without each backend re-implementing a dispatch branch.
        if isinstance(source, Tabular):
            for batch in source.read_arrow_batches():
                self._append_batch(batch)
            self._maybe_spill()
            return

        # Engine-frame namespaces — sniffed via the canonical
        # :class:`ObjectSerde` helper so the dispatch table stays
        # consistent with the rest of the IO layer.
        namespace, _ = ObjectSerde.module_and_name(source)
        root = namespace.split(".", 1)[0] if namespace else ""
        if root == "polars":
            import polars as pl

            # LazyFrame collects here; the in-memory holder is the
            # wrong tool for streaming-laziness anyway. Reach for
            # ``scan_polars_frame`` on a different IO if you want
            # that.
            if isinstance(source, pl.LazyFrame):
                source = source.collect()
            self._ingest(source.to_arrow())
            return
        if root == "pandas":
            self._ingest(pa.Table.from_pandas(source))
            return
        if root == "pyspark":
            to_arrow = getattr(source, "toArrow", None)
            if to_arrow is not None:
                self._ingest(to_arrow())
                return
            self._ingest(pa.Table.from_pandas(source.toPandas()))
            return

        # Pure-Python row-list / column-dict shapes.
        if isinstance(source, list) and source and all(
            isinstance(r, dict) for r in source
        ):
            self._ingest(pa.Table.from_pylist(source))
            return
        if (
            isinstance(source, dict) and source
            and all(isinstance(v, (list, tuple)) for v in source.values())
        ):
            self._ingest(pa.Table.from_pydict({
                k: list(v) for k, v in source.items()
            }))
            return

        # Iterable fallback — recurse so a caller can pass a
        # generator of Tables / batches and have it streamed in.
        try:
            iterator = iter(source)
        except TypeError as exc:
            raise TypeError(
                f"ArrowTabular can't ingest "
                f"{type(source).__module__}.{type(source).__name__}: "
                f"{source!r}. Accepted: pyarrow Table / RecordBatch / "
                "RecordBatchReader, another Tabular, polars "
                "DataFrame/LazyFrame, pandas DataFrame, pyspark DataFrame, "
                "list[dict], dict[str, list], or an iterable of any of "
                "those."
            ) from exc
        for inner in iterator:
            self._ingest(inner)
