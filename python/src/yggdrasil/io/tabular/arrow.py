"""In-memory :class:`Tabular` holding Arrow record batches.

Hot path is fully in-memory: reads yield held batches as-is and
writes mutate the held batch list in place subject to
``options.mode`` (AUTO / OVERWRITE / TRUNCATE → replace, APPEND →
append, IGNORE → no-op when non-empty). Use this when you want a
:class:`Tabular` over Arrow data you already have on the driver
and don't want the IPC serialization round-trip.

Auto-spill to local Arrow IPC
-----------------------------

Same shape as :class:`yggdrasil.io.buffer.bytes_io.BytesIO`: when the
in-memory footprint crosses ``spill_bytes`` (default 128 MiB), the
holder consolidates everything (any previously-spilled table plus
the in-memory tail) into a fresh Arrow IPC *file* under
``tempfile.gettempdir()`` (file-format chosen for random access /
``read_all`` semantics) and re-attaches the result via
:func:`pyarrow.memory_map` — so reads after the spill are
zero-copy from the OS page cache. The spill file uses the same
``tmp-{start}-{end}-{seed}.arrow`` naming and TTL convention
:func:`yggdrasil.io.buffer._concurrency.cleanup_stale_spill_files`
expects, and is unlinked on :meth:`_release` when we own it.

Flip the spill threshold off with ``spill_bytes=0`` (or ``None``).
Pass an explicit ``spill_path=`` to use a caller-owned location
(unlinked-on-close stays off in that case, mirroring the BytesIO
"external spill path" branch).

What we ingest
--------------

:meth:`_ingest` accepts the shapes a real caller actually has on
hand without forcing a manual conversion to Arrow:

- :class:`pyarrow.Table` / :class:`pyarrow.RecordBatch`
- polars :class:`DataFrame` / :class:`LazyFrame` (LazyFrame
  collects on ingest — the holder is in-memory by design)
- pandas :class:`DataFrame`
- pyspark :class:`DataFrame` (driver-side via ``toArrow`` on
  Spark 4+, ``toPandas`` otherwise)
- ``list[dict]`` rows / ``dict[str, list]`` columns
- any iterable yielding the above

That keeps the most common conversion glue on this side of the
API instead of every caller writing the same five-line ``isinstance``
ladder.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, ClassVar, Iterable, Iterator, Optional, Union

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.io.tabular import Tabular
from yggdrasil.data.enums import MimeType, Mode

logger = logging.getLogger(__name__)


__all__ = ["MemoryArrowIO"]


# Anything we know how to ingest into the internal batch list.
# The annotation lists the canonical Arrow shapes plus ``Any`` so
# typecheckers don't fight users handing in a polars / pandas /
# spark frame — :meth:`_ingest` recognises those at runtime via
# module-name sniffing.
ArrowSource = Union[
    pa.RecordBatch,
    pa.Table,
    Iterable[Union[pa.RecordBatch, pa.Table]],
    Any,
    None,
]


# Default spill threshold — matches BytesIO so the two layers behave
# identically on big payloads. Override per-instance via
# ``spill_bytes=`` or per-call via ``MemoryArrowIO.spill_bytes = N``.
_DEFAULT_SPILL_BYTES = 128 * 1024 * 1024
_DEFAULT_SPILL_TTL = 86400


def _deep_copy_table(table: pa.Table) -> pa.Table:
    """Force a buffer-level copy that doesn't reference any external mmap.

    Use case: we're about to close the mmap that backs *table*'s
    buffers and rewrite the same file. Reading *table* after close
    would dereference unmapped pages and segfault, so we have to
    materialise an independent copy first. ``combine_chunks`` does
    exactly that — it walks every column and concatenates chunks
    into a fresh allocation under the default memory pool.
    """
    if table.num_rows == 0:
        # combine_chunks on a zero-row table can keep the original
        # buffer refs in some pyarrow versions; round-trip through
        # IPC-in-memory is the safe answer.
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        buf = sink.getvalue()
        with pa.ipc.open_stream(pa.BufferReader(buf)) as reader:
            return reader.read_all()
    return table.combine_chunks()


class MemoryArrowIO(Tabular[CastOptions]):
    """In-memory Arrow batch holder with auto-spill to local IPC.

    Schema is tracked separately so an empty buffer still answers
    :meth:`collect_schema` correctly when one was supplied at
    construction (or carried over from a write that was later
    overwritten).

    State machine
    -------------

    Three states the holder can be in at any time:

    1. **All in-memory.** ``_spilled_table is None``; ``_batches``
       holds every record batch. The default for small payloads.
    2. **Spilled, empty tail.** ``_spilled_table`` is the mmap-
       backed table; ``_batches`` is empty. Reads stream from the
       spilled table directly.
    3. **Spilled, non-empty tail.** Both ``_spilled_table`` and
       ``_batches`` populated. Reads concat the spilled chunk with
       the in-memory tail. Crossing the threshold again triggers a
       re-spill that consolidates the two.
    """

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    @classmethod
    def default_media_type(cls) -> Optional[MimeType]:
        # In-memory containers don't claim a wire format; returning
        # None keeps them out of the media-type registry so they never
        # win factory dispatch by accident.
        return None

    def __init__(
        self,
        data: ArrowSource = None,
        *,
        schema: Optional[pa.Schema] = None,
        spill_bytes: Optional[int] = _DEFAULT_SPILL_BYTES,
        spill_ttl: int = _DEFAULT_SPILL_TTL,
        spill_path: "Any | None" = None,
        **kwargs: Any,
    ) -> None:
        # ``**kwargs`` forwards :class:`Tabular`-shared init args
        # (``static_values``, ``media_type``, …) without listing them
        # explicitly here.
        super().__init__(**kwargs)
        self._batches: list[pa.RecordBatch] = []
        self._schema: Optional[pa.Schema] = schema

        # Spill state. ``_spill_bytes_threshold == 0`` (or None) keeps
        # the holder permanently in-memory. ``_spill_ttl`` matches the
        # BytesIO convention so the cross-process janitor reaps stale
        # spill files using the same window.
        self._spill_bytes_threshold: int = int(spill_bytes or 0)
        self._spill_ttl: int = int(spill_ttl)
        self._in_memory_bytes: int = 0

        # mmap state. ``_spilled_table`` references buffers that live
        # inside ``_spill_mmap``; we hold the mmap on the instance so
        # the OS-page-cache pages don't get unmapped while still
        # referenced.
        self._spilled_table: "pa.Table | None" = None
        self._spill_mmap: "pa.MemoryMappedFile | None" = None

        # Caller-supplied spill path acts like the BytesIO "external"
        # branch — we honor it as the spill destination but don't
        # unlink on close. Otherwise minted on demand.
        self._spill_path = None
        self._owns_spill_path = True
        if spill_path is not None:
            from yggdrasil.io.path.path import Path  # local import — Path optional in some envs.

            self._spill_path = Path.from_(spill_path)
            self._owns_spill_path = False

        if data is not None:
            self._ingest(data)

        # Tabular leaves are stateless w.r.t. Disposable — there is
        # no separate acquire phase to wait for, so just leave the
        # instance live as soon as construction returns.

    def __repr__(self) -> str:
        spill = ""
        if self._spilled_table is not None and self._spill_path is not None:
            spill = f", spill={self._spill_path!s}"
        return (
            f"MemoryArrowIO(num_batches={self._total_batches()}, "
            f"num_rows={self.num_rows}"
            f"{spill})"
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def batches(self) -> list[pa.RecordBatch]:
        """Defensive copy of every held batch — spilled + in-memory."""
        out: list[pa.RecordBatch] = []
        if self._spilled_table is not None:
            out.extend(self._spilled_table.to_batches())
        out.extend(self._batches)
        return out

    @property
    def schema(self) -> Optional[pa.Schema]:
        """Arrow schema, when known.

        Set by the first ingested batch / table, by an explicit
        constructor argument, or by :meth:`_write_arrow_batches` on
        its first write. ``None`` only when the buffer has never seen
        data and no schema was passed in.
        """
        return self._schema

    @schema.setter
    def schema(self, value: Optional[pa.Schema]) -> None:
        self._schema = value

    def is_empty(self) -> bool:
        return not self._batches and self._spilled_table is None

    @property
    def num_rows(self) -> int:
        n = sum(b.num_rows for b in self._batches)
        if self._spilled_table is not None:
            n += self._spilled_table.num_rows
        return n

    def __len__(self) -> int:
        return self.num_rows

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        if self._spilled_table is not None:
            yield from self._spilled_table.to_batches()
        yield from self._batches

    @property
    def spilled(self) -> bool:
        """Whether any data is currently mmap-backed by an IPC file."""
        return self._spilled_table is not None

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
        """Drop in-memory + spilled state and unlink the owned spill file."""
        self._batches.clear()
        self._in_memory_bytes = 0
        self._drop_spill_table()
        self._unlink_owned_spill_path()

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
        """Sweep mmap + spill file on disposal.

        :class:`Disposable` calls this from ``close()`` — the
        mmap is closed first so the OS releases the page mapping
        before the file is unlinked. Owned-path-only: a caller-
        supplied ``spill_path`` is left intact (mirrors the
        BytesIO external-spill convention).
        """
        super()._release()
        self._batches.clear()
        self._in_memory_bytes = 0
        self._drop_spill_table()
        self._unlink_owned_spill_path()

    # ------------------------------------------------------------------
    # Tabular contract — read / write hooks
    # ------------------------------------------------------------------

    def stat(self):
        return self._stats

    def _read_arrow_batches(self, options: CastOptions) -> Iterator[pa.RecordBatch]:
        # Spilled chunk first — those batches are zero-copy views into
        # the mmap. The in-memory tail follows in append order.
        if self._spilled_table is not None:
            for batch in self._spilled_table.to_batches():
                yield options.cast_arrow_tabular(batch)
        for batch in self._batches:
            yield options.cast_arrow_tabular(batch)

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
            self._drop_spill_table()
            self._unlink_owned_spill_path()
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
                "MemoryArrowIO: spill to local IPC failed; staying "
                "in-memory. The next write attempt will retry.",
            )

    def _consolidate_spill(self) -> None:
        """Merge previously-spilled + in-memory batches → fresh IPC + mmap.

        Two write modes:

        - **Owned spill path** (the default — minted under tempdir):
          mint a new file each consolidation, write to it, mmap from
          it, then unlink the old owned file once the new mmap is
          live. Failure mid-write leaves the old state intact.
        - **Caller-supplied spill path**: rewrite the same file in
          place. We have to close the existing mmap first because
          we're about to truncate the file the mmap covers — once
          torn down the read path is briefly empty, but
          ``_consolidate_spill`` runs synchronously inside the write
          path so callers don't observe the gap.
        """
        tables: list[pa.Table] = []
        if self._spilled_table is not None:
            tables.append(self._spilled_table)
        if self._batches:
            tables.append(pa.Table.from_batches(self._batches))
        if not tables:
            return
        merged = (
            tables[0] if len(tables) == 1
            else pa.concat_tables(tables, promote_options="default")
        )

        caller_owned = (
            self._spill_path is not None and not self._owns_spill_path
        )
        if caller_owned:
            target_path = self._spill_path
        else:
            from yggdrasil.io.bytes_io import _mint_spill_path

            target_path = _mint_spill_path("arrow", self._spill_ttl)

        old_mmap = self._spill_mmap
        old_owned_path = (
            self._spill_path if self._owns_spill_path else None
        )

        # When rewriting in place, the existing mmap covers the file
        # we're about to truncate — but ``merged`` still references
        # buffers backed by that mmap, so writing it after close
        # would dereference unmapped pages. Force a buffer-level copy
        # via :func:`pyarrow.Table.combine_chunks` before tearing the
        # mmap down.
        if caller_owned and old_mmap is not None:
            merged = _deep_copy_table(merged)
            try:
                old_mmap.close()
            except Exception:
                pass
            self._spill_mmap = None
            self._spilled_table = None
            old_mmap = None

        try:
            with pa.OSFile(str(target_path), "wb") as sink:
                with pa.ipc.new_file(sink, merged.schema) as writer:
                    writer.write_table(merged)
        except Exception:
            # Best-effort cleanup of the half-written file before we
            # bubble up — old owned state stays intact when it was a
            # different path.
            if not caller_owned:
                try:
                    pathlib.Path(str(target_path)).unlink(missing_ok=True)
                except Exception:
                    pass
            raise

        # Re-attach via memory_map for zero-copy reads. Both the mmap
        # and the table need to live on the instance so the OS pages
        # don't get unmapped while batches still reference them.
        new_mmap = pa.memory_map(str(target_path), "r")
        new_table = pa.ipc.open_file(new_mmap).read_all()

        if not caller_owned:
            self._spill_path = target_path
            self._owns_spill_path = True
        self._spill_mmap = new_mmap
        self._spilled_table = new_table
        self._batches.clear()
        self._in_memory_bytes = 0

        # Old mmap (when we had one and didn't already close it) plus
        # any old owned file at a different path — tear down now that
        # the new mmap is live.
        if old_mmap is not None:
            try:
                old_mmap.close()
            except Exception:
                pass
        if (
            old_owned_path is not None
            and str(old_owned_path) != str(target_path)
        ):
            try:
                pathlib.Path(str(old_owned_path)).unlink(missing_ok=True)
            except Exception:
                pass

    def _mint_or_reuse_spill_path(self) -> Any:
        """Return the path to write the consolidated IPC into.

        Caller-supplied ``spill_path=`` is honored (we just rewrite
        in place). Otherwise a fresh ``tmp-{start}-{end}-{seed}.arrow``
        name is minted under :func:`tempfile.gettempdir()` —
        time-sortable + matches the
        :func:`cleanup_stale_spill_files` regex so the cross-process
        janitor reaps stale files automatically.
        """
        if self._spill_path is not None and not self._owns_spill_path:
            return self._spill_path
        # Lazy import to avoid pulling the BytesIO module at import time.
        from yggdrasil.io.bytes_io import _mint_spill_path

        return _mint_spill_path("arrow", self._spill_ttl)

    def _drop_spill_table(self) -> None:
        """Tear down the mmap + spilled-table refs (without unlinking)."""
        self._spilled_table = None
        mmap = self._spill_mmap
        self._spill_mmap = None
        if mmap is not None:
            try:
                mmap.close()
            except Exception:
                pass

    def _unlink_owned_spill_path(self) -> None:
        """Unlink the spill file iff we minted it ourselves."""
        path = self._spill_path
        owned = self._owns_spill_path
        self._spill_path = None
        self._owns_spill_path = True
        if path is not None and owned:
            try:
                pathlib.Path(str(path)).unlink(missing_ok=True)
            except FileNotFoundError:
                pass
            except Exception:
                logger.debug(
                    "MemoryArrowIO: failed to unlink spill file %r — "
                    "leaving it for the cross-process janitor.",
                    str(path),
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _total_batches(self) -> int:
        """Best-effort count of held record batches (spilled + in-memory).

        The spilled side's batch count comes from the chunk count of
        the first column — :class:`pyarrow.Table` doesn't expose a
        public ``num_batches``, but every column shares the same
        chunking, so column-0's chunk list answers the question
        without walking the whole table.
        """
        n = len(self._batches)
        if self._spilled_table is not None and self._spilled_table.num_columns:
            n += len(self._spilled_table.column(0).chunks)
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
        if isinstance(source, pa.RecordBatch):
            self._append_batch(source)
            self._maybe_spill()
            return
        if isinstance(source, pa.Table):
            for batch in source.to_batches():
                self._append_batch(batch)
            if self._schema is None:
                self._schema = source.schema
            self._maybe_spill()
            return

        # Module-name sniffing keeps optional engine deps out of
        # the import graph — we only touch a frame's API once we've
        # confirmed it's an instance of one we know how to drain.
        module = (type(source).__module__ or "").split(".", 1)[0]
        if module == "polars":
            import polars as pl

            # LazyFrame collects here; the in-memory holder is the
            # wrong tool for streaming-laziness anyway. Reach for
            # ``scan_polars_frame`` on a different IO if you want
            # that.
            if isinstance(source, pl.LazyFrame):
                source = source.collect()
            self._ingest(source.to_arrow())
            return
        if module == "pandas":
            self._ingest(pa.Table.from_pandas(source))
            return
        if module == "pyspark":
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
                f"MemoryArrowIO can't ingest "
                f"{type(source).__module__}.{type(source).__name__}: "
                f"{source!r}. Accepted: pyarrow Table/RecordBatch, polars "
                "DataFrame/LazyFrame, pandas DataFrame, pyspark DataFrame, "
                "list[dict], dict[str, list], or an iterable of any of "
                "those."
            ) from exc
        for inner in iterator:
            self._ingest(inner)
