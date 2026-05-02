"""Arrow IPC I/O for :class:`PrimitiveIO`.

:class:`ArrowIPCIO` is the concrete leaf for Arrow IPC files
(``.arrow``, ``.feather``). The IPC file format is parquet's nearer
cousin: a header + a sequence of record batches + a footer with batch
offsets, all using the Arrow in-memory representation directly.
Reads are essentially zero-decode, which is the point — Arrow IPC
is the no-translation fast path between processes / disks / arrow
tools.

This class handles the **file** format (random-access, footer-indexed
batches). The streaming format (sequential, no footer) is a thinner
relative.

Lifecycle, codec, and Mode resolution all live on
:class:`DataIO`. This leaf only owns:

- The cached :class:`pa.ipc.RecordBatchFileReader` (footer parse is
  the read-side cost worth amortizing).
- The IPC-specific writer options.
- The native engine overrides (``pds.dataset(format="feather")``,
  ``pl.scan_ipc``, ``pl.read_ipc``) which short-circuit the generic
  shim when the buffer sits at a local path.

Native engine dispatch
----------------------

When the buffer is backed by a real path (no compression wrapper,
no in-flight target cast, non-empty), :meth:`_read_arrow_dataset`,
:meth:`_scan_polars_frame`, and :meth:`_read_polars_frame` dispatch
to the format-aware scanners. They push projection and predicate
into the IPC reader at plan time — much faster than decoding
everything then filtering in-memory through the
:meth:`_read_arrow_batches` shim.
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import TYPE_CHECKING, ClassVar, Iterable, Iterator

import pyarrow as pa
import pyarrow.ipc as ipc

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.io.enums import MimeTypes, Mode
from yggdrasil.lazy_imports import (
    polars_module,
    pyarrow_dataset_module,
)
from .base import PrimitiveIO

if TYPE_CHECKING:
    import polars as pl
    import pyarrow.dataset as pds


__all__ = ["ArrowIPCIO", "ArrowIPCOptions"]


# ---------------------------------------------------------------------------
# ArrowIPCOptions
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ArrowIPCOptions(CastOptions):
    """:class:`CastOptions` extended with IPC-specific knobs.

    All IPC fields default to pyarrow's own defaults, so an
    unparameterized :class:`ArrowIPCOptions()` produces an
    uncompressed file with default metadata version. The most common
    knob to tweak is ``compression`` — ``"lz4"`` is the standard
    "fast and small enough" choice.
    """

    # Reader knobs
    use_threads: bool = True

    # Writer knobs — map onto :class:`pa.ipc.IpcWriteOptions`.
    compression: "str | None" = None  # "lz4" | "zstd" | None
    compression_level: "int | None" = None
    write_legacy_ipc_format: bool = False

    def to_writer_options(self) -> "ipc.IpcWriteOptions":
        """Build a :class:`pa.ipc.IpcWriteOptions` from this options view."""
        return ipc.IpcWriteOptions(
            compression=self.compression,
            use_legacy_format=self.write_legacy_ipc_format,
        )


# ---------------------------------------------------------------------------
# ArrowIPCIO
# ---------------------------------------------------------------------------


class ArrowIPCIO(PrimitiveIO):
    """:class:`PrimitiveIO` for Arrow IPC **file** format.

    File-format reads use :class:`pa.ipc.RecordBatchFileReader`, which
    parses the footer on construction and exposes random access by
    batch index. Writes go through :class:`pa.ipc.RecordBatchFileWriter`.
    """

    # ==================================================================
    # Class-level config
    # ==================================================================

    @classmethod
    def default_mime_type(cls):
        """Canonical :class:`MimeType` — :data:`MimeTypes.ARROW_IPC`."""
        return MimeTypes.ARROW_IPC

    @classmethod
    def options_class(cls):
        return ArrowIPCOptions

    # Class-level switch so subclasses (notably batch-view variants)
    # can opt out of native dispatch without overriding every method.
    _NATIVE_SCANNER_OK: ClassVar[bool] = True

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    def __init__(
        self,
        *args,
        reader: "pa.ipc.RecordBatchFileReader | None" = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._reader: "pa.ipc.RecordBatchFileReader | None" = reader

    # ==================================================================
    # Cached reader — bound to ``self``, not to the codec sibling
    # ==================================================================
    #
    # The reader caches a parsed footer over the IO's bytes. With a
    # codec, those bytes are compressed, so we can't construct a
    # reader against ``self`` directly — :meth:`_read_arrow_batches`
    # routes through ``_reading_context`` which yields the
    # decompressed sibling and the reader is built ad-hoc against
    # that sibling for the duration of the iteration. The cached
    # reader on ``self`` is therefore meaningful only for the
    # no-codec path, which is the hot path.

    @property
    def reader(self) -> "ipc.RecordBatchFileReader":
        """Lazy :class:`pa.ipc.RecordBatchFileReader`, cached.

        Only used on the no-codec path; with a codec the reader is
        a transient over the decompressed sibling (see
        :meth:`_read_arrow_batches`).

        Caller must have opened the IO first — the property raises
        :class:`ValueError` on a closed handle rather than implicitly
        opening. Implicit open used to surprise callers with stale
        materialization on re-access; the explicit-open contract is
        the rework's preferred shape.
        """
        if self._reader is not None:
            return self._reader
        if self.codec is not None:
            raise RuntimeError(
                f"{type(self).__name__}.reader is not cacheable when a "
                "codec is active; route through _reading_context which "
                "yields a decompressed sibling for ad-hoc reader use."
            )
        if not self.opened:
            raise ValueError(
                f"Cannot read from closed {self!r}; call .open() first."
            )

        self.seek(0)
        source = self.arrow_io(mode="rb")
        self._reader = ipc.RecordBatchFileReader(source)
        return self._reader

    def _drop_reader(self) -> None:
        """Drop the cached reader.

        :class:`RecordBatchFileReader` doesn't expose a ``close`` —
        releasing the reference is enough; the underlying memory /
        mmap drops when GC reclaims it.
        """
        self._reader = None

    def _before_release(self) -> None:
        """Tear down the cached reader before the buffer closes.

        Called by :meth:`DataIO._release` while the buffer's bytes
        are still readable. Doing this BEFORE the buffer's own
        cleanup (fd close, spill unlink) is essential — a reader
        holding a memoryview into bytes that just got unmapped is a
        segfault waiting to happen.
        """
        self._drop_reader()
        super()._before_release()

    # ==================================================================
    # Schema — cheap via footer
    # ==================================================================

    def _collect_schema(self, options: ArrowIPCOptions) -> Schema:
        """Read the schema from the IPC footer directly.

        Empty buffer short-circuits to :meth:`Schema.empty` —
        pyarrow's IPC reader raises ``ArrowInvalid: File is too
        small: 0`` on a zero-byte file. The "fresh write target,
        read it back" flow is common enough to deserve the
        short-circuit.
        """
        if self.is_empty():
            return Schema.empty()

        with self._reading_context(options) as io:
            if io is self:
                # No codec, can use the cached reader.
                return Schema.from_arrow(self.reader.schema)
            # Codec branch — sibling has the decompressed bytes;
            # build an ad-hoc reader over its arrow_io.
            source = io.arrow_io(mode="rb")
            reader = ipc.RecordBatchFileReader(source)
            return Schema.from_arrow(reader.schema)

    # ==================================================================
    # Read path
    # ==================================================================

    def _read_arrow_batches(
        self,
        options: ArrowIPCOptions,
    ) -> Iterator[pa.RecordBatch]:
        """Yield Arrow record batches.

        File format gives us random access by batch index; we walk
        them in file order so callers see the same row order they
        wrote. ``options.row_size`` is ignored — IPC batch
        boundaries are baked in at write time.

        Empty buffer short-circuits to no batches: pyarrow raises
        on zero-byte input, and "freshly opened write target, read
        it back" is a legitimate flow that should produce an empty
        iterator rather than an exception.

        ``options.cast_arrow_tabular`` runs unconditionally per
        batch — it short-circuits internally when
        ``options.target_field is None``, so the no-cast path is
        one isinstance check per batch.
        """
        if self.is_empty():
            return

        with self._reading_context(options) as io:
            if io is self:
                # No codec — use the cached, footer-parsed reader.
                reader = self.reader
            else:
                # Codec branch — sibling holds the decompressed
                # bytes; build a transient reader over them.
                source = io.arrow_io(mode="rb")
                reader = ipc.RecordBatchFileReader(source)

            for i in range(reader.num_record_batches):
                yield options.cast_arrow_tabular(reader.get_batch(i))

    # ==================================================================
    # Write path
    # ==================================================================

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: ArrowIPCOptions,
    ) -> None:
        """Persist Arrow record batches as an IPC file.

        Save-mode dispatch:

        - **OVERWRITE / AUTO / TRUNCATE** — single
          :class:`pa.ipc.RecordBatchFileWriter` session.
        - **APPEND** — :meth:`DataIO._arrow_append_via_rewrite`:
          read existing, concat with incoming (schema union),
          recurse with OVERWRITE. The IPC file format has one
          footer for all batches; partial appends would require
          rewriting the footer anyway.
        - **UPSERT** — :meth:`DataIO._arrow_upsert_via_rewrite`:
          merge existing and incoming on
          ``options.match_by_names`` with incoming-wins-on-overlap,
          recurse with OVERWRITE.
        - **IGNORE** — skip.

        Codec round-tripping is handled transparently by
        :meth:`_writing_context`: when ``self.codec`` is set, the
        context yields a decompressed sibling for the body to write
        into and compresses back into ``self`` on successful exit.
        """
        action = self._resolve_save_mode(options.mode)
        if action is Mode.IGNORE:
            return
        if action is Mode.APPEND:
            self.seek(0)
            return self._arrow_append_via_rewrite(batches, options)
        if action is Mode.UPSERT:
            self.seek(0)
            return self._arrow_upsert_via_rewrite(batches, options)
        if action is not Mode.OVERWRITE:
            raise NotImplementedError(
                f"{type(self).__name__}._write_arrow_batches handles "
                f"OVERWRITE / APPEND / UPSERT; got resolved action "
                f"{action!r}."
            )

        iterator = iter(batches)
        first = next(iterator, None)
        if first is None:
            return

        schema = options.check_source(first).check_target(first).merged_schema.to_arrow_schema()
        first = options.cast_arrow_tabular(first)

        # Drop any cached reader BEFORE the write — keeping it alive
        # over a buffer reset would leave it pointing at bytes that
        # are about to be invalidated.
        self._drop_reader()

        lifecycle = options.copy(truncate_before_write=True)

        with self._writing_context(lifecycle) as io:
            with contextlib.ExitStack() as stack:
                sink = io.arrow_io(mode="wb")
                stack.callback(sink.close)
                writer = ipc.RecordBatchFileWriter(
                    sink, schema, options=options.to_writer_options(),
                )
                stack.callback(writer.close)

                if first.num_rows > 0:
                    writer.write_batch(first)

                for batch in iterator:
                    batch = options.cast_arrow_tabular(batch)
                    if batch.num_rows > 0:
                        writer.write_batch(batch)
        return None

    # ==================================================================
    # Native engine overrides — push reads to format-aware scanners
    # ==================================================================
    #
    # When the buffer is backed by a real local path (no compression
    # wrapper, no in-flight target cast, non-empty), the format-native
    # scanners (``pds.dataset(format="feather")``, ``pl.scan_ipc``,
    # ``pl.read_ipc``) are dramatically faster than the generic
    # ``pa.RecordBatchReader`` shim — they do projection / predicate
    # pushdown into the IPC reader at plan time.
    #
    # Reasons to fall back to the base path:
    #
    #   1. Empty buffer — pyarrow/polars raise on zero-byte input.
    #   2. ``options.target_field`` set — implies per-batch casting
    #      that the native scanners don't replicate.
    #   3. Compressed-codec wrapper active — the on-disk bytes are
    #      compressed, native scanners would parse them as raw IPC.
    #   4. No path (pure in-memory IO) — the cached reader lives for
    #      the IO's lifetime and would conflict with a long-lived
    #      dataset / lazy plan over the same buffer.
    #   5. ``_NATIVE_SCANNER_OK`` is False — opt-out switch for
    #      subclasses that scope reads at the batch-index level.

    def _can_use_native_scanner(self, options: ArrowIPCOptions) -> bool:
        """True iff the native IPC scanners can serve *options*."""
        if not type(self)._NATIVE_SCANNER_OK:
            return False
        if self.is_empty():
            return False
        if options.target_field is not None:
            return False
        if self.codec is not None:
            return False
        if self.path is None:
            return False
        if not self.path.is_local:
            return False
        return True

    def _read_arrow_dataset(self, options: ArrowIPCOptions) -> "pds.Dataset":
        """Native :class:`pyarrow.dataset.Dataset` over the IPC file.

        ``"feather"`` is pyarrow's canonical alias for the IPC file
        format (``"ipc"`` / ``"arrow"`` also work). Setting it
        explicitly so a path with a non-standard suffix
        (``.dat``, ``.arrowfile``) routes correctly without
        depending on extension inference.
        """
        if not self._can_use_native_scanner(options):
            return super()._read_arrow_dataset(options)

        pds = pyarrow_dataset_module()
        return pds.dataset(self.path.__fspath__(), format="feather")

    def _scan_polars_frame(self, options: ArrowIPCOptions) -> "pl.LazyFrame":
        """Native :func:`polars.scan_ipc` LazyFrame.

        The polars rust scanner pushes projections and filters into
        the IPC reader at plan time, which the
        ``scan_pyarrow_dataset`` shim path can't do as cleanly.
        """
        if not self._can_use_native_scanner(options):
            return super()._scan_polars_frame(options)

        pl = polars_module()
        return pl.scan_ipc(self.path.__fspath__())

    def _read_polars_frame(self, options: ArrowIPCOptions) -> "pl.DataFrame":
        """Native :func:`polars.read_ipc` eager :class:`pl.DataFrame`.

        Skips the Arrow Table → ``pl.from_arrow`` conversion in the
        base path — the rust-native IPC reader writes straight into
        polars' internal column representation.

        ``memory_map=True`` is polars' default and we leave it that
        way: repeated reads of the same file benefit from OS page
        caching, which is the main reason to pick IPC for working
        data in the first place. Polars silently falls back to a
        non-mmap read for compressed IPC files.

        ``options.use_threads`` (an :class:`ArrowIPCOptions` field
        consumed by the base path's
        :class:`pa.ipc.RecordBatchFileReader`) has no analogue here
        — polars manages its own thread pool. The knob still
        applies on the fallback path.
        """
        if not self._can_use_native_scanner(options):
            return super()._read_polars_frame(options)

        pl = polars_module()
        # ``use_pyarrow=False`` is the default and what we want.
        # Stating it explicitly so a future polars default flip
        # doesn't silently change our code path.
        return pl.read_ipc(self.path.__fspath__(), use_pyarrow=False)