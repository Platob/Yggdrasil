"""Arrow IPC file Tabular leaf over the new :class:`BytesIO` substrate.

:class:`ArrowIPCFile` is a :class:`BytesIO` subclass with
:attr:`mime_type` set to :data:`MimeTypes.ARROW_IPC`, which auto-
registers it in the Tabular registry so :meth:`Tabular.for_holder`
dispatches a holder with that media type to this class.

Reads use :class:`pa.ipc.RecordBatchFileReader` directly against the
underlying buffer through :meth:`view` (so the caller's cursor on the
parent BytesIO is not disturbed). Writes use
:class:`pa.ipc.RecordBatchFileWriter`. The IPC file format has a
single footer for every batch, so APPEND / UPSERT modes degrade into
a read-modify-rewrite over the same buffer.

Why no :class:`pa.BufferReader` materialization
-----------------------------------------------

PyArrow accepts any ``IO[bytes]`` with ``read`` + ``seek`` for both
the reader and the writer side, and the reworked :class:`BytesIO`
satisfies that protocol exactly. Wrapping in
:class:`pa.BufferReader` would force a full bytes copy out of the
holder; passing ``self`` (or ``self.view(pos=0)``) avoids that on the
no-spill path and keeps Arrow IPC's "no decode" promise intact.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, ClassVar, Iterable, Iterator

import pyarrow as pa
import pyarrow.ipc as ipc

from yggdrasil.arrow.cast import get_arrow_nbytes
from yggdrasil.arrow.ops import upsert_arrow_batches
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.data.enums import ByteUnit, MimeTypes, Mode
from yggdrasil.io.base import IO

if TYPE_CHECKING:
    pass


__all__ = ["ArrowIPCFile", "ArrowIPCOptions"]


#: Modes that read existing bytes, merge with the incoming stream,
#: and rewrite the file in one shot. APPEND, UPSERT and MERGE all
#: share the same read-modify-rewrite shape — only the per-row
#: dedup strategy differs.
_MERGE_MODES = frozenset({Mode.APPEND, Mode.UPSERT, Mode.MERGE})


@dataclasses.dataclass(frozen=True, slots=True)
class ArrowIPCOptions(CastOptions):
    """:class:`CastOptions` extended with IPC-specific knobs.

    The most common knob to tweak is ``compression``. ``"auto"``
    (the default) defers the codec choice to write time: pick
    :data:`ArrowIPCFile.AUTO_COMPRESSION_CODEC` when the input is at
    least :data:`ArrowIPCFile.AUTO_COMPRESSION_THRESHOLD` bytes, else
    leave the file uncompressed (small payloads pay codec overhead
    they won't recoup). Pass an explicit ``"lz4"`` / ``"zstd"`` /
    ``None`` to bypass the heuristic.
    """

    use_threads: bool = True
    compression: "str | None" = "auto"  # "auto" | "lz4" | "zstd" | None
    compression_level: "int | None" = None
    write_legacy_ipc_format: bool = False

    def to_writer_options(
        self, nbytes: "int | None" = None
    ) -> "ipc.IpcWriteOptions":
        compression = self.compression
        if compression == "auto":
            if (
                nbytes is not None
                and nbytes >= ArrowIPCFile.AUTO_COMPRESSION_THRESHOLD
            ):
                compression = ArrowIPCFile.AUTO_COMPRESSION_CODEC
            else:
                compression = None
        return ipc.IpcWriteOptions(
            compression=compression,
            use_legacy_format=self.write_legacy_ipc_format,
        )


class ArrowIPCFile(IO[bytes, ArrowIPCOptions]):
    """:class:`Tabular` leaf for the Arrow IPC **file** format.

    File-format reads parse the footer once on construction of the
    :class:`pa.ipc.RecordBatchFileReader`; writes go through
    :class:`pa.ipc.RecordBatchFileWriter`. The streaming format
    (sequential, no footer) is a thinner relative — different leaf,
    different reader type, deliberately not folded in here.
    """

    mime_type: ClassVar[MimeTypes] = MimeTypes.ARROW_IPC

    #: Threshold above which ``compression="auto"`` enables a codec.
    #: 1 MiB is small enough to catch real working sets while leaving
    #: tiny ad-hoc payloads (single-row control messages, schema-only
    #: files) uncompressed where the codec overhead would dominate.
    AUTO_COMPRESSION_THRESHOLD: ClassVar[int] = ByteUnit.MIB

    #: Codec selected by ``compression="auto"`` once the threshold is
    #: crossed. ``"lz4"`` matches pyarrow's "fast and small enough"
    #: default; flip to ``"zstd"`` (or set ``compression`` explicitly)
    #: for tighter compression at write-time CPU cost.
    AUTO_COMPRESSION_CODEC: ClassVar[str] = "lz4"

    @classmethod
    def options_class(cls):
        return ArrowIPCOptions

    # ==================================================================
    # Read path
    # ==================================================================

    def _read_arrow_batches(
        self,
        options: ArrowIPCOptions,
    ) -> Iterator[pa.RecordBatch]:
        """Yield Arrow record batches in file order.

        Empty buffer → no batches. PyArrow raises ``ArrowInvalid: File
        is too small`` on a zero-byte input; the "fresh write target,
        read it back" flow is common enough to deserve the
        short-circuit.
        """
        if self.size_known and self.size == 0:
            return

        try:
            stream_ctx = self.arrow_input_stream()
            stream = stream_ctx.__enter__()
        except FileNotFoundError:
            return
        try:
            try:
                reader = ipc.RecordBatchFileReader(stream)
            except pa.ArrowInvalid:
                return
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                yield options.cast_arrow_tabular(batch)
        finally:
            stream_ctx.__exit__(None, None, None)

    def _read_arrow_table(self, options: ArrowIPCOptions) -> pa.Table:
        """Read every batch in one C++ pass via :meth:`read_all`.

        Overrides the base ``iter_batches`` + ``Table.from_batches``
        shape: :class:`pa.ipc.RecordBatchFileReader.read_all` decodes
        every batch into a single :class:`pa.Table` inside the C++
        runtime, skipping the per-batch hop through Python. The
        post-read :meth:`CastOptions.cast_arrow_tabular` still
        reshapes the table to the caller's ``target_field`` — and
        bypasses for free when the source schema already matches.
        """
        if self.size_known and self.size == 0:
            return super()._read_arrow_table(options)

        try:
            with self.arrow_input_stream() as stream:
                try:
                    reader = ipc.RecordBatchFileReader(stream)
                except pa.ArrowInvalid:
                    return super()._read_arrow_table(options)
                table = reader.read_all()
        except FileNotFoundError:
            return super()._read_arrow_table(options)
        return options.cast_arrow_tabular(table)

    def _collect_schema(self, options: ArrowIPCOptions) -> Schema:
        """Read the schema straight from the IPC footer.

        Empty buffer short-circuits to :meth:`Schema.empty`. Routes
        through :meth:`arrow_input_stream` so a codec'd holder is
        transparently decompressed before the footer probe.
        """
        if options.target:
            return options.target

        if self.size_known and self.size == 0:
            return Schema.empty()
        try:
            with self.arrow_input_stream() as v:
                schema = Schema.from_arrow(ipc.RecordBatchFileReader(v).schema)
                self._persist_schema(schema)
                return schema
        except (FileNotFoundError, pa.ArrowInvalid):
            return Schema.empty()

    # ==================================================================
    # Write path
    # ==================================================================

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: ArrowIPCOptions,
    ) -> None:
        """Persist Arrow record batches as an IPC file.

        Mode dispatch:

        - **OVERWRITE / AUTO / TRUNCATE** — single
          :class:`pa.ipc.RecordBatchFileWriter` session straight into
          the buffer (truncated to zero before the writer opens).
        - **APPEND** — read existing batches, merge with the incoming
          iterator, recurse with OVERWRITE. The IPC file format has
          one footer for all batches; partial appends would require
          rewriting the footer anyway. With ``match_by`` set,
          incoming rows whose key tuple already exists on disk are
          dropped (existing values win); without keys, the incoming
          stream is concatenated as-is.
        - **UPSERT / MERGE** — same read-modify-rewrite shape as
          APPEND, but with ``match_by`` set the existing rows
          whose key tuple is present in the incoming stream are
          dropped (incoming values win). Without keys this is
          undefined at the IPC file level and degrades to plain
          APPEND — pass ``options.match_by=[...]`` to get
          actual upsert semantics.
        - **IGNORE** — skip when the buffer is non-empty.
        - **ERROR_IF_EXISTS** — raise when the buffer is non-empty.

        Key-aware merges are powered by
        :func:`yggdrasil.arrow.ops.upsert_arrow_batches`, which
        streams the existing side through and only buffers the
        smaller of the two key sets needed to drive the dedup.
        """
        # AUTO picks the most useful mode from context:
        # ``match_by`` set → UPSERT (incoming wins on key
        # conflict); otherwise APPEND. This keeps the historical
        # "default = grow the file" behaviour while letting callers
        # opt into key-aware writes purely via ``match_by``.
        action = options.mode
        _skip_existing = self.holder_is_overwrite
        if action is Mode.AUTO:
            action = Mode.UPSERT if options.match_by_keys else Mode.APPEND
        elif action is Mode.TRUNCATE:
            action = Mode.OVERWRITE

        # ``self.size`` is the durable buffer size; ``is_empty()``
        # would only see "no bytes left to read at the cursor" and
        # would mis-fire after an earlier write parked the cursor at
        # EOF. APPEND-into-recently-written is the canonical case.
        has_existing = not _skip_existing and self.size_known and self.size > 0

        if action is Mode.IGNORE:
            if has_existing:
                return None
            action = Mode.OVERWRITE
        elif action is Mode.ERROR_IF_EXISTS:
            if has_existing:
                raise FileExistsError(
                    f"{type(self).__name__} buffer is non-empty "
                    f"({self.size} bytes); refusing to overwrite under "
                    f"mode={options.mode!r}."
                )
            action = Mode.OVERWRITE

        iterator = iter(batches)
        first = next(iterator, None)
        if first is None and action is Mode.OVERWRITE:
            # Empty payload + OVERWRITE → an empty file with the
            # caller's schema (if any) is preferable to leaving stale
            # bytes; truncate and bail.
            self.truncate(0)
            return None
        if first is None:
            return None

        if action in _MERGE_MODES and has_existing:
            rewrite_options = options.with_target(self.collect_schema(options))
            existing = list(self._read_arrow_batches(rewrite_options))
            incoming: Iterator[pa.RecordBatch] = rewrite_options.cast_arrow_batch_iterator(iter([first, *iterator]))
            merged = upsert_arrow_batches(
                iter(existing),
                incoming,
                options.match_by_keys,
                Mode.APPEND if action is Mode.APPEND else Mode.UPSERT,
                memory_pool=options.arrow_memory_pool,
            )
            return self._write_arrow_batches(
                merged,
                rewrite_options.copy(
                    mode=Mode.OVERWRITE,
                    # remove already applied since cast_arrow_batch_iterator does it
                    target=None,
                    row_size=None,
                    byte_size=None
                ),
            )

        # OVERWRITE path — drive the writer against the IO's
        # :meth:`arrow_output_stream`, which yields a
        # :class:`pa.BufferOutputStream` and bulk-commits the encoded
        # bytes (with codec compression when set) on context exit.
        write_options = options.check_source(first.schema)
        first_casted = write_options.cast_arrow_tabular(first)

        # ``compression="auto"`` resolves against the first batch's
        # in-memory size — a representative proxy for the typical
        # single-Table / single-batch payload. Iterators of many
        # tiny batches will under-trigger and stay uncompressed; pass
        # an explicit codec to force the choice.
        nbytes_hint = get_arrow_nbytes(first_casted) if first_casted.num_rows > 0 else 0

        with self.arrow_output_stream() as sink:
            with ipc.RecordBatchFileWriter(
                sink,
                write_options.merged.to_arrow_schema(),
                options=options.to_writer_options(nbytes_hint),
            ) as writer:
                if first_casted.num_rows > 0:
                    writer.write_batch(first_casted)

                for batch in iterator:
                    casted_batch = write_options.cast_arrow_tabular(batch)
                    if casted_batch.num_rows > 0:
                        writer.write_batch(casted_batch)

        return None