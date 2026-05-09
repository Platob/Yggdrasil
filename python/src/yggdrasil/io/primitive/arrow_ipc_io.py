"""Arrow IPC file Tabular leaf over the new :class:`BytesIO` substrate.

:class:`ArrowIPCIO` is a :class:`BytesIO` subclass with
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

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.io.base import IO

if TYPE_CHECKING:
    pass


__all__ = ["ArrowIPCIO", "ArrowIPCOptions"]


@dataclasses.dataclass(frozen=True, slots=True)
class ArrowIPCOptions(CastOptions):
    """:class:`CastOptions` extended with IPC-specific knobs.

    Defaults match pyarrow's so an unparameterized ``ArrowIPCOptions()``
    yields an uncompressed file with the default metadata version.
    The most common knob to tweak is ``compression`` — ``"lz4"`` is
    pyarrow's standard "fast and small enough" choice; ``"zstd"`` is
    the slower-but-tighter alternative.
    """

    use_threads: bool = True
    compression: "str | None" = None  # "lz4" | "zstd" | None
    compression_level: "int | None" = None
    write_legacy_ipc_format: bool = False

    def to_writer_options(self) -> "ipc.IpcWriteOptions":
        return ipc.IpcWriteOptions(
            compression=self.compression,
            use_legacy_format=self.write_legacy_ipc_format,
        )


class ArrowIPCIO(IO[bytes, ArrowIPCOptions]):
    """:class:`Tabular` leaf for the Arrow IPC **file** format.

    File-format reads parse the footer once on construction of the
    :class:`pa.ipc.RecordBatchFileReader`; writes go through
    :class:`pa.ipc.RecordBatchFileWriter`. The streaming format
    (sequential, no footer) is a thinner relative — different leaf,
    different reader type, deliberately not folded in here.
    """

    mime_type: ClassVar[MimeTypes] = MimeTypes.ARROW_IPC

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
        if self.size == 0:
            return

        with self.arrow_input_stream() as v:
            reader = ipc.RecordBatchFileReader(v)
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                yield options.cast_arrow_tabular(batch)

    def _collect_schema(self, options: ArrowIPCOptions) -> Schema:
        """Read the schema straight from the IPC footer.

        Empty buffer short-circuits to :meth:`Schema.empty`. Routes
        through :meth:`arrow_input_stream` so a codec'd holder is
        transparently decompressed before the footer probe.
        """
        if self.size == 0:
            return Schema.empty()
        with self.arrow_input_stream() as v:
            return Schema.from_arrow(ipc.RecordBatchFileReader(v).schema)

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
        - **APPEND** — read existing batches, chain the incoming
          iterator, recurse with OVERWRITE. The IPC file format has
          one footer for all batches; partial appends would require
          rewriting the footer anyway.
        - **IGNORE** — skip when the buffer is non-empty.
        - **ERROR_IF_EXISTS** — raise when the buffer is non-empty.

        UPSERT is not meaningful at the IPC file level (no key
        semantics) and falls back to APPEND behavior with a logged
        note from the merge layer.
        """
        action = options.mode

        if action is Mode.IGNORE:
            if not self.is_empty():
                return None
            action = Mode.OVERWRITE
        elif action is Mode.ERROR_IF_EXISTS:
            if self.is_dirty():
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

        is_empty = self.is_empty()

        if action is Mode.UPSERT and not is_empty:
            upsert_options = options.check_target(self.collect_schema(options))
            raise NotImplementedError
        if action is Mode.APPEND and not is_empty:
            # Read the existing batches, then chain the incoming iter,
            # then recurse with OVERWRITE. Batches are lazily walked
            # so this doesn't materialize the full table — but the
            # rewrite still re-encodes everything.
            append_options = options.check_target(self.collect_schema(options))
            existing = list(self._read_arrow_batches(options))
            chained = append_options.cast_arrow_batch_iterator(
                iter([*existing, first, *iterator])
            )
            return self._write_arrow_batches(
                chained,
                append_options.copy(
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

        with self.arrow_output_stream() as sink:
            with ipc.RecordBatchFileWriter(
                sink,
                write_options.merged_schema.to_arrow_schema(),
                options=options.to_writer_options(),
            ) as writer:
                if first_casted.num_rows > 0:
                    writer.write_batch(first_casted)

                for batch in iterator:
                    casted_batch = write_options.cast_arrow_tabular(batch)
                    if casted_batch.num_rows > 0:
                        writer.write_batch(casted_batch)

        return None