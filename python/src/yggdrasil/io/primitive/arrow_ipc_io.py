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

import contextlib
import dataclasses
from typing import TYPE_CHECKING, ClassVar, Iterable, Iterator

import pyarrow as pa
import pyarrow.ipc as ipc

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.io.bytes_io import BytesIO

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


class ArrowIPCIO(BytesIO):
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

        with self._format_view() as v:
            reader = ipc.RecordBatchFileReader(v)
            for i in range(reader.num_record_batches):
                yield reader.get_batch(i)

    def _collect_schema(self, options: ArrowIPCOptions) -> Schema:
        """Read the schema straight from the IPC footer.

        Empty buffer short-circuits to :meth:`Schema.empty`. Routes
        through :meth:`_format_view` so a codec'd holder is
        transparently decompressed before the footer probe.
        """
        if self.size == 0:
            return Schema.empty()
        with self._format_view() as v:
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
        action = self._resolve_action(options.mode)

        if action is Mode.IGNORE:
            if self.size > 0:
                return
            action = Mode.OVERWRITE
        elif action is Mode.ERROR_IF_EXISTS:
            if self.size > 0:
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
            self.seek(0)
            self.truncate(0)
            return
        if first is None:
            return

        if action is Mode.APPEND and self.size > 0:
            # Read the existing batches, then chain the incoming iter,
            # then recurse with OVERWRITE. Batches are lazily walked
            # so this doesn't materialize the full table — but the
            # rewrite still re-encodes everything.
            existing = list(self._read_arrow_batches(options))
            chained = iter([*existing, first, *iterator])
            return self._write_arrow_batches(
                chained, dataclasses.replace(options, mode=Mode.OVERWRITE),
            )

        # OVERWRITE path — single writer session, codec-aware buffer.
        schema = first.schema

        with self._format_buffer() as buf:
            with contextlib.ExitStack() as stack:
                writer = ipc.RecordBatchFileWriter(
                    buf, schema, options=options.to_writer_options(),
                )
                stack.callback(writer.close)
                if first.num_rows > 0:
                    writer.write_batch(first)
                for batch in iterator:
                    if batch.num_rows > 0:
                        writer.write_batch(batch)

    # ==================================================================
    # Helpers
    # ==================================================================

    def _resolve_action(self, mode: Mode) -> Mode:
        """Pick the disposition for a write call.

        :data:`Mode.AUTO` resolves to :data:`Mode.OVERWRITE` here —
        the IPC file format has no incremental-write story, so
        "let the writer pick" means "rewrite from scratch."
        :data:`Mode.UPSERT` / :data:`Mode.MERGE` degrade to
        :data:`Mode.APPEND`; they need a key the IPC file format
        doesn't carry.
        """
        if mode is Mode.AUTO or mode is Mode.OVERWRITE or mode is Mode.TRUNCATE:
            return Mode.OVERWRITE
        if mode is Mode.APPEND:
            return Mode.APPEND
        if mode is Mode.IGNORE:
            return Mode.IGNORE
        if mode is Mode.ERROR_IF_EXISTS:
            return Mode.ERROR_IF_EXISTS
        if mode is Mode.UPSERT or mode is Mode.MERGE:
            # No key semantics at the IPC layer — closest honest
            # behavior is APPEND.
            return Mode.APPEND
        return Mode.OVERWRITE
