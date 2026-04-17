"""Arrow IPC (Feather v2) I/O on top of :class:`~yggdrasil.io.buffer.BytesIO`.

Reads both the *file* and *stream* IPC layouts (tries file first, falls
back to stream). Writes use the file layout with
:class:`pyarrow.ipc.IpcWriteOptions` for intra-file body compression.

Transport-level compression is handled transparently by the base class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Optional

import pyarrow as pa
import pyarrow.ipc as ipc

from .media_io import MediaIO
from .media_options import MediaOptions

if TYPE_CHECKING:
    import pyarrow

__all__ = ["IPCIO", "IPCOptions"]


@dataclass
class IPCOptions(MediaOptions):
    """Options for Arrow IPC I/O.

    Parameters
    ----------
    compression:
        Intra-file body compression codec used by
        :class:`pyarrow.ipc.IpcWriteOptions`.
    """

    compression: str | None = "zstd"

    def __post_init__(self) -> None:
        """Normalize and validate IPC-specific options."""
        super().__post_init__()

        if self.compression is None:
            return

        if not isinstance(self.compression, str):
            raise TypeError(
                f"compression must be str|None, got {type(self.compression).__name__}"
            )

        if not self.compression:
            raise ValueError("compression must not be empty")

    @classmethod
    def resolve(cls, *, options: "IPCOptions | None" = None, **overrides: Any) -> "IPCOptions":
        """Merge *overrides* into *options* (or a fresh default)."""
        return cls.check_parameters(options=options, **overrides)


@dataclass(slots=True)
class IPCIO(MediaIO[IPCOptions]):
    """Arrow IPC I/O backed by :mod:`pyarrow.ipc`."""

    @classmethod
    def check_options(
        cls,
        options: Optional[IPCOptions],
        *args,
        **kwargs,
    ) -> IPCOptions:
        """Validate and merge caller-supplied options."""
        return IPCOptions.check_parameters(options=options, **kwargs)

    @staticmethod
    def _select_batch_columns(
        batch: "pyarrow.RecordBatch",
        *,
        options: IPCOptions,
    ) -> Iterator["pyarrow.RecordBatch"]:
        """Yield ``batch`` or a projected version when ``columns`` is set."""
        if options.columns is None:
            yield batch
            return

        table = pa.Table.from_batches([batch]).select(options.columns)
        yield from table.to_batches()

    def _read_arrow_batches(
        self,
        *,
        options: IPCOptions,
    ) -> Iterator["pyarrow.RecordBatch"]:
        """Yield record batches from the (uncompressed) IPC buffer.

        Tries the *file* layout first; falls back to *stream*.
        """
        if self.buffer.size <= 0:
            return

        arrow_io = self.buffer.to_arrow_io("r")
        try:
            try:
                reader = ipc.open_file(arrow_io)
                for index in range(reader.num_record_batches):
                    batch = reader.get_batch(index)
                    yield from self._select_batch_columns(batch, options=options)
            except (pa.ArrowInvalid, pa.ArrowIOError):
                arrow_io.seek(0)
                reader = ipc.open_stream(arrow_io)
                for batch in reader:
                    yield from self._select_batch_columns(batch, options=options)
        finally:
            arrow_io.close()

    def _collect_arrow_schema(self, full: bool = False) -> "pyarrow.Schema":
        """Return the IPC schema by reading only the file/stream header."""
        del full
        if self.buffer.size <= 0:
            return pa.schema([])

        buf, decompressed = self._decompressed_buffer()
        orig_buffer = self.buffer
        try:
            if decompressed:
                self.buffer = buf
            arrow_io = self.buffer.to_arrow_io("r")
            try:
                try:
                    return ipc.open_file(arrow_io).schema
                except (pa.ArrowInvalid, pa.ArrowIOError):
                    arrow_io.seek(0)
                    return ipc.open_stream(arrow_io).schema
            finally:
                arrow_io.close()
        finally:
            if decompressed:
                self.buffer = orig_buffer
                buf.close()

    def _write_arrow_batches(
        self,
        *,
        batches: Iterator["pyarrow.RecordBatch"],
        schema: "pyarrow.Schema",
        options: IPCOptions,
    ) -> None:
        """Write record batches as IPC into the (uncompressed) buffer."""
        arrow_io = self.buffer.to_arrow_io("w")
        try:
            write_options = ipc.IpcWriteOptions(
                compression=options.compression,
                use_legacy_format=False,
                use_threads=options.use_threads,
                allow_64bit=True,
            )
            with ipc.new_file(arrow_io, schema, options=write_options) as writer:
                for batch in batches:
                    writer.write_batch(batch)
        finally:
            arrow_io.close()