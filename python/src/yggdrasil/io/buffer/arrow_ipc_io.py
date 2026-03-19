"""Arrow IPC (Feather v2) I/O on top of :class:`~yggdrasil.io.buffer.BytesIO`.

Reads both the *file* and *stream* IPC layouts (tries file first, falls
back to stream).  Writes use the file layout with
:class:`pyarrow.ipc.IpcWriteOptions` for intra-file body compression.

Transport-level compression is handled transparently by the base class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Self

from .media_io import MediaIO
from .media_options import MediaOptions

if TYPE_CHECKING:
    import pyarrow

__all__ = ["IPCIO", "IPCOptions"]


@dataclass(slots=True)
class IPCOptions(MediaOptions):
    """Options for Arrow IPC I/O.

    Parameters
    ----------
    columns:
        Column names to read (``None`` reads all).
    use_threads:
        Enable multi-threaded reading.
    compression:
        Intra-file body compression codec used by
        :class:`pyarrow.ipc.IpcWriteOptions`.
    """

    columns: list[str] | None = None
    use_threads: bool = True

    compression: str | None = "zstd"

    @classmethod
    def resolve(cls, *, options: Self | None = None, **overrides) -> Self:
        """Merge *overrides* into *options* (or a fresh default)."""
        base = options or cls()
        valid = cls.__dataclass_fields__.keys()  # type: ignore[attr-defined]
        unknown = set(overrides) - set(valid)
        if unknown:
            raise TypeError(f"{cls.__name__}.resolve(): unknown option(s): {sorted(unknown)}")
        for k, v in overrides.items():
            setattr(base, k, v)
        return base


@dataclass(slots=True)
class IPCIO(MediaIO[IPCOptions]):
    """Arrow IPC I/O backed by :mod:`pyarrow.ipc`."""

    @classmethod
    def check_options(cls, options: Optional[IPCOptions], *args, **kwargs) -> IPCOptions:
        """Validate and merge caller-supplied options."""
        return IPCOptions.check_parameters(options=options, **kwargs)

    def _read_arrow_table(self, *, options: IPCOptions) -> "pyarrow.Table":
        """Read IPC bytes from the (uncompressed) buffer.

        Tries the *file* layout first; falls back to *stream* on
        :class:`ArrowInvalid`.
        """
        import pyarrow as pa
        import pyarrow.ipc as ipc

        if self.buffer.size <= 0:
            return pa.Table.from_batches([], schema=pa.schema([]))

        arrow_io = self.buffer.to_arrow_io("r")
        try:
            try:
                reader = ipc.open_file(arrow_io)
            except (pa.ArrowInvalid, pa.ArrowIOError):
                arrow_io.seek(0)
                reader = ipc.open_stream(arrow_io)

            table = reader.read_all()

            if options.columns is not None:
                table = table.select(options.columns)

            return table
        finally:
            arrow_io.close()

    def _write_arrow_table(self, *, table: "pyarrow.Table", options: IPCOptions) -> None:
        """Write an Arrow table as IPC into the (uncompressed) buffer."""
        import pyarrow.ipc as ipc

        arrow_io = self.buffer.to_arrow_io("w")
        try:
            write_options = ipc.IpcWriteOptions(
                compression=options.compression,
                use_legacy_format=False,
                use_threads=True,
                allow_64bit=True,
            )

            with ipc.new_file(
                arrow_io,
                table.schema,
                options=write_options,
            ) as writer:
                writer.write_table(table)
        finally:
            arrow_io.close()