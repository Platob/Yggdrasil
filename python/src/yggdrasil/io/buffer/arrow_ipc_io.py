"""Arrow IPC (Feather v2) I/O on top of :class:`~yggdrasil.io.buffer.BytesIO`.

Reads both the *file* and *stream* IPC layouts (tries file first, falls
back to stream).  Writes use the file layout with
:class:`pyarrow.ipc.IpcWriteOptions` for intra-file body compression.

Transport-level compression is handled transparently by the base class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Optional, Self

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
    compression:
        Intra-file body compression codec used by
        :class:`pyarrow.ipc.IpcWriteOptions`.
    """


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

    def _read_arrow_batches(self, *, options: IPCOptions) -> Iterator["pyarrow.RecordBatch"]:
        """Yield record batches from the (uncompressed) IPC buffer.

        Tries the *file* layout first; falls back to *stream*.
        """
        import pyarrow as pa
        import pyarrow.ipc as ipc

        if self.buffer.size <= 0:
            return

        arrow_io = self.buffer.to_arrow_io("r")
        try:
            try:
                reader = ipc.open_file(arrow_io)
                for i in range(reader.num_record_batches):
                    batch = reader.get_batch(i)
                    if options.columns is not None:
                        tbl = pa.Table.from_batches([batch]).select(options.columns)
                        yield from tbl.to_batches()
                    else:
                        yield batch
            except (pa.ArrowInvalid, pa.ArrowIOError):
                arrow_io.seek(0)
                reader = ipc.open_stream(arrow_io)
                for batch in reader:
                    if options.columns is not None:
                        tbl = pa.Table.from_batches([batch]).select(options.columns)
                        yield from tbl.to_batches()
                    else:
                        yield batch
        finally:
            arrow_io.close()

    def _write_arrow_batches(
        self,
        *,
        batches: Iterator["pyarrow.RecordBatch"],
        schema: "pyarrow.Schema",
        options: IPCOptions,
    ) -> None:
        """Write record batches as IPC into the (uncompressed) buffer."""
        import pyarrow.ipc as ipc

        arrow_io = self.buffer.to_arrow_io("w")
        try:
            write_options = ipc.IpcWriteOptions(
                compression=options.compression,
                use_legacy_format=False,
                use_threads=True,
                allow_64bit=True,
            )
            with ipc.new_file(arrow_io, schema, options=write_options) as writer:
                for batch in batches:
                    writer.write_batch(batch)
        finally:
            arrow_io.close()