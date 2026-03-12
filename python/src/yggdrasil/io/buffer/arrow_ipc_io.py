# yggdrasil/io/buffer/ipc_io.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Self

import pyarrow as pa
import pyarrow.ipc as ipc

from .media_io import MediaIO
from .media_options import MediaOptions

if TYPE_CHECKING:
    import pyarrow


__all__ = ["IPCIO", "IPCOptions"]


@dataclass(slots=True)
class IPCOptions(MediaOptions):
    """
    Options for Arrow IPC IO.
    """

    # ---- read options ----
    columns: list[str] | None = None
    use_threads: bool = True

    # ---- write options ----
    # Arrow IPC supports compression via IpcWriteOptions in newer PyArrow versions.
    compression: str | None = "zstd"

    @classmethod
    def resolve(cls, *, options: Self | None = None, **overrides) -> Self:
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
    """
    Concrete IO for Arrow IPC.
    """

    @classmethod
    def check_options(cls, options: Optional[IPCOptions], *args, **kwargs) -> IPCOptions:
        return IPCOptions.check_parameters(options=options, **kwargs)

    def _read_arrow_table(self, *, options: IPCOptions) -> "pyarrow.Table":
        if self.buffer.size <= 0:
            return pa.Table.from_batches([], schema=pa.schema([])) # noqa

        arrow_io = self.buffer.to_arrow_io("r")
        try:
            # Try file format first, then stream format.
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
        arrow_io = self.buffer.to_arrow_io("w")
        try:
            write_options = ipc.IpcWriteOptions(
                compression=options.compression,
                use_legacy_format=False,
                use_threads=True,
                allow_64bit=True
            )

            with ipc.new_file(
                arrow_io,
                table.schema,
                options=write_options,
            ) as writer:
                writer.write_table(table)
        finally:
            arrow_io.close()