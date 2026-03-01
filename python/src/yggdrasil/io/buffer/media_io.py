# yggdrasil/io/buffer/media_io.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar, overload, Optional

from yggdrasil.io.enums import MediaType, MimeType, SaveMode
from .bytes_io import BytesIO
from .media_options import MediaOptions

if TYPE_CHECKING:
    import pandas
    import polars
    import pyarrow

O = TypeVar("O", bound=MediaOptions)

__all__ = ["MediaIO"]


@dataclass(slots=True)
class MediaIO(ABC, Generic[O]):
    buffer: BytesIO

    @classmethod
    @abstractmethod
    def check_options(
        cls,
        options: Optional[O],
        *args,
        **kwargs
    ) -> O:
        pass

    @overload
    @classmethod
    def make(cls, buffer: BytesIO, media: MimeType.PARQUET) -> "ParquetIO": ...
    @overload
    @classmethod
    def make(cls, buffer: BytesIO, media: MimeType.JSON) -> "JsonIO": ...
    @overload
    @classmethod
    def make(cls, buffer: BytesIO, media: MimeType.NDJSON) -> "NDJsonIO": ...
    @overload
    @classmethod
    def make(cls, buffer: BytesIO, media: MimeType.ZIP) -> "ZipIO": ...
    @overload
    @classmethod
    def make(cls, buffer: BytesIO, media: MediaType | str) -> "MediaIO[MediaOptions]": ...

    @classmethod
    def make(cls, buffer: BytesIO, media: MediaType | MimeType | str) -> "MediaIO[MediaOptions]":
        media = MediaType.parse(media)
        mt = media.mime_type

        from .parquet_io import ParquetIO
        from .json_io import JsonIO
        from .zip_io import ZipIO

        if mt is MimeType.PARQUET:
            return ParquetIO(buffer=buffer)
        if mt is MimeType.JSON:
            return JsonIO(buffer=buffer)
        if mt is MimeType.ZIP:
            return ZipIO(buffer=buffer)

        raise NotImplementedError(f"Cannot create media IO for {media!r}")

    def skip_write(self, mode: SaveMode):
        if self.buffer.size > 0:
            if mode == SaveMode.IGNORE:
                return True
            elif mode == SaveMode.ERROR_IF_EXISTS:
                raise IOError(
                    f"Cannot write in already existing {self.buffer!r} with save mode {SaveMode.ERROR_IF_EXISTS.value}"
                )
        return False

    # --- Arrow (generic public API; concrete classes add typed wrappers) ---

    def read_arrow_table(self, *, options: O | None = None, **option_kwargs) -> "pyarrow.Table":
        resolved = self.check_options(options=options, **option_kwargs)
        return self._read_arrow_table(options=resolved)

    def write_arrow_table(
        self,
        table: "pyarrow.Table",
        *,
        mode: SaveMode | str | None = None,
        match_by: list[str] | None = None,
        options: O | None = None,
        **option_kwargs,
    ) -> None:
        resolved = self.check_options(
            options=options,
            mode=mode,
            match_by=match_by,
            **option_kwargs,
        )
        if self.skip_write(mode=resolved.mode):
            return self
        self._write_arrow_table(table=table, options=resolved)

    @abstractmethod
    def _read_arrow_table(self, *, options: O) -> "pyarrow.Table": ...

    @abstractmethod
    def _write_arrow_table(self, *, table: "pyarrow.Table", options: O) -> None: ...

    # --- Pandas/Polars can stay generic the same way ---

    def read_pandas_frame(self, *, options: O | None = None, **option_kwargs) -> "pandas.DataFrame":
        return self.read_arrow_table(options=options, **option_kwargs).to_pandas()

    def read_polars_frame(self, *, options: O | None = None, **option_kwargs) -> "polars.DataFrame":
        from yggdrasil.polars.lib import polars as _pl
        tb = self.read_arrow_table(options=options, **option_kwargs)
        return _pl.from_arrow(tb)