"""databricks.workspaces.io module documentation."""

import base64
import io
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, IO, AnyStr, Union

import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq
from pyarrow.dataset import FileFormat, ParquetFileFormat, CsvFileFormat

from .path_kind import DatabricksPathKind
from ...libs.databrickslib import databricks
from ...types.cast.pandas_cast import PandasDataFrame
from ...types.cast.polars_pandas_cast import PolarsDataFrame
from ...types.cast.registry import convert

if databricks is not None:
    from databricks.sdk.service.workspace import ImportFormat, ExportFormat
    from databricks.sdk.errors.platform import (
        NotFound,
        ResourceDoesNotExist,
        BadRequest,
    )

if TYPE_CHECKING:
    from .path import DatabricksPath


__all__ = [
    "DatabricksIO"
]


class DatabricksIO(ABC, IO):

    def __init__(
        self,
        path: "DatabricksPath",
        mode: str,
        encoding: Optional[str] = None,
        compression: Optional[str] = "detect",
        position: int = 0,
        buffer: Optional[io.BytesIO] = None,
    ):
        """
        __init__ documentation.
        
        Args:
            path: Parameter.
            mode: Parameter.
            encoding: Parameter.
            compression: Parameter.
            position: Parameter.
            buffer: Parameter.
        
        Returns:
            None.
        """

        super().__init__()

        self.encoding = encoding
        self.mode = mode
        self.compression = compression

        self.path = path

        self.buffer = buffer
        self.position = position

        self._write_flag = False

    def __enter__(self) -> "DatabricksIO":
        """
        __enter__ documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return self.connect(clone=False)

    def __exit__(self, exc_type, exc_value, traceback):
        """
        __exit__ documentation.
        
        Args:
            exc_type: Parameter.
            exc_value: Parameter.
            traceback: Parameter.
        
        Returns:
            The result.
        """

        self.close()

    def __del__(self):
        """
        __del__ documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        self.close()

    def __next__(self):
        """
        __next__ documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def __iter__(self):
        """
        __iter__ documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return self

    def __hash__(self):
        """
        __hash__ documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return self.path.__hash__()

    @classmethod
    def create_instance(
        cls,
        path: "DatabricksPath",
        mode: str,
        encoding: Optional[str] = None,
        compression: Optional[str] = "detect",
        position: int = 0,
        buffer: Optional[io.BytesIO] = None,
    ) -> "DatabricksIO":
        """
        create_instance documentation.
        
        Args:
            path: Parameter.
            mode: Parameter.
            encoding: Parameter.
            compression: Parameter.
            position: Parameter.
            buffer: Parameter.
        
        Returns:
            The result.
        """

        if path.kind == DatabricksPathKind.VOLUME:
            return DatabricksVolumeIO(
                path=path,
                mode=mode,
                encoding=encoding,
                compression=compression,
                position=position,
                buffer=buffer,
            )
        elif path.kind == DatabricksPathKind.DBFS:
            return DatabricksDBFSIO(
                path=path,
                mode=mode,
                encoding=encoding,
                compression=compression,
                position=position,
                buffer=buffer,
            )
        elif path.kind == DatabricksPathKind.WORKSPACE:
            return DatabricksWorkspaceIO(
                path=path,
                mode=mode,
                encoding=encoding,
                compression=compression,
                position=position,
                buffer=buffer,
            )
        else:
            raise ValueError(f"Unsupported DatabricksPath kind: {path.kind}")

    @property
    def workspace(self):
        """
        workspace documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return self.path.workspace

    @property
    def name(self):
        """
        name documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return self.path.name

    @property
    def mode(self):
        """
        mode documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return self._mode

    @mode.setter
    def mode(self, value: str):
        """
        mode documentation.
        
        Args:
            value: Parameter.
        
        Returns:
            The result.
        """

        self._mode = value

        # Basic text/binary behavior:
        # - binary -> encoding None
        # - text   -> default utf-8
        if "b" in self._mode:
            self.encoding = None
        else:
            if self.encoding is None:
                self.encoding = "utf-8"

    @property
    def content_length(self) -> int:
        """
        content_length documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return self.path.content_length

    def size(self):
        """
        size documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return self.content_length

    @content_length.setter
    def content_length(self, value: int):
        """
        content_length documentation.
        
        Args:
            value: Parameter.
        
        Returns:
            The result.
        """

        self.path.content_length = value

    @property
    def buffer(self):
        """
        buffer documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        if self._buffer is None:
            self._buffer = io.BytesIO()
            self._buffer.seek(self.position, io.SEEK_SET)
        return self._buffer

    @buffer.setter
    def buffer(self, value: Optional[io.BytesIO]):
        """
        buffer documentation.
        
        Args:
            value: Parameter.
        
        Returns:
            The result.
        """

        self._buffer = value

    def clear_buffer(self):
        """
        clear_buffer documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        if self._buffer is not None:
            self._buffer.close()
            self._buffer = None

    def clone_instance(self, **kwargs):
        """
        clone_instance documentation.
        
        Args:
            **kwargs: Parameter.
        
        Returns:
            The result.
        """

        return self.__class__(
            path=kwargs.get("path", self.path),
            mode=kwargs.get("mode", self.mode),
            encoding=kwargs.get("encoding", self.encoding),
            compression=kwargs.get("compression", self.compression),
            position=kwargs.get("position", self.position),
            buffer=kwargs.get("buffer", self._buffer),
        )

    @property
    def connected(self):
        """
        connected documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return self.path.connected

    def connect(self, clone: bool = False) -> "DatabricksIO":
        """
        connect documentation.
        
        Args:
            clone: Parameter.
        
        Returns:
            The result.
        """

        path = self.path.connect(clone=clone)

        if clone:
            return self.clone_instance(path=path)

        self.path = path
        return self

    def close(self):
        """
        close documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        self.flush()
        if self._buffer is not None:
            self._buffer.close()

    def fileno(self):
        """
        fileno documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return hash(self)

    def isatty(self):
        """
        isatty documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return False

    def tell(self):
        """
        tell documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return self.position

    def seekable(self):
        """
        seekable documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return True

    def seek(self, offset, whence=0, /):
        """
        seek documentation.
        
        Args:
            offset: Parameter.
            whence: Parameter.
        
        Returns:
            The result.
        """

        if whence == io.SEEK_SET:
            new_position = offset
        elif whence == io.SEEK_CUR:
            new_position = self.position + offset
        elif whence == io.SEEK_END:
            end_position = self.content_length
            new_position = end_position + offset
        else:
            raise ValueError("Invalid value for whence")

        if new_position < 0:
            raise ValueError("New position is before the start of the file")

        if self._buffer is not None:
            self._buffer.seek(new_position, io.SEEK_SET)

        self.position = new_position
        return self.position

    def readable(self):
        """
        readable documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return True

    def getvalue(self):
        """
        getvalue documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        if self._buffer is not None:
            return self._buffer.getvalue()
        return self.read_all_bytes()

    def getbuffer(self):
        """
        getbuffer documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return self.buffer

    @abstractmethod
    def read_byte_range(self, start: int, length: int, allow_not_found: bool = False) -> bytes:
        """
        read_byte_range documentation.
        
        Args:
            start: Parameter.
            length: Parameter.
            allow_not_found: Parameter.
        
        Returns:
            The result.
        """

        pass

    def read_all_bytes(self, use_cache: bool = True, allow_not_found: bool = False) -> bytes:
        """
        read_all_bytes documentation.
        
        Args:
            use_cache: Parameter.
            allow_not_found: Parameter.
        
        Returns:
            The result.
        """

        if use_cache and self._buffer is not None:
            buffer_value = self._buffer.getvalue()

            if len(buffer_value) == self.content_length:
                return buffer_value

            self._buffer.close()
            self._buffer = None

        data = self.read_byte_range(0, self.content_length, allow_not_found=allow_not_found)

        # Keep size accurate even if backend didn't know it
        self.content_length = len(data)

        if use_cache and self._buffer is None:
            self._buffer = io.BytesIO(data)
            self._buffer.seek(self.position, io.SEEK_SET)

        return data

    def read(self, n=-1, use_cache: bool = True):
        """
        read documentation.
        
        Args:
            n: Parameter.
            use_cache: Parameter.
        
        Returns:
            The result.
        """

        if not self.readable():
            raise IOError("File not open for reading")

        current_position = self.position
        all_data = self.read_all_bytes(use_cache=use_cache)

        if n == -1:
            n = self.content_length - current_position

        data = all_data[current_position:current_position + n]
        read_length = len(data)

        self.position += read_length

        if self.encoding:
            return data.decode(self.encoding)
        return data

    def readline(self, limit=-1, use_cache: bool = True):
        """
        readline documentation.
        
        Args:
            limit: Parameter.
            use_cache: Parameter.
        
        Returns:
            The result.
        """

        if not self.readable():
            raise IOError("File not open for reading")

        if self.encoding:
            # Text-mode: accumulate characters
            out_chars = []
            read_chars = 0

            while limit == -1 or read_chars < limit:
                ch = self.read(1, use_cache=use_cache)
                if not ch:
                    break
                out_chars.append(ch)
                read_chars += 1
                if ch == "\n":
                    break

            return "".join(out_chars)

        # Binary-mode: accumulate bytes
        line_bytes = bytearray()
        bytes_read = 0

        while limit == -1 or bytes_read < limit:
            b = self.read(1, use_cache=use_cache)
            if not b:
                break
            line_bytes.extend(b)
            bytes_read += 1
            if b == b"\n":
                break

        return bytes(line_bytes)

    def readlines(self, hint=-1, use_cache: bool = True):
        """
        readlines documentation.
        
        Args:
            hint: Parameter.
            use_cache: Parameter.
        
        Returns:
            The result.
        """

        if not self.readable():
            raise IOError("File not open for reading")

        lines = []
        total = 0

        while True:
            line = self.readline(use_cache=use_cache)
            if not line:
                break
            lines.append(line)
            total += len(line)
            if hint != -1 and total >= hint:
                break

        return lines

    def appendable(self):
        """
        appendable documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return "a" in self.mode

    def writable(self):
        """
        writable documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return True

    @abstractmethod
    def write_all_bytes(self, data: bytes):
        """
        write_all_bytes documentation.
        
        Args:
            data: Parameter.
        
        Returns:
            The result.
        """

        pass

    def truncate(self, size=None, /):
        """
        truncate documentation.
        
        Args:
            size: Parameter.
        
        Returns:
            The result.
        """

        if size is None:
            size = self.position

        if self._buffer is not None:
            self._buffer.truncate(size)
        else:
            data = b"\x00" * size
            self.write_all_bytes(data=data)

        self.content_length = size
        self._write_flag = True
        return size

    def flush(self):
        """
        flush documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        if self._write_flag and self._buffer is not None:
            self.write_all_bytes(data=self._buffer.getvalue())
            self._write_flag = False

    def write(self, data: AnyStr) -> int:
        """
        write documentation.
        
        Args:
            data: Parameter.
        
        Returns:
            The result.
        """

        if not self.writable():
            raise IOError("File not open for writing")

        if isinstance(data, str):
            data = data.encode(self.encoding or "utf-8")

        written = self.buffer.write(data)

        self.position += written
        self.content_length = self.position
        self._write_flag = True

        return written

    def writelines(self, lines) -> None:
        """
        writelines documentation.
        
        Args:
            lines: Parameter.
        
        Returns:
            The result.
        """

        for line in lines:
            if isinstance(line, str):
                line = line.encode(self.encoding or "utf-8")
            elif not isinstance(line, (bytes, bytearray)):
                raise TypeError(
                    "a bytes-like or str object is required, not '{}'".format(type(line).__name__)
                )

            data = line + b"\n" if not line.endswith(b"\n") else line
            self.write(data)

    def get_output_stream(self, *args, **kwargs):
        """
        get_output_stream documentation.
        
        Args:
            *args: Parameter.
            **kwargs: Parameter.
        
        Returns:
            The result.
        """

        return self

    def copy_to(
        self,
        dest: Union["DatabricksIO", "DatabricksPath", str]
    ) -> None:
        """
        copy_to documentation.
        
        Args:
            dest: Parameter.
        
        Returns:
            The result.
        """

        if not isinstance(dest, DatabricksIO):
            from .path import DatabricksPath

            dest_path = DatabricksPath.parse(dest, workspace=self.workspace)

            with dest_path.open(mode="wb") as d:
                return self.copy_to(dest=d)

        dest.write_all_bytes(data=self.read_all_bytes(use_cache=False))

    # ---- format helpers ----

    def _reset_for_write(self):
        """
        _reset_for_write documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        if self._buffer is not None:
            self._buffer.seek(0, io.SEEK_SET)
            self._buffer.truncate(0)

        self.position = 0
        self.content_length = 0
        self._write_flag = True

    # ---- Data Querying Helpers ----

    def write_table(
        self,
        table: Union[pa.Table, pa.RecordBatch, PolarsDataFrame, PandasDataFrame],
        file_format: Optional[FileFormat] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ):
        """
        write_table documentation.
        
        Args:
            table: Parameter.
            file_format: Parameter.
            batch_size: Parameter.
            **kwargs: Parameter.
        
        Returns:
            The result.
        """

        if isinstance(table, pa.Table):
            return self.write_arrow_table(table, file_format=file_format, batch_size=batch_size, **kwargs)
        elif isinstance(table, pa.RecordBatch):
            return self.write_arrow_batch(table, file_format=file_format, batch_size=batch_size, **kwargs)
        elif isinstance(table, PolarsDataFrame):
            return self.write_polars(table, file_format=file_format, batch_size=batch_size, **kwargs)
        elif isinstance(table, PandasDataFrame):
            return self.write_pandas(table, file_format=file_format, batch_size=batch_size, **kwargs)
        else:
            raise ValueError(f"Cannot write {type(table)} to {self.path}")

    # ---- Arrow ----

    def read_arrow_table(
        self,
        file_format: Optional[FileFormat] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> pa.Table:
        """
        read_arrow_table documentation.
        
        Args:
            file_format: Parameter.
            batch_size: Parameter.
            **kwargs: Parameter.
        
        Returns:
            The result.
        """

        file_format = self.path.file_format if file_format is None else file_format
        self.seek(0)

        if isinstance(file_format, ParquetFileFormat):
            return pq.read_table(self, **kwargs)

        if isinstance(file_format, CsvFileFormat):
            return pcsv.read_csv(self, parse_options=file_format.parse_options)

        raise ValueError(f"Unsupported file format for Arrow table: {file_format}")

    def write_arrow(
        self,
        table: Union[pa.Table, pa.RecordBatch],
        batch_size: Optional[int] = None,
        **kwargs
    ):
        """
        write_arrow documentation.
        
        Args:
            table: Parameter.
            batch_size: Parameter.
            **kwargs: Parameter.
        
        Returns:
            The result.
        """

        if not isinstance(table, pa.Table):
            table = convert(table, pa.Table)

        return self.write_arrow_table(
            table=table,
            batch_size=batch_size,
            **kwargs
        )

    def write_arrow_table(
        self,
        table: pa.Table,
        file_format: Optional[FileFormat] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ):
        """
        write_arrow_table documentation.
        
        Args:
            table: Parameter.
            file_format: Parameter.
            batch_size: Parameter.
            **kwargs: Parameter.
        
        Returns:
            The result.
        """

        file_format = self.path.file_format if file_format is None else file_format
        buffer = io.BytesIO()

        if isinstance(file_format, ParquetFileFormat):
            pq.write_table(table, buffer, write_batch_size=batch_size, **kwargs)

        elif isinstance(file_format, CsvFileFormat):
            pcsv.write_csv(table, buffer, **kwargs)

        else:
            raise ValueError(f"Unsupported file format for Arrow table: {file_format}")

        self.write_all_bytes(data=buffer.getvalue())

    def write_arrow_batch(
        self,
        batch: pa.RecordBatch,
        file_format: Optional[FileFormat] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ):
        """
        write_arrow_batch documentation.
        
        Args:
            batch: Parameter.
            file_format: Parameter.
            batch_size: Parameter.
            **kwargs: Parameter.
        
        Returns:
            The result.
        """

        table = pa.Table.from_batches([batch])
        self.write_arrow_table(table, file_format=file_format, batch_size=batch_size, **kwargs)

    def read_arrow_batches(
        self,
        batch_size: Optional[int] = None,
        **kwargs
    ):
        """
        read_arrow_batches documentation.
        
        Args:
            batch_size: Parameter.
            **kwargs: Parameter.
        
        Returns:
            The result.
        """

        return (
            self
            .read_arrow_table(batch_size=batch_size, **kwargs)
            .to_batches(max_chunksize=batch_size)
        )

    # ---- Pandas ----

    def read_pandas(
        self,
        batch_size: Optional[int] = None,
        **kwargs
    ):
        """
        read_pandas documentation.
        
        Args:
            batch_size: Parameter.
            **kwargs: Parameter.
        
        Returns:
            The result.
        """

        return self.read_arrow_table(batch_size=batch_size, **kwargs).to_pandas()

    def write_pandas(
        self,
        df,
        batch_size: Optional[int] = None,
        **kwargs
    ):
        """
        write_pandas documentation.
        
        Args:
            df: Parameter.
            batch_size: Parameter.
            **kwargs: Parameter.
        
        Returns:
            The result.
        """

        self.write_arrow_table(pa.table(df), batch_size=batch_size, **kwargs)

    # ---- Polars ----

    def read_polars(
        self,
        file_format: Optional[FileFormat] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ):
        """
        read_polars documentation.
        
        Args:
            file_format: Parameter.
            batch_size: Parameter.
            **kwargs: Parameter.
        
        Returns:
            The result.
        """

        import polars as pl

        file_format = self.path.file_format if file_format is None else file_format
        self.seek(0)

        if isinstance(file_format, ParquetFileFormat):
            return pl.read_parquet(self, **kwargs)

        if isinstance(file_format, CsvFileFormat):
            return pl.read_csv(self, **kwargs)

        raise ValueError(f"Unsupported file format for Polars DataFrame: {file_format}")

    def write_polars(
        self,
        df,
        file_format: Optional[FileFormat] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ):
        """
        write_polars documentation.
        
        Args:
            df: Parameter.
            file_format: Parameter.
            batch_size: Parameter.
            **kwargs: Parameter.
        
        Returns:
            The result.
        """

        file_format = self.path.file_format if file_format is None else FileFormat
        buffer = io.BytesIO()

        if isinstance(file_format, ParquetFileFormat):
            df.write_parquet(buffer, **kwargs)

        elif isinstance(file_format, CsvFileFormat):
            df.write_csv(buffer, **kwargs)

        else:
            raise ValueError(f"Unsupported file format for Polars DataFrame: {file_format}")

        self.write_all_bytes(data=buffer.getvalue())


class DatabricksWorkspaceIO(DatabricksIO):

    def read_byte_range(self, start: int, length: int, allow_not_found: bool = False) -> bytes:
        """
        read_byte_range documentation.
        
        Args:
            start: Parameter.
            length: Parameter.
            allow_not_found: Parameter.
        
        Returns:
            The result.
        """

        if length == 0:
            return b""

        sdk = self.workspace.sdk()
        client = sdk.workspace
        full_path = self.path.workspace_full_path()

        result = client.download(
            path=full_path,
            format=ExportFormat.AUTO,
        )

        if result is None:
            return b""

        data = result.read()

        end = start + length
        return data[start:end]

    def write_all_bytes(self, data: bytes):
        """
        write_all_bytes documentation.
        
        Args:
            data: Parameter.
        
        Returns:
            The result.
        """

        sdk = self.workspace.sdk()
        workspace_client = sdk.workspace
        full_path = self.path.workspace_full_path()

        try:
            workspace_client.upload(
                full_path,
                data,
                format=ImportFormat.AUTO,
                overwrite=True
            )
        except (NotFound, ResourceDoesNotExist, BadRequest):
            self.path.parent.make_workspace_dir(parents=True)

            workspace_client.upload(
                full_path,
                data,
                format=ImportFormat.AUTO,
                overwrite=True
            )

        self.path.reset_metadata(
            is_file=True,
            is_dir=False,
            size=len(data),
            mtime=time.time()
        )

        return self


class DatabricksVolumeIO(DatabricksIO):

    def read_byte_range(self, start: int, length: int, allow_not_found: bool = False) -> bytes:
        """
        read_byte_range documentation.
        
        Args:
            start: Parameter.
            length: Parameter.
            allow_not_found: Parameter.
        
        Returns:
            The result.
        """

        if length == 0:
            return b""

        sdk = self.workspace.sdk()
        client = sdk.files
        full_path = self.path.files_full_path()

        resp = client.download(full_path)
        result = (
            resp.contents
            .seek(start, io.SEEK_SET)
            .read(length)
        )

        return result

    def write_all_bytes(self, data: bytes):
        """
        write_all_bytes documentation.
        
        Args:
            data: Parameter.
        
        Returns:
            The result.
        """

        sdk = self.workspace.sdk()
        client = sdk.files
        full_path = self.path.files_full_path()

        try:
            client.upload(
                full_path,
                io.BytesIO(data),
                overwrite=True
            )
        except (NotFound, ResourceDoesNotExist, BadRequest):
            self.path.parent.mkdir(parents=True, exist_ok=True)

            client.upload(
                full_path,
                io.BytesIO(data),
                overwrite=True
            )

        self.path.reset_metadata(
            is_file=True,
            is_dir=False,
            size=len(data),
            mtime=time.time()
        )

        return self


class DatabricksDBFSIO(DatabricksIO):

    def read_byte_range(self, start: int, length: int, allow_not_found: bool = False) -> bytes:
        """
        read_byte_range documentation.
        
        Args:
            start: Parameter.
            length: Parameter.
            allow_not_found: Parameter.
        
        Returns:
            The result.
        """

        if length == 0:
            return b""

        sdk = self.workspace.sdk()
        client = sdk.dbfs
        full_path = self.path.dbfs_full_path()

        read_bytes = bytearray()
        bytes_to_read = length
        current_position = start

        while bytes_to_read > 0:
            chunk_size = min(bytes_to_read, 2 * 1024 * 1024)

            resp = client.read(
                path=full_path,
                offset=current_position,
                length=chunk_size
            )

            if not resp.data:
                break

            # resp.data is base64; decode and move offsets by *decoded* length
            resp_data_bytes = base64.b64decode(resp.data)

            read_bytes.extend(resp_data_bytes)
            bytes_read = len(resp_data_bytes)  # <-- FIX (was base64 string length)
            current_position += bytes_read
            bytes_to_read -= bytes_read

        return bytes(read_bytes)

    def write_all_bytes(self, data: bytes):
        """
        write_all_bytes documentation.
        
        Args:
            data: Parameter.
        
        Returns:
            The result.
        """

        sdk = self.workspace.sdk()
        client = sdk.dbfs
        full_path = self.path.dbfs_full_path()

        try:
            with client.open(
                path=full_path,
                read=False,
                write=True,
                overwrite=True
            ) as f:
                f.write(data)
        except (NotFound, ResourceDoesNotExist, BadRequest):
            self.path.parent.mkdir(parents=True, exist_ok=True)

            with client.open(
                path=full_path,
                read=False,
                write=True,
                overwrite=True
            ) as f:
                f.write(data)

        self.path.reset_metadata(
            is_file=True,
            is_dir=False,
            size=len(data),
            mtime=time.time()
        )
