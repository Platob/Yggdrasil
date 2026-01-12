"""File-like IO abstractions for Databricks paths."""

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
    """File-like interface for Databricks workspace, volume, or DBFS paths."""

    def __init__(
        self,
        path: "DatabricksPath",
        mode: str,
        encoding: Optional[str] = None,
        compression: Optional[str] = "detect",
        position: int = 0,
        buffer: Optional[io.BytesIO] = None,
    ):
        super().__init__()

        self.encoding = encoding
        self.mode = mode
        self.compression = compression

        self.path = path

        self.buffer = buffer
        self.position = position

        self._write_flag = False

    def __enter__(self) -> "DatabricksIO":
        """Enter a context manager and connect the underlying path."""
        return self.connect(clone=False)

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager and close the buffer."""
        self.close()

    def __del__(self):
        self.close()

    def __next__(self):
        """Iterate over lines in the file."""
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def __iter__(self):
        return self

    def __hash__(self):
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
        """Create the appropriate IO subclass for the given path kind.

        Args:
            path: DatabricksPath to open.
            mode: File mode string.
            encoding: Optional text encoding for text mode.
            compression: Optional compression mode.
            position: Initial file cursor position.
            buffer: Optional pre-seeded buffer.

        Returns:
            A DatabricksIO subclass instance.
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
        """Return the associated Workspace instance.

        Returns:
            The Workspace bound to the path.
        """
        return self.path.workspace

    @property
    def name(self):
        """Return the name of the underlying path.

        Returns:
            The path name component.
        """
        return self.path.name

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value: str):
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
        return self.path.content_length

    def size(self):
        """Return the size of the file in bytes.

        Returns:
            The file size in bytes.
        """
        return self.content_length

    @content_length.setter
    def content_length(self, value: int):
        self.path.content_length = value

    @property
    def buffer(self):
        """Return the in-memory buffer, creating it if necessary.

        Returns:
            A BytesIO buffer for the file contents.
        """
        if self._buffer is None:
            self._buffer = io.BytesIO()
            self._buffer.seek(self.position, io.SEEK_SET)
        return self._buffer

    @buffer.setter
    def buffer(self, value: Optional[io.BytesIO]):
        self._buffer = value

    def clear_buffer(self):
        """Clear any cached in-memory buffer.

        Returns:
            None.
        """
        if self._buffer is not None:
            self._buffer.close()
            self._buffer = None

    def clone_instance(self, **kwargs):
        """Clone this IO instance with optional overrides.

        Args:
            **kwargs: Field overrides for the new instance.

        Returns:
            A cloned DatabricksIO instance.
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
        """Return True if the underlying path is connected.

        Returns:
            True if connected, otherwise False.
        """
        return self.path.connected

    def connect(self, clone: bool = False) -> "DatabricksIO":
        """Connect the underlying path and optionally return a clone.

        Args:
            clone: Whether to return a cloned instance.

        Returns:
            The connected DatabricksIO instance.
        """
        path = self.path.connect(clone=clone)

        if clone:
            return self.clone_instance(path=path)

        self.path = path
        return self

    def close(self):
        """Flush pending writes and close the buffer.

        Returns:
            None.
        """
        self.flush()
        if self._buffer is not None:
            self._buffer.close()

    def fileno(self):
        """Return a pseudo file descriptor based on object hash.

        Returns:
            An integer file descriptor-like value.
        """
        return hash(self)

    def isatty(self):
        return False

    def tell(self):
        """Return the current cursor position.

        Returns:
            The current position in bytes.
        """
        return self.position

    def seekable(self):
        """Return True to indicate seek support.

        Returns:
            True.
        """
        return True

    def seek(self, offset, whence=0, /):
        """Move the cursor to a new position.

        Args:
            offset: Offset in bytes.
            whence: Reference point (start, current, end).

        Returns:
            The new position in bytes.
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
        """Return True to indicate read support.

        Returns:
            True.
        """
        return True

    def getvalue(self):
        """Return the buffer contents, reading from remote if needed.

        Returns:
            File contents as bytes or str depending on mode.
        """
        if self._buffer is not None:
            return self._buffer.getvalue()
        return self.read_all_bytes()

    def getbuffer(self):
        """Return the underlying BytesIO buffer.

        Returns:
            The BytesIO buffer instance.
        """
        return self.buffer

    @abstractmethod
    def read_byte_range(self, start: int, length: int, allow_not_found: bool = False) -> bytes:
        """Read a byte range from the remote path.

        Args:
            start: Starting byte offset.
            length: Number of bytes to read.
            allow_not_found: Whether to suppress missing-path errors.

        Returns:
            The bytes read from the remote path.
        """
        pass

    def read_all_bytes(self, use_cache: bool = True, allow_not_found: bool = False) -> bytes:
        """Read the full contents into memory, optionally caching.

        Args:
            use_cache: Whether to cache contents in memory.
            allow_not_found: Whether to suppress missing-path errors.

        Returns:
            File contents as bytes.
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
        """Read up to ``n`` bytes/characters from the file.

        Args:
            n: Number of bytes/characters to read; -1 for all.
            use_cache: Whether to use cached contents.

        Returns:
            The read bytes or string depending on mode.
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
        """Read a single line from the file.

        Args:
            limit: Max characters/bytes to read; -1 for no limit.
            use_cache: Whether to use cached contents.

        Returns:
            The next line as bytes or string.
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
        """Read all lines from the file.

        Args:
            hint: Optional byte/char count hint; -1 for no hint.
            use_cache: Whether to use cached contents.

        Returns:
            A list of lines.
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
        """Return True when the file is open in append mode.

        Returns:
            True if in append mode.
        """
        return "a" in self.mode

    def writable(self):
        """Return True to indicate write support.

        Returns:
            True.
        """
        return True

    @abstractmethod
    def write_all_bytes(self, data: bytes):
        """Write raw bytes to the remote path.

        Args:
            data: Bytes to write.

        Returns:
            None.
        """
        pass

    def truncate(self, size=None, /):
        """Resize the file to ``size`` bytes.

        Args:
            size: Target size in bytes (defaults to current position).

        Returns:
            The new size in bytes.
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
        """Flush buffered data to the remote path.

        Returns:
            None.
        """
        if self._write_flag and self._buffer is not None:
            self.write_all_bytes(data=self._buffer.getvalue())
            self._write_flag = False

    def write(self, data: AnyStr) -> int:
        """Write data to the buffer and mark for flush.

        Args:
            data: String or bytes to write.

        Returns:
            The number of bytes written.
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
        """Write multiple lines to the buffer.

        Args:
            lines: Iterable of lines to write.

        Returns:
            None.
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
        """Return this instance for compatibility with Arrow APIs.

        Returns:
            The current DatabricksIO instance.
        """
        return self

    def copy_to(
        self,
        dest: Union["DatabricksIO", "DatabricksPath", str]
    ) -> None:
        """Copy the file contents to another Databricks IO/path.

        Args:
            dest: Destination IO, DatabricksPath, or path string.

        Returns:
            None.
        """
        if not isinstance(dest, DatabricksIO):
            from .path import DatabricksPath

            dest_path = DatabricksPath.parse(dest, workspace=self.workspace)

            with dest_path.open(mode="wb") as d:
                return self.copy_to(dest=d)

        dest.write_all_bytes(data=self.read_all_bytes(use_cache=False))

    # ---- format helpers ----

    def _reset_for_write(self):
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
        """Write a table-like object to the path using an inferred format.

        Args:
            table: Table-like object to write.
            file_format: Optional file format override.
            batch_size: Optional batch size for writes.
            **kwargs: Format-specific options.

        Returns:
            The result of the specific write implementation.
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
        """Read the file as an Arrow table.

        Args:
            file_format: Optional file format override.
            batch_size: Optional batch size for reads.
            **kwargs: Format-specific options.

        Returns:
            An Arrow Table with the file contents.
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
        """Write an Arrow table or record batch to the path.

        Args:
            table: Arrow table or batch to write.
            batch_size: Optional batch size for writes.
            **kwargs: Format-specific options.

        Returns:
            None.
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
        """Write an Arrow table using the selected file format.

        Args:
            table: Arrow table to write.
            file_format: Optional file format override.
            batch_size: Optional batch size for writes.
            **kwargs: Format-specific options.

        Returns:
            None.
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
        """Write a single Arrow record batch.

        Args:
            batch: RecordBatch to write.
            file_format: Optional file format override.
            batch_size: Optional batch size for writes.
            **kwargs: Format-specific options.

        Returns:
            None.
        """
        table = pa.Table.from_batches([batch])
        self.write_arrow_table(table, file_format=file_format, batch_size=batch_size, **kwargs)

    def read_arrow_batches(
        self,
        batch_size: Optional[int] = None,
        **kwargs
    ):
        """Yield Arrow record batches from the file.

        Args:
            batch_size: Optional batch size for reads.
            **kwargs: Format-specific options.

        Returns:
            An iterator over Arrow RecordBatch objects.
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
        """Read the file into a pandas DataFrame.

        Args:
            batch_size: Optional batch size for reads.
            **kwargs: Format-specific options.

        Returns:
            A pandas DataFrame with the file contents.
        """
        return self.read_arrow_table(batch_size=batch_size, **kwargs).to_pandas()

    def write_pandas(
        self,
        df,
        batch_size: Optional[int] = None,
        **kwargs
    ):
        """Write a pandas DataFrame to the file.

        Args:
            df: pandas DataFrame to write.
            batch_size: Optional batch size for writes.
            **kwargs: Format-specific options.

        Returns:
            None.
        """
        self.write_arrow_table(pa.table(df), batch_size=batch_size, **kwargs)

    # ---- Polars ----

    def read_polars(
        self,
        file_format: Optional[FileFormat] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ):
        """Read the file into a polars DataFrame.

        Args:
            file_format: Optional file format override.
            batch_size: Optional batch size for reads.
            **kwargs: Format-specific options.

        Returns:
            A polars DataFrame with the file contents.
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
        """Write a polars DataFrame to the file.

        Args:
            df: polars DataFrame to write.
            file_format: Optional file format override.
            batch_size: Optional batch size for writes.
            **kwargs: Format-specific options.

        Returns:
            None.
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
    """IO adapter for Workspace files."""

    def read_byte_range(self, start: int, length: int, allow_not_found: bool = False) -> bytes:
        """Read bytes from a Workspace file.

        Args:
            start: Starting byte offset.
            length: Number of bytes to read.
            allow_not_found: Whether to suppress missing-path errors.

        Returns:
            Bytes read from the file.
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
        """Write bytes to a Workspace file.

        Args:
            data: Bytes to write.

        Returns:
            The DatabricksWorkspaceIO instance.
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
    """IO adapter for Unity Catalog volume files."""

    def read_byte_range(self, start: int, length: int, allow_not_found: bool = False) -> bytes:
        """Read bytes from a volume file.

        Args:
            start: Starting byte offset.
            length: Number of bytes to read.
            allow_not_found: Whether to suppress missing-path errors.

        Returns:
            Bytes read from the file.
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
        """Write bytes to a volume file.

        Args:
            data: Bytes to write.

        Returns:
            The DatabricksVolumeIO instance.
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
    """IO adapter for DBFS files."""

    def read_byte_range(self, start: int, length: int, allow_not_found: bool = False) -> bytes:
        """Read bytes from a DBFS file.

        Args:
            start: Starting byte offset.
            length: Number of bytes to read.
            allow_not_found: Whether to suppress missing-path errors.

        Returns:
            Bytes read from the file.
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
        """Write bytes to a DBFS file.

        Args:
            data: Bytes to write.

        Returns:
            The DatabricksDBFSIO instance.
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
