"""File-like IO abstractions for Databricks paths."""

import base64
import io
import logging
import time
from abc import ABC, abstractmethod
from threading import Thread
from typing import TYPE_CHECKING, Optional, IO, AnyStr, Union, Any

import dill
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq
from pyarrow.dataset import (
    FileFormat,
    ParquetFileFormat,
    CsvFileFormat,
)

from .path_kind import DatabricksPathKind
from ...libs.databrickslib import databricks
from ...libs.pandaslib import PandasDataFrame
from ...libs.polarslib import polars, PolarsDataFrame
from ...pyutils import retry
from ...types.cast.registry import convert
from ...types.file_format import ExcelFileFormat

if databricks is not None:
    from databricks.sdk.service.workspace import ImportFormat, ExportFormat
    from databricks.sdk.errors.platform import (
        NotFound,
        ResourceDoesNotExist,
        BadRequest,
    )
    from databricks.sdk.errors import InternalError

if TYPE_CHECKING:
    from .path import DatabricksPath


__all__ = [
    "DatabricksIO"
]


LOGGER = logging.getLogger(__name__)


class DatabricksIO(ABC, IO):
    """File-like interface for Databricks workspace, volume, or DBFS paths."""

    def __init__(
        self,
        path: "DatabricksPath",
        mode: str,
        encoding: Optional[str] = None,
        position: int = 0,
        buffer: Optional[io.BytesIO] = None,
    ):
        super().__init__()

        self.encoding = encoding
        self.mode = mode

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
        if self._need_flush():
            try:
                Thread(target=self.close).start()
            except BaseException:
                pass

    def __next__(self):
        """Iterate over lines in the file."""
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def __len__(self):
        return self.content_length or 0

    def __iter__(self):
        return self.read_all_bytes().__iter__()

    def __hash__(self):
        return self.path.__hash__()

    def __str__(self):
        return self.path.__str__()

    def __repr__(self):
        return "%s(path=%s)" % (
            self.__class__.__name__,
            self.path.__repr__()
        )

    @classmethod
    def create_instance(
        cls,
        path: "DatabricksPath",
        mode: str,
        encoding: Optional[str] = None,
        position: int = 0,
        buffer: Optional[io.BytesIO] = None,
    ) -> "DatabricksIO":
        """Create the appropriate IO subclass for the given path kind.

        Args:
            path: DatabricksPath to open.
            mode: File mode string.
            encoding: Optional text encoding for text mode.
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
                position=position,
                buffer=buffer,
            )
        elif path.kind == DatabricksPathKind.DBFS:
            return DatabricksDBFSIO(
                path=path,
                mode=mode,
                encoding=encoding,
                position=position,
                buffer=buffer,
            )
        elif path.kind == DatabricksPathKind.WORKSPACE:
            return DatabricksWorkspaceIO(
                path=path,
                mode=mode,
                encoding=encoding,
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

    @content_length.setter
    def content_length(self, value: int):
        self.path.content_length = value

    def size(self):
        """Return the size of the file in bytes.

        Returns:
            The file size in bytes.
        """
        return self.content_length

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

    def close(self, flush: bool = True):
        """Flush pending writes and close the buffer.

        Args:
            flush: Checks flush data to commit to remote location

        Returns:
            None.
        """
        if flush:
            self.flush()
        self.clear_buffer()

    @property
    def closed(self):
        return False

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

    def writable(self):
        """Return True to indicate write support.

        Returns:
            True.
        """
        return True

    @abstractmethod
    def write_all_bytes(self, data: Union[bytes, IO[bytes]]):
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

        if self._buffer is None:
            return self.write_all_bytes(data=b"\x00" * size)

        self._buffer.truncate(size)

        self.content_length = size
        self._write_flag = True

        return size

    def _need_flush(self):
        return self._write_flag and self._buffer is not None

    def flush(self):
        """Flush buffered data to the remote path.

        Returns:
            None.
        """
        if self._need_flush():
            self.write_all_bytes(data=self._buffer.getvalue())
            self._write_flag = False

    def write(self, data: AnyStr) -> int:
        """Write data to the buffer and mark for flush.

        Args:
            data: String or bytes to write.

        Returns:
            The number of bytes written.
        """
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
        data = self.read_all_bytes(use_cache=False)

        if isinstance(dest, DatabricksIO):
            dest.write_all_bytes(data=data)
        elif hasattr(dest, "write"):
            dest.write(data)
        else:
            from .path import DatabricksPath

            dest_path = DatabricksPath.parse(dest, workspace=self.workspace)

            with dest_path.open(mode="wb") as d:
                return self.copy_to(dest=d)

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
    ):
        """Write a table-like object to the path using an inferred format.

        Args:
            table: Table-like object to write.
            file_format: Optional file format override.
            batch_size: Optional batch size for writes.

        Returns:
            The result of the specific write implementation.
        """
        if isinstance(table, pa.Table):
            return self.write_arrow_table(table, file_format=file_format, batch_size=batch_size)
        elif isinstance(table, pa.RecordBatch):
            return self.write_arrow_batch(table, file_format=file_format, batch_size=batch_size)
        elif isinstance(table, PolarsDataFrame):
            return self.write_polars(table, file_format=file_format, batch_size=batch_size)
        elif isinstance(table, PandasDataFrame):
            return self.write_pandas(table, file_format=file_format, batch_size=batch_size)

        return self.write_arrow(
            table=table,
            file_format=file_format,
            batch_size=batch_size
        )

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

        elif isinstance(file_format, CsvFileFormat):
            return pcsv.read_csv(self, parse_options=file_format.parse_options)

        else:
            ValueError(f"Unsupported file format for Arrow table: {file_format}")

    def write_arrow(
        self,
        table: Union[pa.Table, pa.RecordBatch],
        file_format: Optional[FileFormat] = None,
        batch_size: Optional[int] = None,
    ):
        """Write an Arrow table or record batch to the path.

        Args:
            table: Arrow table or batch to write.
            file_format: Optional file format override.
            batch_size: Optional batch size for writes.

        Returns:
            None.
        """
        if not isinstance(table, pa.Table):
            table = convert(table, pa.Table)

        return self.write_arrow_table(
            table=table,
            file_format=file_format,
            batch_size=batch_size,
        )

    def write_arrow_table(
        self,
        table: pa.Table,
        file_format: Optional[FileFormat] = None,
        batch_size: Optional[int] = None,
    ):
        """Write an Arrow table using the selected file format.

        Args:
            table: Arrow table to write.
            file_format: Optional file format override.
            batch_size: Optional batch size for writes.

        Returns:
            None.
        """
        file_format = self.path.file_format if file_format is None else file_format
        buffer = io.BytesIO()

        if isinstance(file_format, ParquetFileFormat):
            pq.write_table(
                table, buffer,
                write_batch_size=batch_size
            )

        elif isinstance(file_format, CsvFileFormat):
            pcsv.write_csv(table, buffer)

        else:
            return self.write_polars(
                df=polars.from_arrow(table),
                file_format=file_format,
                batch_size=batch_size
            )

        self.write_all_bytes(data=buffer.getvalue())

    def write_arrow_batch(
        self,
        batch: pa.RecordBatch,
        file_format: Optional[FileFormat] = None,
        batch_size: Optional[int] = None,
    ):
        """Write a single Arrow record batch.

        Args:
            batch: RecordBatch to write.
            file_format: Optional file format override.
            batch_size: Optional batch size for writes.

        Returns:
            None.
        """
        table = pa.Table.from_batches([batch])
        self.write_arrow_table(table, file_format=file_format, batch_size=batch_size)

    def read_arrow_batches(
        self,
        file_format: Optional[FileFormat] = None,
        batch_size: Optional[int] = None,
    ):
        """Yield Arrow record batches from the file.

        Args:
            file_format: Optional file format override.
            batch_size: Optional batch size for reads.

        Returns:
            An iterator over Arrow RecordBatch objects.
        """
        return (
            self
            .read_arrow_table(
                file_format=file_format,
                batch_size=batch_size,
            )
            .to_batches(max_chunksize=batch_size)
        )

    # ---- Pandas ----

    def read_pandas(
        self,
        file_format: Optional[FileFormat] = None,
        batch_size: Optional[int] = None,
    ):
        """Read the file into a pandas DataFrame.

        Args:
            file_format: Optional file format override.
            batch_size: Optional batch size for reads.

        Returns:
            A pandas DataFrame with the file contents.
        """
        return self.read_arrow_table(
            file_format=file_format,
            batch_size=batch_size
        ).to_pandas()

    def write_pandas(
        self,
        df: PandasDataFrame,
        file_format: Optional[FileFormat] = None,
        batch_size: Optional[int] = None,
    ):
        """Write a pandas DataFrame to the file.

        Args:
            df: pandas DataFrame to write.
            file_format: Optional file format override.
            batch_size: Optional batch size for writes.

        Returns:
            None.
        """
        file_format = self.path.file_format if file_format is None else FileFormat
        buffer = io.BytesIO()

        if isinstance(file_format, ParquetFileFormat):
            df.to_parquet(buffer)

        elif isinstance(file_format, CsvFileFormat):
            df.to_csv(buffer)

        else:
            return self.write_polars(
                df=polars.from_pandas(df),
                file_format=file_format,
                batch_size=batch_size,
            )

        self.write_all_bytes(data=buffer.getvalue())

    # ---- Polars ----

    def read_polars(
        self,
        file_format: Optional[FileFormat] = None,
        batch_size: Optional[int] = None,
    ):
        """Read the file into a polars DataFrame.

        Args:
            file_format: Optional file format override.
            batch_size: Optional batch size for reads.

        Returns:
            A polars DataFrame with the file contents.
        """
        file_format = self.path.file_format if file_format is None else file_format
        self.seek(0)

        if isinstance(file_format, ParquetFileFormat):
            return polars.read_parquet(self)

        elif isinstance(file_format, CsvFileFormat):
            return polars.read_csv(self)

        elif isinstance(file_format, ExcelFileFormat):
            return polars.read_excel(self)

        else:
            raise ValueError(f"Unsupported file format for Polars DataFrame: {file_format}")

    def write_polars(
        self,
        df: PolarsDataFrame,
        file_format: Optional[FileFormat] = None,
        batch_size: Optional[int] = None,
    ):
        """Write a polars DataFrame to the file.

        Args:
            df: polars DataFrame to write.
            file_format: Optional file format override.
            batch_size: Optional batch size for writes.

        Returns:
            None.
        """
        file_format = self.path.file_format if file_format is None else FileFormat
        buffer = io.BytesIO()

        if isinstance(file_format, ParquetFileFormat):
            df.write_parquet(buffer)

        elif isinstance(file_format, CsvFileFormat):
            df.write_csv(buffer)

        elif isinstance(file_format, ExcelFileFormat):
            df.write_excel(buffer)

        else:
            raise ValueError(f"Unsupported file format for Polars DataFrame: {file_format}")

        self.write_all_bytes(data=buffer.getvalue())

    def read_object(
        self,
    ):
        content = self.read_all_bytes()
        obj = dill.loads(content)

        return obj

    def write_object(
        self,
        obj: Any,
        file_format: Optional[FileFormat] = None,
    ):
        buffer = io.BytesIO()

        if isinstance(obj, PandasDataFrame):
            obj.to_pickle(buffer)
        else:
            buffer.write(dill.dumps(obj))

        self.write_all_bytes(data=buffer.getvalue())

class DatabricksWorkspaceIO(DatabricksIO):
    """IO adapter for Workspace files."""

    @retry(exceptions=(InternalError,))
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

    @retry(exceptions=(InternalError,))
    def write_all_bytes(self, data: Union[bytes, IO[bytes]]):
        """Write bytes to a Workspace file.

        Args:
            data: Union[bytes, IO[bytes]] to write.

        Returns:
            The DatabricksWorkspaceIO instance.
        """
        sdk = self.workspace.sdk()
        workspace_client = sdk.workspace
        full_path = self.path.workspace_full_path()

        if isinstance(data, bytes):
            bsize = len(data)
        elif isinstance(data, io.BytesIO):
            bsize = len(data.getvalue())
        else:
            bsize = None

        LOGGER.debug(
            "Writing %s(size=%s) in %s",
            type(data),
            bsize,
            self
        )

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
            size=bsize,
            mtime=time.time()
        )

        LOGGER.info(
            "Written %s bytes in %s",
            bsize,
            self
        )

        return self


class DatabricksVolumeIO(DatabricksIO):
    """IO adapter for Unity Catalog volume files."""

    @retry(exceptions=(InternalError,))
    def read_byte_range(self, start: int, length: int, allow_not_found: bool = False) -> bytes:
        """Read bytes from a volume file.

        Args:
            start: Starting byte offset (0-based).
            length: Number of bytes to read.
            allow_not_found: Whether to suppress missing-path errors.

        Returns:
            Bytes read from the file.
        """
        if length <= 0:
            return b""
        if start < 0:
            raise ValueError(f"start must be >= 0, got {start}")
        if length < 0:
            raise ValueError(f"length must be >= 0, got {length}")

        sdk = self.workspace.sdk()
        client = sdk.files
        full_path = self.path.files_full_path()

        try:
            resp = client.download(full_path)
        except Exception as e:
            # Databricks SDK exceptions vary a bit by version; keep it pragmatic.
            if allow_not_found and any(s in str(e).lower() for s in ("not found", "not exist", "404")):
                return b""
            raise

        data = resp.contents.read()

        # If start is past EOF, return empty (common file-like behavior).
        if start >= len(data):
            return b""

        end = start + length
        return data[start:end]

    @retry(exceptions=(InternalError,))
    def write_all_bytes(self, data: Union[bytes, IO[bytes]]):
        """Write bytes to a volume file.

        Args:
            data: Union[bytes, IO[bytes]] to write.

        Returns:
            The DatabricksVolumeIO instance.
        """
        sdk = self.workspace.sdk()
        client = sdk.files
        full_path = self.path.files_full_path()

        LOGGER.debug(
            "Writing all bytes in %s",
            self
        )

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

        LOGGER.info(
            "Written all bytes in %s",
            self
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

    @retry(exceptions=(InternalError,))
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

    @retry(exceptions=(InternalError,))
    def write_all_bytes(self, data: Union[bytes, IO[bytes]]):
        """Write bytes to a DBFS file.

        Args:
            data: Union[bytes, IO[bytes]] to write.

        Returns:
            The DatabricksDBFSIO instance.
        """
        sdk = self.workspace.sdk()
        client = sdk.dbfs
        full_path = self.path.dbfs_full_path()

        LOGGER.debug(
            "Writing all bytes in %s",
            self
        )

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

        LOGGER.info(
            "Written all bytes in %s",
            self
        )

        self.path.reset_metadata(
            is_file=True,
            is_dir=False,
            size=len(data),
            mtime=time.time()
        )
