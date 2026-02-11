"""File-like IO abstractions for Databricks paths."""

import base64
import io
import logging
import os
import time
from abc import ABC, abstractmethod
from tempfile import SpooledTemporaryFile
from threading import Thread
from typing import TYPE_CHECKING, Optional, IO, AnyStr, Union, BinaryIO

from databricks.sdk.errors import InternalError
from databricks.sdk.errors.platform import (
    NotFound,
    ResourceDoesNotExist,
    BadRequest,
)
from databricks.sdk.service.workspace import ImportFormat, ExportFormat

from .path_kind import DatabricksPathKind
from ...pyutils.retry import retry

if TYPE_CHECKING:
    from .path import DatabricksPath


__all__ = [
    "DatabricksIO"
]


LOGGER = logging.getLogger(__name__)
_SPOOL_MAX = 64 * 1024 * 1024   # 64MB in RAM then spill to disk
_COPY_CHUNK = 8 * 1024 * 1024   # 8MB chunks

def _prepare_binaryio_and_size(
    data: Union[bytes, bytearray, memoryview, BinaryIO]
) -> tuple[int, BinaryIO, bool]:
    """
    Returns (size, bio, should_close).

    - bytes-like -> wrap in BytesIO (closeable by us).
    - seekable file -> compute size via fstat or seek/tell.
    - non-seekable stream -> spool into SpooledTemporaryFile, count bytes.
    """
    # bytes-like
    if isinstance(data, (bytes, bytearray, memoryview)):
        b = bytes(data)
        return len(b), io.BytesIO(b), True

    f: BinaryIO = data

    # 1) try OS-level size for real files
    try:
        fileno = f.fileno()  # type: ignore[attr-defined]
    except Exception:
        fileno = None

    if fileno is not None:
        try:
            st = os.fstat(fileno)
            # rewind if possible
            try:
                f.seek(0)
            except Exception:
                pass
            return int(st.st_size), f, False
        except Exception:
            pass

    # 2) try seek/tell (seekable streams)
    try:
        f.seek(0, io.SEEK_END)
        end = f.tell()
        f.seek(0)
        return int(end), f, False
    except Exception:
        pass

    # 3) non-seekable stream: spool + count
    spooled = SpooledTemporaryFile(max_size=_SPOOL_MAX, mode="w+b")
    size = 0
    while True:
        chunk = f.read(_COPY_CHUNK)
        if not chunk:
            break
        spooled.write(chunk)
        size += len(chunk)
    spooled.seek(0)
    return size, spooled, True

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
        return "%s(path=%s)" % (
            self.__class__.__name__,
            self.path.__repr__()
        )

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
        except (NotFound, ResourceDoesNotExist, BadRequest, InternalError) as e:
            # Databricks SDK exceptions vary a bit by version; keep it pragmatic.
            if allow_not_found:
                return b""
            raise

        data = resp.contents.read()

        # If start is past EOF, return empty (common file-like behavior).
        if start >= len(data):
            return b""

        end = start + length
        return data[start:end]

    def write_all_bytes(
        self,
        data: Union[bytes, bytearray, memoryview, BinaryIO],
        *,
        overwrite: bool = True,
        part_size: Optional[int] = None,
        use_parallel: bool = True,
        parallelism: Optional[int] = None,
    ):
        """Write bytes/stream to a volume file safely (BinaryIO upload)."""
        sdk = self.workspace.sdk()
        client = sdk.files
        full_path = self.path.files_full_path()

        LOGGER.debug("Writing all bytes in %s", self)

        size, bio, should_close = _prepare_binaryio_and_size(data)

        def _upload():
            return client.upload(
                full_path,
                bio,
                overwrite=overwrite,
                part_size=part_size,
                use_parallel=use_parallel,
                parallelism=parallelism,
            )

        try:
            _ = _upload()
        except (NotFound, ResourceDoesNotExist, BadRequest, InternalError):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Important: rewind if possible before retry
            try:
                bio.seek(0)
            except Exception:
                pass
            _ = _upload()
        finally:
            if should_close:
                try:
                    bio.close()
                except Exception:
                    pass

        self.path.reset_metadata(
            is_file=True,
            is_dir=False,
            size=size,
            mtime=time.time(),
        )

        LOGGER.info("Written %s bytes in %s", size or "all", self.path)

        return self  # or return result if your API prefers that


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
