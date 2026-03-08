# yggdrasil/io/path.py
from __future__ import annotations

import os
from collections import deque

from .abstract import AbstractDataPath

"""
Unified path abstraction + local filesystem implementation with fast I/O.

Design:
- Local `open()` returns a normal Python file handle, so we never assume file
  handles have custom helpers like `read_polars()`.
- Instead, the path object owns `read_*_file()` / `write_*_file()`.

Directory sinks:
- If `is_dir_sink()` is True, writers shard into multiple part files.
- When sharding, we now always use the chosen `file_format` (or inferred/default)
  as the file extension for part files (per your request).

Supported formats (from FileFormat enum):
- CSV, PARQUET, JSON, AVRO, ORC, ARROW_IPC, EXCEL, BINARY
"""

import random
import shutil
import string
from pathlib import Path as SystemPath
from typing import Any, IO, Iterator, Optional, Union, TYPE_CHECKING

try:
    from typing import Self

    SelfAbstractDataPath = Self
except:
    SelfAbstractDataPath = "AbstractDataPath"

from pyarrow.fs import FileSystem, LocalFileSystem

if TYPE_CHECKING:
    from ...databricks.workspaces.workspace import Workspace

__all__ = ["LocalDataPath", "SystemPath"]


def _rand_str(n: int) -> str:
    """Generate a random alphanumeric string."""
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choices(alphabet, k=n))


def _ensure_bytes(data: Union[bytes, bytearray, memoryview, IO[bytes]]) -> bytes:
    """Normalize bytes-like or binary buffer to bytes."""
    if hasattr(data, "read"):
        return data.read()  # type: ignore[return-value]
    return bytes(data)


class LocalDataPath(SystemPath, AbstractDataPath):
    """Local filesystem path implementation (+ DatabricksPath factory)."""

    def __new__(
        cls,
        base: Union["LocalDataPath", SystemPath, AbstractDataPath, str] | None = None,
        *args,
        workspace: Optional["Workspace"] = None,
        temporary: bool = False,
        **kwargs: Any,
    ):
        if isinstance(base, str):
            if base.startswith("dbfs://"):
                from ...databricks.workspaces.path import DatabricksPath

                return DatabricksPath.parse(
                    base,
                    client=workspace,
                    temporary=temporary,
                )
        elif not args and isinstance(base, AbstractDataPath):
            return base

        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def parse_any(
        cls,
        obj: Union["LocalDataPath", str, SystemPath],
        *,
        workspace: Optional["Workspace"] = None,
        temporary: bool = False,
    ):
        if isinstance(obj, AbstractDataPath):
            return obj

        return LocalDataPath(
            obj,
            workspace=workspace,
            temporary=temporary
        )

    def path_parts(self):
        return self.parts

    def remove(self, recursive: bool = True, allow_not_found: bool = True) -> None:
        """Remove file or directory."""
        if self.is_dir():
            self.rmdir(recursive=recursive, allow_not_found=allow_not_found)
        else:
            self.rmfile(allow_not_found=allow_not_found)

    def rmfile(self, allow_not_found: bool = True) -> None:
        """Remove file."""
        self.unlink(missing_ok=allow_not_found)

    def rmdir(
        self,
        recursive: bool = True,
        allow_not_found: bool = True,
        with_root: bool = True,
    ) -> None:
        """Remove directory tree (with_root ignored for local FS)."""
        _ = (recursive, with_root)
        shutil.rmtree(self, ignore_errors=allow_not_found)

    def open(
        self,
        mode: str = "r",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
        clone: bool = False,
    ):
        """Open a local file handle and return it (fixes your old bug)."""
        _ = clone

        try:
            return super().open(
                mode=mode,
                buffering=buffering,
                encoding=encoding,
                errors=errors,
                newline=newline,
            )
        except FileNotFoundError:
            # Create parent dirs only when the mode can create/modify a file
            # r / rb should NOT create directories.
            write_like = any(m in mode for m in ("w", "a", "x", "+"))

            if write_like:
                self.parent.mkdir(parents=True, exist_ok=True)

                return super().open(
                    mode=mode,
                    buffering=buffering,
                    encoding=encoding,
                    errors=errors,
                    newline=newline,
                )
            else:
                raise

    def arrow_filesystem(self, **configs) -> FileSystem:
        """Return local PyArrow filesystem adapter."""
        return LocalFileSystem(**configs)

    def read_bytes(self) -> bytes:
        """Read raw bytes from file."""
        with self.open("rb") as f:
            return f.read()

    def write_bytes(self, data: Union[bytes, bytearray, memoryview, IO[bytes]]) -> None:
        """Write raw bytes or binary buffer to file."""
        with self.open("wb") as f:
            return f.write(_ensure_bytes(data))

    def ls(
        self,
        recursive: bool = False,
        fetch_size: Optional[int] = None,
        allow_not_found: bool = True,
    ) -> Iterator["LocalDataPath"]:
        _ = fetch_size  # local FS: no pagination

        if not self.exists():
            if allow_not_found:
                return iter(())
            raise FileNotFoundError(f"Path does not exist: {self}")

        if self.is_file():
            def _one() -> Iterator["LocalDataPath"]:
                yield self

            return _one()

        if not recursive:
            def _iterdir() -> Iterator["LocalDataPath"]:
                with os.scandir(self) as it:
                    for entry in it:
                        yield LocalDataPath(entry.path)

            return _iterdir()

        def _walk_files_only() -> Iterator["LocalDataPath"]:
            q: deque[str] = deque([str(self)])
            while q:
                root = q.popleft()
                try:
                    with os.scandir(root) as it:
                        for entry in it:
                            # Traverse dirs, but don't yield them
                            if entry.is_dir(follow_symlinks=False):
                                q.append(entry.path)
                                continue

                            yield LocalDataPath(entry.path)
                except FileNotFoundError:
                    if not allow_not_found:
                        raise

        return _walk_files_only()

    def sql_engine(self):
        # TODO: Implement SQL engine abstract and for local paths
        raise NotImplementedError
