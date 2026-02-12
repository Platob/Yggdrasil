# yggdrasil/io/path.py
from __future__ import annotations

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import pyarrow

from ..types import cast_arrow_tabular

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
import time
from abc import ABC, abstractmethod
from pathlib import Path as SystemPath
from typing import Any, IO, Iterator, Optional, Union, TYPE_CHECKING

try:
    from typing import Self

    SelfAbstractDataPath = Self
except:
    SelfAbstractDataPath = "AbstractDataPath"

import pyarrow as pa
from pyarrow.fs import FileSystem, FileType, LocalFileSystem

from ..enums import SaveMode
from ..enums.io.file_format import FileFormat
from ..pyutils.serde import ObjectSerde
from ..types.cast.cast_options import CastOptions, CastOptionsArg

if TYPE_CHECKING:
    from ..databricks.workspaces.workspace import Workspace
    from ..polars import polars
    from ..pandas import pandas


__all__ = ["AbstractDataPath", "LocalDataPath", "SystemPath"]


def _rand_str(n: int) -> str:
    """Generate a random alphanumeric string."""
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choices(alphabet, k=n))


def _ensure_bytes(data: Union[bytes, bytearray, memoryview, IO[bytes]]) -> bytes:
    """Normalize bytes-like or binary buffer to bytes."""
    if hasattr(data, "read"):
        return data.read()  # type: ignore[return-value]
    return bytes(data)


class AbstractDataPath(ABC):
    """Backend-agnostic path API with Arrow/Polars/Pandas helpers."""

    # ---------- core path ops ----------

    @abstractmethod
    def __truediv__(self, other: str) -> SelfAbstractDataPath:
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def path_parts(self) -> tuple[str, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def parent(self) -> SelfAbstractDataPath:
        raise NotImplementedError

    @abstractmethod
    def is_file(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_dir(self) -> bool:
        raise NotImplementedError

    def exists(self):
        return self.is_file() or self.is_dir()

    @abstractmethod
    def unlink(self, missing_ok: bool = True) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove(self, recursive: bool = True, allow_not_found: bool = True) -> None:
        raise NotImplementedError

    @abstractmethod
    def rmfile(self, allow_not_found: bool = True) -> None:
        raise NotImplementedError

    @abstractmethod
    def rmdir(
        self,
        recursive: bool = True,
        allow_not_found: bool = True,
        with_root: bool = True,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def ls(
        self,
        recursive: bool = False,
        fetch_size: Optional[int] = None,
        allow_not_found: bool = True,
    ) -> Iterator[SelfAbstractDataPath]:
        raise NotImplementedError

    @abstractmethod
    def open(
        self,
        mode: str = "r",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
        clone: bool = False,
    ):
        raise NotImplementedError

    @abstractmethod
    def read_bytes(self) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def write_bytes(self, data: Union[bytes, bytearray, memoryview, IO[bytes]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def arrow_filesystem(self, **configs) -> FileSystem:
        """Return PyArrow filesystem adapter for this path."""
        raise NotImplementedError

    # ---------- convenience ----------

    @property
    def extension(self) -> str:
        """File extension without leading dot (or empty string)."""
        return self.name.split(".")[-1] if "." in self.name else ""

    @property
    def file_format(self) -> FileFormat:
        """Infer format from extension."""
        return FileFormat.parse_str(self.extension)

    def check_file_format_arg(self, value: FileFormat | str | None) -> FileFormat:
        """Parse file_format arg, defaulting to inferred extension."""
        return FileFormat.parse_any(value=value, default=self.file_format)

    def is_dir_sink(self) -> bool:
        """Heuristic: treat as dataset directory output."""
        if self.is_dir():
            return True
        if self.is_file():
            return False

        parts = self.path_parts()

        if parts and not parts[-1]:
            return True

        return "." not in self.name

    @property
    def file_type(self) -> FileType:
        if self.is_file():
            return FileType.File
        if self.is_dir():
            return FileType.Directory
        return FileType.NotFound

    @abstractmethod
    def sql_engine(self):
        raise NotImplementedError

    # ---------- Move files ------------

    def sync(
        self,
        other: "AbstractDataPath",
        *,
        mode: Optional[SaveMode | str] = None,
        parallel: Optional[int | ThreadPoolExecutor] = 4,
        allow_not_found: bool = True,
    ):
        mode = SaveMode.from_any(mode)

        if not self.exists():
            if allow_not_found:
                return other
            raise FileNotFoundError(f"Path does not exist: {self}")

        if self.is_file():
            return self.sync_file(
                other=other,
                mode=mode,
                parallel=parallel,
                allow_not_found=allow_not_found,
            )

        if self.is_dir():
            return self.sync_dir(
                other=other,
                mode=mode,
                parallel=parallel,
                allow_not_found=allow_not_found,
            )

        if allow_not_found:
            return other
        raise FileNotFoundError(f"Path does not exist: {self}")

    def sync_file(
        self,
        other: "AbstractDataPath",
        *,
        mode: Optional[SaveMode | str] = None,
        parallel: Optional[int | ThreadPoolExecutor] = 4,
        allow_not_found: bool = True,
    ):
        mode = SaveMode.from_any(mode)

        if not self.exists():
            if allow_not_found:
                return other
            raise FileNotFoundError(f"Path does not exist: {self}")

        if not self.is_file():
            raise ValueError(f"sync_file() requires a file source, got: {self}")

        # Async option: if caller passes an executor, return a future
        if isinstance(parallel, ThreadPoolExecutor):
            return parallel.submit(
                self.sync_file,
                other=other,
                mode=mode,
                allow_not_found=allow_not_found,
            )

        def _dst_exists(p: AbstractDataPath) -> bool:
            try:
                return p.is_file() or p.is_dir()
            except Exception:
                return False

        if mode == SaveMode.IGNORE and _dst_exists(other):
            return other
        if mode == SaveMode.ERROR_IF_EXISTS and _dst_exists(other):
            raise FileExistsError(f"Destination exists: {other}")

        if mode == SaveMode.APPEND:
            # Append is only sane for local->local
            if isinstance(other, LocalDataPath):
                with other.open("ab") as out_f, self.open("rb") as in_f:
                    shutil.copyfileobj(in_f, out_f, length=1024 * 1024)
                return other
            raise ValueError("SaveMode.APPEND is only supported for LocalDataPath destinations")

        # OVERWRITE (default semantics)
        # Stream copy to avoid double-buffering full file in memory
        if isinstance(other, LocalDataPath):
            with other.open("wb") as out_f, self.open("rb") as in_f:
                shutil.copyfileobj(in_f, out_f, length=4 * 1024 * 1024)
            return other

        # Generic fallback (may buffer)
        other.write_bytes(self.read_bytes())
        return other

    def sync_dir(
        self,
        other: "AbstractDataPath",
        *,
        mode: Optional[SaveMode | str] = None,
        parallel: Optional[int | ThreadPoolExecutor] = 4,
        allow_not_found: bool = True,
    ):
        mode = SaveMode.from_any(mode)

        if not self.exists():
            if allow_not_found:
                return other
            raise FileNotFoundError(f"Path does not exist: {self}")

        if not self.is_dir():
            raise ValueError(f"sync_dir() requires a directory source, got: {self}")

        # If destination is local, ensure it exists
        if isinstance(other, LocalDataPath):
            other.mkdir(parents=True, exist_ok=True)

        # Executor management
        created_executor: Optional[ThreadPoolExecutor] = None
        executor: Optional[ThreadPoolExecutor]
        if parallel is None:
            executor = None
        elif isinstance(parallel, ThreadPoolExecutor):
            executor = parallel
        else:
            created_executor = ThreadPoolExecutor(max_workers=int(parallel))
            executor = created_executor

        base_parts = self.path_parts()

        def _dst_for(src_file: "LocalDataPath") -> AbstractDataPath:
            file_parts = src_file.parts
            if file_parts[: len(base_parts)] != base_parts:
                # Not under source dir (shouldn't happen), skip safely
                raise ValueError(f"File not under source directory: {src_file}")

            rel_parts = file_parts[len(base_parts) :]
            dst = other
            for part in rel_parts:
                dst = dst / part  # type: ignore[operator]
            return dst

        try:
            futures = []
            for src_file in self.ls(recursive=True, allow_not_found=allow_not_found):
                if not src_file.is_file():
                    continue
                src_file = LocalDataPath(src_file)

                dst_file = _dst_for(src_file)

                if executor is None:
                    src_file.sync_file(
                        other=dst_file,
                        mode=mode,
                        parallel=None,
                        allow_not_found=allow_not_found,
                    )
                else:
                    futures.append(
                        executor.submit(
                            src_file.sync_file,
                            other=dst_file,
                            mode=mode,
                            parallel=None,  # don't nest executors
                            allow_not_found=allow_not_found,
                        )
                    )

            if futures:
                for fut in futures:
                    fut.result()

        finally:
            if created_executor is not None:
                created_executor.shutdown(wait=True)

        return other

    # ---------- Arrow Dataset ----------

    def read_arrow_dataset(
        self,
        filesystem: Optional[FileSystem] = None,
        **kwargs
    ):
        """Create a PyArrow Dataset referencing this path."""
        import pyarrow.dataset as ds

        fs = self.arrow_filesystem(**kwargs) if filesystem is None else filesystem

        return ds.dataset(
            source=str(self),
            filesystem=fs,
            **kwargs
        )

    # ---------- File-level readers/writers (no rich handles assumed) ----------

    def read_arrow_table_file(
        self,
        file_format: FileFormat | str | None = None,
        *,
        batch_size: Optional[int] = None,
        cast_options: Optional[CastOptionsArg] = None,
    ) -> pa.Table:
        file_format = self.check_file_format_arg(file_format)

        with self.open("rb") as f:
            if file_format == FileFormat.PARQUET:
                import pyarrow.parquet as pq

                arrow_table = pq.read_table(f)

            elif file_format == FileFormat.CSV:
                import pyarrow.csv as pacsv

                arrow_table = pacsv.read_csv(f)

            elif file_format == FileFormat.ORC:
                import pyarrow.orc as orc

                arrow_table = orc.ORCFile(f).read()

            elif file_format == FileFormat.ARROW_IPC:
                import pyarrow.ipc as ipc

                arrow_table = ipc.open_file(f).read_all()

            else:
                raise ValueError(
                    "Cannot read %s with file format %s" % (
                        self,
                        file_format
                    )
                )

        if cast_options is not None:
            arrow_table = cast_arrow_tabular(arrow_table, cast_options)

        return arrow_table

    def write_arrow_table_file(
        self,
        table: pa.Table,
        file_format: FileFormat | str | None = None,
        mode: SaveMode | str | None = None,
        *,
        batch_size: Optional[int] = None,
        cast_options: Optional[CastOptionsArg] = None,
    ):
        file_format = self.check_file_format_arg(file_format)
        mode = SaveMode.from_any(mode)
        
        if cast_options is not None:
            table = cast_arrow_tabular(table, cast_options)

        with self.open("wb") as f:
            if file_format == FileFormat.PARQUET:
                import pyarrow.parquet as pq

                pq.write_table(table, f)

            elif file_format == FileFormat.CSV:
                import pyarrow.csv as pacsv

                pacsv.write_csv(table, f)

            elif file_format == FileFormat.ORC:
                import pyarrow.orc as orc

                orc.write_table(table, path)

            elif file_format == FileFormat.ARROW_IPC:
                import pyarrow.ipc as ipc

                with ipc.new_file(f, table.schema) as writer:
                    writer.write_table(table)

            elif file_format == FileFormat.JSON:
                self.write_polars_file(table.to_pandas(), file_format=file_format)

            elif file_format == FileFormat.EXCEL:
                import pandas as pd

                pd.DataFrame(table.to_pandas()).to_excel(path, index=False)

            else:
                raise ValueError(f"Unsupported Arrow write format: {file_format}")

        return self

    def read_polars_file(
        self,
        file_format: FileFormat | str | None = None,
        *,
        batch_size: Optional[int] = None,
        cast_options: Optional[CastOptionsArg] = None,
    ):
        from ..polars.lib import polars as pl
        from ..polars.cast import cast_polars_dataframe

        file_format = self.check_file_format_arg(file_format)
        cast_options = CastOptions.check_arg(cast_options)

        with self.open("rb") as f:
            if file_format == FileFormat.PARQUET:
                df = pl.read_parquet(f)

            elif file_format == FileFormat.CSV:
                if cast_options.safe:
                    null_values = None
                else:
                    null_values = [""]

                df = pl.read_csv(
                    f,
                    null_values=null_values
                )

            elif file_format == FileFormat.JSON:
                df = pl.read_json(f)

            elif file_format == FileFormat.ARROW_IPC:
                df = pl.read_ipc(f)

            elif file_format == FileFormat.ORC:
                df = pl.read_orc(f)

            elif file_format == FileFormat.AVRO:
                df = pl.read_avro(f)

            elif file_format == FileFormat.EXCEL:
                df = pl.read_excel(f)

            else:
                raise ValueError(f"Unsupported polars read format: {file_format}")

        if cast_options is not None:
            df = cast_polars_dataframe(df, cast_options)

        return df

    def write_polars_file(
        self,
        df: "polars.DataFrame",
        file_format: FileFormat | str | None = None,
        mode: SaveMode | str | None = None,
        *,
        batch_size: Optional[int] = None,
        cast_options: Optional[CastOptionsArg] = None,
    ):
        """Write a Polars DataFrame to a single file."""
        from ..polars.lib import polars as pl
        from ..polars.cast import cast_polars_dataframe

        df: pl.DataFrame = df
        fmt = self.check_file_format_arg(file_format)

        if cast_options is not None:
            df = cast_polars_dataframe(df, cast_options)

        with self.open("wb") as f:
            if fmt == FileFormat.PARQUET:
                df.write_parquet(f)

            elif fmt == FileFormat.CSV:
                df.write_csv(f)

            elif fmt == FileFormat.JSON:
                df.write_json(f)

            elif fmt == FileFormat.ARROW_IPC:
                df.write_ipc(f)

            elif fmt == FileFormat.AVRO:
                df.write_avro(f)

            elif fmt == FileFormat.EXCEL:
                df.write_excel(f)

            else:
                raise ValueError(f"Unsupported polars write format: {fmt}")

        return self

    # ---------- High-level path readers/writers (file OR directory) ----------

    def write_table(
        self,
        table: Union[pa.Table, pa.RecordBatch, "polars.DataFrame", "pandas.DataFrame", Any],
        file_format: FileFormat | str | None = None,
        mode: SaveMode | str | None = None,
        *,
        batch_size: Optional[int] = None,
        cast_options: Optional[CastOptionsArg] = None,
    ):
        """Write Arrow/Polars/Pandas-like input to this path."""
        namespace = ObjectSerde.full_namespace(obj=table)

        if namespace.startswith("pyarrow."):
            if isinstance(table, pa.RecordBatch):
                table = pa.Table.from_batches([table]) # type: ignore[arg-type]
            elif hasattr(table, "read_all"):
                table = table.read_all()
            else:
                table = pyarrow.table(table)

            return self.write_arrow_table(
                table,
                file_format=file_format,
                mode=mode,
                batch_size=batch_size,
                cast_options=cast_options,
            )
        elif namespace.startswith("polars."):
            return self.write_polars(
                table,
                file_format=file_format,
                mode=mode,
                batch_size=batch_size,
                cast_options=cast_options,
            )
        elif namespace.startswith("pandas."):
            return self.write_pandas(
                table,
                file_format=file_format,
                mode=mode,
                batch_size=batch_size,
                cast_options=cast_options,
            )

        from ..polars.cast import any_to_polars_dataframe

        return self.write_polars(
            df=any_to_polars_dataframe(table, cast_options),
            file_format=file_format,
            mode=mode,
            batch_size=batch_size,
            cast_options=cast_options,
        )

    def read_arrow_table(
        self,
        file_format: FileFormat | str | None = None,
        *,
        batch_size: Optional[int] = None,
        cast_options: Optional[CastOptionsArg] = None,
    ) -> pa.Table:
        """Read file/dir into Arrow Table."""
        if self.is_file():
            return self.read_arrow_table_file(
                file_format=file_format,
                batch_size=batch_size,
                cast_options=cast_options,
            )

        if self.is_dir():
            tables: list[pa.Table] = []
            for child in self.ls(recursive=True):
                if child.is_file():
                    tables.append(child.read_arrow_table_file(
                        file_format=file_format,
                        batch_size=batch_size,
                        cast_options=cast_options,
                    ))

            if not tables:
                return pa.Table.from_batches([], schema=pa.schema([])) # type: ignore[arg-type]

            try:
                return pa.concat_tables(tables)
            except pa.ArrowInvalid:
                from polars import CompatLevel

                return (
                    self.read_polars(
                        file_format=file_format,
                        batch_size=batch_size,
                        cast_options=cast_options,
                    )
                    .to_arrow(compat_level=CompatLevel.newest())
                )

        raise FileNotFoundError(f"Path does not exist: {self}")

    def write_arrow_table(
        self,
        table: pa.Table,
        file_format: FileFormat | str | None = None,
        mode: SaveMode | str | None = None,
        *,
        batch_size: Optional[int] = None,
        cast_options: Optional[CastOptionsArg] = None,
    ):
        """Write Arrow Table to file or shard to directory sink.

        When writing to a directory sink, part file extension always uses
        `file_format.value` (your request).
        """
        mode = SaveMode.from_any(mode)
        fmt = self.check_file_format_arg(file_format)

        if cast_options is not None:
            table = cast_arrow_tabular(table, cast_options)

        if self.is_dir_sink():
            if mode == SaveMode.OVERWRITE:
                self.rmdir(recursive=True, allow_not_found=True, with_root=False)

            seed = int(time.time() * 1000)
            rows_per_part = batch_size or 1024 * 1024

            for i, batch in enumerate(table.to_batches(max_chunksize=rows_per_part)):
                part = pa.Table.from_batches([batch], schema=table.schema) # type: ignore[arg-type]
                part_path = self / f"part-{i:05d}-{seed}-{_rand_str(4)}.{fmt.value}"
                part_path.write_arrow_table_file(
                    part,
                    file_format=fmt,
                    mode=SaveMode.OVERWRITE,
                    batch_size=batch_size,
                    cast_options=None
                )

            return self

        return self.write_arrow_table_file(
            table,
            file_format=fmt,
            mode=SaveMode.OVERWRITE,
            batch_size=batch_size,
            cast_options=None
        )

    def read_polars(
        self,
        file_format: FileFormat | str | None = None,
        *,
        batch_size: Optional[int] = None,
        cast_options: Optional[CastOptionsArg] = None,
        private_files: bool = False,
        raise_error: bool = True,
        start: Optional[int] = None,
        limit: Optional[int] = None,
    ):
        """Read file/dir into Polars DataFrame with optional start+limit."""
        from ..polars.lib import polars as pl

        fmt = FileFormat.parse_any(file_format, default=None) if file_format is not None else None

        if start is None:
            start = 0
        if start < 0:
            raise ValueError("start must be >= 0")
        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0 or None")

        # Helper: normalize "take all remaining"
        def _want_rows_remaining(total_taken: int) -> Optional[int]:
            if limit is None:
                return None
            remaining = limit - total_taken
            return max(0, remaining)

        # File case: just read then slice (unless you can push down inside read_polars_file)
        if self.is_file():
            df = self.read_polars_file(
                file_format=fmt,
                batch_size=batch_size,
                cast_options=cast_options,
            )
            # Slice is safe even if start beyond end
            if start or limit is not None:
                df = df.slice(start, limit) if limit is not None else df.slice(start)
            return df

        # Dir case: read sequentially and fill [start : start+limit)
        if self.is_dir():
            dfs = []
            skipped_rows = 0  # how many rows we've skipped so far due to start
            taken_rows = 0  # how many rows we've appended so far
            want = _want_rows_remaining(taken_rows)
            if want == 0:
                return pl.DataFrame()

            for child in self.ls(recursive=True):
                if private_files and (child.name.startswith(".") or child.name.startswith("_")):
                    continue
                if not child.is_file():
                    continue

                try:
                    df = child.read_polars_file(
                        file_format=fmt,  # None => infer per file
                        batch_size=batch_size,
                        cast_options=cast_options,
                    )
                except Exception:
                    if raise_error:
                        raise
                    else:
                        continue

                n = df.height
                if n == 0:
                    continue

                # 1) consume start (offset) across files
                if skipped_rows < start:
                    need_skip = start - skipped_rows
                    if need_skip >= n:
                        skipped_rows += n
                        continue
                    # skip part of this file
                    df = df.slice(need_skip)
                    skipped_rows = start
                    n = df.height
                    if n == 0:
                        continue

                # 2) apply remaining limit (stop early)
                want = _want_rows_remaining(taken_rows)
                if want == 0:
                    break
                if want is not None and n > want:
                    df = df.slice(0, want)
                    n = df.height

                dfs.append(df)
                taken_rows += n

                if limit is not None and taken_rows >= limit:
                    break

            if not dfs:
                return pl.DataFrame()

            return pl.concat(dfs, how="diagonal_relaxed")

        raise FileNotFoundError(f"Path does not exist: {self}")

    def write_polars(
        self,
        df: "polars.DataFrame",
        file_format: FileFormat | str | None = None,
        mode: SaveMode | str | None = None,
        *,
        batch_size: Optional[int] = None,
        cast_options: Optional[CastOptionsArg] = None,
    ):
        """Write Polars DataFrame to file or shard to directory sink.

        When writing to a directory sink, part file extension always uses
        `file_format.value`.
        """
        from ..polars.cast import cast_polars_dataframe

        mode = SaveMode.from_any(mode)
        fmt = self.check_file_format_arg(file_format)

        if cast_options is not None:
            df = cast_polars_dataframe(df, cast_options)

        if self.is_dir_sink():
            if mode == SaveMode.OVERWRITE:
                self.rmdir(recursive=True, allow_not_found=True, with_root=False)

            seed = int(time.time() * 1000)
            rows_per_part = batch_size or 1024 * 1024

            for i, chunk in enumerate(df.iter_slices(n_rows=rows_per_part)):
                part_path = self / f"part-{i:05d}-{seed}-{_rand_str(4)}.{fmt.value}"
                part_path.write_polars_file(
                    chunk,
                    file_format=file_format,
                    mode=SaveMode.OVERWRITE,
                    batch_size=batch_size,
                    cast_options=None
                )

            return self

        return self.write_polars_file(
            df,
            file_format=file_format,
            mode=SaveMode.OVERWRITE,
            batch_size=batch_size,
            cast_options=None
        )

    def read_pandas(
        self,
        file_format: FileFormat | str | None = None,
        *,
        batch_size: Optional[int] = None,
        cast_options: Optional[CastOptionsArg] = None,
    ):
        return self.read_polars(
            file_format=file_format,
            batch_size=batch_size,
            cast_options=cast_options
        )

    def write_pandas(
        self,
        df: "pandas.DataFrame",
        file_format: FileFormat | str | None = None,
        mode: SaveMode | str | None = None,
        *,
        batch_size: Optional[int] = None,
        cast_options: Optional[CastOptionsArg] = None,
    ):
        from ..polars.cast import any_to_polars_dataframe

        return self.write_polars(
            any_to_polars_dataframe(df, cast_options),
            file_format=file_format,
            mode=mode,
            batch_size=batch_size,
            cast_options=None
        )

    def sql(self, query: str, engine: str = "auto") -> pa.Table:
        """Run local SQL query referencing this path using DuckDB or Polars."""
        if engine == "auto":
            try:
                import duckdb  # noqa: F401

                engine = "duckdb"
            except ImportError:
                engine = "polars"

        from_table = f"`{self}`"
        if from_table not in query:
            raise ValueError(
                f"SQL query must contain {from_table!r} to execute query:\n{query}"
            )

        if engine == "duckdb":
            import duckdb

            __arrow_dataset__ = self.read_arrow_dataset()
            return (
                duckdb.connect()
                .execute(query=query.replace(from_table, "__arrow_dataset__"))
                .fetch_arrow_table()
            )

        if engine == "polars":
            from polars import CompatLevel

            table_name = "__dbpath__"
            return (
                self.read_polars()
                .sql(query=query.replace(from_table, table_name), table_name=table_name)
            )

        raise ValueError(f"Invalid engine {engine!r}, must be in: duckdb, polars, auto")


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
                from ..databricks.workspaces.path import DatabricksPath

                return DatabricksPath.parse(
                    base,
                    workspace=workspace,
                    temporary=temporary,
                )
        elif not args and isinstance(base, AbstractDataPath):
            return base

        return super().__new__(cls, *args, **kwargs)

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
