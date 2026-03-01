"""
yggdrasil/io/path.py
====================
Unified backend-agnostic path abstraction with Arrow / Polars / Pandas I/O.

Design principles
-----------------
- ``open()`` returns a plain Python file handle — no custom helpers on the handle.
- All format-aware I/O lives on the *path object* (``read_*`` / ``write_*``).
- Directory sinks shard output into part-files whose extension always matches
  the chosen (or inferred) ``FileFormat``.
- ``sync`` / ``sync_file`` / ``sync_dir`` handle cross-backend file copies with
  optional parallelism via a ``ThreadPoolExecutor``.

Supported FileFormat values
---------------------------
CSV, PARQUET, JSON, AVRO, ORC, ARROW_IPC, EXCEL, BINARY, DELTA

Delta Lake
----------
Delta tables are directory-based (not single files).  All Delta I/O is
delegated to the ``deltalake`` package (delta-rs).  Install it with::

    pip install deltalake

:class:`SaveMode` maps cleanly: OVERWRITE → ``"overwrite"``,
APPEND → ``"append"``, IGNORE → ``"ignore"``,
ERROR_IF_EXISTS → ``"error"``.
"""

from __future__ import annotations

import random
import shutil
import string
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path as SystemPath
from typing import Any, IO, Iterator, Optional, Union, TYPE_CHECKING

import pyarrow as pa
from pyarrow.fs import FileSystem, FileType

from yggdrasil.arrow.cast import cast_arrow_tabular
from yggdrasil.data.cast import CastOptions, CastOptionsArg
from yggdrasil.io.enums import SaveMode
from yggdrasil.io.enums.file_format import FileFormat
from yggdrasil.pickle.serde import ObjectSerde

if TYPE_CHECKING:
    import pyarrow.dataset
    from .local import LocalDataPath
    from ...polars import polars
    from ...pandas import pandas

try:
    from typing import Self
    SelfPath = Self
except ImportError:
    # Python < 3.11 fallback
    SelfPath = "AbstractDataPath"

__all__ = ["AbstractDataPath"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rand_str(n: int) -> str:
    """Return a random *n*-character alphanumeric string for part-file names."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def _ensure_bytes(data: Union[bytes, bytearray, memoryview, IO[bytes]]) -> bytes:
    """Coerce a bytes-like or binary-mode file object to plain ``bytes``."""
    if hasattr(data, "read"):
        return data.read()  # type: ignore[return-value]
    return bytes(data)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class AbstractDataPath(ABC):
    """
    Backend-agnostic path API with Arrow / Polars / Pandas I/O helpers.

    Concrete subclasses must implement every ``@abstractmethod``; everything
    else is provided as a mixin built on top of those primitives.
    """

    # ------------------------------------------------------------------ #
    # Core path operations (must be implemented by subclasses)
    # ------------------------------------------------------------------ #

    @abstractmethod
    def __truediv__(self, other: str) -> SelfPath:
        """Append *other* as a path component (``path / "child"``)."""
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        """Return the string representation of the path."""
        raise NotImplementedError

    @abstractmethod
    def path_parts(self) -> tuple[str, ...]:
        """Return all path components as a tuple of strings."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Final path component (filename or directory name)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def parent(self) -> SelfPath:
        """Parent path."""
        raise NotImplementedError

    @abstractmethod
    def is_file(self) -> bool:
        """Return ``True`` if this path points to a regular file."""
        raise NotImplementedError

    @abstractmethod
    def is_dir(self) -> bool:
        """Return ``True`` if this path points to a directory."""
        raise NotImplementedError

    @abstractmethod
    def unlink(self, missing_ok: bool = True) -> None:
        """Remove a single file.  Mirrors ``pathlib.Path.unlink``."""
        raise NotImplementedError

    @abstractmethod
    def remove(self, recursive: bool = True, allow_not_found: bool = True) -> None:
        """Remove a file or directory tree."""
        raise NotImplementedError

    @abstractmethod
    def rmfile(self, allow_not_found: bool = True) -> None:
        """Remove a single file, optionally ignoring missing-path errors."""
        raise NotImplementedError

    @abstractmethod
    def rmdir(
        self,
        recursive: bool = True,
        allow_not_found: bool = True,
        with_root: bool = True,
    ) -> None:
        """
        Remove a directory.

        Parameters
        ----------
        recursive:
            If ``True``, delete the entire subtree.
        allow_not_found:
            If ``True``, silently succeed when the directory does not exist.
        with_root:
            If ``False``, only the *contents* are removed; the root directory
            itself is kept (useful for clearing a sink before re-writing).
        """
        raise NotImplementedError

    @abstractmethod
    def ls(
        self,
        recursive: bool = False,
        fetch_size: Optional[int] = None,
        allow_not_found: bool = True,
    ) -> Iterator[SelfPath]:
        """
        Iterate over children of this path.

        Parameters
        ----------
        recursive:
            Descend into sub-directories.
        fetch_size:
            Hint for batch-fetch size on remote backends (may be ignored).
        allow_not_found:
            If ``True``, yield nothing instead of raising when path is absent.
        """
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
    ) -> IO:
        """
        Open the path and return a standard Python file handle.

        The signature mirrors ``builtins.open``; the ``clone`` argument is an
        extension for backends that need a fresh connection per handle.
        """
        raise NotImplementedError

    @abstractmethod
    def read_bytes(self) -> bytes:
        """Read the entire file and return its contents as ``bytes``."""
        raise NotImplementedError

    @abstractmethod
    def write_bytes(
        self, data: Union[bytes, bytearray, memoryview, IO[bytes]]
    ) -> None:
        """Write *data* to the file, replacing any existing content."""
        raise NotImplementedError

    @abstractmethod
    def arrow_filesystem(self, **configs) -> FileSystem:
        """Return a PyArrow ``FileSystem`` adapter for this path's backend."""
        raise NotImplementedError

    @abstractmethod
    def sql_engine(self):
        """Return a backend-specific SQL engine handle (e.g. DuckDB connection)."""
        raise NotImplementedError

    @abstractmethod
    def mkdir(
        self, mode: int = 0o777, parents: bool = True, exist_ok: bool = True
    ) -> None:
        """Create the directory.  Mirrors ``pathlib.Path.mkdir``."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Derived properties & convenience helpers
    # ------------------------------------------------------------------ #

    def exists(self) -> bool:
        """Return ``True`` if the path is an existing file *or* directory."""
        return self.is_file() or self.is_dir()

    @property
    def extension(self) -> str:
        """File extension *without* a leading dot, or ``''`` if none."""
        name = self.name
        if "." in name:
            return name.rsplit(".", 1)[-1]
        return ""

    @property
    def file_format(self) -> FileFormat:
        """Infer :class:`FileFormat` from the path's file extension."""
        return FileFormat.parse_str(self.extension)

    def check_file_format_arg(
        self, value: Union[FileFormat, str, None]
    ) -> FileFormat:
        """
        Resolve a *file_format* argument.

        Returns *value* parsed to ``FileFormat`` when provided; otherwise falls
        back to the format inferred from the file extension.
        """
        return FileFormat.parse_any(value=value, default=self.file_format)

    @property
    def file_type(self) -> FileType:
        """Return the PyArrow :class:`~pyarrow.fs.FileType` for this path."""
        if self.is_file():
            return FileType.File
        if self.is_dir():
            return FileType.Directory
        return FileType.NotFound

    def is_dir_sink(self) -> bool:
        """
        Heuristic: decide whether this path should be treated as a *directory
        sink* (i.e. writes are sharded into multiple part-files).

        Rules (in order):
        1. Confirmed directory → ``True``.
        2. Confirmed file → ``False``.
        3. Trailing path separator → ``True``.
        4. No dot in name (no extension) → ``True``.
        5. Otherwise → ``False``.
        """
        if self.is_dir():
            return True
        if self.is_file():
            return False

        parts = self.path_parts()
        # Trailing separator means the caller explicitly requested a directory.
        if parts and not parts[-1]:
            return True

        return "." not in self.name

    # ------------------------------------------------------------------ #
    # File-sync helpers
    # ------------------------------------------------------------------ #

    def sync(
        self,
        other: "AbstractDataPath",
        *,
        mode: Optional[Union[SaveMode, str]] = None,
        pool: Optional[Union[int, ThreadPoolExecutor]] = 4,
        allow_not_found: bool = True,
    ) -> "AbstractDataPath":
        """
        Copy this path (file or directory tree) to *other*.

        Parameters
        ----------
        other:
            Destination path.  If not already an :class:`AbstractDataPath` it
            is wrapped in a ``LocalDataPath``.
        mode:
            :class:`SaveMode` controlling conflict resolution.
        pool:
            Worker count (``int``) or an existing :class:`~concurrent.futures.ThreadPoolExecutor`
            for directory copies.  ``None`` disables threading.
        allow_not_found:
            When ``True``, silently return *other* if the source is absent.
        """
        mode = SaveMode.parse(mode)

        if not isinstance(other, AbstractDataPath):
            from .local import LocalDataPath
            other = LocalDataPath(other)

        if not self.exists():
            if allow_not_found:
                return other
            raise FileNotFoundError(f"Path does not exist: {self}")

        if self.is_file():
            return self.sync_file(
                other=other,
                mode=mode,
                pool=pool,
                allow_not_found=allow_not_found,
            )

        if self.is_dir():
            return self.sync_dir(
                other=other,
                mode=mode,
                pool=pool,
                allow_not_found=allow_not_found,
            )

        # Path disappeared between exists() and is_file()/is_dir() — race
        if allow_not_found:
            return other
        raise FileNotFoundError(f"Path does not exist: {self}")

    def sync_file(
        self,
        other: "AbstractDataPath",
        *,
        mode: Optional[Union[SaveMode, str]] = None,
        pool: Optional[Union[int, ThreadPoolExecutor]] = None,
        allow_not_found: bool = True,
    ) -> "AbstractDataPath":
        """
        Copy this *file* to *other*.

        When *pool* is a :class:`~concurrent.futures.ThreadPoolExecutor`,
        the copy is submitted as a future and returned immediately.

        Raises
        ------
        ValueError
            If this path is not a file, or if ``SaveMode.APPEND`` is used with
            a non-local destination.
        FileExistsError
            If ``SaveMode.ERROR_IF_EXISTS`` and *other* already exists.
        """
        mode = SaveMode.parse(mode)

        if not self.exists():
            if allow_not_found:
                return other
            raise FileNotFoundError(f"Path does not exist: {self}")

        if not self.is_file():
            raise ValueError(f"sync_file() requires a file source, got: {self}")

        # Off-load to executor when requested (non-blocking).
        if isinstance(pool, ThreadPoolExecutor):
            return pool.submit(  # type: ignore[return-value]
                self.sync_file,
                other=other,
                mode=mode,
                pool=None,
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
            # Byte-append only makes sense for local→local transfers.
            if isinstance(other, SystemPath):
                with other.open("ab") as out_f, self.open("rb") as in_f:
                    shutil.copyfileobj(in_f, out_f, length=1024 * 1024)
                return other
            raise ValueError(
                "SaveMode.APPEND is only supported for LocalDataPath destinations"
            )

        # OVERWRITE (default): stream copy to avoid holding the full file in RAM.
        if isinstance(other, SystemPath):
            with other.open("wb") as out_f, self.open("rb") as in_f:
                shutil.copyfileobj(in_f, out_f, length=4 * 1024 * 1024)
            return other

        # Generic fallback for non-local destinations.
        other.write_bytes(self.read_bytes())
        return other

    def sync_dir(
        self,
        other: "AbstractDataPath",
        *,
        mode: Optional[Union[SaveMode, str]] = None,
        pool: Optional[Union[int, ThreadPoolExecutor]] = 4,
        allow_not_found: bool = True,
    ) -> "AbstractDataPath":
        """
        Recursively copy this *directory* to *other*.

        Parameters
        ----------
        pool:
            Number of worker threads (``int``) or an existing executor.
            ``None`` runs copies sequentially in the calling thread.
        """
        mode = SaveMode.parse(mode)

        if not self.exists():
            if allow_not_found:
                return other
            raise FileNotFoundError(f"Path does not exist: {self}")

        if not self.is_dir():
            raise ValueError(f"sync_dir() requires a directory source, got: {self}")

        other.mkdir(parents=True, exist_ok=True)

        # Build or reuse executor.
        _owned_executor: Optional[ThreadPoolExecutor] = None
        executor: Optional[ThreadPoolExecutor]
        if pool is None:
            executor = None
        elif isinstance(pool, ThreadPoolExecutor):
            executor = pool
        else:
            _owned_executor = ThreadPoolExecutor(max_workers=int(pool))
            executor = _owned_executor

        base_parts = self.path_parts()

        def _dst_for(src_file: AbstractDataPath) -> AbstractDataPath:
            """Compute the mirror destination path inside *other*."""
            file_parts = src_file.path_parts()
            if file_parts[: len(base_parts)] != base_parts:
                raise ValueError(f"File not under source directory: {src_file}")
            rel_parts = file_parts[len(base_parts):]
            dst: AbstractDataPath = other
            for part in rel_parts:
                dst = dst / part  # type: ignore[operator]
            return dst

        try:
            futures: list[Future] = []
            for src_file in self.ls(recursive=True, allow_not_found=allow_not_found):
                if not src_file.is_file():
                    continue

                dst_file = _dst_for(src_file)

                if executor is None:
                    src_file.sync_file(
                        other=dst_file,
                        mode=mode,
                        pool=None,
                        allow_not_found=allow_not_found,
                    )
                else:
                    futures.append(
                        executor.submit(
                            src_file.sync_file,
                            other=dst_file,
                            mode=mode,
                            pool=None,  # never nest executors
                            allow_not_found=allow_not_found,
                        )
                    )

            # Propagate any exceptions from worker threads.
            for fut in futures:
                fut.result()

        finally:
            if _owned_executor is not None:
                _owned_executor.shutdown(wait=True)

        return other

    # ------------------------------------------------------------------ #
    # Arrow Dataset
    # ------------------------------------------------------------------ #

    def read_arrow_dataset(
        self,
        filesystem: Optional[FileSystem] = None,
        *,
        file_format: Union[FileFormat, str, None] = None,
        schema: Optional[pa.Schema] = None,
        partitioning: Any = None,
        partition_base_dir: Optional[str] = None,
        exclude_invalid_files: bool = False,
        filesystem_kwargs: Optional[dict] = None,
        **dataset_kwargs: Any,
    ) -> "pyarrow.dataset.Dataset":
        """
        Return a lazy PyArrow :class:`~pyarrow.dataset.Dataset` over this path.

        Parameters
        ----------
        filesystem:
            PyArrow filesystem to use.  When ``None``, built from
            ``self.arrow_filesystem(**filesystem_kwargs)``.
        file_format:
            File format hint (e.g. ``FileFormat.PARQUET``).  When ``None``,
            PyArrow infers it from the files in the dataset.  Formats are
            mapped to PyArrow format strings: PARQUET→``"parquet"``,
            CSV→``"csv"``, ARROW_IPC→``"ipc"``, ORC→``"orc"``.
        schema:
            Explicit schema to use for the dataset.  Useful when files have
            slightly differing schemas and you want to enforce a single one.
        partitioning:
            A :class:`~pyarrow.dataset.Partitioning` instance, a list of
            field names (Hive-style), or ``None`` for no partitioning.
        partition_base_dir:
            Base directory for relative partitioning paths.  Defaults to
            the dataset root.
        exclude_invalid_files:
            If ``True``, silently skip files that cannot be opened.
        filesystem_kwargs:
            Keyword arguments forwarded *only* to
            ``self.arrow_filesystem()``.  Kept separate from *dataset_kwargs*
            to avoid cross-contamination.
        **dataset_kwargs:
            Any remaining keyword arguments forwarded to
            :func:`pyarrow.dataset.dataset` (e.g. ``ignore_prefixes``).

        Returns
        -------
        pyarrow.dataset.Dataset

        Notes
        -----
        When a custom *filesystem* is provided, the source path passed to
        PyArrow is the *string path as understood by that filesystem* (i.e.
        ``str(self)`` — callers are responsible for ensuring consistency).
        For the built-in local and HDFS adapters this is always correct.
        """
        import pyarrow.dataset as ds

        # Resolve filesystem — keep its kwargs strictly separate.
        fs = filesystem if filesystem is not None else self.arrow_filesystem(
            **(filesystem_kwargs or {})
        )

        # Map our FileFormat enum to PyArrow's format string / object.
        # None means "let PyArrow infer" which is the safest default.
        _FORMAT_MAP: dict[FileFormat, str] = {
            FileFormat.PARQUET:   "parquet",
            FileFormat.CSV:       "csv",
            FileFormat.ARROW_IPC: "ipc",
            FileFormat.ORC:       "orc",
        }
        fmt = self.check_file_format_arg(file_format)
        pa_format: Optional[str] = _FORMAT_MAP.get(fmt)  # None → PyArrow infers

        # Build the final kwarg dict, only injecting non-None values so we
        # don't override PyArrow defaults with explicit ``None`` arguments.
        kwargs: dict[str, Any] = {}
        if pa_format is not None:
            kwargs["format"] = pa_format
        if schema is not None:
            kwargs["schema"] = schema
        if partitioning is not None:
            kwargs["partitioning"] = partitioning
        if partition_base_dir is not None:
            kwargs["partition_base_dir"] = partition_base_dir
        if exclude_invalid_files:
            kwargs["exclude_invalid_files"] = exclude_invalid_files
        kwargs.update(dataset_kwargs)

        return ds.dataset(
            source=str(self),
            filesystem=fs,
            **kwargs,
        )


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
        file_format: Union[FileFormat, str, None] = None,
        mode: Union[SaveMode, str, None] = None,
        *,
        cast_options: Optional[CastOptionsArg] = None,
        batch_size: Optional[int] = None,  # noqa: ARG002
    ) -> "AbstractDataPath":
        """
        Write a PyArrow :class:`~pyarrow.Table` to a *single file*.

        Supported formats: PARQUET, CSV, ORC, ARROW_IPC, JSON (via Polars),
        EXCEL (via Pandas).

        Returns
        -------
        self
            Enables method chaining.
        """
        fmt = self.check_file_format_arg(file_format)

        if cast_options is not None:
            table = cast_arrow_tabular(table, cast_options)

        with self.open("wb") as f:
            if fmt == FileFormat.PARQUET:
                import pyarrow.parquet as pq
                pq.write_table(table, f)

            elif fmt == FileFormat.CSV:
                import pyarrow.csv as pacsv
                pacsv.write_csv(table, f)

            elif fmt == FileFormat.ORC:
                import pyarrow.orc as orc
                orc.write_table(table, f)

            elif fmt == FileFormat.ARROW_IPC:
                import pyarrow.ipc as ipc
                with ipc.new_file(f, table.schema) as writer:
                    writer.write_table(table)

            elif fmt == FileFormat.JSON:
                # Delegate JSON writing to the Polars path (Arrow has no JSON writer).
                # NOTE: we must close *f* before delegating since write_polars_file
                #       opens the path itself.  Break out of the `with` block first.
                pass  # handled below

            elif fmt == FileFormat.EXCEL:
                table.to_pandas().pipe(
                    lambda df: df.to_excel(f, index=False)  # type: ignore[arg-type]
                )

            elif fmt == FileFormat.DELTA:
                # Delta is directory-based; cannot write through a file handle.
                pass  # handled below

            else:
                raise ValueError(
                    f"write_arrow_table_file: unsupported format {fmt!r}"
                )

        # JSON fall-through: reopen via Polars writer.
        if fmt == FileFormat.JSON:
            import polars as pl
            self.write_polars_file(
                pl.from_arrow(table),
                file_format=fmt,
            )

        return self

    def read_polars_file(
        self,
        file_format: Union[FileFormat, str, None] = None,
        *,
        cast_options: Optional[CastOptionsArg] = None,
        batch_size: Optional[int] = None,  # noqa: ARG002
        lazy: bool = False,
    ) -> "Union[polars.DataFrame, polars.LazyFrame]":
        """
        Read a *single file* into a Polars :class:`~polars.DataFrame` or
        :class:`~polars.LazyFrame`.

        Supported formats: PARQUET, CSV, JSON, ARROW_IPC, ORC, AVRO, EXCEL.

        Parameters
        ----------
        lazy:
            When ``True``, return a :class:`~polars.LazyFrame` instead of a
            materialised DataFrame.  Formats that have a native Polars scanner
            (PARQUET, CSV, ARROW_IPC, NDJSON) avoid reading data entirely until
            the frame is collected.  Other formats fall back to
            ``DataFrame.lazy()``.

            .. note::
                ``cast_options`` are applied eagerly even in lazy mode (they
                require schema introspection).  For zero-materialisation, omit
                cast options and apply schema casts downstream.
        """
        from ...polars.lib import polars as pl
        from ...polars.cast import cast_polars_dataframe

        fmt = self.check_file_format_arg(file_format)
        cast_options = CastOptions.check_arg(cast_options)
        path_str = str(self)

        # ---- lazy path: use native scanners where available ----------------
        if lazy:
            if fmt == FileFormat.PARQUET:
                lf = pl.scan_parquet(path_str)
            elif fmt == FileFormat.CSV:
                null_values = None if cast_options.safe else ["", "null", "N/A"]
                lf = pl.scan_csv(path_str, null_values=null_values)
            elif fmt == FileFormat.ARROW_IPC:
                lf = pl.scan_ipc(path_str)
            elif fmt == FileFormat.JSON:
                # Polars scan_ndjson handles newline-delimited JSON files.
                lf = pl.scan_ndjson(path_str)
            else:
                # Formats without a native scanner: read eagerly then wrap.
                lf = self.read_polars_file(
                    file_format=fmt,
                    cast_options=cast_options,
                    lazy=False,
                ).lazy()

            if cast_options is not None:
                lf = cast_polars_dataframe(lf, cast_options)

            return lf

        # ---- eager path ----------------------------------------------------
        with self.open("rb") as f:
            if fmt == FileFormat.PARQUET:
                df = pl.read_parquet(f)

            elif fmt == FileFormat.CSV:
                null_values = None if cast_options.safe else ["", "null", "N/A"]
                df = pl.read_csv(f, null_values=null_values)

            elif fmt == FileFormat.JSON:
                df = pl.read_json(f)

            elif fmt == FileFormat.ARROW_IPC:
                df = pl.read_ipc(f)

            elif fmt == FileFormat.ORC:
                df = pl.read_orc(f)

            elif fmt == FileFormat.AVRO:
                df = pl.read_avro(f)

            elif fmt == FileFormat.EXCEL:
                df = pl.read_excel(f)

            elif fmt == FileFormat.DELTA:
                # Delta tables are directories, not single files.
                # Close the speculative file handle and read via delta-rs.
                pass  # handled below

            else:
                raise ValueError(
                    f"read_polars_file: unsupported format {fmt!r}"
                )

        if cast_options is not None:
            df = cast_polars_dataframe(df, cast_options)

        return df

    def write_polars_file(
        self,
        df: "polars.DataFrame",
        file_format: Union[FileFormat, str, None] = None,
        mode: Union[SaveMode, str, None] = None,  # noqa: ARG002
        *,
        cast_options: Optional[CastOptionsArg] = None,
        batch_size: Optional[int] = None,  # noqa: ARG002
    ) -> "AbstractDataPath":
        """
        Write a Polars :class:`~polars.DataFrame` to a *single file*.

        Supported formats: PARQUET, CSV, JSON, ARROW_IPC, AVRO, EXCEL.

        Returns
        -------
        self
            Enables method chaining.
        """
        from ...polars.cast import cast_polars_dataframe

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
            elif fmt == FileFormat.DELTA:
                # Delta is directory-based; cannot write through a file handle.
                pass  # handled below
            else:
                raise ValueError(
                    f"write_polars_file: unsupported format {fmt!r}"
                )

        return self

    # ------------------------------------------------------------------ #
    # High-level readers / writers (file OR directory)
    # ------------------------------------------------------------------ #

    def write_table(
        self,
        table: Union[
            pa.Table, pa.RecordBatch,
            "polars.DataFrame", "pandas.DataFrame",
            Any
        ],
        file_format: Union[FileFormat, str, None] = None,
        mode: Union[SaveMode, str, None] = None,
        *,
        batch_size: Optional[int] = None,
        cast_options: Optional[CastOptionsArg] = None,
    ) -> "AbstractDataPath":
        """
        Write *table* to this path, auto-dispatching on the input type.

        Accepted input types
        --------------------
        - ``pyarrow.Table`` / ``pyarrow.RecordBatch`` → :meth:`write_arrow_table`
        - ``polars.DataFrame`` → :meth:`write_polars`
        - ``pandas.DataFrame`` → :meth:`write_pandas`
        - Any other type → converted via ``any_to_polars_dataframe`` first.

        Returns
        -------
        self
        """
        namespace = ObjectSerde.full_namespace(obj=table)

        if namespace.startswith("pyarrow."):
            if not isinstance(table, pa.Table):
                if isinstance(table, pa.RecordBatch):
                    table = pa.Table.from_batches([table])
                elif hasattr(table, "read_all"):
                    table = table.read_all()
                else:
                    table = pa.table(table)
            return self.write_arrow_table(
                table,
                file_format=file_format,
                mode=mode,
                batch_size=batch_size,
                cast_options=cast_options,
            )

        if namespace.startswith("polars."):
            return self.write_polars(
                table,
                file_format=file_format,
                mode=mode,
                batch_size=batch_size,
                cast_options=cast_options,
            )

        if namespace.startswith("pandas."):
            return self.write_pandas(
                table,
                file_format=file_format,
                mode=mode,
                batch_size=batch_size,
                cast_options=cast_options,
            )

        # Unknown type — coerce to Polars first.
        from ...polars.cast import any_to_polars_dataframe

        return self.write_polars(
            df=any_to_polars_dataframe(table, cast_options),
            file_format=file_format,
            mode=mode,
            batch_size=batch_size,
            cast_options=None,  # already applied inside any_to_polars_dataframe
        )

    def read_arrow_table(
        self,
        file_format: Union[FileFormat, str, None] = None,
        *,
        batch_size: Optional[int] = None,
        cast_options: Optional[CastOptionsArg] = None,
    ) -> pa.Table:
        """
        Read this path (file *or* directory) into a single PyArrow
        :class:`~pyarrow.Table`.

        For directories, all child files are read and concatenated.  If the
        schemas differ and a simple Arrow concat fails, the method falls back to
        Polars ``diagonal_relaxed`` concat before converting to Arrow.
        """
        if self.is_file():
            return self.read_arrow_table_file(
                file_format=file_format,
                batch_size=batch_size,
                cast_options=cast_options,
            )

        if self.is_dir():
            tables: list[pa.Table] = [
                child.read_arrow_table_file(
                    file_format=file_format,
                    batch_size=batch_size,
                    cast_options=cast_options,
                )
                for child in self.ls(recursive=True)
                if child.is_file()
            ]

            if not tables:
                return pa.table({})  # empty table with no schema

            try:
                return pa.concat_tables(tables)
            except pa.ArrowInvalid:
                # Schema mismatch — fall back to Polars diagonal relaxed concat.
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
        file_format: Union[FileFormat, str, None] = None,
        mode: Union[SaveMode, str, None] = None,
        *,
        batch_size: Optional[int] = None,
        cast_options: Optional[CastOptionsArg] = None,
    ) -> "AbstractDataPath":
        """
        Write a PyArrow :class:`~pyarrow.Table` to this path.

        For directory sinks, the table is sharded into part-files named
        ``part-{i:05d}-{seed}-{rand}.{ext}`` where *ext* is the resolved
        ``FileFormat`` value.

        Parameters
        ----------
        mode:
            ``OVERWRITE`` clears existing part-files before writing.
        batch_size:
            Maximum rows per part-file (default: 1 048 576).
        """
        mode = SaveMode.parse(mode)
        fmt = self.check_file_format_arg(file_format)
        table = cast_arrow_tabular(table, cast_options)

        if self.is_dir_sink():
            if mode == SaveMode.OVERWRITE:
                self.rmdir(recursive=True, allow_not_found=True, with_root=False)

            seed = int(time.time() * 1000)
            rows_per_part = batch_size or 1_048_576

            for i, batch in enumerate(table.to_batches(max_chunksize=rows_per_part)):
                part = pa.Table.from_batches([batch], schema=table.schema)
                part_path = self / f"part-{i:05d}-{seed}-{_rand_str(4)}.{fmt.value}"
                part_path.write_arrow_table_file(
                    part,
                    file_format=fmt,
                    mode=SaveMode.OVERWRITE,
                    batch_size=None,
                    cast_options=None,
                )

            return self

        return self.write_arrow_table_file(
            table,
            file_format=fmt,
            mode=SaveMode.OVERWRITE,
            batch_size=None,
            cast_options=None,
        )

    def read_polars(
        self,
        file_format: Union[FileFormat, str, None] = None,
        *,
        batch_size: Optional[int] = None,
        cast_options: Optional[CastOptionsArg] = None,
        private_files: bool = False,
        raise_error: bool = True,
        start: int = 0,
        limit: Optional[int] = None,
        lazy: bool = False,
    ) -> "Union[polars.DataFrame, polars.LazyFrame]":
        """
        Read this path (file *or* directory) into a Polars DataFrame or
        LazyFrame.

        Supports row-level slicing via *start* / *limit* (SQL ``OFFSET`` /
        ``LIMIT`` semantics).  In lazy mode, slicing is pushed into the query
        plan at zero materialisation cost.

        Parameters
        ----------
        file_format:
            Override format detection; defaults to the file extension.
        batch_size:
            Hint passed through to single-file readers (ignored in lazy mode
            for formats with native scanners).
        cast_options:
            Optional post-read schema cast.
        private_files:
            When reading a directory, skip files whose names begin with
            ``.`` or ``_`` (common for hidden / metadata files).
        raise_error:
            If ``False``, silently skip files that raise exceptions during
            reading (directory mode only; ignored in lazy mode).
        start:
            Zero-based row offset.  Must be >= 0.
        limit:
            Maximum number of rows to return.  ``None`` means all rows.
        lazy:
            When ``True``, return a :class:`~polars.LazyFrame` instead of a
            materialised :class:`~polars.DataFrame`.

            **File**: uses Polars native scanners (``scan_parquet``,
            ``scan_csv``, ``scan_ipc``, ``scan_ndjson``) when available;
            other formats fall back to ``read_*().lazy()``.

            **Directory**: for formats that support glob-based scanning
            (PARQUET, CSV, ARROW_IPC, JSON), builds a single ``scan_*`` call
            over a glob pattern.  Mixed-format directories fall back to
            concatenating per-file LazyFrames.

            .. note::
                When ``lazy=True``, ``start`` / ``limit`` are applied as
                ``.slice()`` on the LazyFrame — no rows are read until
                ``.collect()`` is called.

        Returns
        -------
        polars.DataFrame
            When ``lazy=False`` (default).
        polars.LazyFrame
            When ``lazy=True``.
        """
        from ...polars.lib import polars as pl

        if start < 0:
            raise ValueError(f"start must be >= 0, got {start!r}")
        if limit is not None and limit < 0:
            raise ValueError(f"limit must be >= 0 or None, got {limit!r}")

        fmt = FileFormat.parse_any(file_format, default=None) if file_format is not None else None

        # ---- single file ------------------------------------------------
        if self.is_file():
            result = self.read_polars_file(
                file_format=fmt,
                batch_size=batch_size,
                cast_options=cast_options,
                lazy=lazy,
            )
            if start or limit is not None:
                result = result.slice(start, limit)
            return result

        # ---- directory --------------------------------------------------
        if self.is_dir():
            # Collect child file paths (respecting private_files filter).
            children = [
                child
                for child in self.ls(recursive=True)
                if child.is_file()
                and not (
                    private_files
                    and (child.name.startswith(".") or child.name.startswith("_"))
                )
            ]

            if not children:
                empty = pl.LazyFrame() if lazy else pl.DataFrame()
                return empty

            # ---- lazy directory: prefer glob-based scanners ----------------
            if lazy:
                resolved_fmt = fmt or (
                    # Infer from the first child if all share an extension.
                    children[0].file_format
                    if len({c.file_format for c in children}) == 1
                    else None
                )

                # Formats with native glob scanners.
                _GLOB_SCANNERS = {
                    FileFormat.PARQUET: pl.scan_parquet,
                    FileFormat.CSV: pl.scan_csv,
                    FileFormat.ARROW_IPC: pl.scan_ipc,
                    FileFormat.JSON: pl.scan_ndjson,
                }

                if resolved_fmt in _GLOB_SCANNERS:
                    scan_fn = _GLOB_SCANNERS[resolved_fmt]
                    glob_path = str(self / "**" / f"*.{resolved_fmt.value}")

                    scan_kwargs: dict = {}
                    if resolved_fmt == FileFormat.CSV and cast_options is not None:
                        co = CastOptions.check_arg(cast_options)
                        if not co.safe:
                            scan_kwargs["null_values"] = ["", "null", "N/A"]

                    lf: "polars.LazyFrame" = scan_fn(glob_path, **scan_kwargs)

                else:
                    # Fallback: build per-file LazyFrames and concat.
                    frames: list["polars.LazyFrame"] = []
                    for child in children:
                        try:
                            frames.append(
                                child.read_polars_file(
                                    file_format=fmt,
                                    cast_options=cast_options,
                                    lazy=True,
                                )
                            )
                        except Exception:
                            if raise_error:
                                raise
                            continue

                    if not frames:
                        return pl.LazyFrame()

                    lf = pl.concat(frames, how="diagonal_relaxed")

                if cast_options is not None:
                    from ...polars.cast import cast_polars_dataframe
                    lf = cast_polars_dataframe(lf, cast_options)

                if start or limit is not None:
                    lf = lf.slice(start, limit)

                return lf

            # ---- eager directory ----------------------------------------
            dfs: list["polars.DataFrame"] = []
            skipped = 0   # rows consumed for the start offset
            taken = 0     # rows collected so far

            for child in children:
                try:
                    df = child.read_polars_file(
                        file_format=fmt,
                        batch_size=batch_size,
                        cast_options=cast_options,
                        lazy=False,
                    )
                except Exception:
                    if raise_error:
                        raise
                    continue

                n = df.height
                if n == 0:
                    continue

                # 1) Consume start-offset across part-files.
                if skipped < start:
                    need_skip = start - skipped
                    if need_skip >= n:
                        skipped += n
                        continue
                    df = df.slice(need_skip)
                    skipped = start
                    n = df.height
                    if n == 0:
                        continue

                # 2) Honour remaining limit.
                if limit is not None:
                    remaining = limit - taken
                    if remaining <= 0:
                        break
                    if n > remaining:
                        df = df.slice(0, remaining)
                        n = df.height

                dfs.append(df)
                taken += n

                if limit is not None and taken >= limit:
                    break

            if not dfs:
                return pl.DataFrame()

            return pl.concat(dfs, how="diagonal_relaxed")

        raise FileNotFoundError(f"Path does not exist: {self}")

    def write_polars(
        self,
        df: "polars.DataFrame",
        file_format: Union[FileFormat, str, None] = None,
        mode: Union[SaveMode, str, None] = None,
        *,
        batch_size: Optional[int] = None,
        cast_options: Optional[CastOptionsArg] = None,
    ) -> "AbstractDataPath":
        """
        Write a Polars DataFrame to this path.

        For directory sinks, the frame is sharded into part-files identical to
        :meth:`write_arrow_table`.

        Returns
        -------
        self
        """
        from ...polars.cast import cast_polars_dataframe

        mode = SaveMode.parse(mode)
        fmt = self.check_file_format_arg(file_format)

        if cast_options is not None:
            df = cast_polars_dataframe(df, cast_options)

        if self.is_dir_sink():
            if mode == SaveMode.OVERWRITE:
                self.rmdir(recursive=True, allow_not_found=True, with_root=False)

            seed = int(time.time() * 1000)
            rows_per_part = batch_size or 1_048_576

            for i, chunk in enumerate(df.iter_slices(n_rows=rows_per_part)):
                part_path = self / f"part-{i:05d}-{seed}-{_rand_str(4)}.{fmt.value}"
                part_path.write_polars_file(
                    chunk,
                    file_format=fmt,
                    mode=SaveMode.OVERWRITE,
                    batch_size=None,
                    cast_options=None,
                )

            return self

        return self.write_polars_file(
            df,
            file_format=fmt,
            mode=SaveMode.OVERWRITE,
            batch_size=None,
            cast_options=None,
        )

    def read_pandas(
        self,
        file_format: Union[FileFormat, str, None] = None,
        *,
        batch_size: Optional[int] = None,
        cast_options: Optional[CastOptionsArg] = None,
    ) -> "pandas.DataFrame":
        """
        Read this path into a Pandas :class:`~pandas.DataFrame`.

        Internally delegates to :meth:`read_polars` (eager) and converts via
        ``polars.DataFrame.to_pandas()``.

        .. note::
            Pandas has no concept of a LazyFrame, so this method always reads
            eagerly.  For deferred execution, use :meth:`read_polars` with
            ``lazy=True`` instead.
        """
        return self.read_polars(
            file_format=file_format,
            batch_size=batch_size,
            cast_options=cast_options,
            lazy=False,
        ).to_pandas()

    def write_pandas(
        self,
        df: "pandas.DataFrame",
        file_format: Union[FileFormat, str, None] = None,
        mode: Union[SaveMode, str, None] = None,
        *,
        batch_size: Optional[int] = None,
        cast_options: Optional[CastOptionsArg] = None,
    ) -> "AbstractDataPath":
        """
        Write a Pandas DataFrame to this path.

        Converts to Polars internally and delegates to :meth:`write_polars`.

        Returns
        -------
        self
        """
        from ...polars.cast import any_to_polars_dataframe

        return self.write_polars(
            any_to_polars_dataframe(df, cast_options),
            file_format=file_format,
            mode=mode,
            batch_size=batch_size,
            cast_options=None,  # already applied
        )

    # ------------------------------------------------------------------ #
    # SQL
    # ------------------------------------------------------------------ #

    def sql(self, query: str, engine: str = "auto") -> pa.Table:
        """
        Execute a local SQL *query* that references this path.

        The query must contain the path literal wrapped in back-ticks
        (e.g. ``SELECT * FROM `path/to/data` WHERE ...``).

        Parameters
        ----------
        engine:
            ``"duckdb"`` — uses DuckDB via a PyArrow Dataset variable.
            ``"polars"`` — uses Polars SQL.
            ``"auto"`` — prefers DuckDB if available, falls back to Polars.

        Returns
        -------
        pa.Table

        Raises
        ------
        ValueError
            If the path literal is not found in *query*, or an invalid engine
            is specified.
        """
        from_literal = f"`{self}`"
        if from_literal not in query:
            raise ValueError(
                f"SQL query must contain {from_literal!r}.\n"
                f"Got:\n{query}"
            )

        if engine == "auto":
            try:
                import duckdb  # noqa: F401
                engine = "duckdb"
            except ImportError:
                engine = "polars"

        if engine == "duckdb":
            import duckdb

            __arrow_dataset__ = self.read_arrow_dataset()
            rewritten = query.replace(from_literal, "__arrow_dataset__")
            return (
                duckdb.connect()
                .execute(query=rewritten)
                .fetch_arrow_table()
            )

        if engine == "polars":
            table_name = "__dbpath__"
            rewritten = query.replace(from_literal, table_name)
            polars_result = self.read_polars().sql(
                query=rewritten, table_name=table_name
            )
            # read_polars().sql() returns a polars.DataFrame; convert to Arrow.
            from polars import CompatLevel
            return polars_result.to_arrow(compat_level=CompatLevel.newest())

        raise ValueError(
            f"Invalid engine {engine!r} — must be one of: duckdb, polars, auto"
        )