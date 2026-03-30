"""Media I/O abstraction layer for :class:`~yggdrasil.io.buffer.BytesIO`.

Each concrete :class:`MediaIO` subclass handles reading and writing a specific
columnar or container format (Parquet, JSON, Arrow IPC, ZIP). The base class
provides:

* **Transparent compression** — when :attr:`media_type` carries a
  :class:`~yggdrasil.io.enums.Codec` (e.g. ``application/json+gzip``),
  reads automatically decompress and writes automatically compress, so
  callers never need to wrap in manual ``decompress()`` / ``compress()``
  calls.
* **Generic adapters** — :meth:`read_pylist`, :meth:`read_pydict`,
  :meth:`read_pandas_frame`, :meth:`read_polars_frame` (and their write
  counterparts) are all routed through the Arrow table read/write path.
* **Batched iteration** — every read method accepts an optional
  ``batch_size`` parameter. When set to a positive integer the method
  returns an ``Iterator`` of chunks instead of a single object, enabling
  memory-efficient streaming over large datasets.

Subclasses only need to implement three methods:

* ``check_options(options, **kwargs) → O`` — validate/merge caller options.
* ``_read_arrow_batches(options) → Iterator[pyarrow.RecordBatch]`` — yield
  record batches from the *uncompressed* buffer.
* ``_write_arrow_batches(batches, schema, options)`` — consume an iterator
  of record batches and write into the *uncompressed* buffer.

Typical usage::

    buf = BytesIO(raw_gzipped_parquet_bytes)
    buf.set_media_type(MediaType("application/vnd.apache.parquet+gzip"), safe=False)
    mio = buf.media_io()                     # returns ParquetIO with auto-decompress
    table = mio.read_arrow_table()           # decompresses then reads

    # Batched iteration
    for batch in mio.read_arrow_table(batch_size=1000):
        process(batch)
"""
from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Iterator, Optional, Sequence, TypeVar, Union

import pyarrow as pa
from yggdrasil.arrow.cast import cast_arrow_tabular
from yggdrasil.io import MimeTypes
from yggdrasil.io.enums import Codec, MediaType, MimeType, SaveMode
from yggdrasil.pickle.serde import ObjectSerde

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
    """Abstract base for format-specific I/O on a :class:`BytesIO` buffer."""

    media_type: MediaType
    buffer: BytesIO

    # ------------------------------------------------------------------
    # Codec helpers (private)
    # ------------------------------------------------------------------

    @property
    def codec(self) -> Codec | None:
        """Return the transport compression codec, or ``None``."""
        return self.media_type.codec if self.media_type else None

    def _decompressed_buffer(self) -> tuple[BytesIO, bool]:
        """Return ``(buffer, was_decompressed)``.

        If the media type has a codec the buffer is decompressed into a
        **copy** (the original buffer is not mutated). The caller must
        close the copy when done.
        """
        codec = self.codec
        if codec is not None and self.buffer.size > 0:
            return self.buffer.decompress(codec=codec, copy=True), True
        return self.buffer, False

    def _compress_into_buffer(self, plain: BytesIO) -> None:
        """Compress *plain* with :attr:`codec` and replace :attr:`buffer` contents."""
        codec = self.codec
        if codec is not None:
            compressed = plain.compress(codec=codec, copy=True)
            self.buffer.replace_with_payload(compressed.to_bytes())
        else:
            self.buffer.replace_with_payload(plain.to_bytes())

    # ------------------------------------------------------------------
    # Cast helpers (private)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_cast(
        data: "pyarrow.Table | pyarrow.RecordBatch",
        *,
        options: O,
    ) -> "pyarrow.Table | pyarrow.RecordBatch":
        """Apply configured Arrow cast when requested."""
        cast = options.cast
        if cast is None:
            return data

        try:
            return cast_arrow_tabular(data, options=cast)
        except Exception:
            if options.raise_error:
                raise
            return data

    # ------------------------------------------------------------------
    # Iterable inference helpers (private)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_nonstring_iterable(obj: Any) -> bool:
        """Return True for iterables not handled by dedicated branches."""
        return (
            isinstance(obj, Iterable)
            and not isinstance(obj, (str, bytes, bytearray, dict, list, pa.Table))
        )

    @staticmethod
    def _peek_iterable(obj: Iterable[Any]) -> tuple[Any, Iterator[Any]]:
        """Return first item and a rebuilt iterator including that first item."""
        iterator = iter(obj)
        first = next(iterator)
        return first, itertools.chain((first,), iterator)

    @classmethod
    def _table_from_tabular_iterable(cls, obj: Iterable[Any]) -> "pyarrow.Table":
        """Infer an Arrow table from an iterable of supported tabular objects."""
        first, iterator = cls._peek_iterable(obj)

        if isinstance(first, dict):
            return pa.Table.from_pylist(list(iterator)) # noqa

        if isinstance(first, pa.RecordBatch):
            return pa.Table.from_batches(list(iterator))

        if isinstance(first, pa.Table):
            tables = list(iterator)
            if not tables:
                return pa.table({})
            return pa.concat_tables(tables, promote_options="default")

        raise TypeError(
            "Unsupported iterable for write_table: expected an iterable of "
            "dict, pyarrow.RecordBatch, or pyarrow.Table, got first item "
            f"of type {type(first)!r}"
        )

    @classmethod
    def _iter_to_batches(
        cls,
        obj: Iterable[Any],
    ) -> tuple[Iterator["pyarrow.RecordBatch"], "pyarrow.Schema"]:
        """Infer a batch iterator and schema from an iterable of tabular objects."""
        first, iterator = cls._peek_iterable(obj)

        if isinstance(first, dict):
            table = pa.Table.from_pylist(list(iterator)) # noqa
            return iter(table.to_batches()), table.schema

        if isinstance(first, pa.RecordBatch):
            return iterator, first.schema

        if isinstance(first, pa.Table):
            def batch_iter() -> Iterator["pyarrow.RecordBatch"]:
                for tb in iterator:
                    yield from tb.to_batches()

            return batch_iter(), first.schema

        raise TypeError(
            "Unsupported iterable for write_table: expected an iterable of "
            "dict, pyarrow.RecordBatch, or pyarrow.Table, got first item "
            f"of type {type(first)!r}"
        )

    def _write_batches_iterable(
        self,
        obj: Iterable[Any],
        *,
        options: O,
    ) -> None:
        """Write an iterable of tabular chunks to the buffer."""
        try:
            batches, schema = self._iter_to_batches(obj)
        except StopIteration:
            self._write_single_table(pa.table({}), options)
            return

        codec = self.codec
        if codec is not None:
            plain = BytesIO(config=self.buffer.config)
            orig_buffer = self.buffer
            self.buffer = plain
            try:
                self._write_arrow_batches(
                    batches=batches,
                    schema=schema,
                    options=options,
                )
            finally:
                self.buffer = orig_buffer
            self._compress_into_buffer(plain)
            plain.close()
        else:
            self._write_arrow_batches(
                batches=batches,
                schema=schema,
                options=options,
            )

    # ------------------------------------------------------------------
    # Abstract protocol
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def check_options(cls, options: Optional[O], *args, **kwargs) -> O:
        """Validate and merge caller-supplied options into a concrete instance."""

    def read_arrow_batches(
        self,
        *args,
        options: Optional[O] = None,
        **media_options,
    ) -> Iterator["pyarrow.RecordBatch"]:
        """Yield Arrow record batches, transparently handling decompression."""
        resolved = self.check_options(
            options=options, *args, **media_options
        )

        buf, decompressed = self._decompressed_buffer()
        orig_buffer = self.buffer
        try:
            if decompressed:
                self.buffer = buf

            for batch in self._read_arrow_batches(options=resolved):
                casted = self._apply_cast(batch, options=resolved)
                if isinstance(casted, pa.Table):
                    yield from casted.to_batches()
                else:
                    yield casted
        finally:
            if decompressed:
                self.buffer = orig_buffer
                buf.close()

    @abstractmethod
    def _read_arrow_batches(self, *, options: O) -> Iterator["pyarrow.RecordBatch"]:
        """Yield record batches from the **uncompressed** buffer."""

    def read_polars_frames(
        self,
        *args,
        options: Optional[O] = None,
        **media_options,
    ) -> Iterator["polars.DataFrame | polars.LazyFrame"]:
        from yggdrasil.polars.lib import polars as _pl

        resolved = self.check_options(
            options=options, *args, **media_options
        )
        for batch in self.read_arrow_batches(options=resolved):
            df = _pl.from_arrow(batch, rechunk=False)
            yield df.lazy() if resolved.lazy else df

    @abstractmethod
    def _write_arrow_batches(
        self,
        *,
        batches: Iterator["pyarrow.RecordBatch"],
        schema: "pyarrow.Schema",
        options: O,
    ) -> None:
        """Consume record batches and write into the **uncompressed** buffer."""

    # ------------------------------------------------------------------
    # Internal helpers: table ↔ batches bridge
    # ------------------------------------------------------------------

    def _read_table_from_batches(self, *, options: O) -> "pyarrow.Table":
        """Collect all batches from :meth:`read_arrow_batches` into one table."""
        batches = list(self.read_arrow_batches(options=options))
        if not batches:
            return pa.table({})
        return pa.Table.from_batches(batches)

    def _write_table_as_batches(self, *, table: "pyarrow.Table", options: O) -> None:
        """Convert *table* to batches and forward to :meth:`_write_arrow_batches`."""
        casted = options.cast.cast_arrow(table)

        self._write_arrow_batches(
            batches=iter(casted.to_batches()),
            schema=casted.schema,
            options=options.with_cast(None),
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def make(
        cls,
        buffer: BytesIO,
        media: MediaType | MimeType | str,
    ) -> "MediaIO[MediaOptions]":
        """Create the appropriate :class:`MediaIO` subclass for *media*."""
        media = MediaType.parse(media)
        mt = media.mime_type
        buffer.set_media_type(media, safe=False)

        if mt is MimeTypes.PARQUET:
            from .parquet_io import ParquetIO
            return ParquetIO(media_type=media, buffer=buffer)
        if mt is MimeTypes.JSON or mt is MimeTypes.NDJSON:
            from .json_io import JsonIO
            return JsonIO(media_type=media, buffer=buffer)
        if mt is MimeTypes.ZIP:
            from .zip_io import ZipIO
            return ZipIO(media_type=media, buffer=buffer)
        if mt is MimeTypes.ARROW_IPC:
            from .arrow_ipc_io import IPCIO
            return IPCIO(media_type=media, buffer=buffer)

        raise NotImplementedError(f"Cannot create media IO for {media!r}")

    # ------------------------------------------------------------------
    # Write guard
    # ------------------------------------------------------------------

    def skip_write(self, mode: SaveMode) -> bool:
        """Return ``True`` when to write should be skipped."""
        if self.buffer.size > 0:
            if mode == SaveMode.IGNORE:
                return True
            if mode == SaveMode.ERROR_IF_EXISTS:
                raise IOError(
                    f"Cannot write in already existing {self.buffer!r} "
                    f"with save mode {SaveMode.ERROR_IF_EXISTS.value}"
                )
        return False

    # ------------------------------------------------------------------
    # Generic write dispatcher
    # ------------------------------------------------------------------

    def write_table(
        self,
        obj: Any,
        *,
        options: O | None = None,
        **option_kwargs,
    ) -> None:
        """Write a supported tabular object to the buffer.

        Supported inputs
        ----------------
        - pyarrow.Table
        - Iterator[pyarrow.Table]
        - Iterator[pyarrow.RecordBatch]
        - pandas.DataFrame
        - polars.DataFrame
        - polars.LazyFrame
        - list[dict]
        - Iterator[dict]
        - dict[str, list]
        """
        if isinstance(obj, pa.Table):
            return self.write_arrow_table(
                table=obj,
                options=options,
                **option_kwargs,
            )
        elif isinstance(obj, pa.RecordBatch):
            return self.write_arrow_table(
                pa.Table.from_batches([obj]),
                options=options,
                **option_kwargs,
            )

        ns, _ = ObjectSerde.module_and_name(obj)

        if ns.startswith("polars"):
            return self.write_polars_frame(
                obj,
                options=options,
                **option_kwargs,
            )
        elif ns.startswith("pandas"):
            return self.write_pandas_frame(
                obj,
                options=options,
                **option_kwargs
            )

        if isinstance(obj, list):
            if obj and not all(isinstance(row, dict) for row in obj):
                raise TypeError(
                    "write_table(list) expects list[dict], "
                    f"got list[{type(obj[0]).__name__}]"
                )
            self.write_pylist(
                data=obj,
                options=options,
                **option_kwargs,
            )
            return

        if isinstance(obj, dict):
            self.write_pydict(
                data=obj,
                options=options,
                **option_kwargs,
            )
            return

        if self._is_nonstring_iterable(obj):
            resolved = self.check_options(options=options, **option_kwargs)
            if self.skip_write(mode=resolved.mode):
                return

            if resolved.mode in (SaveMode.APPEND, SaveMode.UPSERT):
                table = self._table_from_tabular_iterable(obj)
                existing = self._read_existing_table()
                table = self._apply_save_mode(
                    existing,
                    table,
                    mode=resolved.mode,
                    match_by=resolved.match_by,
                )
                self._write_single_table(table, resolved)
                return

            self._write_batches_iterable(obj, options=resolved)
            return

        ns = ObjectSerde.full_namespace(obj)

        if ns.startswith("pandas."):
            self.write_pandas_frame(
                frame=obj,
                options=options,
                **option_kwargs,
            )
            return

        if ns.startswith("polars."):
            self.write_polars_frame(
                frame=obj,
                options=options,
                **option_kwargs,
            )
            return

        raise TypeError(
            "Unsupported tabular object for write_table: "
            f"{type(obj)!r} (namespace={ns!r}). Expected one of: "
            "pyarrow.Table, Iterator[pyarrow.Table], Iterator[pyarrow.RecordBatch], "
            "pandas.DataFrame, polars.DataFrame, polars.LazyFrame, "
            "list[dict], Iterator[dict], dict[str, list]."
        )

    # ------------------------------------------------------------------
    # Save-mode helpers (private)
    # ------------------------------------------------------------------

    def _read_existing_table(self) -> "pyarrow.Table | None":
        """Read the current buffer contents as an Arrow table, or ``None``."""
        if self.buffer.size <= 0:
            return None
        try:
            table = self.read_arrow_table()
            return table if isinstance(table, pa.Table) else None
        except Exception:
            return None

    @staticmethod
    def _apply_save_mode(
        existing: "pyarrow.Table | None",
        incoming: "pyarrow.Table",
        *,
        mode: SaveMode,
        match_by: "Sequence[str] | None",
    ) -> "pyarrow.Table":
        """Combine *existing* and *incoming* tables according to *mode*."""
        if existing is None or existing.num_rows == 0:
            return incoming

        if mode in (SaveMode.AUTO, SaveMode.OVERWRITE, SaveMode.TRUNCATE):
            return incoming

        if mode == SaveMode.APPEND:
            return pa.concat_tables([existing, incoming], promote_options="default")

        if mode == SaveMode.UPSERT:
            if not match_by:
                raise ValueError(
                    "SaveMode.UPSERT requires match_by columns, got None/empty"
                )
            match_by = list(match_by)

            for col in match_by:
                if col not in existing.column_names:
                    raise ValueError(
                        f"UPSERT match_by column {col!r} not in existing table "
                        f"(columns: {existing.column_names})"
                    )
                if col not in incoming.column_names:
                    raise ValueError(
                        f"UPSERT match_by column {col!r} not in incoming table "
                        f"(columns: {incoming.column_names})"
                    )

            incoming_keys: set[tuple] = set()
            incoming_key_arrays = [
                incoming.column(c).to_pylist() for c in match_by
            ]
            for row_idx in range(incoming.num_rows):
                key = tuple(arr[row_idx] for arr in incoming_key_arrays)
                incoming_keys.add(key)

            existing_key_arrays = [
                existing.column(c).to_pylist() for c in match_by
            ]
            keep_mask: list[bool] = []
            for row_idx in range(existing.num_rows):
                key = tuple(arr[row_idx] for arr in existing_key_arrays)
                keep_mask.append(key not in incoming_keys)

            surviving = existing.filter(pa.array(keep_mask, type=pa.bool_()))

            return pa.concat_tables(
                [surviving, incoming],
                promote_options="default",
            )

        return incoming

    # ------------------------------------------------------------------
    # Batching helpers (private)
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_arrow_batches(
        table: "pyarrow.Table",
        batch_size: int,
    ) -> Iterator["pyarrow.Table"]:
        """Yield successive :class:`pyarrow.Table` slices of *batch_size* rows."""
        total = table.num_rows
        for offset in range(0, total, batch_size):
            yield table.slice(offset, min(batch_size, total - offset))

    # ------------------------------------------------------------------
    # Arrow (primary public API)
    # ------------------------------------------------------------------

    def read_arrow_table(
        self,
        *,
        options: O | None = None,
        **option_kwargs,
    ) -> Union["pyarrow.Table", Iterator["pyarrow.Table"]]:
        """Read the buffer into a :class:`pyarrow.Table`."""
        resolved = self.check_options(options=options, **option_kwargs)
        table = self._read_table_from_batches(options=resolved)

        bs = resolved.batch_size
        if bs and bs > 0:
            return self._iter_arrow_batches(table, bs)
        return table

    def write_arrow_table(
        self,
        table: "pyarrow.Table",
        *,
        options: O | None = None,
        **option_kwargs,
    ) -> None:
        """Write a :class:`pyarrow.Table` to the buffer."""
        resolved = self.check_options(
            options=options, **option_kwargs,
        )
        if self.skip_write(mode=resolved.mode):
            return

        if resolved.mode in (SaveMode.APPEND, SaveMode.UPSERT):
            existing = self._read_existing_table()
            table = self._apply_save_mode(
                existing,
                table,
                mode=resolved.mode,
                match_by=resolved.match_by,
            )

        self._write_single_table(table, resolved)

    def _write_single_table(
        self,
        table: "pyarrow.Table",
        options: O,
    ) -> None:
        """Write one table handling codec compression."""
        codec = self.codec
        if codec is not None:
            plain = BytesIO(config=self.buffer.config)
            orig_buffer = self.buffer
            self.buffer = plain
            try:
                self._write_table_as_batches(table=table, options=options)
            finally:
                self.buffer = orig_buffer
            self._compress_into_buffer(plain)
            plain.close()
        else:
            self._write_table_as_batches(table=table, options=options)

    # ------------------------------------------------------------------
    # Python dict / list convenience wrappers
    # ------------------------------------------------------------------

    def read_pylist(
        self,
        *,
        options: O | None = None,
        **option_kwargs,
    ) -> Union[list[dict], Iterator[list[dict]]]:
        """Read the buffer as a list of row dicts."""
        resolved = self.check_options(options=options, **option_kwargs)
        bs = resolved.batch_size
        if bs and bs > 0:
            return (
                chunk.to_pylist()
                for chunk in self.read_arrow_table(options=resolved)
            )
        return self.read_arrow_table(options=resolved).to_pylist()

    def write_pylist(
        self,
        data: list[dict],
        *,
        options: O | None = None,
        **option_kwargs,
    ) -> None:
        """Write a list of row dicts to the buffer."""
        tb = pa.Table.from_pylist(data) # noqa
        self.write_arrow_table(
            table=tb,
            options=options,
            **option_kwargs,
        )

    def read_pydict(
        self,
        *,
        options: O | None = None,
        **option_kwargs,
    ) -> Union[dict[str, list], Iterator[dict[str, list]]]:
        """Read the buffer as a column-oriented dict."""
        resolved = self.check_options(options=options, **option_kwargs)
        bs = resolved.batch_size
        if bs and bs > 0:
            return (
                chunk.to_pydict()
                for chunk in self.read_arrow_table(options=resolved)
            )
        return self.read_arrow_table(options=resolved).to_pydict()

    def write_pydict(
        self,
        data: dict[str, list],
        *,
        options: O | None = None,
        **option_kwargs,
    ) -> None:
        """Write a column-oriented dict to the buffer."""
        tb = pa.Table.from_pydict(data) # noqa
        self.write_arrow_table(
            table=tb,
            options=options,
            **option_kwargs,
        )

    # ------------------------------------------------------------------
    # Pandas / Polars convenience wrappers
    # ------------------------------------------------------------------

    def read_pandas_frame(
        self,
        *,
        options: O | None = None,
        **option_kwargs,
    ) -> Union["pandas.DataFrame", Iterator["pandas.DataFrame"]]:
        """Read the buffer as a :class:`pandas.DataFrame`."""
        resolved = self.check_options(options=options, **option_kwargs)
        bs = resolved.batch_size
        if bs and bs > 0:
            return (
                chunk.to_pandas()
                for chunk in self.read_arrow_table(options=resolved)
            )
        return self.read_arrow_table(options=resolved).to_pandas()

    def write_pandas_frame(
        self,
        frame: "pandas.DataFrame",
        *,
        options: O | None = None,
        **option_kwargs,
    ) -> None:
        """Write a :class:`pandas.DataFrame` to the buffer."""
        tb = pa.Table.from_pandas(
            frame,
            preserve_index=bool(frame.index.name),
        ) # noqa

        self.write_arrow_table(
            table=tb,
            options=options,
            **option_kwargs,
        )

    def read_polars_frame(
        self,
        *,
        options: O | None = None,
        **option_kwargs,
    ) -> Union[
        "polars.DataFrame",
        "polars.LazyFrame",
        Iterator["polars.DataFrame"],
    ]:
        """Read the buffer as a Polars frame."""
        from yggdrasil.polars.lib import polars as _pl

        resolved = self.check_options(options=options, **option_kwargs)
        bs = resolved.batch_size
        if bs and bs > 0:
            return (
                _pl.from_arrow(chunk, rechunk=False)
                for chunk in self.read_arrow_table(options=resolved)
            )

        tb = self.read_arrow_table(options=resolved)
        df = _pl.from_arrow(tb, rechunk=False)
        return df.lazy() if resolved.lazy else df

    def write_polars_frame(
        self,
        frame: "polars.DataFrame | polars.LazyFrame",
        *,
        options: O | None = None,
        **option_kwargs,
    ) -> None:
        """Write a Polars DataFrame or LazyFrame to the buffer."""
        from yggdrasil.polars.cast import polars_dataframe_to_arrow_table

        resolved = self.check_options(
            options=options, **option_kwargs,
        )

        tb = polars_dataframe_to_arrow_table(frame, resolved.cast)

        self.write_arrow_table(
            table=tb,
            options=options,
            **option_kwargs,
        )