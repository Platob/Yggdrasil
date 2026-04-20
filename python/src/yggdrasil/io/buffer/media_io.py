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
from collections.abc import Iterable, Iterator as AbcIterator, Generator as AbcGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Iterator, Optional, Sequence, TypeVar, Union, Generator

import polars as pl
import pyarrow as pa

from yggdrasil.io.enums import MimeTypes, Codec, MediaType, MimeType, SaveMode
from yggdrasil.pickle.serde import ObjectSerde
from .bytes_io import BytesIO
from .media_options import MediaOptions

if TYPE_CHECKING:
    import pandas
    import pyarrow

    from yggdrasil.data.cast.options import CastOptionsArg
    from yggdrasil.data.schema import Schema


O = TypeVar("O", bound=MediaOptions)

__all__ = ["MediaIO", "MediaOptions"]


@dataclass(slots=True)
class MediaIO(ABC, Generic[O]):
    """Abstract base for format-specific I/O on a :class:`BytesIO` buffer."""

    media_type: MediaType
    holder: BytesIO
    buffer: BytesIO | None = field(init=False, default=None)
    _dirty: bool = field(init=False, default=False)

    @property
    def opened(self) -> bool:
        return self.buffer is not None

    @property
    def closed(self) -> bool:
        return self.buffer is None

    def open(self) -> "MediaIO":
        if self.opened:
            raise RuntimeError(f"MediaIO {self!r} is already open")

        codec = self.codec
        if codec is None:
            # share the holder's storage; writes will mark _dirty
            self.buffer = self.holder
        else:
            self.buffer = self.holder.decompress(codec=codec, copy=True)
        return self

    def mark_dirty(self) -> None:
        """Call from _write_arrow_batches paths to signal flush-on-close."""
        self._dirty = True

    def close(self) -> None:
        if not self.opened:
            return

        try:
            # only write back if we actually mutated a detached buffer
            if self._dirty and self.buffer is not self.holder:
                self.holder = self.buffer.compress(codec=self.codec, copy=False)
        finally:
            self.buffer = None
            self._dirty = False

    def __enter__(self) -> "MediaIO":
        return self.open() if not self.opened else self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self):
        # best-effort only; never raise from __del__
        try:
            if self.opened:
                self.close()
        except Exception:
            pass

    def __repr__(self):
        opened: str = "opened" if self.opened else "closed"
        return f"{type(self).__name__}({self.media_type!r}, {self.holder!r}, {opened!r})"

    # ------------------------------------------------------------------
    # Codec helpers (private)
    # ------------------------------------------------------------------

    @property
    def codec(self) -> Codec | None:
        """Return the transport compression codec, or ``None``."""
        return self.media_type.codec if self.media_type else None

    @staticmethod
    def _is_path_input(obj: Any) -> bool:
        """Return True for ``str`` / ``Path`` / ``os.PathLike`` inputs."""
        if isinstance(obj, (str, Path)):
            return True
        return hasattr(obj, "__fspath__")

    @staticmethod
    def polars_to_arrow_table(df: "pl.DataFrame") -> "pyarrow.Table":
        """Convert a polars DataFrame to an Arrow table."""
        return df.rechunk().to_arrow(compat_level=pl.CompatLevel.newest())

    @staticmethod
    def iter_arrow_batches(
        tb: pa.Table | pa.RecordBatch | Iterator[pa.Table | pa.RecordBatch],
    ) -> Iterator[pa.RecordBatch]:
        """Iterate Arrow record batches from a wide range of tabular inputs.

        Dispatch order (important — do NOT reorder without care):

        1. Arrow-native types (Table, RecordBatch).
        2. Column-oriented dicts.
        3. Polars / pandas frames — these happen to be iterable, but
           iterating them yields column names or row scalars rather
           than batches. Must dispatch by namespace BEFORE the generic
           iterable fallback, or else ``pd.DataFrame`` recursively
           iterates strings → characters → infinite recursion.
        4. Lists / tuples — materialized containers that are NOT
           iterators. A list of row dicts is treated as bulk input
           (sparse-key normalized via :meth:`_normalize_records`);
           anything else is walked elementwise.
        5. Generic iterators — narrowed to actual iterators
           (``__next__`` defined) via :class:`collections.abc.Iterator`.
           This deliberately excludes strings, DataFrames, and other
           "iterable" objects that would cause misdispatch.
        6. Fallback TypeError.
        """
        # --- (1) Arrow-native ------------------------------------------
        if isinstance(tb, pa.Table):
            yield from tb.to_batches()
            return
        if isinstance(tb, pa.RecordBatch):
            yield tb
            return

        # --- (2) Column-oriented dict ----------------------------------
        # Only treat dicts as column-oriented when values look like
        # sequences. Row-shaped dicts (scalar values) would fail
        # pa.Table.from_pydict with a confusing error; detect and
        # reject explicitly.
        if isinstance(tb, dict):
            if tb and not all(
                isinstance(v, (list, tuple, pa.Array, pa.ChunkedArray))
                for v in tb.values()
            ):
                raise TypeError(
                    "iter_arrow_batches received a dict with non-sequence "
                    "values; did you mean a list of row dicts?"
                )
            yield from pa.Table.from_pydict(tb).to_batches()
            return

        # --- (3) Namespace dispatch (polars, pandas) BEFORE iterable ----
        ns, _ = ObjectSerde.module_and_name(tb)

        if ns.startswith("polars"):
            if isinstance(tb, pl.LazyFrame):
                tb = tb.collect()
            if isinstance(tb, pl.DataFrame):
                arrow_table = MediaIO.polars_to_arrow_table(tb)
                yield from arrow_table.to_batches()
                return
            raise TypeError(
                f"Unsupported polars object for iter_arrow_batches: {type(tb)!r}"
            )

        if ns.startswith("pandas"):
            import pandas
            if isinstance(tb, pandas.DataFrame):
                arrow_table = pa.Table.from_pandas(tb)
                yield from arrow_table.to_batches()
                return
            raise TypeError(
                f"Unsupported pandas object for iter_arrow_batches: {type(tb)!r}"
            )

        # --- (4) List / tuple (materialized, NOT iterator) --------------
        if isinstance(tb, (list, tuple)):
            if tb and all(isinstance(item, dict) for item in tb):
                # list[dict] → build a table with sparse-key normalization.
                yield from pa.Table.from_pylist(
                    MediaIO._normalize_records(tb)
                ).to_batches()
                return
            # Heterogeneous list of tables/batches/etc — recurse elementwise.
            for sub_tb in tb:
                yield from MediaIO.iter_arrow_batches(sub_tb)
            return

        # --- (5) Generic iterator (collections.abc.Iterator ONLY) --------
        # Catches generators, pyarrow ChunkedArray-based streams, and
        # anything else with __next__. Does NOT catch pandas/polars
        # DataFrames or strings — those either matched earlier or fall
        # through to the TypeError below.
        if isinstance(tb, AbcIterator):
            for sub_tb in tb:
                yield from MediaIO.iter_arrow_batches(sub_tb)
            return

        # --- (6) Nothing matched ----------------------------------------
        raise TypeError(
            f"Unsupported object for iter_arrow_batches: {type(tb)!r}"
        )

    # ------------------------------------------------------------------
    # Schema inspection
    # ------------------------------------------------------------------

    def collect_schema(self, full: bool = False) -> "Schema":
        """Return the yggdrasil :class:`Schema` without collecting all data.

        Delegates to :meth:`_collect_arrow_schema` to obtain the Arrow
        schema through the most efficient path the concrete format offers
        (Parquet footer, Arrow IPC header, CSV header row, first JSON
        record, first ZIP member, …), then wraps it as a yggdrasil
        :class:`~yggdrasil.data.schema.Schema`.

        When *full* is ``True``, container formats that expose multiple
        inner payloads (:class:`~yggdrasil.io.buffer.zip_io.ZipIO`,
        :class:`~yggdrasil.io.buffer.path_io.PathIO`) inspect every
        member/file and return the unified schema instead of stopping at
        the first one. Single-payload formats ignore the flag.
        """
        from yggdrasil.data.schema import Schema

        return Schema.from_arrow(self._collect_arrow_schema(full=full))

    def _collect_arrow_schema(self, full: bool = False) -> "pyarrow.Schema":
        """Return the Arrow schema without collecting all data.

        Default implementation consumes only the first record batch from
        :meth:`read_arrow_batches`. Subclasses should override with a
        metadata-only fast path when the format allows one. The *full*
        flag is accepted for API consistency; container subclasses use
        it to inspect every member instead of just the first.
        """
        del full
        iterator = self.read_arrow_batches()
        try:
            first = next(iterator)
        except StopIteration:
            return pa.schema([])
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        return first.schema

    # ------------------------------------------------------------------
    # Abstract protocol
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def check_options(cls, options: Optional[O], *args, **kwargs) -> O:
        """Validate and merge caller-supplied options into a concrete instance."""

    @abstractmethod
    def _read_arrow_batches(self, options: O) -> Iterator["pyarrow.RecordBatch"]:
        """Yield record batches from the **uncompressed** buffer."""

    @abstractmethod
    def _write_arrow_batches(
        self,
        batches: Iterator["pyarrow.RecordBatch"],
        options: O,
    ) -> None:
        """Consume record batches and write into the **uncompressed** buffer."""

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def make(
        cls,
        buffer: "BytesIO | str | Path | Any" = None,
        media: "MediaType | MimeType | str | None" = None,
    ) -> "MediaIO[MediaOptions]":
        if not isinstance(buffer, BytesIO):
            if media is None:
                media = MediaType.parse(buffer, default=MediaType(MimeTypes.OCTET_STREAM))
            buffer = BytesIO(buffer)

        if media is None:
            if buffer.media_type is None:
                raise ValueError(
                    "MediaIO.make requires a media type when the buffer has none set"
                )
            media = buffer.media_type

        media = MediaType.parse(media)
        mt = media.mime_type
        buffer.set_media_type(media, safe=False)

        if mt is MimeTypes.PARQUET:
            from .parquet_io import ParquetIO
            return ParquetIO(media_type=media, holder=buffer)
        if mt is MimeTypes.CSV or mt is MimeTypes.TSV:
            from .csv_io import CsvIO
            return CsvIO(media_type=media, holder=buffer)
        if mt is MimeTypes.JSON or mt is MimeTypes.NDJSON:
            from .json_io import JsonIO
            return JsonIO(media_type=media, holder=buffer)
        if mt is MimeTypes.XML:
            from .xml_io import XmlIO
            return XmlIO(media_type=media, holder=buffer)
        if mt is MimeTypes.ZIP:
            from .zip_io import ZipIO
            return ZipIO(media_type=media, holder=buffer)
        if mt is MimeTypes.ARROW_IPC:
            from .arrow_ipc_io import IPCIO
            return IPCIO(media_type=media, holder=buffer)
        if mt is MimeTypes.XLSX:
            from .xlsx_io import XlsxIO
            return XlsxIO(media_type=media, holder=buffer)

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

    # ==================================================================
    # PUBLIC API — each method resolves options once, then delegates to
    # the matching `_*` private method that takes only (payload, options).
    # ==================================================================

    # ------------------------------------------------------------------
    # Arrow batches (primary read path)
    # ------------------------------------------------------------------

    def read_arrow_batches(
        self,
        *,
        options: Optional[O] = None,
        columns: "Sequence[str] | None | Any" = ...,
        cast: "CastOptionsArg | Any" = ...,
        use_threads: "bool | Any" = ...,
        ignore_empty: "bool | Any" = ...,
        raise_error: "bool | Any" = ...,
        **option_kwargs,
    ) -> Iterator["pyarrow.RecordBatch"]:
        """Yield Arrow record batches, transparently handling decompression.

        Parameters
        ----------
        options:
            Pre-built options instance for this format. When ``None`` a fresh
            default is constructed and the explicit keyword arguments below
            are merged on top.
        columns:
            Optional column projection — only these columns are materialized
            when the format supports pushdown. Unknown names either raise or
            get dropped depending on the format.
        cast:
            Arrow cast configuration applied to each batch after decode.
        use_threads:
            Enable parallel Arrow decode where the format supports it.
        ignore_empty:
            Suppress zero-row batches so callers never see empty shells.
        raise_error:
            When ``True`` (default) cast failures propagate; when ``False`` the
            uncast batch is yielded instead.
        **option_kwargs:
            Format-specific knobs consumed by the concrete
            :meth:`check_options` implementation.
        """
        resolved = self.check_options(
            options=options,
            columns=columns,
            cast=cast,
            use_threads=use_threads,
            ignore_empty=ignore_empty,
            raise_error=raise_error,
            **option_kwargs,
        )
        yield from self._read_arrow_batches(resolved)

    # ------------------------------------------------------------------
    # Arrow table
    # ------------------------------------------------------------------

    def read_arrow_table(
        self,
        *,
        options: O | None = None,
        columns: "Sequence[str] | None | Any" = ...,
        cast: "CastOptionsArg | Any" = ...,
        use_threads: "bool | Any" = ...,
        ignore_empty: "bool | Any" = ...,
        raise_error: "bool | Any" = ...,
        batch_size: "int | None | Any" = ...,
        **option_kwargs,
    ) -> Union["pyarrow.Table", Iterator["pyarrow.Table"]]:
        """Read the buffer into a :class:`pyarrow.Table`.

        When *batch_size* is a positive integer, returns an iterator of
        :class:`pyarrow.Table` chunks of at most *batch_size* rows.
        """
        resolved = self.check_options(
            options=options,
            columns=columns,
            cast=cast,
            use_threads=use_threads,
            ignore_empty=ignore_empty,
            raise_error=raise_error,
            batch_size=batch_size,
            **option_kwargs,
        )
        return self._read_arrow_table(resolved)

    def _read_arrow_table(
        self,
        options: O,
    ) -> Union["pyarrow.Table", Iterator["pyarrow.Table"]]:
        batches = self._read_arrow_batches(options)

        if getattr(options, "batch_size", 0) and options.batch_size > 0:
            def iter_tables() -> Iterator["pa.Table"]:
                for batch in batches:
                    yield pa.Table.from_batches([batch])
            return iter_tables()

        tables = [pa.Table.from_batches([b]) for b in batches]
        if not tables:
            schema = options.cast.target_schema or options.cast.source_schema
            schema = schema.to_arrow_schema() if schema is None else pa.schema([])
            return schema.empty_table()

        return pa.concat_tables(tables, promote_options="permissive")

    def write_arrow_table(
        self,
        table: "pyarrow.Table",
        *,
        options: O | None = None,
        mode: "SaveMode | str | None | Any" = ...,
        match_by: "Sequence[str] | str | None | Any" = ...,
        cast: "CastOptionsArg | Any" = ...,
        use_threads: "bool | Any" = ...,
        raise_error: "bool | Any" = ...,
        **option_kwargs,
    ) -> None:
        """Write a :class:`pyarrow.Table` to the buffer."""
        resolved = self.check_options(
            options=options,
            mode=mode,
            match_by=match_by,
            cast=cast,
            use_threads=use_threads,
            raise_error=raise_error,
            **option_kwargs,
        )
        return self._write_arrow_table(table, resolved)

    def _write_arrow_table(
        self,
        table: "pyarrow.Table",
        options: O,
    ) -> None:
        return self._write_arrow_batches(
            self.iter_arrow_batches(table),
            options,
        )

    # ------------------------------------------------------------------
    # Generic write dispatcher
    # ------------------------------------------------------------------

    def write_table(
        self,
        obj: Any,
        *,
        options: O | None = None,
        mode: "SaveMode | str | None | Any" = ...,
        match_by: "Sequence[str] | str | None | Any" = ...,
        cast: "CastOptionsArg | Any" = ...,
        use_threads: "bool | Any" = ...,
        raise_error: "bool | Any" = ...,
        **option_kwargs,
    ) -> None:
        """Write a supported tabular object to the buffer.

        Supported inputs
        ----------------
        - ``pyarrow.Table`` / ``pyarrow.RecordBatch``
        - ``Iterable[pyarrow.Table]`` / ``Iterable[pyarrow.RecordBatch]``
        - ``pandas.DataFrame``
        - ``polars.DataFrame`` / ``polars.LazyFrame``
        - ``list[dict]`` or any ``Iterable[dict]`` (including generators)
        - ``dict[str, list]`` (column-oriented)
        - ``str`` / :class:`pathlib.Path` / ``os.PathLike``
        """
        resolved = self.check_options(
            options=options,
            mode=mode,
            match_by=match_by,
            cast=cast,
            use_threads=use_threads,
            raise_error=raise_error,
            **option_kwargs,
        )
        return self._write_table(obj, resolved)

    def _write_table(
        self,
        obj: Any,
        options: O,
    ) -> None:
        # Path-like: stream-convert from source format into this buffer.
        if self._is_path_input(obj):
            from .local_path_io import LocalPathIO

            with LocalPathIO.make(obj) as source:
                return self._write_arrow_batches(
                    source.read_arrow_batches(options=options),
                    options,
                )

        # Arrow natives.
        if isinstance(obj, pa.Table):
            return self._write_arrow_table(obj, options)
        if isinstance(obj, pa.RecordBatch):
            return self._write_arrow_table(pa.Table.from_batches([obj]), options)

        # Polars / pandas frames (non-iterator).
        ns, _ = ObjectSerde.module_and_name(obj)

        if ns.startswith("polars"):
            return self._write_polars_frame(obj, options)
        if ns.startswith("pandas"):
            return self._write_pandas_frame(obj, options)

        # list[dict].
        if isinstance(obj, list):
            if obj:
                bad = next(
                    (row for row in obj if not isinstance(row, (dict, type(None)))),
                    None,
                )
                if bad is not None:
                    raise TypeError(
                        "write_table(list) expects list[dict], "
                        f"got element of type {type(bad).__name__}"
                    )
            return self._write_pylist(obj, options)

        # dict[str, list].
        if isinstance(obj, dict):
            return self._write_pydict(obj, options)

        # Generic iterator: peek to decide the concrete path.
        if isinstance(obj, Iterator):
            first = next(obj, None)
            if first is None:
                return None

            chained = itertools.chain((first,), obj)
            first_ns, _ = ObjectSerde.module_and_name(first)

            if first_ns.startswith("polars"):
                return self._write_polars_frames(chained, options)

            # Everything else (Arrow tables/batches, pandas frames, dicts)
            # is handled by iter_arrow_batches.
            return self._write_arrow_batches(
                self.iter_arrow_batches(chained),
                options,
            )

        raise TypeError(
            "Unsupported tabular object for write_table: "
            f"{type(obj)!r} (namespace={ns!r}). Expected one of: "
            "pyarrow.Table, Iterator[pyarrow.Table], Iterator[pyarrow.RecordBatch], "
            "pandas.DataFrame, polars.DataFrame, polars.LazyFrame, "
            "list[dict], Iterator[dict], dict[str, list]."
        )

    # ------------------------------------------------------------------
    # Python list[dict]
    # ------------------------------------------------------------------

    def read_pylist(
        self,
        *,
        options: O | None = None,
        columns: "Sequence[str] | None | Any" = ...,
        cast: "CastOptionsArg | Any" = ...,
        use_threads: "bool | Any" = ...,
        ignore_empty: "bool | Any" = ...,
        raise_error: "bool | Any" = ...,
        batch_size: "int | None | Any" = ...,
        **option_kwargs,
    ) -> Union[list[dict], Iterator[list[dict]]]:
        """Read the buffer as a list of row dicts (or an iterator of chunks)."""
        resolved = self.check_options(
            options=options,
            columns=columns,
            cast=cast,
            use_threads=use_threads,
            ignore_empty=ignore_empty,
            raise_error=raise_error,
            batch_size=batch_size,
            **option_kwargs,
        )
        return self._read_pylist(resolved)

    def _read_pylist(
        self,
        options: O,
    ) -> Union[list[dict], Iterator[list[dict]]]:
        table_or_iter = self._read_arrow_table(options)

        if isinstance(table_or_iter, pa.Table):
            return table_or_iter.to_pylist(maps_as_pydicts=True)

        def iter_pylists() -> Iterator[list[dict]]:
            for tb in table_or_iter:
                yield tb.to_pylist(maps_as_pydicts=True)
        return iter_pylists()

    def write_pylist(
        self,
        data: Iterable[dict],
        *,
        options: O | None = None,
        mode: "SaveMode | str | None | Any" = ...,
        match_by: "Sequence[str] | str | None | Any" = ...,
        cast: "CastOptionsArg | Any" = ...,
        use_threads: "bool | Any" = ...,
        raise_error: "bool | Any" = ...,
        **option_kwargs,
    ) -> None:
        """Write a list (or iterable) of row dicts to the buffer.

        Sparse/heterogeneous keys are normalized against the union of keys
        seen across rows; missing entries are backfilled with ``None``.
        """
        resolved = self.check_options(
            options=options,
            mode=mode,
            match_by=match_by,
            cast=cast,
            use_threads=use_threads,
            raise_error=raise_error,
            **option_kwargs,
        )
        return self._write_pylist(data, resolved)

    @staticmethod
    def _normalize_records(data: Iterable[dict]) -> list[dict]:
        """Return *data* with every row backfilled to the union of keys.

        :meth:`pyarrow.Table.from_pylist` infers the schema from the first row
        only, silently dropping columns that appear later and producing wrong
        results for sparse / heterogeneous inputs. This helper walks every row
        once, collects the union of keys in first-seen order, then produces a
        new list where each row has every key (missing entries filled with
        ``None``). All-``None`` columns end up with Arrow ``null`` type on
        conversion.
        """
        rows = list(data) if not isinstance(data, list) else data
        if not rows:
            return []

        # Preserve first-seen key order for stable column ordering.
        all_keys: dict[str, None] = {}
        needs_backfill = False
        reference_keys: tuple[str, ...] | None = None

        for row in rows:
            if row is None:
                needs_backfill = True
                continue
            row_keys = tuple(row.keys())
            if reference_keys is None:
                reference_keys = row_keys
            elif row_keys != reference_keys:
                needs_backfill = True
            for key in row_keys:
                if key not in all_keys:
                    all_keys[key] = None

        if not needs_backfill:
            return rows

        keys = tuple(all_keys.keys())
        return [
            {key: (row.get(key) if row is not None else None) for key in keys}
            for row in rows
        ]

    def _write_pylist(
        self,
        data: Iterable[dict],
        options: O,
    ) -> None:
        rows = self._normalize_records(data)
        try:
            tb = pa.Table.from_pylist(rows)
        except pa.ArrowInvalid:
            # Genuinely unrepresentable mixed types — fall back to batch streaming.
            return self._write_arrow_batches(
                self.iter_arrow_batches(rows),
                options,
            )
        return self._write_arrow_table(tb, options)

    # ------------------------------------------------------------------
    # Python dict[str, list]
    # ------------------------------------------------------------------

    def read_pydict(
        self,
        *,
        options: O | None = None,
        columns: "Sequence[str] | None | Any" = ...,
        cast: "CastOptionsArg | Any" = ...,
        use_threads: "bool | Any" = ...,
        ignore_empty: "bool | Any" = ...,
        raise_error: "bool | Any" = ...,
        batch_size: "int | None | Any" = ...,
        **option_kwargs,
    ) -> Union[dict[str, list], Iterator[dict[str, list]]]:
        """Read the buffer as a column-oriented dict."""
        resolved = self.check_options(
            options=options,
            columns=columns,
            cast=cast,
            use_threads=use_threads,
            ignore_empty=ignore_empty,
            raise_error=raise_error,
            batch_size=batch_size,
            **option_kwargs,
        )
        return self._read_pydict(resolved)

    def _read_pydict(
        self,
        options: O,
    ) -> Union[dict[str, list], Iterator[dict[str, list]]]:
        table_or_iter = self._read_arrow_table(options)

        if isinstance(table_or_iter, pa.Table):
            return table_or_iter.to_pydict(maps_as_pydicts=True)

        def iter_pydicts() -> Iterator[dict[str, list]]:
            for tb in table_or_iter:
                yield tb.to_pydict(maps_as_pydicts=True)
        return iter_pydicts()

    def write_pydict(
        self,
        data: dict[str, list],
        *,
        options: O | None = None,
        mode: "SaveMode | str | None | Any" = ...,
        match_by: "Sequence[str] | str | None | Any" = ...,
        cast: "CastOptionsArg | Any" = ...,
        use_threads: "bool | Any" = ...,
        raise_error: "bool | Any" = ...,
        **option_kwargs,
    ) -> None:
        """Write a column-oriented dict to the buffer."""
        resolved = self.check_options(
            options=options,
            mode=mode,
            match_by=match_by,
            cast=cast,
            use_threads=use_threads,
            raise_error=raise_error,
            **option_kwargs,
        )
        return self._write_pydict(data, resolved)

    def _write_pydict(
        self,
        data: dict[str, list],
        options: O,
    ) -> None:
        tb = pa.Table.from_pydict(data)  # noqa
        return self._write_arrow_table(tb, options)

    # ------------------------------------------------------------------
    # Pandas
    # ------------------------------------------------------------------

    def read_pandas_frame(
        self,
        *,
        options: O | None = None,
        columns: "Sequence[str] | None | Any" = ...,
        cast: "CastOptionsArg | Any" = ...,
        use_threads: "bool | Any" = ...,
        ignore_empty: "bool | Any" = ...,
        raise_error: "bool | Any" = ...,
        batch_size: "int | None | Any" = ...,
        **option_kwargs,
    ) -> Union["pandas.DataFrame", Iterator["pandas.DataFrame"]]:
        """Read the buffer as a :class:`pandas.DataFrame`."""
        resolved = self.check_options(
            options=options,
            columns=columns,
            cast=cast,
            use_threads=use_threads,
            ignore_empty=ignore_empty,
            raise_error=raise_error,
            batch_size=batch_size,
            **option_kwargs,
        )
        return self._read_pandas_frame(resolved)

    def _read_pandas_frame(
        self,
        options: O,
    ) -> Union["pandas.DataFrame", Iterator["pandas.DataFrame"]]:
        table_or_iter = self._read_arrow_table(options)

        if isinstance(table_or_iter, pa.Table):
            return table_or_iter.to_pandas()

        def iter_frames() -> Iterator["pandas.DataFrame"]:
            for tb in table_or_iter:
                yield tb.to_pandas()
        return iter_frames()

    def write_pandas_frame(
        self,
        frame: "pandas.DataFrame",
        *,
        options: O | None = None,
        mode: "SaveMode | str | None | Any" = ...,
        match_by: "Sequence[str] | str | None | Any" = ...,
        cast: "CastOptionsArg | Any" = ...,
        use_threads: "bool | Any" = ...,
        raise_error: "bool | Any" = ...,
        **option_kwargs,
    ) -> None:
        """Write a :class:`pandas.DataFrame` to the buffer.

        An unnamed ``RangeIndex`` is dropped; a named index is preserved.
        """
        resolved = self.check_options(
            options=options,
            mode=mode,
            match_by=match_by,
            cast=cast,
            use_threads=use_threads,
            raise_error=raise_error,
            **option_kwargs,
        )
        return self._write_pandas_frame(frame, resolved)

    def _write_pandas_frame(
        self,
        frame: "pandas.DataFrame",
        options: O,
    ) -> None:
        tb = pa.Table.from_pandas(
            frame,
            preserve_index=bool(frame.index.name),
        )  # noqa
        return self._write_arrow_table(tb, options)

    # ------------------------------------------------------------------
    # Polars
    # ------------------------------------------------------------------

    def read_polars_frame(
        self,
        *,
        options: O | None = None,
        columns: "Sequence[str] | None | Any" = ...,
        cast: "CastOptionsArg | Any" = ...,
        use_threads: "bool | Any" = ...,
        ignore_empty: "bool | Any" = ...,
        lazy: "bool | Any" = ...,
        raise_error: "bool | Any" = ...,
        batch_size: "int | None | Any" = ...,
        **option_kwargs,
    ) -> "pl.DataFrame | pl.LazyFrame | Iterator[pl.DataFrame | pl.LazyFrame]":
        """Read the buffer as a Polars frame (or iterator of chunks)."""
        resolved = self.check_options(
            options=options,
            columns=columns,
            cast=cast,
            use_threads=use_threads,
            ignore_empty=ignore_empty,
            lazy=lazy,
            raise_error=raise_error,
            batch_size=batch_size,
            **option_kwargs,
        )
        return self._read_polars_frame(resolved)

    def _read_polars_frame(
        self,
        options: O,
    ) -> "pl.DataFrame | pl.LazyFrame | Iterator[pl.DataFrame | pl.LazyFrame]":
        # Chunked: yield one frame per record batch.
        if getattr(options, "batch_size", 0) and options.batch_size > 0:
            return self._read_polars_frames(options)

        # Single frame: materialize the whole table once.
        arrow_table = self._read_arrow_table(options)
        # _read_arrow_table returns a pa.Table when batch_size<=0.
        df = pl.from_arrow(arrow_table)
        return df.lazy() if getattr(options, "lazy", False) else df

    def read_polars_frames(
        self,
        *,
        options: Optional[O] = None,
        columns: "Sequence[str] | None | Any" = ...,
        cast: "CastOptionsArg | Any" = ...,
        use_threads: "bool | Any" = ...,
        ignore_empty: "bool | Any" = ...,
        lazy: "bool | Any" = ...,
        raise_error: "bool | Any" = ...,
        **option_kwargs,
    ) -> Iterator["pl.DataFrame | pl.LazyFrame"]:
        """Yield one Polars frame per Arrow record batch."""
        resolved = self.check_options(
            options=options,
            columns=columns,
            cast=cast,
            use_threads=use_threads,
            ignore_empty=ignore_empty,
            lazy=lazy,
            raise_error=raise_error,
            **option_kwargs,
        )
        yield from self._read_polars_frames(resolved)

    def _read_polars_frames(
        self,
        options: O,
    ) -> Iterator["pl.DataFrame | pl.LazyFrame"]:
        want_lazy = bool(getattr(options, "lazy", False))
        for arrow_batch in self._read_arrow_batches(options):
            df = pl.from_arrow(arrow_batch, rechunk=False)
            yield df.lazy() if want_lazy else df

    def write_polars_frame(
        self,
        frame: "pl.DataFrame | pl.LazyFrame",
        *,
        options: O | None = None,
        mode: "SaveMode | str | None | Any" = ...,
        match_by: "Sequence[str] | str | None | Any" = ...,
        cast: "CastOptionsArg | Any" = ...,
        use_threads: "bool | Any" = ...,
        raise_error: "bool | Any" = ...,
        **option_kwargs,
    ) -> None:
        """Write a Polars DataFrame or LazyFrame to the buffer."""
        resolved = self.check_options(
            options=options,
            mode=mode,
            match_by=match_by,
            cast=cast,
            use_threads=use_threads,
            raise_error=raise_error,
            **option_kwargs,
        )
        return self._write_polars_frame(frame, resolved)

    def _write_polars_frame(
        self,
        frame: "pl.DataFrame | pl.LazyFrame",
        options: O,
    ) -> None:
        # Collect lazy frames up-front so height / sizing work.
        if isinstance(frame, pl.LazyFrame):
            frame = frame.collect()

        if frame.height == 0:
            return None

        batch_size = getattr(options, "batch_size", 0) or 0
        byte_size = getattr(options, "byte_size", 0) or 0

        if batch_size > 0:
            def iter_chunks() -> Iterator["pl.DataFrame"]:
                yield from frame.iter_slices(n_rows=batch_size)

        elif byte_size > 0:
            total_size = frame.estimated_size(unit="b")
            if total_size == 0:
                def iter_chunks() -> Iterator["pl.DataFrame"]:
                    yield frame
            else:
                rows_per_chunk = max(1, int(frame.height * byte_size / total_size))

                def iter_chunks() -> Iterator["pl.DataFrame"]:
                    yield from frame.iter_slices(n_rows=rows_per_chunk)

        else:
            def iter_chunks() -> Iterator["pl.DataFrame"]:
                yield frame

        return self._write_polars_frames(iter_chunks(), options)

    def _write_polars_frames(
        self,
        frames: Iterator["pl.DataFrame | pl.LazyFrame"],
        options: O,
    ) -> None:
        return self._write_arrow_batches(
            self.iter_arrow_batches(frames),
            options,
        )