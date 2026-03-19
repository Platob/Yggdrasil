"""Media I/O abstraction layer for :class:`~yggdrasil.io.buffer.BytesIO`.

Each concrete :class:`MediaIO` subclass handles reading and writing a specific
columnar or container format (Parquet, JSON, Arrow IPC, ZIP).  The base class
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
  ``batch_size`` parameter.  When set to a positive integer the method
  returns an ``Iterator`` of chunks instead of a single object, enabling
  memory-efficient streaming over large datasets.

Subclasses only need to implement three methods:

* ``check_options(options, **kwargs) → O`` — validate/merge caller options.
* ``_read_arrow_table(options) → pyarrow.Table`` — read the *uncompressed*
  buffer into an Arrow table.
* ``_write_arrow_table(table, options)`` — write an Arrow table into the
  *uncompressed* buffer.

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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Iterator, Sequence, TypeVar, overload, Optional, Union

from yggdrasil.io.enums import Codec, MediaType, MimeType, SaveMode
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
    """Abstract base for format-specific I/O on a :class:`BytesIO` buffer.

    Parameters
    ----------
    media_type:
        The resolved :class:`~yggdrasil.io.enums.MediaType` of the buffer.
        When it carries a :attr:`~MediaType.codec` (e.g. gzip, zstd), reads
        and writes transparently decompress / compress.
    buffer:
        The backing :class:`BytesIO` that holds the raw (possibly compressed)
        payload.
    """

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
        **copy** (the original buffer is not mutated).  The caller must
        close the copy when done.

        Returns
        -------
        tuple[BytesIO, bool]
            ``(buffer_to_read_from, True)`` when decompression happened;
            ``(self.buffer, False)`` otherwise.
        """
        codec = self.codec
        if codec is not None and self.buffer.size > 0:
            return self.buffer.decompress(codec=codec, copy=True), True
        return self.buffer, False

    def _compress_into_buffer(self, plain: BytesIO) -> None:
        """Compress *plain* with :attr:`codec` and replace :attr:`buffer` contents.

        If no codec is set the raw bytes from *plain* are copied directly.

        Parameters
        ----------
        plain:
            A :class:`BytesIO` holding the uncompressed payload.
        """
        codec = self.codec
        if codec is not None:
            compressed = plain.compress(codec=codec, copy=True)
            self.buffer._replace_with_payload(compressed.to_bytes())
        else:
            self.buffer._replace_with_payload(plain.to_bytes())

    # ------------------------------------------------------------------
    # Abstract protocol
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def check_options(cls, options: Optional[O], *args, **kwargs) -> O:
        """Validate and merge caller-supplied options into a concrete instance.

        Subclasses must implement this method.  It is called at the top of
        every public read / write method.

        Parameters
        ----------
        options:
            An existing options instance, or ``None`` for defaults.
        **kwargs:
            Individual overrides merged on top of *options*.

        Returns
        -------
        O
            A fully-resolved, validated options instance.
        """

    @abstractmethod
    def _read_arrow_table(self, *, options: O) -> "pyarrow.Table":
        """Read the **uncompressed** buffer into an Arrow table.

        Subclasses implement this without worrying about transport
        compression — the public :meth:`read_arrow_table` handles
        decompression automatically before calling this method.
        """

    @abstractmethod
    def _write_arrow_table(self, *, table: "pyarrow.Table", options: O) -> None:
        """Write an Arrow table into the **uncompressed** buffer.

        Subclasses implement this without worrying about transport
        compression — the public :meth:`write_arrow_table` handles
        compression automatically after calling this method.
        """

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @overload
    @classmethod
    def make(cls, buffer: BytesIO, media: MimeType.PARQUET) -> "ParquetIO": ...
    @overload
    @classmethod
    def make(cls, buffer: BytesIO, media: MimeType.JSON) -> "JsonIO": ...
    @overload
    @classmethod
    def make(cls, buffer: BytesIO, media: MimeType.NDJSON) -> "JsonIO": ...
    @overload
    @classmethod
    def make(cls, buffer: BytesIO, media: MimeType.ZIP) -> "ZipIO": ...
    @overload
    @classmethod
    def make(cls, buffer: BytesIO, media: MimeType.ARROW_IPC) -> "IPCIO": ...
    @overload
    @classmethod
    def make(cls, buffer: BytesIO, media: MediaType | str) -> "MediaIO[MediaOptions]": ...

    @classmethod
    def make(cls, buffer: BytesIO, media: MediaType | MimeType | str) -> "MediaIO[MediaOptions]":
        """Create the appropriate :class:`MediaIO` subclass for *media*.

        The *media* argument is parsed to a
        :class:`~yggdrasil.io.enums.MediaType`; its ``mime_type`` selects the
        concrete subclass and its ``codec`` (if any) is preserved so that
        reads auto-decompress and writes auto-compress.

        Parameters
        ----------
        buffer:
            The backing byte buffer.
        media:
            A :class:`MediaType`, :class:`MimeType`, or parseable string.

        Returns
        -------
        MediaIO
            A concrete subclass instance ready for I/O.

        Raises
        ------
        NotImplementedError
            When no subclass is registered for the MIME type.
        """
        media = MediaType.parse(media)
        mt = media.mime_type
        buffer.set_media_type(media, safe=False)

        if mt is MimeType.PARQUET:
            from .parquet_io import ParquetIO
            return ParquetIO(media_type=media, buffer=buffer)
        if mt is MimeType.JSON or mt is MimeType.NDJSON:
            from .json_io import JsonIO
            return JsonIO(media_type=media, buffer=buffer)
        if mt is MimeType.ZIP:
            from .zip_io import ZipIO
            return ZipIO(media_type=media, buffer=buffer)
        if mt is MimeType.ARROW_IPC:
            from .arrow_ipc_io import IPCIO
            return IPCIO(media_type=media, buffer=buffer)

        raise NotImplementedError(f"Cannot create media IO for {media!r}")

    # ------------------------------------------------------------------
    # Write guard
    # ------------------------------------------------------------------

    def skip_write(self, mode: SaveMode) -> bool:
        """Return ``True`` when the write should be skipped.

        Parameters
        ----------
        mode:
            The requested save mode.

        Returns
        -------
        bool
            ``True`` when *mode* is ``IGNORE`` and the buffer is non-empty.

        Raises
        ------
        IOError
            When *mode* is ``ERROR_IF_EXISTS`` and the buffer is non-empty.
        """
        if self.buffer.size > 0:
            if mode == SaveMode.IGNORE:
                return True
            elif mode == SaveMode.ERROR_IF_EXISTS:
                raise IOError(
                    f"Cannot write in already existing {self.buffer!r} "
                    f"with save mode {SaveMode.ERROR_IF_EXISTS.value}"
                )
        return False

    # ------------------------------------------------------------------
    # Save-mode helpers (private)
    # ------------------------------------------------------------------

    def _read_existing_table(self) -> "pyarrow.Table | None":
        """Read the current buffer contents as an Arrow table, or ``None``.

        Returns ``None`` when the buffer is empty.  Handles codec
        decompression transparently so that the returned table contains
        plain rows regardless of on-disk encoding.

        Returns
        -------
        pyarrow.Table | None
        """
        if self.buffer.size <= 0:
            return None
        try:
            return self.read_arrow_table()
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
        """Combine *existing* and *incoming* tables according to *mode*.

        Parameters
        ----------
        existing:
            The data currently in the buffer, or ``None`` when the buffer
            is empty.
        incoming:
            The new data to write.
        mode:
            Save mode governing the merge strategy.
        match_by:
            Column names used as the composite key for ``UPSERT``.
            Ignored by other modes.  Required (non-empty) for ``UPSERT``;
            raises :exc:`ValueError` when missing.

        Returns
        -------
        pyarrow.Table
            The final table that should be serialised into the buffer.

        Raises
        ------
        ValueError
            When *mode* is ``UPSERT`` and *match_by* is ``None`` or empty.

        Notes
        -----
        Mode semantics:

        ``AUTO`` / ``OVERWRITE`` / ``TRUNCATE``
            Return *incoming* unchanged (full replace).

        ``APPEND``
            Concatenate ``[existing, incoming]``.  Schema promotion is
            enabled so that columns present in only one side are filled
            with nulls.

        ``UPSERT``
            Rows from *incoming* whose ``match_by`` key already exists in
            *existing* **replace** the old rows.  Rows in *existing* that
            have no match are kept.  New rows from *incoming* that have no
            match in *existing* are appended.  The result is ordered as
            ``[surviving_existing, new_incoming]``.
        """
        import pyarrow as pa

        # --- Fast path: nothing to merge with --------------------------
        if existing is None or existing.num_rows == 0:
            return incoming

        # --- OVERWRITE / AUTO / TRUNCATE: full replace -----------------
        if mode in (SaveMode.AUTO, SaveMode.OVERWRITE, SaveMode.TRUNCATE):
            return incoming

        # --- APPEND: stack rows ----------------------------------------
        if mode == SaveMode.APPEND:
            return pa.concat_tables([existing, incoming], promote_options="default")

        # --- UPSERT: merge by key columns ------------------------------
        if mode == SaveMode.UPSERT:
            if not match_by:
                raise ValueError(
                    "SaveMode.UPSERT requires match_by columns, got None/empty"
                )
            match_by = list(match_by)

            # Validate that key columns exist in both tables
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

            # Build a set of composite keys from the incoming table
            incoming_keys: set[tuple] = set()
            incoming_key_arrays = [
                incoming.column(c).to_pylist() for c in match_by
            ]
            for row_idx in range(incoming.num_rows):
                key = tuple(arr[row_idx] for arr in incoming_key_arrays)
                incoming_keys.add(key)

            # Filter existing: keep rows whose key is NOT in incoming
            existing_key_arrays = [
                existing.column(c).to_pylist() for c in match_by
            ]
            keep_mask = []
            for row_idx in range(existing.num_rows):
                key = tuple(arr[row_idx] for arr in existing_key_arrays)
                keep_mask.append(key not in incoming_keys)

            surviving = existing.filter(pa.array(keep_mask, type=pa.bool_()))

            return pa.concat_tables(
                [surviving, incoming], promote_options="default",
            )

        # --- Unrecognised mode: fall back to overwrite -----------------
        return incoming

    # ------------------------------------------------------------------
    # Batching helpers (private)
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_arrow_batches(
        table: "pyarrow.Table",
        batch_size: int,
    ) -> Iterator["pyarrow.Table"]:
        """Yield successive :class:`pyarrow.Table` slices of *batch_size* rows.

        Parameters
        ----------
        table:
            The full Arrow table to slice.
        batch_size:
            Maximum number of rows per chunk.  Must be > 0.

        Yields
        ------
        pyarrow.Table
            A table slice with at most *batch_size* rows.
        """
        total = table.num_rows
        for offset in range(0, total, batch_size):
            yield table.slice(offset, min(batch_size, total - offset))

    # ------------------------------------------------------------------
    # Arrow (primary public API)
    # ------------------------------------------------------------------

    def read_arrow_table(
        self,
        *,
        batch_size: Optional[int] = None,
        options: O | None = None,
        **option_kwargs,
    ) -> Union["pyarrow.Table", Iterator["pyarrow.Table"]]:
        """Read the buffer into a :class:`pyarrow.Table`.

        Handles transport decompression transparently when the media type
        carries a codec.

        Parameters
        ----------
        batch_size:
            When ``None`` or ≤ 0, return a single :class:`pyarrow.Table`.
            When a positive integer, return an ``Iterator[pyarrow.Table]``
            yielding slices of at most *batch_size* rows.
        options:
            Format-specific options, or ``None`` for defaults.
        **option_kwargs:
            Overrides merged into *options*.

        Returns
        -------
        pyarrow.Table | Iterator[pyarrow.Table]
        """
        resolved = self.check_options(options=options, **option_kwargs)
        buf, decompressed = self._decompressed_buffer()
        try:
            orig_buffer = self.buffer
            if decompressed:
                self.buffer = buf
            table = self._read_arrow_table(options=resolved)
        finally:
            if decompressed:
                self.buffer = orig_buffer

        if batch_size is not None and batch_size > 0:
            return self._iter_arrow_batches(table, batch_size)
        return table

    def write_arrow_table(
        self,
        table: "pyarrow.Table",
        *,
        batch_size: Optional[int] = None,
        mode: SaveMode | str | None = None,
        match_by: list[str] | None = None,
        options: O | None = None,
        **option_kwargs,
    ) -> None:
        """Write a :class:`pyarrow.Table` to the buffer.

        Handles transport compression transparently when the media type
        carries a codec.

        Save-mode semantics
        --------------------
        ``AUTO`` / ``OVERWRITE`` / ``TRUNCATE``
            Replace the buffer contents entirely with *table*.

        ``APPEND``
            Read the current buffer, concatenate *table* after it, and
            write the combined result back.  Schema promotion fills
            missing columns with nulls.

        ``UPSERT``
            Read the current buffer, remove any existing rows whose
            ``match_by`` composite key appears in *table*, then append
            *table*.  Requires ``match_by`` to be a non-empty list of
            column names present in both the existing and incoming data.

        ``IGNORE``
            Skip the write when the buffer already contains data.

        ``ERROR_IF_EXISTS``
            Raise :exc:`IOError` when the buffer already contains data.

        Parameters
        ----------
        table:
            The Arrow table to serialise.
        batch_size:
            When ``None`` or ≤ 0, write the entire table at once.
            When a positive integer, slice the table into chunks of at
            most *batch_size* rows and write each chunk sequentially.
            Only the **last** chunk is kept in the buffer (each write
            overwrites the previous content).  For accumulation, callers
            should iterate and use ``APPEND`` mode.
        mode:
            Save mode governing the merge strategy.
        match_by:
            Columns used as the composite key for ``UPSERT`` mode.
        options:
            Format-specific options, or ``None`` for defaults.
        **option_kwargs:
            Overrides merged into *options*.
        """
        resolved = self.check_options(
            options=options, mode=mode, match_by=match_by, **option_kwargs,
        )
        if self.skip_write(mode=resolved.mode):
            return

        # --- Apply save-mode merge logic ---
        if resolved.mode in (SaveMode.APPEND, SaveMode.UPSERT):
            existing = self._read_existing_table()
            table = self._apply_save_mode(
                existing, table,
                mode=resolved.mode,
                match_by=resolved.match_by,
            )

        if batch_size is not None and batch_size > 0:
            for chunk in self._iter_arrow_batches(table, batch_size):
                self._write_single_table(chunk, resolved)
        else:
            self._write_single_table(table, resolved)

    def _write_single_table(
        self,
        table: "pyarrow.Table",
        options: O,
    ) -> None:
        """Write one table (or chunk) handling codec compression."""
        codec = self.codec
        if codec is not None:
            plain = BytesIO(config=self.buffer.config)
            orig_buffer = self.buffer
            self.buffer = plain
            try:
                self._write_arrow_table(table=table, options=options)
            finally:
                self.buffer = orig_buffer
            self._compress_into_buffer(plain)
        else:
            self._write_arrow_table(table=table, options=options)

    # ------------------------------------------------------------------
    # Python dict / list convenience wrappers
    # ------------------------------------------------------------------

    def read_pylist(
        self,
        *,
        batch_size: Optional[int] = None,
    ) -> Union[list[dict], Iterator[list[dict]]]:
        """Read the buffer as a list of row dicts.

        Parameters
        ----------
        batch_size:
            When ``None`` or ≤ 0, return a single ``list[dict]``.
            When a positive integer, return an ``Iterator[list[dict]]``
            yielding chunks of at most *batch_size* rows.

        Returns
        -------
        list[dict] | Iterator[list[dict]]
        """
        if batch_size is not None and batch_size > 0:
            return (chunk.to_pylist() for chunk in self.read_arrow_table(batch_size=batch_size))
        return self.read_arrow_table().to_pylist()

    def write_pylist(
        self,
        data: list[dict],
        *,
        batch_size: Optional[int] = None,
        mode: SaveMode | str | None = None,
        match_by: list[str] | None = None,
        options: O | None = None,
        **option_kwargs,
    ):
        """Write a list of row dicts to the buffer.

        Parameters
        ----------
        data:
            List of ``{column: value}`` dicts.
        batch_size:
            Forwarded to :meth:`write_arrow_table`.
        mode, match_by, options, **option_kwargs:
            Forwarded to :meth:`write_arrow_table`.
        """
        import pyarrow as _pa
        tb = _pa.Table.from_pylist(data)
        self.write_arrow_table(
            table=tb, batch_size=batch_size, mode=mode, match_by=match_by,
            options=options, **option_kwargs,
        )

    def read_pydict(
        self,
        *,
        batch_size: Optional[int] = None,
    ) -> Union[dict[str, list], Iterator[dict[str, list]]]:
        """Read the buffer as a column-oriented dict.

        Parameters
        ----------
        batch_size:
            When ``None`` or ≤ 0, return a single ``dict[str, list]``.
            When a positive integer, return an ``Iterator[dict[str, list]]``
            yielding chunks of at most *batch_size* rows.

        Returns
        -------
        dict[str, list] | Iterator[dict[str, list]]
        """
        if batch_size is not None and batch_size > 0:
            return (chunk.to_pydict() for chunk in self.read_arrow_table(batch_size=batch_size))
        return self.read_arrow_table().to_pydict()

    def write_pydict(
        self,
        data: dict[str, list],
        *,
        batch_size: Optional[int] = None,
        mode: SaveMode | str | None = None,
        match_by: list[str] | None = None,
        options: O | None = None,
        **option_kwargs,
    ):
        """Write a column-oriented dict to the buffer.

        Parameters
        ----------
        data:
            ``{column_name: [values]}`` dict.
        batch_size:
            Forwarded to :meth:`write_arrow_table`.
        mode, match_by, options, **option_kwargs:
            Forwarded to :meth:`write_arrow_table`.
        """
        import pyarrow as _pa
        tb = _pa.Table.from_pydict(data)
        self.write_arrow_table(
            table=tb, batch_size=batch_size, mode=mode, match_by=match_by,
            options=options, **option_kwargs,
        )

    # ------------------------------------------------------------------
    # Pandas / Polars convenience wrappers
    # ------------------------------------------------------------------

    def read_pandas_frame(
        self,
        *,
        batch_size: Optional[int] = None,
        options: O | None = None,
        **option_kwargs,
    ) -> Union["pandas.DataFrame", Iterator["pandas.DataFrame"]]:
        """Read the buffer as a :class:`pandas.DataFrame`.

        Parameters
        ----------
        batch_size:
            When ``None`` or ≤ 0, return a single :class:`pandas.DataFrame`.
            When a positive integer, return an
            ``Iterator[pandas.DataFrame]`` yielding chunks of at most
            *batch_size* rows.
        options, **option_kwargs:
            Forwarded to :meth:`read_arrow_table`.

        Returns
        -------
        pandas.DataFrame | Iterator[pandas.DataFrame]
        """
        if batch_size is not None and batch_size > 0:
            return (
                chunk.to_pandas()
                for chunk in self.read_arrow_table(
                    batch_size=batch_size, options=options, **option_kwargs,
                )
            )
        return self.read_arrow_table(options=options, **option_kwargs).to_pandas()

    def write_pandas_frame(
        self,
        frame: "pandas.DataFrame",
        *,
        batch_size: Optional[int] = None,
        mode: SaveMode | str | None = None,
        match_by: list[str] | None = None,
        options: O | None = None,
        **option_kwargs,
    ) -> None:
        """Write a :class:`pandas.DataFrame` to the buffer.

        Parameters
        ----------
        frame:
            The DataFrame to serialise.
        batch_size:
            Forwarded to :meth:`write_arrow_table`.
        mode, match_by, options, **option_kwargs:
            Forwarded to :meth:`write_arrow_table`.
        """
        import pyarrow as _pa
        tb = _pa.Table.from_pandas(frame)
        self.write_arrow_table(
            table=tb, batch_size=batch_size, mode=mode, match_by=match_by,
            options=options, **option_kwargs,
        )

    def read_polars_frame(
        self,
        *,
        batch_size: Optional[int] = None,
        lazy: bool = False,
        options: O | None = None,
        **option_kwargs,
    ) -> Union[
        "polars.DataFrame",
        "polars.LazyFrame",
        Iterator["polars.DataFrame"],
    ]:
        """Read the buffer as a Polars frame.

        Parameters
        ----------
        batch_size:
            When ``None`` or ≤ 0, return a single frame.
            When a positive integer, return an
            ``Iterator[polars.DataFrame]`` yielding chunks of at most
            *batch_size* rows.  The *lazy* parameter is ignored when
            batching (each chunk is always a materialised DataFrame).
        lazy:
            When ``True`` and ``batch_size`` is ``None`` / ≤ 0, return a
            :class:`polars.LazyFrame`.
        options, **option_kwargs:
            Forwarded to :meth:`read_arrow_table`.

        Returns
        -------
        polars.DataFrame | polars.LazyFrame | Iterator[polars.DataFrame]
        """
        from yggdrasil.polars.lib import polars as _pl

        if batch_size is not None and batch_size > 0:
            return (
                _pl.from_arrow(chunk, rechunk=False)
                for chunk in self.read_arrow_table(
                    batch_size=batch_size, options=options, **option_kwargs,
                )
            )

        tb = self.read_arrow_table(options=options, **option_kwargs)
        df = _pl.from_arrow(tb, rechunk=False)
        return df.lazy() if lazy else df

    def write_polars_frame(
        self,
        frame: "polars.DataFrame",
        *,
        batch_size: Optional[int] = None,
        mode: SaveMode | str | None = None,
        match_by: list[str] | None = None,
        options: O | None = None,
        **option_kwargs,
    ) -> None:
        """Write a Polars DataFrame to the buffer.

        Parameters
        ----------
        frame:
            The Polars frame to serialise.
        batch_size:
            Forwarded to :meth:`write_arrow_table`.
        mode, match_by, options, **option_kwargs:
            Forwarded to :meth:`write_arrow_table`.
        """
        tb = frame.to_arrow()
        self.write_arrow_table(
            table=tb, batch_size=batch_size, mode=mode, match_by=match_by,
            options=options, **option_kwargs,
        )
