"""JSON / NDJSON I/O on top of :class:`~yggdrasil.io.buffer.BytesIO`.

Two wire formats are routed here via :attr:`MediaIO.media_type.mime_type`:

* ``application/json`` — the buffer holds exactly one JSON value. Reading
  materializes the whole value at once; writing serializes the whole table
  as a single JSON array.
* ``application/x-ndjson`` — the buffer holds one JSON value per line.
  Reading streams line-by-line; writing emits one JSON object per row.

Row normalization (read side):

* top-level **list of dicts** → each dict is a row.
* top-level **list of scalars** → each scalar becomes a row with a
  ``"value"`` column.
* top-level **dict** → one row with that dict's keys as columns.
* top-level **scalar** → one row with a ``"value"`` column.
* empty buffer or empty list → no batches.

Transport-level compression is handled transparently by the base class:
``open()`` decompresses into ``self.buffer``, ``close()`` re-compresses
when ``mark_dirty()`` has been called.
"""
from __future__ import annotations

import json as _json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Optional, Sequence

import pyarrow as pa

import yggdrasil.pickle.json as json_mod
from yggdrasil.io.enums import MimeTypes
from .media_io import MediaIO
from .media_options import MediaOptions

if TYPE_CHECKING:
    import pyarrow

__all__ = ["JsonOptions", "JsonIO"]


_VALUE_COLUMN = "value"


@dataclass
class JsonOptions(MediaOptions):
    """Options for JSON / NDJSON I/O.

    Parameters
    ----------
    encoding:
        Character encoding used when decoding / writing.
    errors:
        Error-handling mode for encoding (``"strict"``, ``"replace"``, …).
    """

    encoding: str = "utf-8"
    errors: str = "strict"

    def __post_init__(self) -> None:
        """Normalize and validate JSON-specific options."""
        super().__post_init__()

        if not isinstance(self.encoding, str):
            raise TypeError(
                f"encoding must be str, got {type(self.encoding).__name__}"
            )
        if not self.encoding:
            raise ValueError("encoding must not be empty")

        if not isinstance(self.errors, str):
            raise TypeError(
                f"errors must be str, got {type(self.errors).__name__}"
            )
        if not self.errors:
            raise ValueError("errors must not be empty")

    @classmethod
    def resolve(cls, *, options: "JsonOptions | None" = None, **overrides: Any) -> "JsonOptions":
        """Merge *overrides* into *options* (or a fresh default)."""
        return cls.check_parameters(options=options, **overrides)


@dataclass(slots=True)
class JsonIO(MediaIO[JsonOptions]):
    """JSON / NDJSON I/O with row-list normalization."""

    @classmethod
    def check_options(
        cls,
        options: Optional[JsonOptions],
        *args,
        **kwargs,
    ) -> JsonOptions:
        """Validate and merge caller-supplied options."""
        return JsonOptions.check_parameters(options=options, **kwargs)

    # ------------------------------------------------------------------
    # Format routing
    # ------------------------------------------------------------------

    @property
    def _is_ndjson(self) -> bool:
        """Return ``True`` when this buffer is newline-delimited JSON."""
        return self.media_type.mime_type is MimeTypes.NDJSON

    # ------------------------------------------------------------------
    # Record normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _value_to_record(value: Any) -> dict:
        """Wrap a single loaded JSON value as a row dict."""
        if isinstance(value, dict):
            return value
        return {_VALUE_COLUMN: value}

    @classmethod
    def _normalize_loaded_json(cls, data: Any) -> list[dict]:
        """Normalize a fully-loaded JSON value into row-oriented records."""
        if isinstance(data, list):
            return [cls._value_to_record(item) for item in data]
        return [cls._value_to_record(data)]

    # ------------------------------------------------------------------
    # Low-level loaders (assume buffer is already opened / decompressed)
    # ------------------------------------------------------------------

    def _iter_ndjson_records(self, options: JsonOptions) -> Iterator[dict]:
        """Stream NDJSON records from ``self.buffer`` line-by-line."""
        raw = self.buffer.to_bytes()
        if not raw:
            return

        text = raw.decode(options.encoding, errors=options.errors)
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            yield self._value_to_record(_json.loads(line))

    def _load_json_records(self, options: JsonOptions) -> list[dict]:
        """Fully load JSON bytes from ``self.buffer`` into records."""
        raw = self.buffer.to_bytes()
        if not raw:
            return []

        data = _json.loads(raw.decode(options.encoding, errors=options.errors))
        return self._normalize_loaded_json(data)

    # ------------------------------------------------------------------
    # Batch construction
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_records(
        records: Iterator[dict],
        chunk_size: int,
    ) -> Iterator[list[dict]]:
        """Yield non-empty chunks of at most *chunk_size* records."""
        buf: list[dict] = []
        for rec in records:
            buf.append(rec)
            if len(buf) >= chunk_size:
                yield buf
                buf = []
        if buf:
            yield buf

    def _records_to_batches(
        self,
        records: Iterator[dict] | list[dict],
        options: JsonOptions,
    ) -> Iterator["pa.RecordBatch"]:
        """Convert row records into Arrow record batches."""
        batch_size = getattr(options, "batch_size", 0) or 0

        if isinstance(records, list):
            if not records:
                return
            if batch_size <= 0 or len(records) <= batch_size:
                yield pa.RecordBatch.from_pylist(records)
                return
            # Slice the list rather than rebuilding an iterator.
            for start in range(0, len(records), batch_size):
                yield pa.RecordBatch.from_pylist(records[start : start + batch_size])
            return

        # Iterator path (NDJSON).
        effective = batch_size if batch_size > 0 else 10_000
        for chunk in self._chunk_records(records, effective):
            yield pa.RecordBatch.from_pylist(chunk)

    # ------------------------------------------------------------------
    # Core read/write protocol
    # ------------------------------------------------------------------

    def _read_arrow_batches(
        self,
        options: JsonOptions,
    ) -> Iterator["pyarrow.RecordBatch"]:
        """Yield record batches from the JSON / NDJSON buffer.

        * ``[]`` or empty buffer → no batches.
        * ``[{…}, {…}]`` → each dict is a row.
        * ``[1, 2, 3]`` → rows with a ``"value"`` column.
        * ``{…}`` → one row.
        * NDJSON → streams line-by-line; honors ``options.batch_size``.
        """
        with self.open() as b:
            if b.buffer.size <= 0:
                return

            if self._is_ndjson:
                records: Iterator[dict] | list[dict] = self._iter_ndjson_records(options)
            else:
                records = self._load_json_records(options)
                if not records:
                    return

            batches = self._records_to_batches(records, options)

            if options.columns is not None:
                batches = (batch.select(options.columns) for batch in batches)

            if options.ignore_empty:
                batches = (batch for batch in batches if batch.num_rows > 0)

            yield from options.cast.cast_iterator(batches)

    def _collect_arrow_schema(self, full: bool = False) -> "pyarrow.Schema":
        """Infer the schema from the minimum records needed.

        * JSON: parse the whole value but only look at the first record.
        * NDJSON: parse just the first non-blank line.
        """
        del full

        with self.open() as b:
            if b.buffer.size <= 0:
                return pa.schema([])

            options = self.check_options(options=None)

            if self._is_ndjson:
                first = next(iter(self._iter_ndjson_records(options)), None)
                if first is None:
                    return pa.schema([])
                return pa.RecordBatch.from_pylist([first]).schema

            records = self._load_json_records(options)
            if not records:
                return pa.schema([])
            return pa.RecordBatch.from_pylist(records[:1]).schema

    def _write_arrow_batches(
        self,
        batches: Iterator["pyarrow.RecordBatch"],
        options: JsonOptions,
    ) -> None:
        """Serialize record batches and write them into the buffer.

        * JSON  → one JSON array (``[{…}, {…}, …]``).
        * NDJSON → one JSON object per line.
        """
        with self.open() as b:
            peeked, cast_options = options.cast.peek_source(batches)

            table = pa.Table.from_batches(
                list(peeked),
                schema=cast_options.source_schema.to_arrow_schema(),
            )
            records = table.to_pylist()

            if self._is_ndjson:
                if records:
                    lines = (
                        json_mod.dumps(
                            row,
                            encoding=options.encoding,
                            errors=options.errors,
                        )
                        for row in records
                    )
                    # json_mod.dumps returns bytes; join with newline bytes.
                    payload = b"\n".join(lines) + b"\n"
                else:
                    payload = b""
            else:
                payload = json_mod.dumps(
                    records,
                    encoding=options.encoding,
                    errors=options.errors,
                )

            b.buffer.replace_with_payload(payload)
            b.mark_dirty()

    # ------------------------------------------------------------------
    # JSON-native read fast path
    # ------------------------------------------------------------------
    #
    # _read_pylist is overridden to bypass Arrow entirely on the read side.
    # The default base-class path routes ``read_pylist → _read_arrow_table
    # → to_pylist`` which forces every value through Arrow type inference.
    # That breaks JSON round-trips in two ways:
    #
    # * Nested Arrow types that have been flattened to JSON (notably
    #   ``map_`` columns serialized as ``[[k, v], ...]`` pair lists) no
    #   longer survive ``from_pylist`` — the pair form infers to
    #   ``list<string>`` and fails when it meets the int in the second
    #   slot. There is no general way to recover ``map_`` from JSON
    #   without a caller-supplied schema.
    # * Backfilled ``None`` columns (from the union-of-keys normalization
    #   in :meth:`MediaIO._write_pylist`) come back as explicit ``None``s
    #   instead of absent keys, breaking sparse-object semantics.
    #
    # Since we already parsed the JSON, the Arrow round-trip is pure
    # overhead. The direct path preserves whatever shape the JSON holds.

    @staticmethod
    def _project_row(row: dict, columns: "Sequence[str]") -> dict:
        """Return *row* with only the requested columns, in caller order."""
        return {col: row[col] for col in columns if col in row}

    @staticmethod
    def _strip_null_keys(row: dict) -> dict:
        """Remove keys with ``None`` values from a row.

        JSON objects are intrinsically sparse — a missing key is not the
        same as a key with value ``null``, but for tabular JSON the two
        are interchangeable. When round-tripping through Arrow the
        base-class write path backfills missing keys with ``None`` so
        every row conforms to the union schema; on read, we drop those
        fill-ins so ``read_pylist(write_pylist(data))`` preserves the
        original sparse shape.
        """
        return {k: v for k, v in row.items() if v is not None}

    def _read_pylist(
        self,
        options: JsonOptions,
    ):
        batch_size = getattr(options, "batch_size", 0) or 0
        columns = options.columns

        with self.open() as b:
            if b.buffer.size <= 0:
                return [] if batch_size <= 0 else iter(())

            if self._is_ndjson:
                return self._read_pylist_ndjson(options, batch_size, columns)
            return self._read_pylist_json(options, batch_size, columns)

    def _read_pylist_json(
        self,
        options: JsonOptions,
        batch_size: int,
        columns: "Sequence[str] | None",
    ):
        records = self._load_json_records(options)

        # Strip backfilled nulls from write-path normalization, and project
        # columns if requested. Keep these two operations together so we
        # only walk each row once.
        if columns is not None:
            records = [
                self._strip_null_keys(self._project_row(row, columns))
                for row in records
            ]
        else:
            records = [self._strip_null_keys(row) for row in records]

        if batch_size <= 0:
            return records

        def iter_chunks():
            for start in range(0, len(records), batch_size):
                yield records[start : start + batch_size]

        return iter_chunks()

    def _read_pylist_ndjson(
        self,
        options: JsonOptions,
        batch_size: int,
        columns: "Sequence[str] | None",
    ):
        def iter_rows():
            for row in self._iter_ndjson_records(options):
                if columns is not None:
                    row = self._project_row(row, columns)
                yield self._strip_null_keys(row)

        if batch_size <= 0:
            return list(iter_rows())

        def iter_chunks():
            buf: list[dict] = []
            for row in iter_rows():
                buf.append(row)
                if len(buf) >= batch_size:
                    yield buf
                    buf = []
            if buf:
                yield buf

        return iter_chunks()