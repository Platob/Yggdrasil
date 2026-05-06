"""CSV I/O for :class:`PrimitiveIO`.

:class:`CsvIO` is the concrete leaf for CSV files. Unlike the
footer-indexed formats (Parquet, Arrow IPC), CSV is a pure stream:
no random access, no metadata, no cheap schema. Reads parse from
byte zero every time; schema collection is "read the first batch."

This makes CSV the simplest leaf and also the slowest for repeated
queries — the format pays for its human-readability with everything
else. The native engine overrides help (``pds.dataset(format="csv")``
and ``pl.scan_csv`` will at least skip columns at parse time) but
the underlying parser still touches every byte.

Save modes
----------

CSV supports honest append: concatenate the new bytes to the
existing file, optionally skipping the header row on the second-
and-later sessions. APPEND is implemented at the leaf level
(unlike Parquet / IPC where it's a NestedIO concern) because the
format itself supports it without footer rewrites.

Lifecycle, codec, and Mode resolution all live on
:class:`DataIO` / :class:`PrimitiveIO`. This leaf only owns the
format-specific options and the read/write bodies.
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import TYPE_CHECKING, ClassVar, Iterable, Iterator

import pyarrow as pa
import pyarrow.csv as pa_csv

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.io.enums import MimeTypes, Mode
from yggdrasil.lazy_imports import (
    polars_module,
    pyarrow_dataset_module,
)
from yggdrasil.io.buffer.bytes_io import BytesIO

if TYPE_CHECKING:
    import polars as pl
    import pyarrow.dataset as pds


__all__ = ["CsvIO", "CsvOptions"]


# ---------------------------------------------------------------------------
# CsvOptions
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class CsvOptions(CastOptions):
    """:class:`CastOptions` extended with CSV-specific knobs."""

    # Reader knobs
    delimiter: str = ","
    quote_char: str = '"'
    escape_char: "str | None" = None
    has_header: bool = True
    skip_rows: int = 0
    null_values: "tuple[str, ...]" = ("", "N/A", "n/a", "NULL", "null", "None")
    encoding: str = "utf-8"

    # Writer knobs
    write_header: bool = True
    line_ending: str = "\n"

    def to_read_options(self) -> "pa_csv.ReadOptions":
        return pa_csv.ReadOptions(
            use_threads=True,
            skip_rows=self.skip_rows,
            autogenerate_column_names=not self.has_header,
            encoding=self.encoding,
        )

    def to_parse_options(self) -> "pa_csv.ParseOptions":
        return pa_csv.ParseOptions(
            delimiter=self.delimiter,
            quote_char=self.quote_char,
            escape_char=self.escape_char if self.escape_char else False,
        )

    def to_convert_options(self) -> "pa_csv.ConvertOptions":
        return pa_csv.ConvertOptions(
            null_values=list(self.null_values),
            strings_can_be_null=True,
        )

    def to_write_options(self) -> "pa_csv.WriteOptions":
        return pa_csv.WriteOptions(
            include_header=self.write_header,
            delimiter=self.delimiter,
        )


# ---------------------------------------------------------------------------
# CsvIO
# ---------------------------------------------------------------------------


class CsvIO(BytesIO):
    """:class:`PrimitiveIO` for CSV files."""

    # No cached reader — CSV has no footer to amortize.
    _FINAL_TABULAR_IO: ClassVar[bool] = True

    @classmethod
    def default_media_type(cls):
        return MimeTypes.CSV

    @classmethod
    def options_class(cls):
        return CsvOptions

    _SUPPORTED_APPEND: ClassVar[bool] = True
    _SUPPORTED_UPSERT: ClassVar[bool] = True
    _NATIVE_SCANNER_OK: ClassVar[bool] = True

    # ==================================================================
    # Schema — read the first batch
    # ==================================================================

    def _collect_schema(self, options: CsvOptions) -> Schema:
        """Read the schema from the first batch.

        CSV has no metadata; the parser must read at least one row
        of data to type-infer.
        """
        if self.is_empty():
            return Schema.empty()

        with self._reading_context(options) as io:
            source = io.arrow_io(mode="rb")
            reader = pa_csv.open_csv(
                source,
                read_options=options.to_read_options(),
                parse_options=options.to_parse_options(),
                convert_options=options.to_convert_options(),
            )
            try:
                return Schema.from_arrow(reader.schema)
            finally:
                reader.close()

    # ==================================================================
    # Read path
    # ==================================================================

    def _read_arrow_batches(
        self,
        options: CsvOptions,
    ) -> Iterator[pa.RecordBatch]:
        """Yield Arrow record batches from a streaming CSV parse."""
        if self.is_empty():
            return

        with self._reading_context(options) as io:
            source = io.arrow_io(mode="rb")
            reader = pa_csv.open_csv(
                source,
                read_options=options.to_read_options(),
                parse_options=options.to_parse_options(),
                convert_options=options.to_convert_options(),
            )
            try:
                for batch in reader:
                    yield options.cast_arrow_tabular(batch)
            finally:
                reader.close()

    # ==================================================================
    # Write path
    # ==================================================================

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: CsvOptions,
    ) -> None:
        """Persist Arrow record batches as a CSV file.

        OVERWRITE goes through one writer session over a truncated
        buffer. APPEND seeks to end and writes without a header
        (unless the existing buffer is empty, which collapses to
        OVERWRITE-with-header semantics). UPSERT goes through the
        generic read-modify-write helper on :class:`DataIO`.
        """
        action = self._resolve_save_mode(options.mode)
        if action is Mode.IGNORE:
            return
        if action is Mode.UPSERT:
            return self._arrow_upsert_via_rewrite(batches, options)
        if action not in (Mode.OVERWRITE, Mode.APPEND):
            raise NotImplementedError(
                f"{type(self).__name__}._write_arrow_batches handles "
                f"OVERWRITE / APPEND / UPSERT; got resolved action "
                f"{action!r}."
            )

        iterator = iter(batches)
        first = next(iterator, None)
        if first is None:
            return

        first = options.cast_arrow_tabular(first)
        schema = options.check_source(first).check_target(first).merged_schema.to_arrow_schema()

        # Decide append-vs-overwrite at the byte level. APPEND on an
        # empty buffer is overwrite-with-header.
        is_append = action is Mode.APPEND and not self.is_empty()

        if is_append:
            write_options = pa_csv.WriteOptions(
                include_header=False,
                delimiter=options.delimiter,
            )
            sink_mode = "ab"
        else:
            write_options = options.to_write_options()
            sink_mode = "wb"

        lifecycle = options.copy(
            truncate_before_write=not is_append,
            write_seek=-1 if is_append else options.write_seek,
        )

        with self._writing_context(lifecycle) as io:
            with contextlib.ExitStack() as stack:
                sink = io.arrow_io(mode=sink_mode)
                stack.callback(sink.close)
                writer = pa_csv.CSVWriter(sink, schema, write_options=write_options)
                stack.callback(writer.close)

                try:
                    writer.write_batch(first)
                    for batch in iterator:
                        batch = options.cast_arrow_tabular(batch)
                        writer.write_batch(batch)
                finally:
                    writer.close()

        return None

    # ==================================================================
    # Native engine overrides
    # ==================================================================

    def _can_use_native_scanner(self, options: CsvOptions) -> bool:
        """True iff the native CSV scanners can serve *options*."""
        if not type(self)._NATIVE_SCANNER_OK:
            return False
        if self.is_empty():
            return False
        if options.target_field is not None:
            return False
        if self.codec is not None:
            return False
        if self.path is None:
            return False
        if not self.path.is_local:
            return False
        return True

    def _read_arrow_dataset(self, options: CsvOptions) -> "pds.Dataset":
        if not self._can_use_native_scanner(options):
            return super()._read_arrow_dataset(options)

        pds = pyarrow_dataset_module()
        return pds.dataset(self.path.__fspath__(), format="csv")

    def _scan_polars_frame(self, options: CsvOptions) -> "pl.LazyFrame":
        if not self._can_use_native_scanner(options):
            return super()._scan_polars_frame(options)

        pl = polars_module()
        return pl.scan_csv(
            self.path.__fspath__(),
            separator=options.delimiter,
            has_header=options.has_header,
            quote_char=options.quote_char,
        )

    def _read_polars_frame(self, options: CsvOptions) -> "pl.DataFrame":
        if not self._can_use_native_scanner(options):
            return super()._read_polars_frame(options)

        pl = polars_module()
        return pl.read_csv(
            self.path.__fspath__(),
            separator=options.delimiter,
            has_header=options.has_header,
            quote_char=options.quote_char,
        )