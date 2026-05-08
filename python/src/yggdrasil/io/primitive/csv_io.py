"""CSV Tabular leaf over the new :class:`BytesIO` substrate.

CSV is a pure stream — no footer, no random access, no cheap
schema. Schema collection reads the first batch; reads parse from
byte zero every call. APPEND is implemented honestly at the leaf
level (concat new bytes; suppress header on the second-and-later
session) since the format itself supports it without a footer
rewrite.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io as _stdlib_io
from typing import TYPE_CHECKING, ClassVar, Iterable, Iterator

import pyarrow as pa
import pyarrow.csv as pa_csv

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.lazy_imports import polars_module, pyarrow_dataset_module
from yggdrasil.io.base import IO

if TYPE_CHECKING:
    import polars as pl
    import pyarrow.dataset as pds


__all__ = ["CsvIO", "CsvOptions"]


@dataclasses.dataclass(frozen=True, slots=True)
class CsvOptions(CastOptions):
    """:class:`CastOptions` extended with CSV-specific knobs."""

    delimiter: str = ","
    quote_char: str = '"'
    escape_char: "str | None" = None
    has_header: bool = True
    skip_rows: int = 0
    null_values: "tuple[str, ...]" = ("", "N/A", "n/a", "NULL", "null", "None")
    encoding: str = "utf-8"

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


class CsvIO(IO[bytes, CsvOptions]):
    """:class:`Tabular` leaf for CSV files."""

    mime_type: ClassVar[MimeTypes] = MimeTypes.CSV

    @classmethod
    def options_class(cls):
        return CsvOptions

    # ==================================================================
    # Helpers
    # ==================================================================

    def _local_path_str(self) -> "str | None":
        holder = self._holder
        if holder is None or not getattr(holder, "is_local_path", False):
            return None
        full_path = getattr(holder, "full_path", None)
        return full_path() if full_path is not None else None

    # ==================================================================
    # Schema — read the first batch
    # ==================================================================

    def _collect_schema(self, options: CsvOptions) -> Schema:
        if self.size == 0:
            return Schema.empty()
        with self._format_input() as v:
            reader = pa_csv.open_csv(
                v,
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
        if self.size == 0:
            return
        with self._format_input() as v:
            reader = pa_csv.open_csv(
                v,
                read_options=options.to_read_options(),
                parse_options=options.to_parse_options(),
                convert_options=options.to_convert_options(),
            )
            try:
                for batch in reader:
                    yield batch
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
        """Persist Arrow record batches as CSV.

        Mode dispatch:

        - **OVERWRITE / AUTO / TRUNCATE** — single
          :class:`pa_csv.CSVWriter` over the truncated buffer with the
          configured header behavior.
        - **APPEND** — seek to EOF, write **without** a header
          (unless the buffer was empty, in which case collapse to
          OVERWRITE-with-header semantics).
        - **IGNORE** — skip when non-empty.
        - **ERROR_IF_EXISTS** — raise when non-empty.
        - **UPSERT / MERGE** — degrade to APPEND. CSV has no key
          model worth honoring at this layer.
        """
        action = self._resolve_action(options.mode)

        if action is Mode.IGNORE:
            if self.size > 0:
                return
            action = Mode.OVERWRITE
        elif action is Mode.ERROR_IF_EXISTS:
            if self.size > 0:
                raise FileExistsError(
                    f"{type(self).__name__} buffer is non-empty "
                    f"({self.size} bytes); refusing to overwrite under "
                    f"mode={options.mode!r}."
                )
            action = Mode.OVERWRITE

        iterator = iter(batches)
        first = next(iterator, None)
        if first is None and action is Mode.OVERWRITE:
            self.seek(0)
            self.truncate(0)
            return
        if first is None:
            return

        codec = self._codec()
        is_append_uncompressed = (
            action is Mode.APPEND and self.size > 0 and codec is None
        )
        is_append_compressed = (
            action is Mode.APPEND and self.size > 0 and codec is not None
        )

        if is_append_compressed:
            # Codec'd buffer can't be appended to byte-wise; fall
            # back to read-modify-rewrite via OVERWRITE chained with
            # the existing batches.
            existing = list(self._read_arrow_batches(options))
            chained = iter([*existing, first, *iterator])
            return self._write_arrow_batches(
                chained, dataclasses.replace(options, mode=Mode.OVERWRITE),
            )

        schema = first.schema

        if is_append_uncompressed:
            # Append path: still encode into an Arrow sink so the
            # CSVWriter's per-row writes don't turn into per-row
            # syscalls, then one bulk seek-to-end + write to self.
            write_options = pa_csv.WriteOptions(
                include_header=False, delimiter=options.delimiter,
            )
            sink = pa.BufferOutputStream()
            with contextlib.ExitStack() as stack:
                writer = pa_csv.CSVWriter(
                    sink, schema, write_options=write_options,
                )
                stack.callback(writer.close)
                if first.num_rows > 0:
                    writer.write_batch(first)
                for batch in iterator:
                    if batch.num_rows > 0:
                        writer.write_batch(batch)
            self._commit_format_payload(sink.getvalue(), append=True)
            return

        write_options = options.to_write_options()
        sink = pa.BufferOutputStream()
        with contextlib.ExitStack() as stack:
            writer = pa_csv.CSVWriter(sink, schema, write_options=write_options)
            stack.callback(writer.close)
            if first.num_rows > 0:
                writer.write_batch(first)
            for batch in iterator:
                if batch.num_rows > 0:
                    writer.write_batch(batch)
        self._commit_format_payload(sink.getvalue())

    def _resolve_action(self, mode: Mode) -> Mode:
        if mode is Mode.AUTO or mode is Mode.OVERWRITE or mode is Mode.TRUNCATE:
            return Mode.OVERWRITE
        if mode is Mode.APPEND:
            return Mode.APPEND
        if mode is Mode.IGNORE:
            return Mode.IGNORE
        if mode is Mode.ERROR_IF_EXISTS:
            return Mode.ERROR_IF_EXISTS
        if mode is Mode.UPSERT or mode is Mode.MERGE:
            return Mode.APPEND
        return Mode.OVERWRITE

    # ==================================================================
    # Native engine overrides
    # ==================================================================

    def _read_arrow_dataset(self, options: CsvOptions) -> "pds.Dataset":
        pds = pyarrow_dataset_module()
        path = self._local_path_str()
        if path is not None:
            return pds.dataset(path, format="csv")
        return super()._read_arrow_dataset(options)

    def _scan_polars_frame(self, options: CsvOptions) -> "pl.LazyFrame":
        pl = polars_module()
        path = self._local_path_str()
        if path is not None:
            return pl.scan_csv(
                path,
                separator=options.delimiter,
                has_header=options.has_header,
                quote_char=options.quote_char,
            )
        return super()._scan_polars_frame(options)

    def _read_polars_frame(self, options: CsvOptions) -> "pl.DataFrame":
        pl = polars_module()
        path = self._local_path_str()
        if path is not None:
            return pl.read_csv(
                path,
                separator=options.delimiter,
                has_header=options.has_header,
                quote_char=options.quote_char,
            )
        return super()._read_polars_frame(options)
