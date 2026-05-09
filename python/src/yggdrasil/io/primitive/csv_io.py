"""CSV Tabular leaf over the new :class:`BytesIO` substrate.

CSV is a pure stream — no footer, no random access, no cheap
schema. Schema collection reads the first batch; reads parse from
byte zero every call. APPEND is implemented honestly at the leaf
level (concat new bytes; suppress header on the second-and-later
session) since the format itself supports it without a footer
rewrite.
"""

from __future__ import annotations

import dataclasses
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


#: Modes that may need to read existing bytes and merge with the
#: incoming stream. APPEND, UPSERT and MERGE all share the dispatch;
#: APPEND keeps the byte-level fast path when no key dedup is asked
#: for and the buffer isn't compressed.
_MERGE_MODES = frozenset({Mode.APPEND, Mode.UPSERT, Mode.MERGE})


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
        with self.arrow_input_stream() as v:
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
        with self.arrow_input_stream() as v:
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
          OVERWRITE-with-header semantics). When
          ``options.match_by_keys`` is set, falls back to
          read-modify-rewrite so the key-aware dedup can run.
        - **UPSERT / MERGE** — read existing, merge against incoming
          via :func:`upsert_arrow_batches` (with ``match_by_names``
          driving the per-row dedup), rewrite via OVERWRITE. Without
          ``match_by_names`` collapses to plain APPEND — CSV has no
          row-level identity at this layer.
        - **IGNORE** — skip when non-empty.
        - **ERROR_IF_EXISTS** — raise when non-empty.
        """
        # Mode resolution. AUTO picks UPSERT when ``match_by_names``
        # is set or APPEND otherwise — APPEND keeps the byte-level
        # fast path on uncompressed buffers, UPSERT triggers a
        # read-modify-rewrite. TRUNCATE collapses to OVERWRITE;
        # APPEND / UPSERT / MERGE keep their identity for the merge
        # branch; IGNORE / ERROR_IF_EXISTS guard the buffer.
        mode = options.mode
        if mode is Mode.AUTO:
            action = Mode.UPSERT if options.match_by_keys else Mode.APPEND
        elif mode is Mode.TRUNCATE:
            action = Mode.OVERWRITE
        elif mode in _MERGE_MODES or mode in (
            Mode.IGNORE, Mode.ERROR_IF_EXISTS, Mode.OVERWRITE,
        ):
            action = mode
        else:
            action = Mode.OVERWRITE

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
        match_by = list(options.match_by_keys or ())
        # Byte-append is only safe for plain APPEND, uncompressed,
        # without a key-aware merge to run. Anything else has to do
        # the full read-modify-rewrite dance.
        is_append_uncompressed = (
            action is Mode.APPEND
            and self.size > 0
            and codec is None
            and not match_by
        )
        needs_rewrite = (
            action in _MERGE_MODES
            and self.size > 0
            and not is_append_uncompressed
        )

        if needs_rewrite:
            from yggdrasil.arrow.ops import upsert_arrow_batches

            existing = list(self._read_arrow_batches(options))
            incoming: Iterator[pa.RecordBatch] = iter([first, *iterator])
            merged = upsert_arrow_batches(
                iter(existing),
                incoming,
                options.match_by_keys,
                Mode.APPEND if action is Mode.APPEND else Mode.UPSERT,
                memory_pool=options.arrow_memory_pool,
            )
            return self._write_arrow_batches(
                merged, dataclasses.replace(options, mode=Mode.OVERWRITE),
            )

        schema = first.schema

        if is_append_uncompressed:
            # Append path: drive the CSVWriter against the IO's
            # :meth:`arrow_output_stream` in append mode, which seeks
            # to EOF and writes the encoded batch on context exit.
            write_options = pa_csv.WriteOptions(
                include_header=False, delimiter=options.delimiter,
            )
            with self.arrow_output_stream(append=True) as sink:
                with pa_csv.CSVWriter(
                    sink, schema, write_options=write_options,
                ) as writer:
                    if first.num_rows > 0:
                        writer.write_batch(first)
                    for batch in iterator:
                        if batch.num_rows > 0:
                            writer.write_batch(batch)
            return

        write_options = options.to_write_options()
        with self.arrow_output_stream() as sink:
            with pa_csv.CSVWriter(
                sink, schema, write_options=write_options,
            ) as writer:
                if first.num_rows > 0:
                    writer.write_batch(first)
                for batch in iterator:
                    if batch.num_rows > 0:
                        writer.write_batch(batch)

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
