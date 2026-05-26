"""CSV Tabular leaf over the new :class:`BytesIO` substrate.

CSV is a pure stream — no footer, no random access, no cheap
schema. Schema collection reads the first batch; reads parse from
byte zero every call. APPEND is implemented honestly at the leaf
level (concat new bytes; suppress header on the second-and-later
session) since the format itself supports it without a footer
rewrite.

Nested columns are serialized to JSON strings on write — pyarrow's
CSV writer rejects ``list`` / ``struct`` / ``map`` / ``dictionary``
types outright, so the only honest representation in a flat text
format is one JSON cell per row. See :func:`_encode_nested_as_json`.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, ClassVar, Iterable, Iterator

import pyarrow as pa
import pyarrow.csv as pa_csv

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.enums import MimeTypes, Mode
from yggdrasil.lazy_imports import polars_module, pyarrow_dataset_module
from yggdrasil.io.base import IO
from yggdrasil.pickle import json as yg_json

if TYPE_CHECKING:
    import polars as pl
    import pyarrow.dataset as pds


__all__ = ["CSVFile", "CsvOptions"]


#: Modes that may need to read existing bytes and merge with the
#: incoming stream. APPEND, UPSERT and MERGE all share the dispatch;
#: APPEND keeps the byte-level fast path when no key dedup is asked
#: for and the buffer isn't compressed.
_MERGE_MODES = frozenset({Mode.APPEND, Mode.UPSERT, Mode.MERGE})


def _is_nested_arrow_type(t: pa.DataType) -> bool:
    """``True`` when *t* can't be written by :class:`pa_csv.CSVWriter`.

    pyarrow's CSV encoder supports scalar primitives + temporals +
    binary/string. Anything column-shaped — list, large_list,
    fixed_size_list, struct, map, dictionary — has to be flattened
    to a single text cell before it reaches the writer. Matches the
    full set the writer rejects with ``Unsupported Type``.
    """
    return (
        pa.types.is_list(t)
        or pa.types.is_large_list(t)
        or pa.types.is_fixed_size_list(t)
        or pa.types.is_struct(t)
        or pa.types.is_map(t)
        or pa.types.is_dictionary(t)
    )


def _csv_nested_schema(schema: pa.Schema) -> "tuple[pa.Schema, tuple[int, ...]]":
    """Project *schema* onto a CSV-writable shape.

    Returns the projected schema (nested fields rewritten as
    ``string``, nullability preserved, original field metadata kept
    around so a downstream reader binding the same target can still
    parse the JSON cells back) and the index tuple of columns that
    need per-batch JSON encoding. Indices are cached upstream so we
    don't re-walk the schema for every batch.
    """
    fields: list[pa.Field] = []
    nested_indices: list[int] = []
    for i, field in enumerate(schema):
        if _is_nested_arrow_type(field.type):
            nested_indices.append(i)
            fields.append(
                pa.field(
                    field.name,
                    pa.string(),
                    nullable=field.nullable,
                    metadata=field.metadata,
                )
            )
        else:
            fields.append(field)
    return pa.schema(fields, metadata=schema.metadata), tuple(nested_indices)


def _encode_nested_as_json(
    batch: pa.RecordBatch,
    schema: pa.Schema,
    indices: "tuple[int, ...]",
    *,
    ensure_ascii: bool,
    sort_keys: bool,
) -> pa.RecordBatch:
    """Replace each nested column of *batch* with a JSON-string column.

    CSV write is a documented row-endpoint — rows are leaving Arrow
    for a flat text format anyway — so ``to_pylist`` on the nested
    columns is the canonical exemption from the no-row-loop rule
    (CLAUDE.md → "Never to_pylist heavy data" → exemption 2). The
    scalar columns sail through unchanged; only the nested ones pay
    the row hop.

    JSON encoding goes through :func:`yggdrasil.pickle.json.dumps`
    (orjson under the hood) for the type coverage we need:
    ``datetime`` / ``date`` / ``time`` / ``UUID`` / ``Decimal`` /
    ``bytes`` all serialize the way callers expect when they pop up
    inside a list or struct cell.
    """
    if not indices:
        return batch

    arrays: list[pa.Array] = list(batch.columns)
    for i in indices:
        rows = arrays[i].to_pylist()
        encoded: list[str | None] = [
            None
            if row is None
            else yg_json.dumps(
                row,
                ensure_ascii=ensure_ascii,
                sort_keys=sort_keys,
                to_bytes=False,
            )
            for row in rows
        ]
        arrays[i] = pa.array(encoded, type=pa.string())

    return pa.RecordBatch.from_arrays(arrays, schema=schema)


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

    #: When ``True`` (the default), nested columns — ``list``,
    #: ``struct``, ``map``, ``dictionary``, ``fixed_size_list`` — are
    #: serialized as JSON strings before reaching the CSV writer.
    #: Without this any nested column trips pyarrow's
    #: "Unsupported Type" guard and the write fails outright. Turn
    #: off only when a caller has already flattened the schema to
    #: scalars upstream.
    nested_as_json: bool = True
    json_ensure_ascii: bool = False
    json_sort_keys: bool = False

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


class CSVFile(IO[bytes, CsvOptions]):
    """:class:`Tabular` leaf for CSV files."""

    mime_type: ClassVar[MimeTypes] = MimeTypes.CSV

    @classmethod
    def options_class(cls):
        return CsvOptions

    # ==================================================================
    # Helpers
    # ==================================================================

    def _local_path_str(self) -> "str | None":
        holder = self._parent
        if holder is None or not getattr(holder, "is_local_path", False):
            return None
        full_path = getattr(holder, "full_path", None)
        return full_path() if full_path is not None else None

    # ==================================================================
    # Schema — read the first batch
    # ==================================================================

    def _collect_schema(self, options: CsvOptions) -> Schema:
        if self.size_known and self.size == 0:
            return Schema.empty()
        try:
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
        except (FileNotFoundError, pa.ArrowInvalid):
            return Schema.empty()

    # ==================================================================
    # Read path
    # ==================================================================

    def _read_arrow_batches(
        self,
        options: CsvOptions,
    ) -> Iterator[pa.RecordBatch]:
        """Stream Arrow record batches out of the CSV reader.

        Each batch is funneled through :meth:`CastOptions.cast_arrow_tabular`
        so a bound ``target_field`` reshapes the rows to the caller's
        schema before they leave the reader — useful when the CSV's
        inferred types (everything looks like a string until pyarrow
        sniffs) need to be coerced to a stricter target. When no
        target is bound the cast is a passthrough.
        """
        if self.size_known and self.size == 0:
            return
        try:
            stream_ctx = self.arrow_input_stream()
            stream = stream_ctx.__enter__()
        except FileNotFoundError:
            return
        try:
            try:
                reader = pa_csv.open_csv(
                    stream,
                    read_options=options.to_read_options(),
                    parse_options=options.to_parse_options(),
                    convert_options=options.to_convert_options(),
                )
            except pa.ArrowInvalid:
                return
            try:
                for batch in reader:
                    yield options.cast_arrow_batch(batch)
            finally:
                reader.close()
        finally:
            stream_ctx.__exit__(None, None, None)

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
          via :func:`upsert_arrow_batches` (with ``match_by``
          driving the per-row dedup), rewrite via OVERWRITE. Without
          ``match_by`` collapses to plain APPEND — CSV has no
          row-level identity at this layer.
        - **IGNORE** — skip when non-empty.
        - **ERROR_IF_EXISTS** — raise when non-empty.
        """
        # Mode resolution. AUTO picks UPSERT when ``match_by``
        # is set or APPEND otherwise — APPEND keeps the byte-level
        # fast path on uncompressed buffers, UPSERT triggers a
        # read-modify-rewrite. TRUNCATE collapses to OVERWRITE;
        # APPEND / UPSERT / MERGE keep their identity for the merge
        # branch; IGNORE / ERROR_IF_EXISTS guard the buffer.
        mode = options.mode
        _skip_existing = self.holder_is_overwrite
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

        _has_existing = not _skip_existing and self.size_known and self.size > 0
        if action is Mode.IGNORE:
            if _has_existing:
                return
            action = Mode.OVERWRITE
        elif action is Mode.ERROR_IF_EXISTS:
            if _has_existing:
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
        is_append_uncompressed = (
            action is Mode.APPEND
            and _has_existing
            and codec is None
            and not match_by
        )
        needs_rewrite = (
            action in _MERGE_MODES
            and _has_existing
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

        # Bind the first batch's schema as the source so
        # :attr:`CastOptions.merged_schema` resolves to the writer
        # schema even when no target_field is set; each batch is
        # cast through :meth:`cast_arrow_tabular` so a bound target
        # reshapes the rows to the caller's schema before the
        # encoder sees them.
        cast_opts = options.check_source(first.schema)
        first_casted = cast_opts.cast_arrow_batch(first)

        # CSV can't encode nested arrow types — pyarrow's CSVWriter
        # raises ``ArrowInvalid: Unsupported Type`` on the first
        # ``list`` / ``struct`` / ``map`` cell. Project the post-cast
        # schema onto a CSV-writable shape (nested fields rewritten
        # as ``string``) once and feed every batch through
        # :func:`_encode_nested_as_json` so the writer sees pure
        # scalar / string columns. When the schema has no nested
        # columns ``nested_indices`` is empty and the encoder is a
        # passthrough.
        writer_schema = cast_opts.merged.to_arrow_schema()
        nested_indices: tuple[int, ...] = ()
        if options.nested_as_json:
            writer_schema, nested_indices = _csv_nested_schema(writer_schema)
            if nested_indices:
                first_casted = _encode_nested_as_json(
                    first_casted,
                    writer_schema,
                    nested_indices,
                    ensure_ascii=options.json_ensure_ascii,
                    sort_keys=options.json_sort_keys,
                )

        schema = writer_schema

        if is_append_uncompressed:
            # Append path: drive the CSVWriter against the IO's
            # :meth:`arrow_output_stream` in append mode, which seeks
            # to EOF and writes the encoded batch on context exit.
            csv_write_options = pa_csv.WriteOptions(
                include_header=False, delimiter=options.delimiter,
            )
            with self.arrow_output_stream(append=True) as sink:
                with pa_csv.CSVWriter(
                    sink, schema, write_options=csv_write_options,
                ) as writer:
                    if first_casted.num_rows > 0:
                        writer.write_batch(first_casted)
                    for batch in iterator:
                        casted = cast_opts.cast_arrow_batch(batch)
                        if nested_indices:
                            casted = _encode_nested_as_json(
                                casted,
                                writer_schema,
                                nested_indices,
                                ensure_ascii=options.json_ensure_ascii,
                                sort_keys=options.json_sort_keys,
                            )
                        if casted.num_rows > 0:
                            writer.write_batch(casted)
            return

        csv_write_options = options.to_write_options()
        with self.arrow_output_stream() as sink:
            with pa_csv.CSVWriter(
                sink, schema, write_options=csv_write_options,
            ) as writer:
                if first_casted.num_rows > 0:
                    writer.write_batch(first_casted)
                for batch in iterator:
                    casted = cast_opts.cast_arrow_batch(batch)
                    if nested_indices:
                        casted = _encode_nested_as_json(
                            casted,
                            writer_schema,
                            nested_indices,
                            ensure_ascii=options.json_ensure_ascii,
                            sort_keys=options.json_sort_keys,
                        )
                    if casted.num_rows > 0:
                        writer.write_batch(casted)

    # ==================================================================
    # Native engine overrides
    # ==================================================================

    def _read_arrow_dataset(self, options: CsvOptions) -> "pds.SparkDataset":
        pds = pyarrow_dataset_module()
        path = self._local_path_str()
        if path is not None:
            return pds.dataset(path, format="csv")
        return super()._read_arrow_dataset(options)

    def _scan_polars_frame(self, options: CsvOptions) -> "pl.LazyFrame":
        pl = polars_module()
        path = self._local_path_str()
        if path is not None:
            lf = pl.scan_csv(
                path,
                separator=options.delimiter,
                has_header=options.has_header,
                quote_char=options.quote_char,
            )
            return options.cast_polars_tabular(lf)
        return super()._scan_polars_frame(options)

    def _read_polars_frame(self, options: CsvOptions) -> "pl.DataFrame":
        pl = polars_module()
        path = self._local_path_str()
        if path is not None:
            df = pl.read_csv(
                path,
                separator=options.delimiter,
                has_header=options.has_header,
                quote_char=options.quote_char,
            )
            return options.cast_polars_tabular(df)
        return super()._read_polars_frame(options)
