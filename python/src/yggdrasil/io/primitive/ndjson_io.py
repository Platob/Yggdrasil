"""NDJSON (newline-delimited JSON) Tabular leaf.

:class:`NDJsonIO` is the streamable counterpart to :class:`JsonIO`.
One JSON object per line, no array wrapper. APPEND is honest at
the byte level (concatenate new lines onto the existing buffer);
no read-modify-rewrite needed.

Reads use :func:`pyarrow.json.open_json`, which streams record
batches and infers types from the head of the file.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, ClassVar, Iterable, Iterator

import pyarrow as pa
import pyarrow.json as pa_json

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.lazy_imports import polars_module, pyarrow_dataset_module
from yggdrasil.io.bytes_io import BytesIO

if TYPE_CHECKING:
    import polars as pl
    import pyarrow.dataset as pds


__all__ = ["NDJsonIO", "NDJsonOptions"]


@dataclasses.dataclass(frozen=True, slots=True)
class NDJsonOptions(CastOptions):
    """:class:`CastOptions` extended with NDJSON-specific knobs."""

    use_threads: bool = True
    block_size: "int | None" = None
    encoding: str = "utf-8"
    line_ending: str = "\n"
    ensure_ascii: bool = False
    sort_keys: bool = False

    def to_read_options(self) -> "pa_json.ReadOptions":
        kwargs: dict = {"use_threads": self.use_threads}
        if self.block_size is not None:
            kwargs["block_size"] = self.block_size
        return pa_json.ReadOptions(**kwargs)

    def to_parse_options(self) -> "pa_json.ParseOptions":
        return pa_json.ParseOptions()


class NDJsonIO(BytesIO):
    """:class:`Tabular` leaf for newline-delimited JSON."""

    mime_type: ClassVar[MimeTypes] = MimeTypes.NDJSON

    @classmethod
    def options_class(cls):
        return NDJsonOptions

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
    # Schema
    # ==================================================================

    def _collect_schema(self, options: NDJsonOptions) -> Schema:
        if self.size == 0:
            return Schema.empty()
        with self._format_input() as v:
            reader = pa_json.open_json(
                v,
                read_options=options.to_read_options(),
                parse_options=options.to_parse_options(),
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
        options: NDJsonOptions,
    ) -> Iterator[pa.RecordBatch]:
        if self.size == 0:
            return
        with self._format_input() as v:
            reader = pa_json.open_json(
                v,
                read_options=options.to_read_options(),
                parse_options=options.to_parse_options(),
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
        options: NDJsonOptions,
    ) -> None:
        """Emit one JSON object per line.

        OVERWRITE truncates and writes from scratch.
        APPEND seeks to EOF and writes — concatenation is a valid
        NDJSON append. IGNORE / ERROR_IF_EXISTS guard non-empty
        buffers.
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

        codec = self._codec()
        is_append_uncompressed = (
            action is Mode.APPEND and self.size > 0 and codec is None
        )
        is_append_compressed = (
            action is Mode.APPEND and self.size > 0 and codec is not None
        )

        if is_append_compressed:
            # Codec'd buffer — read existing batches, chain new ones,
            # rewrite via OVERWRITE so the gzip frame is produced
            # whole.
            existing = list(self._read_arrow_batches(options))
            chained = iter([*existing, *batches])
            return self._write_arrow_batches(
                chained, dataclasses.replace(options, mode=Mode.OVERWRITE),
            )

        line_term = options.line_ending.encode(options.encoding)

        # Encode every row into a pure-Arrow ``BufferOutputStream``
        # before touching ``self``. Pyarrow's sink coalesces the
        # per-row writes into one contiguous Arrow buffer; we hand
        # the whole thing to :meth:`BytesIO._commit_format_payload`
        # at the end so the durable holder sees one bulk write
        # instead of one per row.
        sink = pa.BufferOutputStream()
        for batch in batches:
            for row in batch.to_pylist():
                line = json.dumps(
                    row,
                    ensure_ascii=options.ensure_ascii,
                    sort_keys=options.sort_keys,
                    default=str,
                ).encode(options.encoding)
                sink.write(line)
                sink.write(line_term)

        if is_append_uncompressed:
            # Existing payload may not end with a newline — guard
            # before bulk-appending the freshly-encoded batch.
            if self.size > 0 and self.pread(1, self.size - 1) != b"\n":
                self.seek(0, 2)
                self.write_bytes(line_term)
            self._commit_format_payload(sink.getvalue(), append=True)
            return

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

    def _read_arrow_dataset(self, options: NDJsonOptions) -> "pds.Dataset":
        pds = pyarrow_dataset_module()
        path = self._local_path_str()
        if path is not None:
            return pds.dataset(path, format="json")
        return super()._read_arrow_dataset(options)

    def _scan_polars_frame(self, options: NDJsonOptions) -> "pl.LazyFrame":
        pl = polars_module()
        path = self._local_path_str()
        if path is not None:
            return pl.scan_ndjson(path)
        return super()._scan_polars_frame(options)

    def _read_polars_frame(self, options: NDJsonOptions) -> "pl.DataFrame":
        pl = polars_module()
        path = self._local_path_str()
        if path is not None:
            return pl.read_ndjson(path)
        return super()._read_polars_frame(options)
