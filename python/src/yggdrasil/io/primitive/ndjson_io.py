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
from yggdrasil.io.base import IO

if TYPE_CHECKING:
    import polars as pl
    import pyarrow.dataset as pds


__all__ = ["NDJsonIO", "NDJsonOptions"]


#: Modes that may need to read existing bytes and merge with the
#: incoming stream. APPEND keeps the byte-level fast path when no
#: key dedup is asked for and the buffer isn't compressed; UPSERT /
#: MERGE always trigger the read-modify-rewrite path.
_MERGE_MODES = frozenset({Mode.APPEND, Mode.UPSERT, Mode.MERGE})


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


class NDJsonIO(IO[bytes, NDJsonOptions]):
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
        with self.arrow_input_stream() as v:
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
        with self.arrow_input_stream() as v:
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
        NDJSON append on uncompressed buffers without ``match_by_names``.
        With ``match_by_names`` set, or under :data:`Mode.UPSERT` /
        :data:`Mode.MERGE`, the existing payload is read back and
        merged via :func:`yggdrasil.arrow.ops.upsert_arrow_batches`
        before a single rewrite. IGNORE / ERROR_IF_EXISTS guard
        non-empty buffers.
        """
        # Mode resolution. AUTO picks UPSERT when ``match_by_names``
        # is set or APPEND otherwise — APPEND keeps the byte-level
        # fast path on uncompressed buffers, UPSERT triggers a
        # read-modify-rewrite. TRUNCATE collapses to OVERWRITE;
        # APPEND / UPSERT / MERGE keep their identity for the merge
        # branch; IGNORE / ERROR_IF_EXISTS guard the buffer.
        mode = options.mode
        if mode is Mode.AUTO:
            action = Mode.UPSERT if options.match_by_names else Mode.APPEND
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

        codec = self._codec()
        match_by = list(options.match_by_names or ())
        # Byte-append is only safe for plain APPEND, uncompressed,
        # without a key-aware merge. Anything else has to do the
        # full read-modify-rewrite dance.
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
            merged = upsert_arrow_batches(
                iter(existing),
                iter(batches),
                options.match_by_names,
                Mode.APPEND if action is Mode.APPEND else Mode.UPSERT,
                memory_pool=options.arrow_memory_pool,
            )
            return self._write_arrow_batches(
                merged, dataclasses.replace(options, mode=Mode.OVERWRITE),
            )

        line_term = options.line_ending.encode(options.encoding)

        # Existing payload may not end with a newline — guard so the
        # appended batch starts on a fresh line.
        needs_newline_prefix = (
            is_append_uncompressed
            and self.size > 0
            and self.pread(1, self.size - 1) != b"\n"
        )

        # Drive the per-row writes through the IO's
        # :meth:`arrow_output_stream`, which yields a
        # :class:`pa.BufferOutputStream`; pyarrow's sink coalesces
        # them into one contiguous Arrow buffer that gets bulk-
        # committed (overwrite or append, with codec compression
        # when set) on context exit.
        with self.arrow_output_stream(append=is_append_uncompressed) as sink:
            if needs_newline_prefix:
                sink.write(line_term)
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
