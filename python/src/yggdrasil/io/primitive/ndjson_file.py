"""NDJSON (newline-delimited JSON) Tabular leaf.

:class:`NDJSONFile` is the streamable counterpart to :class:`JSONFile`.
One JSON object per line, no array wrapper. APPEND is honest at
the byte level (concatenate new lines onto the existing buffer);
no read-modify-rewrite needed.

Reads use :func:`pyarrow.json.open_json`, which streams record
batches and infers types from the head of the file.
"""

from __future__ import annotations

import dataclasses
import itertools as _it
import json
from typing import TYPE_CHECKING, Callable, ClassVar, Iterable, Iterator

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


__all__ = ["NDJSONFile", "NDJsonOptions"]


#: Modes that may need to read existing bytes and merge with the
#: incoming stream. APPEND keeps the byte-level fast path when no
#: key dedup is asked for and the buffer isn't compressed; UPSERT /
#: MERGE always trigger the read-modify-rewrite path.
_MERGE_MODES = frozenset({Mode.APPEND, Mode.UPSERT, Mode.MERGE})


def _pick_batch_encoder(
    options: "NDJsonOptions",
    line_term: bytes,
) -> "Callable[[pa.RecordBatch], bytes]":
    """Return a function that serializes a RecordBatch to NDJSON bytes.

    Tier 1 — polars ``write_ndjson``: stays in Rust, ~14x faster
    than the Python-dict path. Available when polars is installed and
    the caller uses default options (no ``ensure_ascii``, no
    ``sort_keys``, newline line-ending, UTF-8 encoding).

    Tier 2 — orjson per-row: ~3-5x faster than stdlib; used when
    polars can't satisfy the options but ``ensure_ascii`` is off.

    Tier 3 — stdlib ``json.dumps`` per-row: fallback when
    ``ensure_ascii=True``.
    """
    use_polars = (
        not options.ensure_ascii
        and not options.sort_keys
        and line_term == b"\n"
        and options.encoding == "utf-8"
    )

    if use_polars:
        try:
            pl = polars_module()
        except ImportError:
            pl = None
        if pl is not None:
            def _encode_polars(batch: pa.RecordBatch) -> bytes:
                return pl.from_arrow(batch).write_ndjson().encode("utf-8")

            return _encode_polars

    if not options.ensure_ascii:
        import orjson

        opt = 0
        if options.sort_keys:
            opt |= orjson.OPT_SORT_KEYS

        def _encode_orjson(batch: pa.RecordBatch) -> bytes:
            rows = batch.to_pylist()
            parts = [orjson.dumps(row, default=str, option=opt) for row in rows]
            parts.append(b"")
            return line_term.join(parts)

        return _encode_orjson

    encoding = options.encoding
    sort_keys = options.sort_keys

    def _encode_stdlib(batch: pa.RecordBatch) -> bytes:
        rows = batch.to_pylist()
        lines = [
            json.dumps(row, ensure_ascii=True, sort_keys=sort_keys,
                       default=str).encode(encoding)
            for row in rows
        ]
        lines.append(b"")
        return line_term.join(lines)

    return _encode_stdlib


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


class NDJSONFile(IO[bytes, NDJsonOptions]):
    """:class:`Tabular` leaf for newline-delimited JSON."""

    mime_type: ClassVar[MimeTypes] = MimeTypes.NDJSON

    @classmethod
    def options_class(cls):
        return NDJsonOptions

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
    # Schema
    # ==================================================================

    def _collect_schema(self, options: NDJsonOptions) -> Schema:
        if self.size_known and self.size == 0:
            return Schema.empty()
        try:
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
        except (FileNotFoundError, pa.ArrowInvalid):
            return Schema.empty()

    # ==================================================================
    # Read path
    # ==================================================================

    def _read_arrow_batches(
        self,
        options: NDJsonOptions,
    ) -> Iterator[pa.RecordBatch]:
        """Stream Arrow record batches out of the NDJSON reader.

        Each batch is funneled through :meth:`CastOptions.cast_arrow_tabular`
        so a bound ``target_field`` reshapes the rows to the caller's
        schema before they leave the reader. When no target is bound
        the cast is a passthrough.
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
                reader = pa_json.open_json(
                    stream,
                    read_options=options.to_read_options(),
                    parse_options=options.to_parse_options(),
                )
            except pa.ArrowInvalid:
                return
            try:
                for batch in reader:
                    yield options.cast_arrow_tabular(batch)
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
        options: NDJsonOptions,
    ) -> None:
        """Emit one JSON object per line.

        OVERWRITE truncates and writes from scratch.
        APPEND seeks to EOF and writes — concatenation is a valid
        NDJSON append on uncompressed buffers without ``match_by``.
        With ``match_by`` set, or under :data:`Mode.UPSERT` /
        :data:`Mode.MERGE`, the existing payload is read back and
        merged via :func:`yggdrasil.arrow.ops.upsert_arrow_batches`
        before a single rewrite. IGNORE / ERROR_IF_EXISTS guard
        non-empty buffers.
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
            merged = upsert_arrow_batches(
                iter(existing),
                iter(batches),
                options.match_by_keys,
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
            and _has_existing
            and self.pread(1, self.size - 1) != b"\n"
        )

        # Peek the first batch to bind a source schema so a bound
        # ``target_field`` reshapes the rows through
        # :meth:`cast_arrow_tabular` before each row is serialized —
        # passthrough when no target is bound.
        iterator = iter(batches)
        first = next(iterator, None)
        cast_opts = (
            options.check_source(first.schema) if first is not None else options
        )

        _encode_batch = _pick_batch_encoder(options, line_term)

        with self.arrow_output_stream(append=is_append_uncompressed) as sink:
            if needs_newline_prefix:
                sink.write(line_term)
            if first is None:
                return
            for batch in _it.chain([first], iterator):
                casted = cast_opts.cast_arrow_tabular(batch)
                sink.write(_encode_batch(casted))

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
            return options.cast_polars_tabular(pl.scan_ndjson(path))
        return super()._scan_polars_frame(options)

    def _read_polars_frame(self, options: NDJsonOptions) -> "pl.DataFrame":
        pl = polars_module()
        path = self._local_path_str()
        if path is not None:
            return options.cast_polars_tabular(pl.read_ndjson(path))
        return super()._read_polars_frame(options)
