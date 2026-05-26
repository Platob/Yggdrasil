"""JSON Tabular leaf over the new :class:`BytesIO` substrate.

:class:`JSONFile` writes a single JSON document — either an array of
objects (``[{...}, {...}, …]``) or a single object — and reads
either shape back. For the streamable line-per-record format use
:class:`NDJSONFile`.

Reads accept three on-disk shapes:

1. NDJSON-flavored input (newline at EOF) routes through
   :func:`pyarrow.json.open_json` — fast, threaded, type-inferred.
2. JSON array of objects — parsed with ``json.loads`` and emitted
   as a single :class:`pa.RecordBatch` from the python list.
3. JSON object — wrapped into a single-row batch.

Writes emit a JSON array under the OVERWRITE / AUTO mode.
APPEND / UPSERT / MERGE round-trip through OVERWRITE since a
top-level array can't be appended to without rewriting the closing
bracket.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, ClassVar, Iterable, Iterator

import pyarrow as pa
import pyarrow.json as pa_json

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.enums import MimeTypes, Mode
from yggdrasil.lazy_imports import pyarrow_dataset_module
from yggdrasil.io.base import IO

if TYPE_CHECKING:
    import pyarrow.dataset as pds


__all__ = ["JSONFile", "JsonOptions"]


@dataclasses.dataclass(frozen=True, slots=True)
class JsonOptions(CastOptions):
    """:class:`CastOptions` extended with JSON-specific knobs."""

    encoding: str = "utf-8"
    use_threads: bool = True
    indent: "int | None" = None
    ensure_ascii: bool = False
    sort_keys: bool = False

    def to_read_options(self) -> "pa_json.ReadOptions":
        return pa_json.ReadOptions(use_threads=self.use_threads)

    def to_parse_options(self) -> "pa_json.ParseOptions":
        return pa_json.ParseOptions()


class JSONFile(IO[bytes, JsonOptions]):
    """:class:`Tabular` leaf for JSON documents."""

    mime_type: ClassVar[MimeTypes] = MimeTypes.JSON

    @classmethod
    def options_class(cls):
        return JsonOptions

    # ==================================================================
    # Schema — first-batch inference (or empty on empty buffer)
    # ==================================================================

    def _collect_schema(self, options: JsonOptions) -> Schema:
        if self.size == 0:
            return Schema.empty()

        first = next(iter(self._read_arrow_batches(options)), None)
        if first is None:
            return Schema.empty()
        return Schema.from_arrow(first.schema)

    # ==================================================================
    # Read path
    # ==================================================================

    def _read_arrow_batches(
        self,
        options: JsonOptions,
    ) -> Iterator[pa.RecordBatch]:
        """Yield Arrow record batches from the JSON payload.

        Each batch is funneled through :meth:`CastOptions.cast_arrow_tabular`
        so a bound ``target_field`` reshapes the rows to the caller's
        schema before they leave the reader. When no target is bound
        the cast is a passthrough.
        """
        if self.size == 0:
            return

        # Route through ``arrow_input_stream`` so a codec on the
        # buffer's MediaType (e.g. ``application/json +
        # application/gzip`` from an HTTP response with
        # ``Content-Encoding: gzip``) is peeled before we sniff and
        # parse — pyarrow's JSON reader and ``json.loads`` both want
        # the decompressed payload.
        with self.arrow_input_stream() as v:
            size = v.size()
            if size == 0:
                return

            # Sniff the buffer shape: NDJSON-ish (line-terminated) goes
            # through pyarrow's streaming reader; everything else through
            # the standard library JSON parser.
            head = v.read_at(1, 0)
            is_array = head.lstrip().startswith(b"[")
            ends_with_newline = v.read_at(1, max(0, size - 1)).endswith(b"\n")
            if is_array or not ends_with_newline:
                yield from self._read_via_json_loads(v, options)
                return

            # Newline-terminated → NDJSON-shaped. Stream via pyarrow.
            # ``pa_json.open_json`` reads in fixed-size blocks (default
            # 1 MiB) and raises ``ArrowInvalid: straddling object …``
            # when a single record exceeds that block — which happens
            # when a single pretty-printed JSON object/array was
            # misclassified as NDJSON by the cheap sniff above. Fall
            # back to the full-buffer ``json.loads`` path in that case.
            v.seek(0)
            try:
                reader = pa_json.open_json(
                    v,
                    read_options=options.to_read_options(),
                    parse_options=options.to_parse_options(),
                )
            except pa.ArrowInvalid:
                yield from self._read_via_json_loads(v, options)
                return
            try:
                for batch in reader:
                    yield options.cast_arrow_batch(batch)
            finally:
                reader.close()

    def _read_via_json_loads(
        self,
        v,
        options: JsonOptions,
    ) -> Iterator[pa.RecordBatch]:
        """Parse the full buffer with :func:`json.loads`.

        Used both for the documented shapes (top-level array, single
        object) and as a fallback when the streaming NDJSON reader
        rejects the payload (typically a single pretty-printed JSON
        object that straddled pyarrow's block boundary).
        """
        v.seek(0)
        data = v.read()
        parsed = json.loads(data.decode(options.encoding))
        if isinstance(parsed, list):
            if parsed:
                yield options.cast_arrow_batch(
                    pa.RecordBatch.from_pylist(parsed)
                )
            return
        if isinstance(parsed, dict):
            yield options.cast_arrow_batch(
                pa.RecordBatch.from_pylist([parsed])
            )
            return
        raise ValueError(
            f"{type(self).__name__}: expected a JSON array of objects "
            f"or a single object; got {type(parsed).__name__}."
        )

    # ==================================================================
    # Write path
    # ==================================================================

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: JsonOptions,
    ) -> None:
        """Persist as a JSON array of objects.

        APPEND / UPSERT / MERGE collapse to a read-modify-rewrite
        because a top-level JSON array can't be honestly appended
        to without rewriting the closing bracket.
        """
        action = self._resolve_action(options.mode)

        _has_existing = not self.holder_is_overwrite and self.size_known and self.size > 0
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

        if action is Mode.APPEND and _has_existing:
            existing = list(self._read_arrow_batches(options))
            chained = iter([*existing, first, *iterator])
            return self._write_arrow_batches(
                chained, dataclasses.replace(options, mode=Mode.OVERWRITE),
            )

        # Bind the first batch's schema as the source so
        # :meth:`cast_arrow_tabular` reshapes each batch to a bound
        # ``target_field`` before it gets serialized — passthrough
        # when no target is set.
        cast_opts = options.check_source(first.schema)

        # Materialize all batches into one pylist; it's a JSON array
        # so the whole document goes out at once.
        rows: list[dict] = list(cast_opts.cast_arrow_batch(first).to_pylist())
        for batch in iterator:
            rows.extend(cast_opts.cast_arrow_batch(batch).to_pylist())

        text = json.dumps(
            rows,
            indent=options.indent,
            ensure_ascii=options.ensure_ascii,
            sort_keys=options.sort_keys,
            default=str,
        )
        # Drive the JSON payload through the IO's
        # :meth:`arrow_output_stream`, which yields a
        # :class:`pa.BufferOutputStream` and bulk-commits the encoded
        # payload — applying any codec on the holder's MediaType
        # (e.g. ``.json.gz``) — on context exit.
        with self.arrow_output_stream() as sink:
            sink.write(text.encode(options.encoding))

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
    # Native engine override — pyarrow.dataset(format="json")
    # ==================================================================

    def _read_arrow_dataset(self, options: JsonOptions) -> "pds.SparkDataset":
        pds = pyarrow_dataset_module()
        holder = self._parent
        if holder is not None and getattr(holder, "is_local_path", False):
            full_path = getattr(holder, "full_path", None)
            if full_path is not None:
                return pds.dataset(full_path(), format="json")
        return super()._read_arrow_dataset(options)
