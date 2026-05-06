"""JSON Tabular leaf over the new :class:`BytesIO` substrate.

:class:`JsonIO` writes a single JSON document — either an array of
objects (``[{...}, {...}, …]``) or a single object — and reads
either shape back. For the streamable line-per-record format use
:class:`NDJsonIO`.

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
from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.lazy_imports import pyarrow_dataset_module
from yggdrasil.io.bytes_io import BytesIO

if TYPE_CHECKING:
    import pyarrow.dataset as pds


__all__ = ["JsonIO", "JsonOptions"]


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


class JsonIO(BytesIO):
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
        if self.size == 0:
            return

        # Sniff the buffer shape: NDJSON-ish (line-terminated) goes
        # through pyarrow's streaming reader; everything else through
        # the standard library JSON parser.
        head = self.pread(1, 0)
        is_array = head.lstrip().startswith(b"[")
        if is_array or not self.pread(1, max(0, self.size - 1)).endswith(b"\n"):
            data = self.to_bytes()
            parsed = json.loads(data.decode(options.encoding))
            if isinstance(parsed, list):
                if parsed:
                    yield pa.RecordBatch.from_pylist(parsed)
                return
            if isinstance(parsed, dict):
                yield pa.RecordBatch.from_pylist([parsed])
                return
            raise ValueError(
                f"{type(self).__name__}: expected a JSON array of objects "
                f"or a single object; got {type(parsed).__name__}."
            )

        # Newline-terminated → NDJSON-shaped. Stream via pyarrow.
        with self.view(pos=0) as v:
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
        options: JsonOptions,
    ) -> None:
        """Persist as a JSON array of objects.

        APPEND / UPSERT / MERGE collapse to a read-modify-rewrite
        because a top-level JSON array can't be honestly appended
        to without rewriting the closing bracket.
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

        if action is Mode.APPEND and self.size > 0:
            existing = list(self._read_arrow_batches(options))
            chained = iter([*existing, first, *iterator])
            return self._write_arrow_batches(
                chained, dataclasses.replace(options, mode=Mode.OVERWRITE),
            )

        # Materialize all batches into one pylist; it's a JSON array
        # so the whole document goes out at once.
        rows: list[dict] = list(first.to_pylist())
        for batch in iterator:
            rows.extend(batch.to_pylist())

        text = json.dumps(
            rows,
            indent=options.indent,
            ensure_ascii=options.ensure_ascii,
            sort_keys=options.sort_keys,
            default=str,
        )
        self.seek(0)
        self.truncate(0)
        self.write(text.encode(options.encoding))

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

    def _read_arrow_dataset(self, options: JsonOptions) -> "pds.Dataset":
        pds = pyarrow_dataset_module()
        holder = self._holder
        if holder is not None and getattr(holder, "is_local_path", False):
            full_path = getattr(holder, "full_path", None)
            if full_path is not None:
                return pds.dataset(full_path(), format="json")
        return super()._read_arrow_dataset(options)
