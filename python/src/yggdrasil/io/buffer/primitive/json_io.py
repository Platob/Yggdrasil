"""JSON I/O for :class:`PrimitiveIO`.

:class:`JsonIO` reads and writes newline-delimited JSON (NDJSON /
JSONL) — one JSON object per line. This is the only JSON shape
that's natively streamable and appendable.

Pretty-printed JSON arrays of objects (the ``[{...}, {...}]``
shape) are out of scope at the leaf level; transform upstream
or use a different leaf.

Save modes
----------

OVERWRITE truncates and writes fresh. APPEND seeks to end and
writes — JSONL's line-per-record framing means concatenation is
a valid append. UPSERT goes through the generic rewrite helper.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, ClassVar, Iterable, Iterator

import pyarrow as pa
import pyarrow.json as pa_json
import yggdrasil.pickle.json as json_module
from yggdrasil.arrow.cast import any_to_arrow_batch_iterator
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


__all__ = ["JsonIO", "JsonOptions"]


# ---------------------------------------------------------------------------
# JsonOptions
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class JsonOptions(CastOptions):
    """:class:`CastOptions` extended with JSON-specific knobs."""

    encoding: str = "utf-8"
    use_threads: bool = True
    line_ending: str = "\n"
    indent: int | None = None

    def to_read_options(self) -> "pa_json.ReadOptions":
        return pa_json.ReadOptions(use_threads=self.use_threads)

    def to_parse_options(self) -> "pa_json.ParseOptions":
        return pa_json.ParseOptions()


# ---------------------------------------------------------------------------
# JsonIO
# ---------------------------------------------------------------------------


class JsonIO(BytesIO):
    """:class:`PrimitiveIO` for newline-delimited JSON."""

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    @classmethod
    def default_mime_type(cls):
        return MimeTypes.JSON

    @classmethod
    def options_class(cls):
        return JsonOptions

    # ==================================================================
    # Schema
    # ==================================================================

    def _collect_schema(self, options: JsonOptions) -> Schema:
        if self.is_empty():
            return Schema.empty()

        with self._reading_context(options) as io:
            source = io.arrow_io(mode="rb")
            reader = pa_json.open_json(
                source,
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
        options: JsonOptions,
    ) -> Iterator[pa.RecordBatch]:
        with self._reading_context(options) as io:
            if io.remaining_bytes() == 0:
                return

            tail = io.tail(5)
            line_defined = tail.endswith(b"\n")

            if line_defined:
                source = io.arrow_io(mode="rb")

                try:
                    reader = pa_json.open_json(
                        source,
                        read_options=options.to_read_options(),
                        parse_options=options.to_parse_options(),
                    )
                except Exception as e:
                    raise ValueError(
                        f"Cannot parse JSON {io}:\n{io.synthetic_content()}"
                    ) from e

                try:
                    for batch in reader:
                        yield options.cast_arrow_tabular(batch)
                finally:
                    reader.close()
            else:
                parsed = json_module.load(io)

                if isinstance(parsed, list):
                    yield pa.RecordBatch.from_pylist(parsed)
                elif isinstance(parsed, dict):
                    yield pa.RecordBatch.from_pylist([parsed])
                else:
                    raise ValueError(
                        f"Cannot parse JSON {io} as list or dict: {io.synthetic_content()}"
                    )

    # ==================================================================
    # Write path
    # ==================================================================

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: JsonOptions,
    ) -> None:
        """Persist batches as one JSON object per line.

        pyarrow has no ``open_json_writer`` analogue; we emit lines
        manually via ``Table.to_pylist`` plus ``json.dumps``. For
        large writes prefer Parquet or IPC.
        """
        action = self._resolve_save_mode(options.mode)
        if action is Mode.IGNORE:
            return
        if action is Mode.APPEND:
            return self._arrow_append_via_rewrite(batches, options)
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

        all_batches = [first] + list(any_to_arrow_batch_iterator(iterator))

        collected = pa.concat_tables(
            [pa.Table.from_batches([batch]) for batch in all_batches],
            promote_options="permissive",
            memory_pool=options.arrow_memory_pool,
        ).to_pylist(maps_as_pydicts=True)

        with self._writing_context(options) as io:
            io.write_bytes(
                json_module.dumps(collected, indent=options.indent)
            )

    # ==================================================================
    # Native engine overrides
    # ==================================================================

    def _can_use_native_scanner(self, options: JsonOptions) -> bool:
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

    def _read_arrow_dataset(self, options: JsonOptions) -> "pds.Dataset":
        if not self._can_use_native_scanner(options):
            return super()._read_arrow_dataset(options)

        pds = pyarrow_dataset_module()
        return pds.dataset(self.path.__fspath__(), format="json")

    # No native polars JSON scanner — we serialize as a JSON array
    # and polars' scan_ndjson / read_ndjson expect newline-delimited
    # objects. Fall through to the base path (pyarrow JSON reader →
    # pl.from_arrow), which handles both shapes correctly.