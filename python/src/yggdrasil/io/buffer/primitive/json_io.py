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

import contextlib
import dataclasses
from typing import TYPE_CHECKING, ClassVar, Iterable, Iterator

import pyarrow as pa
import pyarrow.json as pa_json

from yggdrasil.data.cast.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.io.enums import MimeTypes, Mode
from yggdrasil.lazy_imports import (
    polars_module,
    pyarrow_dataset_module,
)
from .base import PrimitiveIO

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

    def to_read_options(self) -> "pa_json.ReadOptions":
        return pa_json.ReadOptions(use_threads=self.use_threads)

    def to_parse_options(self) -> "pa_json.ParseOptions":
        return pa_json.ParseOptions()


# ---------------------------------------------------------------------------
# JsonIO
# ---------------------------------------------------------------------------


class JsonIO(PrimitiveIO):
    """:class:`PrimitiveIO` for newline-delimited JSON."""

    __slots__ = ()

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    @classmethod
    def default_mime_type(cls):
        return MimeTypes.JSON

    @classmethod
    def options_class(cls):
        return JsonOptions

    _SUPPORTED_APPEND: ClassVar[bool] = True
    _SUPPORTED_UPSERT: ClassVar[bool] = True
    _NATIVE_SCANNER_OK: ClassVar[bool] = True

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

            source = io.arrow_io(mode="rb")
            reader = pa_json.open_json(
                source,
                read_options=options.to_read_options(),
                parse_options=options.to_parse_options(),
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
        options: JsonOptions,
    ) -> None:
        """Persist batches as one JSON object per line.

        pyarrow has no ``open_json_writer`` analogue; we emit lines
        manually via ``Table.to_pylist`` plus ``json.dumps``. For
        large writes prefer Parquet or IPC.
        """
        import json as _json

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

        if options.target_field is not None:
            first = options.cast_arrow_tabular(first)

        is_append = action is Mode.APPEND and not self.is_empty()

        lifecycle = options.copy(
            truncate_before_write=not is_append,
            write_seek=-1 if is_append else None,
        )

        line_sep = options.line_ending.encode(options.encoding)

        def emit(batch: pa.RecordBatch, sink) -> None:
            for row in batch.to_pylist(maps_as_pydicts=True):
                sink.write(_json.dumps(row, ensure_ascii=False).encode(options.encoding))
                sink.write(line_sep)

        with self._writing_context(lifecycle) as io:
            with contextlib.ExitStack() as stack:
                sink = io.arrow_io(mode="ab" if is_append else "wb")
                stack.callback(sink.close)

                emit(first, sink)
                for batch in iterator:
                    batch = options.cast_arrow_tabular(batch)
                    emit(batch, sink)

    # ==================================================================
    # Native engine overrides
    # ==================================================================

    def _can_use_native_scanner(self, options: JsonOptions) -> bool:
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

    def _read_arrow_dataset(self, options: JsonOptions) -> "pds.Dataset":
        if not self._can_use_native_scanner(options):
            return super()._read_arrow_dataset(options)

        pds = pyarrow_dataset_module()
        return pds.dataset(self.path.__fspath__(), format="json")

    def _scan_polars_frame(self, options: JsonOptions) -> "pl.LazyFrame":
        if not self._can_use_native_scanner(options):
            return super()._scan_polars_frame(options)

        pl = polars_module()
        return pl.scan_ndjson(self.path.__fspath__())

    def _read_polars_frame(self, options: JsonOptions) -> "pl.DataFrame":
        if not self._can_use_native_scanner(options):
            return super()._read_polars_frame(options)

        pl = polars_module()
        return pl.read_ndjson(self.path.__fspath__())