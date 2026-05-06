"""XLSX Tabular leaf over the new :class:`BytesIO` substrate.

:class:`XlsxIO` handles a single-sheet xlsx workbook. Multi-sheet
workbooks are out of scope at the leaf level — they belong to a
folder/nested IO that maps each sheet onto a child fragment.

Reads use :func:`openpyxl.load_workbook(read_only=True)`; writes
use :func:`openpyxl.Workbook(write_only=True)`. Both modes are
streaming. Save modes: OVERWRITE only — a workbook is one ZIP
archive, so APPEND would defeat the streaming write path.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar, Iterable, Iterator

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.io.bytes_io import BytesIO

__all__ = ["XlsxIO", "XlsxOptions"]


@dataclasses.dataclass(frozen=True, slots=True)
class XlsxOptions(CastOptions):
    """:class:`CastOptions` extended with XLSX-specific knobs."""

    sheet_name: str = "Sheet1"
    has_header: bool = True


class XlsxIO(BytesIO):
    """:class:`Tabular` leaf for single-sheet xlsx workbooks."""

    mime_type: ClassVar[MimeTypes] = MimeTypes.XLSX

    @classmethod
    def options_class(cls):
        return XlsxOptions

    _APPEND_REJECTED_HINT: ClassVar[str] = (
        "XLSX append (whether new rows or new sheets) defeats the "
        "streaming write path — the workbook would need to be reopened "
        "in editable mode and re-saved. Use a folder-oriented writer "
        "to add new xlsx files alongside, or convert to a streamable "
        "format (parquet, jsonl) for incremental ingest."
    )

    # ==================================================================
    # Lazy openpyxl import
    # ==================================================================

    @staticmethod
    def _openpyxl():
        try:
            import openpyxl
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "XlsxIO requires openpyxl. Install with: pip install openpyxl"
            ) from e
        return openpyxl

    # ==================================================================
    # Schema
    # ==================================================================

    def _collect_schema(self, options: XlsxOptions) -> Schema:
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
        options: XlsxOptions,
    ) -> Iterator[pa.RecordBatch]:
        """Stream rows from the named sheet, batch them, yield."""
        if self.size == 0:
            return

        openpyxl = self._openpyxl()
        with self.view(pos=0) as v:
            wb = openpyxl.load_workbook(v, read_only=True, data_only=True)
            try:
                ws = (
                    wb[options.sheet_name]
                    if options.sheet_name in wb.sheetnames
                    else wb[wb.sheetnames[0]]
                )

                row_iter = ws.iter_rows(values_only=True)

                if options.has_header:
                    headers = next(row_iter, None)
                    if headers is None:
                        return
                    columns = [
                        str(h) if h is not None else f"col_{i}"
                        for i, h in enumerate(headers)
                    ]
                else:
                    first_row = next(row_iter, None)
                    if first_row is None:
                        return
                    columns = [f"col_{i}" for i in range(len(first_row))]
                    row_iter = self._chain_one(first_row, row_iter)

                row_size = options.row_size if options.row_size else 4096
                rows: list[dict] = []
                for raw_row in row_iter:
                    rows.append(dict(zip(columns, raw_row)))
                    if len(rows) >= row_size:
                        yield from self._rows_to_batches(rows)
                        rows = []
                if rows:
                    yield from self._rows_to_batches(rows)
            finally:
                wb.close()

    @staticmethod
    def _chain_one(first, rest):
        yield first
        yield from rest

    @staticmethod
    def _rows_to_batches(rows: list[dict]) -> Iterator[pa.RecordBatch]:
        if not rows:
            return
        table = pa.Table.from_pylist(rows)
        yield from table.to_batches()

    # ==================================================================
    # Write path
    # ==================================================================

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: XlsxOptions,
    ) -> None:
        """Stream rows into a write-only workbook, save into the buffer.

        openpyxl's ``write_only`` workbook can only ``save(file_like)``
        once — the workbook closes itself on save. We hand it the
        BytesIO directly.
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

        if action is not Mode.OVERWRITE:
            raise NotImplementedError(
                f"{type(self).__name__}._write_arrow_batches only handles "
                f"OVERWRITE; got {action!r}. {self._APPEND_REJECTED_HINT}"
            )

        iterator = iter(batches)
        first = next(iterator, None)
        if first is None:
            self.seek(0)
            self.truncate(0)
            return

        openpyxl = self._openpyxl()
        wb = openpyxl.Workbook(write_only=True)
        try:
            ws = wb.create_sheet(title=options.sheet_name)
            column_names = first.schema.names
            if options.has_header:
                ws.append(column_names)

            self._append_batch(ws, first, column_names)
            for batch in iterator:
                self._append_batch(ws, batch, column_names)

            self.seek(0)
            self.truncate(0)
            wb.save(self)
        finally:
            wb.close()

    @staticmethod
    def _append_batch(ws, batch: pa.RecordBatch, column_names: list[str]) -> None:
        for row in batch.to_pylist():
            ws.append([row.get(c) for c in column_names])

    def _resolve_action(self, mode: Mode) -> Mode:
        if mode is Mode.AUTO or mode is Mode.OVERWRITE or mode is Mode.TRUNCATE:
            return Mode.OVERWRITE
        if mode is Mode.IGNORE:
            return Mode.IGNORE
        if mode is Mode.ERROR_IF_EXISTS:
            return Mode.ERROR_IF_EXISTS
        # APPEND / UPSERT / MERGE all reject — XLSX has no streaming
        # append story; raise loudly.
        return mode
