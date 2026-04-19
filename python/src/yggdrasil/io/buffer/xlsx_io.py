"""XLSX (Office Open XML Spreadsheet) I/O on top of :class:`~yggdrasil.io.buffer.BytesIO`.

Reading tries, in order:

1. **polars** (``pl.read_excel``) — tries the ``calamine`` engine first, then
   ``openpyxl``, then ``xlsx2csv``. Fastest when ``fastexcel`` is installed.
2. **pandas** (``pd.read_excel`` with ``openpyxl``) — the usual fallback when
   polars isn't available.
3. **Built-in fallback** — a minimal :mod:`zipfile` + :mod:`xml.etree.ElementTree`
   parser. Covers shared strings, inline strings, numbers, booleans, dates
   (stored as ISO strings) and ``null`` cells. Enough to read files produced
   by common writers without pulling an optional dependency.

Writing tries, in order:

1. **polars** (``df.write_excel`` → ``xlsxwriter``)
2. **pandas** (``df.to_excel`` → ``openpyxl`` or ``xlsxwriter``)
3. **Built-in fallback** — a minimal XLSX writer that composes the required
   OOXML parts into a ZIP archive using only the standard library.

Transport-level compression is handled transparently by the base class, but
XLSX files are already ZIP-compressed containers so an outer codec is rarely
useful.
"""
from __future__ import annotations

import io as _io
import re as _re
import zipfile as _zipfile
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import TYPE_CHECKING, Any, Iterator, Optional
from xml.etree import ElementTree as _ET

import pyarrow as pa

from .media_io import MediaIO
from .media_options import MediaOptions

if TYPE_CHECKING:
    import pyarrow

__all__ = ["XlsxOptions", "XlsxIO"]


_ENGINES = ("auto", "polars", "pandas", "fallback")
_CELL_RE = _re.compile(r"^([A-Z]+)(\d+)$")
_NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _col_letter_to_index(letters: str) -> int:
    """Return the 0-based column index for an Excel column letter (``A`` → 0)."""
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _index_to_col_letter(index: int) -> str:
    """Return the Excel column letter for a 0-based column index (0 → ``A``)."""
    letters = ""
    n = index + 1
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return letters


def _xml_escape(text: str) -> str:
    """Escape XML text content (``&``, ``<``, ``>``)."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


@dataclass
class XlsxOptions(MediaOptions):
    """Options for XLSX I/O.

    Parameters
    ----------
    sheet_name:
        Sheet to read. When ``None`` (the default) the first sheet is used.
        Accepts a sheet name (``str``) or a 0-based sheet index (``int``).
    sheet_id:
        Alias for *sheet_name* when given as an int index. Kept for parity
        with :mod:`polars`. Ignored when *sheet_name* is set.
    has_header:
        Treat the first data row as the column names. Defaults to ``True``.
    include_header:
        Write a header row. Defaults to ``True``.
    skip_rows:
        Number of leading rows to skip before the header/data. Defaults to 0.
    write_sheet_name:
        Sheet name to create when writing. Defaults to ``"Sheet1"``.
    engine:
        Force a specific backend: ``"polars"``, ``"pandas"``, ``"fallback"``
        or ``"auto"`` (the default). ``"auto"`` tries polars, then pandas,
        then the built-in fallback.
    """

    sheet_name: str | int | None = None
    sheet_id: int | None = None
    has_header: bool = True
    include_header: bool = True
    skip_rows: int = 0
    write_sheet_name: str = "Sheet1"
    engine: str = "auto"

    def __post_init__(self) -> None:
        """Normalize and validate XLSX-specific options."""
        super().__post_init__()

        if self.sheet_name is not None and not isinstance(self.sheet_name, (str, int)):
            raise TypeError(
                f"sheet_name must be str|int|None, got {type(self.sheet_name).__name__}"
            )
        if isinstance(self.sheet_name, bool):
            raise TypeError("sheet_name must be str|int|None, got bool")

        if self.sheet_id is not None:
            if isinstance(self.sheet_id, bool) or not isinstance(self.sheet_id, int):
                raise TypeError(
                    f"sheet_id must be int|None, got {type(self.sheet_id).__name__}"
                )
            if self.sheet_id < 0:
                raise ValueError(f"sheet_id must be >= 0, got {self.sheet_id}")

        if not isinstance(self.has_header, bool):
            raise TypeError(
                f"has_header must be bool, got {type(self.has_header).__name__}"
            )
        if not isinstance(self.include_header, bool):
            raise TypeError(
                f"include_header must be bool, got {type(self.include_header).__name__}"
            )

        if not isinstance(self.write_sheet_name, str) or not self.write_sheet_name:
            raise ValueError("write_sheet_name must be a non-empty str")

        if not isinstance(self.engine, str):
            raise TypeError(f"engine must be str, got {type(self.engine).__name__}")
        engine = self.engine.lower()
        if engine not in _ENGINES:
            raise ValueError(
                f"engine must be one of {_ENGINES}, got {self.engine!r}"
            )
        self.engine = engine

    @classmethod
    def resolve(cls, *, options: "XlsxOptions | None" = None, **overrides: Any) -> "XlsxOptions":
        """Merge *overrides* into *options* (or a fresh default)."""
        return cls.check_parameters(options=options, **overrides)


@dataclass(slots=True)
class XlsxIO(MediaIO[XlsxOptions]):
    """XLSX I/O with polars → pandas → built-in fallback dispatch."""

    @classmethod
    def check_options(
        cls,
        options: Optional[XlsxOptions],
        *args,
        **kwargs,
    ) -> XlsxOptions:
        """Validate and merge caller-supplied options."""
        return XlsxOptions.check_parameters(options=options, **kwargs)

    # ------------------------------------------------------------------
    # Sheet target resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _sheet_target(options: XlsxOptions) -> str | int:
        """Return the sheet identifier to request from the backend.

        Resolves the ambiguity between *sheet_name* and *sheet_id*: when
        *sheet_name* is set it wins; otherwise *sheet_id* is used; otherwise
        the first sheet (index ``0``) is returned.
        """
        if options.sheet_name is not None:
            return options.sheet_name
        if options.sheet_id is not None:
            return options.sheet_id
        return 0

    # ------------------------------------------------------------------
    # Read dispatch
    # ------------------------------------------------------------------

    def _read_arrow_batches(
        self,
        *,
        options: XlsxOptions,
    ) -> Iterator["pyarrow.RecordBatch"]:
        """Yield record batches from the XLSX buffer using the best backend."""
        if self.buffer.size <= 0:
            return

        raw = self.buffer.to_bytes()
        table = self._read_table(raw, options=options)

        if options.columns is not None:
            keep = [c for c in options.columns if c in table.column_names]
            table = table.select(keep)

        if table.num_rows == 0 and options.ignore_empty:
            return

        yield from table.to_batches()

    def _read_table(self, raw: bytes, *, options: XlsxOptions) -> "pyarrow.Table":
        """Read XLSX bytes into an Arrow table, picking the best backend."""
        engine = options.engine
        errors: list[str] = []

        if engine in ("auto", "polars"):
            try:
                return self._read_polars(raw, options=options)
            except ImportError as e:
                errors.append(f"polars: {e}")
                if engine == "polars":
                    raise
            except Exception as e:
                errors.append(f"polars: {e}")
                if engine == "polars":
                    raise

        if engine in ("auto", "pandas"):
            try:
                return self._read_pandas(raw, options=options)
            except ImportError as e:
                errors.append(f"pandas: {e}")
                if engine == "pandas":
                    raise
            except Exception as e:
                errors.append(f"pandas: {e}")
                if engine == "pandas":
                    raise

        if engine in ("auto", "fallback"):
            try:
                return self._read_fallback(raw, options=options)
            except Exception as e:
                errors.append(f"fallback: {e}")
                if engine == "fallback":
                    raise

        raise RuntimeError(
            "XlsxIO: could not read workbook with any backend. "
            f"Tried: {', '.join(errors) if errors else 'none'}"
        )

    # ------------------------------------------------------------------
    # Polars backend
    # ------------------------------------------------------------------

    @staticmethod
    def _read_polars(raw: bytes, *, options: XlsxOptions) -> "pyarrow.Table":
        """Read via :func:`polars.read_excel`, trying each engine in order."""
        from yggdrasil.polars.lib import polars as pl

        target = XlsxIO._sheet_target(options)
        kwargs: dict[str, Any] = {"has_header": options.has_header}
        if isinstance(target, str):
            kwargs["sheet_name"] = target
        else:
            kwargs["sheet_id"] = int(target) + 1
        if options.skip_rows:
            kwargs["read_options"] = {"skip_rows": options.skip_rows}

        last_err: Exception | None = None
        for pl_engine in ("calamine", "openpyxl", "xlsx2csv"):
            try:
                df = pl.read_excel(_io.BytesIO(raw), engine=pl_engine, **kwargs)
                break
            except ImportError as e:
                last_err = e
                continue
            except ModuleNotFoundError as e:
                last_err = e
                continue
        else:
            raise ImportError(
                f"polars.read_excel requires one of calamine/openpyxl/xlsx2csv: {last_err}"
            )

        if isinstance(df, dict):
            df = next(iter(df.values()))
        return df.to_arrow()

    # ------------------------------------------------------------------
    # Pandas backend
    # ------------------------------------------------------------------

    @staticmethod
    def _read_pandas(raw: bytes, *, options: XlsxOptions) -> "pyarrow.Table":
        """Read via :func:`pandas.read_excel` (openpyxl-backed)."""
        from yggdrasil.pandas.lib import pandas as pd

        target = XlsxIO._sheet_target(options)
        header: int | None = options.skip_rows if options.has_header else None
        skiprows = None if options.has_header else options.skip_rows

        frame = pd.read_excel(
            _io.BytesIO(raw),
            sheet_name=target,
            header=header,
            skiprows=skiprows,
            engine="openpyxl",
        )
        if isinstance(frame, dict):
            frame = next(iter(frame.values()))

        if not options.has_header:
            frame.columns = [f"f{i}" for i in range(frame.shape[1])]

        return pa.Table.from_pandas(frame, preserve_index=False)

    # ------------------------------------------------------------------
    # Built-in fallback reader
    # ------------------------------------------------------------------

    @classmethod
    def _read_fallback(cls, raw: bytes, *, options: XlsxOptions) -> "pyarrow.Table":
        """Read XLSX using only the stdlib (zipfile + ElementTree)."""
        with _zipfile.ZipFile(_io.BytesIO(raw)) as zf:
            shared = cls._parse_shared_strings(zf)
            sheet_path = cls._resolve_sheet_path(zf, options)
            rows = cls._parse_sheet_rows(zf, sheet_path, shared)

        rows = rows[options.skip_rows:]
        if not rows:
            return pa.table({})

        width = max(len(r) for r in rows)
        if options.has_header:
            header_row = rows[0]
            headers = [
                str(header_row[i]) if i < len(header_row) and header_row[i] is not None
                else f"f{i}"
                for i in range(width)
            ]
            data_rows = rows[1:]
        else:
            headers = [f"f{i}" for i in range(width)]
            data_rows = rows

        records: list[dict] = []
        for row in data_rows:
            padded = list(row) + [None] * (width - len(row))
            records.append({headers[i]: padded[i] for i in range(width)})

        if not records:
            return pa.Table.from_pydict({h: [] for h in headers})
        return pa.Table.from_pylist(records)

    @staticmethod
    def _parse_shared_strings(zf: _zipfile.ZipFile) -> list[str]:
        """Return the shared-strings table (``xl/sharedStrings.xml``)."""
        try:
            data = zf.read("xl/sharedStrings.xml")
        except KeyError:
            return []

        strings: list[str] = []
        root = _ET.fromstring(data)
        for si in root.findall("main:si", _NS):
            text_parts: list[str] = []
            for t in si.iter(f"{{{_NS['main']}}}t"):
                text_parts.append(t.text or "")
            strings.append("".join(text_parts))
        return strings

    @classmethod
    def _resolve_sheet_path(cls, zf: _zipfile.ZipFile, options: XlsxOptions) -> str:
        """Return the zip path for the requested sheet."""
        wb = _ET.fromstring(zf.read("xl/workbook.xml"))
        sheets_el = wb.find("main:sheets", _NS)
        if sheets_el is None:
            raise ValueError("XLSX fallback reader: workbook has no <sheets> element")

        sheets: list[tuple[str, str]] = []
        for s in sheets_el.findall("main:sheet", _NS):
            name = s.get("name", "")
            rid = s.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id", "")
            sheets.append((name, rid))

        if not sheets:
            raise ValueError("XLSX fallback reader: workbook declares no sheets")

        target = cls._sheet_target(options)
        if isinstance(target, str):
            match = next((rid for n, rid in sheets if n == target), None)
            if match is None:
                names = [n for n, _ in sheets]
                raise ValueError(
                    f"XLSX fallback reader: sheet {target!r} not found. Available: {names}"
                )
            target_rid = match
        else:
            if target >= len(sheets):
                raise IndexError(
                    f"XLSX fallback reader: sheet index {target} out of range "
                    f"(workbook has {len(sheets)} sheets)"
                )
            target_rid = sheets[target][1]

        rels = _ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        for rel in rels:
            if rel.get("Id") == target_rid:
                path = rel.get("Target", "")
                if path.startswith("/"):
                    return path.lstrip("/")
                return f"xl/{path}" if not path.startswith("xl/") else path

        raise ValueError(
            f"XLSX fallback reader: relationship {target_rid!r} missing from workbook rels"
        )

    @staticmethod
    def _parse_sheet_rows(
        zf: _zipfile.ZipFile,
        sheet_path: str,
        shared: list[str],
    ) -> list[list[Any]]:
        """Return the sheet as a list of row lists (values in column order)."""
        data = zf.read(sheet_path)
        root = _ET.fromstring(data)
        sheet_data = root.find("main:sheetData", _NS)
        if sheet_data is None:
            return []

        rows: list[list[Any]] = []
        for row_el in sheet_data.findall("main:row", _NS):
            row: list[Any] = []
            next_col = 0
            for cell in row_el.findall("main:c", _NS):
                ref = cell.get("r", "")
                m = _CELL_RE.match(ref) if ref else None
                if m:
                    col_idx = _col_letter_to_index(m.group(1))
                else:
                    col_idx = next_col

                while len(row) < col_idx:
                    row.append(None)

                row.append(_decode_cell(cell, shared))
                next_col = col_idx + 1

            rows.append(row)
        return rows

    # ------------------------------------------------------------------
    # Write dispatch
    # ------------------------------------------------------------------

    def _write_arrow_batches(
        self,
        *,
        batches: Iterator["pyarrow.RecordBatch"],
        schema: "pyarrow.Schema",
        options: XlsxOptions,
    ) -> None:
        """Write record batches as XLSX using the best available backend."""
        batch_list = list(batches)
        if batch_list:
            table = pa.Table.from_batches(batch_list, schema=schema)
        else:
            table = pa.Table.from_pydict(
                {name: [] for name in schema.names},
                schema=schema,
            )

        payload = self._write_table_bytes(table, options=options)
        self.buffer.replace_with_payload(payload)

    def _write_table_bytes(self, table: "pyarrow.Table", *, options: XlsxOptions) -> bytes:
        """Serialise *table* to XLSX bytes, picking the best backend."""
        engine = options.engine
        errors: list[str] = []

        if engine in ("auto", "polars"):
            try:
                return self._write_polars(table, options=options)
            except ImportError as e:
                errors.append(f"polars: {e}")
                if engine == "polars":
                    raise
            except Exception as e:
                errors.append(f"polars: {e}")
                if engine == "polars":
                    raise

        if engine in ("auto", "pandas"):
            try:
                return self._write_pandas(table, options=options)
            except ImportError as e:
                errors.append(f"pandas: {e}")
                if engine == "pandas":
                    raise
            except Exception as e:
                errors.append(f"pandas: {e}")
                if engine == "pandas":
                    raise

        if engine in ("auto", "fallback"):
            try:
                return self._write_fallback(table, options=options)
            except Exception as e:
                errors.append(f"fallback: {e}")
                if engine == "fallback":
                    raise

        raise RuntimeError(
            "XlsxIO: could not write workbook with any backend. "
            f"Tried: {', '.join(errors) if errors else 'none'}"
        )

    @staticmethod
    def _write_polars(table: "pyarrow.Table", *, options: XlsxOptions) -> bytes:
        """Write via :meth:`polars.DataFrame.write_excel` (uses xlsxwriter)."""
        from yggdrasil.polars.lib import polars as pl

        df = pl.from_arrow(table)
        if isinstance(df, pl.Series):
            df = df.to_frame()

        buf = _io.BytesIO()
        df.write_excel(
            workbook=buf,
            worksheet=options.write_sheet_name,
            include_header=options.include_header,
            autofit=False,
        )
        return buf.getvalue()

    @staticmethod
    def _write_pandas(table: "pyarrow.Table", *, options: XlsxOptions) -> bytes:
        """Write via :meth:`pandas.DataFrame.to_excel` (openpyxl-backed)."""
        from yggdrasil.pandas.lib import pandas as pd

        frame = table.to_pandas()
        buf = _io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            frame.to_excel(
                writer,
                sheet_name=options.write_sheet_name,
                header=options.include_header,
                index=False,
            )
        return buf.getvalue()

    @classmethod
    def _write_fallback(cls, table: "pyarrow.Table", *, options: XlsxOptions) -> bytes:
        """Write a minimal valid XLSX using only the stdlib."""
        headers = list(table.column_names)
        columns = [table.column(name).to_pylist() for name in headers]
        n_rows = table.num_rows

        rows_xml: list[str] = []
        row_idx = 1
        if options.include_header and headers:
            rows_xml.append(_render_row(row_idx, headers, is_string_row=True))
            row_idx += 1

        for r in range(n_rows):
            values = [columns[c][r] for c in range(len(headers))]
            rows_xml.append(_render_row(row_idx, values))
            row_idx += 1

        sheet_name = options.write_sheet_name
        sheet_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            "<sheetData>"
            f"{''.join(rows_xml)}"
            "</sheetData></worksheet>"
        ).encode("utf-8")

        workbook_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            "<sheets>"
            f'<sheet name="{_xml_escape(sheet_name)}" sheetId="1" r:id="rId1"/>'
            "</sheets></workbook>"
        ).encode("utf-8")

        workbook_rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet1.xml"/>'
            "</Relationships>"
        ).encode("utf-8")

        root_rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/>'
            "</Relationships>"
        ).encode("utf-8")

        content_types = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" '
            'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            "</Types>"
        ).encode("utf-8")

        buf = _io.BytesIO()
        with _zipfile.ZipFile(buf, "w", _zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", content_types)
            zf.writestr("_rels/.rels", root_rels)
            zf.writestr("xl/workbook.xml", workbook_xml)
            zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
            zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)
        return buf.getvalue()


# ----------------------------------------------------------------------
# Cell-level helpers (used by the built-in fallback reader/writer)
# ----------------------------------------------------------------------


def _decode_cell(cell: _ET.Element, shared: list[str]) -> Any:
    """Decode one ``<c>`` element into a Python scalar (shared-string aware)."""
    t = cell.get("t")

    if t == "s":
        v = cell.find("main:v", _NS)
        if v is None or v.text is None:
            return None
        try:
            return shared[int(v.text)]
        except (ValueError, IndexError):
            return v.text

    if t == "inlineStr":
        is_el = cell.find("main:is", _NS)
        if is_el is None:
            return None
        return "".join(t_el.text or "" for t_el in is_el.iter(f"{{{_NS['main']}}}t"))

    if t == "str":
        v = cell.find("main:v", _NS)
        return v.text if v is not None else None

    if t == "b":
        v = cell.find("main:v", _NS)
        if v is None or v.text is None:
            return None
        return v.text.strip() not in ("0", "false", "False", "")

    if t == "e":
        v = cell.find("main:v", _NS)
        return v.text if v is not None else None

    # default: numeric (or date serial — reported as float; callers that care
    # about dates should use the polars/pandas backends, which know about
    # the workbook's number formats)
    v = cell.find("main:v", _NS)
    if v is None or v.text is None:
        return None
    text = v.text
    try:
        if "." in text or "e" in text or "E" in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _render_row(row_idx: int, values: list[Any], *, is_string_row: bool = False) -> str:
    """Render one ``<row>`` element with inline-string cells for strings."""
    cells: list[str] = []
    for col_idx, value in enumerate(values):
        ref = f"{_index_to_col_letter(col_idx)}{row_idx}"
        cells.append(_render_cell(ref, value, force_string=is_string_row))
    return f'<row r="{row_idx}">{"".join(cells)}</row>'


def _render_cell(ref: str, value: Any, *, force_string: bool = False) -> str:
    """Render one ``<c>`` element, picking the right ``t`` attribute."""
    if value is None:
        return f'<c r="{ref}"/>'

    if force_string or isinstance(value, str):
        text = _xml_escape(str(value))
        return f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>'

    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{1 if value else 0}</v></c>'

    if isinstance(value, (int, float)):
        return f'<c r="{ref}"><v>{value}</v></c>'

    if isinstance(value, (datetime, date, time)):
        return (
            f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">'
            f"{_xml_escape(value.isoformat())}"
            "</t></is></c>"
        )

    text = _xml_escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>'
