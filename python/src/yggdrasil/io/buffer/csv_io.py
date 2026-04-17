"""CSV / TSV I/O on top of :class:`~yggdrasil.io.buffer.BytesIO`.

Uses :mod:`pyarrow.csv` for parsing and writing. Read-time defaults try to be
helpful for loosely structured text buffers:

* delimiter is inferred from a small text sample when not provided
* header presence is inferred from the first rows when not provided
* Arrow conversion is left enabled so numeric / boolean / temporal-like values
  become typed Arrow columns instead of plain strings where possible

Transport-level compression is handled transparently by the base class.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Optional, Sequence

from yggdrasil.io.enums import MimeTypes

from .media_io import MediaIO
from .media_options import MediaOptions

if TYPE_CHECKING:
    import pyarrow

__all__ = ["CsvIO", "CsvOptions"]


_DEFAULT_NULL_VALUES = ("", "null", "NULL", "none", "None", "nan", "NaN", "N/A", "n/a")
_DEFAULT_TRUE_VALUES = ("true", "True", "TRUE", "1", "yes", "Yes", "YES")
_DEFAULT_FALSE_VALUES = ("false", "False", "FALSE", "0", "no", "No", "NO")
_DEFAULT_TIMESTAMP_PARSERS = ("ISO8601",)
_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+|\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_DELIMITER_CANDIDATES = (",", "\t", ";", "|")


def _ensure_char(name: str, value: str | None) -> str | None:
    """Validate a one-character CSV control token."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str|None, got {type(value).__name__}")
    if len(value) != 1:
        raise ValueError(f"{name} must be exactly one character, got {value!r}")
    return value


def _normalize_str_values(
    name: str,
    values: Sequence[str] | None,
    *,
    default: Sequence[str],
) -> tuple[str, ...]:
    """Normalize optional string-sequence options."""
    if values is None:
        return tuple(default)

    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings, not {type(values).__name__}")

    out = tuple(values)
    if not all(isinstance(value, str) for value in out):
        raise TypeError(f"{name} must contain only str values")
    return out


def _classify_token(value: str) -> str:
    """Classify one CSV token for header inference."""
    token = value.strip()
    if not token:
        return "null"
    if token.lower() in {"true", "false", "yes", "no"}:
        return "bool"
    if _NUMERIC_RE.match(token):
        return "number"
    if _DATE_RE.match(token):
        return "date"
    return "string"


@dataclass
class CsvOptions(MediaOptions):
    """Options for CSV / TSV I/O."""

    delimiter: str | None = None
    has_header: bool | None = None
    include_header: bool = True
    quote_char: str | None = '"'
    escape_char: str | None = None
    newlines_in_values: bool = False
    encoding: str = "utf-8"
    skip_rows: int = 0
    null_values: Sequence[str] | None = None
    true_values: Sequence[str] | None = None
    false_values: Sequence[str] | None = None
    timestamp_parsers: Sequence[str] | None = None
    strings_can_be_null: bool = True
    quoted_strings_can_be_null: bool = True

    def __post_init__(self) -> None:
        """Normalize and validate CSV-specific options."""
        super().__post_init__()

        self.delimiter = _ensure_char("delimiter", self.delimiter)
        self.quote_char = _ensure_char("quote_char", self.quote_char)
        self.escape_char = _ensure_char("escape_char", self.escape_char)

        if self.has_header is not None and not isinstance(self.has_header, bool):
            raise TypeError(
                f"has_header must be bool|None, got {type(self.has_header).__name__}"
            )

        if not isinstance(self.include_header, bool):
            raise TypeError(
                f"include_header must be bool, got {type(self.include_header).__name__}"
            )

        if not isinstance(self.newlines_in_values, bool):
            raise TypeError(
                "newlines_in_values must be bool, "
                f"got {type(self.newlines_in_values).__name__}"
            )

        if not isinstance(self.encoding, str):
            raise TypeError(f"encoding must be str, got {type(self.encoding).__name__}")
        if not self.encoding:
            raise ValueError("encoding must not be empty")

        if not isinstance(self.skip_rows, int) or self.skip_rows < 0:
            raise ValueError("skip_rows must be a non-negative int")

        if not isinstance(self.strings_can_be_null, bool):
            raise TypeError(
                "strings_can_be_null must be bool, "
                f"got {type(self.strings_can_be_null).__name__}"
            )
        if not isinstance(self.quoted_strings_can_be_null, bool):
            raise TypeError(
                "quoted_strings_can_be_null must be bool, "
                f"got {type(self.quoted_strings_can_be_null).__name__}"
            )

        self.null_values = _normalize_str_values(
            "null_values",
            self.null_values,
            default=_DEFAULT_NULL_VALUES,
        )
        self.true_values = _normalize_str_values(
            "true_values",
            self.true_values,
            default=_DEFAULT_TRUE_VALUES,
        )
        self.false_values = _normalize_str_values(
            "false_values",
            self.false_values,
            default=_DEFAULT_FALSE_VALUES,
        )
        self.timestamp_parsers = _normalize_str_values(
            "timestamp_parsers",
            self.timestamp_parsers,
            default=_DEFAULT_TIMESTAMP_PARSERS,
        )

    @classmethod
    def resolve(cls, *, options: "CsvOptions | None" = None, **overrides: Any) -> "CsvOptions":
        """Merge *overrides* into *options* (or a fresh default)."""
        return cls.check_parameters(options=options, **overrides)


@dataclass(slots=True)
class CsvIO(MediaIO[CsvOptions]):
    """CSV / TSV I/O backed by :mod:`pyarrow.csv`."""

    @classmethod
    def check_options(
        cls,
        options: Optional[CsvOptions],
        *args,
        **kwargs,
    ) -> CsvOptions:
        """Validate and merge caller-supplied options."""
        return CsvOptions.check_parameters(options=options, **kwargs)

    def _sample_text(
        self,
        *,
        options: CsvOptions,
        limit: int = 8192,
    ) -> str:
        """Decode a small prefix for delimiter/header inference."""
        if self.buffer.size <= 0:
            return ""

        raw = self.buffer.pread(min(limit, self.buffer.size), 0)
        return raw.decode(options.encoding, errors="replace").lstrip("\ufeff")

    def _infer_delimiter(self, *, options: CsvOptions) -> str:
        """Infer delimiter from media type or from a text sample."""
        if options.delimiter is not None:
            return options.delimiter

        if self.media_type.mime_type is MimeTypes.TSV:
            return "\t"

        sample = self._sample_text(options=options)
        lines = [line for line in sample.splitlines() if line.strip()]
        if not lines:
            return ","

        best_delimiter = ","
        best_score = (-1, float("inf"))

        for delimiter in _DELIMITER_CANDIDATES:
            counts = [line.count(delimiter) for line in lines[:10]]
            positive = [count for count in counts if count > 0]
            if not positive:
                continue

            score = (len(positive), max(positive) - min(positive))
            if score > best_score:
                best_delimiter = delimiter
                best_score = score

        return best_delimiter

    def _infer_has_header(
        self,
        *,
        options: CsvOptions,
        delimiter: str,
    ) -> bool:
        """Infer whether the buffer has a header row."""
        if options.has_header is not None:
            return options.has_header

        sample = self._sample_text(options=options)
        rows = [
            row
            for row in csv.reader(
                sample.splitlines(),
                delimiter=delimiter,
                quotechar=options.quote_char or '"',
                escapechar=options.escape_char,
            )
            if row and any(cell.strip() for cell in row)
        ]

        if not rows:
            return True
        if len(rows) == 1:
            first_types = [_classify_token(value) for value in rows[0]]
            return all(kind == "string" for kind in first_types)

        first = rows[0]
        second = rows[1]
        first_types = [_classify_token(value) for value in first]
        second_types = [_classify_token(value) for value in second]

        if any(kind in {"number", "bool", "date"} for kind in first_types):
            return False

        second_has_typed_values = any(kind in {"number", "bool", "date"} for kind in second_types)
        first_is_stringy = all(kind in {"string", "null"} for kind in first_types)
        first_is_unique = len({value.strip() for value in first if value.strip()}) == len(
            [value for value in first if value.strip()]
        )

        return first_is_stringy and first_is_unique and second_has_typed_values

    def _read_options(self, *, options: CsvOptions, has_header: bool):
        """Build :mod:`pyarrow.csv` read options."""
        import pyarrow.csv as csv_pa

        return csv_pa.ReadOptions(
            use_threads=options.use_threads,
            skip_rows=options.skip_rows,
            autogenerate_column_names=not has_header,
            encoding=options.encoding,
        )

    def _parse_options(self, *, options: CsvOptions, delimiter: str):
        """Build :mod:`pyarrow.csv` parse options."""
        import pyarrow.csv as csv_pa

        quote_char = options.quote_char if options.quote_char is not None else False
        escape_char = options.escape_char if options.escape_char is not None else False

        return csv_pa.ParseOptions(
            delimiter=delimiter,
            quote_char=quote_char,
            escape_char=escape_char,
            newlines_in_values=options.newlines_in_values,
        )

    def _convert_options(self, *, options: CsvOptions):
        """Build :mod:`pyarrow.csv` convert options with Arrow inference enabled."""
        import pyarrow.csv as csv_pa

        return csv_pa.ConvertOptions(
            strings_can_be_null=options.strings_can_be_null,
            quoted_strings_can_be_null=options.quoted_strings_can_be_null,
            null_values=list(options.null_values),
            true_values=list(options.true_values),
            false_values=list(options.false_values),
            timestamp_parsers=list(options.timestamp_parsers),
        )

    def _read_arrow_batches(
        self,
        *,
        options: CsvOptions,
    ) -> Iterator["pyarrow.RecordBatch"]:
        """Yield record batches from the CSV / TSV buffer."""
        import pyarrow as pa
        import pyarrow.csv as csv_pa

        if self.buffer.size <= 0:
            return

        delimiter = self._infer_delimiter(options=options)
        has_header = self._infer_has_header(options=options, delimiter=delimiter)
        arrow_io = self.buffer.to_arrow_io("r")
        try:
            table = csv_pa.read_csv(
                arrow_io,
                read_options=self._read_options(options=options, has_header=has_header),
                parse_options=self._parse_options(options=options, delimiter=delimiter),
                convert_options=self._convert_options(options=options),
            )
        finally:
            arrow_io.close()

        if options.columns is not None:
            table = table.select(options.columns)

        if table.num_rows == 0 and options.ignore_empty:
            return

        yield from pa.Table.from_batches(table.to_batches()).to_batches()

    def _write_arrow_batches(
        self,
        *,
        batches: Iterator["pyarrow.RecordBatch"],
        schema: "pyarrow.Schema",
        options: CsvOptions,
    ) -> None:
        """Write record batches as CSV / TSV into the buffer."""
        import pyarrow as pa
        import pyarrow.csv as csv_pa

        delimiter = self._infer_delimiter(options=options)
        table = pa.Table.from_batches(list(batches), schema=schema)

        arrow_io = self.buffer.to_arrow_io("w")
        try:
            csv_pa.write_csv(
                table,
                arrow_io,
                write_options=csv_pa.WriteOptions(
                    include_header=options.include_header,
                    delimiter=delimiter,
                ),
            )
        finally:
            arrow_io.close()
