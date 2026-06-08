"""Render an Arrow table as an aligned, typed text preview.

:func:`arrow_display` is the engine behind
:meth:`yggdrasil.io.tabular.Tabular.display` — give it a
``pa.Table`` (and, optionally, the project :class:`~yggdrasil.data.schema.Schema`
that carries the column markers) and it lays out a first-*n*-rows preview:

- a two-row header — the column names, then each column's short type tag
  (:meth:`yggdrasil.data.types.base.DataType.short`, recursive for nested
  types) plus its main schema markers (PK / partition / required);
- a ``│``-delimited body with a ``─┼─`` rule under the header and a ``─┴─``
  rule to close; numbers / booleans right-align;
- every cell clipped to *max_width* (headers a touch wider) and measured by
  *display* width — combining marks count 0, East-Asian wide / emoji glyphs
  count 2 — so CJK / accented values line up on what the terminal shows and
  one long value can never balloon the table.

It reads the Arrow columns directly (no per-row dicts): one pass per column
builds ``[name, type, *values]``, the width is the widest cell each column
must fit, then the grid is stitched row by row.
"""
from __future__ import annotations

import json
import unicodedata
from typing import TYPE_CHECKING

try:
    import pyarrow as pa
except ImportError:  # heavy optional dep — install on first import (project guard)
    from yggdrasil.lazy_imports import _lazy_import

    pa = _lazy_import("pyarrow", install=True)

if TYPE_CHECKING:
    from yggdrasil.data.schema import Schema

#: Whitespace that would break column alignment, mapped to a single space.
_WS_TRANS = {ord(c): " " for c in "\n\t\r\v\f"}


def _char_width(ch: str) -> int:
    """A character's *display* width — 0 for combining marks, 2 for East-Asian
    wide / fullwidth glyphs (CJK, emoji), else 1. So a table aligns on what the
    terminal actually shows, not the code-point count."""
    if unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _disp_width(text: str) -> int:
    return sum(_char_width(c) for c in text)


def _clip_width(text: str, cap: int) -> str:
    """Clip *text* to a *display-width* budget (not code points), ending in ``…``
    — so one long value (or a wide-glyph value) can't balloon the display."""
    if _disp_width(text) <= cap:
        return text
    out: list[str] = []
    width = 0
    for ch in text:
        cw = _char_width(ch)
        if width + cw > cap - 1:
            break
        out.append(ch)
        width += cw
    return "".join(out) + "…"


def arrow_display(
    table: pa.Table,
    schema: Schema | None = None,
    *,
    n: int = 10,
    max_width: int = 32,
) -> str:
    """Render *table*'s first *n* rows as an aligned, typed text table.

    *schema* supplies the per-column type tags and markers; when omitted it is
    derived from ``table.schema``. A table holding more than *n* rows shows a
    ``… (first n rows)`` marker; otherwise the footer is the shape.

        print(arrow_display(some_table, n=5))
    """
    if schema is None:
        from yggdrasil.data.schema import Schema as _Schema

        schema = _Schema.from_arrow_schema(table.schema)
    fields = list(schema.children)
    if not fields:                                       # empty / no schema
        return "(no rows)"

    truncated = table.num_rows > n
    body = table.slice(0, n)

    # Build the grid one Arrow column at a time, leaning on the project Field
    # for the header: its recursive short type tag and main schema markers
    # (PK / partition / required), plus whether the type right-aligns (numbers
    # / booleans). Each cell is clipped to *max_width* (headers a touch wider)
    # so one long value or name can never balloon the table; nested values
    # compact to a single line and nulls read as a dot.
    head_cap = max(max_width, 44)
    grid: list[list[str]] = []          # per column: [name, type, *values]
    right: list[bool] = []
    for i, field in enumerate(fields):
        marks = field.markers()
        tag = f"{field.dtype.short()} {marks}" if marks else field.dtype.short()
        column = [_clip_width(field.name, head_cap), _clip_width(tag, head_cap)]
        for value in body.column(i).to_pylist():
            if value is None:
                text = "·"
            elif isinstance(value, (dict, list, tuple)):
                text = json.dumps(value, separators=(",", ":"),
                                  default=str, ensure_ascii=False)
            else:
                text = str(value).translate(_WS_TRANS)
            column.append(_clip_width(text, max_width))
        grid.append(column)
        right.append(field.dtype.type_id.is_numeric or field.dtype.type_id.is_boolean)

    # Column width = the widest cell it must fit (by display width). Pad each
    # cell — right for numbers, else left — join with │; a ─┼─ rule under the
    # two-row header, a ─┴─ rule to close.
    widths = [max(_disp_width(cell) for cell in column) for column in grid]
    lines: list[str] = []
    for r in range(2 + body.num_rows):
        row = []
        for c, column in enumerate(grid):
            cell = column[r]
            gap = " " * (widths[c] - _disp_width(cell))
            row.append(gap + cell if right[c] else cell + gap)
        lines.append(" │ ".join(row))
        if r == 1:
            lines.append("─┼─".join("─" * w for w in widths))
    lines.append("─┴─".join("─" * w for w in widths))
    if truncated:
        lines.append(f"… (first {n} rows)")
    else:
        cols = len(fields)
        lines.append(f"{body.num_rows} row{'s' * (body.num_rows != 1)} × "
                     f"{cols} col{'s' * (cols != 1)}")
    return "\n".join(lines)
