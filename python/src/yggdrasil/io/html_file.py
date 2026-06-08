"""HTML :class:`Tabular` leaf — read ``<table>`` elements into a frame.

HTML is a blob format, but it routinely *carries* tabular data. This leaf
makes ``text/html`` a first-class tabular input: it parses the page's
``<table>`` elements (via ``pandas.read_html``) and yields the selected one as
Arrow batches, so ``IO.from_("page.html").to_polars()`` and
``response.to_polars()`` (for a fetched HTML page) "just work". A page with no
table degrades to a one-row readable-text projection. Writing a frame emits an
HTML ``<table>``.
"""
from __future__ import annotations

import dataclasses
import io as _io
import re
from typing import ClassVar, Iterable, Iterator

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.enums import MimeTypes
from yggdrasil.io.base import IO

__all__ = ["HTMLFile", "HtmlOptions"]


@dataclasses.dataclass(frozen=True, slots=True)
class HtmlOptions(CastOptions):
    """:class:`CastOptions` extended with HTML-specific knobs."""

    table: int = 0          #: which ``<table>`` to read (0-based)
    encoding: str = "utf-8"
    border: int = 1         #: write: ``<table border=...>``
    index: bool = False     #: write: include the DataFrame index column


def _strip_html(html: str) -> str:
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return " ".join(text.split())[:8000]


class HTMLFile(IO[bytes, HtmlOptions]):
    """:class:`Tabular` leaf for HTML — parses ``<table>`` elements."""

    mime_type: ClassVar[MimeTypes] = MimeTypes.HTML

    @classmethod
    def options_class(cls):
        return HtmlOptions

    def _read_arrow_batches(self, options: HtmlOptions) -> Iterator[pa.RecordBatch]:
        data = self.read_bytes()
        if not data:
            return
        text = data.decode(options.encoding, errors="replace")
        import pandas as pd

        try:
            tables = pd.read_html(_io.StringIO(text))
        except ValueError:  # "No tables found"
            tables = []
        if tables:
            idx = options.table if 0 <= options.table < len(tables) else 0
            table = pa.Table.from_pandas(tables[idx], preserve_index=False)
        else:
            table = pa.table({"text": [_strip_html(text)]})
        for batch in table.to_batches():
            yield options.cast_arrow_batch(batch)

    def _write_arrow_batches(
        self, batches: Iterable[pa.RecordBatch], options: HtmlOptions
    ) -> None:
        table = pa.Table.from_batches(list(batches))
        html = table.to_pandas().to_html(index=options.index, border=options.border)
        self.write_bytes(html.encode(options.encoding), 0)
