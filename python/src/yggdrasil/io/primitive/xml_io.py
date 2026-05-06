"""XML I/O for :class:`PrimitiveIO`.

XML is the awkward leaf. Pyarrow has no XML reader, the format
isn't naturally tabular, and there's no native polars scanner.
:class:`XmlIO` makes one opinionated choice and sticks to it: the
document is a sequence of row-shaped elements under a single root.

Default shape::

    <rows>
      <row>
        <colname>value</colname>
        ...
      </row>
      ...
    </rows>

Save modes: OVERWRITE only — XML's single-root rule means
appending new rows would require seeking past the closing tag,
rewinding, and re-emitting.
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, Iterable, Iterator

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.io.bytes_io import BytesIO


__all__ = ["XmlIO", "XmlOptions"]


# ---------------------------------------------------------------------------
# XmlOptions
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class XmlOptions(CastOptions):
    """:class:`CastOptions` extended with XML-specific knobs."""

    root_tag: str = "rows"
    row_tag: str = "row"
    encoding: str = "utf-8"
    pretty_print: bool = False


# ---------------------------------------------------------------------------
# XmlIO
# ---------------------------------------------------------------------------


class XmlIO(BytesIO):
    """:class:`PrimitiveIO` for flat row-shaped XML documents."""

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    @classmethod
    def default_media_type(cls):
        return MimeTypes.XML

    @classmethod
    def options_class(cls):
        return XmlOptions

    _APPEND_REJECTED_HINT: ClassVar[str] = (
        "XML append would require rewriting the document end — "
        "stripping the closing root element, inserting new rows, "
        "and re-emitting the close. Out of scope at the leaf level. "
        "Use a folder-oriented writer to add new XML files alongside, "
        "or migrate to JSON Lines for an appendable shape."
    )
    _NATIVE_SCANNER_OK: ClassVar[bool] = False

    @staticmethod
    def _lxml():
        try:
            from lxml import etree  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "XmlIO requires lxml. Install with: pip install lxml"
            ) from e
        from lxml import etree
        return etree

    # ==================================================================
    # Schema
    # ==================================================================

    def _collect_schema(self, options: XmlOptions) -> Schema:
        if self.is_empty():
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
        options: XmlOptions,
    ) -> Iterator[pa.RecordBatch]:
        """Iterparse the document, yielding row dicts grouped into batches."""
        if self.is_empty():
            return

        with self._reading_context(options) as io:
            etree = self._lxml()
            row_size = options.row_size if options.row_size else 4096
            tag = options.row_tag

            rows: list[dict] = []
            context = etree.iterparse(io, events=("end",), tag=tag)
            for _, element in context:
                rows.append(self._element_to_row(element))

                # Free child memory; keep the element shell, drop
                # preceding siblings.
                element.clear()
                while element.getprevious() is not None:
                    del element.getparent()[0]

                if len(rows) >= row_size:
                    yield from self._rows_to_batches(rows, options)
                    rows = []

            if rows:
                yield from self._rows_to_batches(rows, options)

    @staticmethod
    def _element_to_row(element) -> dict:
        row: dict[str, Any] = {}
        for child in element:
            row[child.tag] = child.text if child.text is not None else None
        return row

    def _rows_to_batches(
        self,
        rows: list[dict],
        options: XmlOptions,
    ) -> Iterator[pa.RecordBatch]:
        normalized = self._normalize_records(rows)
        if not normalized:
            return
        table = pa.Table.from_pylist(normalized)
        for batch in table.to_batches():
            yield options.cast_arrow_tabular(batch)

    # ==================================================================
    # Write path
    # ==================================================================

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: XmlOptions,
    ) -> None:
        """Serialize batches as a row-element-per-record XML document."""
        action = self._resolve_save_mode(options.mode)
        if action is Mode.IGNORE:
            return
        if action is not Mode.OVERWRITE:
            raise NotImplementedError(
                f"{type(self).__name__}._write_arrow_batches only handles "
                f"OVERWRITE; got {action!r}. {self._APPEND_REJECTED_HINT}"
            )

        iterator = iter(batches)
        first = next(iterator, None)
        if first is None:
            return

        if options.target_field is not None:
            first = options.cast_arrow_tabular(first)

        lifecycle = options.copy(truncate_before_write=True)

        with self._writing_context(lifecycle) as io:
            etree = self._lxml()
            with etree.xmlfile(io, encoding=options.encoding, buffered=True) as xf:
                xf.write_declaration()
                with xf.element(options.root_tag):
                    self._write_batch(xf, etree, first, options)
                    for batch in iterator:
                        if options.target_field is not None:
                            batch = options.cast_arrow_tabular(batch)
                        self._write_batch(xf, etree, batch, options)

    @staticmethod
    def _write_batch(xf, etree, batch: pa.RecordBatch, options: XmlOptions) -> None:
        for row in batch.to_pylist():
            element = etree.Element(options.row_tag)
            for key, value in row.items():
                child = etree.SubElement(element, key)
                if value is not None:
                    child.text = str(value)
            xf.write(element, pretty_print=options.pretty_print)