"""XML I/O on top of :class:`~yggdrasil.io.buffer.BytesIO`.

Reading
-------
* Parses XML with :mod:`xml.etree.ElementTree`.
* Detects common row-oriented layouts such as ``<rows><row>...</row></rows>``
  and nested containers like ``<root><records><record>...</record></records>``.
* Normalizes the document into ``list[dict]``; repeated child tags become
  Python lists, attributes use a configurable prefix, and mixed-content
  text is stored under a configurable key.
* Sparse/heterogeneous rows are normalized against the union of keys so
  :func:`pa.RecordBatch.from_pylist` doesn't silently drop columns that
  appear only in later rows.

Writing
-------
* Serializes Arrow rows or a ``list[dict]`` payload into a predictable
  row-oriented XML document.
* Save modes: OVERWRITE (truncate + write), IGNORE / ERROR_IF_EXISTS
  (via the base-class guard), APPEND (read-old-then-rewrite), UPSERT
  (anti-join on ``match_by``). XML doesn't stream-append — the root
  element wraps everything — so APPEND/UPSERT always re-materialize the
  full document.

Transport-level compression (``MediaType.codec``) is handled by the
base class via ``open()`` / ``close()`` / ``mark_dirty()``.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Optional, Sequence

import pyarrow as pa

from yggdrasil.io.enums import SaveMode
from .media_io import MediaIO
from .media_options import MediaOptions

if TYPE_CHECKING:
    import pyarrow

__all__ = ["XmlOptions", "XmlIO"]


_VALUE_COLUMN = "value"
_INT_RE = re.compile(r"^[+-]?(?:0|[1-9]\d*)$")
_FLOAT_RE = re.compile(r"^[+-]?(?:(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)|\d+\.\d*)$")
_ROW_CONTAINER_TAGS = {"rows", "records", "items", "entries"}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _strip_tag(tag: str) -> str:
    """Remove XML namespace prefixes from a tag name."""
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    if ":" in tag:
        return tag.rsplit(":", 1)[-1]
    return tag


def _coerce_scalar(value: str) -> Any:
    """Convert simple XML text values into Python scalars when safe."""
    text = value.strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    if _INT_RE.match(text):
        # Preserve literals with redundant leading zeros as strings — they
        # might be identifiers (ZIP codes, part numbers) rather than numbers.
        if text.startswith("0") and len(text) > 1:
            return text
        if text.startswith("-0") and len(text) > 2:
            return text
        try:
            return int(text)
        except Exception:
            return text

    if _FLOAT_RE.match(text):
        try:
            return float(text)
        except Exception:
            return text

    return text


def _normalize_records(records: list[dict]) -> list[dict]:
    """Backfill sparse rows with ``None`` against the union of keys.

    Mirrors :meth:`MediaIO._normalize_records` in the base class but scoped
    to XML records (which are frequently sparse: optional elements and
    attributes mean consecutive rows rarely have identical key sets).
    Without this, :func:`pa.RecordBatch.from_pylist` infers the schema
    from the first row and silently drops anything that appears later.
    """
    if not records:
        return records

    all_keys: dict[str, None] = {}
    needs_backfill = False
    reference_keys: tuple[str, ...] | None = None

    for row in records:
        if not isinstance(row, dict):
            needs_backfill = True
            continue
        row_keys = tuple(row.keys())
        if reference_keys is None:
            reference_keys = row_keys
        elif row_keys != reference_keys:
            needs_backfill = True
        for key in row_keys:
            if key not in all_keys:
                all_keys[key] = None

    if not needs_backfill:
        return records

    keys = tuple(all_keys.keys())
    return [
        {k: row.get(k) if isinstance(row, dict) else None for k in keys}
        for row in records
    ]


# ---------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------

@dataclass
class XmlOptions(MediaOptions):
    """Options for XML I/O.

    Parameters
    ----------
    encoding:
        Character encoding. Default ``"utf-8"``.
    errors:
        Encode/decode error policy. Default ``"strict"``.
    root_tag:
        Top-level element name on write. Default ``"rows"``.
    row_tag:
        Per-row element name on write. Default ``"row"``.
    text_key:
        Python dict key under which mixed-content text is stored when an
        element has both attributes and inner text. Default ``"value"``.
    attr_prefix:
        Prefix used in the normalized dict to distinguish XML attributes
        from child elements. Default ``"@"``. May be empty.
    xml_declaration:
        Emit the ``<?xml version="1.0" encoding="..."?>`` header on write.
        Default ``True``.
    pretty_print:
        Indent emitted XML for human inspection. Default ``False``
        (compact output). Uses :func:`xml.etree.ElementTree.indent`.
    list_item_tag:
        Tag name used for list items when the containing value is a
        Python list under a key that isn't itself a list of dicts.
        Default ``"item"``. The old behavior of stripping a trailing
        ``"s"`` from the container tag was unreliable (``"data"`` →
        ``"dat"``) and has been removed.
    """

    encoding: str = "utf-8"
    errors: str = "strict"
    root_tag: str = "rows"
    row_tag: str = "row"
    text_key: str = _VALUE_COLUMN
    attr_prefix: str = "@"
    xml_declaration: bool = True
    pretty_print: bool = False
    list_item_tag: str = "item"

    def __post_init__(self) -> None:
        """Normalize and validate XML-specific options."""
        super().__post_init__()

        for name in (
            "encoding",
            "errors",
            "root_tag",
            "row_tag",
            "text_key",
            "attr_prefix",
            "list_item_tag",
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be str, got {type(value).__name__}")
            # attr_prefix is the only tag-ish option that may be empty.
            if not value and name != "attr_prefix":
                raise ValueError(f"{name} must not be empty")

        if not isinstance(self.xml_declaration, bool):
            raise TypeError(
                f"xml_declaration must be bool, got {type(self.xml_declaration).__name__}"
            )
        if not isinstance(self.pretty_print, bool):
            raise TypeError(
                f"pretty_print must be bool, got {type(self.pretty_print).__name__}"
            )

    @classmethod
    def resolve(cls, *, options: "XmlOptions | None" = None, **overrides: Any) -> "XmlOptions":
        """Merge *overrides* into *options* (or a fresh default)."""
        return cls.check_parameters(options=options, **overrides)


# ---------------------------------------------------------------------
# XmlIO
# ---------------------------------------------------------------------

@dataclass(slots=True)
class XmlIO(MediaIO[XmlOptions]):
    """XML I/O with row detection, cast integration, and save-mode support."""

    @classmethod
    def check_options(
        cls,
        options: Optional[XmlOptions],
        *args,
        **kwargs,
    ) -> XmlOptions:
        """Validate and merge caller-supplied options."""
        return XmlOptions.check_parameters(options=options, **kwargs)

    # ------------------------------------------------------------------
    # Parsing (assumes buffer is open)
    # ------------------------------------------------------------------

    def _parse_root(self, options: XmlOptions) -> ET.Element | None:
        """Parse the XML document from ``self.buffer``. Returns ``None`` if empty."""
        if self.buffer is None or self.buffer.size <= 0:
            return None

        raw = self.buffer.to_bytes()
        if not raw:
            return None

        text = raw.decode(options.encoding, errors=options.errors)
        return ET.fromstring(text)

    @staticmethod
    def _homogeneous_children(element: ET.Element) -> list[ET.Element] | None:
        """Return direct children when they look like repeated row elements."""
        children = list(element)
        if not children:
            return None

        names = [_strip_tag(child.tag) for child in children]
        if len(children) >= 2 and len(set(names)) == 1:
            return children
        # Single-child case: only treat as a row list if the parent is
        # one of the known row-container tags (avoids misdetecting
        # arbitrary singleton wrappers).
        if len(children) == 1 and _strip_tag(element.tag).lower() in _ROW_CONTAINER_TAGS:
            return children
        return None

    def _find_row_elements(self, root: ET.Element) -> list[ET.Element] | None:
        """Find the best repeated child collection to treat as rows."""
        direct = self._homogeneous_children(root)
        if direct is not None:
            return direct

        # BFS one level at a time. Stops at the first homogeneous group
        # discovered at any depth — this matches how people typically
        # nest row containers (``<root><data><rows><row>…</row></rows>``).
        queue: list[ET.Element] = list(root)
        while queue:
            node = queue.pop(0)
            repeated = self._homogeneous_children(node)
            if repeated is not None:
                return repeated
            queue.extend(list(node))

        return None

    def _element_to_python(
        self,
        element: ET.Element,
        options: XmlOptions,
    ) -> Any:
        """Convert one XML element into a Python scalar / dict."""
        data: dict[str, Any] = {
            f"{options.attr_prefix}{_strip_tag(key)}": _coerce_scalar(value)
            for key, value in element.attrib.items()
        }

        children = list(element)
        if not children:
            text = _coerce_scalar(element.text or "")
            if data:
                if text is not None:
                    data[options.text_key] = text
                return data
            return text

        grouped: dict[str, list[Any]] = {}
        for child in children:
            name = _strip_tag(child.tag)
            grouped.setdefault(name, []).append(
                self._element_to_python(child, options)
            )

        for name, values in grouped.items():
            data[name] = values[0] if len(values) == 1 else values

        text = _coerce_scalar(element.text or "")
        if text is not None:
            data[options.text_key] = text

        return data

    def _normalize_root(
        self,
        root: ET.Element,
        options: XmlOptions,
    ) -> list[dict[str, Any]]:
        """Normalize the parsed document into row-oriented ``list[dict]``."""
        rows = self._find_row_elements(root)
        if rows is not None:
            normalized = [self._element_to_python(row, options) for row in rows]
        else:
            normalized = [self._element_to_python(root, options)]

        out: list[dict[str, Any]] = []
        for row in normalized:
            if isinstance(row, dict):
                out.append(row)
            else:
                out.append({_VALUE_COLUMN: row})
        return out

    def _load_records(self, options: XmlOptions) -> list[dict[str, Any]]:
        """Parse and normalize the XML document into row records.

        Caller is responsible for opening the buffer first.
        """
        root = self._parse_root(options)
        if root is None:
            return []
        return self._normalize_root(root, options)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _value_to_xml(
        self,
        parent: ET.Element,
        key: str,
        value: Any,
        options: XmlOptions,
    ) -> None:
        """Append one (key, value) pair under ``parent`` as XML."""
        # Attribute: strip prefix, set on parent.
        if options.attr_prefix and key.startswith(options.attr_prefix):
            attr_name = key[len(options.attr_prefix):]
            parent.set(attr_name, "" if value is None else str(value))
            return

        # Mixed-content text marker.
        if key == options.text_key:
            parent.text = "" if value is None else str(value)
            return

        # List: emit one element per item under the same key name.
        if isinstance(value, list):
            for item in value:
                child = ET.SubElement(parent, key)
                self._object_to_xml(child, item, options)
            return

        child = ET.SubElement(parent, key)
        self._object_to_xml(child, value, options)

    def _object_to_xml(
        self,
        element: ET.Element,
        value: Any,
        options: XmlOptions,
    ) -> None:
        """Populate ``element`` from a Python scalar / mapping / list."""
        if isinstance(value, dict):
            for key, item in value.items():
                self._value_to_xml(element, key, item, options)
            return

        if isinstance(value, list):
            # Lists appearing where a scalar / dict was expected:
            # emit items under the configured list_item_tag. This avoids
            # the old tag-pluralization heuristic that mangled names like
            # ``"data"`` → ``"dat"``.
            for item in value:
                child = ET.SubElement(element, options.list_item_tag)
                self._object_to_xml(child, item, options)
            return

        if value is not None:
            element.text = str(value)

    def _records_to_xml(
        self,
        records: list[dict],
        options: XmlOptions,
    ) -> bytes:
        """Serialize a list of row records into an XML byte string."""
        root = ET.Element(options.root_tag)
        for record in records:
            row = ET.SubElement(root, options.row_tag)
            self._object_to_xml(row, record, options)

        if options.pretty_print:
            # ET.indent is available from Python 3.9. Operates in-place.
            ET.indent(root, space="  ", level=0)

        return ET.tostring(
            root,
            encoding=options.encoding,
            xml_declaration=options.xml_declaration,
        )

    # ------------------------------------------------------------------
    # Core read/write protocol
    # ------------------------------------------------------------------

    def _read_arrow_batches(
        self,
        options: XmlOptions,
    ) -> Iterator["pyarrow.RecordBatch"]:
        """Yield Arrow batches from the XML buffer."""
        with self.open() as b:
            if b.buffer.size <= 0:
                return

            records = self._load_records(options)
            if not records:
                return

            # Handle sparse/heterogeneous rows before from_pylist sees them.
            records = _normalize_records(records)

            # Build one batch; respect batch_size by slicing the list.
            batch_size = getattr(options, "batch_size", 0) or 0

            def iter_batches() -> Iterator[pa.RecordBatch]:
                if batch_size <= 0 or len(records) <= batch_size:
                    yield pa.RecordBatch.from_pylist(records)
                    return
                for start in range(0, len(records), batch_size):
                    yield pa.RecordBatch.from_pylist(
                        records[start : start + batch_size]
                    )

            batches: Iterator[pa.RecordBatch] = iter_batches()

            if options.columns is not None:
                batches = (batch.select(options.columns) for batch in batches)

            if options.ignore_empty:
                batches = (batch for batch in batches if batch.num_rows > 0)

            yield from options.cast.cast_iterator(batches)

    def _collect_arrow_schema(self, full: bool = False) -> "pyarrow.Schema":
        """Infer the schema from a single parsed row."""
        del full

        with self.open() as b:
            if b.buffer.size <= 0:
                return pa.schema([])

            options = self.check_options(options=None)
            records = self._load_records(options)
            if not records:
                return pa.schema([])
            records = _normalize_records(records)
            return pa.RecordBatch.from_pylist(records[:1]).schema

    def _write_arrow_batches(
        self,
        batches: Iterator["pyarrow.RecordBatch"],
        options: XmlOptions,
    ) -> None:
        """Serialize record batches into a row-oriented XML document."""
        with self.open() as b:
            if self.skip_write(options.mode):
                return

            # APPEND / UPSERT: read the existing document once, merge
            # with the new records, then rewrite. XML documents are tree-
            # structured, so there's no way to stream-append without
            # invalidating the closing root tag.
            existing_records: list[dict] = []
            if b.buffer.size > 0 and options.mode in (SaveMode.APPEND, SaveMode.UPSERT):
                existing_records = self._load_records(options)

            # Route the new stream through the write-side cast. This is
            # the canonical write entry point for options.cast.
            cast_batches = options.cast.cast_iterator(batches)

            # Collect new-side records from the cast batches. We can't
            # stream-encode XML anyway, so materialization is honest here.
            new_records: list[dict] = []
            for batch in cast_batches:
                if batch.num_rows == 0:
                    continue
                new_records.extend(batch.to_pylist())

            merged = self._merge_records(
                existing=existing_records,
                new=new_records,
                mode=options.mode,
                match_by=options.match_by,
            )

            if not merged and not existing_records and not new_records:
                # Nothing to write and nothing was there — no-op.
                return

            # OVERWRITE (or empty buffer): truncate before writing so a
            # partial write can't leave a hybrid document.
            if options.mode not in (SaveMode.APPEND, SaveMode.UPSERT):
                b.buffer.truncate(0)

            payload = self._records_to_xml(merged, options)
            b.buffer.replace_with_payload(payload)
            b.mark_dirty()

    # ------------------------------------------------------------------
    # Save-mode merging
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_records(
        *,
        existing: list[dict],
        new: list[dict],
        mode: SaveMode,
        match_by: Any,
    ) -> list[dict]:
        """Combine existing + new records according to the save mode."""
        if mode == SaveMode.APPEND:
            return existing + new

        if mode == SaveMode.UPSERT:
            keys = _normalize_match_by(match_by)
            if not keys:
                raise ValueError("SaveMode.UPSERT requires options.match_by to be set")

            new_index: set[tuple] = set()
            for row in new:
                new_index.add(tuple(row.get(k) for k in keys))

            kept_old = [
                row for row in existing
                if tuple(row.get(k) for k in keys) not in new_index
            ]
            return kept_old + new

        # OVERWRITE / AUTO / TRUNCATE / anything else: new wins outright.
        return new

    # ------------------------------------------------------------------
    # JSON-native-style read fast path (bypasses Arrow type inference)
    # ------------------------------------------------------------------
    #
    # Like JsonIO, XmlIO has a cheap path for callers who just want rows:
    # parse once, return the dicts directly. Routing through
    # _read_arrow_table would force Arrow type inference on XML-parsed
    # values — wasteful and occasionally fragile for mixed-scalar fields.

    @staticmethod
    def _project_row(row: dict, columns: "Sequence[str]") -> dict:
        return {col: row[col] for col in columns if col in row}

    def _read_pylist(
        self,
        options: XmlOptions,
    ):
        batch_size = getattr(options, "batch_size", 0) or 0
        columns = options.columns

        with self.open() as b:
            if b.buffer.size <= 0:
                return [] if batch_size <= 0 else iter(())

            records = self._load_records(options)

            if columns is not None:
                records = [self._project_row(r, columns) for r in records]

            if batch_size <= 0:
                return records

            def iter_chunks():
                for start in range(0, len(records), batch_size):
                    yield records[start : start + batch_size]

            return iter_chunks()


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _normalize_match_by(match_by: Any) -> tuple[str, ...]:
    """Return *match_by* as a tuple of column names, or ``()`` if unset."""
    if match_by is None or match_by is ...:
        return ()
    if isinstance(match_by, str):
        return (match_by,)
    return tuple(match_by)