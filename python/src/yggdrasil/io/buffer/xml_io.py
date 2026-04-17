"""XML I/O on top of :class:`~yggdrasil.io.buffer.BytesIO`.

Reading:
    * Parses XML with :mod:`xml.etree.ElementTree`.
    * Detects common row-oriented layouts such as ``<rows><row>...</row></rows>``
      and nested containers like ``<root><records><record>...</record></records>``.
    * Normalizes XML into ``list[dict]`` so Arrow can infer the tabular schema.
    * Repeated child tags become Python lists, attributes use a configurable
      prefix, and mixed-content text is stored under a configurable key.

Writing:
    * Serializes Arrow rows or ``list[dict]`` payloads back into a predictable
      row-oriented XML document.

Transport-level compression is handled transparently by the base class.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import pyarrow as pa

from yggdrasil.io.enums import SaveMode
from .media_io import MediaIO
from .media_options import MediaOptions

__all__ = ["XmlOptions", "XmlIO"]


_VALUE_COLUMN = "value"
_INT_RE = re.compile(r"^[+-]?(?:0|[1-9]\d*)$")
_FLOAT_RE = re.compile(r"^[+-]?(?:(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)|\d+\.\d*)$")
_ROW_CONTAINER_TAGS = {"rows", "records", "items", "entries"}


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


@dataclass
class XmlOptions(MediaOptions):
    """Options for XML I/O."""

    encoding: str = "utf-8"
    errors: str = "strict"
    root_tag: str = "rows"
    row_tag: str = "row"
    text_key: str = _VALUE_COLUMN
    attr_prefix: str = "@"
    xml_declaration: bool = True

    def __post_init__(self) -> None:
        """Normalize and validate XML-specific options."""
        super().__post_init__()

        for name in ("encoding", "errors", "root_tag", "row_tag", "text_key", "attr_prefix"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be str, got {type(value).__name__}")
            if not value and name != "attr_prefix":
                raise ValueError(f"{name} must not be empty")

        if not isinstance(self.xml_declaration, bool):
            raise TypeError(
                f"xml_declaration must be bool, got {type(self.xml_declaration).__name__}"
            )

    @classmethod
    def resolve(cls, *, options: "XmlOptions | None" = None, **overrides: Any) -> "XmlOptions":
        """Merge *overrides* into *options* (or a fresh default)."""
        return cls.check_parameters(options=options, **overrides)


@dataclass(slots=True)
class XmlIO(MediaIO[XmlOptions]):
    """XML I/O with smart row detection and predictable row-oriented writing."""

    @classmethod
    def check_options(
        cls,
        options: Optional[XmlOptions],
        *args,
        **kwargs,
    ) -> XmlOptions:
        """Validate and merge caller-supplied options."""
        return XmlOptions.check_parameters(options=options, **kwargs)

    def _load_root(self, *, options: XmlOptions) -> ET.Element | None:
        """Parse the XML document from the buffer."""
        if self.buffer.size <= 0:
            return None

        raw = self.buffer.to_bytes()
        root = ET.fromstring(raw.decode(options.encoding, errors=options.errors))
        return root

    @staticmethod
    def _homogeneous_children(element: ET.Element) -> list[ET.Element] | None:
        """Return direct children when they look like repeated row elements."""
        children = list(element)
        if not children:
            return None

        names = [_strip_tag(child.tag) for child in children]
        if len(children) >= 2 and len(set(names)) == 1:
            return children
        if len(children) == 1 and _strip_tag(element.tag).lower() in _ROW_CONTAINER_TAGS:
            return children
        return None

    def _find_row_elements(self, root: ET.Element) -> list[ET.Element] | None:
        """Find the best repeated child collection to treat as rows."""
        direct = self._homogeneous_children(root)
        if direct is not None:
            return direct

        queue = list(root)
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
        *,
        options: XmlOptions,
    ) -> Any:
        """Convert one XML element into a Python scalar / dict / list-friendly object."""
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
                self._element_to_python(child, options=options)
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
        *,
        options: XmlOptions,
    ) -> list[dict[str, Any]]:
        """Normalize the parsed XML document into row-oriented ``list[dict]`` form."""
        rows = self._find_row_elements(root)
        if rows is not None:
            normalized = [self._element_to_python(row, options=options) for row in rows]
        else:
            normalized = [self._element_to_python(root, options=options)]

        out: list[dict[str, Any]] = []
        for row in normalized:
            if isinstance(row, dict):
                out.append(row)
            else:
                out.append({_VALUE_COLUMN: row})
        return out

    def _load_xml_records(self, *, options: XmlOptions) -> list[dict[str, Any]]:
        """Parse and normalize the XML document into row records."""
        root = self._load_root(options=options)
        if root is None:
            return []
        return self._normalize_root(root, options=options)

    def _read_arrow_batches(
        self,
        *,
        options: XmlOptions,
    ) -> Iterator["pa.RecordBatch"]:
        """Yield Arrow batches from the XML buffer."""
        records = self._load_xml_records(options=options)
        if not records:
            return

        batch = pa.RecordBatch.from_pylist(records)

        if options.columns is not None:
            table = pa.Table.from_batches([batch]).select(options.columns)
            yield from table.to_batches()
            return

        yield batch

    def _value_to_xml(
        self,
        parent: ET.Element,
        key: str,
        value: Any,
        *,
        options: XmlOptions,
    ) -> None:
        """Append one Python value under ``parent`` as XML."""
        if key.startswith(options.attr_prefix):
            parent.set(key[len(options.attr_prefix):], "" if value is None else str(value))
            return

        if key == options.text_key:
            parent.text = "" if value is None else str(value)
            return

        if isinstance(value, list):
            for item in value:
                child = ET.SubElement(parent, key)
                self._object_to_xml(child, item, options=options)
            return

        child = ET.SubElement(parent, key)
        self._object_to_xml(child, value, options=options)

    def _object_to_xml(
        self,
        element: ET.Element,
        value: Any,
        *,
        options: XmlOptions,
    ) -> None:
        """Populate ``element`` from a Python scalar or mapping."""
        if isinstance(value, dict):
            for key, item in value.items():
                self._value_to_xml(element, key, item, options=options)
            return

        if isinstance(value, list):
            singular = element.tag[:-1] if element.tag.endswith("s") and len(element.tag) > 1 else "item"
            for item in value:
                child = ET.SubElement(element, singular)
                self._object_to_xml(child, item, options=options)
            return

        if value is not None:
            element.text = str(value)

    def _write_arrow_batches(
        self,
        *,
        batches: Iterator["pa.RecordBatch"],
        schema: "pa.Schema",
        options: XmlOptions,
    ) -> None:
        """Write Arrow batches as a row-oriented XML document."""

        all_batches = list(batches)
        if not all_batches:
            records: list[dict[str, Any]] = []
        else:
            table = pa.Table.from_batches(all_batches, schema=schema)
            records = table.to_pylist()

        root = ET.Element(options.root_tag)
        for record in records:
            row = ET.SubElement(root, options.row_tag)
            self._object_to_xml(row, record, options=options)

        payload = ET.tostring(
            root,
            encoding=options.encoding,
            xml_declaration=options.xml_declaration,
        )
        self.buffer.replace_with_payload(payload)

    def read_pylist(
        self,
        *,
        options: XmlOptions | None = None,
        **option_kwargs,
    ):
        """Read XML directly into row dictionaries when batching is not requested."""
        resolved = self.check_options(options=options, **option_kwargs)
        batch_size = resolved.batch_size
        if batch_size and batch_size > 0:
            return super().read_pylist(options=resolved)

        codec = self.codec
        if codec is None:
            return self._load_xml_records(options=resolved)

        buf, _ = self._decompressed_buffer()
        original = self.buffer
        try:
            self.buffer = buf
            return self._load_xml_records(options=resolved)
        finally:
            self.buffer = original
            buf.close()

    def write_pylist(
        self,
        data: list[dict],
        *,
        options: XmlOptions | None = None,
        **option_kwargs,
    ):
        """Write a list of row dictionaries directly as XML."""
        resolved = self.check_options(options=options, **option_kwargs)

        if self.skip_write(mode=resolved.mode):
            return

        batch_size = resolved.batch_size
        if resolved.mode in (SaveMode.APPEND, SaveMode.UPSERT) or (batch_size and batch_size > 0):
            return super().write_pylist(
                data,
                options=resolved,
            )

        root = ET.Element(resolved.root_tag)
        for record in data:
            row = ET.SubElement(root, resolved.row_tag)
            self._object_to_xml(row, record, options=resolved)

        payload = ET.tostring(
            root,
            encoding=resolved.encoding,
            xml_declaration=resolved.xml_declaration,
        )

        codec = self.codec
        if codec is not None:
            from .bytes_io import BytesIO as _BIO

            plain = _BIO(payload, config=self.buffer.config)
            self._compress_into_buffer(plain)
        else:
            self.buffer.replace_with_payload(payload)

        return None
