from __future__ import annotations

import pyarrow as pa
from yggdrasil.io import MimeTypes
from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.buffer.media_io import MediaIO
from yggdrasil.io.buffer.xml_io import XmlOptions
from yggdrasil.io.enums import GZIP, MediaType


def test_xmlio_read_repeated_rows():
    buf = BytesIO(
        b"""<?xml version="1.0" encoding="utf-8"?>
<rows>
  <row><id>1</id><name>alice</name><active>true</active></row>
  <row><id>2</id><name>bob</name><active>false</active></row>
</rows>
"""
    )

    table = MediaIO.make(buf, MimeTypes.XML).read_arrow_table()

    assert table.to_pylist() == [
        {"id": 1, "name": "alice", "active": True},
        {"id": 2, "name": "bob", "active": False},
    ]


def test_xmlio_reads_nested_record_container():
    buf = BytesIO(
        b"""<root>
  <meta><source>demo</source></meta>
  <records>
    <record id="1"><name>alice</name><score>10.5</score></record>
    <record id="2"><name>bob</name><score>11.5</score></record>
  </records>
</root>"""
    )

    records = MediaIO.make(buf, MimeTypes.XML).read_pylist()

    assert records == [
        {"@id": 1, "name": "alice", "score": 10.5},
        {"@id": 2, "name": "bob", "score": 11.5},
    ]


def test_xmlio_preserves_repeated_children_as_lists():
    buf = BytesIO(
        b"""<rows>
  <row>
    <id>1</id>
    <tag>a</tag>
    <tag>b</tag>
    <meta><created_by>qa</created_by></meta>
  </row>
</rows>"""
    )

    records = MediaIO.make(buf, MimeTypes.XML).read_pylist()

    assert records == [
        {"id": 1, "tag": ["a", "b"], "meta": {"created_by": "qa"}}
    ]


def test_xmlio_write_and_read_roundtrip():
    buf = BytesIO()
    io_ = MediaIO.make(buf, MimeTypes.XML)
    table = pa.table({"id": [1, 2], "name": ["x", "y"]})

    io_.write_arrow_table(table)
    out = io_.read_arrow_table()

    assert out.to_pylist() == table.to_pylist()


def test_xmlio_gzip_roundtrip():
    buf = BytesIO()
    io_ = MediaIO.make(buf, MediaType(MimeTypes.XML, codec=GZIP))
    table = pa.table({"id": [1, 2], "score": [10.5, 11.5]})

    io_.write_arrow_table(table)

    raw = buf.to_bytes()
    assert raw[:2] == b"\x1f\x8b"
    assert io_.read_arrow_table().to_pylist() == table.to_pylist()


def test_xmlio_custom_text_key_for_mixed_content():
    buf = BytesIO(
        b"""<rows>
  <row id="1">hello<name>alice</name></row>
</rows>"""
    )

    records = MediaIO.make(buf, MimeTypes.XML).read_pylist(
        options=XmlOptions(text_key="text")
    )

    assert records == [{"@id": 1, "name": "alice", "text": "hello"}]
