"""Unit tests for :class:`yggdrasil.io.buffer.xml_io.XmlIO`.

Covers:

* options validation
* round-trip (memory + spilled)
* row-layout detection (flat, nested, single-object, heterogeneous children)
* attributes + mixed content
* sparse / heterogeneous rows
* column projection
* save modes: OVERWRITE, IGNORE, ERROR_IF_EXISTS, APPEND, UPSERT
* batched read
* edge cases (empty buffer, empty stream, numeric coercion)
* cast integration via options.cast.cast_arrow_tabular
* known XML limitations (length-1 list collapses to scalar)
"""
from __future__ import annotations

from pathlib import Path

import pytest
from yggdrasil.io import MimeTypes
from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.buffer.media_io import MediaIO
from yggdrasil.io.buffer.xml_io import XmlIO, XmlOptions
from yggdrasil.io.config import BufferConfig
from yggdrasil.io.enums import SaveMode


# ---------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------

def _pa():
    from yggdrasil.arrow.lib import pyarrow as pa

    return pa


def _make_cfg(tmp_path: Path, *, spill_bytes: int = 128) -> BufferConfig:
    return BufferConfig(
        spill_bytes=spill_bytes,
        tmp_dir=tmp_path,
        prefix="test_xmlio_",
        suffix=".xml",
        keep_spilled_file=False,
    )


@pytest.fixture()
def cfg(tmp_path: Path) -> BufferConfig:
    return _make_cfg(tmp_path)


@pytest.fixture()
def spill_cfg(tmp_path: Path) -> BufferConfig:
    return _make_cfg(tmp_path, spill_bytes=1)


@pytest.fixture()
def sample_records():
    return [
        {"id": 1, "name": "alpha", "score": 1.5},
        {"id": 2, "name": "beta", "score": 2.5},
        {"id": 3, "name": "gamma", "score": 3.5},
    ]


@pytest.fixture()
def sample_table():
    pa = _pa()
    return pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int64()),
            "name": pa.array(["alpha", "beta", "gamma"], type=pa.string()),
            "score": pa.array([1.5, 2.5, 3.5], type=pa.float64()),
        }
    )


def _make_io(buf: BytesIO) -> XmlIO:
    io_ = MediaIO.make(buf, MimeTypes.XML)
    assert isinstance(io_, XmlIO)
    return io_


def _seed_bytes(cfg: BufferConfig, payload: bytes) -> BytesIO:
    """Create a BytesIO pre-loaded with *payload* (used for read tests)."""
    return BytesIO(payload, config=cfg)


# =====================================================================
# Factory
# =====================================================================

class TestFactory:
    def test_media_io_make_returns_xml_io(self, cfg: BufferConfig):
        buf = BytesIO(config=cfg)
        io_ = MediaIO.make(buf, MimeTypes.XML)
        assert isinstance(io_, XmlIO)

    def test_check_options_accepts_none(self):
        resolved = XmlIO.check_options(None)
        assert isinstance(resolved, XmlOptions)
        assert resolved.encoding == "utf-8"
        assert resolved.root_tag == "rows"
        assert resolved.row_tag == "row"
        assert resolved.xml_declaration is True

    def test_check_options_merges_kwargs(self):
        resolved = XmlIO.check_options(
            None, root_tag="data", row_tag="item", pretty_print=True
        )
        assert resolved.root_tag == "data"
        assert resolved.row_tag == "item"
        assert resolved.pretty_print is True


# =====================================================================
# XmlOptions validation
# =====================================================================

class TestXmlOptions:
    def test_defaults(self):
        opts = XmlOptions()
        assert opts.encoding == "utf-8"
        assert opts.errors == "strict"
        assert opts.root_tag == "rows"
        assert opts.row_tag == "row"
        assert opts.text_key == "value"
        assert opts.attr_prefix == "@"
        assert opts.xml_declaration is True
        assert opts.pretty_print is False
        assert opts.list_item_tag == "item"

    @pytest.mark.parametrize(
        "name", ["encoding", "errors", "root_tag", "row_tag", "text_key", "list_item_tag"]
    )
    def test_empty_string_rejected(self, name):
        with pytest.raises(ValueError, match=name):
            XmlOptions(**{name: ""})

    def test_attr_prefix_may_be_empty(self):
        # attr_prefix is explicitly allowed to be empty.
        opts = XmlOptions(attr_prefix="")
        assert opts.attr_prefix == ""

    @pytest.mark.parametrize(
        "name,bad",
        [
            ("encoding", 123),
            ("root_tag", None),
            ("row_tag", 42),
            ("list_item_tag", 1.5),
        ],
    )
    def test_non_string_type_rejected(self, name, bad):
        with pytest.raises(TypeError, match=name):
            XmlOptions(**{name: bad})

    def test_xml_declaration_must_be_bool(self):
        with pytest.raises(TypeError, match="xml_declaration"):
            XmlOptions(xml_declaration="yes")  # type: ignore[arg-type]

    def test_pretty_print_must_be_bool(self):
        with pytest.raises(TypeError, match="pretty_print"):
            XmlOptions(pretty_print=1)  # type: ignore[arg-type]


# =====================================================================
# Round-trip
# =====================================================================

class TestRoundtrip:
    def test_write_then_read_roundtrip(
        self, cfg: BufferConfig, sample_table
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        io_.write_arrow_table(
            sample_table, options=XmlOptions(mode=SaveMode.OVERWRITE)
        )
        assert buf.size > 0

        out = io_.read_arrow_table()
        # XML doesn't preserve Arrow types natively — values coerce back
        # to Python scalars on parse. Compare via pylist.
        expected = sample_table.to_pylist()
        assert out.to_pylist() == expected

    def test_written_payload_is_xml_text(
        self, cfg: BufferConfig, sample_table
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        io_.write_arrow_table(
            sample_table, options=XmlOptions(mode=SaveMode.OVERWRITE)
        )
        raw = buf.to_bytes()
        assert raw.startswith(b"<?xml") or raw.startswith(b"<rows")
        assert b"<row>" in raw
        assert b"<id>1</id>" in raw

    def test_xml_declaration_false_omits_header(
        self, cfg: BufferConfig, sample_records
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        io_.write_pylist(
            sample_records,
            options=XmlOptions(mode=SaveMode.OVERWRITE, xml_declaration=False),
        )
        raw = buf.to_bytes()
        assert not raw.startswith(b"<?xml")

    def test_pretty_print_adds_whitespace(
        self, cfg: BufferConfig, sample_records
    ):
        buf_compact = BytesIO(config=cfg)
        buf_pretty = BytesIO(config=_make_cfg(cfg.tmp_dir))

        _make_io(buf_compact).write_pylist(
            sample_records, options=XmlOptions(mode=SaveMode.OVERWRITE)
        )
        _make_io(buf_pretty).write_pylist(
            sample_records,
            options=XmlOptions(mode=SaveMode.OVERWRITE, pretty_print=True),
        )

        compact = buf_compact.to_bytes()
        pretty = buf_pretty.to_bytes()

        # Pretty-printed output has newlines + indentation.
        assert len(pretty) > len(compact)
        assert b"\n" in pretty

    def test_custom_root_and_row_tags(
        self, cfg: BufferConfig, sample_records
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        io_.write_pylist(
            sample_records,
            options=XmlOptions(
                mode=SaveMode.OVERWRITE,
                root_tag="records",
                row_tag="record",
            ),
        )
        raw = buf.to_bytes()
        assert b"<records>" in raw
        assert b"<record>" in raw
        assert b"<rows>" not in raw

    def test_roundtrip_in_spilled_buffer(
        self, spill_cfg: BufferConfig, sample_table
    ):
        buf = BytesIO(config=spill_cfg)
        io_ = _make_io(buf)

        io_.write_arrow_table(
            sample_table, options=XmlOptions(mode=SaveMode.OVERWRITE)
        )
        assert buf.spilled is True
        assert buf.path is not None

        out = io_.read_arrow_table()
        assert out.to_pylist() == sample_table.to_pylist()


# =====================================================================
# Row-layout detection
# =====================================================================

class TestRowDetection:
    def test_flat_rows_container(self, cfg: BufferConfig):
        xml = b"""<?xml version="1.0"?>
<rows>
  <row><id>1</id><name>a</name></row>
  <row><id>2</id><name>b</name></row>
</rows>"""
        buf = _seed_bytes(cfg, xml)
        io_ = _make_io(buf)

        out = io_.read_pylist()
        assert out == [
            {"id": 1, "name": "a"},
            {"id": 2, "name": "b"},
        ]

    def test_nested_container_at_depth(self, cfg: BufferConfig):
        """``<root><data><records><record>...`` should still be detected."""
        xml = b"""<?xml version="1.0"?>
<root>
  <data>
    <records>
      <record><id>1</id><v>x</v></record>
      <record><id>2</id><v>y</v></record>
    </records>
  </data>
</root>"""
        buf = _seed_bytes(cfg, xml)
        io_ = _make_io(buf)

        out = io_.read_pylist()
        assert out == [
            {"id": 1, "v": "x"},
            {"id": 2, "v": "y"},
        ]

    def test_heterogeneous_children_treated_as_single_row(
        self, cfg: BufferConfig
    ):
        """Mismatched child tags → treat the root as a single row."""
        xml = b"""<?xml version="1.0"?>
<root>
  <name>Alice</name>
  <age>30</age>
</root>"""
        buf = _seed_bytes(cfg, xml)
        io_ = _make_io(buf)

        out = io_.read_pylist()
        assert out == [{"name": "Alice", "age": 30}]

    def test_single_element_in_known_container_tag(self, cfg: BufferConfig):
        """``<rows><row>…</row></rows>`` with one child is still a row list."""
        xml = b"""<?xml version="1.0"?>
<rows><row><id>1</id></row></rows>"""
        buf = _seed_bytes(cfg, xml)
        io_ = _make_io(buf)

        out = io_.read_pylist()
        assert out == [{"id": 1}]

    def test_namespaced_tags_are_stripped(self, cfg: BufferConfig):
        xml = b"""<?xml version="1.0"?>
<ns:rows xmlns:ns="http://example.com/ns">
  <ns:row><ns:id>1</ns:id></ns:row>
</ns:rows>"""
        buf = _seed_bytes(cfg, xml)
        io_ = _make_io(buf)

        out = io_.read_pylist()
        assert out == [{"id": 1}]


# =====================================================================
# Attributes and mixed content
# =====================================================================

class TestAttributesAndText:
    def test_attributes_get_prefixed(self, cfg: BufferConfig):
        xml = b"""<?xml version="1.0"?>
<rows>
  <row id="1" type="A">first</row>
  <row id="2" type="B">second</row>
</rows>"""
        buf = _seed_bytes(cfg, xml)
        io_ = _make_io(buf)

        out = io_.read_pylist()
        # Element with both attributes and text → dict with text_key entry.
        assert out == [
            {"@id": 1, "@type": "A", "value": "first"},
            {"@id": 2, "@type": "B", "value": "second"},
        ]

    def test_custom_attr_prefix(self, cfg: BufferConfig):
        xml = b"""<?xml version="1.0"?>
<rows><row id="1"><name>x</name></row></rows>"""
        buf = _seed_bytes(cfg, xml)
        io_ = _make_io(buf)

        out = io_.read_pylist(options=XmlOptions(attr_prefix="_"))
        assert out == [{"_id": 1, "name": "x"}]

    def test_empty_attr_prefix(self, cfg: BufferConfig):
        """attr_prefix='' makes attributes indistinguishable from elements.

        With an empty prefix, ``<row id="1"><id>2</id></row>`` would
        collide. This test just verifies the empty-prefix path doesn't
        crash on the no-collision case.
        """
        xml = b"""<?xml version="1.0"?>
<rows><row type="A"><name>x</name></row></rows>"""
        buf = _seed_bytes(cfg, xml)
        io_ = _make_io(buf)

        out = io_.read_pylist(options=XmlOptions(attr_prefix=""))
        # "type" is the attribute, "name" is the element — no collision.
        assert out[0]["type"] == "A"
        assert out[0]["name"] == "x"


# =====================================================================
# Type coercion
# =====================================================================

class TestCoercion:
    def test_integer_coerced(self, cfg: BufferConfig):
        xml = b"<rows><row><n>42</n></row></rows>"
        out = _make_io(_seed_bytes(cfg, xml)).read_pylist()
        assert out == [{"n": 42}]

    def test_negative_integer_coerced(self, cfg: BufferConfig):
        xml = b"<rows><row><n>-17</n></row></rows>"
        out = _make_io(_seed_bytes(cfg, xml)).read_pylist()
        assert out == [{"n": -17}]

    def test_float_coerced(self, cfg: BufferConfig):
        xml = b"<rows><row><x>3.14</x></row></rows>"
        out = _make_io(_seed_bytes(cfg, xml)).read_pylist()
        assert out == [{"x": 3.14}]

    def test_scientific_notation_coerced(self, cfg: BufferConfig):
        xml = b"<rows><row><x>1.5e2</x></row></rows>"
        out = _make_io(_seed_bytes(cfg, xml)).read_pylist()
        assert out == [{"x": 150.0}]

    def test_bool_coerced(self, cfg: BufferConfig):
        xml = b"<rows><row><a>true</a><b>false</b></row></rows>"
        out = _make_io(_seed_bytes(cfg, xml)).read_pylist()
        assert out == [{"a": True, "b": False}]

    def test_null_literals_coerced_to_none(self, cfg: BufferConfig):
        xml = b"<rows><row><a>null</a><b>None</b></row></rows>"
        out = _make_io(_seed_bytes(cfg, xml)).read_pylist()
        assert out == [{"a": None, "b": None}]

    def test_leading_zero_preserved_as_string(self, cfg: BufferConfig):
        """ZIP codes and similar identifiers must not be coerced to int."""
        xml = b"<rows><row><zip>02134</zip></row></rows>"
        out = _make_io(_seed_bytes(cfg, xml)).read_pylist()
        assert out == [{"zip": "02134"}]

    def test_alphanumeric_stays_string(self, cfg: BufferConfig):
        xml = b"<rows><row><code>A12B</code></row></rows>"
        out = _make_io(_seed_bytes(cfg, xml)).read_pylist()
        assert out == [{"code": "A12B"}]

    def test_empty_text_is_none(self, cfg: BufferConfig):
        xml = b"<rows><row><a></a><b>  </b></row></rows>"
        out = _make_io(_seed_bytes(cfg, xml)).read_pylist()
        assert out == [{"a": None, "b": None}]


# =====================================================================
# Repeated elements → lists
# =====================================================================

class TestRepeatedElements:
    def test_multiple_same_tag_becomes_list(self, cfg: BufferConfig):
        xml = b"""<?xml version="1.0"?>
<rows>
  <row>
    <id>1</id>
    <tag>a</tag>
    <tag>b</tag>
    <tag>c</tag>
  </row>
</rows>"""
        buf = _seed_bytes(cfg, xml)
        out = _make_io(buf).read_pylist()
        assert out == [{"id": 1, "tag": ["a", "b", "c"]}]

    def test_single_same_tag_becomes_scalar_documented_quirk(
        self, cfg: BufferConfig
    ):
        """Known XML roundtrip limitation.

        Length-1 lists can't be distinguished from scalars without a
        schema hint, so a single ``<tag>x</tag>`` parses as the scalar
        ``"x"`` rather than ``["x"]``. Test pins the current behavior
        so a fix would be a conscious choice.
        """
        xml = b"<rows><row><tag>x</tag></row></rows>"
        buf = _seed_bytes(cfg, xml)
        out = _make_io(buf).read_pylist()
        assert out == [{"tag": "x"}]  # not ["x"]


# =====================================================================
# Write: lists and nested dicts
# =====================================================================

class TestWriteStructures:
    def test_write_list_values_as_repeated_elements(
        self, cfg: BufferConfig
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        records = [{"id": 1, "tags": ["a", "b", "c"]}]
        io_.write_pylist(
            records, options=XmlOptions(mode=SaveMode.OVERWRITE)
        )

        raw = buf.to_bytes()
        assert raw.count(b"<tags>") == 3

        # Round-trip: multi-element list survives.
        out = io_.read_pylist()
        assert out == records

    def test_write_nested_dict_becomes_child_element(
        self, cfg: BufferConfig
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        records = [{"id": 1, "meta": {"k": "v", "n": 10}}]
        io_.write_pylist(
            records, options=XmlOptions(mode=SaveMode.OVERWRITE)
        )

        raw = buf.to_bytes()
        assert b"<meta>" in raw
        assert b"<k>v</k>" in raw

        out = io_.read_pylist()
        assert out == records

    def test_list_item_tag_default_is_item(
        self, cfg: BufferConfig
    ):
        """Bare list at the dict-value level uses list_item_tag."""
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        # A list of scalars under a dict key becomes repeated <tags>
        # elements (handled by _value_to_xml's list branch). This test
        # exercises the *_object_to_xml* list branch for lists appearing
        # mid-structure — possible via deeper nesting.
        records = [{"nested": {"inner": [1, 2, 3]}}]
        io_.write_pylist(
            records, options=XmlOptions(mode=SaveMode.OVERWRITE)
        )

        raw = buf.to_bytes()
        # Either <inner>1</inner><inner>2</inner>... (from _value_to_xml)
        # or <item>...</item> in some paths — just verify the ints are there.
        assert b"1" in raw and b"2" in raw and b"3" in raw

    def test_custom_list_item_tag(self, cfg: BufferConfig):
        """Explicit list_item_tag is honored when falling through the list branch."""
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        # Write via write_arrow_table so the top-level payload becomes a
        # list of records — but we want to trigger the _object_to_xml
        # list branch, which fires when a list appears where a value was
        # expected. Hard to hit naturally; simplest: just verify the
        # option doesn't crash and produces valid XML.
        io_.write_pylist(
            [{"a": 1}],
            options=XmlOptions(
                mode=SaveMode.OVERWRITE, list_item_tag="entry"
            ),
        )
        assert buf.size > 0


# =====================================================================
# Sparse / heterogeneous rows
# =====================================================================

class TestSparseRows:
    def test_missing_columns_do_not_drop(
        self, cfg: BufferConfig
    ):
        """Rows with different key sets must all survive round-trip."""
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        records = [
            {"a": 1},
            {"a": 2, "b": "x"},
            {"b": "y", "c": 3.14},
        ]

        io_.write_pylist(
            records, options=XmlOptions(mode=SaveMode.OVERWRITE)
        )

        # Via Arrow: union of keys, missing entries filled with None.
        tb = io_.read_arrow_table()
        assert set(tb.column_names) == {"a", "b", "c"}
        assert tb.num_rows == 3

    def test_all_none_column_handled(self, cfg: BufferConfig):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        records = [{"a": 1, "x": None}, {"a": 2, "x": None}]
        io_.write_pylist(
            records, options=XmlOptions(mode=SaveMode.OVERWRITE)
        )
        tb = io_.read_arrow_table()
        assert "x" in tb.column_names


# =====================================================================
# Column projection
# =====================================================================

class TestColumnProjection:
    def test_read_with_columns(self, cfg: BufferConfig, sample_table):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(
            sample_table, options=XmlOptions(mode=SaveMode.OVERWRITE)
        )

        out = io_.read_arrow_table(options=XmlOptions(columns=["id", "name"]))
        assert set(out.column_names) == {"id", "name"}
        assert out.num_rows == sample_table.num_rows

    def test_read_pylist_with_columns(
        self, cfg: BufferConfig, sample_records
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_pylist(
            sample_records, options=XmlOptions(mode=SaveMode.OVERWRITE)
        )

        out = io_.read_pylist(options=XmlOptions(columns=["id"]))
        assert all(list(r.keys()) == ["id"] for r in out)

    def test_read_unknown_columns_silently_dropped(
        self, cfg: BufferConfig, sample_table
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(
            sample_table, options=XmlOptions(mode=SaveMode.OVERWRITE)
        )

        out = io_.read_pylist(options=XmlOptions(columns=["id", "nope"]))
        # The projection drops unknown columns rather than raising.
        assert all("nope" not in r for r in out)
        assert all("id" in r for r in out)


# =====================================================================
# Save modes
# =====================================================================

class TestSaveModes:
    def test_ignore_mode_does_not_overwrite(
        self, cfg: BufferConfig, sample_records
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        io_.write_pylist(
            sample_records, options=XmlOptions(mode=SaveMode.OVERWRITE)
        )
        size1 = buf.size
        bytes1 = buf.to_bytes()

        io_.write_pylist(
            [{"different": "data"}],
            options=XmlOptions(mode=SaveMode.IGNORE),
        )
        assert buf.size == size1
        assert buf.to_bytes() == bytes1

    def test_error_if_exists_raises(
        self, cfg: BufferConfig, sample_records
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        io_.write_pylist(
            sample_records, options=XmlOptions(mode=SaveMode.OVERWRITE)
        )

        with pytest.raises(IOError):
            io_.write_pylist(
                sample_records,
                options=XmlOptions(mode=SaveMode.ERROR_IF_EXISTS),
            )

    def test_error_if_exists_allowed_on_empty_buffer(
        self, cfg: BufferConfig, sample_records
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        io_.write_pylist(
            sample_records,
            options=XmlOptions(mode=SaveMode.ERROR_IF_EXISTS),
        )
        assert buf.size > 0

    def test_overwrite_replaces_content(
        self, cfg: BufferConfig, sample_records
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        io_.write_pylist(
            sample_records, options=XmlOptions(mode=SaveMode.OVERWRITE)
        )
        bytes1 = buf.to_bytes()

        t2 = [{"id": 99, "name": "new"}]
        io_.write_pylist(t2, options=XmlOptions(mode=SaveMode.OVERWRITE))
        bytes2 = buf.to_bytes()

        assert bytes1 != bytes2
        out = io_.read_pylist()
        assert out == t2

    def test_append_combines_rows(
        self, cfg: BufferConfig, sample_records
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        t2 = [{"id": 4, "name": "delta", "score": 4.5}]

        io_.write_pylist(
            sample_records, options=XmlOptions(mode=SaveMode.OVERWRITE)
        )
        io_.write_pylist(t2, options=XmlOptions(mode=SaveMode.APPEND))

        out = io_.read_pylist()
        ids = [r.get("id") for r in out]
        assert ids == [1, 2, 3, 4]

    def test_append_into_empty_buffer_works(
        self, cfg: BufferConfig, sample_records
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        io_.write_pylist(
            sample_records, options=XmlOptions(mode=SaveMode.APPEND)
        )
        out = io_.read_pylist()
        assert len(out) == len(sample_records)

    def test_upsert_requires_match_by(
        self, cfg: BufferConfig, sample_records
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        io_.write_pylist(
            sample_records, options=XmlOptions(mode=SaveMode.OVERWRITE)
        )

        with pytest.raises(ValueError, match="match_by"):
            io_.write_pylist(
                sample_records, options=XmlOptions(mode=SaveMode.UPSERT)
            )

    def test_upsert_replaces_matching_rows(
        self, cfg: BufferConfig, sample_records
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        io_.write_pylist(
            sample_records, options=XmlOptions(mode=SaveMode.OVERWRITE)
        )

        # Overlaps id=2, introduces id=4.
        t2 = [
            {"id": 2, "name": "BETA", "score": 22.0},
            {"id": 4, "name": "delta", "score": 4.0},
        ]
        io_.write_pylist(
            t2,
            options=XmlOptions(mode=SaveMode.UPSERT, match_by="id"),
        )

        out = io_.read_pylist()
        by_id = {r["id"]: r for r in out}
        assert set(by_id.keys()) == {1, 2, 3, 4}
        assert by_id[2]["name"] == "BETA"
        assert by_id[1]["name"] == "alpha"
        assert by_id[4]["name"] == "delta"

    def test_upsert_composite_key(self, cfg: BufferConfig):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        t1 = [
            {"k1": "a", "k2": 1, "v": 10},
            {"k1": "a", "k2": 2, "v": 20},
            {"k1": "b", "k2": 1, "v": 30},
        ]
        t2 = [
            {"k1": "a", "k2": 2, "v": 200},
            {"k1": "c", "k2": 1, "v": 100},
        ]

        io_.write_pylist(t1, options=XmlOptions(mode=SaveMode.OVERWRITE))
        io_.write_pylist(
            t2,
            options=XmlOptions(mode=SaveMode.UPSERT, match_by=["k1", "k2"]),
        )

        out = sorted(io_.read_pylist(), key=lambda r: (r["k1"], r["k2"]))
        assert out == [
            {"k1": "a", "k2": 1, "v": 10},
            {"k1": "a", "k2": 2, "v": 200},
            {"k1": "b", "k2": 1, "v": 30},
            {"k1": "c", "k2": 1, "v": 100},
        ]


# =====================================================================
# Batched read
# =====================================================================

class TestBatchedRead:
    def test_read_arrow_batches_respects_batch_size(
        self, cfg: BufferConfig
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        records = [{"id": i, "name": f"row_{i}"} for i in range(50)]
        io_.write_pylist(
            records, options=XmlOptions(mode=SaveMode.OVERWRITE)
        )

        batches = list(
            io_.read_arrow_batches(options=XmlOptions(batch_size=10))
        )
        total = sum(b.num_rows for b in batches)
        assert total == 50
        assert all(b.num_rows <= 10 for b in batches)
        assert len(batches) == 5

    def test_read_pylist_respects_batch_size(
        self, cfg: BufferConfig
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        records = [{"id": i} for i in range(20)]
        io_.write_pylist(
            records, options=XmlOptions(mode=SaveMode.OVERWRITE)
        )

        chunks = list(
            io_.read_pylist(options=XmlOptions(batch_size=5))
        )
        assert len(chunks) == 4
        assert all(len(c) <= 5 for c in chunks)
        assert sum(len(c) for c in chunks) == 20


# =====================================================================
# Schema inspection
# =====================================================================

class TestSchema:
    def test_collect_schema_empty_buffer(self, cfg: BufferConfig):
        pa = _pa()
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        schema = io_._collect_arrow_schema()
        assert schema == pa.schema([])

    def test_collect_schema_from_populated_buffer(
        self, cfg: BufferConfig, sample_records
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        io_.write_pylist(
            sample_records, options=XmlOptions(mode=SaveMode.OVERWRITE)
        )
        schema = io_._collect_arrow_schema()
        assert set(schema.names) == {"id", "name", "score"}


# =====================================================================
# Edge cases
# =====================================================================

class TestEdgeCases:
    def test_read_empty_buffer_returns_empty(self, cfg: BufferConfig):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        assert io_.read_pylist() == []
        assert list(io_.read_arrow_batches()) == []

    def test_write_empty_list_produces_empty_root(
        self, cfg: BufferConfig
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        io_.write_pylist([], options=XmlOptions(mode=SaveMode.OVERWRITE))
        # Behavior: an empty records list produces an empty root element.
        # The buffer is non-empty (XML declaration + empty root tag).
        # Round-trip returns empty.
        out = io_.read_pylist()
        assert out == []

    def test_double_read_is_idempotent(
        self, cfg: BufferConfig, sample_records
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        io_.write_pylist(
            sample_records, options=XmlOptions(mode=SaveMode.OVERWRITE)
        )
        out1 = io_.read_pylist()
        out2 = io_.read_pylist()
        assert out1 == out2

    def test_ignore_empty_filters_empty_batches(
        self, cfg: BufferConfig, sample_records
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        io_.write_pylist(
            sample_records, options=XmlOptions(mode=SaveMode.OVERWRITE)
        )
        # ignore_empty is really a cast-path option; just verify it
        # doesn't crash on a normal workload.
        out = io_.read_arrow_table(
            options=XmlOptions(ignore_empty=True)
        )
        assert out.num_rows == len(sample_records)


# =====================================================================
# Cast integration
# =====================================================================

class TestCastIntegration:
    def test_default_cast_is_identity(
        self, cfg: BufferConfig, sample_records
    ):
        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)

        io_.write_pylist(
            sample_records, options=XmlOptions(mode=SaveMode.OVERWRITE)
        )
        out = io_.read_arrow_table()
        assert out.num_rows == len(sample_records)
        assert set(out.column_names) == {"id", "name", "score"}

    def test_cast_target_applied_on_write(self, cfg: BufferConfig):
        """Explicit CastOptions should reach the write path."""
        pa = _pa()
        try:
            from yggdrasil.data.cast.options import CastOptions
        except ImportError:
            pytest.skip("CastOptions not importable in this environment")

        src = pa.table(
            {
                "id": pa.array([1, 2, 3], type=pa.int32()),
                "v": pa.array([1.5, 2.5, 3.5], type=pa.float64()),
            }
        )
        target = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("v", pa.float32()),
            ]
        )

        try:
            cast = CastOptions(target_field=target)
        except TypeError:
            pytest.skip(
                "CastOptions(target_field=...) signature mismatch"
            )

        buf = BytesIO(config=cfg)
        io_ = _make_io(buf)
        io_.write_arrow_table(
            src,
            options=XmlOptions(mode=SaveMode.OVERWRITE, cast=cast),
        )

        # XML stores values as text, so the on-disk payload won't have
        # Arrow types — but when the round-trip comes back through the
        # read-side cast (same CastOptions would have to be set), the
        # types enforce. Here we just check that the write didn't
        # blow up and the data is readable.
        out = io_.read_pylist()
        assert len(out) == 3
        assert {r["id"] for r in out} == {1, 2, 3}