"""Unit tests for :class:`yggdrasil.data.record.Record`.

Covers Mapping protocol, schema-singleton invariant (same Schema
identity across sibling records), positional + name-keyed access,
construction from tuple / dict, and the Arrow-batch streaming
constructor.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data.record import Record
from yggdrasil.data.schema import Schema


def _schema() -> Schema:
    return Schema.from_arrow(
        pa.schema([("x", pa.int64()), ("y", pa.string())])
    )


class TestRecordConstruction:
    def test_tuple_values_align_with_schema_fields(self):
        s = _schema()
        r = Record((1, "a"), s)
        assert r["x"] == 1
        assert r["y"] == "a"
        assert r[0] == 1
        assert r[1] == "a"
        assert r.values == (1, "a")
        assert r.schema is s

    def test_dict_values_realigned_to_schema_field_order(self):
        s = _schema()
        # Pass dict in REVERSE order — Record should re-align to the
        # schema's declared order so values[0]==x, values[1]==y.
        r = Record({"y": "a", "x": 1}, s)
        assert r.values == (1, "a")

    def test_dict_missing_key_lands_as_none(self):
        s = _schema()
        r = Record({"x": 1}, s)
        assert r["y"] is None

    def test_wrong_arity_raises(self):
        s = _schema()
        with pytest.raises(ValueError, match="does not match"):
            Record((1, "a", "extra"), s)


class TestRecordMappingProtocol:
    def test_iter_yields_field_names(self):
        s = _schema()
        r = Record((1, "a"), s)
        assert list(iter(r)) == ["x", "y"]
        assert list(r.keys()) == ["x", "y"]

    def test_items_returns_name_value_pairs(self):
        s = _schema()
        r = Record((1, "a"), s)
        assert list(r.items()) == [("x", 1), ("y", "a")]

    def test_get_with_default(self):
        s = _schema()
        r = Record((1, "a"), s)
        assert r.get("x") == 1
        assert r.get("missing") is None
        assert r.get("missing", 42) == 42

    def test_in_operator(self):
        s = _schema()
        r = Record((1, "a"), s)
        assert "x" in r
        assert "missing" not in r

    def test_unknown_name_raises_keyerror(self):
        s = _schema()
        r = Record((1, "a"), s)
        with pytest.raises(KeyError, match="No field named 'missing'"):
            _ = r["missing"]

    def test_eq_dict(self):
        s = _schema()
        r = Record((1, "a"), s)
        assert r == {"x": 1, "y": "a"}

    def test_unhashable(self):
        s = _schema()
        r = Record((1, "a"), s)
        with pytest.raises(TypeError):
            hash(r)


class TestRecordFromArrowBatches:
    def test_shares_one_schema_across_all_records(self):
        # Singleton-schema invariant: every Record from one stream
        # carries the SAME Schema instance, not copies.
        batches = pa.table({"x": [1, 2, 3], "y": ["a", "b", "c"]}).to_batches()
        records = list(Record.from_arrow_batches(batches))
        assert len(records) == 3
        first_schema = records[0].schema
        for r in records[1:]:
            assert r.schema is first_schema

    def test_values_round_trip_from_arrow(self):
        batches = pa.table({"x": [1, 2], "y": ["a", "b"]}).to_batches()
        records = list(Record.from_arrow_batches(batches))
        assert [r.to_dict() for r in records] == [
            {"x": 1, "y": "a"},
            {"x": 2, "y": "b"},
        ]

    def test_explicit_schema_overrides_first_batch_inference(self):
        s = _schema()
        batches = pa.table({"x": [1], "y": ["a"]}).to_batches()
        records = list(Record.from_arrow_batches(batches, schema=s))
        assert records[0].schema is s
