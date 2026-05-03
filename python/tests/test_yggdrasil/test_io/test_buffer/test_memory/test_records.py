"""TabularIO record streaming surface, exercised on `MemoryArrowIO`.

The base-class default impl of `_read_records` and `_write_records`
shouldn't depend on a specific concrete leaf — verify on the
in-memory holder so the tests stay snappy and skip-free.
"""

from __future__ import annotations

from collections.abc import Mapping

import pyarrow as pa

from yggdrasil.data.record import Record
from yggdrasil.data.schema import Schema
from yggdrasil.io.buffer.memory import MemoryArrowIO


def _table() -> pa.Table:
    return pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})


class TestTabularIOReadRecords:
    def test_read_records_yields_record_objects(self):
        io = MemoryArrowIO(_table())
        records = list(io.read_records())
        assert len(records) == 3
        assert all(isinstance(r, Record) for r in records)
        assert all(isinstance(r, Mapping) for r in records)

    def test_read_records_share_one_schema_instance(self):
        io = MemoryArrowIO(_table())
        records = list(io.read_records())
        first = records[0].schema
        for r in records[1:]:
            assert r.schema is first

    def test_to_records_is_alias(self):
        io = MemoryArrowIO(_table())
        assert [r.to_dict() for r in io.to_records()] == [
            r.to_dict() for r in io.read_records()
        ]


class TestTabularIOWriteRecords:
    def test_write_records_round_trips(self):
        # Build records by hand against a known schema, write them
        # into a fresh MemoryArrowIO, read them back as records.
        schema = Schema.from_arrow(
            pa.schema([("a", pa.int64()), ("b", pa.string())])
        )
        records = [Record((i, chr(ord("x") + i)), schema) for i in range(3)]

        io = MemoryArrowIO()
        io.write_records(records)

        out = list(io.read_records())
        assert [r.to_dict() for r in out] == [
            {"a": 0, "b": "x"},
            {"a": 1, "b": "y"},
            {"a": 2, "b": "z"},
        ]

    def test_write_records_accepts_plain_dict_rows(self):
        # The default impl accepts any Mapping — Record subclass not
        # required. Useful for callers that have row dicts already.
        io = MemoryArrowIO()
        io.write_records([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
        out = list(io.read_records())
        assert [r.to_dict() for r in out] == [
            {"a": 1, "b": "x"},
            {"a": 2, "b": "y"},
        ]
