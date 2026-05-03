"""Unit tests for :class:`MemoryArrowIO`.

Covers construction, schema retention, the save-mode branches the
class exposes (OVERWRITE / APPEND / IGNORE / ERROR_IF_EXISTS), and
that the buffer stays out of the media-type registry.
"""

from __future__ import annotations

import pytest

from yggdrasil.io.buffer.base import TabularIO
from yggdrasil.io.buffer.memory import MemoryArrowIO
from yggdrasil.io.enums import Mode
from .._helpers import sample_table


class TestMemoryArrowIOBasics:
    def test_default_mime_type_is_none_so_not_registered(self):
        # Returning None keeps the class out of the media-type
        # registry — in-memory holders never want factory dispatch.
        assert MemoryArrowIO.default_mime_type() is None

    def test_is_a_tabular_io(self):
        assert isinstance(MemoryArrowIO(), TabularIO)

    def test_empty_construction(self):
        io = MemoryArrowIO()
        assert io.is_empty()
        assert len(io) == 0
        assert not io
        assert io.cached
        assert io.schema is None
        assert list(io.read_arrow_batches()) == []

    def test_construction_from_table_keeps_schema(self):
        table = sample_table()
        io = MemoryArrowIO(table)
        assert not io.is_empty()
        assert io.num_rows == table.num_rows
        assert io.schema is not None
        assert io.schema.names == table.schema.names

    def test_construction_from_record_batch(self):
        table = sample_table()
        batch = table.to_batches()[0]
        io = MemoryArrowIO(batch)
        assert io.num_rows == batch.num_rows
        assert io.schema == batch.schema

    def test_explicit_schema_preserved_when_empty(self):
        table = sample_table()
        io = MemoryArrowIO(schema=table.schema)
        assert io.is_empty()
        assert io.schema == table.schema


class TestMemoryArrowIOReadWrite:
    def test_read_arrow_table_round_trips(self):
        table = sample_table()
        io = MemoryArrowIO(table)
        out = io.read_arrow_table()
        assert out.equals(table)

    def test_overwrite_replaces_contents(self):
        io = MemoryArrowIO(sample_table())
        replacement = sample_table().slice(0, 1)
        io.write_arrow_batches(replacement.to_batches())
        assert io.num_rows == 1

    def test_append_grows_contents(self):
        io = MemoryArrowIO(sample_table())
        io.write_arrow_batches(sample_table().to_batches(), mode=Mode.APPEND)
        assert io.num_rows == 2 * sample_table().num_rows

    def test_ignore_skips_when_non_empty(self):
        io = MemoryArrowIO(sample_table())
        original = io.num_rows
        io.write_arrow_batches(
            sample_table().slice(0, 1).to_batches(), mode=Mode.IGNORE,
        )
        assert io.num_rows == original

    def test_error_if_exists_raises_on_non_empty(self):
        io = MemoryArrowIO(sample_table())
        with pytest.raises(FileExistsError):
            io.write_arrow_batches(
                sample_table().to_batches(), mode=Mode.ERROR_IF_EXISTS,
            )

    def test_first_write_seeds_schema(self):
        io = MemoryArrowIO()
        io.write_arrow_batches(sample_table().to_batches())
        assert io.schema is not None


class TestMemoryArrowIOPersist:
    def test_persist_with_data_replaces_contents(self):
        io = MemoryArrowIO(sample_table())
        replacement = sample_table().slice(0, 2)
        io.persist(data=replacement)
        assert io.num_rows == 2

    def test_unpersist_clears_batches_keeps_schema(self):
        # Schema is kept on the side so an empty buffer can still
        # answer schema queries — `unpersist` only drops the rows.
        io = MemoryArrowIO(sample_table())
        schema = io.schema
        io.unpersist()
        assert io.is_empty()
        assert io.schema == schema
