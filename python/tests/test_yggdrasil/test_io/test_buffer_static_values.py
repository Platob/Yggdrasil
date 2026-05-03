"""Tests for :attr:`TabularIO.static_values`.

The injection helpers
(:func:`inject_static_values_into_batch` /
:func:`inject_static_values_into_table`) get standalone
coverage; the IO-side wiring is exercised through
:class:`MemoryArrowIO` since every subclass funnels through the
same :meth:`TabularIO.read_arrow_batches` /
:meth:`TabularIO.read_arrow_table` entry points.
"""

from __future__ import annotations

import pyarrow as pa

from yggdrasil.io.buffer.base import (
    inject_static_values_into_batch,
    inject_static_values_into_table,
)
from yggdrasil.io.buffer.memory import MemoryArrowIO


def _table():
    return pa.Table.from_pylist([{"a": 1}, {"a": 2}, {"a": 3}])


class TestStaticValuesField:
    def test_default_is_none(self):
        # A plain MemoryArrowIO carries no static values.
        assert MemoryArrowIO().static_values is None

    def test_constructor_accepts_static_values_kwarg(self):
        io = MemoryArrowIO(static_values={"tag": "cache"})
        assert io.static_values == {"tag": "cache"}

    def test_static_values_input_is_defensively_copied(self):
        # Mutating the caller's mapping after construction must
        # not leak through onto the IO — a constructor that
        # captures the same dict reference would silently drop
        # rows on read after the caller deletes a key.
        source = {"tag": "cache"}
        io = MemoryArrowIO(static_values=source)
        del source["tag"]
        assert io.static_values == {"tag": "cache"}

    def test_empty_mapping_normalizes_to_none(self):
        # ``static_values={}`` ⇒ ``None`` so the read-path guard
        # stays a single ``is None`` check on the hot path.
        assert MemoryArrowIO(static_values={}).static_values is None


class TestInjectionOnRead:
    def test_static_columns_appear_on_every_batch(self):
        t = _table()
        io = MemoryArrowIO(
            schema=t.schema, static_values={"tag": "cache", "year": 2025},
        )
        io.write_arrow_table(t)

        batches = list(io.read_arrow_batches())
        assert batches  # at least one batch
        for b in batches:
            assert "tag" in b.schema.names
            assert "year" in b.schema.names
            # Constant value across every row.
            assert b.column("tag").to_pylist() == ["cache"] * b.num_rows
            assert b.column("year").to_pylist() == [2025] * b.num_rows

    def test_static_columns_appear_on_read_arrow_table(self):
        t = _table()
        io = MemoryArrowIO(schema=t.schema, static_values={"tag": "cache"})
        io.write_arrow_table(t)
        out = io.read_arrow_table()
        assert "tag" in out.column_names
        assert out.column("tag").to_pylist() == ["cache"] * out.num_rows

    def test_existing_column_is_left_untouched(self):
        # Source already carries a ``tag`` column → static value
        # is *not* injected over it. Additive, never overriding.
        t = pa.Table.from_pylist([{"a": 1, "tag": "src"}, {"a": 2, "tag": "src"}])
        io = MemoryArrowIO(schema=t.schema, static_values={"tag": "static"})
        io.write_arrow_table(t)
        out = io.read_arrow_table()
        assert out.column("tag").to_pylist() == ["src", "src"]

    def test_post_construction_assignment_works(self):
        # The slot is intentionally mutable — callers tagging an
        # existing IO can do ``io.static_values = {...}`` without
        # rebuilding it.
        t = _table()
        io = MemoryArrowIO(schema=t.schema)
        io.write_arrow_table(t)
        io.static_values = {"injected": True}
        out = io.read_arrow_table()
        assert "injected" in out.column_names
        assert out.column("injected").to_pylist() == [True] * out.num_rows


class TestStandaloneHelpers:
    def test_inject_into_batch_skips_existing(self):
        rb = pa.RecordBatch.from_pylist([{"a": 1, "tag": "src"}])
        out = inject_static_values_into_batch(rb, {"tag": "stat", "extra": 99})
        # ``tag`` survives, ``extra`` is appended.
        assert out.column("tag").to_pylist() == ["src"]
        assert out.column("extra").to_pylist() == [99]

    def test_inject_into_table_skips_existing(self):
        t = pa.Table.from_pylist([{"a": 1, "tag": "src"}])
        out = inject_static_values_into_table(t, {"tag": "stat", "extra": 99})
        assert out.column("tag").to_pylist() == ["src"]
        assert out.column("extra").to_pylist() == [99]

    def test_inject_none_or_empty_is_passthrough(self):
        rb = pa.RecordBatch.from_pylist([{"a": 1}])
        assert inject_static_values_into_batch(rb, None) is rb
        assert inject_static_values_into_batch(rb, {}) is rb

    def test_inject_pa_scalar_keeps_declared_dtype(self):
        # Wrapping the value in :class:`pyarrow.Scalar` lets the
        # caller pin the dtype explicitly — useful when the
        # default Python-value inference would pick the wrong
        # type (e.g. timestamp-as-int for a datetime).
        rb = pa.RecordBatch.from_pylist([{"a": 1}, {"a": 2}])
        out = inject_static_values_into_batch(
            rb, {"v": pa.scalar(7, type=pa.int32())},
        )
        assert out.schema.field("v").type == pa.int32()
