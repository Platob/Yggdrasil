"""Arrow-side casts and streaming for :class:`StructType`.

Three layers under test:

* **Array casts** — struct/map/list source arrays cast to a struct
  target. Reorder children, fill missing children with defaults,
  preserve null rows.
* **Tabular cast** — Table / RecordBatch column rebuild against a
  merged schema. Same reorder + fill semantics, just at the table
  level.
* **Streaming** — :func:`cast_arrow_batch_iterator` flattens
  per-batch tabular cast over an iterator and hands the cast stream
  to :func:`rechunk_arrow_batches`, which honours
  ``byte_size`` (target bytes per batch) and ``row_size`` (hard cap
  on rows per batch).
"""
from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data import Field, Schema
from yggdrasil.data.options import CastOptions
from yggdrasil.data.types import IntegerType
from yggdrasil.data.types.nested.array import ArrayType
from yggdrasil.arrow.cast import rechunk_arrow_batches
from yggdrasil.data.types.nested.struct import (
    StructType,
    cast_arrow_batch_iterator,
    cast_arrow_list_array,
    cast_arrow_map_array,
    cast_arrow_struct_array,
    cast_arrow_tabular,
)


def _batch(rows: list[dict], schema: pa.Schema) -> pa.RecordBatch:
    return pa.RecordBatch.from_pylist(rows, schema=schema)


# ---------------------------------------------------------------------------
# struct → struct
# ---------------------------------------------------------------------------


class TestCastStructArray:
    def test_reorders_fields_and_fills_missing(
        self,
        source_struct_field: Field,
        target_struct_field: Field,
    ) -> None:
        array = pa.array(
            [
                {"a": 1, "b": "x"},
                {"a": 2, "b": "y"},
                None,
            ],
            type=pa.struct(
                [pa.field("a", pa.int64()), pa.field("b", pa.string())]
            ),
        )

        options = CastOptions(
            source_field=source_struct_field,
            target_field=target_struct_field,
        )

        result = cast_arrow_struct_array(array, options)

        assert isinstance(result, pa.StructArray)
        assert result.type == pa.struct(
            [
                pa.field("b", pa.string()),
                pa.field("c", pa.int64()),
                pa.field("a", pa.int64()),
            ]
        )
        assert result.to_pylist() == [
            {"b": "x", "c": None, "a": 1},
            {"b": "y", "c": None, "a": 2},
            None,
        ]

    def test_returns_input_when_no_cast_needed(
        self,
        source_struct_field: Field,
    ) -> None:
        # Same source as target → need_cast() is False → identity.
        arrow_struct = pa.struct(
            [pa.field("a", pa.int64()), pa.field("b", pa.string())]
        )
        array = pa.array([{"a": 1, "b": "x"}], type=arrow_struct)

        options = CastOptions(
            source_field=source_struct_field,
            target_field=source_struct_field,
        )
        assert cast_arrow_struct_array(array, options) is array

    def test_returns_input_when_target_is_none(self) -> None:
        arrow_struct = pa.struct([pa.field("a", pa.int64())])
        array = pa.array([{"a": 1}], type=arrow_struct)

        source_field = Field(
            name="src",
            dtype=StructType(
                fields=[
                    Field(
                        name="a",
                        dtype=IntegerType(byte_size=8, signed=True),
                        nullable=True,
                    )
                ]
            ),
            nullable=True,
        )

        options = CastOptions(source_field=source_field, target_field=None)
        assert cast_arrow_struct_array(array, options) is array

    def test_rejects_non_struct_source(self) -> None:
        array = pa.array([[1, 2]], type=pa.list_(pa.int64()))

        source_field = Field(
            name="src",
            dtype=ArrayType.from_item(
                IntegerType(byte_size=8, signed=True).to_field(name="item"),
            ),
            nullable=True,
        )
        target_field = Field(
            name="tgt",
            dtype=StructType(
                fields=[
                    Field(
                        name="a",
                        dtype=IntegerType(byte_size=8, signed=True),
                        nullable=True,
                    )
                ]
            ),
            nullable=True,
        )

        options = CastOptions(
            source_field=source_field, target_field=target_field
        )

        with pytest.raises(pa.ArrowInvalid, match="Cannot cast"):
            cast_arrow_struct_array(array, options)


# ---------------------------------------------------------------------------
# map → struct (named-key extraction)
# ---------------------------------------------------------------------------


class TestCastMapArray:
    def test_extracts_named_keys_to_struct(
        self,
        source_map_field: Field,
        target_struct_field: Field,
    ) -> None:
        array = pa.array(
            [[("a", 1), ("b", 2)], [("b", 3)], None],
            type=pa.map_(pa.string(), pa.int64()),
        )

        options = CastOptions(
            source_field=source_map_field,
            target_field=target_struct_field,
        )

        result = cast_arrow_map_array(array, options)

        assert isinstance(result, pa.StructArray)
        assert result.to_pylist() == [
            {"b": "2", "c": None, "a": 1},
            {"b": "3", "c": None, "a": None},
            None,
        ]


# ---------------------------------------------------------------------------
# list → struct (positional mapping)
# ---------------------------------------------------------------------------


class TestCastListArray:
    def test_maps_by_position_and_fills_missing(
        self,
        source_list_field: Field,
        target_list_to_struct_field: Field,
    ) -> None:
        array = pa.array(
            [[1, 2, 3], [4], None],
            type=pa.list_(pa.int64()),
        )

        options = CastOptions(
            source_field=source_list_field,
            target_field=target_list_to_struct_field,
        )

        result = cast_arrow_list_array(array, options)

        assert isinstance(result, pa.StructArray)
        assert result.to_pylist() == [
            {"first": 1, "second": "2", "third": 3},
            {"first": 4, "second": None, "third": None},
            None,
        ]


# ---------------------------------------------------------------------------
# Tabular casts
# ---------------------------------------------------------------------------


class TestCastTabular:
    def test_table_reorders_columns_and_adds_missing(
        self,
        source_tabular_schema: Schema,
        target_tabular_schema: Schema,
    ) -> None:
        table = pa.table(
            {
                "a": pa.array([1, 2, None], type=pa.int64()),
                "b": pa.array(["x", "y", "z"], type=pa.string()),
            }
        )

        options = CastOptions(
            source_field=source_tabular_schema,
            target_field=target_tabular_schema,
        )

        result = cast_arrow_tabular(table, options)

        assert isinstance(result, pa.Table)
        assert result.schema == pa.schema(
            [
                pa.field("b", pa.string()),
                pa.field("c", pa.int64()),
                pa.field("a", pa.int64()),
            ]
        )
        assert result.to_pylist() == [
            {"b": "x", "c": None, "a": 1},
            {"b": "y", "c": None, "a": 2},
            {"b": "z", "c": None, "a": None},
        ]

    def test_record_batch_round_trip_keeps_shape(
        self,
        source_tabular_schema: Schema,
        target_tabular_schema: Schema,
    ) -> None:
        batch = pa.record_batch(
            [
                pa.array([1, 2], type=pa.int64()),
                pa.array(["x", "y"], type=pa.string()),
            ],
            names=["a", "b"],
        )

        options = CastOptions(
            source_field=source_tabular_schema,
            target_field=target_tabular_schema,
        )

        result = cast_arrow_tabular(batch, options)

        assert isinstance(result, pa.RecordBatch)
        assert result.to_pylist() == [
            {"b": "x", "c": None, "a": 1},
            {"b": "y", "c": None, "a": 2},
        ]

    def test_returns_input_when_target_is_none(self) -> None:
        table = pa.table({"a": [1, 2, 3]})
        options = CastOptions(target_field=None)

        assert cast_arrow_tabular(table, options) is table

    def test_rejects_non_table_input(self) -> None:
        options = CastOptions(
            source_field=Schema(inner_fields=[]),
            target_field=Schema(inner_fields=[]),
        )

        with pytest.raises(TypeError, match="Unsupported tabular type"):
            cast_arrow_tabular({"a": [1, 2, 3]}, options)


# ---------------------------------------------------------------------------
# Tabular casts — tricky column dtypes.
#
# Cover the column shapes that historically slip past a "reorder by
# name" sweep: fixed-precision decimal, timestamp with a non-naive
# timezone, a nested struct whose children themselves swap order, and a
# list<struct> whose inner struct children also swap order. The same
# fixtures back the Pandas / Polars / Spark equivalents so divergences
# show up as one engine failing a peer's assertion.
# ---------------------------------------------------------------------------


class TestCastTabularTrickyTypes:
    def _build_table(
        self, source_schema: Schema, as_record_batch: bool = False
    ):
        from datetime import datetime, timezone
        from decimal import Decimal

        arrow_schema = source_schema.to_arrow_schema()
        rows = [
            {
                "drop_me": 99,
                "amount": Decimal("1.23"),
                "ts": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "nested": {"x": 10, "y": "a"},
                "items": [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}],
                "name": "row-1",
            },
            {
                "drop_me": 100,
                "amount": Decimal("4.56"),
                "ts": datetime(2024, 6, 30, 12, 30, tzinfo=timezone.utc),
                "nested": {"x": 20, "y": "b"},
                "items": [{"x": 3, "y": "c"}],
                "name": "row-2",
            },
            {
                "drop_me": None,
                "amount": None,
                "ts": None,
                "nested": None,
                "items": None,
                "name": None,
            },
        ]
        if as_record_batch:
            return pa.RecordBatch.from_pylist(rows, schema=arrow_schema)
        return pa.Table.from_pylist(rows, schema=arrow_schema)

    def test_table_reorders_selects_and_preserves_tricky_dtypes(
        self,
        tricky_source_schema: Schema,
        tricky_target_schema: Schema,
    ) -> None:
        table = self._build_table(tricky_source_schema)

        options = CastOptions(
            source_field=tricky_source_schema,
            target_field=tricky_target_schema,
        )

        result = cast_arrow_tabular(table, options)

        # Column order matches the target schema and ``drop_me`` is gone.
        assert result.column_names == [
            "ts",
            "amount",
            "items",
            "nested",
            "name",
            "missing",
        ]
        # Tricky dtypes survive the rebuild without precision / tz loss.
        assert result.schema == tricky_target_schema.to_arrow_schema()
        assert result.schema.field("amount").type == pa.decimal128(10, 2)
        assert result.schema.field("ts").type == pa.timestamp("us", tz="UTC")
        # Nested children also reordered (y before x).
        assert [f.name for f in result.schema.field("nested").type] == ["y", "x"]
        assert [
            f.name for f in result.schema.field("items").type.value_type
        ] == ["y", "x"]

        rows = result.to_pylist()
        # Inner struct rows came through with swapped child order — and
        # the ``missing`` column is filled with nulls because no source
        # field maps to it.
        assert rows[0]["nested"] == {"y": "a", "x": 10}
        assert rows[0]["items"] == [{"y": "a", "x": 1}, {"y": "b", "x": 2}]
        assert rows[0]["missing"] is None
        assert rows[1]["nested"] == {"y": "b", "x": 20}
        # Final row is all-null source → result row is null per column.
        assert rows[2]["nested"] is None
        assert rows[2]["items"] is None
        assert rows[2]["amount"] is None
        assert rows[2]["ts"] is None

    def test_record_batch_path_matches_table_path(
        self,
        tricky_source_schema: Schema,
        tricky_target_schema: Schema,
    ) -> None:
        # Both code paths share the per-column rebuild — confirm
        # RecordBatch input → RecordBatch output and the same row data.
        batch = self._build_table(tricky_source_schema, as_record_batch=True)
        options = CastOptions(
            source_field=tricky_source_schema,
            target_field=tricky_target_schema,
        )

        result = cast_arrow_tabular(batch, options)

        assert isinstance(result, pa.RecordBatch)
        assert result.schema == tricky_target_schema.to_arrow_schema()
        assert result.to_pylist() == cast_arrow_tabular(
            self._build_table(tricky_source_schema), options
        ).to_pylist()

    def test_widens_integer_dtype_during_reorder(
        self,
        string_type,
    ) -> None:
        # int32 source column must cast to int64 inside the same call
        # that reorders columns — the per-column dtype cast and the
        # column rebuild have to compose.
        from yggdrasil.data.types import IntegerType

        int32 = IntegerType(byte_size=4, signed=True)
        int64 = IntegerType(byte_size=8, signed=True)

        source = Schema(
            inner_fields=[
                Field(name="small", dtype=int32, nullable=True),
                Field(name="label", dtype=string_type, nullable=True),
            ]
        )
        target = Schema(
            inner_fields=[
                Field(name="label", dtype=string_type, nullable=True),
                Field(name="small", dtype=int64, nullable=True),
            ]
        )

        table = pa.table(
            {
                "small": pa.array([1, 2, 3], type=pa.int32()),
                "label": pa.array(["a", "b", "c"], type=pa.string()),
            }
        )

        result = cast_arrow_tabular(
            table, CastOptions(source_field=source, target_field=target)
        )

        assert result.column_names == ["label", "small"]
        assert result.schema.field("small").type == pa.int64()
        assert result.to_pylist() == [
            {"label": "a", "small": 1},
            {"label": "b", "small": 2},
            {"label": "c", "small": 3},
        ]

    def test_resolves_target_alias_to_source_column(self, string_type) -> None:
        # Source carries the legacy name ``old_amount``; target asks for
        # ``amount`` with the old name configured as an alias. The Arrow
        # tabular cast resolves via ``Field.select_in_field`` which walks
        # name → alias, so the renamed column must come through.
        from yggdrasil.data.types import IntegerType

        int64 = IntegerType(byte_size=8, signed=True)

        amount = Field(name="amount", dtype=int64, nullable=True).with_alias(
            "old_amount"
        )
        source = Schema(
            inner_fields=[
                Field(name="old_amount", dtype=int64, nullable=True),
                Field(name="name", dtype=string_type, nullable=True),
            ]
        )
        target = Schema(inner_fields=[amount, Field(name="name", dtype=string_type, nullable=True)])

        table = pa.table({"old_amount": [10, 20], "name": ["x", "y"]})

        result = cast_arrow_tabular(
            table, CastOptions(source_field=source, target_field=target)
        )

        assert result.column_names == ["amount", "name"]
        assert result.to_pylist() == [
            {"amount": 10, "name": "x"},
            {"amount": 20, "name": "y"},
        ]


# ---------------------------------------------------------------------------
# Streaming entry point — cast + rechunk in one call.
# ---------------------------------------------------------------------------


class TestCastBatchIterator:
    def test_casts_each_batch_against_target(
        self,
        source_tabular_schema: Schema,
        target_tabular_schema: Schema,
    ) -> None:
        source_arrow = source_tabular_schema.to_arrow_schema()
        batches = [
            _batch([{"a": 1, "b": "x"}], source_arrow),
            _batch([{"a": 2, "b": "y"}, {"a": 3, "b": "z"}], source_arrow),
        ]

        options = CastOptions(
            source_field=source_tabular_schema,
            target_field=target_tabular_schema,
        )

        result = list(cast_arrow_batch_iterator(iter(batches), options))

        assert len(result) == 2
        target_arrow = target_tabular_schema.to_arrow_schema()
        for r in result:
            assert r.schema == target_arrow
        assert [row for b in result for row in b.to_pylist()] == [
            {"b": "x", "c": None, "a": 1},
            {"b": "y", "c": None, "a": 2},
            {"b": "z", "c": None, "a": 3},
        ]

    def test_passes_through_when_no_sizing_set(
        self,
        source_tabular_schema: Schema,
    ) -> None:
        """No row/byte cap → batches flow through 1:1."""
        schema = source_tabular_schema.to_arrow_schema()
        inputs = [_batch([{"a": i, "b": "x"}], schema) for i in range(5)]

        options = CastOptions(
            source_field=source_tabular_schema,
            target_field=source_tabular_schema,
        )

        result = list(cast_arrow_batch_iterator(iter(inputs), options))

        assert len(result) == len(inputs)
        assert sum(r.num_rows for r in result) == len(inputs)

    def test_byte_size_coalesces_small_inputs(
        self,
        source_tabular_schema: Schema,
    ) -> None:
        schema = source_tabular_schema.to_arrow_schema()
        inputs = [_batch([{"a": i, "b": "x"}], schema) for i in range(20)]

        # Pick byte_size large enough to coalesce most inputs into 1-2 batches.
        target_bytes = inputs[0].nbytes * 8

        options = CastOptions(
            source_field=source_tabular_schema,
            target_field=source_tabular_schema,
            byte_size=target_bytes,
        )

        result = list(cast_arrow_batch_iterator(iter(inputs), options))

        assert len(result) < len(inputs)
        assert sum(r.num_rows for r in result) == len(inputs)

    def test_row_size_caps_rows_per_batch(
        self,
        source_tabular_schema: Schema,
    ) -> None:
        """row_size alone → fixed-size chunks, no byte tracking."""
        schema = source_tabular_schema.to_arrow_schema()
        big = _batch([{"a": i, "b": "x"} for i in range(25)], schema)

        options = CastOptions(
            source_field=source_tabular_schema,
            target_field=source_tabular_schema,
            row_size=10,
        )

        result = list(cast_arrow_batch_iterator(iter([big]), options))

        assert [b.num_rows for b in result] == [10, 10, 5]

    def test_handles_empty_iterator(
        self,
        target_tabular_schema: Schema,
    ) -> None:
        options = CastOptions(target_field=target_tabular_schema)
        assert list(cast_arrow_batch_iterator(iter([]), options)) == []

    def test_rejects_non_record_batch_items(self) -> None:
        options = CastOptions()
        with pytest.raises(TypeError, match="expected pa.RecordBatch"):
            list(cast_arrow_batch_iterator(iter([{"a": 1}]), options))

    def test_rejects_non_record_batch_items_after_first(
        self,
        source_tabular_schema: Schema,
    ) -> None:
        schema = source_tabular_schema.to_arrow_schema()
        good = _batch([{"a": 1, "b": "x"}], schema)
        options = CastOptions(target_field=source_tabular_schema)

        gen = cast_arrow_batch_iterator(iter([good, "not-a-batch"]), options)
        next(gen)
        with pytest.raises(TypeError, match="expected pa.RecordBatch"):
            next(gen)


# ---------------------------------------------------------------------------
# Direct rechunker behavior — exposed for callers that already cast.
# ---------------------------------------------------------------------------


class TestRechunk:
    def test_passthrough_when_no_caps_given(
        self,
        source_tabular_schema: Schema,
    ) -> None:
        schema = source_tabular_schema.to_arrow_schema()
        inputs = [_batch([{"a": i, "b": "x"}], schema) for i in range(3)]

        out = list(rechunk_arrow_batches(iter(inputs)))

        assert out == inputs

    def test_passthrough_when_byte_size_is_zero(
        self,
        source_tabular_schema: Schema,
    ) -> None:
        schema = source_tabular_schema.to_arrow_schema()
        inputs = [_batch([{"a": i, "b": "x"}], schema) for i in range(3)]

        out = list(rechunk_arrow_batches(iter(inputs), byte_size=0))

        assert out == inputs

    def test_byte_size_slices_oversized_batch(
        self,
        source_tabular_schema: Schema,
    ) -> None:
        schema = source_tabular_schema.to_arrow_schema()
        big = _batch(
            [{"a": i, "b": "x"} for i in range(100)],
            schema,
        )

        bytes_per_row = max(1, big.nbytes // big.num_rows)
        target = bytes_per_row * 10

        out = list(
            rechunk_arrow_batches(iter([big]), byte_size=target)
        )

        assert len(out) > 1
        assert sum(b.num_rows for b in out) == big.num_rows
        for b in out[:-1]:
            assert b.nbytes <= target * 2  # generous upper bound

    def test_byte_size_coalesces_small_batches(
        self,
        source_tabular_schema: Schema,
    ) -> None:
        schema = source_tabular_schema.to_arrow_schema()
        inputs = [_batch([{"a": i, "b": "x"}], schema) for i in range(10)]
        target = inputs[0].nbytes * 5

        out = list(
            rechunk_arrow_batches(iter(inputs), byte_size=target)
        )

        assert len(out) < len(inputs)
        assert sum(b.num_rows for b in out) == sum(b.num_rows for b in inputs)

    def test_drops_empty_input_batches(
        self,
        source_tabular_schema: Schema,
    ) -> None:
        schema = source_tabular_schema.to_arrow_schema()
        empty = _batch([], schema)
        one = _batch([{"a": 1, "b": "x"}], schema)

        out = list(
            rechunk_arrow_batches(
                iter([empty, one, empty]),
                byte_size=one.nbytes * 100,
            )
        )

        assert sum(b.num_rows for b in out) == 1

    def test_row_size_only_emits_fixed_chunks(
        self,
        source_tabular_schema: Schema,
    ) -> None:
        schema = source_tabular_schema.to_arrow_schema()
        big = _batch([{"a": i, "b": "x"} for i in range(13)], schema)

        out = list(
            rechunk_arrow_batches(iter([big]), row_size=5)
        )

        assert [b.num_rows for b in out] == [5, 5, 3]
        assert sum(b.num_rows for b in out) == big.num_rows

    def test_row_size_only_drops_empty_batches(
        self,
        source_tabular_schema: Schema,
    ) -> None:
        schema = source_tabular_schema.to_arrow_schema()
        empty = _batch([], schema)
        one = _batch([{"a": 1, "b": "x"}], schema)

        out = list(
            rechunk_arrow_batches(
                iter([empty, one, empty]),
                row_size=10,
            )
        )

        assert [b.num_rows for b in out] == [1]

    def test_row_size_caps_byte_size_target(
        self,
        source_tabular_schema: Schema,
    ) -> None:
        """With both knobs set, row_size is the hard upper bound."""
        schema = source_tabular_schema.to_arrow_schema()
        inputs = [_batch([{"a": i, "b": "x"}], schema) for i in range(40)]

        # Generous byte target — would otherwise pull large chunks.
        big_byte_target = inputs[0].nbytes * 1000
        row_cap = 5

        out = list(
            rechunk_arrow_batches(
                iter(inputs),
                byte_size=big_byte_target,
                row_size=row_cap,
            )
        )

        assert all(b.num_rows <= row_cap for b in out)
        assert sum(b.num_rows for b in out) == sum(b.num_rows for b in inputs)
