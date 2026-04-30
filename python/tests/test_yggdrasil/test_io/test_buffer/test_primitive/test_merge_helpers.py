"""Tests for the pure-function merge helpers on :class:`DataIO`.

These three helpers are pure functions over pyarrow Tables — no IO
state, no side effects. Tests exercise:

- ``concat_with_schema_union`` — schema alignment, null-fill, empty
  table handling, single-table fast-path.
- ``_filter_out_matches`` — single-column fast path (pyarrow ``is_in``)
  and multi-column tuple-set path.
- ``merge_upsert_tables`` — the four corner cases (both empty, only
  existing, only incoming, both populated) plus a realistic overlap
  scenario.
"""

from __future__ import annotations

import pytest
import pyarrow as pa

from yggdrasil.io.buffer.primitive.csv_io import CsvIO  # any leaf will do


@pytest.fixture
def io() -> CsvIO:
    """Any DataIO subclass works for testing the pure helpers."""
    return CsvIO()


# ===========================================================================
# concat_with_schema_union
# ===========================================================================


class TestConcatWithSchemaUnion:

    def test_identical_schemas_passthrough(self, io):
        t1 = pa.table({"a": [1, 2], "b": ["x", "y"]})
        t2 = pa.table({"a": [3], "b": ["z"]})
        result = io.concat_with_schema_union([t1, t2])
        assert result.column_names == ["a", "b"]
        assert result.num_rows == 3
        assert result["a"].to_pylist() == [1, 2, 3]
        assert result["b"].to_pylist() == ["x", "y", "z"]

    def test_left_only_columns_null_filled_on_right(self, io):
        t1 = pa.table({"a": [1, 2], "b": ["x", "y"]})
        t2 = pa.table({"a": [3]})
        result = io.concat_with_schema_union([t1, t2])
        assert result.column_names == ["a", "b"]
        assert result["a"].to_pylist() == [1, 2, 3]
        assert result["b"].to_pylist() == ["x", "y", None]

    def test_right_only_columns_null_filled_on_left(self, io):
        t1 = pa.table({"a": [1, 2]})
        t2 = pa.table({"a": [3], "c": [9.9]})
        result = io.concat_with_schema_union([t1, t2])
        assert result.column_names == ["a", "c"]
        assert result["a"].to_pylist() == [1, 2, 3]
        assert result["c"].to_pylist() == [None, None, 9.9]

    def test_three_way_drift_first_seen_order(self, io, schema_drift_tables):
        """Column union preserves first-seen order across all inputs."""
        result = io.concat_with_schema_union(schema_drift_tables)
        assert result.column_names == ["a", "b", "c", "d"]
        assert result.num_rows == 5  # 2 + 2 + 1

    def test_empty_input_returns_empty_table(self, io):
        result = io.concat_with_schema_union([])
        assert result.num_rows == 0
        assert result.num_columns == 0

    def test_skips_truly_empty_tables(self, io):
        empty = pa.table({})
        t1 = pa.table({"a": [1]})
        result = io.concat_with_schema_union([empty, t1, empty])
        assert result.column_names == ["a"]
        assert result.num_rows == 1

    def test_single_table_returned_unchanged(self, io):
        t = pa.table({"a": [1, 2]})
        result = io.concat_with_schema_union([t])
        # The fast path returns the same object; identity check.
        assert result is t


# ===========================================================================
# _filter_out_matches
# ===========================================================================


class TestFilterOutMatches:

    def test_single_column_drops_overlapping_keys(self, io):
        existing = pa.table({"key": [1, 2, 3], "v": ["a", "b", "c"]})
        incoming = pa.table({"key": [2, 3], "v": ["b2", "c2"]})
        result = io._filter_out_matches(existing, incoming, ["key"])
        assert result["key"].to_pylist() == [1]
        assert result["v"].to_pylist() == ["a"]

    def test_single_column_no_overlap(self, io):
        existing = pa.table({"key": [1, 2], "v": ["a", "b"]})
        incoming = pa.table({"key": [99], "v": ["z"]})
        result = io._filter_out_matches(existing, incoming, ["key"])
        assert result.num_rows == 2

    def test_single_column_full_overlap(self, io):
        existing = pa.table({"key": [1, 2]})
        incoming = pa.table({"key": [1, 2]})
        result = io._filter_out_matches(existing, incoming, ["key"])
        assert result.num_rows == 0

    def test_multi_column_match(self, io):
        existing = pa.table({
            "k1": [1, 1, 2, 2],
            "k2": ["a", "b", "a", "b"],
            "v": [10, 20, 30, 40],
        })
        # Match on (1, "a") and (2, "b") → drop rows 0 and 3, keep 1, 2.
        incoming = pa.table({"k1": [1, 2], "k2": ["a", "b"], "v": [99, 99]})
        result = io._filter_out_matches(existing, incoming, ["k1", "k2"])
        assert result["v"].to_pylist() == [20, 30]

    def test_existing_missing_match_column_returns_unchanged(self, io):
        existing = pa.table({"other": [1, 2]})
        incoming = pa.table({"key": [1, 2], "other": [9, 9]})
        result = io._filter_out_matches(existing, incoming, ["key"])
        # Existing has no "key" column → no overlap possible.
        assert result.num_rows == 2


# ===========================================================================
# merge_upsert_tables
# ===========================================================================


class TestMergeUpsertTables:

    def test_both_empty(self, io):
        existing = pa.table({"key": pa.array([], type=pa.int64())})
        incoming = pa.table({"key": pa.array([], type=pa.int64())})
        result = io.merge_upsert_tables(existing, incoming, match_by=["key"])
        assert result.num_rows == 0

    def test_only_existing(self, io):
        existing = pa.table({"key": [1, 2], "v": ["a", "b"]})
        incoming = pa.table({"key": pa.array([], type=pa.int64()), "v": pa.array([], type=pa.string())})
        result = io.merge_upsert_tables(existing, incoming, match_by=["key"])
        assert result.num_rows == 2
        assert result["v"].to_pylist() == ["a", "b"]

    def test_only_incoming(self, io):
        existing = pa.table({"key": pa.array([], type=pa.int64()), "v": pa.array([], type=pa.string())})
        incoming = pa.table({"key": [1, 2], "v": ["x", "y"]})
        result = io.merge_upsert_tables(existing, incoming, match_by=["key"])
        assert result.num_rows == 2
        assert result["v"].to_pylist() == ["x", "y"]

    def test_overlapping_existing_loses_to_incoming(self, io, upsert_tables):
        existing, incoming, match_by = upsert_tables
        result = io.merge_upsert_tables(existing, incoming, match_by=match_by)

        assert result.num_rows == 4  # key 1 + key 2/3/4
        # key 1 from existing, keys 2/3/4 from incoming.
        rows = {row["key"]: row["value"] for row in result.to_pylist()}
        assert rows == {1: "old-1", 2: "new-2", 3: "new-3", 4: "new-4"}

    def test_match_by_missing_in_incoming_raises(self, io):
        existing = pa.table({"key": [1]})
        incoming = pa.table({"other": [2]})
        with pytest.raises(ValueError, match="missing from incoming"):
            io.merge_upsert_tables(existing, incoming, match_by=["key"])

    def test_empty_match_by_raises(self, io):
        with pytest.raises(ValueError, match="at least one"):
            io.merge_upsert_tables(
                pa.table({"k": [1]}),
                pa.table({"k": [1]}),
                match_by=[],
            )
