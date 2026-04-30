"""Tests for :class:`JsonIO` (newline-delimited JSON).

Coverage:

- Round-trip of canonical types via JSONL.
- One JSON object per line — no array wrapping.
- APPEND honest concatenation.
- UPSERT via rewrite.
"""

from __future__ import annotations

import json

import pytest
from yggdrasil.io.enums import Mode
from yggdrasil.io.buffer.primitive.json_io import JsonIO


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_round_trip(arrow_table):
    io = JsonIO()
    with io:
        io.write_arrow_table(arrow_table)
        io.seek(0)
        result = io.read_arrow_table()

    # JSON preserves types fairly well except for some int/float
    # boundary cases.
    actual = result.to_pydict()
    expected = arrow_table.to_pydict()

    for col in expected:
        assert actual[col] == expected[col] or actual[col] == pytest.approx(expected[col])


def test_output_is_one_object_per_line(arrow_table):
    io = JsonIO()
    with io:
        io.write_arrow_table(arrow_table)
        io.seek(0)
        raw = io.read().decode("utf-8")

    lines = [ln for ln in raw.split("\n") if ln.strip()]
    assert len(lines) == arrow_table.num_rows

    # Each line should be a parseable JSON object.
    for line in lines:
        parsed = json.loads(line)
        assert isinstance(parsed, dict)


def test_empty_buffer_read_yields_no_batches():
    io = JsonIO()
    with io:
        batches = list(io.read_arrow_batches())
    assert batches == []


# ---------------------------------------------------------------------------
# APPEND
# ---------------------------------------------------------------------------


def test_append_concatenates_lines(arrow_table):
    io = JsonIO()
    with io:
        io.write_arrow_table(arrow_table, mode=Mode.OVERWRITE)
        io.write_arrow_table(arrow_table, mode=Mode.APPEND)
        io.seek(0)
        raw = io.read().decode("utf-8")

    lines = [ln for ln in raw.split("\n") if ln.strip()]
    assert len(lines) == 2 * arrow_table.num_rows


def test_append_to_empty_collapses_to_overwrite(arrow_table):
    io = JsonIO()
    with io:
        io.write_arrow_table(arrow_table, mode=Mode.APPEND)
        io.seek(0)
        result = io.read_arrow_table()
    assert result.num_rows == arrow_table.num_rows


# ---------------------------------------------------------------------------
# UPSERT
# ---------------------------------------------------------------------------


def test_upsert_replaces_overlapping(upsert_tables):
    existing, incoming, match_by = upsert_tables

    io = JsonIO()
    with io:
        io.write_arrow_table(existing, mode=Mode.OVERWRITE)
        io.write_arrow_table(
            incoming, mode=Mode.UPSERT, match_by_names=match_by,
        )
        io.seek(0)
        result = io.read_arrow_table()

    rows = {r["key"]: r["value"] for r in result.to_pylist()}
    assert rows == {1: "old-1", 2: "new-2", 3: "new-3", 4: "new-4"}
