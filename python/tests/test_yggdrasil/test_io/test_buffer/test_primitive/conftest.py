"""Shared fixtures for the IO leaf test suite.

These fixtures provide:

- ``arrow_table`` / ``arrow_batches`` — canonical small datasets used
  across every leaf's round-trip tests.
- ``schema_drift_tables`` — three tables with overlapping but distinct
  column sets, used to exercise ``concat_with_schema_union`` and the
  append-via-rewrite schema-merge path.
- ``upsert_tables`` — existing/incoming pair with overlap on a
  match-by key, used to exercise UPSERT semantics.
- ``tmp_path``-derived path bindings — a Path object pointing into
  the test's tempdir, plus a ``LocalIO`` factory that builds
  bound-IO instances.
- ``fake_codec`` — an instrumented codec that records compress/
  decompress calls so we can verify the round-trip fired.

All fixtures are scoped to ``function`` so per-test mutations don't
leak.
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from yggdrasil.io.enums import Codec


# ---------------------------------------------------------------------------
# Canonical datasets
# ---------------------------------------------------------------------------


@pytest.fixture
def arrow_table() -> pa.Table:
    """A small canonical table — three rows, mixed types.

    Used as the round-trip target across every leaf. Mixed types
    (int, str, float, bool) flush out string-coercion bugs in
    formats like CSV/XML/XLSX that don't preserve types natively.
    """
    return pa.table({
        "id": pa.array([1, 2, 3], type=pa.int64()),
        "name": pa.array(["alpha", "beta", "gamma"], type=pa.string()),
        "value": pa.array([1.5, 2.5, 3.5], type=pa.float64()),
        "active": pa.array([True, False, True], type=pa.bool_()),
    })


@pytest.fixture
def arrow_batches(arrow_table: pa.Table) -> list[pa.RecordBatch]:
    """The canonical table re-chunked as two batches.

    Two batches catches code that processes the first batch
    differently from subsequent ones (the ``first = next(iter, None)``
    pattern in every leaf).
    """
    return arrow_table.to_batches(max_chunksize=2)


@pytest.fixture
def empty_arrow_table() -> pa.Table:
    """A schema-only table with no rows."""
    return pa.table({
        "id": pa.array([], type=pa.int64()),
        "name": pa.array([], type=pa.string()),
    })


@pytest.fixture
def schema_drift_tables() -> list[pa.Table]:
    """Three tables with overlapping-but-distinct column sets.

    - t0: {a, b}
    - t1: {a, c}
    - t2: {b, c, d}

    Column union order is [a, b, c, d] (first-seen across all).
    Used by ``concat_with_schema_union`` and ``append_via_rewrite``
    schema-merge tests.
    """
    return [
        pa.table({"a": [1, 2], "b": ["x", "y"]}),
        pa.table({"a": [3, 4], "c": [1.1, 2.2]}),
        pa.table({"b": ["z"], "c": [9.9], "d": [True]}),
    ]


@pytest.fixture
def upsert_tables() -> tuple[pa.Table, pa.Table, list[str]]:
    """An (existing, incoming, match_by) triple for UPSERT tests.

    Existing rows have keys 1, 2, 3. Incoming rows have keys 2, 3, 4
    — keys 2 and 3 overlap and should be replaced; key 1 should
    survive verbatim; key 4 should be appended.
    """
    existing = pa.table({
        "key": [1, 2, 3],
        "value": ["old-1", "old-2", "old-3"],
    })
    incoming = pa.table({
        "key": [2, 3, 4],
        "value": ["new-2", "new-3", "new-4"],
    })
    return existing, incoming, ["key"]


# ---------------------------------------------------------------------------
# Tabular equality helpers
# ---------------------------------------------------------------------------


def assert_tables_equal(actual: pa.Table, expected: pa.Table, *, ignore_order: bool = False):
    """Assert two pa.Tables are equal, with optional row-order ignore.

    pyarrow's Table.equals checks column order strictly. For the
    UPSERT path where row order is implementation-defined, pass
    ``ignore_order=True`` and the helper sorts both sides by the
    first column before comparing.
    """
    assert actual.column_names == expected.column_names, (
        f"Column names differ: {actual.column_names} vs {expected.column_names}"
    )

    if ignore_order:
        first_col = actual.column_names[0]
        actual = actual.sort_by(first_col)
        expected = expected.sort_by(first_col)

    assert actual.equals(expected), (
        f"Tables differ.\nActual:\n{actual.to_pydict()}\n"
        f"Expected:\n{expected.to_pydict()}"
    )


def assert_round_trip(io_factory, table: pa.Table, *, options=None):
    """Write *table* through a fresh IO, read it back, assert equal.

    *io_factory* is a zero-arg callable returning a fresh IO. Two
    instances are built — one for write, one for read — to verify
    the round-trip survives a clean re-open rather than relying on
    in-memory state.
    """
    writer = io_factory()
    with writer:
        writer.write_arrow_table(table, options=options)
        # Capture the bytes after writing — we'll restore them in
        # the reader to simulate a fresh-open scenario.
        writer.seek(0)
        payload = writer.read()

    reader = io_factory()
    with reader:
        reader.write(payload)
        reader.seek(0)
        result = reader.read_arrow_table(options=options)

    assert_tables_equal(result, table)
