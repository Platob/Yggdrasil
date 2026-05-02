"""Shared helpers for the per-media-type buffer tests.

Each ``test_{media_io}/`` package mirrors the same three-file layout:

- ``test_base.py``   — construction, mime registration, save-mode
                       resolution, byte-level round-trip.
- ``test_arrow.py``  — Arrow Table / RecordBatch round-trip.
- ``test_polars.py`` — Polars DataFrame round-trip (skipped on
                       missing optional dep).

These helpers centralize the sample data and the
``importorskip`` boilerplate so the per-media test files stay
focused on the format-specific behavior.
"""

from __future__ import annotations

import pytest
import pyarrow as pa


def sample_table() -> pa.Table:
    """Three-row, two-column Arrow table the suites round-trip."""
    return pa.Table.from_pylist(
        [
            {"a": 1, "b": "henry"},
            {"a": 2, "b": "hub"},
            {"a": 3, "b": "settle"},
        ]
    )


def sample_batches() -> list[pa.RecordBatch]:
    """Two RecordBatches sharing the same schema."""
    schema = pa.schema([("a", pa.int64()), ("b", pa.string())])
    first = pa.RecordBatch.from_pylist(
        [{"a": 1, "b": "henry"}, {"a": 2, "b": "hub"}], schema=schema
    )
    second = pa.RecordBatch.from_pylist(
        [{"a": 3, "b": "settle"}], schema=schema
    )
    return [first, second]


def require_polars():
    """``pytest.importorskip`` on polars; returns the module."""
    return pytest.importorskip("polars")


def sample_polars_frame():
    """Polars DataFrame matching :func:`sample_table`."""
    pl = require_polars()
    return pl.DataFrame({"a": [1, 2, 3], "b": ["henry", "hub", "settle"]})
