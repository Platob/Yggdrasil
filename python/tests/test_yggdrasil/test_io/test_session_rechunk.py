"""Unit tests for the Spark response-batch rechunker.

The rechunker is the inner helper that the ``_spark_fetch_misses``
worker uses to keep emitted batches under a byte cap. Driving it
directly with synthetic record batches is much cheaper than spinning
Spark for an end-to-end pass and lets us assert the boundary cases
(grouping vs. solo emission) precisely.
"""

from __future__ import annotations

import pyarrow as pa

from yggdrasil.arrow.cast import rechunk_arrow_batches_by_byte_size


def _row_batch(payload: bytes) -> pa.RecordBatch:
    """One-row record batch carrying *payload* in a binary column."""
    return pa.RecordBatch.from_pydict({"body": [payload]})


def _rows(out: list[pa.RecordBatch]) -> list[bytes]:
    if not out:
        return []
    return pa.Table.from_batches(out).column("body").to_pylist()


class TestRechunkBelowLimit:
    def test_groups_small_rows_into_a_single_batch(self):
        # Five tiny rows, well under any reasonable cap.
        rows = [_row_batch(b"x" * 16) for _ in range(5)]
        out = list(rechunk_arrow_batches_by_byte_size(iter(rows), byte_size=10_000))
        assert len(out) == 1
        assert out[0].num_rows == 5

    def test_empty_input_yields_nothing(self):
        assert list(
            rechunk_arrow_batches_by_byte_size(iter([]), byte_size=10_000)
        ) == []


class TestRechunkAtLimit:
    def test_splits_when_running_total_would_exceed(self):
        # Each row is ~256 B; cap at 600 B forces a split.
        rows = [_row_batch(b"x" * 256) for _ in range(5)]
        out = list(rechunk_arrow_batches_by_byte_size(iter(rows), byte_size=600))
        # Exact counts depend on Arrow's serialization overhead — only
        # assert the cap is honored on multi-row batches and every row
        # makes it through.
        assert sum(b.num_rows for b in out) == 5
        for b in out:
            if b.num_rows > 1:
                assert b.nbytes <= 600


class TestRechunkOversizedRow:
    def test_solo_row_when_single_payload_exceeds_cap(self):
        big = _row_batch(b"x" * 4096)
        # Cap below the row size — the row comes out alone (the
        # rechunker never splits a single row across batches).
        out = list(rechunk_arrow_batches_by_byte_size(iter([big]), byte_size=512))
        assert len(out) == 1
        assert out[0].num_rows == 1


class TestRechunkPreservesRows:
    def test_total_rows_match_input(self):
        rows = [_row_batch(b"x" * 100) for _ in range(20)]
        out = list(rechunk_arrow_batches_by_byte_size(iter(rows), byte_size=512))
        assert sum(b.num_rows for b in out) == 20

    def test_payloads_round_trip(self):
        payloads = [b"alpha", b"beta", b"gamma", b"delta"]
        rows = [_row_batch(p) for p in payloads]
        out = list(rechunk_arrow_batches_by_byte_size(iter(rows), byte_size=10_000))
        assert _rows(out) == payloads

    def test_payload_order_preserved_with_oversized_row(self):
        small = _row_batch(b"alpha")
        big = _row_batch(b"x" * 4096)
        trailing = _row_batch(b"omega")
        out = list(
            rechunk_arrow_batches_by_byte_size(
                iter([small, small, big, trailing]),
                byte_size=512,
            )
        )
        assert _rows(out) == [b"alpha", b"alpha", b"x" * 4096, b"omega"]
