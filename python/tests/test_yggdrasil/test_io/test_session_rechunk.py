"""Unit tests for the Spark response-batch rechunker.

The rechunker is the inner helper that the ``_spark_fetch_misses``
worker uses to keep emitted batches under a byte cap. Driving it
directly with synthetic record batches is much cheaper than spinning
Spark for an end-to-end pass and lets us assert the boundary cases
(grouping vs. solo emission) precisely.
"""

from __future__ import annotations

import pyarrow as pa

from yggdrasil.io.session import _rechunk_to_byte_limit


def _row_batch(payload: bytes) -> pa.RecordBatch:
    """One-row record batch carrying *payload* in a binary column."""
    return pa.RecordBatch.from_pydict({"body": [payload]})


class TestRechunkBelowLimit:
    def test_groups_small_rows_into_a_single_batch(self):
        # Five tiny rows, well under any reasonable cap.
        rows = [_row_batch(b"x" * 16) for _ in range(5)]
        out = list(_rechunk_to_byte_limit(iter(rows), byte_limit=10_000))
        assert len(out) == 1
        assert out[0].num_rows == 5

    def test_empty_input_yields_nothing(self):
        assert list(_rechunk_to_byte_limit(iter([]), byte_limit=10_000)) == []


class TestRechunkAtLimit:
    def test_splits_when_running_total_would_exceed(self):
        # Each row is ~256 B; cap at 600 B forces a split after row 2.
        rows = [_row_batch(b"x" * 256) for _ in range(5)]
        out = list(_rechunk_to_byte_limit(iter(rows), byte_limit=600))
        # We don't assert exact counts per batch (depends on Arrow's
        # serialization overhead) — only that the cap is honored and
        # every input row makes it through.
        assert sum(b.num_rows for b in out) == 5
        for b in out:
            if b.num_rows > 1:
                assert b.nbytes <= 600


class TestRechunkOversizedRow:
    def test_solo_row_when_single_payload_exceeds_cap(self):
        big = _row_batch(b"x" * 4096)
        # Cap below the row size — the row must come out alone.
        out = list(_rechunk_to_byte_limit(iter([big]), byte_limit=512))
        assert len(out) == 1
        assert out[0].num_rows == 1

    def test_oversized_row_flushes_pending_first(self):
        small = _row_batch(b"x" * 16)
        big = _row_batch(b"x" * 4096)
        smaller_after = _row_batch(b"x" * 16)

        out = list(
            _rechunk_to_byte_limit(
                iter([small, small, big, smaller_after]),
                byte_limit=512,
            )
        )

        # Three batches in order: the two small rows, the oversized
        # row alone, then the trailing small row.
        assert [b.num_rows for b in out] == [2, 1, 1]


class TestRechunkPreservesRows:
    def test_total_rows_match_input(self):
        rows = [_row_batch(b"x" * 100) for _ in range(20)]
        out = list(_rechunk_to_byte_limit(iter(rows), byte_limit=512))
        assert sum(b.num_rows for b in out) == 20

    def test_payloads_round_trip(self):
        payloads = [b"alpha", b"beta", b"gamma", b"delta"]
        rows = [_row_batch(p) for p in payloads]
        out = list(_rechunk_to_byte_limit(iter(rows), byte_limit=10_000))
        flat = pa.Table.from_batches(out).column("body").to_pylist()
        assert flat == payloads
