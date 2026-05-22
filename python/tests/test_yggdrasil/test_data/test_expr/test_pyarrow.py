"""Tests for the pyarrow backend.

Round-trips the AST through :class:`pyarrow.compute.Expression`
and verifies the result against actual table filtering — that's
the user-visible behaviour we care about.
"""

from __future__ import annotations

import pyarrow as pa

from yggdrasil.io.tabular.execution.expr import col


def _table():
    return pa.Table.from_pylist([
        {"price": 50, "side": "buy"},
        {"price": 200, "side": "sell"},
        {"price": 150, "side": "hold"},
        {"price": None, "side": "buy"},
    ])


class TestArrowFiltering:
    def test_basic_comparison_filters_correctly(self):
        t = _table()
        out = t.filter((col("price") >= 100).to_pyarrow())
        assert out.column("price").to_pylist() == [200, 150]

    def test_and_combination_works(self):
        t = _table()
        p = (col("price") >= 100) & col("side").is_in(["buy", "sell"])
        out = t.filter(p.to_pyarrow())
        assert out.to_pylist() == [{"price": 200, "side": "sell"}]

    def test_is_null_filters_to_null_rows_only(self):
        t = _table()
        out = t.filter(col("price").is_null().to_pyarrow())
        assert out.column("side").to_pylist() == ["buy"]

    def test_between_inclusive(self):
        t = _table()
        out = t.filter(col("price").between(100, 200).to_pyarrow())
        # NULL rows excluded by default arrow semantics.
        assert sorted(out.column("price").to_pylist()) == [150, 200]

    def test_negated_in_excludes_listed_values(self):
        t = _table()
        out = t.filter(col("side").not_in(["hold"]).to_pyarrow())
        # ``hold`` is dropped; the buy / sell rows survive, plus
        # the NULL-side row (Arrow's NOT IN keeps NULL rows by
        # default; SQL would drop them — choose the surface based
        # on the engine semantics).
        sides = sorted(out.column("side").to_pylist(), key=str)
        assert "hold" not in sides
        assert "buy" in sides and "sell" in sides


class TestFilterMethods:
    """Direct ``Predicate.filter_arrow_*`` helpers (no Python row loop)."""

    def test_filter_arrow_table_keeps_matching_rows(self):
        t = _table()
        out = (col("price") >= 100).filter_arrow_table(t)
        assert out.column("price").to_pylist() == [200, 150]

    def test_filter_arrow_table_empty_input_passthrough(self):
        empty = _table().slice(0, 0)
        out = (col("price") > 0).filter_arrow_table(empty)
        assert out.num_rows == 0
        assert out.schema == empty.schema

    def test_filter_arrow_batch_returns_record_batch(self):
        batch = _table().to_batches()[0]
        kept = (col("side") == "buy").filter_arrow_batch(batch)
        assert isinstance(kept, pa.RecordBatch)
        assert kept.column("side").to_pylist() == ["buy", "buy"]

    def test_filter_arrow_batch_preserves_schema_when_empty(self):
        batch = _table().to_batches()[0]
        kept = (col("side") == "nope").filter_arrow_batch(batch)
        assert kept.num_rows == 0
        assert kept.schema == batch.schema

    def test_filter_arrow_batches_streams_survivors(self):
        batches = _table().to_batches()
        out = list(
            (col("price") >= 100).filter_arrow_batches(batches)
        )
        merged = pa.Table.from_batches(out) if out else _table().slice(0, 0)
        assert sorted(merged.column("price").to_pylist()) == [150, 200]

    def test_filter_arrow_batches_skips_zero_row_batches(self):
        # Mix an empty batch in and confirm it doesn't appear in output.
        full = _table().to_batches()[0]
        empty = full.slice(0, 0)
        out = list(
            (col("price") < 1000).filter_arrow_batches([empty, full, empty])
        )
        assert all(b.num_rows > 0 for b in out)
        # All non-null prices survive ``< 1000``.
        rows = pa.Table.from_batches(out).column("price").to_pylist()
        assert sorted(r for r in rows if r is not None) == [50, 150, 200]


class TestNullInListExpansion:
    def test_includes_null_matches_null_rows(self):
        # ``IN (1, 2, None)`` — explicit NULL in the value list
        # should make NULL rows match. Arrow expands to
        # ``isin OR is_null``.
        t = pa.Table.from_pylist([{"x": 1}, {"x": 2}, {"x": None}, {"x": 5}])
        p = col("x").is_in([1, None])
        out = t.filter(p.to_pyarrow())
        assert sorted(
            v for v in out.column("x").to_pylist() if v is not None
        ) == [1]
        assert None in out.column("x").to_pylist()


class TestFilterArrowBatchViaHashset:
    """``Predicate.filter_arrow_batch_via_hashset`` — static-value fast path.

    Decomposes ``InList`` and ``AND(InList, InList, ...)`` predicates
    into Python :class:`frozenset` probes against materialised
    columns. On the typical cache-lookup shape (1-row leaf batches,
    multi-column IN-lists) it's ~30x faster than
    :meth:`pa.RecordBatch.filter` — the per-call compile + scan
    overhead dominates pyarrow's path on tiny inputs. Falls back to
    :meth:`pa.RecordBatch.filter` for anything else by returning
    ``None`` to the caller.
    """

    def test_single_in_list_keeps_matching_rows(self):
        t = pa.Table.from_pylist([{"x": 1}, {"x": 2}, {"x": 3}])
        batch = t.to_batches()[0]
        kept = col("x").is_in([1, 3]).filter_arrow_batch_via_hashset(batch)
        assert kept is not None
        assert kept.column("x").to_pylist() == [1, 3]

    def test_and_of_in_lists_runs_via_hashset(self):
        from yggdrasil.io.tabular.execution.expr import simplify

        t = pa.Table.from_pylist([
            {"x": 1, "y": "a"},
            {"x": 2, "y": "b"},
            {"x": 3, "y": "a"},
            {"x": 4, "y": "c"},
        ])
        batch = t.to_batches()[0]
        pred = simplify(col("x").is_in([1, 3]) & col("y").is_in(["a"]))
        kept = pred.filter_arrow_batch_via_hashset(batch)
        assert kept is not None
        assert kept.column("x").to_pylist() == [1, 3]

    def test_returns_input_when_every_row_matches(self):
        # All-pass case: the helper hands back the original batch
        # (no take call) so a downstream caller sees ``kept is batch``.
        t = pa.Table.from_pylist([{"x": 1}, {"x": 2}])
        batch = t.to_batches()[0]
        kept = col("x").is_in([1, 2, 3]).filter_arrow_batch_via_hashset(batch)
        assert kept is batch

    def test_returns_zero_row_slice_when_nothing_matches(self):
        t = pa.Table.from_pylist([{"x": 1}, {"x": 2}])
        batch = t.to_batches()[0]
        kept = col("x").is_in([99]).filter_arrow_batch_via_hashset(batch)
        assert kept is not None and kept.num_rows == 0
        # Schema preserved.
        assert kept.schema.equals(batch.schema)

    def test_includes_null_matches_null_rows(self):
        from yggdrasil.io.tabular.execution.expr import simplify

        t = pa.Table.from_pylist([{"x": 1}, {"x": None}, {"x": 5}])
        batch = t.to_batches()[0]
        # ``simplify`` folds ``IN [1] | IS NULL`` into ``InList(includes_null=True)``.
        pred = simplify(col("x").is_in([1]) | col("x").is_null())
        kept = pred.filter_arrow_batch_via_hashset(batch)
        assert kept is not None
        out = kept.column("x").to_pylist()
        assert sorted(v for v in out if v is not None) == [1]
        assert None in out

    def test_excludes_null_when_not_requested(self):
        # ``IN (1, 2)`` with no NULL flag — null rows must be dropped,
        # matching ``pa.RecordBatch.filter(pa.field('x').isin([1, 2]))``.
        t = pa.Table.from_pylist([{"x": 1}, {"x": None}, {"x": 5}])
        batch = t.to_batches()[0]
        kept = col("x").is_in([1, 2]).filter_arrow_batch_via_hashset(batch)
        assert kept is not None
        assert kept.column("x").to_pylist() == [1]

    def test_non_inlist_predicate_returns_none(self):
        # ``col(x) > 100`` is a Comparison, not InList. The fast path
        # bows out; caller should use ``pa.RecordBatch.filter``.
        t = pa.Table.from_pylist([{"x": 1}])
        batch = t.to_batches()[0]
        kept = (col("x") > 100).filter_arrow_batch_via_hashset(batch)
        assert kept is None

    def test_or_returns_none(self):
        # OR of InLists isn't decomposable into a hashset-per-column
        # (the row needs to match *one* of them, not all). Falls back.
        t = pa.Table.from_pylist([{"x": 1, "y": "a"}])
        batch = t.to_batches()[0]
        pred = col("x").is_in([1]) | col("y").is_in(["b"])
        kept = pred.filter_arrow_batch_via_hashset(batch)
        assert kept is None

    def test_unhashable_values_return_none(self):
        # ``IN ([1, 2], [3, 4])`` — list literals are unhashable, so
        # :class:`frozenset` raises. The helper returns ``None`` and
        # the caller falls back to the pyarrow path.
        t = pa.Table.from_pylist([{"x": [1, 2]}])
        batch = t.to_batches()[0]
        kept = col("x").is_in([[1, 2], [3, 4]]).filter_arrow_batch_via_hashset(batch)
        assert kept is None

    def test_matches_pyarrow_filter_on_mixed_input(self):
        """The hashset path and ``pa.RecordBatch.filter`` agree row-for-row."""
        from yggdrasil.io.tabular.execution.expr import simplify

        t = pa.Table.from_pylist([
            {"x": 1, "y": "a"},
            {"x": 2, "y": "b"},
            {"x": None, "y": "a"},
            {"x": 3, "y": None},
            {"x": 4, "y": "a"},
        ])
        batch = t.to_batches()[0]
        pred = simplify(col("x").is_in([1, 3, 4]) & col("y").is_in(["a"]))
        hs_out = pred.filter_arrow_batch_via_hashset(batch)
        pa_out = batch.filter(pred.to_arrow())
        assert hs_out is not None
        assert hs_out.to_pylist() == pa_out.to_pylist()
