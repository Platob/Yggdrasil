"""Tests for the predicate-typed fields on :class:`CastOptions`.

``predicate`` is the unified row-level filter; ``children_predicate``
is the orthogonal child-discovery filter. These tests verify that
:class:`CastOptions` accepts them, round-trips them through
:meth:`copy`, and that the per-IO consumers honour the contract.
"""

from __future__ import annotations

from yggdrasil.execution.expr import Predicate, col
from yggdrasil.data.options import CastOptions


class TestCastOptionsPredicateFields:
    def test_defaults_are_none(self):
        opts = CastOptions()
        assert opts.predicate is None
        assert opts.children_predicate is None

    def test_legacy_split_predicate_fields_are_gone(self):
        # ``source_predicate`` and ``target_predicate`` used to be
        # separate fields. They've been collapsed into a single
        # ``predicate`` slot — every IO that filters rows reaches
        # for the same field, and backends that pushed the filter
        # down clear it before reaching the per-batch evaluator.
        opts = CastOptions()
        assert not hasattr(opts, "source_predicate")
        assert not hasattr(opts, "target_predicate")

    def test_legacy_where_field_is_gone(self):
        # ``where`` used to be a generic predicate slot on CastOptions;
        # it was replaced by ``predicate``.
        assert not hasattr(CastOptions(), "where")

    def test_predicate_round_trips_through_copy(self):
        # ``CastOptions.copy`` must round-trip every field.
        pred = col("price") >= 100
        children = ~col("name").like(".%")

        opts = CastOptions(
            predicate=pred,
            children_predicate=children,
        )
        copied = opts.copy()
        assert copied.predicate is pred
        assert copied.children_predicate is children

    def test_predicate_field_accepts_predicate_instance(self):
        opts = CastOptions(predicate=col("a") > 1)
        assert isinstance(opts.predicate, Predicate)
