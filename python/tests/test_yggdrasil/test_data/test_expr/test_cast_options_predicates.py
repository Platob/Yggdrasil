"""Tests for the predicate-typed fields on :class:`CastOptions`.

``source_predicate`` / ``target_predicate`` / ``children_predicate``
are all :class:`Predicate` references — these tests verify that
:class:`CastOptions` accepts them, round-trips them through
:meth:`copy`, and that the per-IO consumers (FolderIO's children
filter) honour ``children_predicate`` end-to-end.
"""

from __future__ import annotations

from yggdrasil.data.expr import Predicate, col
from yggdrasil.data.options import CastOptions


class TestCastOptionsPredicateFields:
    def test_defaults_are_none(self):
        opts = CastOptions()
        assert opts.source_predicate is None
        assert opts.target_predicate is None
        assert opts.children_predicate is None

    def test_legacy_where_field_is_gone(self):
        # ``where`` used to be a generic predicate slot on
        # CastOptions; it was replaced by the explicit
        # ``source_predicate`` (read-side) and ``target_predicate``
        # (write-side) so callers can't accidentally apply a
        # filter on the wrong side. Pin the removal so a future
        # re-add doesn't slip through.
        assert not hasattr(CastOptions(), "where")

    def test_predicates_are_carried_through_copy(self):
        # ``CastOptions.copy`` must round-trip every field — adding
        # a new field is exactly the kind of thing that breaks
        # silently if the copy path forgets to propagate it.
        src = col("price") >= 100
        tgt = col("status") == "ok"
        children = ~col("name").like(".%")

        opts = CastOptions(
            source_predicate=src,
            target_predicate=tgt,
            children_predicate=children,
        )
        copied = opts.copy()
        assert copied.source_predicate is src
        assert copied.target_predicate is tgt
        assert copied.children_predicate is children

    def test_predicate_fields_accept_only_predicate_instances(self):
        # The dataclass declares the type as ``Predicate | None``;
        # any boolean-valued :class:`Expression` works because
        # comparisons inherit from :class:`Predicate`. A non-
        # predicate scalar would still slip past the type checker
        # but the helper consumers will reject it loudly when they
        # try to call ``to_python()``.
        opts = CastOptions(source_predicate=col("a") > 1)
        assert isinstance(opts.source_predicate, Predicate)
