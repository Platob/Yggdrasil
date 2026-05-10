"""Lazy, pushable execution plans for :class:`Tabular`.

An :class:`ExecutionPlan` is an ordered, immutable sequence of
:class:`PlanOp` nodes. Each node is a tiny dataclass that describes
*what* should happen (filter on a predicate, project columns, group +
aggregate, run an arbitrary callable on the LazyFrame); compilation to
a backend happens through :meth:`PlanOp.apply_polars` (today) and is
the natural place to add pyarrow / SQL emitters later.

Why this layer
--------------

:class:`yggdrasil.io.tabular.lazy.LazyTabular` used to keep its op
list as anonymous tuples — that was fine for one method per shape but
made it hard to ask plan-shaped questions ("is this op
schema-preserving?", "does this op commute with a vertical union?")
without scattering `if op[0] == ...` ladders across the codebase.
:class:`PlanOp` puts those questions on the node itself, and the
plan-level helpers (:meth:`ExecutionPlan.append`,
:meth:`ExecutionPlan.split_pushdownable`,
:meth:`ExecutionPlan.is_schema_preserving`) read off them.

The full :mod:`.sql` plan-node tree (Scan / Project / Aggregate / …)
is the heavier IR used when parsing SQL — it carries enough metadata
for a real planner to rewrite. :class:`ExecutionPlan` is the lighter
companion the builder API threads.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Tuple

from yggdrasil.data.data_field import Field
from yggdrasil.io.tabular.execution.expr import Expression, Predicate

if TYPE_CHECKING:
    import polars as pl


__all__ = [
    "PlanOp",
    "Filter",
    "Select",
    "GroupByAgg",
    "Apply",
    "Join",
    "ExecutionPlan",
]


# ---------------------------------------------------------------------------
# Compilation helpers — keep "lift to polars" out of every node body
# ---------------------------------------------------------------------------


def _to_polars(value: Any) -> Any:
    """Compile a stored selector / predicate to a polars-friendly value.

    Yggdrasil :class:`Expression` nodes go through ``to_polars``.
    :class:`yggdrasil.data.data_field.Field` — the canonical
    selector — is translated to ``pl.col(source).cast(target)
    .alias(out_name)``, where the source-side label is
    :attr:`Field.alias` (when distinct from :attr:`Field.name`)
    and the cast target is :attr:`Field.dtype`. Everything else
    (column-name strings, polars / pyarrow native expressions)
    passes through untouched — round-tripping a polars ``Expr``
    through the AST can lose dtype information, so we only lift
    the canonical-AST / canonical-Field inputs.
    """
    if isinstance(value, Expression):
        return value.to_polars()
    if isinstance(value, Field):
        import polars as pl

        source = value.alias if value.has_alias else value.name
        expr = pl.col(source)
        if value.dtype is not None and hasattr(value.dtype, "to_polars"):
            try:
                expr = expr.cast(value.dtype.to_polars())
            except Exception:
                pass
        if value.name and value.name != source:
            expr = expr.alias(value.name)
        return expr
    return value


# ---------------------------------------------------------------------------
# Plan ops
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class PlanOp:
    """Abstract base — one node in an :class:`ExecutionPlan`.

    Concrete nodes override :meth:`apply_polars` (the only required
    backend hook) and the two introspection flags
    :meth:`is_schema_preserving` and
    :meth:`commutes_with_vertical_union`. Defaults are conservative:
    new opaque ops register as schema-changing and non-commutative,
    which is the right call for ``apply``.
    """

    def apply_polars(self, lf: "pl.LazyFrame") -> "pl.LazyFrame":
        raise NotImplementedError(
            f"{type(self).__name__}.apply_polars must be overridden."
        )

    def is_schema_preserving(self) -> bool:
        """Whether running this op leaves the column set unchanged.

        Schema-preserving ops let :meth:`LazyTabular._collect_schema`
        skip the polars round-trip and answer from the source's own
        Arrow schema.
        """
        return False

    def commutes_with_vertical_union(self) -> bool:
        """Whether this op can be pushed *into* each branch of a
        vertical concat without changing the result.

        ``select`` / ``filter`` commute (they read the same way
        before or after the concat); ``group_by`` doesn't (it needs
        all rows). :class:`UnionTabular` uses this to decide which
        ops form the "broadcast prefix" and which run post-union.
        """
        return False


@dataclasses.dataclass(frozen=True, slots=True)
class Filter(PlanOp):
    """``WHERE`` clause — one or more predicates AND-combined.

    Stored items are either yggdrasil :class:`Predicate` nodes
    (canonical, multi-backend) or backend-native expressions kept as-is
    to avoid lossy round-tripping. :meth:`apply_polars` compiles them
    just before they hit the LazyFrame.
    """

    predicates: Tuple[Any, ...]

    def apply_polars(self, lf: "pl.LazyFrame") -> "pl.LazyFrame":
        return lf.filter(*(_to_polars(p) for p in self.predicates))

    def is_schema_preserving(self) -> bool:
        return True

    def commutes_with_vertical_union(self) -> bool:
        return True

    def extend(self, other: "Filter") -> "Filter":
        """Fuse another :class:`Filter`'s predicates onto this one.

        Adjacent yggdrasil predicates AND-merge into a single canonical
        :class:`Logical` node via :meth:`Predicate.merge_with` so the
        stored form is the merged tree. Native predicates stay
        separate; polars' planner ANDs them anyway.
        """
        merged: list[Any] = list(self.predicates)
        for p in other.predicates:
            if (
                merged
                and isinstance(merged[-1], Predicate)
                and isinstance(p, Predicate)
            ):
                merged[-1] = merged[-1].merge_with(p)
            else:
                merged.append(p)
        return Filter(tuple(merged))


@dataclasses.dataclass(frozen=True, slots=True)
class Select(PlanOp):
    """``SELECT`` clause — column projection."""

    columns: Tuple[Any, ...]

    def apply_polars(self, lf: "pl.LazyFrame") -> "pl.LazyFrame":
        return lf.select(*(_to_polars(c) for c in self.columns))

    def commutes_with_vertical_union(self) -> bool:
        return True


@dataclasses.dataclass(frozen=True, slots=True)
class GroupByAgg(PlanOp):
    """``GROUP BY ... AGG ...`` — keys + aggregation expressions.

    Aggregations stay backend-native (the AST doesn't model
    aggregations yet). Empty ``aggs`` is the polars "distinct keys"
    behavior.
    """

    keys: Tuple[Any, ...]
    aggs: Tuple[Any, ...]

    def apply_polars(self, lf: "pl.LazyFrame") -> "pl.LazyFrame":
        gb = lf.group_by(*(_to_polars(k) for k in self.keys))
        return gb.agg(*self.aggs) if self.aggs else gb.agg()


@dataclasses.dataclass(frozen=True, slots=True)
class Apply(PlanOp):
    """Escape hatch — arbitrary ``LazyFrame -> LazyFrame`` callable.

    Use only when the dedicated nodes don't fit (custom ``join``,
    ``with_columns``, ``sort``, …). Schema-changing and
    non-commutative by default since the callable is opaque.
    """

    fn: Callable[["pl.LazyFrame"], "pl.LazyFrame"]

    def apply_polars(self, lf: "pl.LazyFrame") -> "pl.LazyFrame":
        return self.fn(lf)


@dataclasses.dataclass(frozen=True, slots=True)
class Join(PlanOp):
    """``JOIN`` clause — right-side source + on-keys + kind.

    *right* may be a :class:`Tabular` (lifted to a polars
    ``LazyFrame`` via :meth:`Tabular.scan_polars_frame`), a polars
    ``DataFrame`` / ``LazyFrame`` (used as-is), or a string name to
    resolve against :data:`yggdrasil.io.tabular.engine.SYSTEM_ENGINE`
    at apply-time. *on* is shared join keys; pass ``left_on`` /
    ``right_on`` for asymmetric keys. Reshapes rows, so neither
    schema-preserving nor union-commutative.
    """

    right: Any
    on: Tuple[Any, ...] = ()
    how: str = "inner"
    left_on: Tuple[Any, ...] = ()
    right_on: Tuple[Any, ...] = ()
    suffix: str = "_right"

    def apply_polars(self, lf: "pl.LazyFrame") -> "pl.LazyFrame":
        right_lf = self._resolve_right_polars()
        kwargs: dict[str, Any] = {"how": self.how, "suffix": self.suffix}
        if self.left_on or self.right_on:
            left_keys = self.left_on or self.on
            right_keys = self.right_on or self.on
            if not left_keys or not right_keys:
                raise ValueError(
                    "Join requires either ``on`` or both ``left_on`` and "
                    f"``right_on``; got on={self.on!r}, "
                    f"left_on={self.left_on!r}, right_on={self.right_on!r}."
                )
            kwargs["left_on"] = [_to_polars(c) for c in left_keys]
            kwargs["right_on"] = [_to_polars(c) for c in right_keys]
        elif self.how != "cross":
            if not self.on:
                raise ValueError(
                    "Join requires ``on`` (or ``left_on`` / ``right_on``) "
                    f"unless how='cross'; got how={self.how!r}."
                )
            kwargs["on"] = [_to_polars(c) for c in self.on]
        return lf.join(right_lf, **kwargs)

    def _resolve_right_polars(self) -> "pl.LazyFrame":
        from yggdrasil.io.tabular.base import Tabular

        right = self.right
        if isinstance(right, str):
            from yggdrasil.io.tabular.engine import SYSTEM_ENGINE
            right = SYSTEM_ENGINE.resolve(right)
        if isinstance(right, Tabular):
            return right.scan_polars_frame()
        # polars DataFrame / LazyFrame — both work as the join right;
        # ``DataFrame.lazy()`` is cheap, ``LazyFrame`` passes through.
        lazy = getattr(right, "lazy", None)
        if callable(lazy):
            try:
                return lazy()
            except TypeError:
                pass
        return right


# ---------------------------------------------------------------------------
# ExecutionPlan
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Immutable, ordered sequence of :class:`PlanOp`.

    Construction normalizes the input to a tuple. :meth:`append` is
    the canonical "add one op and return a new plan" entry point —
    it also fuses adjacent :class:`Filter` nodes via
    :meth:`Filter.extend` so two stacked ``where`` calls land in a
    single op. Empty plans are valid and read as zero ops.
    """

    ops: Tuple[PlanOp, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.ops, tuple):
            object.__setattr__(self, "ops", tuple(self.ops))

    # ------------------------------------------------------------------
    # Construction / mutation (always returns a new plan)
    # ------------------------------------------------------------------

    @classmethod
    def empty(cls) -> "ExecutionPlan":
        return cls(())

    def append(self, op: PlanOp) -> "ExecutionPlan":
        if (
            isinstance(op, Filter)
            and self.ops
            and isinstance(self.ops[-1], Filter)
        ):
            return ExecutionPlan((*self.ops[:-1], self.ops[-1].extend(op)))
        return ExecutionPlan((*self.ops, op))

    def extend(self, others: Iterable[PlanOp]) -> "ExecutionPlan":
        out = self
        for op in others:
            out = out.append(op)
        return out

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[PlanOp]:
        return iter(self.ops)

    def __len__(self) -> int:
        return len(self.ops)

    def __bool__(self) -> bool:
        return bool(self.ops)

    def is_empty(self) -> bool:
        return not self.ops

    def is_schema_preserving(self) -> bool:
        """``True`` iff every op leaves the column set unchanged."""
        return all(op.is_schema_preserving() for op in self.ops)

    def split_pushdownable(self) -> "tuple[ExecutionPlan, ExecutionPlan]":
        """Slice into ``(prefix, tail)`` at the first non-commutative op.

        The prefix is safe to broadcast into each branch of a vertical
        union; the tail must run on the unioned frame. The split stops
        at the *first* non-commutative op even if commutative ops
        appear later — order matters once aggregation reshapes rows.
        """
        for i, op in enumerate(self.ops):
            if not op.commutes_with_vertical_union():
                return ExecutionPlan(self.ops[:i]), ExecutionPlan(self.ops[i:])
        return self, ExecutionPlan.empty()

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def apply_polars(self, lf: "pl.LazyFrame") -> "pl.LazyFrame":
        """Replay every op against *lf* in order."""
        for op in self.ops:
            lf = op.apply_polars(lf)
        return lf
