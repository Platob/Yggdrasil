"""Logical / physical execution-plan tree for :mod:`yggdrasil.sql`.

Two layers, one set of node classes:

- The shape is logical — every node says *what* should happen
  (scan a Tabular, filter on a predicate, project columns, group +
  aggregate, order by, limit). No physical implementation choices
  baked in.
- Evaluation is physical — :meth:`PlanNode.execute` materializes the
  node into a :class:`Tabular` (typically :class:`ArrowTabular`) in
  one pass, recursing through children. The base evaluator works in
  Arrow throughout; engine-aware nodes (a future
  :class:`PolarsScan` / :class:`SparkScan`) can override
  :meth:`execute` to stay in their native engine until materialization.

What we deliberately don't do here
----------------------------------

- No cost-based optimization. Predicate pushdown and column pruning
  are surfaced as explicit ``Scan(predicate=, projection=)`` slots
  the planner sets when it can prove they're safe; everything else
  evaluates strictly bottom-up. Adding a real optimizer means
  walking the tree and rewriting nodes — pure data, no plumbing.
- No join planning. We support nested-loop and hash-join inside
  :class:`Join.execute`; the planner picks the strategy based on
  the join shape, but neither node knows about cardinality.

The node tree is the single intermediate representation: the parser
hands one back, the engine runs it, the result handle ships it. Tests
that need to inspect the plan can look at :attr:`Engine.plan` after
:meth:`Engine.prepare`.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.compute as pc

from yggdrasil.data.expr import Expression, Predicate
from yggdrasil.data.options import CastOptions
from yggdrasil.io.tabular import ArrowTabular, Tabular


if TYPE_CHECKING:
    from yggdrasil.sql.dynamic_catalog import DynamicCatalog


__all__ = [
    "PlanNode",
    "Scan",
    "Filter",
    "Project",
    "ProjectionItem",
    "Aggregate",
    "AggregateSpec",
    "Sort",
    "SortKey",
    "Limit",
    "Join",
    "JoinKind",
    "evaluate",
]


# ---------------------------------------------------------------------------
# Projection / aggregate / sort-key value types
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class ProjectionItem:
    """One entry in a :class:`Project` node.

    ``source`` is either a column name (string) — a straight passthrough
    column rename — or an :class:`Expression` evaluated row-wise on the
    input batch. ``alias`` is the output column name.
    """

    source: Any  # str | Expression
    alias: str


@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class AggregateSpec:
    """One entry in an :class:`Aggregate` node's aggregate list.

    ``func`` is a normalized aggregate name (``count``, ``sum``,
    ``avg``, ``min``, ``max``); ``column`` is the input column or
    ``None`` for ``COUNT(*)``. ``alias`` is the output column.
    """

    func: str
    column: Optional[str]
    alias: str
    distinct: bool = False


@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class SortKey:
    """One entry in a :class:`Sort` node's ``order by`` list."""

    column: str
    descending: bool = False
    nulls_first: bool = False


class JoinKind(str):
    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    FULL = "full"
    CROSS = "cross"


# ---------------------------------------------------------------------------
# PlanNode base
# ---------------------------------------------------------------------------


class PlanNode(ABC):
    """Base class for every operator in the execution plan.

    The contract is two methods:

    - :meth:`execute` — return a :class:`Tabular` over this node's
      output. Default impl evaluates :meth:`_execute_arrow` and wraps
      the result in :class:`ArrowTabular`; subclasses with a native
      engine path (Polars / Spark) override :meth:`execute` directly.
    - :meth:`children` — every child PlanNode in left-to-right order,
      so the optimizer / formatter can walk the tree without each
      node hand-rolling its own iteration.

    Nodes are dataclasses so equality / repr / hashing come for free
    — useful for cache keys, tests, and debug dumps.
    """

    #: Friendly name used in :meth:`format` debug dumps.
    op_name: ClassVar[str] = "PlanNode"

    @abstractmethod
    def children(self) -> "Sequence[PlanNode]":
        """Iterate child nodes in left-to-right order."""

    @abstractmethod
    def _execute_arrow(self, ctx: "DynamicCatalog") -> pa.Table:
        """Materialize this node's output as a :class:`pyarrow.Table`.

        Recursion is the subclass's responsibility — most operators
        delegate to ``self._child_table(ctx)`` for their single input
        node and then run the operator-specific Arrow code on top.
        """

    def execute(self, ctx: "DynamicCatalog") -> Tabular:
        """Evaluate the plan and return a :class:`Tabular` result.

        The default builds an :class:`ArrowTabular` over the table
        materialized by :meth:`_execute_arrow`. Override on engine-
        aware operators that can stay in their native engine until
        the final materialize.
        """
        return ArrowTabular(self._execute_arrow(ctx))

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def format(self, indent: int = 0) -> str:
        """Indented multi-line dump of this subtree.

        Used by :meth:`PlanNode.__repr__` so a printed plan reads
        top-down like an EXPLAIN output:

        ::

            Sort by=name
              Filter pred=age > 18
                Scan src=users
        """
        head = " " * indent + f"{self.op_name} {self._format_args()}"
        kids = "\n".join(c.format(indent + 2) for c in self.children())
        return head + (("\n" + kids) if kids else "")

    def _format_args(self) -> str:
        """Per-subclass extra info on the head line. Default: empty."""
        return ""

    def __repr__(self) -> str:
        return self.format()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _child_table(
        self, ctx: "DynamicCatalog", *, index: int = 0,
    ) -> pa.Table:
        """Materialize the *index*-th child as a single :class:`pa.Table`."""
        kids = list(self.children())
        return kids[index]._execute_arrow(ctx)


# ---------------------------------------------------------------------------
# Scan — read a Tabular out of the catalog with optional pushdown
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class Scan(PlanNode):
    """Read rows from a named :class:`Tabular` in the catalog.

    ``predicate`` and ``projection`` are pushdown hints. The evaluator
    threads them into the source's :class:`CastOptions` so a
    Tabular that supports native filtering (Parquet, Delta) can drop
    rows / columns before they reach memory. When the source ignores
    them, :meth:`_execute_arrow` re-applies the filter locally so the
    contract holds either way.
    """

    op_name: ClassVar[str] = "Scan"

    name: str
    alias: Optional[str] = None
    predicate: Optional[Predicate] = None
    projection: Optional[Tuple[str, ...]] = None
    limit: Optional[int] = None

    def children(self) -> "Sequence[PlanNode]":
        return ()

    def _format_args(self) -> str:
        bits: list[str] = [f"src={self.name!r}"]
        if self.alias and self.alias != self.name:
            bits.append(f"alias={self.alias!r}")
        if self.projection:
            bits.append(f"projection={list(self.projection)}")
        if self.predicate is not None:
            bits.append(f"predicate={self.predicate.to_sql()}")
        if self.limit is not None:
            bits.append(f"limit={self.limit}")
        return " ".join(bits)

    def _execute_arrow(self, ctx: "DynamicCatalog") -> pa.Table:
        source = ctx.resolve(self.name)
        options = CastOptions(predicate=self.predicate)
        table = source.read_arrow_table(options=options)
        # Pushdown is a hint — the source may or may not honour it.
        # Re-apply both filter and projection so the output of Scan
        # is what the planner asked for, regardless of the source.
        if self.predicate is not None:
            mask = pc.cast(_eval_predicate_arrow(self.predicate, table), pa.bool_())
            table = table.filter(mask)
        if self.projection:
            keep = [c for c in self.projection if c in table.column_names]
            table = table.select(keep)
        if self.limit is not None and table.num_rows > self.limit:
            table = table.slice(0, self.limit)
        return table


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class Filter(PlanNode):
    """Apply a row-level predicate to the child node's output."""

    op_name: ClassVar[str] = "Filter"

    child: PlanNode
    predicate: Predicate

    def children(self) -> "Sequence[PlanNode]":
        return (self.child,)

    def _format_args(self) -> str:
        return f"pred={self.predicate.to_sql()}"

    def _execute_arrow(self, ctx: "DynamicCatalog") -> pa.Table:
        table = self._child_table(ctx)
        mask = pc.cast(_eval_predicate_arrow(self.predicate, table), pa.bool_())
        return table.filter(mask)


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class Project(PlanNode):
    """Pick / rename / compute output columns from the child."""

    op_name: ClassVar[str] = "Project"

    child: PlanNode
    items: Tuple[ProjectionItem, ...]

    def children(self) -> "Sequence[PlanNode]":
        return (self.child,)

    def _format_args(self) -> str:
        bits = []
        for it in self.items:
            if isinstance(it.source, str) and it.source == it.alias:
                bits.append(it.alias)
            elif isinstance(it.source, str):
                bits.append(f"{it.source} AS {it.alias}")
            else:
                bits.append(f"({it.source.to_sql()}) AS {it.alias}")
        return "items=[" + ", ".join(bits) + "]"

    def _execute_arrow(self, ctx: "DynamicCatalog") -> pa.Table:
        table = self._child_table(ctx)
        out_arrays: list[pa.ChunkedArray] = []
        out_names: list[str] = []
        for item in self.items:
            if isinstance(item.source, str):
                if item.source not in table.column_names:
                    raise KeyError(
                        f"Project: column {item.source!r} not found in input. "
                        f"Available: {table.column_names!r}."
                    )
                out_arrays.append(table.column(item.source))
            else:
                # Expression — evaluate row-wise via the pyarrow.compute
                # backend, then materialize as a single chunked column.
                arr = _eval_expression_arrow(item.source, table)
                out_arrays.append(_as_chunked(arr))
            out_names.append(item.alias)
        return pa.Table.from_arrays(out_arrays, names=out_names)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class Aggregate(PlanNode):
    """``GROUP BY group_keys`` + per-group aggregates.

    ``group_keys`` may be empty: that's the global-aggregate case
    (``SELECT COUNT(*) FROM t`` produces one row).
    """

    op_name: ClassVar[str] = "Aggregate"

    child: PlanNode
    group_keys: Tuple[str, ...]
    aggregates: Tuple[AggregateSpec, ...]

    def children(self) -> "Sequence[PlanNode]":
        return (self.child,)

    def _format_args(self) -> str:
        bits = []
        if self.group_keys:
            bits.append(f"keys={list(self.group_keys)}")
        if self.aggregates:
            agg = []
            for a in self.aggregates:
                col = a.column or "*"
                d = "DISTINCT " if a.distinct else ""
                agg.append(f"{a.func.upper()}({d}{col}) AS {a.alias}")
            bits.append("aggs=[" + ", ".join(agg) + "]")
        return " ".join(bits)

    def _execute_arrow(self, ctx: "DynamicCatalog") -> pa.Table:
        table = self._child_table(ctx)

        # Build the (function, options, target) triples pyarrow's
        # ``group_by(...).aggregate`` expects. ``count(*)`` / ``count_all``
        # don't take a column; we pass the aggregate count function on
        # the first key (or any column) since pyarrow's signature is
        # ``(column, function, options)``.
        agg_inputs: list[Any] = []
        for spec in self.aggregates:
            fn = _AGG_NAME_MAP.get(spec.func.lower())
            if fn is None:
                raise NotImplementedError(
                    f"Aggregate function {spec.func!r} is not supported. "
                    f"Supported: {sorted(_AGG_NAME_MAP)}."
                )
            if fn == "count" and (spec.column is None or spec.column == "*"):
                # Pyarrow's count_all has no column; we synthesize a
                # constant column to count.
                if "_ygg_count_all" not in table.column_names:
                    table = table.append_column(
                        "_ygg_count_all",
                        pa.array([1] * table.num_rows, type=pa.int64()),
                    )
                agg_inputs.append(("_ygg_count_all", "count", None))
                continue
            if spec.column is None:
                raise ValueError(
                    f"Aggregate {spec.func!r} requires a column."
                )
            agg_inputs.append((spec.column, fn, None))

        if not self.group_keys:
            # Global aggregate — one row per aggregate.
            results: dict[str, Any] = {}
            for spec, (col, fn, _opts) in zip(self.aggregates, agg_inputs):
                results[spec.alias] = [_global_agg(fn, table.column(col))]
            return pa.table(results)

        # Group-wise.
        keyed = table.group_by(list(self.group_keys)).aggregate(agg_inputs)
        # Pyarrow names the aggregate columns ``<col>_<func>``; rebuild
        # with our caller-supplied aliases. The grouped-by keys keep
        # their original names.
        renamed: list[str] = list(self.group_keys)
        for spec, (col, fn, _opts) in zip(self.aggregates, agg_inputs):
            # Pyarrow appends a hash suffix for hash_count etc.; use
            # the column position instead of a brittle name match.
            renamed.append(spec.alias)
        # The aggregate columns come after the keys, in spec order.
        return keyed.rename_columns(renamed)


_AGG_NAME_MAP = {
    "count": "count",
    "sum": "sum",
    "min": "min",
    "max": "max",
    "avg": "mean",
    "mean": "mean",
}


def _global_agg(fn: str, column: Any) -> Any:
    """One-shot scalar aggregation for the no-group-by case."""
    if fn == "count":
        return pc.count(column).as_py()
    if fn == "sum":
        return pc.sum(column).as_py()
    if fn == "min":
        return pc.min(column).as_py()
    if fn == "max":
        return pc.max(column).as_py()
    if fn == "mean":
        return pc.mean(column).as_py()
    raise NotImplementedError(f"Global aggregate {fn!r} not supported.")


# ---------------------------------------------------------------------------
# Sort + Limit
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class Sort(PlanNode):
    """``ORDER BY``. Stable sort across all keys at once."""

    op_name: ClassVar[str] = "Sort"

    child: PlanNode
    keys: Tuple[SortKey, ...]

    def children(self) -> "Sequence[PlanNode]":
        return (self.child,)

    def _format_args(self) -> str:
        bits = [f"{k.column}{' DESC' if k.descending else ''}" for k in self.keys]
        return "by=[" + ", ".join(bits) + "]"

    def _execute_arrow(self, ctx: "DynamicCatalog") -> pa.Table:
        table = self._child_table(ctx)
        if table.num_rows == 0:
            return table
        keys = [
            (k.column, "descending" if k.descending else "ascending")
            for k in self.keys
        ]
        indices = pc.sort_indices(table, sort_keys=keys)
        return table.take(indices)


@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class Limit(PlanNode):
    """``LIMIT n [OFFSET m]`` slice."""

    op_name: ClassVar[str] = "Limit"

    child: PlanNode
    n: int
    offset: int = 0

    def children(self) -> "Sequence[PlanNode]":
        return (self.child,)

    def _format_args(self) -> str:
        return f"n={self.n} offset={self.offset}"

    def _execute_arrow(self, ctx: "DynamicCatalog") -> pa.Table:
        table = self._child_table(ctx)
        return table.slice(self.offset, self.n)


# ---------------------------------------------------------------------------
# Join
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class Join(PlanNode):
    """Two-input join.

    ``kind`` is one of :class:`JoinKind`'s constants. ``on`` is the
    list of equality keys ``(left_col, right_col)``; pyarrow's
    :meth:`pa.Table.join` does the hash join. Non-equi joins are
    not currently supported — fall back to an explicit ``Filter``
    on a ``Cross`` join when needed.
    """

    op_name: ClassVar[str] = "Join"

    left: PlanNode
    right: PlanNode
    kind: str = JoinKind.INNER
    on: Tuple[Tuple[str, str], ...] = ()
    left_alias: Optional[str] = None
    right_alias: Optional[str] = None

    def children(self) -> "Sequence[PlanNode]":
        return (self.left, self.right)

    def _format_args(self) -> str:
        on_bits = ", ".join(f"{l}={r}" for l, r in self.on)
        return f"kind={self.kind} on=[{on_bits}]"

    def _execute_arrow(self, ctx: "DynamicCatalog") -> pa.Table:
        left = self._child_table(ctx, index=0)
        right = self._child_table(ctx, index=1)

        if self.kind == JoinKind.CROSS:
            # Pyarrow has no built-in cross join — synthesize one via a
            # constant join key.
            if "_ygg_cross" not in left.column_names:
                left = left.append_column(
                    "_ygg_cross", pa.array([1] * left.num_rows, type=pa.int64())
                )
            if "_ygg_cross" not in right.column_names:
                right = right.append_column(
                    "_ygg_cross", pa.array([1] * right.num_rows, type=pa.int64())
                )
            joined = left.join(
                right, keys=["_ygg_cross"], join_type="inner",
            )
            if "_ygg_cross" in joined.column_names:
                joined = joined.drop_columns(["_ygg_cross"])
            return joined

        if not self.on:
            raise ValueError(
                f"Join kind={self.kind!r} requires equality keys; got empty on=[]. "
                "Cross-product joins should pass kind='cross' instead."
            )
        keys = [k[0] for k in self.on]
        right_keys = [k[1] for k in self.on]
        join_type_map = {
            JoinKind.INNER: "inner",
            JoinKind.LEFT: "left outer",
            JoinKind.RIGHT: "right outer",
            JoinKind.FULL: "full outer",
        }
        join_type = join_type_map.get(self.kind)
        if join_type is None:
            raise NotImplementedError(
                f"Join kind {self.kind!r} not supported. "
                f"Supported: {sorted(join_type_map)}."
            )
        return left.join(
            right,
            keys=keys,
            right_keys=right_keys,
            join_type=join_type,
        )


# ---------------------------------------------------------------------------
# Top-level evaluator
# ---------------------------------------------------------------------------


def evaluate(plan: PlanNode, ctx: "DynamicCatalog") -> Tabular:
    """One-shot evaluation entry point — equivalent to ``plan.execute(ctx)``."""
    return plan.execute(ctx)


# ---------------------------------------------------------------------------
# Predicate / expression eval helpers (Arrow path)
# ---------------------------------------------------------------------------


def _eval_predicate_arrow(predicate: Predicate, table: pa.Table) -> pa.Array:
    """Render *predicate* against *table* and return a boolean mask.

    Path: lift the predicate to a :class:`pyarrow.compute.Expression`
    and evaluate against a Dataset wrapper. Pyarrow doesn't support
    direct expression evaluation on a Table without going through the
    Dataset API, so we wrap the table once and run the filter.
    """
    import pyarrow.dataset as pds

    expr = predicate.to_arrow()
    mask = pds.dataset(table).to_table(filter=expr, columns=[]).num_rows
    # The above gets us the count, not the mask. Build the mask the
    # straightforward way: filter and re-derive membership.
    filtered = pds.dataset(table).to_table(filter=expr)
    if filtered.num_rows == table.num_rows:
        return pa.array([True] * table.num_rows, type=pa.bool_())
    if filtered.num_rows == 0:
        return pa.array([False] * table.num_rows, type=pa.bool_())
    # Build a key that identifies each input row by position via a
    # synthetic ordinal column, then test membership on the filtered
    # ordinals.
    ord_table = table.append_column(
        "_ygg_row_ord", pa.array(range(table.num_rows), type=pa.int64()),
    )
    kept = pds.dataset(ord_table).to_table(
        filter=expr, columns=["_ygg_row_ord"],
    )
    kept_set = set(kept.column("_ygg_row_ord").to_pylist())
    return pa.array(
        [i in kept_set for i in range(table.num_rows)], type=pa.bool_(),
    )


def _eval_expression_arrow(expr: Expression, table: pa.Table) -> pa.Array:
    """Evaluate a non-aggregate :class:`Expression` row-wise on *table*.

    Mirrors :func:`_eval_predicate_arrow` but for arbitrary expression
    output (column projections like ``UPPER(name)`` or ``a + b``).
    Path: lift to a pyarrow compute expression, run via Dataset.
    """
    import pyarrow.dataset as pds

    arrow_expr = expr.to_arrow()
    # Use the dataset projection trick to compute the expression as a
    # named column, then pull the single resulting column.
    ds = pds.dataset(table)
    out = ds.to_table(columns={"__ygg_expr_out": arrow_expr})
    return out.column("__ygg_expr_out").combine_chunks()


def _as_chunked(arr: Any) -> pa.ChunkedArray:
    if isinstance(arr, pa.ChunkedArray):
        return arr
    if isinstance(arr, pa.Array):
        return pa.chunked_array([arr])
    return pa.chunked_array([pa.array(arr)])
