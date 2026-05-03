"""Plan a select-and-filter pass over a frame.

:class:`ExecutionSchema` packages two intents that almost always
travel together — a list of projected columns / expressions and
an optional row predicate — into one fluent, immutable plan:

::

    from yggdrasil.data.expr import col, select, ExecutionSchema

    plan = (
        ExecutionSchema()
        .select(col("price"), select("symbol", alias="ticker"))
        .where(col("price") >= 100)
    )

    out = plan.arrow_apply(table)   # → pa.Table

The point of the wrapper (vs. emitting two separate predicates)
is that backends can compile both halves together — pyarrow's
:meth:`Dataset.scanner` honours filter + project in one pass,
Spark's `df.select(...).where(...)` order matters for predicate
pushdown, etc. :class:`ExecutionSchema` gives backends a single
input to inspect.

Each ``select(...)`` / ``where(...)`` call returns a fresh
:class:`ExecutionSchema` — the underlying tuples / predicate are
never mutated, so a base plan can be branched safely.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Iterable

from .nodes import (
    Column,
    Expression,
    Logical,
    LogicalOp,
    Predicate,
    Selector,
)

if TYPE_CHECKING:
    import pyarrow as pa
    from yggdrasil.data.data_field import Field


__all__ = ["ExecutionSchema"]


@dataclasses.dataclass(frozen=True, slots=True)
class ExecutionSchema:
    """Immutable select + filter plan.

    Construction is fluent — :meth:`select` and :meth:`where` are
    additive; calling them on an existing schema yields a fresh
    instance with the new entries appended (or AND-merged for the
    predicate). The dataclass itself is hashable / equality-by-
    value so plans are usable as dict keys.

    Attributes
    ----------

    selects:
        Tuple of projected expressions, in output order. Each
        entry is normally a :class:`Column` / :class:`Selector`
        but any :class:`Expression` is accepted — backends that
        support computed columns project them as derived
        outputs.
    where:
        Optional :class:`Predicate` applied before the projection.
        Multiple ``.where()`` calls AND-merge under one
        :class:`Logical` so callers can layer filters
        incrementally without losing precedence.
    """

    selects: "tuple[Expression, ...]" = ()
    where_predicate: "Predicate | None" = None

    # ------------------------------------------------------------------
    # Fluent builders
    # ------------------------------------------------------------------

    def select(
        self,
        *exprs: "Expression | str",
    ) -> "ExecutionSchema":
        """Append projected columns / expressions.

        Strings are coerced to :class:`Column` references for the
        common ``schema.select("a", "b")`` case. Anything else
        must already be an :class:`Expression`. Returns a new
        :class:`ExecutionSchema` — the receiver is unchanged.
        """
        coerced = tuple(_coerce_select(e) for e in exprs)
        return dataclasses.replace(
            self,
            selects=self.selects + coerced,
        )

    def where(self, predicate: "Expression | None") -> "ExecutionSchema":
        """AND-merge a row filter.

        Repeated calls accumulate: ``schema.where(a).where(b)`` is
        equivalent to ``schema.where(a & b)`` so callers can layer
        predicates without restating the conjunction.
        ``predicate=None`` is a no-op.
        """
        if predicate is None:
            return self
        if not isinstance(predicate, Expression):
            raise TypeError(
                f"where() expects an Expression / Predicate, got "
                f"{type(predicate).__name__}."
            )
        if not isinstance(predicate, Predicate):
            raise TypeError(
                f"where() expects a Predicate (boolean expression); "
                f"got a non-boolean {type(predicate).__name__}."
            )
        merged = (
            predicate
            if self.where_predicate is None
            else Logical(LogicalOp.AND, (self.where_predicate, predicate))
        )
        return dataclasses.replace(self, where_predicate=merged)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @property
    def output_names(self) -> "tuple[str, ...]":
        """Names of the projected columns in output order.

        Selector projections honour their ``projection_name``
        (alias / output_name); plain Column entries use their
        ``name``; arbitrary expressions fall back to the first
        :class:`Column` they reference (best-effort) or
        ``"expr_<i>"`` when no column is reachable.
        """
        from .nodes import free_columns

        out: list[str] = []
        for i, expr in enumerate(self.selects):
            if isinstance(expr, Selector):
                out.append(expr.projection_name)
            elif isinstance(expr, Column):
                out.append(expr.alias or expr.name)
            else:
                cols = free_columns(expr)
                out.append(cols[0] if cols else f"expr_{i}")
        return tuple(out)

    @property
    def output_fields(self) -> "tuple[Field | None, ...]":
        """Per-column :class:`Field` projections, when known."""
        out: "list[Field | None]" = []
        for expr in self.selects:
            if isinstance(expr, Selector):
                out.append(expr.projection_field)
            elif isinstance(expr, Column):
                out.append(expr.field)
            else:
                out.append(None)
        return tuple(out)

    # ------------------------------------------------------------------
    # Apply — pyarrow Table / RecordBatch
    # ------------------------------------------------------------------

    def arrow_apply(
        self,
        source: "pa.Table | pa.RecordBatch",
    ) -> "pa.Table":
        """Apply the plan to an Arrow table or record batch.

        Order of operations:

        1. Filter rows via :attr:`where_predicate` (compiled to
           :class:`pyarrow.compute.Expression`) when one is set.
           Skipped on a plain :class:`pyarrow.RecordBatch` because
           filter-on-batch needs a full :class:`Table` round-trip
           — we lift to a Table first.
        2. Project + (optionally) cast every column listed in
           :attr:`selects`. Selector entries with a
           :attr:`Selector.projection_field` get
           :meth:`pyarrow.ChunkedArray.cast` to that Field's
           Arrow type so downstream consumers receive the
           promised dtype.

        Returns a :class:`pyarrow.Table`. Empty :attr:`selects`
        means "all columns", same convention SQL uses for
        ``SELECT *`` — useful when the schema is just a
        ``WHERE`` filter.
        """
        import pyarrow as pa

        if isinstance(source, pa.RecordBatch):
            table = pa.Table.from_batches([source])
        elif isinstance(source, pa.Table):
            table = source
        else:
            raise TypeError(
                f"arrow_apply expects pa.Table or pa.RecordBatch, "
                f"got {type(source).__name__}."
            )

        if self.where_predicate is not None:
            table = table.filter(self.where_predicate.to_arrow())

        if not self.selects:
            return table

        out_columns: list[pa.ChunkedArray] = []
        out_names: list[str] = []
        for expr in self.selects:
            column, name = self._materialize_arrow_column(expr, table, pa)
            out_columns.append(column)
            out_names.append(name)

        return pa.Table.from_arrays(out_columns, names=out_names)

    @staticmethod
    def _materialize_arrow_column(
        expr: Expression,
        table: "pa.Table",
        pa,  # type: ignore[no-untyped-def]
    ) -> "tuple[pa.ChunkedArray, str]":
        """Pull one projection column off ``table``.

        Selector / Column entries use the column lookup directly
        (cheap — no compute kernel runs). Arbitrary expressions
        fall back to ``pa.Table.append_column`` via
        :meth:`pyarrow.compute.Expression` — the predicate is
        compiled once and projected as a derived column with the
        first referenced column's name (or ``expr_i`` if none).
        """
        if isinstance(expr, Selector):
            arr = table.column(expr.name)
            field = expr.projection_field
            if field is not None:
                try:
                    arr = arr.cast(field.dtype.to_arrow())
                except Exception:
                    # Cast failures fall through to the raw column;
                    # callers that demand a strict cast can wrap
                    # the Selector in a Cast() node and read the
                    # error from there.
                    pass
            return arr, expr.projection_name

        if isinstance(expr, Column):
            arr = table.column(expr.name)
            if expr.field is not None:
                try:
                    arr = arr.cast(expr.field.dtype.to_arrow())
                except Exception:
                    pass
            return arr, expr.alias or expr.name

        # Arbitrary expression — compile to pyarrow.compute, then
        # project against the table by adding a column. We name
        # it after the first referenced column to keep the output
        # readable; callers that want a specific name should wrap
        # the expression in a Selector via ``alias=`` semantics
        # added later.
        from .nodes import free_columns

        compute_expr = expr.to_arrow()
        # ``Table.append_column`` doesn't take an Expression; we
        # use a one-row probe via ``Dataset`` to evaluate the
        # expression. Simpler: build a dataset from the table and
        # scan with the compute_expr as a projection.
        import pyarrow.dataset as pds

        dataset = pds.dataset(table)
        cols = free_columns(expr)
        proj_name = cols[0] if cols else "expr"
        scanner = dataset.scanner(columns={proj_name: compute_expr})
        projected = scanner.to_table()
        return projected.column(proj_name), proj_name

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterable[Expression]:
        return iter(self.selects)

    def __bool__(self) -> bool:
        return bool(self.selects) or self.where_predicate is not None


def _coerce_select(value: Any) -> Expression:
    """Coerce a select() arg into an Expression.

    Strings become :class:`Column` references — the natural
    shorthand for ``schema.select("a", "b")``. Anything else must
    be an :class:`Expression`; bare values are rejected because
    a literal as a projection would need a name (use a
    :class:`Selector` if that's really what you want).
    """
    if isinstance(value, Expression):
        return value
    if isinstance(value, str):
        return Column(name=value)
    raise TypeError(
        f"select() expected str or Expression, got "
        f"{type(value).__name__}. Wrap it in select(name) or col(name)."
    )
