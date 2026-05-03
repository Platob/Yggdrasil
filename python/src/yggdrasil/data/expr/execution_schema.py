"""Plan a select-and-filter pass over a :class:`TabularIO` source.

:class:`ExecutionSchema` is an immutable dataclass packaging four
intents that travel together when describing "read from this
source, filter rows, project columns":

::

    ExecutionSchema(
        alias=...,        # SQL alias for joined / MERGE planning
        source=...,       # bound TabularIO read target
        select=[...],     # projected expressions
        where=...,        # row predicate
    )

The fields are exposed both as constructor arguments and as
``with_*`` builders (``with_alias`` / ``with_source`` /
``with_select`` / ``with_where``) so callers can either materialize
the whole plan up front or build it fluently:

::

    from yggdrasil.data.expr import col, select, ExecutionSchema

    plan = (
        ExecutionSchema(alias="t", source=parquet_io)
        .with_select(col("price"), select("symbol", alias="ticker"))
        .with_where(col("price") >= 100)
    )

    # Apply against an Arrow Table
    out = plan.arrow_apply(parquet_io.read_arrow_table())
    # ...or against a single RecordBatch (streaming)
    out_batch = plan.arrow_batch_apply(batch)

When :attr:`select` is empty the plan implicitly projects every
source column — same convention SQL uses for ``SELECT *``. When
:attr:`source` is bound, the no-arg :meth:`arrow_apply` /
:meth:`arrow_batch_apply` read directly from it; without a source
callers must pass the table / batch explicitly.

Each ``with_*`` call returns a fresh :class:`ExecutionSchema`;
the underlying tuples / predicate are never mutated, so a base
plan can be branched and shared safely.
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
    from yggdrasil.io.buffer.base import TabularIO


__all__ = ["ExecutionSchema"]


@dataclasses.dataclass(frozen=True, slots=True)
class ExecutionSchema:
    """Immutable bound select + filter plan.

    Attributes
    ----------

    alias:
        SQL alias for this source — used by emitters that need
        ``T.col`` qualification (joins, MERGE planning,
        partition-side IN predicates). Default ``""`` means
        "no alias"; columns render unqualified.
    source:
        Bound :class:`TabularIO`. When set, no-arg
        :meth:`arrow_apply` reads from it directly. When
        ``None``, callers pass the input explicitly per call.
    select:
        Projected expressions in output order — typically
        :class:`Column` / :class:`Selector` but any
        :class:`Expression` is accepted (computed columns).
        Empty means ``SELECT *``: arrow_apply projects every
        source column.
    where:
        Optional :class:`Predicate` applied before the
        projection. Multiple :meth:`with_where` calls AND-merge
        under one :class:`Logical` so callers can layer filters
        without restating the conjunction.
    """

    alias: str = ""
    source: "TabularIO | None" = None
    select: "tuple[Expression, ...]" = ()
    where: "Predicate | None" = None

    def __post_init__(self) -> None:
        # ``select`` accepts any iterable but stays as a tuple of
        # :class:`Expression` so the dataclass remains hashable
        # and the AST stays canonical. Strings (the
        # ``select=["a", "b"]`` shorthand) get coerced to bare
        # column references here.
        coerced: tuple[Expression, ...]
        if self.select and not isinstance(self.select, tuple):
            coerced = tuple(_coerce_select(e) for e in self.select)
        else:
            coerced = tuple(_coerce_select(e) for e in self.select)
        object.__setattr__(self, "select", coerced)

        if self.where is not None:
            if not isinstance(self.where, Expression):
                raise TypeError(
                    f"where= expects an Expression / Predicate, got "
                    f"{type(self.where).__name__}."
                )
            if not isinstance(self.where, Predicate):
                raise TypeError(
                    f"where= expects a Predicate (boolean expression); "
                    f"got non-boolean {type(self.where).__name__}."
                )

    # ------------------------------------------------------------------
    # Fluent builders — every method returns a fresh plan
    # ------------------------------------------------------------------

    def with_alias(self, alias: str) -> "ExecutionSchema":
        """Return a copy of this plan rebound to ``alias``."""
        return dataclasses.replace(self, alias=alias)

    def with_source(self, source: "TabularIO") -> "ExecutionSchema":
        """Return a copy of this plan rebound to ``source``."""
        return dataclasses.replace(self, source=source)

    def with_select(
        self,
        *exprs: "Expression | str",
    ) -> "ExecutionSchema":
        """Append projected columns / expressions.

        Strings coerce to :class:`Column` references for the
        ``schema.with_select("a", "b")`` shortcut. Anything else
        must already be an :class:`Expression`.
        """
        coerced = tuple(_coerce_select(e) for e in exprs)
        return dataclasses.replace(self, select=self.select + coerced)

    def with_where(
        self,
        predicate: "Expression | None",
    ) -> "ExecutionSchema":
        """AND-merge a row filter.

        Repeated calls accumulate: ``schema.with_where(a).with_where(b)``
        is equivalent to ``schema.with_where(a & b)``.
        ``predicate=None`` is a no-op.
        """
        if predicate is None:
            return self
        if not isinstance(predicate, Expression):
            raise TypeError(
                f"with_where expects an Expression / Predicate, got "
                f"{type(predicate).__name__}."
            )
        if not isinstance(predicate, Predicate):
            raise TypeError(
                f"with_where expects a Predicate (boolean expression); "
                f"got non-boolean {type(predicate).__name__}."
            )
        merged = (
            predicate if self.where is None
            else Logical(LogicalOp.AND, (self.where, predicate))
        )
        return dataclasses.replace(self, where=merged)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @property
    def output_names(self) -> "tuple[str, ...]":
        """Names of the projected columns in output order.

        With an empty :attr:`select` and a bound :attr:`source`,
        this resolves the source's :meth:`collect_schema` field
        names (``SELECT *``). Otherwise Selector projections
        honour their ``projection_name`` (alias / output_name);
        plain Column entries use their ``name``; arbitrary
        expressions fall back to the first :class:`Column` they
        reference, or ``"expr_<i>"`` when no column is reachable.
        """
        from .nodes import free_columns

        if not self.select:
            return self._source_column_names()

        out: list[str] = []
        for i, expr in enumerate(self.select):
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
        for expr in self.select:
            if isinstance(expr, Selector):
                out.append(expr.projection_field)
            elif isinstance(expr, Column):
                out.append(expr.field)
            else:
                out.append(None)
        return tuple(out)

    def _source_column_names(self) -> "tuple[str, ...]":
        """Best-effort source column list for the ``SELECT *`` path."""
        if self.source is None:
            return ()
        try:
            schema = self.source.collect_schema
            if callable(schema):
                schema = schema()
        except Exception:
            return ()
        names = getattr(schema, "names", None)
        if names is None:
            return ()
        return tuple(names)

    # ------------------------------------------------------------------
    # Apply — pyarrow Table / RecordBatch
    # ------------------------------------------------------------------

    def arrow_apply(
        self,
        source: "pa.Table | pa.RecordBatch | None" = None,
    ) -> "pa.Table":
        """Apply the plan to an Arrow table or record batch.

        Order of operations:

        1. Resolve the input — ``source`` arg wins; otherwise
           the bound :attr:`source` is read via
           :meth:`TabularIO.read_arrow_table`. Without either,
           a :class:`TypeError` surfaces immediately.
        2. Filter rows via :attr:`where` (compiled to
           :class:`pyarrow.compute.Expression`) when one is set.
        3. Project + (optionally) cast every column listed in
           :attr:`select`. Empty ``select`` keeps every column
           (``SELECT *`` semantics).

        Returns a :class:`pyarrow.Table`.
        """
        import pyarrow as pa

        table = self._resolve_arrow_table(source, pa)

        if self.where is not None:
            table = table.filter(self.where.to_arrow())

        if not self.select:
            return table

        out_columns: list[pa.ChunkedArray] = []
        out_names: list[str] = []
        for expr in self.select:
            column, name = self._materialize_arrow_column(expr, table, pa)
            out_columns.append(column)
            out_names.append(name)
        return pa.Table.from_arrays(out_columns, names=out_names)

    def arrow_batch_apply(
        self,
        batch: "pa.RecordBatch | None" = None,
    ) -> "pa.RecordBatch":
        """Apply the plan to a single :class:`pyarrow.RecordBatch`.

        The streaming counterpart of :meth:`arrow_apply` — wraps
        the batch in a one-batch :class:`pyarrow.Table` for the
        filter / project pass (PyArrow's filter / project APIs
        operate on Tables) and combines the result back into a
        single batch on the way out.

        ``batch=None`` requires a bound :attr:`source`; the
        first batch from :meth:`TabularIO.read_arrow_batches` is
        consumed. Most callers in streaming pipelines pass the
        batch explicitly inside their own iterator.

        Returns a :class:`pyarrow.RecordBatch`. An empty filter
        result returns a zero-row batch with the projected
        schema so downstream consumers don't have to special-case
        the empty path.
        """
        import pyarrow as pa

        if batch is None:
            if self.source is None:
                raise TypeError(
                    "arrow_batch_apply needs a batch argument or a "
                    "bound source. Pass a pa.RecordBatch or set "
                    "ExecutionSchema(source=...) first."
                )
            batch = next(iter(self.source.read_arrow_batches()), None)
            if batch is None:
                # Source produced no batches — return an empty
                # batch matching the projected schema for the
                # caller to keep iterating cleanly.
                return self._empty_arrow_batch(pa)

        if not isinstance(batch, pa.RecordBatch):
            raise TypeError(
                f"arrow_batch_apply expects a pa.RecordBatch, got "
                f"{type(batch).__name__}."
            )

        result = self.arrow_apply(batch)
        if result.num_rows == 0:
            # ``Table.combine_chunks().to_batches()`` returns
            # ``[]`` for empty tables — synthesize a zero-row
            # batch with the right schema so we always return
            # the documented type.
            return pa.RecordBatch.from_pylist([], schema=result.schema)
        combined = result.combine_chunks()
        batches = combined.to_batches()
        if not batches:
            return pa.RecordBatch.from_pylist([], schema=result.schema)
        if len(batches) == 1:
            return batches[0]
        # ``combine_chunks`` should leave one batch when the
        # inputs were uniform; if PyArrow re-chunked anyway,
        # concatenate via Table → bytes → RecordBatch is
        # heavier than a single ``pa.concat_arrays`` per column,
        # which is what we do here.
        merged_columns = [
            pa.concat_arrays([b.column(i) for b in batches])
            for i in range(combined.num_columns)
        ]
        return pa.RecordBatch.from_arrays(merged_columns, schema=result.schema)

    # ------------------------------------------------------------------
    # Internal — Arrow plumbing
    # ------------------------------------------------------------------

    def _resolve_arrow_table(
        self,
        source: "pa.Table | pa.RecordBatch | None",
        pa,  # type: ignore[no-untyped-def]
    ) -> "pa.Table":
        if source is None:
            if self.source is None:
                raise TypeError(
                    "arrow_apply needs a source argument or a bound "
                    "ExecutionSchema.source. Pass a pa.Table / "
                    "pa.RecordBatch or set source= on the plan."
                )
            return self.source.read_arrow_table()
        if isinstance(source, pa.RecordBatch):
            return pa.Table.from_batches([source])
        if isinstance(source, pa.Table):
            return source
        raise TypeError(
            f"arrow_apply expects pa.Table or pa.RecordBatch, got "
            f"{type(source).__name__}."
        )

    def _empty_arrow_batch(self, pa):  # type: ignore[no-untyped-def]
        """Build a zero-row RecordBatch matching the projected schema.

        Falls back to an empty batch with no columns when no
        source schema is reachable — keeps the helper from
        raising on a fully unbound plan.
        """
        # Use ``arrow_apply`` against an empty table to avoid
        # duplicating the projection / cast logic. The probe
        # table is built from the bound source's schema when
        # available.
        if self.source is None:
            return pa.RecordBatch.from_pylist([], schema=pa.schema([]))
        try:
            schema = self.source.collect_schema
            if callable(schema):
                schema = schema()
            arrow_schema = schema.to_arrow_schema()
        except Exception:
            return pa.RecordBatch.from_pylist([], schema=pa.schema([]))
        empty_table = arrow_schema.empty_table()
        result = self.arrow_apply(empty_table)
        return pa.RecordBatch.from_pylist([], schema=result.schema)

    @staticmethod
    def _materialize_arrow_column(
        expr: Expression,
        table: "pa.Table",
        pa,  # type: ignore[no-untyped-def]
    ) -> "tuple[pa.ChunkedArray, str]":
        """Pull one projection column off ``table``.

        Selector / Column entries use the column lookup directly
        (cheap — no compute kernel runs). Arbitrary expressions
        fall back to a one-shot :class:`pyarrow.dataset.Scanner`
        evaluation so the predicate is compiled once and projected
        as a derived column.
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

        from .nodes import free_columns

        compute_expr = expr.to_arrow()
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
        return iter(self.select)

    def __bool__(self) -> bool:
        return (
            bool(self.select)
            or self.where is not None
            or self.source is not None
            or bool(self.alias)
        )


def _coerce_select(value: Any) -> Expression:
    """Coerce a select() arg into an Expression.

    Strings become :class:`Column` references — the natural
    shorthand for ``schema.with_select("a", "b")``. Anything
    else must be an :class:`Expression`; bare values are
    rejected because a literal as a projection would need a
    name (use a :class:`Selector` if that's really what you
    want).
    """
    if isinstance(value, Expression):
        return value
    if isinstance(value, str):
        return Column(name=value)
    raise TypeError(
        f"select expected str or Expression, got {type(value).__name__}. "
        "Wrap it in select(name) or col(name)."
    )
