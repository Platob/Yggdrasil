"""Planner — turn a sqlglot AST into an :class:`ExecutionPlan` tree.

The planner is intentionally small: one method per SQL clause, each
appending one node on top of the running tree. Pipeline order:

1. ``FROM`` + ``JOIN`` → :class:`Scan` / :class:`Join`
2. ``WHERE``           → :class:`Filter`
3. ``GROUP BY``        → :class:`Aggregate` (also covers global aggregate)
4. ``HAVING``          → :class:`Filter`
5. ``SELECT``          → :class:`Project`
6. ``ORDER BY``        → :class:`Sort`
7. ``LIMIT`` / ``OFFSET`` → :class:`Limit`

We don't implement any optimizer pass here — pushdown happens in two
narrow places:

- :meth:`_lower_filter_into_scan` folds a single-source ``WHERE`` into
  the underlying :class:`Scan`'s ``predicate`` slot so the source
  Tabular gets a chance at native filter pushdown (Parquet, Delta).
- :meth:`_lower_projection_into_scan` folds the column list into the
  Scan's ``projection`` slot so wide tables only materialize what
  the SELECT actually wants.

Both rewrites are conservative: a JOIN, GROUP BY, or HAVING blocks
the lowering since the predicate / projection might reference
columns the Scan can't see in isolation.

Why we still go through sqlglot rather than parsing ourselves
-------------------------------------------------------------

Sqlglot already speaks every flavor we care about (Databricks,
Postgres, MySQL, SQLite, ANSI). The translator below only has to
walk the *normalized* node types — sqlglot does the heavy lifting of
turning ``ILIKE`` into ``LIKE``, dialect-specific functions into
canonical names, etc. That keeps :mod:`yggdrasil.sql.planner` tight
and lets us inherit dialect support for free as sqlglot grows.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from yggdrasil.data.expr import Expression, Predicate

from yggdrasil.sql.dialect import Dialect, resolve_dialect
from yggdrasil.sql.lib import sqlglot_expressions
from yggdrasil.sql.parser import parse, SqlParseError
from yggdrasil.sql.plan import (
    Aggregate,
    AggregateSpec,
    Filter,
    Join,
    JoinKind,
    Limit,
    PlanNode,
    Project,
    ProjectionItem,
    Scan,
    Sort,
    SortKey,
)


if TYPE_CHECKING:
    from yggdrasil.sql.dynamic_catalog import DynamicCatalog


__all__ = ["Planner", "PlanError", "plan"]


class PlanError(ValueError):
    """Raised when the planner can't translate a SQL feature.

    Distinct from :class:`SqlParseError` (which is "the SQL is
    syntactically broken"); a :class:`PlanError` means "we parsed it
    but our engine doesn't yet know how to execute it." The message
    always names the offending clause and links to a workaround
    where one exists.
    """


# ---------------------------------------------------------------------------
# Aggregate detection
# ---------------------------------------------------------------------------


_AGG_FUNCTION_NAMES = {
    "Count": "count",
    "Sum": "sum",
    "Avg": "avg",
    "Min": "min",
    "Max": "max",
    "Mean": "avg",
}


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class Planner:
    """Walk a sqlglot AST and produce a :class:`PlanNode` tree.

    One :class:`Planner` per :meth:`plan` call — internal counters /
    alias maps live on the instance and would otherwise leak between
    independent plans.
    """

    def __init__(self, dialect: "Dialect | str | None" = None) -> None:
        self.dialect = resolve_dialect(dialect)
        self._sge = sqlglot_expressions()
        self._alias_to_table: dict[str, str] = {}

    # ==================================================================
    # Public entry
    # ==================================================================

    def plan(self, query: str | Any) -> PlanNode:
        """Parse *query* (or accept a sqlglot AST) and return the plan tree."""
        if isinstance(query, str):
            root = parse(query, dialect=self.dialect)
        else:
            root = query
        sge = self._sge
        if not isinstance(root, sge.Select):
            raise PlanError(
                f"yggdrasil.sql.engine handles SELECT statements only; got "
                f"{type(root).__name__} via {getattr(root, 'sql', lambda **_: '?')()!r}. "
                "DDL / DML are not supported on the in-process catalog."
            )
        return self._plan_select(root)

    # ==================================================================
    # SELECT
    # ==================================================================

    def _plan_select(self, node: Any) -> PlanNode:
        sge = self._sge

        # 1. FROM + JOINs.
        plan = self._plan_from(node)

        # 2. WHERE.
        where = node.args.get("where")
        if where is not None:
            predicate = self._lift_predicate(where.this)
            plan = self._maybe_lower_filter(plan, predicate)

        # 3. GROUP BY (or implicit global aggregate driven by SELECT).
        group_by = node.args.get("group")
        having = node.args.get("having")
        agg_specs = self._collect_aggregates(node)

        if group_by is not None or agg_specs:
            keys = self._collect_group_keys(group_by)
            plan = Aggregate(
                child=plan,
                group_keys=tuple(keys),
                aggregates=tuple(agg_specs),
            )

        # 4. HAVING (Filter on top of Aggregate).
        if having is not None:
            having_pred = self._lift_predicate(having.this)
            plan = Filter(child=plan, predicate=having_pred)

        # 5. ORDER BY runs *before* SELECT projection in our pipeline
        # so ``ORDER BY <col>`` can reference any physical column the
        # source carries — even when SELECT later drops it. Standard
        # SQL semantics: ORDER BY's column space is the FROM rows
        # plus the projection aliases. For aggregate plans the
        # Aggregate node has already materialized the aggregate
        # aliases, so they're visible to Sort here.
        order = node.args.get("order")
        if order is not None:
            keys = self._collect_sort_keys(order)
            plan = Sort(child=plan, keys=tuple(keys))

        # 6. SELECT projection — only when not pure aggregate. The
        # Aggregate node already produced columns named after its
        # ``alias`` slots; pass-through aggregate-only SELECTs need
        # no extra projection.
        if node.expressions and not (agg_specs and not group_by):
            project_items = self._collect_projection(node, has_aggregate=bool(agg_specs))
            if project_items is not None:
                plan = self._maybe_lower_projection(plan, project_items)

        # 7. LIMIT / OFFSET.
        limit = node.args.get("limit")
        offset = node.args.get("offset")
        if limit is not None or offset is not None:
            n = _int_from_limit(limit) if limit is not None else (1 << 62)
            o = _int_from_limit(offset) if offset is not None else 0
            plan = Limit(child=plan, n=n, offset=o)

        return plan

    # ==================================================================
    # FROM + JOINs
    # ==================================================================

    def _plan_from(self, node: Any) -> PlanNode:
        sge = self._sge
        from_clause = node.args.get("from") or node.args.get("from_")
        if from_clause is None:
            raise PlanError(
                "SELECT without a FROM clause is not supported "
                "(VALUES-only / scalar SELECTs)."
            )
        # sqlglot's From holds the lead table on ``this``; multiple
        # comma-separated tables go on ``expressions``.
        leads = [from_clause.this] if from_clause.this is not None else []
        leads += list(from_clause.args.get("expressions") or [])
        if not leads:
            raise PlanError("FROM clause did not yield any tables.")

        # First lead → starting plan; remaining leads are implicit
        # cross joins (``FROM a, b``).
        plan = self._plan_table_or_subquery(leads[0])
        for extra in leads[1:]:
            right = self._plan_table_or_subquery(extra)
            plan = Join(left=plan, right=right, kind=JoinKind.CROSS, on=())

        # Explicit JOIN clauses.
        for join_node in (node.args.get("joins") or []):
            plan = self._apply_join(plan, join_node)
        return plan

    def _plan_table_or_subquery(self, node: Any) -> PlanNode:
        sge = self._sge
        # Strip alias wrapper but remember the alias.
        alias_name: Optional[str] = None
        if isinstance(node, sge.Alias):
            alias_name = node.alias
            node = node.this
        # Subquery → recursively plan, wrap in a Project that just
        # forwards the columns under the alias if any.
        if isinstance(node, (sge.Subquery, sge.Select)):
            inner = node.this if isinstance(node, sge.Subquery) else node
            sub_plan = self._plan_select(inner)
            if alias_name:
                self._alias_to_table[alias_name] = alias_name
            return sub_plan

        if isinstance(node, sge.Table):
            qualified = self._qualified_table_name(node)
            tab_alias = node.alias_or_name
            if alias_name:
                tab_alias = alias_name
            self._alias_to_table[tab_alias] = qualified
            return Scan(name=qualified, alias=tab_alias)

        raise PlanError(
            f"Unsupported FROM entry: {type(node).__name__} "
            f"({_safe_sql(node, self.dialect)})."
        )

    def _qualified_table_name(self, node: Any) -> str:
        parts: list[str] = []
        for attr in ("catalog", "db"):
            v = getattr(node, attr, None)
            if v:
                parts.append(str(v))
        parts.append(node.name)
        return ".".join(parts)

    def _apply_join(self, left: PlanNode, join_node: Any) -> PlanNode:
        sge = self._sge

        right = self._plan_table_or_subquery(join_node.this)
        kind = self._join_kind(join_node)
        on_node = join_node.args.get("on")
        using = join_node.args.get("using")
        on_pairs: list[Tuple[str, str]] = []
        residual: Optional[Predicate] = None

        if using:
            # ``USING (col1, col2)`` — equality on shared columns.
            for col_id in using:
                name = col_id.name if hasattr(col_id, "name") else str(col_id)
                on_pairs.append((name, name))
        elif on_node is not None:
            on_pairs, residual = self._extract_equi_join(on_node)

        if kind == JoinKind.CROSS and on_pairs:
            kind = JoinKind.INNER
        plan: PlanNode = Join(left=left, right=right, kind=kind, on=tuple(on_pairs))
        if residual is not None:
            plan = Filter(child=plan, predicate=residual)
        return plan

    def _join_kind(self, join_node: Any) -> str:
        side = (join_node.args.get("side") or "").lower()
        kind = (join_node.args.get("kind") or "").lower()
        if kind == "cross":
            return JoinKind.CROSS
        if side == "left":
            return JoinKind.LEFT
        if side == "right":
            return JoinKind.RIGHT
        if side == "full":
            return JoinKind.FULL
        return JoinKind.INNER

    def _extract_equi_join(
        self, on_node: Any,
    ) -> "Tuple[list[Tuple[str, str]], Optional[Predicate]]":
        """Split a join ``ON`` clause into equality pairs + a residual filter.

        ``a.x = b.y AND a.z > 5`` becomes ``([(x, y)], a.z > 5)`` —
        the equality goes into the hash-join key list, the rest stays
        as a post-join Filter.
        """
        sge = self._sge
        equi: list[Tuple[str, str]] = []
        residuals: list[Any] = []
        stack: list[Any] = [on_node]
        while stack:
            cur = stack.pop()
            if isinstance(cur, sge.And):
                stack.append(cur.this)
                stack.append(cur.expression)
                continue
            if isinstance(cur, sge.EQ):
                left_col = self._column_name(cur.this)
                right_col = self._column_name(cur.expression)
                if left_col and right_col:
                    equi.append((left_col, right_col))
                    continue
            residuals.append(cur)
        residual: Optional[Predicate] = None
        if residuals:
            combined = residuals[0]
            for r in residuals[1:]:
                combined = sge.And(this=combined, expression=r)
            lifted = self._lift_predicate(combined)
            residual = lifted
        return equi, residual

    @staticmethod
    def _column_name(node: Any) -> Optional[str]:
        if hasattr(node, "name") and node.name:
            return node.name
        return None

    # ==================================================================
    # WHERE / HAVING — predicate lift
    # ==================================================================

    def _lift_predicate(self, node: Any) -> Predicate:
        """Turn a sqlglot expression into our :class:`Predicate` AST.

        Round-trip via :meth:`Expression.from_sql`: render the node
        back to SQL using the active dialect, then parse it through
        the expression frontend. That gives us ``IS NULL`` / ``BETWEEN``
        / ``IN (list)`` / ``LIKE`` / arithmetic ops for free without
        re-implementing the translator here.
        """
        sql = _safe_sql(node, self.dialect)
        if not sql:
            raise PlanError("Empty predicate; sqlglot rendered to nothing.")
        expr = Expression.from_sql(sql, dialect=self.dialect.value)
        if not isinstance(expr, Predicate):
            # SQL boolean fragments lift to a Predicate; if not, we
            # got a scalar expression where one wasn't expected.
            raise PlanError(
                f"WHERE / HAVING clause did not lift to a boolean predicate: "
                f"{sql!r} → {type(expr).__name__}."
            )
        return expr

    # ==================================================================
    # SELECT projections + aggregates
    # ==================================================================

    def _collect_projection(
        self, node: Any, *, has_aggregate: bool,
    ) -> Optional[List[ProjectionItem]]:
        """Build the projection list, or ``None`` for ``SELECT *``.

        ``SELECT *`` returns ``None`` so the planner skips the
        :class:`Project` node — the underlying Scan / Aggregate
        already returns every column of interest.
        """
        sge = self._sge
        items: list[ProjectionItem] = []
        for raw in node.expressions:
            alias = None
            inner = raw
            if isinstance(raw, sge.Alias):
                alias = raw.alias
                inner = raw.this

            if isinstance(inner, sge.Star):
                # Wildcard — let the upstream node fan out.
                return None

            # Aggregate calls inside SELECT are pre-handled by
            # :meth:`_collect_aggregates`; here we just emit a passthrough
            # ``ProjectionItem`` referencing the aggregate's alias.
            agg_alias = self._aggregate_alias_if_any(raw)
            if agg_alias is not None:
                items.append(ProjectionItem(source=agg_alias, alias=agg_alias))
                continue

            if isinstance(inner, sge.Column):
                col_name = inner.name
                items.append(ProjectionItem(
                    source=col_name,
                    alias=alias or col_name,
                ))
                continue

            # Computed expression — lift to Expression AST.
            sql = _safe_sql(inner, self.dialect)
            try:
                expr = Expression.from_sql(sql, dialect=self.dialect.value)
            except Exception as exc:
                raise PlanError(
                    f"Unsupported SELECT expression {sql!r}: {exc}."
                ) from exc
            output_alias = alias or sql
            items.append(ProjectionItem(source=expr, alias=output_alias))

        return items

    def _aggregate_alias_if_any(self, raw: Any) -> Optional[str]:
        """Return the output alias for an aggregate-call SELECT entry, else None.

        Aggregate columns are emitted by :class:`Aggregate` under the
        same alias the SELECT requested, so the projection node only
        needs to forward by name.
        """
        sge = self._sge
        alias = None
        inner = raw
        if isinstance(raw, sge.Alias):
            alias = raw.alias
            inner = raw.this
        if not self._is_aggregate(inner):
            return None
        if alias:
            return alias
        # No explicit alias → use the rendered SQL as the synthetic
        # alias. This matches the same key :meth:`_collect_aggregates`
        # uses, so projection forwarding hits the right column.
        return _safe_sql(inner, self.dialect)

    def _collect_aggregates(self, node: Any) -> List[AggregateSpec]:
        """Walk the SELECT list + HAVING for aggregate calls.

        Each aggregate is registered under its rendered SQL (or the
        explicit alias when present) so ``COUNT(*)`` referenced in
        both SELECT and HAVING resolves to one Aggregate slot.
        """
        sge = self._sge
        seen: dict[str, AggregateSpec] = {}

        def _walk(item: Any) -> None:
            if item is None:
                return
            if self._is_aggregate(item):
                spec = self._aggregate_to_spec(item, alias=None)
                # Use the synthetic alias key for de-duplication —
                # pre-aliased SELECT entries get their alias as key.
                seen.setdefault(spec.alias, spec)
                return
            args = getattr(item, "args", None)
            if not args:
                return
            for child in args.values():
                if isinstance(child, list):
                    for c in child:
                        _walk(c)
                else:
                    _walk(child)

        for raw in node.expressions or []:
            inner = raw.this if isinstance(raw, sge.Alias) else raw
            alias = raw.alias if isinstance(raw, sge.Alias) else None
            if self._is_aggregate(inner):
                spec = self._aggregate_to_spec(inner, alias=alias)
                seen[spec.alias] = spec
            else:
                _walk(inner)

        having = node.args.get("having")
        if having is not None:
            _walk(having.this)
        return list(seen.values())

    def _is_aggregate(self, node: Any) -> bool:
        sge = self._sge
        # sqlglot exposes individual aggregate classes (Count / Sum /
        # Avg / Min / Max). We type-check against a tuple so we don't
        # depend on a hypothetical AggFunc base.
        agg_classes = tuple(
            getattr(sge, name) for name in _AGG_FUNCTION_NAMES
            if hasattr(sge, name)
        )
        return isinstance(node, agg_classes)

    def _aggregate_to_spec(
        self, node: Any, *, alias: Optional[str],
    ) -> AggregateSpec:
        sge = self._sge
        cls_name = type(node).__name__
        func = _AGG_FUNCTION_NAMES.get(cls_name, cls_name.lower())
        column: Optional[str] = None
        argument = node.this
        if argument is None or isinstance(argument, sge.Star):
            column = None
        elif isinstance(argument, sge.Column):
            column = argument.name
        else:
            # Computed argument inside an aggregate — not yet supported
            # (e.g. ``SUM(a + b)``). Project the inner expression first
            # if you need this.
            raise PlanError(
                f"Aggregate over a computed expression is not supported: "
                f"{cls_name}({_safe_sql(argument, self.dialect)}). "
                "Wrap the inner expression in a subquery's SELECT first."
            )
        distinct = bool(node.args.get("distinct"))
        out_alias = alias or _safe_sql(node, self.dialect)
        return AggregateSpec(
            func=func, column=column, alias=out_alias, distinct=distinct,
        )

    # ==================================================================
    # GROUP BY / ORDER BY
    # ==================================================================

    def _collect_group_keys(self, group_by: Any) -> List[str]:
        if group_by is None:
            return []
        keys: list[str] = []
        for entry in group_by.expressions:
            name = self._column_name(entry)
            if not name:
                raise PlanError(
                    f"GROUP BY supports column references only; got "
                    f"{_safe_sql(entry, self.dialect)!r}."
                )
            keys.append(name)
        return keys

    def _collect_sort_keys(self, order: Any) -> List[SortKey]:
        sge = self._sge
        keys: list[SortKey] = []
        for entry in order.expressions:
            desc = bool(entry.args.get("desc"))
            nulls_first = bool(entry.args.get("nulls_first"))
            target = entry.this
            name = self._column_name(target)
            if not name:
                raise PlanError(
                    f"ORDER BY supports column references only; got "
                    f"{_safe_sql(entry, self.dialect)!r}."
                )
            keys.append(SortKey(
                column=name, descending=desc, nulls_first=nulls_first,
            ))
        return keys

    # ==================================================================
    # Pushdown
    # ==================================================================

    def _maybe_lower_filter(
        self, plan: PlanNode, predicate: Predicate,
    ) -> PlanNode:
        """Try to fold *predicate* into the underlying Scan.

        Conservative: only when the plan is a single Scan (no joins
        below it). Anything else gets a regular Filter on top.
        """
        if isinstance(plan, Scan):
            merged = (
                predicate
                if plan.predicate is None
                else plan.predicate.merge_with(predicate)
            )
            return dataclasses.replace(plan, predicate=merged)
        return Filter(child=plan, predicate=predicate)

    def _maybe_lower_projection(
        self, plan: PlanNode, items: List[ProjectionItem],
    ) -> PlanNode:
        """When the projection is a pure column-passthrough over a
        single Scan, fold the column list into the Scan and skip the
        Project node.
        """
        if isinstance(plan, Scan) and all(
            isinstance(it.source, str) and it.source == it.alias
            for it in items
        ):
            cols = tuple(it.source for it in items)
            return dataclasses.replace(plan, projection=cols)
        return Project(child=plan, items=tuple(items))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_sql(node: Any, dialect: Dialect) -> str:
    """Render *node* back to SQL via sqlglot, swallowing render errors."""
    try:
        return node.sql(dialect=dialect.value) if node is not None else ""
    except Exception:
        return ""


def _int_from_limit(node: Any) -> int:
    """Pull an integer out of a sqlglot ``Limit`` / ``Offset`` arg."""
    expr = node.expression if hasattr(node, "expression") else node
    if expr is None:
        return 0
    if hasattr(expr, "this"):
        # ``Literal`` carries the value on ``this``.
        try:
            return int(expr.this)
        except (TypeError, ValueError):
            pass
    try:
        return int(_safe_sql(expr, Dialect.DATABRICKS))
    except (TypeError, ValueError):
        return 0


def plan(
    query: str | Any,
    *,
    dialect: "Dialect | str | None" = None,
) -> PlanNode:
    """Convenience wrapper — instantiate a :class:`Planner` and run it once."""
    return Planner(dialect=dialect).plan(query)
