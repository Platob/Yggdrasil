"""Emit SQL from plan nodes.

Walks a :class:`PlanNode` tree and renders SQL text in the chosen
dialect. Expression-level emission delegates to the existing
:func:`yggdrasil.saga.expr.backends.sql.to_sql`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from yggdrasil.enums.dialect import Dialect
from yggdrasil.saga.expr.backends.sql import (
    _quote_ident,
    _resolve_dialect,
    to_sql as _expr_to_sql,
)
from yggdrasil.saga.expr.nodes import (
    Alias,
    Column,
    Expression,
    FunctionCall,
    Literal,
    SortOrder,
    Star,
    WindowFunction,
    WindowSpec,
)

from .nodes import InsertNode, MergeNode, PlanNode, ScanNode, SelectNode
from .ops import CTE, JoinClause, LateralViewItem, SetOp, SubqueryRef, TableRef, ValuesRef

if TYPE_CHECKING:
    pass


def emit_sql(node: PlanNode, *, dialect: "Dialect | str | None" = None) -> str:
    return _emit(node, _resolve_dialect(dialect))


def _emit(node: PlanNode, d: Dialect) -> str:
    if isinstance(node, SelectNode): return _emit_select(node, d)
    if isinstance(node, InsertNode): return _emit_insert(node, d)
    if isinstance(node, MergeNode): return _emit_merge(node, d)
    if isinstance(node, ScanNode):
        n = node.name or "?"
        return f"{_quote_ident(n, d)} {_quote_ident(node.alias, d)}" if node.alias else _quote_ident(n, d)
    raise NotImplementedError(f"Cannot emit SQL for {type(node).__name__}")


def _emit_select(node: SelectNode, d: Dialect) -> str:
    parts: list[str] = []

    # CTEs
    if node.ctes:
        cte_parts = []
        for cte in node.ctes:
            body = _emit(cte.plan, d)
            cte_parts.append(f"{_quote_ident(cte.name, d)} AS ({body})")
        parts.append("WITH " + ", ".join(cte_parts))

    # SELECT [DISTINCT]
    select_kw = "SELECT DISTINCT" if node.distinct else "SELECT"
    proj = ", ".join(_emit_expr(e, d) for e in node.projections) if node.projections else "*"
    parts.append(f"{select_kw} {proj}")

    # FROM
    if node.from_clause is not None:
        parts.append("FROM " + _emit_from(node.from_clause, d))

    # LATERAL VIEW
    if node.lateral_views:
        for lv in node.lateral_views:
            func_sql = _emit_expr(lv.function, d)
            cols = ", ".join(_quote_ident(c, d) for c in lv.column_aliases)
            lv_sql = f"LATERAL VIEW {func_sql} {_quote_ident(lv.table_alias, d)}"
            if cols:
                lv_sql += f" AS {cols}"
            parts.append(lv_sql)

    # WHERE
    if node.where is not None:
        parts.append("WHERE " + _emit_expr(node.where, d))

    # GROUP BY
    if node.group_by:
        parts.append("GROUP BY " + ", ".join(_emit_expr(e, d) for e in node.group_by))

    # HAVING
    if node.having is not None:
        parts.append("HAVING " + _emit_expr(node.having, d))

    # QUALIFY
    if node.qualify is not None:
        parts.append("QUALIFY " + _emit_expr(node.qualify, d))

    # ORDER BY
    if node.order_by:
        parts.append("ORDER BY " + ", ".join(_emit_order(o, d) for o in node.order_by))

    # LIMIT
    if node.limit is not None:
        parts.append(f"LIMIT {node.limit}")

    # OFFSET
    if node.offset is not None:
        parts.append(f"OFFSET {node.offset}")

    sql = " ".join(parts)

    # Set operations
    if node.set_ops:
        for sop in node.set_ops:
            right_sql = _emit(sop.plan, d)
            sql = f"{sql} {sop.kind} {right_sql}"

    return sql


def _emit_from(item: Any, d: Dialect) -> str:
    if isinstance(item, TableRef):
        parts = []
        if item.catalog:
            parts.append(_quote_ident(item.catalog, d))
        if item.schema:
            parts.append(_quote_ident(item.schema, d))
        parts.append(_quote_ident(item.name, d))
        name = ".".join(parts)
        if item.alias:
            return f"{name} {_quote_ident(item.alias, d)}"
        return name
    if isinstance(item, SubqueryRef):
        inner = _emit(item.plan, d)
        return f"({inner}) {_quote_ident(item.alias, d)}"
    if isinstance(item, ValuesRef):
        rows = ", ".join(
            "(" + ", ".join(_emit_expr(v, d) for v in row) + ")"
            for row in item.values
        )
        sql = f"(VALUES {rows}) {_quote_ident(item.alias, d)}"
        if item.columns:
            cols = ", ".join(_quote_ident(c, d) for c in item.columns)
            sql += f" ({cols})"
        return sql
    if isinstance(item, JoinClause):
        left_sql = _emit_from(item.left, d)
        right_sql = _emit_from(item.right, d)
        jt = item.join_type.sql if hasattr(item.join_type, "sql") else str(item.join_type)
        on_sql = f" ON {_emit_expr(item.on, d)}" if item.on else ""
        return f"{left_sql} {jt} {right_sql}{on_sql}"
    if isinstance(item, PlanNode):
        return _emit(item, d)
    return str(item)


def _emit_expr(expr: Any, d: Dialect) -> str:
    if isinstance(expr, Star):
        if expr.qualifier:
            return f"{_quote_ident(expr.qualifier, d)}.*"
        return "*"
    if isinstance(expr, Alias):
        inner = _emit_expr(expr.expr, d)
        return f"{inner} AS {_quote_ident(expr.name, d)}"
    if isinstance(expr, FunctionCall):
        # INTERVAL 'value' unit — special syntax, not parenthesized
        if expr.name == "INTERVAL" and len(expr.args) == 2:
            val = _emit_expr(expr.args[0], d)
            unit = expr.args[1].value if isinstance(expr.args[1], Literal) else _emit_expr(expr.args[1], d)
            return f"INTERVAL {val} {unit}"
        # EXTRACT(field FROM source) — special keyword syntax
        if expr.name == "EXTRACT" and len(expr.args) == 2:
            field = expr.args[0].value if isinstance(expr.args[0], Literal) else _emit_expr(expr.args[0], d)
            source = _emit_expr(expr.args[1], d)
            return f"EXTRACT({field} FROM {source})"
        args = ", ".join(_emit_expr(a, d) for a in expr.args)
        distinct = "DISTINCT " if expr.distinct else ""
        return f"{expr.name}({distinct}{args})"
    if isinstance(expr, WindowFunction):
        func_sql = _emit_expr(expr.function, d)
        win_sql = _emit_window_spec(expr.window, d)
        return f"{func_sql} OVER ({win_sql})"
    if isinstance(expr, SortOrder):
        return _emit_order(expr, d)
    if isinstance(expr, Expression):
        return _expr_to_sql(expr, dialect=d)
    return str(expr)


def _emit_order(item: Any, d: Dialect) -> str:
    if isinstance(item, SortOrder):
        expr, asc, nf = item.expr, item.ascending, item.nulls_first
    elif isinstance(item, dict):
        expr, asc, nf = item["expr"], item.get("ascending", True), item.get("nulls_first")
    else:
        return _emit_expr(item, d)
    s = _emit_expr(expr, d) + (" ASC" if asc else " DESC")
    if nf is True: s += " NULLS FIRST"
    elif nf is False: s += " NULLS LAST"
    return s


def _emit_window_spec(spec: Any, d: Dialect) -> str:
    pb = spec.partition_by if isinstance(spec, WindowSpec) else spec.get("partition_by", ())
    ob = spec.order_by if isinstance(spec, WindowSpec) else spec.get("order_by", ())
    fs = spec.frame_start if isinstance(spec, WindowSpec) else spec.get("frame_start")
    fe = spec.frame_end if isinstance(spec, WindowSpec) else spec.get("frame_end")
    parts: list[str] = []
    if pb: parts.append("PARTITION BY " + ", ".join(_emit_expr(e, d) for e in pb))
    if ob: parts.append("ORDER BY " + ", ".join(_emit_order(o, d) for o in ob))
    if fs: parts.append(f"ROWS BETWEEN {fs} AND {fe}" if fe else f"ROWS {fs}")
    return " ".join(parts)


def _emit_insert(node: InsertNode, d: Dialect) -> str:
    target = _emit_from(node.target, d) if node.target else "?"
    cols = ""
    if node.columns:
        cols = " (" + ", ".join(_quote_ident(c, d) for c in node.columns) + ")"
    if node.source:
        src = _emit(node.source, d)
        return f"INSERT INTO {target}{cols} {src}"
    if node.values:
        rows = ", ".join(
            "(" + ", ".join(_emit_expr(v, d) for v in row) + ")"
            for row in node.values
        )
        return f"INSERT INTO {target}{cols} VALUES {rows}"
    return f"INSERT INTO {target}{cols}"


def _emit_merge(node: MergeNode, d: Dialect) -> str:
    target = _emit_from(node.target, d) if node.target else "?"
    source = _emit_from(node.source, d) if node.source else "?"
    on_sql = _emit_expr(node.on, d) if node.on else "?"
    def _action(a: dict) -> str:
        if a["type"] == "DELETE": return "DELETE"
        if a["type"] == "UPDATE":
            return "UPDATE SET " + ", ".join(f"{_quote_ident(k, d)} = {_emit_expr(v, d)}" for k, v in a["set"].items())
        if a["type"] == "INSERT":
            cols = " (" + ", ".join(_quote_ident(c, d) for c in a["columns"]) + ")" if a.get("columns") else ""
            return f"INSERT{cols} VALUES ({', '.join(_emit_expr(v, d) for v in a['values'])})"
        return a["type"]

    parts = [f"MERGE INTO {target} USING {source} ON {on_sql}"]
    for label, clauses in [("WHEN MATCHED", node.when_matched), ("WHEN NOT MATCHED", node.when_not_matched)]:
        for wc in (clauses or []):
            cond = f" AND {_emit_expr(wc['condition'], d)}" if wc.get("condition") else ""
            parts.append(f"{label}{cond} THEN {_action(wc['action'])}")
    return " ".join(parts)
