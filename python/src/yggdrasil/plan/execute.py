"""Execute a plan-node tree against concrete Tabular sources.

Pushes predicates and row_limit into CastOptions for I/O-level
optimization (partition pruning, row-group skipping). Uses Tabular
interface methods (.filter/.select/.unique) which dispatch to the
correct engine (Spark DataFrame API or Arrow C++ kernels).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from .nodes import InsertNode, MergeNode, PlanNode, ScanNode, SelectNode
from .ops import JoinClause, SubqueryRef, TableRef

if TYPE_CHECKING:
    from yggdrasil.io.tabular import Tabular

logger = logging.getLogger(__name__)

_AGG_MAP = {"COUNT": "count", "SUM": "sum", "AVG": "mean", "MIN": "min", "MAX": "max"}
_AGG_NAMES = frozenset(_AGG_MAP) | {"MEAN", "COLLECT_LIST", "COLLECT_SET",
                                     "STDDEV", "VARIANCE", "APPROX_COUNT_DISTINCT"}


def execute_plan(node: PlanNode, tables: dict[str, "Tabular"] | None = None) -> "Tabular":
    return _exec(node, dict(tables or {}))


def _exec(node: PlanNode, tables: dict[str, "Tabular"]) -> "Tabular":
    if isinstance(node, SelectNode):
        return _exec_select(node, tables)
    if isinstance(node, ScanNode):
        if node.tabular is not None:
            return node.tabular
        if node.name and node.name in tables:
            return tables[node.name]
        raise ValueError(f"Table {node.name!r} not found. Available: {sorted(tables)}")
    if isinstance(node, InsertNode):
        return _exec_insert(node, tables)
    raise NotImplementedError(f"Cannot execute {type(node).__name__}")


def _resolve_from(item: Any, tables: dict[str, "Tabular"]) -> "Tabular":
    if isinstance(item, TableRef):
        if item.name in tables:
            return tables[item.name]
        raise ValueError(f"Table {item.name!r} not found. Available: {sorted(tables)}")
    if isinstance(item, SubqueryRef):
        return _exec(item.plan, tables)
    if isinstance(item, JoinClause):
        return _exec_join(item, tables)
    if isinstance(item, PlanNode):
        return _exec(item, tables)
    raise TypeError(f"Cannot resolve FROM item: {type(item).__name__}")


def _exec_select(node: SelectNode, tables: dict[str, "Tabular"]) -> "Tabular":
    from yggdrasil.arrow.tabular import ArrowTabular
    from yggdrasil.execution.expr.nodes import Alias, Column, FunctionCall, Literal, Star

    # CTEs
    if node.ctes:
        for cte in node.ctes:
            tables[cte.name] = _exec(cte.plan, tables)

    # FROM
    result: "Tabular | None" = _resolve_from(node.from_clause, tables) if node.from_clause else None
    if result is None:
        row = {}
        for p in (node.projections or []):
            if isinstance(p, Alias):
                row[p.name] = p.expr.value if isinstance(p.expr, Literal) else None
            elif isinstance(p, Column):
                row[p.alias or p.name] = None
            else:
                row[str(p)] = p.value if isinstance(p, Literal) else None
        return ArrowTabular(pa.table({k: [v] for k, v in row.items()}) if row else pa.table({}))

    # Spark SQL passthrough for GROUP BY / SET ops
    spark_frame = getattr(result, "_native_spark_frame", lambda: None)()
    if spark_frame is not None and (node.group_by is not None or node.set_ops):
        try:
            from yggdrasil.plan.sql_emitter import emit_sql
            from yggdrasil.enums import Dialect
            from yggdrasil.spark.tabular import SparkDataset
            return SparkDataset(frame=spark_frame.sparkSession.sql(
                emit_sql(node, dialect=Dialect.DATABRICKS),
            ))
        except Exception:
            pass

    # WHERE — push predicate into CastOptions for I/O-level optimization,
    # or use Tabular.filter() for Spark/Arrow engine-native dispatch.
    if node.where is not None:
        from yggdrasil.execution.expr import Predicate
        if isinstance(node.where, Predicate) and not isinstance(result, ArrowTabular) and spark_frame is None:
            try:
                opts = result.check_options(None, overrides={"predicate": node.where})
                result = result.read_arrow_tabular(opts)
            except Exception:
                result = result.filter(node.where)
        else:
            result = result.filter(node.where)

    # GROUP BY + aggregation
    agg_alias_map: dict[str, str] | None = None
    has_group = node.group_by is not None
    has_agg = has_group or any(
        isinstance(p.expr if isinstance(p, Alias) else p, FunctionCall)
        and (p.expr if isinstance(p, Alias) else p).name.upper() in _AGG_NAMES
        for p in (node.projections or [])
    )
    if has_agg:
        result, agg_alias_map = _exec_group_by(result, node)

    # HAVING
    if node.having is not None and agg_alias_map:
        rewritten = _rewrite_having(node.having, result, agg_alias_map)
        if rewritten is not None:
            result = result.filter(rewritten)

    # Projection — use Tabular.select() (routes to Spark/Arrow natively)
    if node.projections and not has_group:
        cols = [
            (p.expr if isinstance(p, Alias) else p).name
            for p in node.projections
            if isinstance(p.expr if isinstance(p, Alias) else p, Column)
        ]
        if cols and len(cols) == len(node.projections):
            result = result.select(*cols)

    # DISTINCT
    if node.distinct:
        sf = getattr(result, "_native_spark_frame", lambda: None)()
        if sf is not None:
            from yggdrasil.spark.tabular import SparkDataset
            result = SparkDataset(frame=sf.distinct())
        else:
            try:
                col_names = result.collect_schema().names
            except Exception:
                col_names = result.read_arrow_table().column_names
            if col_names:
                result = result.unique(col_names)

    # ORDER BY
    if node.order_by:
        from yggdrasil.execution.expr.nodes import SortOrder
        sort_keys = [
            (item.expr.name, "ascending" if item.ascending else "descending")
            for item in node.order_by
            if isinstance(item, SortOrder) and isinstance(item.expr, Column)
        ]
        if not sort_keys:
            sort_keys = [
                (item["expr"].name, "ascending" if item.get("ascending", True) else "descending")
                for item in node.order_by
                if isinstance(item, dict) and isinstance(item.get("expr"), Column)
            ]
        if sort_keys:
            sf = getattr(result, "_native_spark_frame", lambda: None)()
            if sf is not None:
                try:
                    from pyspark.sql.functions import asc, desc
                    from yggdrasil.spark.tabular import SparkDataset
                    result = SparkDataset(frame=sf.orderBy(*[
                        desc(c) if d == "descending" else asc(c) for c, d in sort_keys
                    ]))
                except Exception:
                    import pyarrow.compute as pc
                    table = result.read_arrow_table()
                    result = ArrowTabular(table.take(pc.sort_indices(table, sort_keys=sort_keys)))
            else:
                import pyarrow.compute as pc
                table = result.read_arrow_table()
                result = ArrowTabular(table.take(pc.sort_indices(table, sort_keys=sort_keys)))

    # LIMIT / OFFSET
    if node.limit is not None:
        offset = node.offset or 0
        sf = getattr(result, "_native_spark_frame", lambda: None)()
        if sf is not None and offset == 0:
            from yggdrasil.spark.tabular import SparkDataset
            result = SparkDataset(frame=sf.limit(node.limit))
        else:
            table = result.read_arrow_table()
            result = ArrowTabular(table.slice(offset, node.limit) if offset else
                                  table.slice(0, node.limit) if table.num_rows > node.limit else table)

    # SET ops (UNION / INTERSECT / EXCEPT)
    if node.set_ops:
        from yggdrasil.enums import Mode
        for sop in node.set_ops:
            result = result.union(_exec(sop.plan, tables), mode=Mode.IGNORE)

    return result


def _exec_join(jc: JoinClause, tables: dict[str, "Tabular"]) -> "Tabular":
    from yggdrasil.arrow.tabular import ArrowTabular
    from yggdrasil.execution.expr.nodes import Column, Comparison, Logical
    from yggdrasil.execution.expr.operators import CompareOp, LogicalOp

    left_table = _resolve_from(jc.left, tables).read_arrow_table()
    right_table = _resolve_from(jc.right, tables).read_arrow_table()
    left_cols, right_cols = set(left_table.column_names), set(right_table.column_names)

    left_keys: list[str] = []
    right_keys: list[str] = []
    if jc.on is not None:
        comparisons = ([jc.on] if isinstance(jc.on, Comparison) and jc.on.op == CompareOp.EQ
                       else [o for o in getattr(jc.on, "operands", ())
                             if isinstance(o, Comparison) and o.op == CompareOp.EQ])
        for cmp in comparisons:
            ln = cmp.left.name if isinstance(cmp.left, Column) else None
            rn = cmp.right.name if isinstance(cmp.right, Column) else None
            if ln and rn:
                if ln in left_cols and rn in right_cols:
                    left_keys.append(ln); right_keys.append(rn)
                elif rn in left_cols and ln in right_cols:
                    left_keys.append(rn); right_keys.append(ln)
                elif ln in left_cols and ln in right_cols:
                    left_keys.append(ln); right_keys.append(ln)

    if left_keys and right_keys:
        kw = {"keys": left_keys, "join_type": jc.join_type.arrow}
        if left_keys != right_keys:
            kw["right_keys"] = right_keys
        return ArrowTabular(left_table.join(right_table, **kw))

    common = [c for c in left_table.column_names if c in right_cols]
    return ArrowTabular(left_table.join(
        right_table, keys=common[:1] or left_table.column_names[:1],
        join_type=jc.join_type.arrow,
    ))


def _exec_group_by(result: "Tabular", node: SelectNode) -> "tuple[Tabular, dict | None]":
    from yggdrasil.arrow.tabular import ArrowTabular
    from yggdrasil.execution.expr.nodes import Alias, Column, FunctionCall, Star
    import pyarrow.compute as pc

    table = result.read_arrow_table()
    group_keys = [g.name for g in (node.group_by or []) if isinstance(g, Column)]
    agg_specs: list[tuple[str, str, str]] = []
    agg_alias_map: dict[str, str] = {}

    for proj in (node.projections or []):
        expr, out_name = (proj.expr, proj.name) if isinstance(proj, Alias) else (proj, None)
        if isinstance(expr, FunctionCall) and expr.name.upper() in _AGG_MAP:
            col_name = (expr.args[0].name if expr.args and isinstance(expr.args[0], Column)
                        else group_keys[0] if group_keys else table.column_names[0])
            pa_func = _AGG_MAP[expr.name.upper()]
            target = out_name or f"{expr.name.lower()}_{col_name}"
            agg_specs.append((col_name, pa_func, target))
            agg_alias_map[f"{expr.name.upper()}:{col_name}"] = target

    if not agg_specs:
        return result, None

    resolved = [(gk[0] if gk[0] != "*" else (group_keys[0] if group_keys else table.column_names[0]),
                 gk[1]) for gk in [(c, f) for c, f, _ in agg_specs]]

    if group_keys:
        result_table = table.group_by(group_keys).aggregate(resolved)
        rename = {f"{c}_{f}": n for c, f, n in agg_specs
                  if f"{c}_{f}" in result_table.column_names and f"{c}_{f}" != n}
        if rename:
            result_table = result_table.rename_columns([rename.get(c, c) for c in result_table.column_names])
        return ArrowTabular(result_table), agg_alias_map
    else:
        row: dict[str, list] = {}
        for (col_name, func), (_, _, out_name) in zip(resolved, agg_specs):
            arr = table.column(col_name)
            compute = {"count": lambda a: len(a) - a.null_count, "sum": lambda a: pc.sum(a).as_py(),
                       "mean": lambda a: pc.mean(a).as_py(), "min": lambda a: pc.min(a).as_py(),
                       "max": lambda a: pc.max(a).as_py()}
            row[out_name] = [compute.get(func, lambda a: None)(arr)]
        return ArrowTabular(pa.table(row)), agg_alias_map


def _rewrite_having(having: Any, result: "Tabular", agg_alias_map: dict) -> Any:
    from yggdrasil.execution.expr.nodes import Column, Comparison, FunctionCall, Logical, Not, Star
    from yggdrasil.execution.expr import walk

    try:
        col_names = set(result.collect_schema().names)
    except Exception:
        col_names = set(result.read_arrow_table().column_names)

    pa_map = {"count": "count", "sum": "sum", "avg": "mean", "min": "min", "max": "max", "mean": "mean"}

    def _rw(expr: Any) -> Any:
        if isinstance(expr, FunctionCall):
            fu = expr.name.upper()
            if expr.args and isinstance(expr.args[0], Column):
                key = f"{fu}:{expr.args[0].name}"
                if key in agg_alias_map and agg_alias_map[key] in col_names:
                    return Column(name=agg_alias_map[key])
            elif expr.args and isinstance(expr.args[0], Star):
                for k, v in agg_alias_map.items():
                    if k.startswith(f"{fu}:") and v in col_names:
                        return Column(name=v)
            for k, v in agg_alias_map.items():
                if k.startswith(f"{fu}:") and v in col_names:
                    return Column(name=v)
            return expr
        if isinstance(expr, Comparison):
            return Comparison(_rw(expr.left), expr.op, _rw(expr.right))
        if isinstance(expr, Logical):
            return Logical(expr.op, tuple(_rw(o) for o in expr.operands))
        if isinstance(expr, Not):
            return Not(_rw(expr.operand))
        return expr

    rewritten = _rw(having)
    return None if any(isinstance(n, FunctionCall) for n in walk(rewritten)) else rewritten


def _exec_insert(node: InsertNode, tables: dict[str, "Tabular"]) -> "Tabular":
    from yggdrasil.arrow.tabular import ArrowTabular
    from yggdrasil.execution.expr.nodes import Literal

    if not node.target or node.target.name not in tables:
        raise ValueError(f"Target table {getattr(node.target, 'name', None)!r} not found")
    target = tables[node.target.name]
    if node.source:
        target.write_table(_exec(node.source, tables))
    elif node.values:
        cols = node.columns or [f"col{i}" for i in range(len(node.values[0]))]
        pydict = {c: [r[i].value if isinstance(r[i], Literal) else r[i] if i < len(r) else None
                       for r in node.values] for i, c in enumerate(cols)}
        target.write_table(pa.table(pydict))
    return target
