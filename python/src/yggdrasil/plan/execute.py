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


def execute_plan(node: PlanNode, tables: "dict[str, Tabular] | None" = None) -> "Tabular":
    return _exec(node, dict(tables) if tables else {})


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


def _resolve_from(item: Any, tables: dict[str, "Tabular"], predicate: Any = None) -> "Tabular":
    if isinstance(item, TableRef):
        name = item.name
        # Qualified names: try catalog.schema.table, schema.table, table
        full = ".".join(filter(None, [item.catalog, item.schema, name]))
        for key in (full, f"{item.schema}.{name}" if item.schema else None, name):
            if key and key in tables:
                result = tables[key]
                # Push predicate into I/O-backed sources
                if predicate is not None and not _is_in_memory(result):
                    try:
                        opts = result.check_options(None, overrides={"predicate": predicate})
                        return result.read_arrow_tabular(opts)
                    except Exception:
                        pass
                return result
        # Auto-resolve URLs/paths as tabular sources
        from yggdrasil.io.tabular.base import is_tabular_source, Tabular as _Tab
        if is_tabular_source(name) or is_tabular_source(full):
            resolved = _Tab.from_(full if is_tabular_source(full) else name, default=None)
            if resolved is not None:
                tables[name] = resolved
                return resolved
        raise ValueError(f"Table {name!r} not found. Available: {sorted(tables)}")
    if isinstance(item, SubqueryRef):
        return _exec(item.plan, tables)
    if isinstance(item, JoinClause):
        return _exec_join(item, tables)
    if isinstance(item, PlanNode):
        return _exec(item, tables)
    raise TypeError(f"Cannot resolve FROM item: {type(item).__name__}")


def _is_in_memory(t: "Tabular") -> bool:
    from yggdrasil.arrow.tabular import ArrowTabular
    return isinstance(t, ArrowTabular)


def _exec_select(node: SelectNode, tables: dict[str, "Tabular"]) -> "Tabular":
    from yggdrasil.arrow.tabular import ArrowTabular
    from yggdrasil.execution.expr.nodes import Alias, Column, FunctionCall, Literal, Star

    # CTEs
    if node.ctes:
        for cte in node.ctes:
            tables[cte.name] = _exec(cte.plan, tables)

    # FROM — push predicate to source when no joins (enables partition pruning)
    pushable_pred = node.where if (node.where and not isinstance(node.from_clause, JoinClause)) else None
    result: "Tabular | None" = _resolve_from(node.from_clause, tables, predicate=pushable_pred) if node.from_clause else None
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

    # LATERAL VIEW EXPLODE / POSEXPLODE — apply before WHERE so the
    # exploded column can be referenced in the predicate.
    if getattr(node, "lateral_views", None):
        result = _apply_lateral_views(result, node.lateral_views)

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

    # QUALIFY (Databricks window-function filter) — runs before projection so
    # it can reference columns that aren't in the final SELECT list.
    if getattr(node, "qualify", None) is not None:
        result = _apply_qualify(result, node.qualify)

    # Projection — apply scalar functions via Arrow kernels, then select
    if node.projections and not has_agg:
        result = _apply_projections(result, node.projections)

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
        # Auto-cast mismatched join key types
        for lk, rk in zip(left_keys, right_keys):
            lt, rt = left_table.schema.field(lk).type, right_table.schema.field(rk).type
            if lt != rt:
                try:
                    target = pa.lib.unify_schemas([
                        pa.schema([(lk, lt)]), pa.schema([(rk, rt)])
                    ]).field(0).type
                except Exception:
                    target = pa.utf8()
                if left_table.schema.field(lk).type != target:
                    left_table = left_table.set_column(
                        left_table.column_names.index(lk), lk,
                        left_table.column(lk).cast(target))
                if right_table.schema.field(rk).type != target:
                    right_table = right_table.set_column(
                        right_table.column_names.index(rk), rk,
                        right_table.column(rk).cast(target))
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


def _apply_projections(result: "Tabular", projections: list) -> "Tabular":
    """Apply SELECT projections: columns pass through, FunctionCall nodes
    execute via Arrow kernels from the function registry."""
    from yggdrasil.arrow.tabular import ArrowTabular
    from yggdrasil.execution.expr.nodes import Alias, Column, FunctionCall, Literal, Star
    from .func_registry import BUILTIN_REGISTRY

    if any(isinstance(p.expr if isinstance(p, Alias) else p, Star) for p in projections):
        return result

    # Fast path: all simple columns
    simple_cols = []
    has_func = False
    for p in projections:
        expr = p.expr if isinstance(p, Alias) else p
        if isinstance(expr, Column):
            simple_cols.append(expr.name)
        elif isinstance(expr, FunctionCall):
            has_func = True
        else:
            return result

    if not has_func and simple_cols and len(simple_cols) == len(projections):
        return result.select(*simple_cols)

    # Slow path: materialize table and apply function kernels column by column
    table = result.read_arrow_table()
    out_arrays: list[pa.Array] = []
    out_names: list[str] = []
    for p in projections:
        out_name = p.name if isinstance(p, Alias) else None
        expr = p.expr if isinstance(p, Alias) else p

        if isinstance(expr, Column):
            out_names.append(out_name or expr.name)
            out_arrays.append(table.column(expr.name))
        elif isinstance(expr, FunctionCall):
            args = []
            for a in expr.args:
                if isinstance(a, Column):
                    args.append(table.column(a.name))
                elif isinstance(a, Literal):
                    args.append(a.value)
                else:
                    return result
            arr = BUILTIN_REGISTRY.apply_arrow(expr.name, *args)
            if arr is None:
                return result
            out_names.append(out_name or expr.name.lower())
            out_arrays.append(arr)
        elif isinstance(expr, Literal):
            out_names.append(out_name or str(expr.value))
            out_arrays.append(pa.array([expr.value] * table.num_rows))
        else:
            return result

    return ArrowTabular(pa.table(dict(zip(out_names, out_arrays))))


def _apply_lateral_views(result: "Tabular", lateral_views: list) -> "Tabular":
    """Apply LATERAL VIEW EXPLODE / POSEXPLODE clauses.

    Supports the Meteologica-style ``LATERAL VIEW EXPLODE(col) tbl AS x``
    and ``LATERAL VIEW POSEXPLODE(col) tbl AS pos, val`` patterns. For
    each view, the named list column is flattened, scalar columns are
    repeated per element, and the result is rebound to the column
    aliases.
    """
    from yggdrasil.arrow.tabular import ArrowTabular
    from yggdrasil.execution.expr.nodes import Column, FunctionCall
    from .func_registry import explode_table, posexplode_table

    table = result.read_arrow_table()
    for lv in lateral_views:
        func = lv.function
        if not isinstance(func, FunctionCall):
            continue
        fn = func.name.upper()
        if not func.args or not isinstance(func.args[0], Column):
            continue
        src_col = func.args[0].name
        if src_col not in table.column_names:
            continue
        if fn in ("EXPLODE", "EXPLODE_OUTER"):
            out_col = lv.column_aliases[0] if lv.column_aliases else src_col
            table = explode_table(table, src_col, out_col=out_col)
        elif fn in ("POSEXPLODE", "POSEXPLODE_OUTER"):
            pos_col = lv.column_aliases[0] if lv.column_aliases else "pos"
            val_col = lv.column_aliases[1] if len(lv.column_aliases) > 1 else src_col
            table = posexplode_table(table, src_col, pos_col=pos_col, out_col=val_col)
    return ArrowTabular(table)


def _apply_qualify(result: "Tabular", qualify: Any) -> "Tabular":
    """Execute a QUALIFY clause on the result.

    Currently handles the most common Meteologica-style pattern:
    ``QUALIFY ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...) = N``
    by computing row numbers per partition and filtering. Other window
    expressions fall back to a no-op so the query still completes.
    """
    from yggdrasil.arrow.tabular import ArrowTabular
    from yggdrasil.execution.expr.nodes import (
        Column, Comparison, FunctionCall, Literal, SortOrder, WindowFunction,
    )
    from yggdrasil.execution.expr.operators import CompareOp
    import pyarrow.compute as pc

    if not (isinstance(qualify, Comparison)
            and isinstance(qualify.left, WindowFunction)
            and isinstance(qualify.right, Literal)):
        return result  # fallback for unsupported QUALIFY shapes

    wf = qualify.left
    if not (isinstance(wf.function, FunctionCall)
            and wf.function.name.upper() in ("ROW_NUMBER", "RANK", "DENSE_RANK")):
        return result

    target_rank = qualify.right.value
    cmp_op = qualify.op
    win = wf.window
    pb_cols = [c.name for c in win.partition_by if isinstance(c, Column)]
    sort_keys: list[tuple[str, str]] = []
    for so in win.order_by:
        if isinstance(so, SortOrder) and isinstance(so.expr, Column):
            sort_keys.append(
                (so.expr.name, "ascending" if so.ascending else "descending"))

    table = result.read_arrow_table()
    if table.num_rows == 0 or table.num_columns == 0:
        return result

    # Partition keys first so equal partitions are contiguous; sort keys then
    # rank rows within each partition.
    full_sort = [(c, "ascending") for c in pb_cols] + sort_keys
    if full_sort:
        table = table.take(pc.sort_indices(table, sort_keys=full_sort))

    # Compute row numbers within partitions
    if pb_cols:
        partition_arrays = [table.column(c).to_pylist() for c in pb_cols]
        row_numbers = []
        last_key = object()
        n = 0
        for i in range(table.num_rows):
            key = tuple(arr[i] for arr in partition_arrays)
            if key != last_key:
                n = 1
                last_key = key
            else:
                n += 1
            row_numbers.append(n)
    else:
        row_numbers = list(range(1, table.num_rows + 1))

    rn_array = pa.array(row_numbers, type=pa.int64())
    rank_literal = pa.scalar(target_rank, type=pa.int64())
    cmp_map = {
        CompareOp.EQ: pc.equal,
        CompareOp.NE: pc.not_equal,
        CompareOp.LT: pc.less,
        CompareOp.LE: pc.less_equal,
        CompareOp.GT: pc.greater,
        CompareOp.GE: pc.greater_equal,
    }
    mask = cmp_map.get(cmp_op, pc.equal)(rn_array, rank_literal)
    filtered = table.filter(mask)
    return ArrowTabular(filtered)


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
