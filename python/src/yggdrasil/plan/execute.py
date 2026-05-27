"""Execute a plan-node tree against concrete Tabular sources.

Dispatches Spark-native operations when the source is a SparkDataset,
otherwise falls back to Arrow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa

from .nodes import InsertNode, MergeNode, PlanNode, ScanNode, SelectNode
from .ops import JoinClause, LateralViewItem, SubqueryRef, TableRef

if TYPE_CHECKING:
    from yggdrasil.io.tabular import Tabular


def execute_plan(
    node: PlanNode,
    tables: dict[str, "Tabular"] | None = None,
) -> "Tabular":
    ctx = _Context(tables or {})
    return ctx.execute(node)


class _Context:
    __slots__ = ("tables",)

    def __init__(self, tables: dict[str, "Tabular"]) -> None:
        self.tables = dict(tables)

    def execute(self, node: PlanNode) -> "Tabular":
        if isinstance(node, SelectNode):
            return self._exec_select(node)
        if isinstance(node, ScanNode):
            return self._exec_scan(node)
        if isinstance(node, InsertNode):
            return self._exec_insert(node)
        raise NotImplementedError(f"Cannot execute {type(node).__name__}")

    def _exec_scan(self, node: ScanNode) -> "Tabular":
        if node.tabular is not None:
            return node.tabular
        name = node.name
        if name and name in self.tables:
            return self.tables[name]
        raise ValueError(
            f"Table {name!r} not found. Available: {sorted(self.tables)}"
        )

    def _exec_select(self, node: SelectNode) -> "Tabular":
        from yggdrasil.arrow.tabular import ArrowTabular
        from yggdrasil.execution.expr.nodes import (
            Alias,
            Column,
            FunctionCall,
            Star,
        )

        # Materialize CTEs
        if node.ctes:
            for cte in node.ctes:
                self.tables[cte.name] = self.execute(cte.plan)

        # FROM clause
        result: "Tabular | None" = None
        if node.from_clause is not None:
            result = self._exec_from(node.from_clause)

        if result is None:
            if node.projections:
                row = {}
                for proj in node.projections:
                    if isinstance(proj, Alias):
                        row[proj.name] = self._eval_literal(proj.expr)
                    elif isinstance(proj, Column):
                        row[proj.alias or proj.name] = None
                    else:
                        row[str(proj)] = self._eval_literal(proj)
                return ArrowTabular(pa.table(
                    {k: [v] for k, v in row.items()}
                ))
            return ArrowTabular(pa.table({}))

        # Spark SQL passthrough for complex plans
        spark_frame = result._native_spark_frame() if hasattr(result, '_native_spark_frame') else None
        if spark_frame is not None and self._can_spark_passthrough(node):
            return self._exec_spark_passthrough(node, result)

        # WHERE
        if node.where is not None:
            result = result.filter(node.where)

        # GROUP BY + aggregation
        agg_alias_map: dict | None = None
        if node.group_by is not None:
            result, agg_alias_map = self._exec_group_by(result, node)
        elif node.projections:
            has_agg = any(self._is_aggregate(p) for p in node.projections)
            if has_agg and not node.group_by:
                result, agg_alias_map = self._exec_group_by(result, node)

        # HAVING — rewrite aggregate references to materialized column names
        if node.having is not None:
            rewritten = self._rewrite_having(node.having, result, agg_alias_map)
            if rewritten is not None:
                result = result.filter(rewritten)

        # Projection (SELECT list) — only column selection for now
        if node.projections and not node.group_by:
            cols = self._resolve_projection_columns(node.projections, result)
            if cols:
                result = result.select(*cols)

        # DISTINCT
        if node.distinct:
            table = result.read_arrow_table()
            col_names = table.column_names
            if col_names:
                result = result.unique(col_names)

        # ORDER BY
        if node.order_by:
            result = self._exec_order_by(result, node.order_by)

        # LIMIT
        if node.limit is not None:
            table = result.read_arrow_table()
            offset = node.offset or 0
            if offset > 0:
                table = table.slice(offset, node.limit)
            elif table.num_rows > node.limit:
                table = table.slice(0, node.limit)
            result = ArrowTabular(table)

        # Set operations
        if node.set_ops:
            for sop in node.set_ops:
                right = self.execute(sop.plan)
                from yggdrasil.enums import Mode
                result = result.union(right, mode=Mode.IGNORE)

        return result

    def _exec_from(self, item: Any) -> "Tabular":
        if isinstance(item, TableRef):
            name = item.name
            if name in self.tables:
                return self.tables[name]
            raise ValueError(
                f"Table {name!r} not found. Available: {sorted(self.tables)}"
            )
        if isinstance(item, SubqueryRef):
            return self.execute(item.plan)
        if isinstance(item, JoinClause):
            return self._exec_join(item)
        if isinstance(item, PlanNode):
            return self.execute(item)
        raise TypeError(f"Cannot resolve FROM item: {type(item).__name__}")

    def _exec_join(self, jc: JoinClause) -> "Tabular":
        from yggdrasil.arrow.tabular import ArrowTabular
        from yggdrasil.execution.expr.nodes import Column, Comparison

        left = self._exec_from(jc.left)
        right = self._exec_from(jc.right)
        left_table = left.read_arrow_table()
        right_table = right.read_arrow_table()

        left_cols = set(left_table.column_names)
        right_cols = set(right_table.column_names)

        if jc.on is not None:
            left_keys, right_keys = self._extract_join_keys(
                jc.on, left_cols, right_cols,
            )
            if left_keys and right_keys and left_keys != right_keys:
                result_table = left_table.join(
                    right_table,
                    keys=left_keys,
                    right_keys=right_keys,
                    join_type=jc.join_type.arrow,
                )
                return ArrowTabular(result_table)
            if left_keys:
                result_table = left_table.join(
                    right_table,
                    keys=left_keys,
                    join_type=jc.join_type.arrow,
                )
                return ArrowTabular(result_table)

        common = [c for c in left_table.column_names if c in right_cols]
        keys = common[:1] if common else left_table.column_names[:1]
        result_table = left_table.join(
            right_table, keys=keys, join_type=jc.join_type.arrow,
        )
        return ArrowTabular(result_table)

    @staticmethod
    def _extract_join_keys(
        on: Any, left_cols: set[str], right_cols: set[str],
    ) -> tuple[list[str], list[str]]:
        """Extract (left_keys, right_keys) from a join ON predicate."""
        from yggdrasil.execution.expr.nodes import Column, Comparison, Logical
        from yggdrasil.execution.expr.operators import CompareOp, LogicalOp

        comparisons = []
        if isinstance(on, Comparison) and on.op == CompareOp.EQ:
            comparisons = [on]
        elif isinstance(on, Logical) and on.op == LogicalOp.AND:
            comparisons = [
                o for o in on.operands
                if isinstance(o, Comparison) and o.op == CompareOp.EQ
            ]

        left_keys: list[str] = []
        right_keys: list[str] = []
        for cmp in comparisons:
            l_name = cmp.left.name if isinstance(cmp.left, Column) else None
            r_name = cmp.right.name if isinstance(cmp.right, Column) else None
            if l_name and r_name:
                if l_name in left_cols and r_name in right_cols:
                    left_keys.append(l_name)
                    right_keys.append(r_name)
                elif r_name in left_cols and l_name in right_cols:
                    left_keys.append(r_name)
                    right_keys.append(l_name)
                elif l_name in left_cols and l_name in right_cols:
                    left_keys.append(l_name)
                    right_keys.append(l_name)
        return left_keys, right_keys

    def _exec_group_by(
        self, result: "Tabular", node: SelectNode,
    ) -> "tuple[Tabular, dict | None]":
        from yggdrasil.arrow.tabular import ArrowTabular
        from yggdrasil.execution.expr.nodes import (
            Alias,
            Column,
            FunctionCall,
            Star,
        )

        table = result.read_arrow_table()
        import pyarrow.compute as pc

        group_keys: list[str] = []
        if node.group_by:
            for g in node.group_by:
                if isinstance(g, Column):
                    group_keys.append(g.name)
                elif isinstance(g, str):
                    group_keys.append(g)

        agg_specs: list[tuple[str, str, str]] = []
        agg_alias_map: dict[str, str] = {}
        output_names: list[str] = []

        for proj in (node.projections or []):
            expr = proj
            out_name = None
            if isinstance(proj, Alias):
                out_name = proj.name
                expr = proj.expr

            if isinstance(expr, Column) and expr.name in group_keys:
                output_names.append(out_name or expr.name)
                continue

            if isinstance(expr, FunctionCall):
                func = expr.name.upper()
                if expr.args and isinstance(expr.args[0], (Column, Star)):
                    col_name = expr.args[0].name if isinstance(expr.args[0], Column) else group_keys[0] if group_keys else table.column_names[0]
                    agg_map = {
                        "COUNT": "count", "SUM": "sum", "AVG": "mean",
                        "MIN": "min", "MAX": "max",
                    }
                    if func in agg_map:
                        target_name = out_name or f"{func.lower()}_{col_name}"
                        agg_specs.append((col_name, agg_map[func], target_name))
                        output_names.append(target_name)
                        agg_key = f"{func}:{col_name}"
                        agg_alias_map[agg_key] = target_name
                        continue
            if isinstance(expr, Column):
                output_names.append(out_name or expr.name)

        if agg_specs:
            resolved_agg: list[tuple[str, str]] = []
            for col_name, func, name in agg_specs:
                if col_name == "*" and func == "count":
                    col_name = group_keys[0] if group_keys else table.column_names[0]
                resolved_agg.append((col_name, func))

            if group_keys:
                result_table = table.group_by(group_keys).aggregate(resolved_agg)
            else:
                import pyarrow.compute as pc
                row: dict[str, list] = {}
                for (col_name, func), (_, _, out_name) in zip(resolved_agg, agg_specs):
                    arr = table.column(col_name)
                    if func == "count":
                        row[out_name] = [len(arr) - arr.null_count]
                    elif func == "sum":
                        row[out_name] = [pc.sum(arr).as_py()]
                    elif func == "mean":
                        row[out_name] = [pc.mean(arr).as_py()]
                    elif func == "min":
                        row[out_name] = [pc.min(arr).as_py()]
                    elif func == "max":
                        row[out_name] = [pc.max(arr).as_py()]
                    else:
                        row[out_name] = [None]
                return ArrowTabular(pa.table(row)), agg_alias_map or None

            rename_map = {}
            for col_name, func, name in agg_specs:
                if col_name == "*" and func == "count":
                    col_name = group_keys[0] if group_keys else table.column_names[0]
                src = f"{col_name}_{func}"
                if src in result_table.column_names and src != name:
                    rename_map[src] = name
            if rename_map:
                result_table = result_table.rename_columns([
                    rename_map.get(c, c) for c in result_table.column_names
                ])
            return ArrowTabular(result_table), agg_alias_map or None

        return result, None

    def _exec_order_by(self, result: "Tabular", order_by: list) -> "Tabular":
        from yggdrasil.arrow.tabular import ArrowTabular
        from yggdrasil.execution.expr.nodes import Column, SortOrder
        import pyarrow.compute as pc

        table = result.read_arrow_table()
        sort_keys = []
        for item in order_by:
            if isinstance(item, SortOrder):
                if isinstance(item.expr, Column):
                    order = "ascending" if item.ascending else "descending"
                    sort_keys.append((item.expr.name, order))
            elif isinstance(item, dict):
                expr = item.get("expr")
                if isinstance(expr, Column):
                    order = "ascending" if item.get("ascending", True) else "descending"
                    sort_keys.append((expr.name, order))

        if sort_keys:
            indices = pc.sort_indices(table, sort_keys=sort_keys)
            table = table.take(indices)
        return ArrowTabular(table)

    def _exec_insert(self, node: InsertNode) -> "Tabular":
        from yggdrasil.arrow.tabular import ArrowTabular
        from yggdrasil.execution.expr.nodes import Literal

        if node.target is None:
            raise ValueError("INSERT requires a target table")
        target_name = node.target.name
        if target_name not in self.tables:
            raise ValueError(f"Target table {target_name!r} not found")
        target = self.tables[target_name]
        if node.source:
            source_data = self.execute(node.source)
            target.write_table(source_data)
        elif node.values:
            columns = node.columns
            if not columns:
                columns = [f"col{i}" for i in range(len(node.values[0]))]
            pydict: dict[str, list] = {c: [] for c in columns}
            for row in node.values:
                for i, c in enumerate(columns):
                    val = row[i] if i < len(row) else None
                    if isinstance(val, Literal):
                        val = val.value
                    pydict[c].append(val)
            target.write_table(pa.table(pydict))
        return target

    def _resolve_projection_columns(
        self, projections: list, result: "Tabular",
    ) -> list[str] | None:
        from yggdrasil.execution.expr.nodes import Alias, Column, Star

        cols: list[str] = []
        for p in projections:
            if isinstance(p, Star):
                return None
            if isinstance(p, Alias):
                p = p.expr
            if isinstance(p, Column):
                cols.append(p.name)
            else:
                return None
        return cols or None

    def _is_aggregate(self, expr: Any) -> bool:
        from yggdrasil.execution.expr.nodes import Alias, FunctionCall

        if isinstance(expr, Alias):
            expr = expr.expr
        if isinstance(expr, FunctionCall):
            return expr.name.upper() in (
                "COUNT", "SUM", "AVG", "MIN", "MAX", "MEAN",
                "COLLECT_LIST", "COLLECT_SET",
                "STDDEV", "VARIANCE", "APPROX_COUNT_DISTINCT",
            )
        return False

    def _eval_literal(self, expr: Any) -> Any:
        from yggdrasil.execution.expr.nodes import Literal
        if isinstance(expr, Literal):
            return expr.value
        return None

    def _rewrite_having(
        self, having: Any, result: "Tabular",
        agg_alias_map: dict | None = None,
    ) -> Any:
        """Rewrite HAVING predicate: replace aggregate FunctionCalls with
        column references pointing to materialized aggregate column names."""
        from yggdrasil.execution.expr.nodes import (
            Column, Comparison, FunctionCall, Logical, Not, Star,
        )

        col_names = set(result.read_arrow_table().column_names)
        alias_map = agg_alias_map or {}
        agg_pyarrow = {
            "count": "count", "sum": "sum", "avg": "mean",
            "min": "min", "max": "max", "mean": "mean",
        }

        def _rewrite(expr: Any) -> Any:
            if isinstance(expr, FunctionCall):
                func_upper = expr.name.upper()
                func_lower = expr.name.lower()
                pa_name = agg_pyarrow.get(func_lower, func_lower)
                # Check alias map first (from the GROUP BY projection aliases)
                if expr.args and isinstance(expr.args[0], Column):
                    key = f"{func_upper}:{expr.args[0].name}"
                    if key in alias_map and alias_map[key] in col_names:
                        return Column(name=alias_map[key])
                    candidate = f"{expr.args[0].name}_{pa_name}"
                    if candidate in col_names:
                        return Column(name=candidate)
                elif expr.args and isinstance(expr.args[0], Star):
                    # COUNT(*) → try alias map with wildcard keys
                    for k, v in alias_map.items():
                        if k.startswith(f"{func_upper}:") and v in col_names:
                            return Column(name=v)
                    for cn in col_names:
                        if cn.endswith(f"_{pa_name}"):
                            return Column(name=cn)
                if not expr.args:
                    for k, v in alias_map.items():
                        if k.startswith(f"{func_upper}:") and v in col_names:
                            return Column(name=v)
                return expr
            if isinstance(expr, Comparison):
                return Comparison(_rewrite(expr.left), expr.op, _rewrite(expr.right))
            if isinstance(expr, Logical):
                return Logical(expr.op, tuple(_rewrite(o) for o in expr.operands))
            if isinstance(expr, Not):
                return Not(_rewrite(expr.operand))
            return expr

        rewritten = _rewrite(having)
        has_func = False
        from yggdrasil.execution.expr import walk
        for node in walk(rewritten):
            if isinstance(node, FunctionCall):
                has_func = True
                break
        if has_func:
            return None
        return rewritten

    def _can_spark_passthrough(self, node: SelectNode) -> bool:
        return node.group_by is not None or bool(node.set_ops)

    def _exec_spark_passthrough(
        self, node: SelectNode, source: "Tabular",
    ) -> "Tabular":
        from yggdrasil.plan.sql_emitter import emit_sql
        from yggdrasil.enums import Dialect

        sql = emit_sql(node, dialect=Dialect.DATABRICKS)
        spark_frame = source._native_spark_frame()
        if spark_frame is not None:
            spark = spark_frame.sparkSession
            result_frame = spark.sql(sql)
            from yggdrasil.spark.tabular import SparkDataset
            return SparkDataset(frame=result_frame)
        return source
