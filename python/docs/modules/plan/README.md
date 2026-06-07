# Execution Plans

SQL parsing, lazy execution, and Arrow-native UDF execution for `Tabular`
data — `yggdrasil.plan`.

For the full walkthrough (SQL features, joins, CTEs, UDF registration), see
the [Execution Plans & SQL guide](../../guides/plan.md). Full signatures live
in the auto-generated [API reference](../../reference/yggdrasil/plan/).

```python
from yggdrasil.plan import parse_sql
from yggdrasil.arrow.tabular import ArrowTabular
import pyarrow as pa

users = ArrowTabular(pa.table({
    "id": [1, 2, 3],
    "name": ["alice", "bob", "carol"],
    "score": [90, 80, 95],
}))

node = parse_sql("SELECT name, score FROM users WHERE score > 80 ORDER BY score DESC")
result = node.execute(tables={"users": users})
print(result.read_arrow_table().to_pylist())
# [{'name': 'carol', 'score': 95}, {'name': 'alice', 'score': 90}]
```

## Surface

| Symbol | Role |
|---|---|
| [`parse_sql`][yggdrasil.plan.parse_sql] | Parse a SQL string into an executable plan node |
| [`SQLQueryParser`][yggdrasil.plan.sql_parser.SQLQueryParser] | The underlying parser |
| [`ExecutionPlan`][yggdrasil.plan.ExecutionPlan] / [`SelectPlan`][yggdrasil.plan.SelectPlan] | Plan objects |
| [`LazyTabular`][yggdrasil.plan.LazyTabular] | Deferred, lazily-evaluated `Tabular` |
| [`PlanNode`][yggdrasil.plan.nodes.PlanNode] | Base node — [`SelectNode`][yggdrasil.plan.nodes.SelectNode], [`InsertNode`][yggdrasil.plan.nodes.InsertNode], [`MergeNode`][yggdrasil.plan.nodes.MergeNode], [`ScanNode`][yggdrasil.plan.nodes.ScanNode] |
| [`FunctionRegistry`][yggdrasil.plan.func_registry.FunctionRegistry] / `FunctionMeta` | UDF registry + metadata |
| `BUILTIN_REGISTRY` | Built-in functions (incl. `explode_table`, `posexplode_table`) |
| `yggdrasil.plan.ops` | Relational ops — joins, group-by, order-by, union, CTEs |
