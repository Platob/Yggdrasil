# Saga — the data engine

Saga is yggdrasil's unified, autonomous, lazy data engine: schema-aware
expressions (`yggdrasil.saga.expr`), execution plans + SQL parse/emit/execute
(`yggdrasil.saga.plan`), and the `Saga` engine facade that ties them together
over a catalog of `Tabular` sources.

For the full walkthrough (SQL features, joins, CTEs, UDF registration), see
the [Saga Engine guide](../../guides/saga.md). Full signatures live in the
auto-generated [API reference](../../reference/yggdrasil/saga/).

```python
from yggdrasil.saga import Saga
from yggdrasil.arrow.tabular import ArrowTabular
import pyarrow as pa

users = ArrowTabular(pa.table({
    "id": [1, 2, 3],
    "name": ["alice", "bob", "carol"],
    "score": [90, 80, 95],
}))

saga = Saga(dialect="databricks").register("users", users)
result = saga.sql("SELECT name, score FROM users WHERE score > 80 ORDER BY score DESC")
print(result.read_arrow_table().to_pylist())
# [{'name': 'carol', 'score': 95}, {'name': 'alice', 'score': 90}]

# Or build a deferred pipeline that executes on the first read:
out = saga.scan("users").filter("score > 80").select("name", "score").read_arrow_table()
```

## Surface

| Symbol | Role |
|---|---|
| [`Saga`][yggdrasil.saga.Saga] | Engine facade — register tables, parse SQL, build/execute lazy plans |
| [`col`][yggdrasil.saga.col] / [`lit`][yggdrasil.saga.lit] / [`Predicate`][yggdrasil.saga.Predicate] | Expression / predicate AST (`yggdrasil.saga.expr`) |
| [`parse_sql`][yggdrasil.saga.plan.parse_sql] | Parse a SQL string into an executable plan node |
| [`SQLQueryParser`][yggdrasil.saga.plan.sql_parser.SQLQueryParser] | The underlying parser |
| [`ExecutionPlan`][yggdrasil.saga.plan.ExecutionPlan] / [`SelectPlan`][yggdrasil.saga.plan.SelectPlan] | Plan objects |
| [`LazyTabular`][yggdrasil.saga.plan.LazyTabular] | Deferred, lazily-evaluated `Tabular` |
| [`PlanNode`][yggdrasil.saga.plan.nodes.PlanNode] | Base node — [`SelectNode`][yggdrasil.saga.plan.nodes.SelectNode], [`InsertNode`][yggdrasil.saga.plan.nodes.InsertNode], [`MergeNode`][yggdrasil.saga.plan.nodes.MergeNode], [`ScanNode`][yggdrasil.saga.plan.nodes.ScanNode] |
| [`FunctionRegistry`][yggdrasil.saga.plan.func_registry.FunctionRegistry] / `FunctionMeta` | UDF registry + metadata |
| `BUILTIN_REGISTRY` | Built-in functions (incl. `explode_table`, `posexplode_table`) |
| `yggdrasil.saga.plan.ops` | Relational ops — joins, group-by, order-by, union, CTEs |
