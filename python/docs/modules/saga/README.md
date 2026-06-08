# Saga — the data engine

Saga is yggdrasil's unified, autonomous, lazy data engine: schema-aware
expressions (`yggdrasil.saga.expr`), execution plans + SQL parse/emit/execute
(`yggdrasil.saga.plan`), and the `Saga` engine facade that ties them together.
Saga holds no catalog — it parses the `FROM` sources and live-builds them
(path/URL via IO, in-memory frames via `Tabular.new`) — and owns a
local-disk staging session for spilling large results.

For the full walkthrough (SQL features, joins, CTEs, UDF registration), see
the [Saga Engine guide](../../guides/saga.md). Full signatures live in the
auto-generated [API reference](../../reference/yggdrasil/saga/).

```python
from yggdrasil.saga import Saga

saga = Saga(dialect="databricks")

# FROM resolves the source live — no registration needed
result = saga.sql("SELECT name, score FROM 'users.parquet' WHERE score > 80")
print(result.read_arrow_table().to_pylist())

# Or build a deferred pipeline that executes on the first read:
out = saga.scan("users.parquet").filter("score > 80").select("name", "score").read_arrow_table()
```

## Surface

| Symbol | Role |
|---|---|
| [`Saga`][yggdrasil.saga.Saga] | Engine facade — parse SQL, live-resolve sources, build/execute lazy plans |
| [`SagaSession`][yggdrasil.saga.SagaSession] | Local-disk staging area for result spill, auto-cleaned |
| [`ExecutionResult`][yggdrasil.saga.ExecutionResult] | Lazy, awaitable handle to a plan run — a `Tabular` + `Awaitable` |
| [`col`][yggdrasil.saga.col] / [`lit`][yggdrasil.saga.lit] / [`Predicate`][yggdrasil.saga.Predicate] | Expression / predicate AST (`yggdrasil.saga.expr`) |
| [`parse_sql`][yggdrasil.saga.plan.parse_sql] | Parse a SQL string into an executable plan node |
| [`SQLQueryParser`][yggdrasil.saga.plan.sql_parser.SQLQueryParser] | The underlying parser |
| [`ExecutionPlan`][yggdrasil.saga.plan.ExecutionPlan] / [`SelectPlan`][yggdrasil.saga.plan.SelectPlan] | Plan objects |
| [`LazyTabular`][yggdrasil.saga.plan.LazyTabular] | Deferred, lazily-evaluated `Tabular` |
| [`PlanNode`][yggdrasil.saga.plan.nodes.PlanNode] | Base node — [`SelectNode`][yggdrasil.saga.plan.nodes.SelectNode], [`InsertNode`][yggdrasil.saga.plan.nodes.InsertNode], [`MergeNode`][yggdrasil.saga.plan.nodes.MergeNode], [`ScanNode`][yggdrasil.saga.plan.nodes.ScanNode] |
| [`FunctionRegistry`][yggdrasil.saga.plan.func_registry.FunctionRegistry] / `FunctionMeta` | UDF registry + metadata |
| `BUILTIN_REGISTRY` | Built-in functions (incl. `explode_table`, `posexplode_table`) |
| `yggdrasil.saga.plan.ops` | Relational ops — joins, group-by, order-by, union, CTEs |
