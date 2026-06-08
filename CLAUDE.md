# Yggdrasil

Schema-aware data interchange + Databricks tooling for Python, with a
JS/TS port. The reference implementation is **Python**
(`python/src/yggdrasil/`); the **JS/TS** library mirrors it
(`packages/yggdrasil/`).

> **Mid-refactor.** The legacy node server (`yggdrasil.node`) and the
> Next.js frontend (`nextjs/`) were removed — they were unmaintained. The
> project is now: a Python data-interchange library
> (Arrow/Polars/pandas/Spark/Databricks), Databricks tooling
> (`yggdrasil.databricks` + `ygg databricks`), and the **Loki** agent
> (`yggdrasil.loki` + `ygg loki`). Build the great project from this base.

## ⚠️ Cross-language parity (IMPORTANT — global rule)

Yggdrasil is replicated across languages so the same concepts run on the
backend and client-side. The reference implementation is **Python**
(`python/src/yggdrasil/`); the **JS/TS** port lives in
`packages/yggdrasil/` (mirrors the Python package layout — `enums/`,
…). More languages may follow.

**These ports are ONE contract, not independent code.** When you touch a
concept that exists in more than one language, you MUST keep them
synchronized — same names, same structure, same behavior:

- Changing a Python concept (e.g. a value in `enums/mime_type.py`, a new
  field, a renamed method) → mirror it in the JS/TS port (`enums/mimeType.ts`)
  in the same commit, and vice-versa.
- Keep the **structure** parallel: file-for-file, class-for-class,
  method-for-method (Python `MimeType.from_` ↔ TS `MimeType.fromName`/`from`).
- Each port's module header points at its counterpart; update both when the
  mapping changes. See `packages/yggdrasil/README.md` for the file map.
- If you can only do one side now, leave an explicit `// PARITY:` /
  `# PARITY:` note at both sites describing the gap — never let the two
  drift silently.

Treat a cross-language divergence as a bug.

## Coding Style

1. **Inline over micro-functions** — prefer inlining logic directly where it's used. Don't extract one-liner helpers, don't create utility functions called from a single site, don't wrap stdlib calls in project-specific wrappers. A 15-line method is better than 5 three-line functions calling each other. Extract only when the same logic appears 3+ times or the extracted function has genuine standalone semantics (e.g. `make_id`, `serialize_result`).
2. **Flat call stacks** — avoid deep delegation chains like `service → helper → sub-helper → util`. Services do the work. Two levels max for the common path.
3. **No defensive wrappers** — don't wrap `json.loads` in a `safe_parse_json`, don't wrap `Path.mkdir` in an `ensure_dir`. Call stdlib directly. Handle errors at the boundary where they matter.
4. **No premature abstraction** — three similar blocks of code is fine. A `BaseService` with one subclass is not. Don't create interfaces, registries, or plugin systems until the third concrete user exists. (Where a contract is explicitly wanted — e.g. Loki `LokiBehavior` / `TokenEngine` — that IS the abstraction; keep it minimal.)
5. **Collocate related code** — keep schemas, service logic, and the entry point for one concept readable together. Jumping between 5 files to understand one thing is worse than a 200-line service file.
6. **Delete dead code** — no commented-out blocks, no `# TODO: maybe later`, no unused imports. If it's not called, it doesn't exist.
7. **Prefer data over code** — dicts and lists over class hierarchies. Pydantic models over hand-rolled validation. Enum values over if/elif chains.
8. **Real type hints, never stringified** — annotate with the **real type object**, not a quoted string (`def f(x: str | None)`, not `def f(x: "str | None")`). Every module carries `from __future__ import annotations`, so an annotation that names a not-yet-imported type evaluates lazily and needs no quotes. When the type can't be imported at runtime (circular import, optional/heavy dependency), import it under `if typing.TYPE_CHECKING:` and reference it bare — still no quotes. Don't sprinkle `"Any"`/string forward-refs to dodge an import; add the real import (top-level when free of cycles, `TYPE_CHECKING` when not). The mirror JS/TS port keeps its real types too.

## Principles

1. **Exceptions** — derive from `YGGException`. API errors use `yggdrasil.exceptions.api`. No ad-hoc exception classes.
2. **Services own logic** — thin entry points (CLI handlers, properties); business rules + state live in services.
3. **Schemas are contracts** — `StrictModel` (extra="forbid") for request/response types.
4. **Int64 IDs** — prefer `int` over string UUIDs. Use xxhash-based composites: `xxh32(semantic_key) << 32 | timestamp_ms`. Never use cryptographic hashes for IDs.
5. **Upsert by default** — create if name not found, update if it exists. ID is immutable once assigned.
6. **Prefer uv over pip** — use `uv` for all Python tooling: `uv venv`, `uv pip install`, `uv build --wheel`. Reach for `pip` only when `uv` isn't on PATH. Run tests/scripts with `uv run`.

## Saga

`yggdrasil.saga` is the unified, autonomous, lazy data engine. It is the
single home for schema-aware lazy compute over `Tabular` data and centralizes
everything that used to live in `plan/` and `execution/expr/`:

- `saga.expr` — the expression / predicate AST with multi-backend emitters
  (python / arrow / polars / spark / sql).
- `saga.plan` — mutable execution plans (`SelectPlan` / `InsertPlan` /
  `MergePlan`), the immutable plan-node tree, `LazyTabular`, `ExecutionResult`
  (a lazy, awaitable `Tabular`+`Awaitable` handle to a plan run — read-side
  sibling of `StatementResult`), SQL parse/emit across dialects, and the
  Arrow-native UDF registry.
- `Saga` (`saga.engine`) — the engine facade. Holds **no catalog**
  (named-table registration comes later): it parses the `FROM` sources and
  **live-builds** them — path/URL via the IO layer, in-memory frames via
  `Tabular.new`. Runs on Saga's own plan executor (no statement executors).
  Each engine owns a `SagaSession` with a local-disk staging area
  (`~/.saga/<session>/staging`) for spilling results, auto-cleaned on close.
  Driven from code: `Saga(dialect=...).sql("SELECT ... FROM 'x.parquet'")` /
  `.scan(source).filter(...)` / `.parse(...)` / `.plan(...)` /
  `.collect(query, spill=True)`.

Reach for `from yggdrasil.saga import Saga, col, lit, parse_sql, SelectPlan`.
See `docs/guides/saga.md`.

In-memory Tabulars: `ArrowTabular` (`arrow/`), `PolarsTabular` (`polars/`),
`PandasTabular` (`pandas/`), `SparkDataset` (`spark/`). `Tabular.new(data)`
and `Tabular.from_(obj)` dispatch a raw engine frame to its native in-memory
Tabular (arrow / polars / pandas / spark); `from_` otherwise falls back to the
IO layer, which is specialized on serialized data (paths, URLs, byte streams).

## Databricks

`yggdrasil.databricks` wraps the Databricks SDK behind `DatabricksClient` and
its `dbc.<service>` accessors (`sql`, `tables`, `volumes`, `warehouses`,
`compute`, `jobs`, `job_runs`, `wheels`, `environments`, `secrets`, `iam`,
`ai`, `genie`, …). Tabular
data moves as Arrow; `Table` is itself a `Tabular`. See `docs/guides/databricks.md`
and `docs/guides/databricks-cli.md`. Primary compute rules: prefer
**serverless** for inner Databricks I/O, a **single-user cluster** for
external-resource access, and always the **pre-built ygg wheel
environments** deployed by `ygg databricks deploy`.

## Loki

`yggdrasil.loki` is the global yggdrasil agent: it detects the backends it
can reach (offline, never raises), acts as a token/credential provider
(chiefly Databricks), and dispatches pluggable `LokiBehavior` actions
(feature contracts). `yggdrasil.databricks.loki` is the specialized
Databricks agent. Driven from code (`Loki`) or the terminal (`ygg loki`).
See `docs/guides/loki.md`.

## Layout

```
python/src/yggdrasil/
  data/                 Schema / Field / DataType, casting, statement results
  arrow/                Arrow helpers + ops
  io/                   Tabular IO (parquet, arrow IPC, xlsx, delta), Holder
  http_/                HTTPSession + local/remote (Tabular) response cache
  pickle/               yggdrasil pickle + json (orjson) serialization
  saga/                 Saga — the unified lazy data engine
    engine.py           Saga facade — catalog + SQL + lazy plan execution
    expr/               Predicate / expression engine (polars/pyarrow/sql backends)
    plan/               Execution plans + SQL parse/emit/execute + UDF registry + LazyTabular
  databricks/           Databricks SDK integrations
    client.py           DatabricksClient + dbc.<service> accessors
    sql/ table/ volume/ warehouse/ compute/ cluster/ job/ secrets/ iam/ ai/
    wheels/             dbc.wheels — build/upload/deploy/browse the wheel registry
    environments/       dbc.environments — assemble base envs + deploy projects
    genie/              dbc.genie — AI/BI Genie spaces
    cli/                ygg databricks ... (configure/deploy/sql/jobs/fs/…)
    assistant/          Databricks Assistant skills + guidance bundle
    loki/               specialized Loki Databricks agent
  loki/                 Loki — the global yggdrasil agent + TokenEngine + behaviors
  cli/                  ygg CLI (main → databricks, loki)
  exceptions/api.py     APIError hierarchy
packages/yggdrasil/     JS/TS port (mirrors the Python package layout)
docs/                   mkdocs (Material) — guides + auto-generated API reference
```

## CLI

`ygg databricks` — Databricks management CLI (configure, deploy, wheel, environment, sql, jobs, fs, …)
`ygg loki` — global yggdrasil agent (status / capabilities / behaviors / engines / tools / reason / do / token / run)

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABRICKS_HOST` / `DATABRICKS_TOKEN` | — | Databricks auth (or use `ygg databricks configure`) |
| `YGG_NODE_SAGA_DIALECT` | postgres | Default SQL dialect (where applicable) |
