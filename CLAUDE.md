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
8. **Real type hints, never string-quoted** — every module starts with `from __future__ import annotations`, then annotate with the **real** types (`Optional[Loki]`, not `"Optional[Loki]"`). When a type isn't importable at runtime (circular import, optional dep), import it under `if typing.TYPE_CHECKING:` and reference it unquoted — future-annotations makes that safe. No quoted forward-refs.
9. **One CLI entry point** — `ygg` is the only console script. New commands are subcommands of `ygg` (e.g. `ygg run`, `ygg loki`), never separate `ygg-*` scripts.

## Principles

1. **Exceptions** — derive from `YGGException`. API errors use `yggdrasil.exceptions.api`. No ad-hoc exception classes.
2. **Services own logic** — thin entry points (CLI handlers, properties); business rules + state live in services.
3. **Schemas are contracts** — `StrictModel` (extra="forbid") for request/response types.
4. **Int64 IDs** — prefer `int` over string UUIDs. Use xxhash-based composites: `xxh32(semantic_key) << 32 | timestamp_ms`. Never use cryptographic hashes for IDs.
5. **Upsert by default** — create if name not found, update if it exists. ID is immutable once assigned.
6. **Prefer uv over pip** — use `uv` for all Python tooling: `uv venv`, `uv pip install`, `uv build --wheel`. Reach for `pip` only when `uv` isn't on PATH. Run tests/scripts with `uv run`.

## Databricks

`yggdrasil.databricks` wraps the Databricks SDK behind `DatabricksClient` and
its `dbc.<service>` accessors (`sql`, `tables`, `volumes`, `warehouses`,
`compute`, `jobs`, `job_runs`, `secrets`, `iam`, `ai`, `genie`, …). Tabular
data moves as Arrow; `Table` is itself a `Tabular`. See `docs/guides/databricks.md`
and `docs/guides/databricks-cli.md`. Primary compute rules: prefer
**serverless** for inner Databricks I/O, a **single-user cluster** for
external-resource access, and always the **pre-built ygg wheel
environments** seeded by `ygg databricks seed`.

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
  plan/                 Execution plans + SQL parse/emit/execute
  http_/                HTTPSession + local/remote (Tabular) response cache
  pickle/               yggdrasil pickle + json (orjson) serialization
  execution/expr/       Predicate / expression engine (polars/pyarrow/sql backends)
  databricks/           Databricks SDK integrations
    client.py           DatabricksClient + dbc.<service> accessors
    sql/ table/ volume/ warehouse/ compute/ cluster/ job/ secrets/ iam/ ai/
    genie/              dbc.genie — AI/BI Genie spaces
    cli/                ygg databricks ... (configure/seed/deploy/sql/jobs/fs/…)
    assistant/          Databricks Assistant skills + guidance (deployed by seed)
    loki/               specialized Loki Databricks agent
  loki/                 Loki — the global yggdrasil agent + TokenEngine + behaviors
  cli/                  ygg CLI (main → databricks, loki)
  exceptions/api.py     APIError hierarchy
packages/yggdrasil/     JS/TS port (mirrors the Python package layout)
docs/                   mkdocs (Material) — guides + auto-generated API reference
```

## CLI

`ygg databricks` — Databricks management CLI (configure, seed, deploy, sql, jobs, fs, …)
`ygg loki` — global yggdrasil agent (status / capabilities / behaviors / token / run)

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABRICKS_HOST` / `DATABRICKS_TOKEN` | — | Databricks auth (or use `ygg databricks configure`) |
| `YGG_NODE_SAGA_DIALECT` | postgres | Default SQL dialect (where applicable) |
