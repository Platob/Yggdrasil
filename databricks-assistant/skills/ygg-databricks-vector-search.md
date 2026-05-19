# Skill: Databricks Vector Search — typed indexes, Arrow-first queries, time-series-aware

## When to use

The user mentions "vector search", "vector index", "vector store",
"semantic search", "similarity search", "embeddings", "RAG",
"retrieval-augmented generation", "find similar documents", "find
similar tickers / products / customers", "build an embedding index
on this table", "query the index by text" / "by vector",
"hybrid / full-text + vector search", or "rerank these results".
Also when the conversation is about wiring Databricks Vector Search
into an ingestion pipeline (the curated → index → query loop).

Builds on [`ygg-curated-views`](ygg-curated-views.md) (indexes only
ever read curated, never `raw_`), [`ygg-data-modeling`](ygg-data-modeling.md)
(schema-per-source, `<source>.<entity>` naming),
[`ygg-schema-fields`](ygg-schema-fields.md) (the `Schema` / `Field`
that drives both the Delta source table *and* the index row schema),
[`ygg-cast`](ygg-cast.md) (typed result materialization via
`target_schema=`), and [`ygg-mlops`](ygg-mlops.md) (the broader AI
service surface — vector search is one of three: `client.ai.vector_search`
today, `client.ai.serving` and `client.ai.registry` next).

## Primary surface

```python
from yggdrasil.databricks import DatabricksClient

dbc = DatabricksClient()
vs = dbc.ai.vector_search              # VectorSearch service (cached on client.ai)
```

| Call | Returns | For |
| --- | --- | --- |
| `vs.endpoint(name)` | `VectorSearchEndpoint` | one endpoint handle |
| `vs.list_endpoints()` | iterator | discover endpoints |
| `vs.find_endpoint(name=…)` | `VectorSearchEndpoint \| None` | name-keyed lookup |
| `vs.index(full_name, endpoint_name=…)` | `VectorSearchIndex` | one index handle |
| `vs.list_indexes(endpoint_name=…)` | iterator | discover indexes |

Endpoints host compute, indexes hold the data. One endpoint can back
many indexes; one index lives on exactly one endpoint.

## Set the defaults once

```python
from dataclasses import replace

vs.defaults = replace(
    vs.defaults,
    endpoint_name="rag-endpoint",                      # avoids repeating it everywhere
    embedding_model_endpoint_name="databricks-bge-large-en",
    endpoint_type="STANDARD",                          # or "STORAGE_OPTIMIZED"
    pipeline_type="TRIGGERED",                         # cheap; "CONTINUOUS" stays in lockstep
)
```

Every subsequent `vs.index(...)` / `vs.endpoint(...)` /
`create_delta_sync(...)` call inherits these and stops asking for
them inline.

## Endpoint lifecycle

```python
ep = vs.endpoint("rag-endpoint")

ep.ensure_created()            # idempotent — no-op when it exists
ep.create(wait=True)           # block until ONLINE (default budget = 20 min)
ep.is_online                   # bool
ep.state                       # "ONLINE" | "PROVISIONING" | ...
ep.wait_online(wait=600)       # block up to 10 min
ep.delete(missing_ok=True)     # idempotent teardown
```

Default `create(wait=...)` is **non-blocking** — endpoint provisioning
takes 5–10 minutes; only opt in to waiting when the next step actually
needs the endpoint online. The opt-in shape matches every other
`WaitingConfigArg`: `True` / seconds / `timedelta` / a full
`WaitingConfig`.

## Schema-per-source for indexes

Indexes are UC three-part identifiers. Default layout:

```
main.<source>.raw_<entity>               ← bronze (immutable, never indexed)
└── main.<source>.<entity>                ← silver (curated — what feeds the index)
    ├── main.<source>.idx_<entity>        ← vector index (delta-sync over curated)
    └── main.<source>.idx_<entity>_writeback   ← optional embedding writeback table
```

Rules:

- `idx_*` is the convention for vector-search indexes — analysts /
  ML callers know it at a glance.
- **Never** delta-sync an index against `raw_<entity>` — drift on
  text shape / timestamp formats / decimal scale silently breaks
  retrieval. Build on curated, the same way models do (see
  [`ygg-mlops`](ygg-mlops.md)).
- Per-shard sources (see
  [`ygg-data-modeling`](ygg-data-modeling.md#hard-table-split))
  get one index per shard or a `UNION ALL` curated view; do not
  point the index at the wildcard.

## Delta-sync index — the common path

The source is a UC Delta table; the index re-embeds on each sync.
Managed embeddings (Databricks calls the serving endpoint for you)
are the simpler choice — self-managed embeddings only when you
already write a precomputed vector column.

### Managed embeddings (source column = raw text)

```python
idx = vs.index("main.rag.idx_docs").create_delta_sync(
    source_table="main.rag.docs",                # curated
    primary_key="id",
    embedding_source_columns=["text"],           # raw text column on curated
    # embedding_model_endpoint_name=...,         # inherits from defaults
    # pipeline_type="TRIGGERED",                 # inherits from defaults
    columns_to_sync=["id", "text", "language_iso", "created_at_utc"],
    wait=True,                                   # block until status.ready (no-op when False/None)
)
```

`columns_to_sync` is the projection from the source Delta table into
the index — keep it tight (the index storage is per-column). UTC
timestamps, ISO codes, and `lat` / `lon` from
[`ygg-curated-views`](ygg-curated-views.md) all index cleanly and
become available for `filters=`.

### Self-managed embeddings (source has precomputed vector)

```python
idx = vs.index("main.rag.idx_docs").create_delta_sync(
    source_table="main.rag.docs_with_vectors",
    primary_key="id",
    embedding_vector_columns=[
        {"name": "vec", "embedding_dimension": 768},
    ],
    pipeline_type="CONTINUOUS",                  # keep in lockstep
)
```

The vector column has to live on the source table at that exact
dimension — no server-side coercion. Run the embedding job upstream
and write into `<entity>_with_vectors` (or a `_v2` projection of the
curated entity).

### Optional embedding writeback

```python
idx.create_delta_sync(
    source_table="main.rag.docs",
    primary_key="id",
    embedding_source_columns=["text"],
    embedding_writeback_table="main.rag.idx_docs_writeback",  # UC table
)
```

The managed embeddings get written back into this UC table so you
can join them in downstream curated / display views (e.g. expose a
`dash_docs_embeddings` for analyst inspection).

## Direct-access index — caller-managed rows

Use this when the data doesn't live in a Delta table — ad-hoc
ingestion, on-the-fly enrichment, hot-path inserts.

### Schema from a yggdrasil `Schema` (preferred)

The same `Schema` / `Field` / `DataType` surface that drives Delta
DDL (see [`ygg-schema-fields`](ygg-schema-fields.md)) drives the
index row schema. Define once, reuse for the source table *and*
the index:

```python
from yggdrasil.data import DataType, Field, Schema

DOCS_INDEX_SCHEMA = Schema.from_fields([
    Field("id",             DataType.from_("string"),         nullable=False,
          tags={"primary_key": True}),
    Field("text",           DataType.from_("string"),         nullable=False),
    Field("language_iso",   DataType.from_("string"),         nullable=True,
          comment="ISO 639-1 two-letter code — see yggdrasil.data.enums."),
    Field("created_at_utc", DataType.from_("timestamp(UTC)"), nullable=False),
    Field("vec",            DataType.from_("list<float>"),    nullable=False,
          comment="768-dim BGE embedding."),
])

vs.index("main.rag.idx_docs").create_direct_access(
    primary_key="id",
    schema=DOCS_INDEX_SCHEMA,                                  # ← ygg Schema
    embedding_vector_columns=[{"name": "vec", "embedding_dimension": 768}],
)
```

Behind the scenes the service serialises each field via
`DataType.to_spark_name().lower()` so the on-the-wire `schema_json`
stays one-to-one with the Databricks SQL types Delta speaks. **Do
not hand-roll the JSON** — let the service project from `Schema` so
the index and the source table can never drift on type / case /
nullability.

Accepted shapes: `Schema`, `pa.Schema`, `Mapping[str, str]` (already
Databricks-flavoured), or the legacy `schema_json="…"` string.

### Upsert with the data shape you already have

```python
import pyarrow as pa
# Arrow Table (preferred — same shape Delta / cast registry hand back)
idx.upsert(pa.table({"id": ["a"], "text": ["alpha"], "vec": [[0.1] * 768]}))

# Polars / pandas / row dicts — pick whichever the upstream produces
idx.upsert(polars_df)
idx.upsert(pandas_df)
idx.upsert([{"id": "a", "text": "alpha", "vec": [0.1] * 768}])

idx.delete_rows(["a", "b", "c"])
```

`upsert` accepts `pa.Table` / `pl.DataFrame` / `pd.DataFrame` /
`Sequence[Mapping]`. The conversion to the JSON wire shape happens
in one `to_pylist` hop at the row-endpoint boundary (see the
"genuine row endpoint" exemption in `CLAUDE.md`). Empty inputs are
skipped — no API call, returns `None`.

## Query — typed result through the cast registry

```python
result = vs.index("main.rag.idx_docs").query(
    columns=["id", "text", "language_iso", "created_at_utc"],
    query_text="how do I onboard a new customer?",
    num_results=10,
    filters={"language_iso": "en"},          # mapping → serialised via ygg_json
    query_type="HYBRID",                     # "ANN" (default) / "HYBRID" / "FULL_TEXT"
)

result.row_count                             # int
result.column_names                          # tuple[str, ...]
result.next_page_token                       # cursor for next page
result.to_dicts()                            # [{"id": ..., "score": ...}, ...]
result.to_arrow_table()                      # pyarrow.Table
result.to_polars()                           # polars.DataFrame
result.to_pandas()                           # pandas.DataFrame
```

### Pin the result schema for typed downstream consumption

The wire returns every cell as a JSON-string; the default
`to_arrow_table()` casts each column to the type resolved from
`type_text` (`pa.string()` fallback for unknown / complex shapes).
For tight downstream consumption, pin a `target_schema=`:

```python
from yggdrasil.data import DataType, Field, Schema

QUERY_RESULT_SCHEMA = Schema.from_fields([
    Field("id",             DataType.from_("string"),          nullable=False),
    Field("text",           DataType.from_("string"),          nullable=False),
    Field("language_iso",   DataType.from_("string"),          nullable=True),
    Field("created_at_utc", DataType.from_("timestamp(UTC)"),  nullable=False),
    Field("score",          DataType.from_("float32"),         nullable=False,
          comment="Databricks `__db_score`. Promoted to f32 for compactness."),
])

result = idx.query(
    columns=["id", "text", "language_iso", "created_at_utc"],
    query_text="onboarding",
    target_schema=QUERY_RESULT_SCHEMA,        # threaded onto VectorSearchQueryResult
)
df = result.to_polars()                       # already typed; no per-column cast at the call site
```

`target_schema` runs the assembled Arrow table through
`Schema.cast_arrow` — the same registered path
[`ygg-statement-result`](ygg-statement-result.md) and
[`ygg-databricks-genie`](ygg-databricks-genie.md) use. Timezone
intent, decimal scale, nullability, and field order all stay
honored without a hand-rolled per-column cast loop. Inherited by
`to_polars()` / `to_pandas()`; `next_page()` carries the pinned
target forward across pagination.

## Time-series-aware retrieval

Vector indexes work best when the source has been *standardised*
in curated (UTC timestamps, ISO codes, pre-rolled time buckets).
For time-series sources:

1. **Chunk by time bucket, not character count.** Curated already
   has `bucket_start_utc` (see
   [`ygg-curated-views`](ygg-curated-views.md) §2 + the trading /
   commodity time-series shapes). Generate one row per
   `(entity, bucket_start_utc)` with the bucket's narrative as
   `text`; that bucket becomes the retrievable unit.
2. **Pre-roll text the way you pre-roll prices.** If a dashboard
   renders the last 90 days at 1-hour granularity, the index for
   "explain this period" should hold `dash_*_1h`-shaped windows,
   not raw tick rows.
3. **Always carry `created_at_utc` / `bucket_start_utc` in
   `columns_to_sync`.** Filter retrieval by time at query time:
   `filters={"bucket_start_utc": {">=": "2026-04-01T00:00:00Z"}}`.
   The metadata stays in the index — no second round trip to
   Delta to time-window the candidates.
4. **Use a `points: list<struct<lat, lon, observation_utc>>` column**
   for spatiotemporal sources (see
   [`ygg-curated-views`](ygg-curated-views.md) §3b). Direct-access
   indexes can carry it as `array<struct<…>>`.

```python
RAG_PRICE_NARRATIVE_SCHEMA = Schema.from_fields([
    Field("entity_id",         DataType.from_("string"),         nullable=False,
          tags={"primary_key": True}),
    Field("bucket_start_utc",  DataType.from_("timestamp(UTC)"), nullable=False,
          tags={"primary_key": True, "partition_by": True}),
    Field("text",              DataType.from_("string"),         nullable=False,
          comment="One-sentence narrative of the bucket (OHLCV summary, anomalies, regime label)."),
    Field("language_iso",      DataType.from_("string"),         nullable=False),
    Field("currency_iso",      DataType.from_("string"),         nullable=False),
])
# Same Schema feeds the curated table AND `idx.create_delta_sync(source_table=..., ...)`.
```

## Pagination

```python
for page in result.iter_pages():
    process_page(page.to_arrow_table())
```

`iter_pages()` yields the current result, then `next_page()` until
exhausted. Useful when the dashboard / RAG prompt iterates the
full candidate set; the inherited `target_schema` applies on every
page.

## Auto-discovery — which curated tables deserve an index?

Use the same heuristics as the autoML candidate pass in
[`ygg-mlops`](ygg-mlops.md), tightened for retrieval:

```python
def discover_vector_search_candidates(dbc) -> list[dict]:
    """Walk curated schemas, return tables that look indexable."""
    candidates = []
    for schema in dbc.schemas.list(catalog="main"):
        if schema.name.startswith(("iso", "_meta")):
            continue
        for tbl in dbc.tables.list(catalog="main", schema=schema.name):
            if tbl.name.startswith(("raw_", "ml_", "idx_", "dash_")):
                continue
            info = tbl.read_info()
            text_cols = [f for f in info.fields
                         if f.dtype.is_string and (f.name or "").lower() not in
                            {"id", "currency_iso", "country_iso", "language_iso",
                             "region_iso", "timezone_iana", "mic_iso", "eic_code"}]
            pks = info.primary_keys
            if not text_cols or not pks:
                continue
            candidates.append({
                "table": tbl.full_name(),
                "primary_key": pks[0].name,
                "text_candidates": [f.name for f in text_cols],
            })
    return candidates
```

Then for each candidate, build the index via `create_delta_sync` in
a `dbc.jobs.create_or_update(name=f"idx_{table.replace('.', '_')}", ...)`
that runs on the same compute story as the curated rebuild — see
[`ygg-databricks-job-workflows`](ygg-databricks-job-workflows.md).

## Compute pinning

Vector-search endpoints are **their own compute** — they do not
share the warehouse or all-purpose cluster pool. Pin the
provisioning task wherever it's cheapest (it only calls the
control plane); the actual embedding / query work runs server-side
on the endpoint Databricks provisions. The serverless ingestion
task that triggers `idx.sync()` runs fine on the same serverless
pool as the curated rebuild.

## Where it slots in the pipeline DAG

End-to-end alongside the existing eight steps of
[`ygg-ingestion-pipeline`](ygg-ingestion-pipeline.md):

```
1. fetch          ┐  classic cluster (HTTP egress)
2. cast           │
3. raw_ insert    ┘
4. curate         ┐
5. dash_*         │  serverless
6. idx_* sync     │  ← this skill
7. retrain        ┘
```

The index sync runs **after** the curated rebuild and **alongside**
the dashboard / ML refresh. Add it as a downstream task on the same
Job DAG with `depends_on=["curate"]`:

```python
job.pytask(
    lambda: vs.index("main.rag.idx_docs").sync(),
    task_key="idx_docs_sync",
    depends_on=["curate"],
).create()
```

## Don'ts

- Don't delta-sync an index against `raw_<entity>` — drift on text
  shape / dtype / nullability silently breaks retrieval. Curated
  only, same rule as ML training.
- Don't hand-roll `schema_json` for direct-access indexes when a
  `Schema` already describes the row shape. Pass `schema=` and let
  the service project — the index can't drift from the source if
  both read the same `Schema` literal.
- Don't put curated rows into a direct-access index when they
  already live in a UC Delta table. Use delta-sync — server-side
  pipeline + free re-embedding on table mutation, no caller
  upsert loop to keep alive.
- Don't blow up the row count by re-indexing every minor edit on
  `TRIGGERED`. Sync as a downstream task on the ingestion Job DAG,
  not on a side schedule that can race the curated rebuild.
- Don't `.to_pylist()` the result rows to build a chart input or a
  Polars frame — pass `target_schema=` and call `to_arrow_table()` /
  `to_polars()` / `to_pandas()` once.
- Don't query with both `query_text=` and `query_vector=` — the
  service raises (mutually exclusive). Pick one path: managed
  embedding endpoint embeds your text *or* you pre-compute the
  vector.
- Don't `pa.Table` → `to_pandas()` → row-loop-fill → upsert. Hand
  the Arrow Table straight to `idx.upsert(table)`; the row-endpoint
  conversion is one `to_pylist` hop at the API boundary, no Python
  loop in the call site.
- Don't pickle a `VectorSearchEndpoint` / `VectorSearchIndex`
  hoping it preserves the SDK handle. Re-resolve at the other side
  via `client.ai.vector_search.index(full_name)` — handles are
  cheap and the service holds the singleton-by-config wiring.
- Don't fall back to `ws.vector_search_endpoints.create_endpoint(...)`
  / `ws.vector_search_indexes.query_index(...)` directly. The
  resource singleton (`vs.endpoint(...)` / `vs.index(...)`) wraps
  the SDK with `if_not_exists` / `missing_ok` / waiting / cached
  infos / typed results — same rule as
  `Volume.create` / `Schema.create` in
  [`ygg-databricks-client`](ygg-databricks-client.md).
