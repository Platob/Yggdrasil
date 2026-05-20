# Skill: business-display views — dashboard-shaped, pivoted, time-series-optimized

## When to use

The user has curated tables (see
[`ygg-curated-views`](ygg-curated-views.md)) and asks to "build a
dashboard view", "make this queryable in Databricks SQL", "expose a
read-shape for the BI tool", "pivot this for analyst use", "precompute
this aggregate", "this is too slow on the curated table", "build a
KPI view", "feed Power BI / Tableau / Hex / Mode / Sigma / Lightdash
from this", or "what shape should the analyst query?".

Comes **after** curated. Curated is the canonical source of truth;
display is the consumer-shaped projection over it.

## The three layers, end to end

```
main.<source>.raw_<entity>          ← bronze   — source-shaped, immutable, audit
└── main.<source>.<entity>           ← silver   — curated, standardised, stable
    └── main.<source>.dash_<view>     ← gold     — display, dashboard-shaped
```

Hard rules for `dash_*`:

1. **Read-only contract.** Analysts query, never write. No `INSERT`
   from the BI tool — refresh is pipeline-driven.
2. **Wide and denormalised.** Joins, lookups, FK resolutions all
   pre-computed. The dashboard query is `SELECT * FROM dash_X WHERE
   <filter>` — no joins.
3. **Stable column set per `<view>`.** Adding a column is fine;
   renaming / dropping is a versioned change (`dash_<view>_v2`).
4. **Materialised when read-cost > rebuild-cost.** Pick table vs
   view vs streaming table from the decision matrix below.

## Pick the right materialisation

| Shape | When to use | DDL |
| --- | --- | --- |
| **View** | Cheap renames + casts only; data is already aggregated upstream. Read traffic moderate. | `CREATE VIEW dash_X AS SELECT …` |
| **Table** (rebuild on refresh) | Expensive joins / window functions, weekly+ refresh acceptable. | `INSERT OVERWRITE dash_X SELECT …` inside a scheduled job. |
| **Materialised view** | Reads dominate writes, the source updates predictably. Auto-refresh on the underlying tables. | `CREATE MATERIALIZED VIEW dash_X AS SELECT …` (Databricks SQL, requires DLT in some workspaces). |
| **Streaming table** | Latency-sensitive (sub-minute), source is itself a streaming table or Delta change feed. | DLT pipeline (`@dlt.table`). |

Default to **view** when standardisation cost is small, **table** for
anything expensive — pre-aggregate during the ingestion job, not at
query time.

## Time-series display tables

The dominant shape in trading / commodity / energy / IoT use cases.
The curated table is **long** (one row per `(entity, timestamp)`);
the display table is **wide** (one row per timestamp, columns =
entities) **or** the curated layer with rolled-up windows.

### Pivot long → wide

For dashboard tiles showing N series side-by-side:

```sql
CREATE OR REPLACE TABLE main.entsoe.dash_dayahead_prices_eu AS
SELECT
    delivery_start_utc,
    MAX(CASE WHEN eic_code = '10YFR-RTE------C' THEN price END) AS price_fr,
    MAX(CASE WHEN eic_code = '10YDE-VE-------2' THEN price END) AS price_de,
    MAX(CASE WHEN eic_code = '10YGB----------A' THEN price END) AS price_gb,
    MAX(CASE WHEN eic_code = '10YES-REE------0' THEN price END) AS price_es
FROM main.entsoe.dayahead
WHERE delivery_start_utc >= current_date() - INTERVAL 90 DAYS
GROUP BY delivery_start_utc
```

Rules:

- **One time-axis column** (`delivery_start_utc`), `UTC`-suffixed.
- **One column per entity** with a stable, kebab-`_`-cased name
  derived from the curated id (`price_fr`, `price_de`).
- **Window** the source down to the dashboard's useful horizon
  (`>= current_date() - 90 days`) — don't ship 10 years of history
  if the dashboard only renders the last quarter.
- **`MAX(CASE WHEN …)`** is the portable pivot — Databricks SQL
  also supports `PIVOT (…)` but the `CASE` form survives engine
  migrations and reads more obviously in code review.

### Pre-rolled time buckets

Dashboards rarely render raw 1-second / 1-minute resolution; pre-roll
to the granularity the consumer needs:

```sql
CREATE OR REPLACE TABLE main.cme.dash_ohlcv_5m AS
SELECT
    vendor_symbol,
    mic_iso,
    date_trunc('5 MINUTES', bucket_start) AS bucket_start_5m_utc,
    FIRST(open ORDER BY bucket_start)  AS open,
    MAX(high)                          AS high,
    MIN(low)                           AS low,
    LAST(close ORDER BY bucket_start)  AS close,
    SUM(volume)                        AS volume,
    SUM(close * volume) / NULLIF(SUM(volume), 0) AS vwap
FROM main.cme.ohlcv_1m
WHERE bucket_start >= current_date() - INTERVAL 30 DAYS
GROUP BY vendor_symbol, mic_iso, date_trunc('5 MINUTES', bucket_start)
```

Naming: `dash_<entity>_<bucket-iso>` — the bucket size is part of
the table name (`_5m`, `_1h`, `_1d`). Analysts pick the right one
without scanning the column comments.

### Liquid-cluster the dashboard's filter dimensions

```python
DASH_PRICES_SCHEMA = Schema.from_fields([
    Field("delivery_start_utc", DataType.from_("timestamp(UTC)"),
          nullable=False,
          tags={"primary_key": True, "partition_by": True}),
    Field("price_fr", DataType.from_("decimal(18, 6)"),
          nullable=True, tags={"cluster_by": True}),
    Field("price_de", DataType.from_("decimal(18, 6)"),
          nullable=True, tags={"cluster_by": True}),
    # ...
])
```

`cluster_by` on the columns the dashboard filters on (date range,
country, currency) makes the read fast. See
[`ygg-data-modeling`](ygg-data-modeling.md) for the `Field` metadata
→ DDL mapping.

## Geo-display tables

When the dashboard renders a map, the display layer must carry the
geometry inline — frontend plugins (kepler.gl, pydeck, folium,
Databricks SQL geo visual) need `lat` / `lon` or a polygon in the
same row as the metric.

```python
from yggdrasil.data import Field, DataType, Schema, geo_point

DASH_ZONE_PRICES_SCHEMA = Schema.from_fields([
    Field("delivery_start_utc", DataType.from_("timestamp(UTC)"), nullable=False,
          tags={"primary_key": True, "partition_by": True}),
    Field("eic_code",           DataType.from_("string"), nullable=False,
          tags={"primary_key": True, "cluster_by": True}),
    Field("zone_name",          DataType.from_("string"), nullable=False),
    # GeoPoint struct — frontend renders as a marker / centroid.
    geo_point("centroid", nullable=True,
              comment="Bidding-zone centroid (WGS84). Renders as a marker pin."),
    # Polygon for shading the zone.
    Field("boundary_geojson",   DataType.from_("string"), nullable=True,
          metadata={b"comment": b"GeoJSON Feature. Pre-resolved from main.iso.bidding_zone."}),
    Field("price",              DataType.from_("decimal(18, 6)"), nullable=False),
    Field("currency_iso",       DataType.from_("string"), nullable=False),
])
```

For tracks / time-series-of-points (vessel routes, wind patterns):

```python
Field("points",
      DataType.from_(
          "list<struct<lat: float64, lon: float64, observation_utc: timestamp(UTC)>>"
      ),
      nullable=True,
      metadata={b"comment": b"Ordered track. Each cell independently renderable."}),
```

See [`ygg-curated-views`](ygg-curated-views.md#3b-geographic-data--always-carry-latlon--optional-polygon)
for the full geo convention. The display layer **inlines** what the
curated layer's FK join would produce — analysts shouldn't have to
know about `main.iso.bidding_zone`.

## KPI / aggregate display tables

For single-number tiles (today's revenue, last hour's average
price, current open positions):

```sql
CREATE OR REPLACE TABLE main.vendor_orders.dash_kpis AS
SELECT
    'orders_today_count'   AS kpi,
    COUNT(*)               AS value,
    'orders'               AS unit,
    current_timestamp()    AS computed_at_utc
FROM main.vendor_orders.orders
WHERE date(created_at_utc) = current_date()

UNION ALL

SELECT
    'orders_today_total_eur',
    SUM(amount * fx.rate),
    'EUR',
    current_timestamp()
FROM main.vendor_orders.orders o
LEFT JOIN main.fx.spot fx
       ON fx.pair_iso = concat(o.currency_iso, 'EUR')
      AND fx.observed_utc = date_trunc('HOUR', o.created_at_utc)
WHERE date(o.created_at_utc) = current_date()
```

One KPI per row, schema `(kpi: string, value: decimal,
unit: string, computed_at_utc: timestamp)` — keeps the column set
stable as KPIs are added / removed; the dashboard tile filters by
`kpi` name.

## Wiring it into the ingestion pipeline

Display tables refresh **after** curated, on the same Job DAG. Use
multi-task jobs (see
[`ygg-databricks-job-workflows`](ygg-databricks-job-workflows.md)):

```python
job = dbc.jobs.create_or_update(name="vendor_orders_etl", tasks=[])

ingest    = job.pytask(ingest_raw,      task_key="ingest").create()
curate    = job.pytask(rebuild_curated, task_key="curate",
                        depends_on=["ingest"]).create()
dash_kpis = job.pytask(rebuild_dash_kpis, task_key="dash_kpis",
                        depends_on=["curate"]).create()
dash_ts   = job.pytask(rebuild_dash_timeseries, task_key="dash_ts",
                        depends_on=["curate"]).create()
```

Display tasks run in parallel after curated lands. A failure on one
display task doesn't roll back the upstream curated refresh — that
stays available to other consumers.

## When a `dash_*` table isn't enough — graduate to a Databricks App

Display tables are read through whatever the consumer prefers:
Databricks SQL, Power BI, Tableau, Hex, an AI/BI Dashboard, or
a Databricks App. The decision is mostly the consumer's, but two
shapes specifically call for [`ygg-databricks-apps`](ygg-databricks-apps.md):

- **Interactive controls or write-back** — slider that re-runs a
  scoring job, button that promotes a model challenger, form that
  triggers a counterfactual simulation. AI/BI tiles can't write;
  an App backend can call `dbc.jobs.run(...)`.
- **Custom layout, map + table + chart combined, sub-second renders
  on a hot pre-aggregated table** — the UX of a real React frontend
  (`react-leaflet` / `deck.gl` over the geo columns this layer
  already inlines) plus Arrow IPC over HTTP for the wire format.

Apps still read from `dash_*` tables — they don't replace this
layer, they sit downstream of it. Build the display table first;
point the App at it. See
[`ygg-databricks-apps`](ygg-databricks-apps.md) for the two
recipes (FastAPI + Next.js, or Next.js full-stack), OAuth OBO
auth, the world-map shapes, and the deploy story.

## When NOT to build a display table

- **Ad-hoc analyst exploration.** Curated tables are the right read
  surface; an analyst writing exploratory SQL doesn't need
  pre-aggregation.
- **One-off charts.** Build the chart against curated; if it ships
  to production / a dashboard, then promote.
- **The query is already < 1 s on curated.** Materialisation costs
  storage + a refresh job; only worth it when read cost is real.

## Don'ts

- Don't write to a `dash_*` table from a notebook. The pipeline
  owns refresh; ad-hoc writes corrupt the contract.
- Don't ship a `dash_*` view that does multi-table joins at query
  time when read traffic is high. Materialise it.
- Don't include columns the dashboard doesn't render — every extra
  column is bytes the BI tool scans. `SELECT only the projection`
  the dashboard tile uses.
- Don't pivot to > 30 columns by hand. At that scale, keep it long
  and pivot in the BI tool (Tableau / Power BI handle it cleanly).
- Don't reuse a vendor-specific id (`eic_code`, `mic_iso`) as a
  column name in a pivoted display table without an alias —
  ``price_10YFR-RTE------C`` reads worse than ``price_fr``. Pick a
  stable kebab-case label per entity and document the mapping.
- Don't forget to UTC-suffix the time axis on dash tables either.
  Same rule as curated.
- Don't gate a `dash_*` refresh on `_ingested_at`-style provenance
  columns. Those belong on `raw_*`; the display layer is consumer-
  facing.
