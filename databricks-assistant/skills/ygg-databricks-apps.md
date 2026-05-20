# Skill: Databricks Apps — Next.js + FastAPI frontends over the curated layer, with world-map plugins for trading KPIs

## When to use

The user has a curated / `dash_*` layer (see
[`ygg-curated-views`](ygg-curated-views.md) +
[`ygg-display-views`](ygg-display-views.md)) and asks to "build an
app", "build a Databricks App", "ship a custom dashboard",
"build a Next.js frontend over our data", "expose trading KPIs on
a world map", "render bidding zones / countries / pipelines on a
map with live numbers", "the Databricks AI/BI Dashboard isn't
flexible enough", "the SQL warehouse is too slow for our
dashboard read pattern", "we need write-back / controls in the
UI", or "deploy a React frontend on Databricks".

Databricks Apps is the right surface **after** AI/BI Dashboards
fall short. Reach for Apps when at least one of these is true:

| Need | Why a Databricks App | Why not AI/BI Dashboard |
| --- | --- | --- |
| Interactive controls (sliders, multi-select that drive joined queries, re-run a scoring job, override a model champion) | Apps run a real backend you control. | AI/BI is read-only over saved queries. |
| Write-back to a table or trigger a Job | Apps can call `dbc.jobs.run()` server-side. | Dashboards can't write. |
| Sub-second renders on a hot pre-aggregated table | App backend can cache + serve Arrow IPC over HTTP in tens of ms. | SQL Warehouse cold-start + every render = a new query. |
| Custom UX — map + table + chart + form on one canvas, dependent dropdowns, optimistic UI | React / Next.js full design freedom. | AI/BI tile system is fixed. |
| Per-row map-rendered KPIs (per-zone, per-country, per-route) | Apps load GeoJSON + WGS84 coordinates from your curated layer and render with `react-leaflet` / `deck.gl`. | The Databricks SQL geo visual is functional but inflexible. |
| External auth (OAuth OBO carries user RBAC into the queries) | First-class on Apps. | Same on AI/BI, but you lose the rest of the surface. |

**Don't** reach for Apps when:

- The use case is "scroll and filter on a SQL table" — AI/BI is faster to ship and cheaper to maintain.
- You don't have a `dash_*` table yet — build it first (see [`ygg-display-views`](ygg-display-views.md)); Apps over raw curated is fine only when the read is one-row-per-tile.
- You'd be re-implementing model training / scoring in the app. Apps are a UI shell; **business logic stays in scheduled Job tasks**, the app calls `dbc.jobs.run(...)`.

## The whole stack at a glance

```
main.<source>.dash_<view>         ← Delta table (curated wide-form)
        │
        ▼
   ┌────────────────────────────────────────────────────────┐
   │  Databricks SQL Warehouse  (read path)                  │
   └────────────────────────────────────────────────────────┘
        │
        ▼  (Arrow IPC over HTTP, sub-second when warm)
   ┌────────────────────────────────────────────────────────┐
   │  Backend  — FastAPI (Python) OR Next.js API routes      │
   │    yggdrasil.databricks.DatabricksClient → dbc.sql      │
   │    yggdrasil.data.cast.convert / Schema / DataField     │
   │    OAuth on-behalf-of token forwarded from the browser  │
   └────────────────────────────────────────────────────────┘
        │
        ▼  (JSON or Arrow IPC over fetch)
   ┌────────────────────────────────────────────────────────┐
   │  Frontend  — Next.js (React)                            │
   │    react-leaflet (zones / boundaries / markers)         │
   │    deck.gl (large point sets, animated flows)           │
   │    recharts / visx / observable-plot (line / bar / KPI) │
   └────────────────────────────────────────────────────────┘
        │
        ▼
   Databricks Apps runtime  (serves the SPA; runs the backend process)
```

The whole thing deploys with `databricks apps deploy` from a
workspace directory; runtime is Apps' managed compute, **not** a
SQL warehouse and **not** a job cluster.

## Pick the architecture

Two recipes — neither is wrong, but they have different trade-offs.

### Recipe A — FastAPI backend + Next.js frontend (recommended for ygg-heavy work)

Use when:
- The backend needs `yggdrasil` (`DatabricksClient`, `Schema`,
  `cast.convert`, `FxRate`, `GeoZoneCatalog`, `ml_*` services). ygg
  is Python-only; the FastAPI process is where it lives.
- You want to serve Arrow IPC bytes directly to the browser
  (`pa.ipc.RecordBatchStreamWriter` → fetch → `apache-arrow` JS
  parser). Cheapest wire format when the same schema repeats.
- The team has Python data engineers already; the frontend is
  rendered in Next.js but the data plane is owned by Python.

Repo layout:

```
my_app/
├── app.yaml                    ← Databricks Apps spec
├── backend/                    ← FastAPI
│   ├── pyproject.toml
│   ├── requirements.txt        ← ygg[databricks,api] + fastapi + uvicorn
│   └── src/my_app/
│       ├── main.py             ← FastAPI(..); uvicorn entrypoint
│       ├── deps.py             ← DatabricksClient singleton; OBO auth
│       ├── routes/
│       │   ├── kpis.py         ← /api/kpis/<task>
│       │   ├── map.py          ← /api/map/<view>     (GeoJSON + values)
│       │   └── timeseries.py   ← /api/timeseries/<view>?start=&end=
│       └── arrow_response.py   ← Helper: Tabular → Arrow IPC bytes
└── frontend/                   ← Next.js
    ├── package.json            ← next, react, react-leaflet, deck.gl, recharts, apache-arrow
    ├── next.config.js          ← rewrites: /api/* → http://localhost:8000/api/*
    └── src/
        ├── app/                ← Next.js 14 app router
        │   ├── page.tsx        ← Dashboard root
        │   └── map/page.tsx    ← Map view
        └── components/
            ├── WorldMap.tsx        ← react-leaflet
            ├── ZoneChoropleth.tsx  ← GeoJSON + per-zone fill colour
            ├── FlowLayer.tsx       ← deck.gl arc layer (cross-zone flows)
            └── KpiTile.tsx
```

`app.yaml` (Databricks Apps spec) for Recipe A — two processes,
Next.js fronts traffic, proxies `/api/*` to FastAPI on a local port:

```yaml
command:
  # Apps runs both processes; foreman-style supervisor not needed —
  # Next.js proxies /api to localhost:8000 via next.config.js.
  - sh
  - -c
  - >
    uvicorn my_app.main:app --host 127.0.0.1 --port 8000 &
    node frontend/.next/standalone/server.js
env:
  - name: NODE_ENV
    value: production
  # Databricks Apps inject host + OBO token into the request headers
  # automatically; no static service-principal secret needed.
```

### Recipe B — Next.js full-stack (API routes + React)

Use when:
- The backend logic is thin — straight pass-through SQL +
  formatting, no ygg-heavy transforms.
- The team prefers one runtime / one language (TypeScript everywhere).
- The data access is via `@databricks/sql` (the official Node.js
  driver) directly from Next.js route handlers.

Repo layout:

```
my_app/
├── app.yaml
├── package.json                ← next, @databricks/sql, react-leaflet, deck.gl
├── next.config.js
└── src/
    ├── app/
    │   ├── page.tsx
    │   ├── map/page.tsx
    │   └── api/
    │       ├── kpis/route.ts          ← GET → JSON
    │       ├── map/route.ts           ← GET → GeoJSON FeatureCollection
    │       └── timeseries/route.ts    ← GET → JSON time series
    ├── lib/
    │   ├── databricks.ts              ← @databricks/sql client + OBO token forwarding
    │   └── arrow.ts                   ← apache-arrow IPC decode (optional)
    └── components/
        ├── WorldMap.tsx
        ├── ZoneChoropleth.tsx
        └── KpiTile.tsx
```

`app.yaml` for Recipe B — one process, simpler:

```yaml
command:
  - node
  - ./.next/standalone/server.js
env:
  - name: NODE_ENV
    value: production
```

### Picking between A and B

| Situation | Pick |
| --- | --- |
| ygg cast / schema / FxRate / GeoZoneCatalog on the read path | A — Python backend can call ygg directly. |
| Write-back triggers a Job task (`dbc.jobs.run(...)`) | A — ygg JobTask shape is the cleanest call site. |
| Backend is "SELECT, format, return JSON" only | B — fewer moving parts, one deploy unit. |
| Team is FE-heavy, BE-light | B. |
| Already have a FastAPI service serving other consumers (mobile, internal API) | A — reuse it. |
| Arrow IPC wire format for hot pre-aggregated tables | A — `pyarrow.ipc` from Python is well-trodden; the JS-only path through `@databricks/sql` returns rows, not Arrow. |

## Data access patterns

### Read path — SQL Warehouse with OAuth on-behalf-of

The app shouldn't hold a static service-principal token. Databricks
Apps inject the **user's OAuth on-behalf-of (OBO)** access token
into every incoming request as a header
(`X-Forwarded-Access-Token`). The backend forwards that token when
opening the SQL connection — RBAC on the underlying Delta tables /
warehouse carries through automatically.

**Recipe A** (FastAPI, the canonical pattern):

```python
# backend/src/my_app/deps.py
from fastapi import Depends, Header, HTTPException
from yggdrasil.databricks import DatabricksClient

def databricks_client(
    x_forwarded_access_token: str | None = Header(None),
    x_forwarded_host: str | None = Header(None),
) -> DatabricksClient:
    if not x_forwarded_access_token:
        raise HTTPException(401, "Missing OAuth on-behalf-of token. "
                                 "Are you running this outside Databricks Apps?")
    # DatabricksClient is singleton-by-config; same (host, token) →
    # same instance, so per-request creation is cheap.
    return DatabricksClient(
        host=f"https://{x_forwarded_host}",
        token=x_forwarded_access_token,
    )
```

```python
# backend/src/my_app/routes/kpis.py
from fastapi import APIRouter, Depends
from yggdrasil.databricks import DatabricksClient
from ..deps import databricks_client
from ..arrow_response import arrow_ipc_response

router = APIRouter(prefix="/api/kpis")

@router.get("/{task}")
def kpis(task: str, dbc: DatabricksClient = Depends(databricks_client)):
    # Read the pre-aggregated dash_kpis row-per-KPI table — no joins,
    # filtered server-side, returns Arrow IPC bytes.
    result = dbc.sql.execute(
        "SELECT kpi, value, unit, computed_at_utc "
        f"FROM main.{task}.dash_kpis "
        "WHERE computed_at_utc > current_timestamp() - INTERVAL 24 HOURS"
    )
    return arrow_ipc_response(result.read_arrow_table())
```

`arrow_ipc_response` — the helper every read endpoint uses:

```python
# backend/src/my_app/arrow_response.py
import io
import pyarrow as pa
from fastapi import Response

def arrow_ipc_response(table: pa.Table) -> Response:
    """Serialise a pa.Table as Arrow IPC stream and ship it.

    Per AGENTS.md §6b — when the same-schema rows cross the
    stage boundary, the unit IS an Arrow batch, not a JSON array.
    The JS side decodes with the apache-arrow package; the schema
    travels for free (field names, dtypes, nullability, nested
    structure all on the wire).
    """
    buf = io.BytesIO()
    with pa.ipc.new_stream(buf, table.schema) as writer:
        writer.write_table(table)
    return Response(
        content=buf.getvalue(),
        media_type="application/vnd.apache.arrow.stream",
    )
```

Frontend side (Next.js, both recipes):

```typescript
// frontend/src/lib/fetchArrow.ts
import { tableFromIPC, Table } from "apache-arrow";

export async function fetchArrow(url: string): Promise<Table> {
  const r = await fetch(url, { credentials: "include" });
  if (!r.ok) throw new Error(`${url}: ${r.status} ${r.statusText}`);
  return tableFromIPC(await r.arrayBuffer());
}
```

```typescript
// frontend/src/components/KpiTile.tsx
"use client";
import { useEffect, useState } from "react";
import { fetchArrow } from "@/lib/fetchArrow";

export function KpiTile({ task, kpi }: { task: string; kpi: string }) {
  const [value, setValue] = useState<number | null>(null);
  useEffect(() => {
    fetchArrow(`/api/kpis/${task}`).then((tbl) => {
      const row = tbl.toArray().find((r: any) => r.kpi === kpi);
      setValue(row?.value ?? null);
    });
  }, [task, kpi]);
  return <div className="kpi">{value !== null ? value.toFixed(2) : "…"}</div>;
}
```

**Recipe B** (Next.js full-stack, `@databricks/sql` driver):

```typescript
// src/lib/databricks.ts
import { DBSQLClient } from "@databricks/sql";
import { headers } from "next/headers";

export async function sqlClient() {
  const h = headers();
  const token = h.get("x-forwarded-access-token");
  const host  = h.get("x-forwarded-host");
  if (!token) throw new Error("Missing OBO token");
  const client = new DBSQLClient();
  await client.connect({
    host,
    path: process.env.DATABRICKS_WAREHOUSE_HTTP_PATH!,
    token,
    authType: "access-token",
  });
  return client;
}
```

```typescript
// src/app/api/kpis/[task]/route.ts
import { NextResponse } from "next/server";
import { sqlClient } from "@/lib/databricks";

export async function GET(
  _req: Request,
  { params }: { params: { task: string } },
) {
  const client = await sqlClient();
  const session = await client.openSession();
  const operation = await session.executeStatement(
    `SELECT kpi, value, unit, computed_at_utc
     FROM main.${params.task}.dash_kpis
     WHERE computed_at_utc > current_timestamp() - INTERVAL 24 HOURS`,
  );
  const rows = await operation.fetchAll();
  await operation.close();
  await session.close();
  return NextResponse.json(rows);
}
```

Recipe B's row shape is JSON over the wire (the Node driver
returns objects, not Arrow). Fine for KPI tiles (a handful of rows
per request); not what you want for the map / time-series
endpoints — those go through Arrow IPC. If Recipe B's app
**also** needs Arrow IPC for one endpoint, that's the signal you
should be on Recipe A.

### Write path — never call business logic from the app

The app **never** runs the ML training / scoring / curation logic
itself. Every write goes through a Job task already deployed
elsewhere; the app's role is to **trigger** the Job and surface its
status. This keeps RBAC, retry, audit, and observability where they
already live.

```python
# backend/src/my_app/routes/promote.py
@router.post("/promote-champion/{task}")
def promote(task: str, dbc: DatabricksClient = Depends(databricks_client)):
    # The job exists in the workspace, deployed via
    # `dbc.jobs.create_or_update(...)`. The app only invokes it.
    job = dbc.jobs.find(name=f"{task}-promote-champion")
    if job is None:
        raise HTTPException(404, f"No promote job for task={task!r}.")
    run = job.run(wait=False)  # async; the UI polls its status
    return {"run_id": run.run_id, "state": run.state}
```

Same idea for a re-run / recompute / counterfactual button — the
app is a thin shell over the Jobs API. **Don't** load ygg cast
helpers in the request handler to recompute a feature; that
duplicates the Job task's logic and drifts.

## World maps for trading KPIs

The whole point of building an App over an AI/BI Dashboard is
often the map: per-zone day-ahead clearing prices, cross-zone
flow arrows, country-level positions, vessel routes. The data is
already there in the curated layer (see
[`ygg-curated-views`](ygg-curated-views.md#3b-geographic-data--always-carry-latlon--optional-polygon)
— every curated row with a location reference carries
`lat: float64` + `lon: float64`, and zone polygons add
`boundary_geojson: string`).

### Pick the library

Two picks cover almost every trading-floor map use case:

| Library | When | Bundle | Renders |
| --- | --- | --- | --- |
| `react-leaflet` | Country / bidding-zone / region choropleths, simple markers, < 10 k features. The default for "show the EU power map with prices per zone". | ~150 KB | Raster basemap (OSM) + SVG vector overlays. |
| `deck.gl` | Large point clouds (vessel positions, generation assets, well sites), animated flow arcs, heatmaps, > 10 k features, GPU-accelerated. | ~500 KB | WebGL on `<canvas>`; pairs with `react-map-gl` for a basemap. |

Skip these unless you have a specific reason:

- **Mapbox GL JS / MapLibre GL JS** — vector basemaps with full styling control. Worth it if the design team requires a custom basemap; otherwise `react-leaflet` + free OSM tiles is enough.
- **kepler.gl** — great for exploratory analysis, not a production component.
- **react-simple-maps** — country-only, no basemap. Smaller bundle, but you give up the zoom/pan UX users expect.
- **Databricks SQL geo visual** — built-in but the AI/BI tile boundary is exactly what you're escaping by building the App.

### Country choropleth (ISO 3166)

For "colour each country by a metric", pair the curated table's
`country_iso` column with a world-atlas GeoJSON:

Backend endpoint (Recipe A):

```python
@router.get("/api/map/country-positions")
def country_positions(dbc: DatabricksClient = Depends(databricks_client)):
    result = dbc.sql.execute("""
        SELECT country_iso, exposure_eur
        FROM main.book.dash_exposure_by_country
        WHERE as_of_date = current_date()
    """)
    return arrow_ipc_response(result.read_arrow_table())
```

Frontend (Next.js + react-leaflet):

```typescript
// frontend/src/components/CountryChoropleth.tsx
"use client";
import { MapContainer, GeoJSON, TileLayer } from "react-leaflet";
import { useEffect, useState } from "react";
import { fetchArrow } from "@/lib/fetchArrow";
import worldGeo from "@/data/world-countries-110m.json"; // ne_110m_admin_0 dump

export function CountryChoropleth() {
  const [byIso, setByIso] = useState<Map<string, number>>(new Map());
  useEffect(() => {
    fetchArrow("/api/map/country-positions").then((tbl) => {
      const m = new Map<string, number>();
      for (const row of tbl.toArray()) m.set(row.country_iso, row.exposure_eur);
      setByIso(m);
    });
  }, []);

  const style = (feature: any) => {
    const iso = feature.properties.iso_a2;        // ISO 3166-1 alpha-2
    const v = byIso.get(iso) ?? 0;
    return {
      fillColor: colorFor(v),                     // diverging scale
      weight: 0.5, color: "#444", fillOpacity: 0.7,
    };
  };

  return (
    <MapContainer center={[50, 10]} zoom={4} style={{ height: 600 }}>
      <TileLayer
        attribution="© OpenStreetMap"
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <GeoJSON data={worldGeo as any} style={style} />
    </MapContainer>
  );
}
```

The GeoJSON fixture (`world-countries-110m.json`) ships in the
frontend repo — pulled once from
[Natural Earth](https://www.naturalearthdata.com/) or
[world-atlas](https://github.com/topojson/world-atlas). Don't refetch
it per render; it's static reference data.

### Bidding-zone / region choropleth (custom polygons)

For EU power's ENTSO-E bidding zones — or any region set that
isn't a country — the boundary lives in the **curated** layer
already: the `<catalog>.iso.bidding_zone` table ships
`boundary_geojson: string` per zone (resolved at curate time from
[`GeoZoneCatalog`](../../python/src/yggdrasil/data/enums/geozone/)).
The display layer inlines that column so the app doesn't have to
join.

```python
@router.get("/api/map/dayahead-prices")
def dayahead_prices(dbc: DatabricksClient = Depends(databricks_client)):
    result = dbc.sql.execute("""
        SELECT eic_code,
               zone_name,
               boundary_geojson,        -- inlined GeoJSON Feature string
               centroid_lat AS lat,
               centroid_lon AS lon,
               price,
               currency_iso
        FROM main.entsoe.dash_dayahead_prices_eu
        WHERE delivery_start_utc = (
            SELECT MAX(delivery_start_utc) FROM main.entsoe.dash_dayahead_prices_eu
        )
    """)
    return arrow_ipc_response(result.read_arrow_table())
```

Frontend assembles a `FeatureCollection` from the per-row
`boundary_geojson` strings:

```typescript
// frontend/src/components/ZonePriceMap.tsx
"use client";
import { MapContainer, GeoJSON, TileLayer, Tooltip } from "react-leaflet";
import { useMemo, useEffect, useState } from "react";
import { fetchArrow } from "@/lib/fetchArrow";

export function ZonePriceMap() {
  const [rows, setRows] = useState<any[]>([]);
  useEffect(() => {
    fetchArrow("/api/map/dayahead-prices").then((tbl) =>
      setRows(tbl.toArray() as any[])
    );
  }, []);

  const fc = useMemo(() => ({
    type: "FeatureCollection",
    features: rows.flatMap((r) => {
      if (!r.boundary_geojson) return [];
      const feature = JSON.parse(r.boundary_geojson);
      return [{
        ...feature,
        properties: {
          ...feature.properties,
          eic_code: r.eic_code,
          zone_name: r.zone_name,
          price: r.price,
          currency_iso: r.currency_iso,
        },
      }];
    }),
  }), [rows]);

  const style = (feature: any) => ({
    fillColor: priceColor(feature.properties.price),
    weight: 0.5, color: "#222", fillOpacity: 0.7,
  });

  const onEach = (feature: any, layer: any) => {
    layer.bindTooltip(
      `<b>${feature.properties.zone_name}</b><br/>` +
      `${feature.properties.price.toFixed(2)} ${feature.properties.currency_iso}/MWh`
    );
  };

  return (
    <MapContainer center={[50, 10]} zoom={4} style={{ height: 600 }}>
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      <GeoJSON data={fc as any} style={style} onEachFeature={onEach} />
    </MapContainer>
  );
}
```

The `boundary_geojson` column is GeoJSON text per the curated
convention — the frontend just `JSON.parse`s it; nothing to
re-project, no Spark on the client.

### Cross-zone flow arrows (deck.gl ArcLayer)

For power flow / LNG shipment / cross-border trade flows, render
arcs between origin and destination coordinates. The curated row
carries both endpoints as `geo_point` structs (`origin.lat /
origin.lon`, `destination.lat / destination.lon`):

```python
@router.get("/api/map/flows")
def flows(dbc: DatabricksClient = Depends(databricks_client)):
    result = dbc.sql.execute("""
        SELECT origin_zone,
               destination_zone,
               origin.lat   AS o_lat, origin.lon   AS o_lon,
               destination.lat AS d_lat, destination.lon AS d_lon,
               mw_flow,
               direction
        FROM main.entsoe.dash_cross_zone_flows
        WHERE delivery_start_utc = current_timestamp()::date
    """)
    return arrow_ipc_response(result.read_arrow_table())
```

```typescript
// frontend/src/components/FlowLayer.tsx
"use client";
import { useEffect, useState } from "react";
import DeckGL from "@deck.gl/react";
import { ArcLayer } from "@deck.gl/layers";
import { Map } from "react-map-gl/maplibre"; // free MapLibre basemap
import { fetchArrow } from "@/lib/fetchArrow";

export function FlowLayer() {
  const [rows, setRows] = useState<any[]>([]);
  useEffect(() => {
    fetchArrow("/api/map/flows").then((tbl) => setRows(tbl.toArray() as any[]));
  }, []);

  const layer = new ArcLayer({
    id: "flow-arcs",
    data: rows,
    getSourcePosition: (d: any) => [d.o_lon, d.o_lat],
    getTargetPosition: (d: any) => [d.d_lon, d.d_lat],
    getWidth: (d: any) => Math.log(Math.max(d.mw_flow, 1)) * 0.5,
    getSourceColor: (d: any) => [80, 200, 120],
    getTargetColor: (d: any) => [200, 80, 80],
  });

  return (
    <DeckGL
      initialViewState={{ longitude: 10, latitude: 50, zoom: 4 }}
      controller
      layers={[layer]}
    >
      <Map mapStyle="https://demotiles.maplibre.org/style.json" />
    </DeckGL>
  );
}
```

### Trading-KPI map shapes — what to render

A trading-floor map view typically combines two or three layers
on one canvas. Default tile set:

1. **Zone choropleth** — `dash_dayahead_prices_<region>` filled
   by current clearing price; tooltip carries
   `(zone_name, price, currency_iso/MWh, last_observed_utc)`.
2. **Position markers** — `analyst_<task>_positions_proposed`
   pinned at zone centroid; size encodes `notional_eur`, color
   encodes `direction` (long / short). Click pops a side-panel
   with the rationale.
3. **Flow arcs** — `dash_cross_zone_flows` rendered as deck.gl
   ArcLayer; thickness ∝ MW; colour gradient origin → destination.
4. **Single-number KPI strip** — pinned bottom of the map: book
   `pnl_today_eur`, `exposure_eur`, `var_95_eur`, all reading
   from `dash_book_kpis` (one KPI per row, per
   [`ygg-display-views`](ygg-display-views.md#kpi--aggregate-display-tables)).

Layout: KPI strip + map + table side-by-side. The map is the
primary surface; the table is a fallback when the user wants the
exact numbers.

## Performance — why apps beat SQL Warehouse for hot reads

The SQL Warehouse is great at "one analyst types ad-hoc SQL once a
minute". It is **not** great at "fifteen browser tabs polling the
same `dash_*` table every 5 seconds" — every query is a fresh
warehouse round-trip (~50–300 ms even on cluster-warm queries
against a small table).

Apps fix this two ways:

1. **In-process LRU on the backend.** A FastAPI dependency that
   memoises the last `N` Arrow tables, keyed by `(query, user_iso8601_hour)`,
   serves repeat hits in ~5 ms.

   ```python
   # backend/src/my_app/cache.py
   from yggdrasil.dataclasses import ExpiringDict
   import pyarrow as pa

   _ARROW_CACHE: ExpiringDict[tuple, pa.Table] = ExpiringDict(
       default_ttl=60.0, max_size=128,
   )

   def cached_query(dbc, sql: str, ttl: float = 60.0) -> pa.Table:
       """Memoise an Arrow result with an `ExpiringDict` per AGENTS.md
       "Reach for ExpiringDict for any concurrent / expiring cache"."""
       key = (sql,)
       hit = _ARROW_CACHE.get(key)
       if hit is not None:
           return hit
       tbl = dbc.sql.execute(sql).read_arrow_table()
       _ARROW_CACHE.set(key, tbl, ttl=ttl)
       return tbl
   ```

   Use `ExpiringDict` (`yggdrasil.dataclasses.ExpiringDict`) per
   the AGENTS.md rule — already process-wide thread-safe, picklable,
   bounded.

2. **Arrow IPC on the wire.** A pre-aggregated `dash_*` table with
   200 rows × 20 columns serialised as Arrow IPC is ~30 KB; the
   same data as JSON is ~120 KB. Across hundreds of map redraws
   per session, the bytes add up — and the JS parser walks Arrow
   columns at memcpy speed, not per-row `JSON.parse` allocation.

The pre-aggregation rule still applies: **never query a curated
table from the app** for high-frequency reads. Build a
`dash_<view>` first (see [`ygg-display-views`](ygg-display-views.md)),
point the app at it, refresh on a Job DAG. The app exists because
you wanted *more* control over UX than AI/BI gives — it doesn't
buy you out of the curated → display layering.

## Deploy

```bash
# Authenticate the Databricks CLI against the target workspace.
databricks auth login --host https://<workspace>.cloud.databricks.com

# Validate the app.yaml + run a local dev cycle.
databricks apps run-local --source ./my_app

# Ship to the workspace.
databricks apps deploy --source ./my_app --name my-trading-app
```

`databricks apps deploy` uploads the directory to
`/Workspace/Users/<you>/.bundles/apps/<name>/`, builds the runtime
image, and exposes the SPA at
`https://<name>-<workspace-id>.databricksapps.com`. Re-deploy is
in-place; rolling restart, no downtime configurable from the UI.

CI/CD: wrap the deploy in a GitHub Action or a Databricks Asset
Bundle (`databricks bundle deploy` against a
`databricks.yml`). Both work; pick whatever the rest of your jobs
use.

## Routing summary

| Need | Pick | Where |
| --- | --- | --- |
| Backend exposes ygg / Arrow IPC | Recipe A (FastAPI + Next.js) | This skill |
| Backend is thin SELECT → JSON | Recipe B (Next.js full-stack) | This skill |
| Read a `dash_*` table for a tile | `dbc.sql.execute(...)` + `arrow_ipc_response` | This skill |
| Trigger a Job from the UI | `dbc.jobs.find(name=...).run()` | [`ygg-databricks-jobs`](ygg-databricks-jobs.md) |
| Build the underlying `dash_*` table | [`ygg-display-views`](ygg-display-views.md) | — |
| ML candidate leaderboard / promote champion UI | [`ygg-modelist`](ygg-modelist.md) + this skill | — |
| Trader-facing decision map | [`ygg-energy-trading-analyst`](ygg-energy-trading-analyst.md) + this skill | — |
| Cache hot reads in the app process | `yggdrasil.dataclasses.ExpiringDict` | AGENTS.md |
| OBO auth header forwarding | `X-Forwarded-Access-Token` → `DatabricksClient(token=...)` | This skill |

## Don'ts

- **Don't embed a service-principal token in the app.** Databricks
  Apps inject the user's OBO token; use it. Static credentials
  defeat RBAC and audit.
- **Don't re-implement ygg logic on the frontend.** ygg is Python;
  cast / schema / FxRate / GeoZoneCatalog calls stay in the
  backend (Recipe A) or in the Job task the app triggers (Recipe B
  + write path).
- **Don't run heavy compute inside the request handler.** If a
  read takes > 1 s on the warehouse, materialise it as a
  `dash_*` table on a scheduled refresh, point the handler at the
  pre-aggregate. The app is a UI, not a compute layer.
- **Don't query curated directly from the app for production
  reads.** Curated is the silver layer; build the `dash_*` view,
  point the app there. AI/BI follows the same rule for the same
  reason.
- **Don't iterate rows on the JS side when you can pass a Polars /
  Arrow Table around.** `apache-arrow` JS has `.get(i)` / `.toArray()`
  for the cases you genuinely need a row; chart libraries
  (`recharts`, `visx`, `observable-plot`) eat column-shaped data
  directly. See AGENTS.md §6b.
- **Don't fetch the world GeoJSON per render.** It's static
  reference data; bundle it in the frontend (`world-atlas`,
  `naturalearth`). For dynamic boundaries (custom zones, regulator
  changes), ship them from a curated table column
  (`boundary_geojson`) and let the app cache the response.
- **Don't ship a write button without an audit row.** Every UI
  write should land in an `audit_app_actions` table with
  `(user_email, action, payload_json, performed_at_utc)`. Apps run
  outside the curated layer's normal provenance; the audit table
  is how you reconstruct who did what.
- **Don't put the SQL Warehouse path in client-side env vars.**
  The warehouse HTTP path goes through the backend; the frontend
  only knows `/api/*` endpoints. Keep credentials and infra IDs
  server-side.
