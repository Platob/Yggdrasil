# Skill: curated views over `raw_` tables — standardisation rules

## When to use

The user has a `raw_<entity>` table (vendor-shaped, immutable, see
[`ygg-data-modeling`](ygg-data-modeling.md)) and asks to "clean it
up", "build the curated layer", "expose this to analysts",
"standardise the timestamps", "harmonise country codes", "convert to
UTC", or "build a view I can join on". The curated layer is what
downstream consumers (BI, ML, other ingestion pipelines) read; the
raw layer is the audit / replay trail.

This skill covers the **rules** for the curated layer. The mechanics
(`Schema`, `Field`, ensure_created, PK/FK) live in
[`ygg-data-modeling`](ygg-data-modeling.md) — the rules below apply
to the curated tables those rules build.

## The curated contract

```
main.<source>.raw_<entity>        ← bronze   — source-shaped, immutable
└── main.<source>.<entity>         ← silver   — curated, standardised, stable
    └── main.<source>.dash_<view>   ← gold     — business-display, dashboard-shaped
```

The display layer (`dash_*`) is consumer-facing — wide, pivoted,
pre-aggregated, geometry inlined. Built on top of curated; built
**after** curated in the same ingestion Job DAG. See
[`ygg-display-views`](ygg-display-views.md).

A curated table / view:

1. **Standardises units.** UTC for timestamps, decimal for money, ISO
   codes for country / currency / language.
2. **Stable columns.** Renaming / dropping is a versioned change
   (`<entity>_v2`), not a silent migration.
3. **Documents every column.** Every `Field` carries `comment=...`.
   The raw layer can be terse; the curated layer can't.
4. **Has a primary key.** Even when the source doesn't, the curated
   layer picks one (often the source id, otherwise a synthetic).
5. **Joinable on ISO codes** — see
   [`ygg-data-modeling`](ygg-data-modeling.md#cross-schema-joins-via-shared-iso-dimensions).

Implement as either a Delta table (rewritten from `raw_` on each
load) or a SQL view (cheap, always fresh, slower at read time). Pick
the view when the standardisation is cheap (renames, casts), pick
the table when there's expensive joins / deduplication.

## Standardisation rules

### 1. Timestamps → UTC

```python
Field(
    "paid_at_utc",
    DataType.timestamp("UTC"),
    nullable=False,
    comment="Source paid_at, normalised to UTC.",
)
```

Cast in the curated SELECT:

```sql
SELECT
    CAST(paid_at AS TIMESTAMP)              AS paid_at_utc,        -- naive → assume UTC
    from_utc_timestamp(paid_at, src_tz)     AS paid_at_utc,        -- tz-tagged
    paid_at AT TIME ZONE 'UTC'              AS paid_at_utc         -- Delta-native
FROM main.<source>.raw_orders
```

The yggdrasil cast registry already does the right thing —
`convert(raw_table, target_schema)` with a `timestamp("UTC")` target
field handles naive → aware, tz-shift, and microsecond truncation
through one Arrow C++ pass. See [`ygg-cast`](ygg-cast.md).

Rules:

- All timestamp columns end in `_utc` unless they're business-local
  (rare; needs justification in the column comment).
- No `string` timestamps. The cast registry refuses to store
  `"2026-05-19T10:00:00Z"` as a string.
- One epoch column max (`event_time_utc`). Don't ship `created`,
  `created_at`, `creation_ts`, `time` all on the same table —
  consumers can't tell which to use.

### 2. Numbers → decimal for money, int for counts

```python
Field(
    "amount",
    DataType.decimal(18, 2),       # money — never float64
    nullable=False,
    comment="Order total, in `currency_iso` units.",
)
Field(
    "item_count",
    DataType.int32(),              # countable → integer, narrowest fit
    nullable=False,
)
Field(
    "fx_rate",
    DataType.decimal(18, 8),       # rate — high scale, fixed precision
    nullable=False,
)
```

Rules:

- `decimal(18, 2)` for currency amounts. `(18, 8)` for FX rates.
  `(36, 18)` for crypto.
- Integer counts → `int32` (`int64` only when you legitimately need
  > 2 G).
- Booleans → `bool`, not `int(0|1)`, not `string("Y"|"N")`. The cast
  registry handles the demote.
- Percentages → `decimal(5, 4)` (range -1 .. 1) **or** `decimal(7, 4)`
  for "0 .. 100" — document which in the field comment.

### 3. Country / currency / language → ISO codes

```python
from yggdrasil.data.enums import Currency
from yggdrasil.data.enums.geozone.catalog import GeoZoneCatalog

# At curate time, parse the source value through the right enum.
Currency.parse("usd").code            # → "USD"
Currency.parse("Euro").code           # → "EUR"
GeoZoneCatalog.default().parse("Germany").country_iso  # → "DE"
```

In the schema:

```python
Field(
    "currency_iso",                          # column name carries the standard
    DataType.string(),
    nullable=False,
    comment="ISO 4217 — see yggdrasil.data.enums.Currency.",
    tags={"cluster_by": True},               # most queries filter on it
)
Field(
    "country_iso",
    DataType.string(),
    nullable=True,
    comment="ISO 3166-1 alpha-2 — see yggdrasil.data.enums.geozone.",
)
Field(
    "region_iso",
    DataType.string(),
    nullable=True,
    comment="ISO 3166-2 subdivision code (e.g. 'FR-75').",
)
Field(
    "language_iso",
    DataType.string(),
    nullable=True,
    comment="ISO 639-1 two-letter code.",
)
Field(
    "timezone_iana",
    DataType.string(),
    nullable=True,
    comment="IANA timezone name (e.g. 'Europe/Paris'). "
            "See yggdrasil.data.enums.Timezone.from_(...).",
)
```

Column naming convention: `<concept>_<standard>` so the column name
itself signals which catalog to join (`country_iso` → `main.iso.country`).

### 3b. Geographic data → always carry lat/lon (+ optional polygon)

Anything with a place attached gets **renderable** coordinates in
the curated layer — a frontend, BI plugin, or notebook map widget
(folium, kepler.gl, pydeck, Databricks SQL geo visual) should be
able to plot rows without a second lookup.

Rules:

- **Add `lat` and `lon` to every curated table that ships a location
  reference**, even when the source only ships an opaque code
  (`country_iso`, `city_name`, `eic_code`, `mic_iso`). Resolve at
  curate time via `yggdrasil.data.enums.geozone.GeoZoneCatalog`
  (every `GeoZone` carries `lat` + `lon`).
- **For zone / region / polygon shapes, add `boundary_geojson`** —
  a string column holding a GeoJSON `Feature` or `FeatureCollection`
  so plugins can draw the polygon. Stored as `string` (Delta has no
  native geometry type; downstream Spark/Polars/plotly accept GeoJSON).
- **For a list of points** (route, time-series of positions),
  emit `points` as a `list<struct<lat: float64, lon: float64,
  observation_utc: timestamp("UTC")>>` so each cell is itself
  renderable.
- **Coordinate types:** `lat: float64 [-90, 90]`, `lon: float64
  [-180, 180]`. Don't store as `decimal` — frontends and geo
  libraries expect `float64` / WGS84.

Two equivalent shapes — pick whichever reads better at the call site:

**Flat columns** — simplest when the row has one location:

```python
from yggdrasil.data import Field, DataType

Field("lat", DataType.from_("float64"), nullable=True,
      metadata={b"comment": b"WGS84 latitude [-90, 90]. "
                            b"Resolved at curate time from country_iso / eic_code / city_name."})
Field("lon", DataType.from_("float64"), nullable=True,
      metadata={b"comment": b"WGS84 longitude [-180, 180]."})
```

**`geo_point()` struct** — preferred when the row has multiple
locations (`origin` / `destination`) or a downstream sink speaks
the `{"lat": …, "lon": …}` shape natively:

```python
from yggdrasil.data import geo_point

geo_point("origin",      comment="Pickup point, WGS84.")
geo_point("destination", comment="Drop-off point, WGS84.")
```

`geo_point("name")` returns a `Field` whose dtype is the canonical
`GEO_POINT_TYPE` (a `StructType` of two non-nullable `float64`
children). `nullable=True` by default — `nullable=False` on the
parent struct prevents NULL rows, while the child non-nullable
contract prevents partial coordinates (one of the two NULL). The
type is a singleton (same instance for every `geo_point()` call) so
the cast registry hits its exact-match fast path on every read.
`is_geo_point_type(dtype)` matches the shape if you need to detect
it downstream (e.g. when emitting a GeoJSON encoder).

**Zone polygons** — shade the area on a frontend map:

```python
Field("boundary_geojson", DataType.from_("string"), nullable=True,
      metadata={b"comment": b"GeoJSON Feature/FeatureCollection. "
                            b"Used by map plugins (kepler.gl, pydeck, folium)."})
```

**Tracks / time-series of points** — vessel routes, wind patterns,
sensor sweeps:

```python
# Each point in the list is independently renderable.
Field("points",
      DataType.from_(
          "list<struct<lat: float64, lon: float64, observation_utc: timestamp(UTC)>>"
      ),
      nullable=True,
      metadata={b"comment": b"Ordered WGS84 track."})
```

Resolve coords from the curated dim:

```python
# At curate time, look up lat/lon from the shared geozone catalog.
from yggdrasil.data.enums.geozone.catalog import GeoZoneCatalog

cat = GeoZoneCatalog.default()
def lookup_coords(country_iso: str) -> tuple[float | None, float | None]:
    z = cat.parse(country_iso)
    return (z.lat, z.lon) if z else (None, None)
```

For `main.iso.country` / `main.iso.region` / `main.iso.exchange`
shared dims, **always include `lat` + `lon` + `boundary_geojson`**
so any curated table that FK-joins them inherits the geo columns
without re-resolving:

```sql
SELECT o.*, c.lat AS country_lat, c.lon AS country_lon, c.boundary_geojson
FROM main.vendor_orders.orders o
JOIN main.iso.country c USING (country_iso)
```

Polygon sources:

- ENTSO-E bidding zones — `entsoe-py` / ENTSO-E transparency ships
  the EIC boundaries as GeoJSON.
- Countries — Natural Earth (public domain) at `1:50m` is the
  default; load once, write to `main.iso.country`.
- ISO 3166-2 subdivisions — same Natural Earth source.
- MIC exchanges — single point (exchange HQ city's lat/lon) is
  enough; boundary not meaningful.

### 4. Strings → trimmed, normalised, enum-validated

The raw layer keeps the source's exact bytes. The curated layer:

- **Trims** leading / trailing whitespace.
- **Lower-cases** identifiers that are case-insensitive in the
  domain (email, URL host).
- **Validates** against an enum / lookup. If `status` should be one
  of `("pending", "paid", "refunded")`, enforce it via a CHECK
  constraint or refuse the load.
- **Splits** structured strings: `"Paris, FR"` → `(city, country_iso)`.

Build a CHECK constraint via a constraint `Field` (see
[`ygg-data-modeling`](ygg-data-modeling.md)) when the universe is
small and known:

```python
Field(
    "status",
    DataType.string(),
    nullable=False,
    metadata={"check": "status IN ('pending', 'paid', 'refunded')"},
    tags={"constraint_key": True},
)
```

### 5. NULLs are real values — pick one missing-value rule

The raw layer mirrors the source (often: empty string vs missing key
vs literal `null` all coexist). The curated layer collapses them to
one:

```sql
SELECT
    NULLIF(TRIM(currency), '')   AS currency_iso,   -- empty string → NULL
    ...
```

Then `nullable=True` on the field. Don't ship a curated `string`
column where some rows are `""` and others are `NULL` for "missing".

### 6. Identifiers → typed + commented

```python
Field(
    "order_id",
    DataType.string(),
    nullable=False,
    comment="ULID — 26 char Crockford-base32, monotonic.",
    tags={"primary_key": True},
)
```

Document the format. `int` ids → `int64` and document the issuer.
UUIDs → `string` with `comment="UUID v4 hex (lowercase, no hyphens)"`.

## End-to-end example

Raw landing (vendor shape):

```python
RAW_ORDERS_SCHEMA = Schema.from_fields([
    Field("order_id",   DataType.string(), nullable=False,
              tags={"primary_key": True}),
    Field("created",    DataType.string(),  nullable=False,
              comment="Source `created` field — ISO-8601 string, +offset varies."),
    Field("amount",     DataType.float64(), nullable=False),
    Field("ccy",        DataType.string(),  nullable=False),
    Field("country",    DataType.string(),  nullable=True),
    # ... + provenance columns ...
])

dbc.table("main.vendor_orders.raw_orders").ensure_created(
    schema=RAW_ORDERS_SCHEMA,
)
```

Curated standardisation:

```python
ORDERS_SCHEMA = Schema.from_fields([
    Field(
        "order_id", DataType.string(), nullable=False,
        comment="Vendor order id. ULID format.",
        tags={"primary_key": True},
    ),
    Field(
        "created_at_utc", DataType.timestamp("UTC"), nullable=False,
        comment="Order creation time, normalised to UTC.",
        tags={"partition_by": True},
    ),
    Field(
        "amount", DataType.decimal(18, 2), nullable=False,
        comment="Order total in `currency_iso`.",
    ),
    Field(
        "currency_iso", DataType.string(), nullable=False,
        comment="ISO 4217. Joinable on main.iso.currency(iso_alpha3).",
        metadata={"references": "main.iso.currency(iso_alpha3)"},
        tags={"foreign_key": True, "cluster_by": True},
    ),
    Field(
        "country_iso", DataType.string(), nullable=True,
        comment="ISO 3166-1 alpha-2 (NULL when vendor didn't ship one).",
        metadata={"references": "main.iso.country(iso_alpha2)"},
        tags={"foreign_key": True},
    ),
])

dbc.table("main.vendor_orders.orders").ensure_created(schema=ORDERS_SCHEMA)
```

The standardisation SQL (rebuild-on-load shape):

```python
from yggdrasil.data.enums import Currency

dbc.sql.execute("""
    INSERT OVERWRITE main.vendor_orders.orders
    SELECT
        order_id,
        CAST(created AS TIMESTAMP) AT TIME ZONE 'UTC'    AS created_at_utc,
        CAST(amount AS DECIMAL(18, 2))                    AS amount,
        UPPER(NULLIF(TRIM(ccy), ''))                      AS currency_iso,
        c.iso_alpha2                                      AS country_iso
    FROM main.vendor_orders.raw_orders r
    LEFT JOIN main.iso.country_aliases c
           ON UPPER(TRIM(r.country)) = c.alias_upper
""")
```

Vendor-specific normalisation (the `country_aliases` join) lives in
the curated query; the raw layer kept the original bytes for replay.

## Curated table or curated view?

| Pick a Delta **table** when… | Pick a SQL **view** when… |
| --- | --- |
| Standardisation is expensive (joins, dedup, window functions). | Standardisation is renames + casts only. |
| Downstream consumers expect predictable scan costs. | Read traffic is low or always recent. |
| You want to `OPTIMIZE` / `ZORDER` / Liquid-cluster the result. | `raw_` is already partitioned the right way. |
| You need point-in-time snapshots (`VERSION AS OF`). | History doesn't matter beyond `raw_`. |

Default to **table** for high-traffic curated entities. Views are a
fine starting point but every "let's just create the view" decision
eventually owes a rewrite.

## Don'ts

- Don't ship a curated column called `created`, `time`, `timestamp`,
  `date` — too ambiguous. Use `created_at_utc`, `processed_on_date`.
- Don't keep `float64` for money in the curated layer because "the
  source ships float". The cast registry will demote with strict
  precision checking; let it raise.
- Don't fork standardisation logic per consumer ("BI wants
  `country_full_name`, ML wants `country_iso`"). Ship `country_iso`
  in curated; downstream joins `main.iso.country` for the full name.
- Don't migrate the curated schema in place with column renames.
  Ship `<entity>_v2` and let consumers cut over.
- Don't omit `comment=` on a curated column. The curated layer is
  the documentation.
- Don't put curated and raw in the same table with a `is_curated`
  flag. Separate tables, separate contracts.
