# Skill: energy trading analyst — forecasts, signals, decisions, FX + cross-zone risk

## When to use

The user is acting as a **business / trading analyst** for an energy
commodity book (power, gas, oil, or any combination) and asks to
"produce a trading signal", "build a forecast", "size a position",
"compute P&L attribution", "stress-test the book", "hedge FX
exposure", "aggregate exposure by zone / country", "flag a
cross-border spread opportunity", "publish today's trade ideas",
"build the morning analyst pack". The user is not ingesting raw
data — they sit downstream of the data engineer and one step
upstream of the trader / decision-maker.

This is the **shared** analyst skill — read it first. Then route to
the commodity-specific skill for the desk you're working on:

| Desk | Skill |
| --- | --- |
| Power (day-ahead, intraday, balancing, capacity) | [`ygg-trading-analyst-power`](ygg-trading-analyst-power.md) |
| Natural gas + LNG | [`ygg-trading-analyst-gas`](ygg-trading-analyst-gas.md) |
| Crude oil + refined products | [`ygg-trading-analyst-oil`](ygg-trading-analyst-oil.md) |

Builds on [`ygg-curated-views`](ygg-curated-views.md) (the canonical
read surface), [`ygg-display-views`](ygg-display-views.md) (the
`dash_*` tables the analyst consumes), [`ygg-trading-commodity`](ygg-trading-commodity.md)
(the trading domain conventions the data engineer landed), and
[`ygg-mlops`](ygg-mlops.md) / [`ygg-modelist`](ygg-modelist.md)
(when the forecast needs a real model).

## Don't invent things — ask, or use a placeholder

This is the **hard rule** across every analyst, modelist, and
data-engineering skill: **never reference a data table, vendor
source, schema, column, catalog, or upstream API by a concrete name
unless you can point at where it actually lives** — a Unity Catalog
table that exists, a yggdrasil module that ships it, a vendor docs
URL, or a user-provided spec in this session.

What this means in practice:

- **If you don't know the table name, say so.** Use a placeholder
  (`<curated_day_ahead_table>`, `<your_weather_feature_table>`) and
  add a one-line note: *"Confirm the actual name with the data
  engineer / check `dbc.tables.list(...)`."*
- **If the user names a vendor / source you haven't seen wired up,
  ask.** "Is `main.entsoe.dayahead` already curated, or do we need
  to add it to the ingestion pipeline first?" beats writing SQL
  against a table that doesn't exist.
- **Refined specialised notions are fair game.** Units (`EUR/MWh`,
  `USD/bbl`, `USD/MMBtu`, `GBp/therm`), ISO standards (4217 / 3166
  / 10383 / EIC), commodity universes (TTF, NBP, Henry Hub, JKM,
  Brent, WTI, Dubai), market conventions (day-ahead / intraday /
  balancing, contango / backwardation, crack 3-2-1, spark / dark
  spread), and yggdrasil's own API surface (`yggdrasil.fxrate.FxRate`,
  `GeoZoneCatalog`, `DataField`, `Schema`, `DataType`,
  `DatabricksClient`) — those are real, ship in the library, and
  can be referenced freely.
- **Always be able to explain it.** If you write a feature, a
  formula, a metric, a column name into a skill or a notebook,
  you must be able to point at the source (vendor docs, well-known
  market convention, yggdrasil module, user-supplied spec). If
  you can't, drop it and ask for validation.
- **Output schemas the analyst / modelist OWN are fair to specify.**
  `analyst_<task>_signals`, `analyst_<task>_positions_proposed`,
  `ml_<task>_run_metrics` — these are *conventions for tables this
  role creates*, not assumptions about pre-existing inputs. Speccing
  the output is the point of the skill.

When in doubt, ask the user — "Which curated table is the
day-ahead clearing on your workspace?" / "Do you have a weather
feed already curated, or do we need to add one?" — before writing
code or SQL that pretends it exists.

## Where the analyst sits in the stack

```
<raw ingestion>            ← data engineer (yggdrasil ingestion + ygg-trading-commodity)
└── <curated entity>        ← data engineer (yggdrasil cast + ygg-curated-views)
    ├── <dashboard view>     ← data engineer (ygg-display-views)
    ├── <ml feature store>   ← MODELIST (yggdrasil + mlflow; ygg-modelist)
    └── <analyst artifacts>  ← ANALYST output (this skill)
        ├── analyst_<task>_signals
        ├── analyst_<task>_positions_proposed
        ├── analyst_<task>_pnl
        └── analyst_<task>_risk
        └── (downstream) trader consumes
```

Hard rules:

1. **Read-only on curated / dash / ml-feature tables.** The analyst
   never `INSERT`s into them; corrections go through the data
   engineer's ingestion pipeline or the modelist's training task.
2. **`analyst_*` is the analyst's namespace.** Co-locate output
   under the same schema as the inputs the task depends on so
   lineage is local; don't carve out a global `analyst.*` catalog.
3. **Decisions are artifacts, not notebooks.** Every recommendation
   lands in `analyst_<task>_signals` or `analyst_<task>_positions_proposed`
   so it's queryable, joinable to realised P&L, and auditable.
4. **Same standardisation as curated** — UTC timestamps, decimal
   money, ISO codes, lat/lon on geo-bearing rows. See
   [`ygg-curated-views`](ygg-curated-views.md).

## The analyst's feature universe (conceptual)

Energy alpha rarely comes from one feature class — the production
analyst pulls from all of these. Each class corresponds to a
curated table family the data engineer lands; **confirm the actual
table names with the data engineer before writing SQL**.

| Feature class | What it drives | Refined specialised notions |
| --- | --- | --- |
| **Prices / curves** | Auto-regressive base, term-structure trades, vol regime | OHLCV bars, settles, forward / futures curves; native units per market (EUR/MWh, USD/bbl, USD/MMBtu, GBp/therm, USD/gal); ISO 10383 MIC |
| **FX rates** | Reporting-currency P&L, cross-currency hedge, arb gating | ISO 4217 pairs; spot vs forward; `yggdrasil.fxrate.FxRate` is the canonical access (it actually ships) |
| **Weather** | Power demand, renewable supply, gas heating load, cooling load | Temperature (HDD/CDD on 18.3 °C / 65 °F basis), wind speed at hub height, solar GHI/DNI/DHI, precipitation, dewpoint, pressure |
| **Storage / inventory** | Gas / oil fundamentals, hydro power | Working volume vs working capacity, z-score vs N-year normals, days-of-cover |
| **Freight & shipping** | LNG / crude arb economics, port-congestion premium | Worldscale (WS) points + flat rates, USD/MMBtu / USD/bbl conversions, vessel class (VLCC / Suezmax / Aframax / LNGC TFDE / MEGI), choke-point routing |
| **Refinery / pipeline ops** | Crack spreads, regional differentials, supply disruption | Capacity (kb/d), utilisation %, outage severity scale, planned vs unplanned |
| **Carbon / emissions** | Clean spark / dark, cross-jurisdiction competitiveness | EUA / CCA / RGGI / UK ETS prices; tonnes of CO₂ per MWh by fuel (~0.36 t/MWh CCGT, ~0.90 t/MWh hard-coal) |
| **Macro & rates** | Term-structure regime, USD-denominated commodity beta | DXY index, 2y / 10y nominal, 5y / 10y real yields, central-bank policy meetings |
| **Calendar / holidays** | Day-of-week, seasonality, contract-roll dates | DST gaps in delivery hours, holiday calendars per jurisdiction, last-trade-day per contract |
| **Geopolitics / outages** | Event-driven repricing, supply-shock signals | Event type, severity (1..5), affected zones / grades, duration estimate |
| **Geography** | Cross-zone exposure aggregation, freight routing | ISO 3166 country, ENTSO-E EIC bidding zone, gas hubs by country, refining regions (USGC / NWE / SING / MED), shipping lanes + choke points (Suez / Hormuz / Malacca / Panama / Bab-el-Mandeb) |
| **Position / book state** | Inventory risk, VaR, concentration limits, crowding | Net vs gross by zone / country / tenor / currency, mandate limits |

Two operating rules across the universe:

1. **One feature class = one curated namespace.** Don't sprinkle
   weather columns onto a price table; analysts join. The data
   engineer keeps them separate; you keep them joinable via shared
   ISO dimensions.
2. **Persist what you used.** A signal row stores `model_uri` or
   `rule_name`; a position row stores `fx_rate_used` and the
   feature vector hash. Otherwise back-tests don't reproduce.

### Weather — feature shapes the analyst expects

For an EU power book you want, per zone, per delivery hour
(materialised by the data engineer / the [weather modelist](ygg-modelist-weather.md)):

| Column shape | Refined notion | Notes |
| --- | --- | --- |
| `temp_c_pop_weighted` | Demand basis | Population-polygon-weighted, HDD/CDD reference. |
| `wind_speed_ms_capacity_weighted` | Wind supply | Installed-capacity-polygon-weighted, ideally at hub height (~100 m). |
| `ghi_wm2_capacity_weighted` | Solar supply | Global horizontal irradiance, capacity-weighted. |
| `precip_mm_basin` | Hydro refill | Per hydrological basin (Nordic / Alpine power). |
| `temp_anomaly_c` | Seasonality-adjusted demand | Temp vs 30y rolling normal. |
| `forecast_age_hours` | Trustworthiness | Older NWP runs get downweighted. |

For gas: replace zone-weighted by **demand-region-weighted**; for
oil, weather matters less for crude but a lot for distillates
(heating oil ≈ degree-days × inventory).

### Freight & shipping — the cross-hub arbitrage gate

A "TTF–JKM arb is open" signal is only actionable when **delivered**
LNG cost — freight included — beats the local clearing price. The
data the analyst needs the data engineer to land:

- Per-route freight cost in the destination's price unit
  (`USD/MMBtu` for LNG, `USD/bbl` for crude tankers).
- Vessel utilisation per class (signal-actionable filter when
  `utilisation > 90 %`).
- Port / terminal congestion (dwell days) at origin and destination.
- Boil-off loss percentage per vessel class.
- Route polyline / choke-points crossed (for chokepoint exposure
  aggregation).

Worldscale → USD/bbl conversion math (oil / dirty tanker market):

```
freight_usd_bbl = WS_route_points * flat_rate_usd_t / 100 / bbl_per_tonne
```

`bbl_per_tonne ≈ 7.33` for crude (varies by API); flat rates are
published annually. Don't recompute in signal SQL — build the
converter into the curated freight table or its `dash_*` view.

### Storage / inventory — the slow signal

Two transforms the analyst desk reuses across every storage
report (gas, oil, hydro reservoir):

- `z_score_vs_5y` — same calendar day across 5 years, not last-5
  points.
- `days_of_cover = inventory / recent_avg_demand`.

Bake both into the curated layer once; don't reinvent in every
signal SQL.

### Carbon — the cross-commodity tax

`co2_<jurisdiction>_<unit>` (EUR/t for EUA, USD/t for CCA / RGGI /
UK ETS). Required for any clean-spark / clean-dark / fuel-switching
signal. Don't treat carbon as power-only — gas-export / refining
jurisdictions carry embedded carbon obligation too.

### Macro & rates — regime knob, not point feature

Two derived quantities used as gating, not as a forecasting input:

- DXY (USD index) — crude beta; strong USD → weak USD-denominated
  commodity prices.
- Real yield (`nominal_5y - tips_5y`) — storage-cost arbitrage
  band; cheap real money widens contango.

### Calendar / holidays

`is_trading_day`, `is_holiday`, `holiday_name`, `dst_offset_hours`
per relevant jurisdiction. Day-of-week and hour-of-day come from
the **delivery** timestamp, not observation. Contract-roll dates
(last-trade-day of the front month) trigger an automatic conviction
haircut on proposals expiring within `N` days.

### Geopolitics & outages — event feed

Schema the analyst wants in the event table (negotiate the actual
column names with the data engineer):

- Stable `event_id`, `event_utc`, `event_type`, `severity` (1..5),
  `affected_zones` / `affected_grades`, `source_url`, optional
  `estimated_duration_iso`.

Two signal patterns:

1. **Event filter on existing signals.** When `affected_zone`
   overlaps a proposed-position zone, downweight conviction or pin
   a "manual review required" flag.
2. **Pure event-driven signals.** A `severity >= 4` outage in a
   capacity-meaningful facility triggers a regional crack / spread
   signal — but only when the market hasn't already moved
   `+2σ` in the prior session.

### Position / book state — yes, the book is a feature

Crowding signals: when the desk is already at maximum mandate on
FR-DE spread, the marginal signal flips from "open more" to "trim".
The signal job aggregates current book by zone / country / tenor
before scoring.

## Standard `analyst_*` output tables

Each desk lands these four artifacts per task. The schemas below
are the **convention this role establishes** — they are the
analyst's own output, not assumed pre-existing inputs. Adapt names
to your catalog; the *shape* is what matters.

### `analyst_<task>_signals` — model / rule output

```python
from yggdrasil.data import Field, DataType, Schema

ANALYST_SIGNALS_SCHEMA = Schema.from_fields([
    Field("signal_id", DataType.string(), nullable=False,
          tags={"primary_key": True},
          comment="Stable per (task, entity_id, observation_utc, model_version)."),
    Field("task", DataType.string(), nullable=False,
          comment="'price_forecast' | 'spread_signal' | 'curve_arb' | 'vol_regime'."),
    Field("entity_id", DataType.string(), nullable=False,
          tags={"cluster_by": True},
          comment="What the signal is on — bidding zone EIC, contract code, hub, refinery."),
    Field("observation_utc", DataType.timestamp("UTC"), nullable=False,
          tags={"partition_by": True}),
    Field("horizon_iso", DataType.string(), nullable=False,
          comment="ISO duration ('PT1H', 'P1D', 'P30D'). Empty string for spot signals."),
    Field("direction", DataType.string(), nullable=False,
          comment="'long' | 'short' | 'flat'."),
    Field("conviction", DataType.decimal(5, 4), nullable=False,
          comment="0..1 model confidence / rule strength. Calibrated per task."),
    Field("expected_value", DataType.decimal(28, 10), nullable=True,
          comment="Forecast point estimate in `currency_iso` per unit."),
    Field("expected_lower", DataType.decimal(28, 10), nullable=True,
          comment="Lower bound of forecast interval (e.g. p5)."),
    Field("expected_upper", DataType.decimal(28, 10), nullable=True),
    Field("currency_iso", DataType.string(), nullable=False),
    Field("unit", DataType.string(), nullable=False,
          comment="'MWh' | 'bbl' | 'MMBtu' | 'therm' — instrument's native unit."),
    Field("model_uri", DataType.string(), nullable=True,
          comment="MLflow URI when ML-derived. NULL when rule-based."),
    Field("rule_name", DataType.string(), nullable=True,
          comment="Rule identifier when not ML."),
    Field("rationale", DataType.string(), nullable=True,
          comment="One-line human-readable why. Truncate to ~200 chars."),
    # Provenance — _ingested_at is the analyst run timestamp, not source.
    Field("_ingested_at",  DataType.timestamp("UTC"), nullable=False,
          tags={"primary_key": True}),
    Field("_source",       DataType.string(),  nullable=False,
          comment="'analyst:<task>'."),
    Field("_payload_hash", DataType.string(),  nullable=False),
    Field("_batch_id",     DataType.string(),  nullable=False),
])
```

### `analyst_<task>_positions_proposed` — sized trade idea

```python
ANALYST_POSITIONS_SCHEMA = Schema.from_fields([
    Field("proposal_id", DataType.string(), nullable=False,
          tags={"primary_key": True}),
    Field("signal_id", DataType.string(), nullable=False,
          tags={"foreign_key": True}),
    Field("instrument", DataType.string(), nullable=False,
          comment="Contract code, hub+tenor, or spread legs joined by '|'."),
    Field("mic_iso", DataType.string(), nullable=True,
          comment="Listing exchange when applicable. ISO 10383."),
    Field("direction", DataType.string(), nullable=False),
    Field("quantity", DataType.decimal(28, 4), nullable=False,
          comment="Position size in `unit`. Lots / MWh / bbl / MMBtu."),
    Field("unit", DataType.string(), nullable=False),
    Field("entry_target", DataType.decimal(28, 10), nullable=False),
    Field("stop_loss", DataType.decimal(28, 10), nullable=True),
    Field("take_profit", DataType.decimal(28, 10), nullable=True),
    Field("currency_iso", DataType.string(), nullable=False),
    Field("base_currency_iso", DataType.string(), nullable=False,
          comment="Book's reporting currency. FX exposure = instrument ccy ≠ base ccy."),
    Field("notional_base", DataType.decimal(28, 4), nullable=False),
    Field("fx_rate_used", DataType.decimal(20, 10), nullable=True,
          comment="Rate from FxRate.latest at proposal time. NULL when ccy==base."),
    Field("fx_pair_iso", DataType.string(), nullable=True,
          comment="'EURUSD' shape."),
    Field("eic_code", DataType.string(), nullable=True),
    Field("country_iso", DataType.string(), nullable=True),
    Field("expiration_utc", DataType.timestamp("UTC"), nullable=True),
    Field("proposed_at_utc", DataType.timestamp("UTC"), nullable=False,
          tags={"partition_by": True}),
    Field("rationale", DataType.string(), nullable=True),
    # + provenance, _source = 'analyst:<task>'
])
```

### `analyst_<task>_pnl` — realised + unrealised attribution

One row per `(proposal_id, observation_utc)`. Columns:
`gross_pnl_<ccy>`, `fx_pnl_<base_ccy>`, `carry_pnl_<base_ccy>`,
`total_pnl_<base_ccy>`. The split between `gross` and `fx` is the
whole point — without it the analyst can't tell a winning view
from a winning FX move.

### `analyst_<task>_risk` — book-level exposure snapshots

One row per `(snapshot_utc, dimension, bucket)`. Dimensions:
`zone` (EIC / hub), `country_iso`, `currency_iso`, `tenor_iso`,
`commodity`. Buckets are the values inside the dimension. Columns:
`net_quantity`, `gross_quantity`, `delta_base_ccy`, `var_95_base_ccy`,
`concentration_pct` (this bucket's % of book gross).

## FX risk — always route through `yggdrasil.fxrate`

Energy is multi-currency by default — most EU power / gas hubs
quote EUR, NBP quotes GBp, ICE Brent / NYMEX / JKM quote USD. A
book reported in one currency carries FX risk on every position
outside it.

**Don't roll an FX lookup.** The library ships `yggdrasil.fxrate.FxRate`
— one `HTTPSession` singleton, multi-source fallback, multi-source
cached, polars-frame output. Use it.

### Convert proposal notionals to base currency

```python
from yggdrasil.fxrate import FxRate

fx = FxRate()  # singleton — call freely, pool / cache stay shared

def to_base_ccy(amount: float, ccy: str, base_ccy: str) -> tuple[float, float | None]:
    """Returns (notional_in_base, fx_rate_used). fx_rate_used is NULL when ccy==base."""
    if ccy == base_ccy:
        return amount, None
    rate = fx.latest([(ccy, base_ccy)]).select("value").item()
    return amount * rate, rate
```

Persist both `notional_base` and `fx_rate_used` on every
`analyst_<task>_positions_proposed` row — the rate is what you
captured at proposal time, not a "current" lookup.

### Historical re-statement (back-tests, attribution)

For the P&L / risk job that re-prices a window in base currency:

```python
hist = fx.fetch(
    pairs=[("EUR", "USD"), ("GBP", "USD"), ("NOK", "USD")],
    start="<window-start>",
    end="<window-end>",
    sampling="1d",
)
# Long-format polars frame: source, target, from_timestamp,
# to_timestamp, sampling, value.
```

**Cadence rule.** Day-ahead and longer use `sampling="1d"`. Intraday
power / gas (sub-hour) gets a `fx.latest()` snapshot pinned at the
proposal time and stored on the row; do not chase intraday FX ticks
for a settlement-on-D+2 trade — the FX move over those 18h is not
your alpha.

### Hedge ratio for cross-currency commodity exposure

When the book is long an EUR-denominated commodity and reports in
USD, aggregate the EUR notional and propose the matching FX leg.
The hedge itself lands in a separate task (`task='fx_hedge'`,
`instrument='EURUSD-spot'` / `'EURUSD-3m-fwd'`) so the trader can
execute it next to the commodity trades.

## Cross-geo-zone risk

Energy markets are geographically fragmented in a way oil / equities
aren't — neighbouring power bidding zones are different instruments
with congestion-bounded correlation; gas hubs are bounded by
pipeline capacity and LNG freight; oil grades are bounded by
refinery acceptance and tanker routes.

### Aggregate by zone, country, continent

`yggdrasil.data.enums.geozone.GeoZoneCatalog` is the canonical
bidirectional index. EIC ↔ ISO 3166-1 alpha-2 ↔ country name ↔
centroid lat/lon are all one lookup away — and it actually ships
in the library, so you can use it without coordinating with the
data engineer.

```python
from yggdrasil.data.enums.geozone import load_geozones

zones = load_geozones(include_countries=True)  # cached after first call

def zone_to_country(eic: str) -> str | None:
    zone = zones.lookup(eic)
    return zone.country_iso if zone else None
```

For the `analyst_<task>_risk` snapshot, drive the bucketing off this
catalog rather than a hand-rolled `CASE WHEN`. Confirm the actual
joinable shared-ISO dimension table with the data engineer before
writing the SQL.

### Cross-zone spread positions are two-leg, one rationale

A trade like "long zone A base, short zone B base" is one *idea*;
persist it as **two `analyst_<task>_positions_proposed` rows** that
share a `signal_id`, with opposite `direction` and the *same*
`entry_target` expressed as a spread. The `rationale` on both rows
references the spread thesis ("FR-DE 1.50 EUR/MWh, mean-reverting
to 0.60").

Don't try to encode "two legs in one row" — every downstream
aggregation (zone risk, country risk, VaR) treats one row as one
exposure; collapsing two legs hides the gross.

### Choke-point dimension on freight-dependent trades

LNG and crude-tanker trades crossing Suez / Hormuz / Malacca /
Panama / Bab-el-Mandeb are correlated *through the chokepoint*,
not through origin or destination. Add `route_choke_points`
(`array<string>`) to the proposal row when the trade is
route-dependent, and bucket the `analyst_<task>_risk` snapshot by
chokepoint as well as by country.

### Geo concentration limits

The `dash_<task>_risk` view (built downstream of `analyst_<task>_risk`
by the data engineer / display-views skill) surfaces concentration %
and flags when one zone / country / currency / chokepoint crosses
the book's mandate (typically 20–30%). The flag is a column on
the dash table, not a notebook log line.

## Building a forecast vs. a rule

| Pattern | Use it when | Lands in |
| --- | --- | --- |
| **Rule** (`spread > X`, `realised_vol > P90`, `inventory < 5y_min`) | Signal is mechanical, no historical fit needed, < 5 features. | `analyst_<task>_signals` with `rule_name`, `model_uri=NULL`. |
| **Statistical forecast** (`AutoARIMA`, `AutoETS`) | Univariate / few-feature horizon ≤ 30d, no regime breaks expected. | MLflow run + UC registry + `analyst_<task>_signals` row per scoring window. |
| **ML forecast** (`lightgbm`, `NHITS`, `TFT`) | Multi-feature, regime-aware, ≥ 90d of history. | Same as statistical, owned by the modelist (see [`ygg-modelist`](ygg-modelist.md)). |
| **Ensemble** | Production. Multiple of the above, blended. | One signal row per ensemble output, `model_uri` points at the wrapper. |

All four shapes land in the same `analyst_<task>_signals` schema —
`rule_name` and `model_uri` are mutually exclusive but the rest of
the row is the same.

## Wiring the analyst job into the DAG

Analyst tasks run **after** curated lands, on the same Job DAG (see
[`ygg-databricks-job-workflows`](ygg-databricks-job-workflows.md)).
Pattern shape (task keys are the convention; the actual upstream
task keys depend on the data engineer's DAG — confirm):

```python
job = dbc.jobs.create_or_update(name="<desk>_<task>_analyst", tasks=[])

# Upstream — data engineer's responsibility, shown only for the dep chain.
# (task keys here are placeholders — match the actual keys the data
# engineer uses on this DAG)

signals   = job.pytask(score_signals,    task_key="signals",
                       depends_on=["<curate_task>"]).create()
positions = job.pytask(size_positions,   task_key="positions",
                       depends_on=["signals"]).create()
risk      = job.pytask(refresh_risk,     task_key="risk",
                       depends_on=["positions"]).create()
pnl       = job.pytask(refresh_pnl,      task_key="pnl",
                       depends_on=["positions"]).create()
dash      = job.pytask(rebuild_analyst_dash, task_key="dash",
                       depends_on=["risk", "pnl"]).create()
```

Compute split (per CLAUDE.md "Pick compute by workload type"):
signal scoring + position sizing + risk + P&L + dashboard refresh
are all UC-Delta jobs → **serverless** (`environment_key=DEFAULT_ENVIRONMENT_KEY`).
Only the ingestion upstream needs classic compute.

## Trader-facing surfaces — dashboard vs Databricks App

The trader consumes through one of two surfaces; pick by
interactivity, the same way the modelist does:

| Surface | When |
| --- | --- |
| **AI/BI Dashboard** over `dash_analyst_<task>_*` | Read-only morning pack; embedded share link; "scroll and filter" UX. |
| **Databricks App** ([Next.js + FastAPI or Next.js full-stack](ygg-databricks-apps.md)) | Trader needs a **world map** with per-zone clearing prices + position markers + cross-border flow arcs on one canvas; or a button that re-scores / promotes a signal; or sub-second tile renders the SQL Warehouse can't deliver. |

When the desk's map view matters — EU bidding-zone choropleth
coloured by today's price, position pins at zone centroids
sized by `notional_eur`, deck.gl arc layer for cross-zone
flows — graduate to a Databricks App. The curated geo columns
this skill (and [`ygg-curated-views`](ygg-curated-views.md#3b-geographic-data--always-carry-latlon--optional-polygon))
already ship (`lat`/`lon`, `boundary_geojson`, `geo_point`)
are exactly what `react-leaflet` and `deck.gl` consume; no
extra ingestion needed. Full recipe (backend split, OAuth OBO,
Arrow IPC wire format, map plugins, trading-KPI tile set, deploy)
lives in [`ygg-databricks-apps`](ygg-databricks-apps.md).

## Picking the analyst's reporting currency

The book's `base_currency_iso` is a configuration knob, not a
per-task one. Pick once per analyst desk (typically `EUR` for
EU power / gas, `USD` for global crude / LNG, `GBP` for UK gas)
and propagate everywhere. Store it in a per-desk config table the
tasks read at start-up so re-denominating the whole book is a
single-row update.

## Don'ts

- **Don't invent a curated table name** to make example SQL run.
  Placeholder + a note ("confirm with the data engineer") is the
  rule. See [§ Don't invent things](#dont-invent-things--ask-or-use-a-placeholder).
- Don't query the raw landing layer from an analyst task. That's
  bronze, for provenance / audit only.
- Don't compute FX inline via `requests` / a hand-rolled rate dict.
  Use `yggdrasil.fxrate.FxRate` — it exists.
- Don't store an FX-converted price without persisting the rate.
- Don't aggregate exposure by vendor-specific zone codes. Resolve
  via `GeoZoneCatalog` and the shared ISO dimensions.
- Don't collapse a two-leg spread into one row.
- Don't write trade ideas to a notebook output cell. The artifact
  is `analyst_<task>_positions_proposed`.
- Don't ship a signal without `rationale`.
- Don't auto-execute. The analyst layer ends at `positions_proposed`.
- Don't mix commodities in one `analyst_<task>_*` table.
- Don't reach for an ML model when a rule fits. Default to a rule;
  promote to a model when residuals look forecastable.
