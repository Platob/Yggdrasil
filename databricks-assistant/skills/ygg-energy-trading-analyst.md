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
| Natural gas + LNG (TTF, NBP, Henry Hub, JKM) | [`ygg-trading-analyst-gas`](ygg-trading-analyst-gas.md) |
| Crude oil + refined products (Brent, WTI, Dubai, RBOB, ULSD) | [`ygg-trading-analyst-oil`](ygg-trading-analyst-oil.md) |

Builds on [`ygg-curated-views`](ygg-curated-views.md) (the canonical
read surface), [`ygg-display-views`](ygg-display-views.md) (the
`dash_*` tables the analyst consumes), [`ygg-trading-commodity`](ygg-trading-commodity.md)
(the trading domain conventions the data engineer landed), and
[`ygg-mlops`](ygg-mlops.md) (when the forecast needs a real model).

## Where the analyst sits in the stack

```
raw_<entity>      ← data engineer ingests it (ygg-trading-commodity)
└── <entity>       ← curated, the analyst's ONLY input surface
    ├── dash_<view> ← display, also analyst-readable
    └── analyst_<task>_* ← analyst's OUTPUT (signals / positions / P&L / risk)
        └── (downstream) trader consumes, optionally promotes to OMS
```

Hard rules:

1. **Read-only on `<entity>` / `dash_*`.** The analyst never `INSERT`s
   into curated; corrections go through the data engineer's ingestion
   pipeline.
2. **`analyst_*` is the analyst's namespace.** Same schema as the
   data source (`main.<source>.analyst_<task>_<artifact>`) so the
   analyst's output co-locates with the inputs it depends on. No
   shared `analyst.*` catalog — keeps lineage local.
3. **Decisions are artifacts, not notebooks.** Every recommendation
   lands in `analyst_<task>_signals` or `analyst_<task>_positions_proposed`
   so it's queryable, joinable to realised P&L, and auditable.
4. **Same standardisation as curated** — UTC timestamps, decimal
   money, ISO codes, lat/lon on geo-bearing rows. See
   [`ygg-curated-views`](ygg-curated-views.md).

## The analyst's feature universe

Energy alpha rarely comes from one feature class — the production
analyst pulls from **all** of these and blends in a signal model.
Each feature class is a curated table the data engineer lands; the
analyst consumes it, doesn't ingest it.

| Feature class | Curated tables (typical) | What it drives |
| --- | --- | --- |
| **Prices / curves** | `<source>.dayahead`, `<source>.futures`, `<source>.ohlcv_*` | Auto-regressive base, term-structure trades, vol regime |
| **FX rates** | `<source>.fx_spot` + `yggdrasil.fxrate.FxRate` live | Reporting-currency P&L, cross-currency hedge, arb |
| **Weather** | `weather.<provider>.forecast / actual` (temp, wind, solar GHI, precip, dewpoint, pressure) | Power demand, renewable supply, gas heating load, cooling load |
| **Storage / inventory** | `eia.weekly_petroleum`, `agsi.gas_storage`, `entso-e.reservoir`, `iea.oecd_oil_stocks` | Gas / oil fundamentals, hydro power |
| **Freight & shipping** | `baltic.bdi_bdti_bcti`, `spark.lng_freight`, `flexport.ais_vessels`, `eia.crude_tanker_rates` | LNG arb economics, crude-by-tanker delivered cost, port-congestion premium |
| **Refinery / pipeline ops** | `genscape.refinery_status`, `argus.pipeline_flows`, `iir.outages` | Crack spreads, regional differentials, supply disruption |
| **Carbon / emissions** | `eex.eua_settle`, `ice.cca_wci`, `epa.rggi_auction` | Clean spark / dark, cross-jurisdiction power competitiveness |
| **Macro & rates** | `fred.dgs2_dgs10_dxy`, `ecb.rates`, `cot.reports` | Term-structure regime, positioning crowding, USD-denominated commodity beta |
| **Calendar / holidays** | `<source>.calendar` (built-in `pandas_market_calendars` mirror) | Day-of-week, holiday, summer / winter seasonality, contract roll dates |
| **Geopolitics / outages** | `enex.events`, `ogj.refinery_news`, `entso-e.unplanned_outages` | Event-driven repricing, supply-shock signals |
| **Geography** | `iso.bidding_zone`, `iso.gas_hub`, `iso.exchange`, `iso.shipping_lane` + `GeoZoneCatalog` | Cross-zone exposure aggregation, freight routing |
| **Position / book state** | `analyst_<task>_positions_proposed` joined to OMS fills | Inventory risk, VaR, concentration limits |

Two operating rules across the universe:

1. **One feature class = one curated namespace.** Don't sprinkle
   weather columns onto a price table; analysts join. The data
   engineer's [`ygg-data-modeling`](ygg-data-modeling.md) keeps
   them separate; you keep them joinable via shared dims.
2. **Persist what you used.** A signal row stores `model_uri` or
   `rule_name`; a position row stores `fx_rate_used`. Same rule
   for the feature snapshot — the analyst job writes
   `ml_<task>_features` with the *materialised* feature vector,
   not "look it up at scoring time". Otherwise back-tests don't
   reproduce.

### Weather — the feature class energy people consistently underweight

For an EU power book you want, per zone, per delivery hour:

| Column | Source | Notes |
| --- | --- | --- |
| `temp_c_pop_weighted` | gridded temp × population polygon | HDD / CDD basis. |
| `wind_speed_ms_capacity_weighted` | gridded wind × installed-capacity-weighted polygon | Wind supply driver. |
| `ghi_wm2_capacity_weighted` | gridded GHI × solar-installed-capacity | Solar supply driver. |
| `precip_mm_basin` | per hydrological basin | Reservoir refill (Nordic / Alpine power). |
| `dewpoint_c`, `humidity_pct` | gridded | Cooling-load driver in summer. |
| `temp_anomaly_c` | temp vs 30y rolling normal | Seasonality-adjusted demand. |
| `forecast_age_hours` | timestamp of last NWP run | Trustworthiness; older runs get downweighted. |

Build per-zone polygon weighting **once** in `main.weather.dash_zone_features`
— a join key of `(eic_code, observation_utc)` is all the analyst
task needs. Don't re-weight inside every signal SQL.

For gas: replace zone-weighted by **demand-region-weighted**
(`gas_demand_region_id` from `main.iso.gas_demand_region`); for oil,
weather matters less for crude but a lot for distillates (heating
oil = degree-days × inventory).

### Freight & shipping — the cross-hub arbitrage gate

A "TTF–JKM arb is open" signal is only actionable when **delivered**
LNG cost — freight included — beats the local clearing price.

| Field | Curated table |
| --- | --- |
| `lng_freight_usd_mmbtu` per route (USGC→TFDE→NW Europe, USGC→NE Asia, Qatar→NW Europe, …) | `main.spark.lng_freight` |
| `crude_tanker_rate_ws` (Worldscale, dirty / clean) | `main.baltic.tanker` |
| `port_congestion_days` per terminal | `main.flexport.ais_dwell` |
| `boiloff_pct` | constant per vessel class, but on the freight row | as above |
| `route_geojson` | shipping-lane polyline in WGS84 | `main.iso.shipping_lane` |

Compute "delivered cost" in the dash layer, not at signal time:

```sql
-- main.gas.dash_lng_delivered_cost
SELECT
    f.route_from_port, f.route_to_port,
    f.observation_utc,
    src.price + f.lng_freight_usd_mmbtu + f.boiloff_pct * src.price AS delivered_cost_usd_mmbtu,
    dst.price                                                       AS local_clearing_usd_mmbtu,
    dst.price - (src.price + f.lng_freight_usd_mmbtu
                + f.boiloff_pct * src.price)                        AS arb_usd_mmbtu,
    f.route_geojson
FROM main.spark.lng_freight f
JOIN main.gas.dash_hub_prices src ON src.hub_name = f.origin_hub
JOIN main.gas.dash_hub_prices dst ON dst.hub_name = f.dest_hub
WHERE src.observation_utc = f.observation_utc
  AND dst.observation_utc = f.observation_utc
```

The geo aggregation rule from
[`#cross-geo-zone-risk`](#cross-geo-zone-risk) extends here: a
trade that sits "long delivered LNG to NW Europe" is exposed to
the **route polyline**, not just origin and destination — a Suez
closure stresses the entire route. Persist the polyline on the
proposal row when the trade is route-dependent.

### Storage / inventory — the slow signal

Weekly storage reports drive 5-day-ahead repricing on every gas /
oil curve, and seasonally drive every power hydro market:

- **EU gas (AGSI / GIE).** Weekly fill % per country, % of 5y-avg.
- **US gas (EIA Weekly Natural Gas Storage Report).** Thursday 10:30 ET.
- **US crude (EIA Weekly Petroleum Status).** Wednesday 10:30 ET.
- **OECD oil stocks (IEA monthly).** First week of month.
- **Hydro reservoir (NVE, Statnett, EDF).** Daily; Nordics + alpine.

Two analyst-facing transforms:

| Transform | Why |
| --- | --- |
| `z_score_vs_5y` | "Stocks at -2σ" beats "stocks at 95 bcm" as a feature. |
| `days_of_cover = inventory / recent_avg_demand` | Comparable across hubs, normalises for size. |

Both land in `main.<source>.dash_inventory` and are read by every
signal task on the desk. Avoid the temptation to hard-code 5y
windows in each signal SQL.

### Carbon — the cross-commodity tax

EUA (EEX, `eex.eua_settle`), California CCA, RGGI, UK ETS — the
right curated table joined as `co2_<jurisdiction>_eur_t` makes
clean-spark / clean-dark and gas-vs-coal switching trivial. Don't
treat carbon as power-only: a 2026-onwards US-gas-export project
carries embedded carbon obligation in some jurisdictions.

### Macro & rates — the slow regime knob

Two derived columns the analyst desk uses without overthinking it:

| Column | Source | Used for |
| --- | --- | --- |
| `dxy_index` (`fred.dgs10_dxy`) | DXY USD index | Crude beta — strong DXY → weak USD-denominated commodity prices. |
| `real_yield_5y` (`fred.dgs5 - fred.tipsy5`) | TIPS-implied real | Storage-cost arbitrage band. Cheap storage = wider contango. |

For most signals this is a *regime filter* (don't fade contango when
real yields are sub-zero) rather than a forecasting feature; persist
on the dash but treat as gating logic, not point input.

### Calendar / holidays

`main.<source>.calendar` carries `(date, is_trading_day, is_holiday,
holiday_name, dst_offset_hours)` per relevant jurisdiction. **Use
it on every signal**, not just for the obvious holiday filter:

- Day-of-week and hour-of-day from the *delivery* timestamp, not the
  observation timestamp.
- Contract-roll dates (last-trade-day of the front month) for futures
  — proposals expiring within `N` days of roll get an automatic
  conviction haircut.
- DST handling — see [`ygg-trading-commodity#time-series-hygiene`](ygg-trading-commodity.md#time-series-hygiene).
  Curated lands it correctly; the analyst just consumes
  `delivery_start_utc`.

### Geopolitics & outages — the event feed

`main.enex.events` / `main.iir.outages` is the event-typed feed: 
`event_id`, `event_utc`, `type` (`refinery_outage` / `pipeline_disruption`
/ `sanctions_announcement` / `weather_warning` / `strike` / …),
`affected_zone_eics` (`array<string>`), `severity` (1..5), `source_url`.
Two signal patterns:

1. **Event filter on existing signals.** When `affected_zone` overlaps
   a proposed-position zone, downweight conviction or pin a "manual
   review required" flag.
2. **Pure event-driven signals.** A `severity >= 4` outage in a 
   capacity-meaningful refinery triggers a regional crack-spread 
   long signal — but only when the crack hasn't already moved 
   `+2σ` in the prior session (the market saw it first).

Both flows land in the same `analyst_<task>_signals` schema; 
event-driven rows carry `rule_name='event_<type>_<id>'` and 
`rationale='outage at <facility>: <severity>/5'`.

### Position / book state — yes, the book is a feature

Crowding signals: when the desk is already long FR-DE spread at
maximum mandate, the marginal signal flips from "open more" to
"trim". The current book is read from `analyst_<task>_positions_proposed`
joined to the OMS fill feed (`<source>.fills`); the signal job
aggregates by zone / country / tenor before scoring. Don't propose
beyond mandate.

## Standard `analyst_*` output tables

Each desk lands these four artifacts per task. Names are stable;
the columns inside scale with the task.

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
          comment="'long' | 'short' | 'flat'. Lowercase, alpha only."),
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
          comment="Rule identifier when not ML — 'spark_spread_gt_5' etc."),
    Field("rationale", DataType.string(), nullable=True,
          comment="One-line human-readable why. Truncate to ~200 chars."),
    # Provenance — same shape as raw_/curated. _ingested_at is the
    # ANALYST run timestamp, not the source data timestamp.
    Field("_ingested_at",  DataType.timestamp("UTC"), nullable=False,
          tags={"primary_key": True}),
    Field("_source",       DataType.string(),  nullable=False,
          comment="'analyst:<task>' — distinguishes from data-engineering provenance."),
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
          tags={"foreign_key": True},
          metadata={"references": "main.<source>.analyst_<task>_signals(signal_id)"}),
    Field("instrument", DataType.string(), nullable=False,
          comment="Contract code, hub+tenor, or spread legs joined by '|'."),
    Field("mic_iso", DataType.string(), nullable=True,
          comment="Listing exchange when applicable. ISO 10383."),
    Field("direction", DataType.string(), nullable=False),
    Field("quantity", DataType.decimal(28, 4), nullable=False,
          comment="Position size in `unit`. Lots / MWh / bbl / MMBtu."),
    Field("unit", DataType.string(), nullable=False),
    Field("entry_target", DataType.decimal(28, 10), nullable=False,
          comment="Price at which the proposal is actionable, in `currency_iso`."),
    Field("stop_loss", DataType.decimal(28, 10), nullable=True),
    Field("take_profit", DataType.decimal(28, 10), nullable=True),
    Field("currency_iso", DataType.string(), nullable=False),
    Field("base_currency_iso", DataType.string(), nullable=False,
          comment="Book's reporting currency. FX exposure = instrument ccy ≠ base ccy."),
    Field("notional_base", DataType.decimal(28, 4), nullable=False,
          comment="Notional in `base_currency_iso` at the FX rate captured below."),
    Field("fx_rate_used", DataType.decimal(20, 10), nullable=True,
          comment="Rate from FxRate.latest at proposal time. NULL when ccy==base."),
    Field("fx_pair_iso", DataType.string(), nullable=True,
          comment="'EURUSD' shape, only when `fx_rate_used` is set."),
    Field("eic_code", DataType.string(), nullable=True,
          comment="Bidding zone (power) or delivery hub zone (gas / oil)."),
    Field("country_iso", DataType.string(), nullable=True,
          comment="ISO 3166-1 alpha-2 — for cross-country exposure aggregation."),
    Field("expiration_utc", DataType.timestamp("UTC"), nullable=True),
    Field("proposed_at_utc", DataType.timestamp("UTC"), nullable=False,
          tags={"partition_by": True}),
    Field("rationale", DataType.string(), nullable=True),
    # Provenance + _source = 'analyst:<task>'
])
```

### `analyst_<task>_pnl` — realised + unrealised attribution

Joins proposals to realised fills (when the OMS writes back) and to
the curated mark-to-market price; one row per `(proposal_id,
observation_utc)`. Columns: `gross_pnl_<ccy>`, `fx_pnl_<base_ccy>`,
`carry_pnl_<base_ccy>`, `total_pnl_<base_ccy>`. The split between
`gross` and `fx` is the whole point — without it the analyst can't
tell a winning view from a winning FX move.

### `analyst_<task>_risk` — book-level exposure snapshots

One row per `(snapshot_utc, dimension, bucket)`. Dimensions:
`zone` (EIC / hub), `country_iso`, `currency_iso`, `tenor_iso`,
`commodity`. Buckets are the values inside the dimension. Columns:
`net_quantity`, `gross_quantity`, `delta_base_ccy`, `var_95_base_ccy`,
`concentration_pct` (this bucket's % of book gross). Refreshed by
the same job that lands proposals — see "Wiring" below.

## FX risk — always route through `yggdrasil.fxrate`

Energy is multi-currency by default — EEX / EPEX / Nord Pool quote
EUR, NBP quotes GBp, ICE Brent / NYMEX quote USD, Asian LNG (JKM)
quotes USD. A book reported in one currency carries FX risk on
every position outside it.

**Don't roll an FX lookup.** The library ships `yggdrasil.fxrate.FxRate`
— one `HTTPSession` singleton, multi-source fallback, multi-source
cached, polars-frame output. Use it.

### Convert proposal notionals to base currency

```python
from yggdrasil.fxrate import FxRate

fx = FxRate()  # singleton — call FxRate() freely, the pool / cache stay shared

def to_base_ccy(amount: float, ccy: str, base_ccy: str) -> tuple[float, float | None]:
    """Returns (notional_in_base, fx_rate_used). fx_rate_used is NULL when ccy==base."""
    if ccy == base_ccy:
        return amount, None
    rate = fx.latest([(ccy, base_ccy)]).select("value").item()
    return amount * rate, rate
```

Persist both `notional_base` and `fx_rate_used` on every
`analyst_<task>_positions_proposed` row — the rate is what you
captured at proposal time, not a "current" lookup. P&L attribution
later joins on `fx_pair_iso = concat(currency_iso, base_currency_iso)`
to compute the realised FX move.

### Historical re-statement (back-tests, attribution)

For the P&L / risk job that re-prices a window in base currency:

```python
hist = fx.fetch(
    pairs=[("EUR", "USD"), ("GBP", "USD"), ("NOK", "USD")],
    start="2026-01-01",
    end="2026-05-19",
    sampling="1d",
)
# Long-format polars frame: source, target, from_timestamp, to_timestamp,
# sampling, value. Pivot to wide per pair when joining on a price table.
```

**Cadence rule.** Day-ahead and longer use `sampling="1d"`. Intraday
power / gas (sub-hour) gets a `fx.latest()` snapshot pinned at the
proposal time and stored on the row; do not chase intraday FX ticks
for a settlement-on-D+2 trade — the FX move over those 18h is not
your alpha.

### Hedge ratio for cross-currency commodity exposure

When the book is long 100 MWh of FR power (EUR) and reports in USD:

```sql
-- Net EUR-denominated exposure for the book.
WITH eur_book AS (
  SELECT
      sum(notional_base) FILTER (WHERE currency_iso = 'EUR') AS notional_usd_from_eur,
      sum(quantity * fx_rate_used) FILTER (WHERE currency_iso = 'EUR') AS eur_notional_usd_equiv
  FROM main.entsoe.analyst_da_signals_positions_proposed
  WHERE proposed_at_utc >= current_date() - INTERVAL 1 DAY
)
SELECT eur_notional_usd_equiv AS eur_hedge_size_usd FROM eur_book
```

The hedge proposal itself lands in `analyst_fx_hedge_positions_proposed`
(`task='fx_hedge'`, `instrument='EURUSD-spot'` / `'EURUSD-3m-fwd'`)
so the trader can execute it next to the commodity trades.

## Cross-geo-zone risk

Energy markets are geographically fragmented in a way oil / equities
aren't — a French day-ahead position and a German one are different
instruments with congestion-bounded correlation. Three patterns:

### Aggregate by zone, country, continent

`GeoZoneCatalog` (`yggdrasil.data.enums.geozone.GeoZoneCatalog`) is
the canonical bidirectional index. EIC ↔ ISO 3166-1 alpha-2 ↔
country name ↔ centroid lat/lon are all one lookup away.

```python
from yggdrasil.data.enums.geozone import load_geozones

zones = load_geozones(include_countries=True)  # cached after first call

def zone_to_country(eic: str) -> str | None:
    zone = zones.lookup(eic)
    return zone.country_iso if zone else None
```

For the `analyst_<task>_risk` snapshot, drive the bucketing off this
catalog rather than a hand-rolled `CASE WHEN`:

```sql
-- Net exposure rolled up to country, joined to the shared dim
-- so the analyst inherits lat/lon for the map view in dash_*.
INSERT OVERWRITE main.entsoe.analyst_da_risk
SELECT
    current_timestamp()                    AS snapshot_utc,
    'country_iso'                          AS dimension,
    bz.country_iso                         AS bucket,
    sum(p.quantity * sign_dir(p.direction)) AS net_quantity,
    sum(abs(p.quantity))                    AS gross_quantity,
    sum(p.notional_base * sign_dir(p.direction)) AS delta_base_ccy,
    -- VaR computed upstream by the risk task; null here when not run.
    NULL                                    AS var_95_base_ccy,
    sum(abs(p.notional_base)) /
        nullif(sum(abs(p.notional_base)) OVER (), 0)  AS concentration_pct
FROM main.entsoe.analyst_da_positions_proposed p
LEFT JOIN main.iso.bidding_zone bz USING (eic_code)
WHERE p.proposed_at_utc >= current_date() - INTERVAL 1 DAY
GROUP BY bz.country_iso
```

### Cross-zone spread positions are two-leg, one rationale

A trade like "long FR base, short DE base" is one *idea*; persist it
as **two `analyst_<task>_positions_proposed` rows** that share a
`signal_id`, with opposite `direction` and the *same* `entry_target`
expressed as a spread. The `rationale` on both rows references the
spread thesis ("FR-DE 1.50 EUR/MWh, mean-reverting to 0.60").

Don't try to encode "two legs in one row" — every downstream
aggregation (zone risk, country risk, VaR) treats one row as one
exposure; collapsing two legs hides the gross.

### Geo concentration limits

The `dash_<task>_risk` view (built downstream of `analyst_<task>_risk`)
surfaces the concentration % and flags when one zone / country /
currency crosses the book's mandate (typically 20–30%). The flag
is a column on the dash table, not a notebook log line — that's how
risk officers consume it.

## Building a forecast vs. a rule

| Pattern | Use it when | Lands in |
| --- | --- | --- |
| **Rule** (`spread > X`, `realised_vol > P90`, `inventory < 5y_min`) | Signal is mechanical, no historical fit needed, < 5 features. | `analyst_<task>_signals` with `rule_name`, `model_uri=NULL`. |
| **Statistical forecast** (`AutoARIMA`, `AutoETS`) | Univariate / few-feature horizon ≤ 30d, no regime breaks expected. | MLflow run + UC registry + `analyst_<task>_signals` row per scoring window. |
| **ML forecast** (`lightgbm`, `NHITS`, `TFT`) | Multi-feature, regime-aware, ≥ 90d of history. | Same as statistical, via [`ygg-mlops`](ygg-mlops.md). |
| **Ensemble** | Production. Multiple of the above, blended. | One signal row per ensemble output, `model_uri` points at the wrapper. |

All four shapes land in the same `analyst_<task>_signals` schema —
`rule_name` and `model_uri` are mutually exclusive but the rest of
the row is the same. Means the trader's downstream consumer reads
one shape.

## Wiring the analyst job into the DAG

Analyst tasks run **after** curated lands, on the same Job DAG (see
[`ygg-databricks-job-workflows`](ygg-databricks-job-workflows.md)):

```python
job = dbc.jobs.create_or_update(name="entsoe_da_analyst", tasks=[])

# Upstream — data engineer's responsibility, not the analyst's.
# (referenced here only to show the dependency chain)
ingest    = job.pytask(ingest_da_raw,     task_key="ingest").create()
curate    = job.pytask(rebuild_da_curated, task_key="curate",
                       depends_on=["ingest"]).create()

# Analyst tasks — parallel after curate.
signals   = job.pytask(score_da_signals,  task_key="signals",
                       depends_on=["curate"]).create()
positions = job.pytask(size_da_positions, task_key="positions",
                       depends_on=["signals"]).create()
risk      = job.pytask(refresh_da_risk,   task_key="risk",
                       depends_on=["positions"]).create()
pnl       = job.pytask(refresh_da_pnl,    task_key="pnl",
                       depends_on=["positions"]).create()

# Dash refresh sits at the tail — the trader's morning pack.
dash      = job.pytask(rebuild_da_analyst_dash, task_key="dash",
                       depends_on=["risk", "pnl"]).create()
```

The serverless-vs-classic compute split applies: signal scoring +
position sizing + risk + P&L are all UC-Delta jobs → **serverless**
(`environment_key=DEFAULT_ENVIRONMENT_KEY`). The data ingestion
upstream is the only piece that needs the classic cluster (FX +
ENTSO-E HTTP egress). See `CLAUDE.md` → "Pick compute by workload
type".

## Picking the analyst's reporting currency

The book's `base_currency_iso` is a configuration knob, not a
per-task one. Pick once per analyst desk (typically `EUR` for
EU power / gas, `USD` for global crude / LNG, `GBP` for UK gas)
and propagate everywhere. Store it in a per-desk config table
(`main.<source>.analyst_config`) the tasks read at start-up so
re-denominating the whole book is a single-row update.

## Don'ts

- Don't query `raw_<entity>` from an analyst task. That's the bronze
  layer — provenance / audit only. Read curated.
- Don't compute FX inline via `requests` / a hand-rolled rate dict.
  Use `yggdrasil.fxrate.FxRate` — the fallback chain, retry policy,
  cache, and geography enrichment already exist.
- Don't store an FX-converted price without persisting the rate.
  `fx_rate_used` + `fx_pair_iso` on the proposal row is what makes
  the back-test possible six months later.
- Don't aggregate exposure by vendor zone codes (`'10YFR-RTE------C'`,
  `'NBP'`, `'TTF-Henry'`) — always resolve via the shared dim
  (`main.iso.bidding_zone`, `main.iso.gas_hub`) so cross-desk reports
  are joinable. See `GeoZoneCatalog`.
- Don't collapse a two-leg spread into one row. Risk, P&L, and VaR
  all need the gross, not the net.
- Don't write trade ideas to a notebook output cell. The artifact
  is `analyst_<task>_positions_proposed`. Notebooks are exploration;
  proposals are queryable.
- Don't ship a signal without `rationale`. The trader rejecting a
  trade idea wants to know "what does the analyst think?" — a
  one-liner is the contract.
- Don't auto-execute. The analyst layer ends at `positions_proposed`;
  the OMS / trader integration is a separate concern with its own
  approvals.
- Don't mix commodities in one `analyst_<task>_*` table. Power, gas,
  and oil have different units, tenors, and risk shapes; keep them
  per-desk and let the dash layer union them.
- Don't reach for an ML model when a rule fits. Two-line rules ship
  in a day; a model needs an MLflow pipeline, drift detection,
  retraining, and ops on-call. Default to a rule, promote to a model
  when the rule's residuals look forecastable.
