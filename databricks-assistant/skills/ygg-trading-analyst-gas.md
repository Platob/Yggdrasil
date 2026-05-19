# Skill: gas trading analyst — TTF, NBP, Henry Hub, JKM, LNG arb, storage, weather

## When to use

The user is doing analyst work on a **natural gas / LNG** book and
asks to "forecast TTF / NBP / Henry Hub prices", "find an
LNG arbitrage", "compute the inter-hub spread", "stress storage
draws", "build a heating-degree-day signal", "size a TTF–JKM trade",
"publish the gas morning pack", "aggregate by country / hub". The
instrument universe is **hubs (TTF, NBP, Henry Hub, JKM, AECO, PEG,
THE, ZTP, PSV)** and the contracts that settle to them — day-ahead,
within-day, prompt month, balance-of-month, seasonal strips.

Routes off [`ygg-energy-trading-analyst`](ygg-energy-trading-analyst.md)
(shared analyst conventions — read first), uses
[`ygg-curated-views`](ygg-curated-views.md) /
[`ygg-trading-commodity`](ygg-trading-commodity.md) (curated
`<source>.spot`, `<source>.futures`, `<source>.lng_freight`,
`<source>.storage_fill`), and [`ygg-mlops`](ygg-mlops.md) for the
forecast model.

## Gas-desk inputs (curated tables you read)

| Curated table | Carries | Refresh |
| --- | --- | --- |
| `main.ice.ttf_settle` | TTF prompt-month / seasons, EUR/MWh | Daily settle 19:30 CET |
| `main.ice.nbp_settle` | NBP prompt + seasons, GBp/therm | Daily settle, ICE LSE |
| `main.nymex.hh_settle` | Henry Hub, USD/MMBtu | Daily NYMEX settle |
| `main.platts.jkm` | JKM Asia LNG spot, USD/MMBtu | Daily Platts assessment |
| `main.<source>.spot` | Day-ahead clearing, hub-local ccy/unit | Daily, per hub |
| `main.agsi.gas_storage` | EU hub fill %, working volume, withdrawal/injection | Daily, T+1 |
| `main.eia.weekly_storage` | US Lower-48 storage by region, bcf | Weekly Thursday 10:30 ET |
| `main.spark.lng_freight` | Per-route LNG freight, USD/MMBtu, vessel type | Daily |
| `main.kpler.lng_flows` | Cargo-tracked LNG flows, origin/destination, ETA | Continuous |
| `main.weather.<provider>.hdd_cdd` | HDD/CDD per gas-demand region | Hourly |
| `main.enex.events` | Pipeline disruptions, sanctions, refinery outages | Event-driven |

Currency / unit conventions per hub:

| Hub | Currency | Unit | Conversion to USD/MMBtu |
| --- | --- | --- | --- |
| TTF | EUR | MWh | `price * 1.0` then EUR→USD; `1 MWh = 3.412 MMBtu` so multiply by 0.293 |
| NBP | GBp | therm | `price / 100` to GBP, then GBP→USD; `1 therm = 0.1 MMBtu` so multiply by 10 |
| Henry Hub | USD | MMBtu | identity |
| JKM | USD | MMBtu | identity |
| PEG (FR) | EUR | MWh | same as TTF |
| THE (DE) | EUR | MWh | same as TTF |
| AECO | CAD | GJ | CAD→USD; `1 GJ ≈ 0.948 MMBtu` |

**Don't hand-roll unit math in signal SQL.** Build the conversion
into a `main.iso.gas_hub_unit_convert(hub_name, target_unit)` view
once. Same rule for FX — go through
[`yggdrasil.fxrate.FxRate`](ygg-energy-trading-analyst.md#fx-risk--always-route-through-yggdrasilfxrate).

## The dominant gas features

### Heating-degree-days (HDD) and cooling-degree-days (CDD)

The strongest single feature for prompt gas. EU is heating-dominated
in winter, CDD-driven (via power demand) in summer. US has both.

```
HDD_t  = max(0, 65 °F - daily_avg_temp_f)   -- 18.3 °C basis
CDD_t  = max(0, daily_avg_temp_f - 65 °F)
```

Build per gas-demand region (NWE = `BE+NL+DE_NW+FR_N`, S_EU,
US_Central, US_East, …) as `main.weather.dash_gas_hdd_cdd` with one
row per `(gas_region, observation_utc)`. Population- *and*
demand-weight the grid points; cold-snap weighting on Belgium needs
to count Antwerp / Brussels more than the Ardennes.

Signal template — cold-snap front-month rally:

```sql
WITH hdd_anom AS (
  SELECT
      observation_utc,
      gas_region,
      hdd,
      hdd - avg(hdd) OVER (PARTITION BY gas_region
                           ORDER BY observation_utc
                           ROWS BETWEEN 365 PRECEDING AND 1 PRECEDING) AS hdd_anomaly,
      stddev(hdd) OVER (PARTITION BY gas_region
                        ORDER BY observation_utc
                        ROWS BETWEEN 365 PRECEDING AND 1 PRECEDING) AS hdd_std
  FROM main.weather.dash_gas_hdd_cdd
  WHERE observation_utc >= current_date() - INTERVAL 60 DAYS
)
SELECT
    observation_utc,
    gas_region,
    hdd_anomaly,
    CASE
      WHEN hdd_anomaly > 2.0 * hdd_std THEN 'long_prompt_cold_snap'
      WHEN hdd_anomaly < -2.0 * hdd_std AND gas_region = 'NWE' THEN 'short_prompt_warm_anomaly'
      ELSE NULL
    END AS rule_name
FROM hdd_anom
```

### Storage z-score — slow, but it's *the* fundamentals signal

```sql
-- TTF storage stress signal — fills <5y P10 → long winter
WITH eu_storage AS (
  SELECT
      observation_utc,
      sum(working_volume_gwh) AS eu_working_volume_gwh,
      sum(working_volume_gwh) /
        sum(max_working_volume_gwh) * 100 AS eu_fill_pct
  FROM main.agsi.gas_storage
  WHERE country_iso IN ('BE','NL','DE','FR','AT','IT')
  GROUP BY observation_utc
)
SELECT
    observation_utc,
    eu_fill_pct,
    avg(eu_fill_pct) OVER same_day_5y AS eu_fill_pct_5y_avg,
    eu_fill_pct - avg(eu_fill_pct) OVER same_day_5y AS eu_fill_pct_anomaly,
    percentile_approx(eu_fill_pct, 0.10) OVER same_day_5y AS eu_fill_pct_p10
FROM eu_storage
WINDOW same_day_5y AS (
  PARTITION BY date_format(observation_utc, 'MM-dd')
  ORDER BY observation_utc
  ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
)
```

Signal: `eu_fill_pct < eu_fill_pct_p10` between Oct–Mar → long
winter TTF. Note the partition on `MM-dd` — same calendar day
across years, not last-5-points.

### LNG arbitrage — the cross-hub freight-gated trade

The most-watched cross-commodity gas signal: deliver US gas to
Europe or to Asia at a profit. Three legs and freight matters more
than people think.

```
arb_USGC_to_NWE_usd_mmbtu = TTF_usd_mmbtu
                          - (HH_usd_mmbtu * (1 + liquefaction_fee_pct)
                             + lng_freight_USGC_NWE_usd_mmbtu
                             + boiloff_loss_pct * HH_usd_mmbtu
                             + regas_fee_usd_mmbtu)

arb_USGC_to_JKM_usd_mmbtu = JKM_usd_mmbtu - (… same shape via Panama or Cape route)
```

Build `main.gas.dash_lng_arb` with one row per
`(origin_hub, destination_hub, observation_utc)` — see
[`ygg-energy-trading-analyst#freight--shipping`](ygg-energy-trading-analyst.md#freight--shipping--the-cross-hub-arbitrage-gate)
for the SQL pattern. Add `route_geojson` so the morning dash can
render the route on a map.

Signal: when `arb > 1.5 USD/MMBtu` and *the destination's
clearing market hasn't already priced it in* (`destination_price -
destination_price_lag_5d > arb`), long the destination, short the
origin. Both legs are two rows in
`analyst_lng_arb_positions_proposed` sharing a `signal_id`.

The signal is only actionable when:

- **Vessels available.** `main.spark.lng_freight` includes vessel
  utilisation; tight freight (utilisation > 90%) means the arb is
  trapped — quote it but don't size it.
- **Re-gas slot available.** `main.kpler.lng_regas_capacity` —
  destination terminals oversubscribed for the window kills the
  trade. Filter by `regas_slot_available_window`.
- **Sanctions / origin restrictions clear.** Cross-reference
  `main.enex.events` — a fresh sanction on the origin country
  re-prices in days.

### Inter-hub spreads

```
TTF_NBP_spread_eur_mwh = TTF_eur_mwh - NBP_eur_mwh
```

Mind the unit / FX conversion: convert NBP `GBp/therm → EUR/MWh`
via `(price_GBp / 100) * GBP_EUR * (1 / 0.029307)` (1 therm
= 0.029307 MWh). Persist the converted column in
`main.gas.dash_hub_prices_normalised` — one row per
`(hub_name, observation_utc)`, all-USD and all-EUR columns
side-by-side, FX rate `fx_rate_used` persisted per row. See
[`ygg-energy-trading-analyst#fx-risk`](ygg-energy-trading-analyst.md#fx-risk--always-route-through-yggdrasilfxrate).

Spread signals are the same z-score mean-reversion shape as power
cross-zone spreads (see
[`ygg-trading-analyst-power`](ygg-trading-analyst-power.md#cross-zone-spread-signals--the-bread-and-butter))
— with **one critical difference**: gas hubs are connected by
**pipeline capacity** (NBP↔TTF via the Interconnector, etc.) or
**LNG freight**, not by power interconnectors. The "gating capacity"
column on `dash_hub_prices_normalised` is
`pipeline_capacity_gwh_per_day` for pipe-connected pairs and
`lng_freight_utilisation` for LNG-connected ones.

## Cross-geography risk on a gas book

`GeoZoneCatalog` for gas runs through `main.iso.gas_hub` (hub →
country) and `main.iso.gas_demand_region`. Aggregations:

```sql
INSERT OVERWRITE main.gas.analyst_gas_risk
SELECT
    current_timestamp()           AS snapshot_utc,
    'country_iso'                 AS dimension,
    h.country_iso                 AS bucket,
    sum(p.quantity * sign_dir(p.direction))                AS net_quantity_mmbtu,
    sum(p.notional_base * sign_dir(p.direction))           AS delta_base_ccy,
    sum(abs(p.notional_base)) /
        nullif(sum(abs(p.notional_base)) OVER (), 0)       AS concentration_pct
FROM main.gas.analyst_lng_arb_positions_proposed p
LEFT JOIN main.iso.gas_hub h ON h.hub_name = p.eic_code
GROUP BY h.country_iso
```

LNG-specific geo dimensions worth bucketing separately:

- **`origin_country_iso`** — concentration to one source country
  (post-2022, US dominance is real; pre-2022, Qatar / Russia were).
- **`route_choke_point`** — Suez / Panama / Bab-el-Mandeb. A trade
  long delivered NWE via Suez and a trade long delivered JKM via
  Suez are correlated through a single chokepoint, not through
  origin / destination — the risk snapshot should expose it.
- **`destination_region`** — NWE / Iberia / Med / NE Asia / SE Asia.

The choke-point dimension comes from the `route_geojson` polyline
joined to `main.iso.shipping_lane.choke_points` (`array<string>`
of chokepoint codes). Geo-concentration limits then apply per
choke-point.

## Forecast model — typical feature set

For a TTF prompt-month price forecast (horizon = D+1 to D+5):

| Feature | Source |
| --- | --- |
| `lag_1d`, `lag_5d`, `lag_30d` price | curated |
| `realised_vol_5d`, `realised_vol_30d` | dash_vol |
| `hdd_anomaly_nwe`, `cdd_anomaly_nwe` | dash_gas_hdd_cdd |
| `eu_fill_pct_anomaly` | storage |
| `lng_inflow_5d_avg_gwh` | dash_lng_flows |
| `arb_USGC_NWE_usd_mmbtu` (lag 1) | dash_lng_arb |
| `pipeline_flow_NWE_5d_avg` | pipeline ops |
| `co2_eua_eur_t` | EUA settle (gas-vs-coal switching) |
| `power_dayahead_de` (lag 1) | power desk's curated DE clearing — fuel-switching feedback |
| `is_winter`, `dow`, `holiday_eu` | calendar |
| `usd_eur_fx`, `dxy` | FxRate fetch + dash_macro |
| `event_severity_max_5d` | enex.events scored 0..5 |

Standard MLflow pipeline via [`ygg-mlops`](ygg-mlops.md). LightGBM
with quantile regression for prediction intervals
(`prediction_lower` / `prediction_upper` on the signal row) is the
production default for prompt; statsforecast `AutoARIMAX` works for
back-of-curve seasons.

## Fuel-switching — the cross-commodity feedback loop

EU power's CCGT plants set the gas demand floor. When **clean spark
> clean dark**, CCGTs win the dispatch and lift gas demand; when
**clean dark > clean spark**, coal wins and gas demand collapses.

Persist the switch indicator on `main.gas.dash_fuel_switching`:

```
switch_signal = sign(clean_spark_eur_mwh - clean_dark_eur_mwh)
               -- +1 = gas wins, -1 = coal wins
```

Read from the power desk's `main.entsoe.dash_zone_economics` (see
[`ygg-trading-analyst-power`](ygg-trading-analyst-power.md#spark--dark--clean-spreads--the-cross-commodity-view))
— this is *intentional* cross-desk dependency, both desks need the
view.

Signal: persistent `switch_signal = +1` for ≥ 5 days → 
gas-demand-up signal, long prompt TTF. Persistent `-1` → short 
prompt. The feedback works on weekly cadence; don't intra-day it.

## Morning gas pack — the dash table

`main.gas.dash_morning_gas` rolls everything into a one-row-per-day
analyst view:

| Column | Source |
| --- | --- |
| `price_ttf_eur_mwh`, `price_nbp_eur_mwh`, `price_hh_usd_mmbtu`, `price_jkm_usd_mmbtu` | curated settle |
| `eu_fill_pct`, `eu_fill_pct_anomaly` | storage |
| `arb_USGC_NWE_usd_mmbtu`, `arb_USGC_JKM_usd_mmbtu` | dash_lng_arb |
| `hdd_anomaly_nwe`, `cdd_anomaly_us` | dash_gas_hdd_cdd |
| `switch_signal_de`, `switch_signal_uk` | dash_fuel_switching |
| `flagged_signals` | array<string> of active signal_ids |
| `top_rationale` | denormalised one-liner |
| `route_choke_points_at_risk` | array<string> from enex.events |

Refreshed by the analyst job after the morning settle + AGSI / EIA
storage windows.

## Don'ts (gas-specific — also see the shared skill)

- Don't compute LNG arb without freight. The 0.5–1.0 USD/MMBtu
  freight component is what kills three out of four "arb open"
  alerts on a tight-vessel day.
- Don't mix `GBp/therm` and `EUR/MWh` in the same column without an
  explicit conversion view. NBP signals will silently look 100×
  bigger than reality.
- Don't forget vessel availability + regas slot. The arb is
  hypothetical until you can book a ship and a re-gas window.
- Don't drive a winter signal off "absolute fill %". Compare to
  same-calendar-day 5y norms — 70% fill in October is bullish,
  70% in March is bearish.
- Don't treat AECO / CEGT / NCG separately when they've merged. THE
  is now the German hub (NCG + GASPOOL collapsed in 2021); curated
  layer should already map old codes through, but verify before
  back-testing pre-2022 data.
- Don't aggregate exposure by hub name string. Resolve to
  `country_iso` and `route_choke_point` via shared dims so cross-
  desk reports join.
- Don't ignore `event_severity` on the `enex.events` feed. A
  pipeline outage at Severity 4 will reprice the front month before
  your storage-z-score signal even has fresh data.
- Don't drive a JKM forecast off ARIMA alone. JKM is shock-driven
  (cargo diversions, weather, sanctions) — pure auto-regressive
  models under-predict tail moves; carry an event-feature in the
  model or quote prediction intervals widely.
