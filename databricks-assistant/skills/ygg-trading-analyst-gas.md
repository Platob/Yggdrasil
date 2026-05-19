# Skill: gas trading analyst — TTF, NBP, Henry Hub, JKM, LNG arb, storage, weather

## When to use

The user is doing analyst work on a **natural gas / LNG** book and
asks to "forecast TTF / NBP / Henry Hub prices", "find an LNG
arbitrage", "compute the inter-hub spread", "stress storage draws",
"build a heating-degree-day signal", "size a TTF–JKM trade",
"publish the gas morning pack", "aggregate by country / hub". The
instrument universe is the **major hubs (TTF, NBP, Henry Hub, JKM,
AECO, PEG, THE, ZTP, PSV)** and the contracts that settle to them
— day-ahead, within-day, prompt month, balance-of-month, seasonal
strips.

Routes off [`ygg-energy-trading-analyst`](ygg-energy-trading-analyst.md)
(shared analyst conventions — **read first**, especially the "don't
invent things" rule), uses [`ygg-curated-views`](ygg-curated-views.md) /
[`ygg-trading-commodity`](ygg-trading-commodity.md) (curated hub
settles, freight, storage), and [`ygg-modelist`](ygg-modelist.md)
for the forecast model.

## Don't invent tables — applies here too

Every example below uses placeholder names — confirm the actual
table names with the data engineer before writing executable SQL.
The conceptual structure (HDD/CDD, storage z-score, LNG arb,
inter-hub spreads, fuel switching) is the part that travels.

## Hub universe — refined notions you can quote

These are real markets with stable conventions; persist as
`hub_name` lookup, *don't* invent table names that sit behind them:

| Hub | Currency | Native unit | Convert to USD/MMBtu |
| --- | --- | --- | --- |
| TTF (NL) | EUR | EUR/MWh | `price * USDEUR / 3.412`  (1 MWh = 3.412 MMBtu) |
| NBP (UK) | GBp | GBp/therm | `price / 100 * GBPUSD * 10` (1 therm = 0.1 MMBtu) |
| Henry Hub (US) | USD | USD/MMBtu | identity |
| JKM (Asia LNG) | USD | USD/MMBtu | identity |
| PEG (FR) | EUR | EUR/MWh | same shape as TTF |
| THE (DE — NCG + GASPOOL merged 2021) | EUR | EUR/MWh | same shape as TTF |
| ZTP (BE) | EUR | EUR/MWh | same shape as TTF |
| PSV (IT) | EUR | EUR/MWh | same shape as TTF |
| AECO (CA) | CAD | CAD/GJ | CAD→USD, then `1 GJ ≈ 0.948 MMBtu` |

**Don't hand-roll unit math in signal SQL.** Build the conversion
into a normalised hub-price view once. Same rule for FX — route
through [`yggdrasil.fxrate.FxRate`](ygg-energy-trading-analyst.md#fx-risk--always-route-through-yggdrasilfxrate).

## The dominant gas features

### Heating-degree-days (HDD) and cooling-degree-days (CDD)

The strongest single feature for prompt gas. EU is heating-dominated
in winter, CDD-driven (via power demand) in summer. US has both.

```
HDD_t  = max(0, 65 °F - daily_avg_temp_f)   -- 18.3 °C basis
CDD_t  = max(0, daily_avg_temp_f - 65 °F)
```

Built per gas-demand region (NWE, S_EU, US_Central, US_East, …) —
population- *and* demand-weighted; cold-snap weighting needs the
urban centres counted more than the periphery. The
[weather modelist](ygg-modelist-weather.md) owns the
demand-region-weighted HDD/CDD table.

Signal template — cold-snap front-month rally (placeholder
table names):

```sql
WITH hdd_anom AS (
  SELECT
      observation_utc,
      gas_region,
      hdd,
      hdd - avg(hdd) OVER w_365d AS hdd_anomaly,
      stddev(hdd) OVER w_365d    AS hdd_std
  FROM <weather_hdd_cdd_table>
  WHERE observation_utc >= current_date() - INTERVAL 60 DAYS
  WINDOW w_365d AS (PARTITION BY gas_region
                    ORDER BY observation_utc
                    ROWS BETWEEN 365 PRECEDING AND 1 PRECEDING)
)
SELECT
    observation_utc,
    gas_region,
    hdd_anomaly,
    CASE
      WHEN hdd_anomaly > 2.0 * hdd_std THEN 'long_prompt_cold_snap'
      WHEN hdd_anomaly < -2.0 * hdd_std AND gas_region = 'NWE'
        THEN 'short_prompt_warm_anomaly'
      ELSE NULL
    END AS rule_name
FROM hdd_anom
```

### Storage z-score — slow, but it's *the* fundamentals signal

Two transforms the gas desk reuses across every storage report:

- `z_score_vs_5y` — same calendar day across 5 years, not last-5
  points.
- `days_of_cover = inventory / recent_avg_demand`.

The window choice matters: partition on `MM-dd` so you're comparing
"this March 12" to "previous five March 12ths", not "this week" to
"last week".

Signal: `eu_fill_pct < eu_fill_pct_p10` between Oct–Mar → long
winter TTF. The actual EU storage feed (typically AGSI / GIE for
EU, EIA Weekly Natural Gas Storage Report for US) lands in a
curated table — **ask the data engineer for the name and the
working-volume / max-working-volume columns** before writing the
SQL above.

### LNG arbitrage — the cross-hub freight-gated trade

The most-watched cross-commodity gas signal. Three legs and
freight matters more than people think:

```
arb_origin_to_dest_usd_mmbtu
  = dest_price_usd_mmbtu
  - (origin_price_usd_mmbtu * (1 + liquefaction_fee_pct)
     + lng_freight_origin_dest_usd_mmbtu
     + boiloff_pct * origin_price_usd_mmbtu
     + regas_fee_usd_mmbtu)
```

Build the delivered-cost calculation in a dash view once
(`<dash_lng_delivered_cost>`); the signal task reads it.

Signal: when `arb > 1.5 USD/MMBtu` **and** the destination's
clearing hasn't already priced it in
(`dest_price - dest_price_lag_5d > arb`), long the destination,
short the origin. Two-leg in `analyst_lng_arb_positions_proposed`
sharing a `signal_id`.

Actionability filters (the signal is only tradable when all hold):

- **Vessels available.** Vessel utilisation `< 90 %` for the
  relevant class (LNGC TFDE / MEGI for Atlantic / Pacific).
- **Re-gas slot available.** Destination terminal not oversubscribed
  for the delivery window.
- **Sanctions / origin restrictions clear.** Cross-reference the
  events feed — a fresh sanction on origin re-prices in days.

The data the analyst needs the data engineer to land per route:
`(origin_port, dest_port, observation_utc, freight_usd_mmbtu,
vessel_utilisation_pct, regas_slot_available_window, route_geojson,
route_choke_points)`. Confirm the table name before SQL.

### Inter-hub spreads

```
TTF_NBP_spread_eur_mwh = TTF_eur_mwh - NBP_eur_mwh
```

Unit / FX conversion for NBP (`GBp/therm → EUR/MWh`):
`(price_GBp / 100) * GBPEUR / 0.029307` (1 therm = 0.029307 MWh).
Persist the converted column in a normalised hub-price view —
one row per `(hub_name, observation_utc)`, all-USD and all-EUR
columns side-by-side, `fx_rate_used` persisted per row.

Spread signals are the same z-score mean-reversion shape as power
cross-zone spreads — with **one critical difference**: gas hubs
are connected by **pipeline capacity** (NBP↔TTF via the
Interconnector, etc.) or **LNG freight**, not by power
interconnectors. The "gating capacity" column on the normalised
view is `pipeline_capacity_gwh_per_day` for pipe-connected pairs
and `lng_freight_utilisation_pct` for LNG-connected ones.

## Cross-geography risk on a gas book

Aggregations follow the shared rule (see
[`ygg-energy-trading-analyst#cross-geo-zone-risk`](ygg-energy-trading-analyst.md#cross-geo-zone-risk));
gas-specific dimensions worth bucketing separately:

- **`origin_country_iso`** — concentration to one source country
  (post-2022, US dominance is real).
- **`route_choke_point`** — Suez, Panama, Bab-el-Mandeb, Hormuz.
  A trade long delivered NWE via Suez and a trade long delivered
  JKM via Suez are correlated through a single chokepoint — the
  risk snapshot should expose it.
- **`destination_region`** — NWE / Iberia / Med / NE Asia / SE Asia.

## Forecast model — typical feature set (prompt TTF)

Conceptual features the modelist will materialise; see
[`ygg-modelist`](ygg-modelist.md) for the training contract:

| Feature | Shape |
| --- | --- |
| `lag_1d`, `lag_5d`, `lag_30d` price | curated settle |
| `realised_vol_5d`, `realised_vol_30d` | derived |
| `hdd_anomaly_nwe`, `cdd_anomaly_nwe` | weather features |
| `eu_fill_pct_anomaly` | storage z-score |
| `lng_inflow_5d_avg_gwh` | LNG flow tracking |
| `arb_USGC_NWE_usd_mmbtu` (lag 1) | LNG arb dash |
| `co2_eua_eur_t` | EUA settle (gas-vs-coal switching) |
| Cross-desk: DE power day-ahead (lag 1) | fuel-switching feedback |
| `is_winter`, `dow`, `holiday_eu` | calendar |
| `usd_eur_fx`, `dxy` | `FxRate.fetch` + macro |
| `event_severity_max_5d` | events feed, scored 0..5 |

Default modelling pattern: LightGBM with quantile regression for
prediction intervals; statsforecast `AutoARIMAX` for back-of-curve
seasons. Both via the modelist's multi-candidate loop.

## Fuel-switching — the cross-commodity feedback loop

EU power's CCGT plants set the gas demand floor. When **clean spark
> clean dark**, CCGTs win the dispatch and lift gas demand; when
**clean dark > clean spark**, coal wins and gas demand collapses.

```
switch_signal = sign(clean_spark_eur_mwh - clean_dark_eur_mwh)
               -- +1 = gas wins, -1 = coal wins
```

Read from the power desk's clean-spread output (see
[`ygg-trading-analyst-power#spark--dark--clean-spreads`](ygg-trading-analyst-power.md#spark--dark--clean-spreads--the-cross-commodity-view))
— this is *intentional* cross-desk dependency.

Signal: persistent `switch_signal = +1` for ≥ 5 days → gas-demand-up
signal, long prompt TTF. Persistent `-1` → short prompt. The
feedback works on weekly cadence; don't intra-day it.

## Don'ts (gas-specific — also see the shared skill)

- Don't compute LNG arb without freight. The 0.5–1.0 USD/MMBtu
  freight component is what kills three out of four "arb open"
  alerts on a tight-vessel day.
- Don't mix `GBp/therm` and `EUR/MWh` in the same column without an
  explicit conversion view. NBP signals will silently look 100× off.
- Don't forget vessel availability + regas slot. The arb is
  hypothetical until you can book a ship and a re-gas window.
- Don't drive a winter signal off "absolute fill %". Compare to
  same-calendar-day 5y norms.
- Don't treat AECO / CEGT / NCG separately when they've merged.
  THE is now the German hub (NCG + GASPOOL collapsed in 2021).
- Don't aggregate exposure by hub name string alone. Resolve to
  `country_iso` and `route_choke_point`.
- Don't ignore severity on the events feed. A pipeline outage at
  severity 4 will reprice the front month before your
  storage-z-score has fresh data.
- Don't drive a JKM forecast off ARIMA alone. JKM is shock-driven
  (cargo diversions, weather, sanctions) — pure auto-regressive
  models under-predict tail moves.
