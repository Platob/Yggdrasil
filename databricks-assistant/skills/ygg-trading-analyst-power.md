# Skill: power trading analyst — day-ahead, intraday, cross-zone spreads, spark/dark

## When to use

The user is doing analyst work on a **power** book and asks to
"forecast day-ahead prices", "build an intraday signal", "find a
cross-zone spread", "compute spark / dark / clean spreads", "size
the FR-DE position", "publish today's power morning pack", "stress
the book under a heatwave / cold snap", "aggregate exposure by
bidding zone", "compare residual load forecasts". The instrument
universe is **bidding zones (ENTSO-E EIC) and the contracts that
settle to them** — day-ahead hourly, intraday 15-minute / 30-minute
(depending on jurisdiction), balancing reserves, EPEX / Nord Pool /
N2EX auctions, futures on EEX / ICE Endex.

Routes off [`ygg-energy-trading-analyst`](ygg-energy-trading-analyst.md)
(shared analyst conventions — **read first**, especially the "don't
invent things" rule), uses [`ygg-curated-views`](ygg-curated-views.md) /
[`ygg-trading-commodity`](ygg-trading-commodity.md) (curated power
clearing, load, generation, cross-border flow), and
[`ygg-modelist`](ygg-modelist.md) (when the forecast needs a real
model — leave the training to the modelist).

## Don't invent tables — applies here too

Every example below uses placeholder names (`<curated_dayahead>`,
`<curated_load>`, `<weather_features>`, `<dash_zone_economics>`) —
**confirm the actual table names with the data engineer**, or ask
the user to share their catalog layout, before writing executable
SQL. The conceptual structure (residual load, cross-zone spread,
spark / dark formulas) is the part that travels; the table names
do not.

## Power-desk inputs (conceptual feature classes)

| Feature class | Refined notion | Refresh cadence |
| --- | --- | --- |
| Day-ahead clearing | `(eic_code, delivery_start_utc, price, currency_iso, unit='MWh')`. EU clears 12:42 CET D-1; GB clears 09:20 GMT D-1; Nord Pool clears 12:42 CET D-1. | Daily |
| Intraday clearing | Same shape + `auction_id` or "continuous" flag, 15-minute (continental) or 30-minute (GB) resolution. | Continuous |
| Load actual / forecast | `(eic_code, observation_utc, load_mw)`. Vendor-shipped TSO forecasts beat home-rolled. | Hourly |
| Generation actual / forecast | `(eic_code, observation_utc, psr_type, mw)` where `psr_type` is the ENTSO-E codelist (B16 solar, B19 wind onshore, B18 wind offshore, B14 nuclear, B04 fossil gas, …). | Hourly |
| Cross-border flow | `(in_eic, out_eic, observation_utc, flow_mw)`. | Hourly |
| Net transfer capacity (NTC) | `(in_eic, out_eic, delivery_start_utc, transfer_capacity_mw)`. Bounds the spread; without it a spread signal is half a signal. | Daily / per auction |
| Futures / forwards | `(contract_code, underlying_eic, delivery_period_iso, settle)` — EEX, ICE Endex, Nasdaq Commodities. | Daily settle |
| Weather features (per zone) | Temp pop-weighted, wind capacity-weighted at hub height (~100 m), solar GHI capacity-weighted. See the [weather modelist](ygg-modelist-weather.md). | Hourly |
| Carbon (EUA) | EU ETS settle, EUR/t. Required for clean spark / dark. | Daily settle |
| Fuel: gas (TTF / NBP) | Required for CCGT economics. See the gas desk for full coverage. | Daily settle + intraday |
| Fuel: coal (API2 / API4) | Required for hard-coal / lignite economics. | Daily |
| Events / outages | Unplanned outages, line trips, planned maintenance. | Event-driven |

`psr_type` follows the ENTSO-E codelist — keep the curated layer
using those codes; map to human labels at the dash / signal-rationale
layer.

## Residual load = the headline feature

Most power signals decompose into a view on **residual load** — the
load left after renewable generation is netted out:

```
residual_load_mw = load_forecast_mw - wind_forecast_mw - solar_forecast_mw
```

It captures the supply-stack position (low residual → cheap, high
residual → expensive marginal plant) more cleanly than load alone.
The data engineer should materialise it once per zone-hour so every
analyst / modelist task can reuse it; build the SQL against the
real curated load / generation tables once you've confirmed their
names.

Coverage rule: don't `coalesce(wind_mw, 0)` when wind data is
*missing* (vs *zero*). Flag the NULL; the row is unreliable and
shouldn't drive a signal.

## Day-ahead price forecast — feature set

Inputs that consistently predict EU day-ahead hourly clearing
prices (conceptual list — the modelist will materialise these into
the feature store; see [`ygg-modelist`](ygg-modelist.md)):

| Feature | Why it matters |
| --- | --- |
| `residual_load_mw` per zone | Strongest single signal. |
| Gas price (TTF or NBP) in EUR/MWh equivalent | Marginal-plant fuel cost (CCGT). |
| Coal price (API2 / API4) in EUR/MWh equivalent | Marginal lignite / hard-coal cost. |
| EUA price (EUR/t) | EU ETS carbon — adds to marginal cost. |
| Temperature pop-weighted (HDD / CDD) | Demand driver. |
| `hour_of_day`, `dow`, `is_holiday` | Strong seasonality. |
| NTC `import_capacity_mw` per border | Caps congestion spread. |
| `lag_24h_price`, `lag_168h_price` | Auto-regressive. |

The modelist owns the feature table, training, and the
`ml_<task>_*` artifacts. The analyst reads predictions and decides
how to trade them.

## Cross-zone spread signals — the bread-and-butter

A cross-zone spread is the price difference between two
*interconnected* bidding zones for the same delivery period:

```
spread_A_B(t) = dayahead_A(t) - dayahead_B(t)
```

The spread is bounded by the available **net transfer capacity
(NTC)** — when interconnectors are full, the spread can blow out;
when capacity is loose, it mean-reverts toward the marginal-fuel
delta between the two zones' supply stacks.

### Signal template — NTC-bounded mean reversion

Conceptual SQL — substitute your real curated table names; the
*pattern* is what matters:

```sql
-- placeholder table names
WITH spread AS (
  SELECT
      a.delivery_start_utc,
      a.price - b.price                          AS spread,
      ntc.transfer_capacity_mw                   AS ntc_mw,
      flow.flow_mw                               AS realised_flow_mw,
      flow.flow_mw / nullif(ntc.transfer_capacity_mw, 0) AS utilisation
  FROM <curated_dayahead> a
  JOIN <curated_dayahead> b
        ON b.delivery_start_utc = a.delivery_start_utc
       AND b.eic_code = '<zone_B_eic>'
  LEFT JOIN <curated_ntc> ntc
        ON ntc.in_eic = '<zone_B_eic>'
       AND ntc.out_eic = '<zone_A_eic>'
       AND ntc.delivery_start_utc = a.delivery_start_utc
  LEFT JOIN <curated_cross_border_flow> flow
        ON flow.in_eic = '<zone_B_eic>'
       AND flow.out_eic = '<zone_A_eic>'
       AND flow.observation_utc = a.delivery_start_utc
  WHERE a.eic_code = '<zone_A_eic>'
)
SELECT
    delivery_start_utc,
    spread,
    avg(spread) OVER w_30d            AS spread_mean_30d,
    stddev(spread) OVER w_30d         AS spread_std_30d,
    (spread - avg(spread) OVER w_30d)
        / nullif(stddev(spread) OVER w_30d, 0) AS spread_zscore,
    utilisation,
    CASE
      WHEN utilisation > 0.95 AND
           (spread - avg(spread) OVER w_30d)
             / nullif(stddev(spread) OVER w_30d, 0) > 2.0
      THEN 'mean_reversion_short_spread'
      WHEN utilisation < 0.30 AND
           (spread - avg(spread) OVER w_30d)
             / nullif(stddev(spread) OVER w_30d, 0) < -2.0
      THEN 'mean_reversion_long_spread'
      ELSE NULL
    END AS rule_name
FROM spread
WINDOW w_30d AS (ORDER BY delivery_start_utc
                 ROWS BETWEEN 720 PRECEDING AND 1 PRECEDING)
```

`rule_name` falls into `analyst_dayahead_signals` directly. The
matching `analyst_dayahead_positions_proposed` row is the
**two-leg spread** convention from
[`ygg-energy-trading-analyst#cross-zone-spread-positions-are-two-leg-one-rationale`](ygg-energy-trading-analyst.md#cross-zone-spread-positions-are-two-leg-one-rationale).

### Cross-zone FX nuance

Most central-EU bidding zones quote EUR — no FX between FR / DE /
NL / BE. **Exceptions** that bite analysts:

| Zone | Currency | FX exposure when book is EUR |
| --- | --- | --- |
| GB (N2EX, EPEX UK) | GBP | Real — hedge with `EURGBP`. |
| Nordics (NO, SE, DK1/2 — Nord Pool) | EUR | None (the auction is EUR). |
| CH | CHF | Real. |
| PL | PLN | Real. |
| HU, CZ, RO | EUR (M7 day-ahead) but balancing in local ccy | Mixed — flag by `currency_iso` per row. |

The rule: read `currency_iso` from the curated row; if it doesn't
match `base_currency_iso`, call `FxRate.latest([(ccy, base_ccy)])`
at proposal time and store both `fx_rate_used` and `fx_pair_iso` on
the position. See
[`ygg-energy-trading-analyst#fx-risk`](ygg-energy-trading-analyst.md#fx-risk--always-route-through-yggdrasilfxrate).

## Spark / dark / clean spreads — the cross-commodity view

The most-watched fundamental signal on a power book: is the
marginal generating unit profitable at the current power price?

```
spark_spread_eur_mwh   = power_eur_mwh
                       - gas_eur_mwh / gas_to_power_efficiency

dark_spread_eur_mwh    = power_eur_mwh
                       - coal_eur_mwh / coal_to_power_efficiency

clean_spark_eur_mwh    = spark_spread_eur_mwh
                       - co2_eua_eur_t * tco2_per_mwh_ccgt

clean_dark_eur_mwh     = dark_spread_eur_mwh
                       - co2_eua_eur_t * tco2_per_mwh_coal
```

Reference values you can quote with confidence (industry standards;
double-check for your specific plant fleet):

- `gas_to_power_efficiency`: CCGT ≈ 0.50 (50 % HHV)
- `coal_to_power_efficiency`: hard-coal ≈ 0.38–0.45; lignite ≈ 0.36
- `tco2_per_mwh_ccgt` ≈ 0.36
- `tco2_per_mwh_coal` (hard-coal) ≈ 0.90; lignite ≈ 1.10

Persist these as desk-config rather than hardcoded literals in
the SQL; the fleet mix evolves.

Signal patterns:

- **Negative clean spark** with high wind forecast → short the zone
  (gas plant uneconomic, renewables crowd out demand).
- **Positive clean spark** in summer heat + low wind → long the
  zone (gas dispatching at high cost).
- **Clean dark > clean spark by EUR 5+** in coal-favouring
  jurisdictions → coal beats gas on the stack; signal long that
  zone vs short a gas-heavy neighbour.

## Intraday & balancing — separate signal task

Intraday is a different game — 15-minute (continental) or 30-minute
(GB) resolution, continuous auction, dominated by renewable forecast
revisions. Don't reuse the day-ahead model.

Feature classes the modelist will want for intraday:

| Feature | Why |
| --- | --- |
| Wind / solar forecast revision (delta from prior cycle) | Strongest IDA signal. |
| Imbalance price (T-1, T-2) | Carries TSO scarcity. |
| Last clearing within-day | Auto-regressive. |
| Gas intraday price | Within-day fuel cost. |

Lands in `analyst_intraday_signals` (separate task from day-ahead).
Horizon `PT15M` to `PT4H` only — anything longer belongs on the
day-ahead desk.

## Geographic risk patterns specific to power

- **Sole-zone concentration risk.** If > 30% of the book's
  `gross_quantity` sits in one EIC, the `analyst_dayahead_risk`
  snapshot flags it. EIC → `country_iso` → continent rollup via
  `GeoZoneCatalog` (see
  [`ygg-energy-trading-analyst#aggregate-by-zone-country-continent`](ygg-energy-trading-analyst.md#aggregate-by-zone-country-continent)).
- **Interconnector dependence.** Spread trades are bounded by NTC;
  the risk snapshot carries `ntc_utilisation_pct` per zone-pair so
  the analyst sees when the spread can't widen further.
- **Weather correlation.** A long-FR / short-DE spread is *also*
  long northwest-Europe weather (cold continental anticyclone pumps
  both prices). Quantify by computing the spread's beta to a
  pop-weighted EU temperature anomaly; surface it on the risk
  dashboard.
- **DST gaps.** Spring-forward = 23 hours in the day, fall-back = 25.
  Curated layer already lands the rows correctly (see
  [`ygg-trading-commodity#time-series-hygiene`](ygg-trading-commodity.md));
  the analyst's `GROUP BY delivery_start_utc` handles both naturally
  — don't hand-roll a `for hour in range(24)` loop.

## Don'ts (power-specific — also see the shared skill)

- Don't model day-ahead and intraday in one task.
- Don't treat Nordic and continental zones identically. Nord Pool
  is EUR but bidding-zone topology + reservoir constraints differ —
  train per zone cluster.
- Don't ignore congestion. A spread signal without an NTC check is
  half a signal.
- Don't aggregate exposure by vendor-specific zone codes. Resolve
  through `GeoZoneCatalog` and shared ISO dims.
- Don't store GB-zone clearing in EUR-denominated fields. GB zone
  = GBP/MWh; persist that and FX-convert on the proposal row.
- Don't backfill `residual_load_mw` with `coalesce(wind_mw, 0)`
  when wind data is *missing* — flag the NULL.
- Don't propose a spread without naming both legs in
  `analyst_dayahead_positions_proposed`.
