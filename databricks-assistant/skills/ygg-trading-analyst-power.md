# Skill: power trading analyst — day-ahead, intraday, cross-zone spreads, spark/dark

## When to use

The user is doing analyst work on a **power** book and asks to
"forecast day-ahead prices", "build an intraday signal", "find a
cross-zone spread", "compute spark / dark / clean spreads", "size
the FR-DE position", "publish today's power morning pack", "stress
the book under a heatwave / cold snap", "aggregate exposure by
bidding zone", "compare residual load forecasts". The instrument
universe is **bidding zones (EIC) and the contracts that settle to
them** — day-ahead hourly, intraday 15-minute, balancing reserves,
EPEX / Nord Pool / N2EX auctions, futures on EEX.

Routes off [`ygg-energy-trading-analyst`](ygg-energy-trading-analyst.md)
(shared analyst conventions — read first), uses
[`ygg-curated-views`](ygg-curated-views.md) /
[`ygg-trading-commodity`](ygg-trading-commodity.md) (curated
`<source>.dayahead`, `<source>.load`, `<source>.generation`,
`<source>.cross_border_flow`), and
[`ygg-mlops`](ygg-mlops.md) (statsforecast / NHITS templates for
hour-ahead and day-ahead forecasts).

## Power-desk inputs (curated tables you read)

| Curated table | Carries | Typical refresh |
| --- | --- | --- |
| `main.entsoe.dayahead` | `eic_code`, `delivery_start_utc`, `price`, `currency_iso`, `unit='MWh'` | D-1 16:00 CET (EU) |
| `main.entsoe.intraday` | same shape + `auction_id`, 15-minute resolution | Continuous |
| `main.entsoe.load_actual` / `load_forecast` | `eic_code`, `observation_utc`, `load_mw` | Hourly |
| `main.entsoe.generation_actual` | `eic_code`, `psr_type` (wind, solar, nuclear, gas, …), `mw` | Hourly |
| `main.entsoe.cross_border_flow` | `in_eic`, `out_eic`, `observation_utc`, `flow_mw` | Hourly |
| `main.eex.futures` | `contract_code`, `underlying_eic`, `delivery_period_iso`, `settle` | Daily settle |
| `main.weather.<provider>.forecast` | `lat`, `lon`, `observation_utc`, `temp_c`, `wind_ms`, `ghi_wm2` | Hourly |

`psr_type` follows the ENTSO-E codelist (`B16` solar, `B19` wind
onshore, `B18` wind offshore, `B14` nuclear, `B04` fossil gas, …).
Keep the curated layer using those codes; map to human labels at the
dash / signal-rationale layer.

## Residual load = the headline feature

Most power signals decompose into a view on **residual load** — the
load left after renewable generation is netted out:

```
residual_load_mw = load_forecast_mw - wind_forecast_mw - solar_forecast_mw
```

Build it once in `main.entsoe.dash_residual_load` (UTC time-axis,
one column per EIC) and reuse across every forecast / signal task.
It captures the supply-stack position (low residual → cheap, high
residual → expensive marginal plant) more cleanly than load alone.

```sql
CREATE OR REPLACE TABLE main.entsoe.dash_residual_load AS
SELECT
    l.eic_code,
    l.observation_utc,
    l.load_mw,
    coalesce(w_on.mw, 0) + coalesce(w_off.mw, 0)         AS wind_mw,
    coalesce(s.mw,    0)                                  AS solar_mw,
    l.load_mw - coalesce(w_on.mw,0) - coalesce(w_off.mw,0)
              - coalesce(s.mw, 0)                         AS residual_load_mw
FROM main.entsoe.load_forecast l
LEFT JOIN main.entsoe.generation_forecast w_on
       ON  w_on.eic_code = l.eic_code
      AND  w_on.observation_utc = l.observation_utc
      AND  w_on.psr_type = 'B19'
LEFT JOIN main.entsoe.generation_forecast w_off
       ON  w_off.eic_code = l.eic_code
      AND  w_off.observation_utc = l.observation_utc
      AND  w_off.psr_type = 'B18'
LEFT JOIN main.entsoe.generation_forecast s
       ON  s.eic_code = l.eic_code
      AND  s.observation_utc = l.observation_utc
      AND  s.psr_type = 'B16'
WHERE l.observation_utc >= current_date() - INTERVAL 30 DAYS
```

## Day-ahead price forecast — feature set

Inputs that consistently predict EU day-ahead hourly clearing prices:

| Feature | Source | Notes |
| --- | --- | --- |
| `residual_load_mw` per zone | `dash_residual_load` | Strongest single signal. |
| `gas_ttf_eur_mwh` | `main.ice.dash_ttf_settle` (gas desk) | Marginal-plant fuel cost (CCGT). |
| `coal_api2_eur_t` | `main.ice.dash_coal_api2_settle` | Marginal lignite / hard-coal cost. |
| `co2_eua_eur_t` | `main.eex.dash_eua_settle` | EU ETS carbon — adds to marginal cost. |
| `temp_c_pop_weighted` | weather × population polygon | Demand driver. |
| `hour_of_day`, `dow`, `is_holiday` | calendar | Strong seasonality. |
| `import_capacity_mw_<neighbour>` per border | NTC publications | Caps congestion spread. |
| `lag_24h_price`, `lag_168h_price` | curated dayahead | Auto-regressive. |

Build the feature snapshot in `main.entsoe.ml_dayahead_features`
(see [`ygg-mlops`](ygg-mlops.md) for the standard shape), train per
EIC with `AutoARIMA` + `lightgbm` ensemble. Predictions land in
`main.entsoe.ml_dayahead_predictions`; the analyst signal layer
reads predictions, applies the desk's conviction calibration, and
writes to `analyst_dayahead_signals`.

## Cross-zone spread signals — the bread-and-butter

A cross-zone spread is the price difference between two
*interconnected* bidding zones for the same delivery period:

```
spread_FR_DE(t) = dayahead_FR(t) - dayahead_DE(t)
```

The spread is bounded by the available **net transfer capacity
(NTC)** — when interconnectors are full, the spread can blow out;
when capacity is loose, it mean-reverts toward the marginal-fuel
delta between the two zones' supply stacks.

### Signal template — NTC-bounded mean reversion

```sql
WITH spread AS (
  SELECT
      fr.delivery_start_utc,
      fr.price - de.price                       AS spread_eur_mwh,
      ntc.transfer_capacity_mw                  AS ntc_mw,
      flow.flow_mw                              AS realised_flow_mw,
      flow.flow_mw / nullif(ntc.transfer_capacity_mw, 0) AS utilisation
  FROM main.entsoe.dayahead fr
  JOIN main.entsoe.dayahead de
        ON de.delivery_start_utc = fr.delivery_start_utc
       AND de.eic_code = '10YDE-VE-------2'
  LEFT JOIN main.entsoe.ntc_published ntc
        ON ntc.in_eic = '10YDE-VE-------2'
       AND ntc.out_eic = '10YFR-RTE------C'
       AND ntc.delivery_start_utc = fr.delivery_start_utc
  LEFT JOIN main.entsoe.cross_border_flow flow
        ON flow.in_eic = '10YDE-VE-------2'
       AND flow.out_eic = '10YFR-RTE------C'
       AND flow.observation_utc = fr.delivery_start_utc
  WHERE fr.eic_code = '10YFR-RTE------C'
    AND fr.delivery_start_utc >= current_date() - INTERVAL 30 DAYS
)
SELECT
    delivery_start_utc,
    spread_eur_mwh,
    avg(spread_eur_mwh) OVER w_30d AS spread_mean_30d,
    stddev(spread_eur_mwh) OVER w_30d AS spread_std_30d,
    (spread_eur_mwh - avg(spread_eur_mwh) OVER w_30d)
        / nullif(stddev(spread_eur_mwh) OVER w_30d, 0) AS spread_zscore,
    utilisation,
    CASE
      WHEN utilisation > 0.95 AND
           (spread_eur_mwh - avg(spread_eur_mwh) OVER w_30d)
             / nullif(stddev(spread_eur_mwh) OVER w_30d, 0) > 2.0
      THEN 'mean_reversion_short_spread'
      WHEN utilisation < 0.30 AND
           (spread_eur_mwh - avg(spread_eur_mwh) OVER w_30d)
             / nullif(stddev(spread_eur_mwh) OVER w_30d, 0) < -2.0
      THEN 'mean_reversion_long_spread'
      ELSE NULL
    END AS rule_name
FROM spread
WINDOW w_30d AS (ORDER BY delivery_start_utc ROWS BETWEEN 720 PRECEDING AND 1 PRECEDING)
```

`rule_name` falls into `analyst_dayahead_signals` directly. The
matching `analyst_dayahead_positions_proposed` row is the **two-leg
spread** convention from
[`ygg-energy-trading-analyst`](ygg-energy-trading-analyst.md#cross-zone-spread-positions-are-two-leg-one-rationale):
one row long the cheap zone, one row short the rich zone, same
`signal_id`, same `entry_target` (the spread level), same
`rationale`. Don't collapse them.

### Cross-zone FX nuance

Most central-EU bidding zones quote EUR — no FX between FR / DE / NL
/ BE. **Exceptions** that bite analysts:

| Zone | Currency | FX exposure when book is EUR |
| --- | --- | --- |
| GB (N2EX, EPEX UK) | GBP | Real — hedge with `EURGBP` |
| Nordics (NO, SE, DK1/2) | EUR (Nord Pool) | None (the auction is EUR) |
| CH | CHF | Real |
| PL | PLN | Real |
| HU, CZ, RO | EUR (M7) but balancing in local ccy | Mixed — flag by `currency_iso` per row |

The rule: read `currency_iso` from the curated `dayahead` row; if
it doesn't match `base_currency_iso`, call `FxRate.latest([(ccy,
base_ccy)])` at proposal time and store both `fx_rate_used` and
`fx_pair_iso` on the position. See
[`ygg-energy-trading-analyst#fx-risk`](ygg-energy-trading-analyst.md#fx-risk--always-route-through-yggdrasilfxrate).

## Spark / dark / clean spreads — the cross-commodity view

The most-watched fundamental signal on a power book: is the
marginal generating unit profitable at the current power price?

```
spark_spread_eur_mwh   = power_eur_mwh
                       - gas_ttf_eur_mwh / gas_to_power_efficiency

dark_spread_eur_mwh    = power_eur_mwh
                       - coal_api2_eur_mwh / coal_to_power_efficiency

clean_spark_eur_mwh    = spark_spread_eur_mwh
                       - co2_eua_eur_t * 0.36  -- ~0.36 tCO2 per MWh for CCGT

clean_dark_eur_mwh     = dark_spread_eur_mwh
                       - co2_eua_eur_t * 0.90  -- ~0.90 tCO2 per MWh for hard-coal
```

Land the four columns on `main.entsoe.dash_zone_economics` (one row
per `(eic_code, delivery_start_utc)`) and signal off them:

- **Negative clean spark** with `wind_forecast > P75` → short the
  zone (gas plant uneconomic, renewables crowd out demand).
- **Positive clean spark** in summer heat + low wind → long the
  zone (gas dispatching at high cost).
- **Clean dark > clean spark by EUR 5+** in Germany → coal beats gas
  on the stack; coal-favouring jurisdictions outperform gas-heavy ones.

The conversion factors (`gas_to_power_efficiency=0.50` CCGT,
`coal_to_power_efficiency=0.38` lignite, `0.45` hard-coal) are
desk-config; persist in `main.entsoe.analyst_config` not as a
hardcoded literal in the SQL.

## Intraday & balancing — separate signal task

Intraday is a different game — 15-minute resolution, continuous
auction, dominated by renewable forecast revisions. Don't reuse
the day-ahead model.

| Feature | Source | Notes |
| --- | --- | --- |
| `wind_forecast_revision_mw` (delta from prior cycle) | `dash_residual_load` lag join | Strongest IDA signal. |
| `imbalance_price_eur_mwh` (T-1, T-2) | `main.entsoe.imbalance` | Carries TSO scarcity. |
| `last_15m_clearing` | `intraday` | Auto-regressive. |
| `gas_intraday_eur_mwh` | `main.ice.dash_ttf_intraday` | Within-day fuel. |

Lands in `main.entsoe.analyst_intraday_signals` (separate task from
day-ahead). Horizon `PT15M` or `PT1H` only — anything longer belongs
on the day-ahead desk.

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
  pop-weighted EU temperature anomaly; surface on
  `dash_dayahead_risk` so the trader sees they're not running a
  pure spread.
- **DST gaps.** Spring-forward = 23 hours in the day, fall-back =
  25. Curated layer already lands the rows correctly (see
  `ygg-trading-commodity#time-series-hygiene`); the analyst's
  `GROUP BY delivery_start_utc` handles both naturally — don't
  hand-roll a `for hour in range(24)` loop.

## Today's-power-pack — the morning analyst dash

The trader's morning view sits in `main.entsoe.dash_morning_power`:

| Column | Source |
| --- | --- |
| `delivery_start_utc` | curated dayahead |
| `price_<zone>` per major EIC (FR, DE, NL, BE, GB, ES, IT-N, NO2, NO3) | pivot from dayahead |
| `residual_load_<zone>_mw` | dash_residual_load |
| `clean_spark_<zone>_eur_mwh` | dash_zone_economics |
| `spread_zscore_<pair>` per major pair (FR-DE, DE-NL, ES-FR, GB-FR) | analyst signal task |
| `flagged_signals` | `array<string>` of `signal_id` values active today |
| `top_proposal_rationale` | denormalised one-liner from highest-conviction proposal |

One row per delivery hour for `today()` + `today() + 1`. Refreshed
05:00 UTC by the analyst job (after ENTSO-E publishes the D-1
auction at 12:42 CET / D-A clearing). See
[`ygg-display-views`](ygg-display-views.md) for the pivot pattern.

## Don'ts (power-specific — also see the shared skill)

- Don't model day-ahead and intraday in one task. Different feature
  sets, different horizons, different operational pace.
- Don't treat Nordic and continental zones identically. Nord Pool
  is EUR but the bidding-zone topology + reservoir constraints are
  different — train per zone-cluster.
- Don't ignore congestion. A spread signal without an NTC check is
  half a signal — when interconnectors are saturated the spread is
  structurally wide, not mispriced.
- Don't aggregate exposure by vendor-specific zone codes. Always
  resolve through `main.iso.bidding_zone` → `country_iso` → continent.
- Don't store `power_in_gbp` in EUR-denominated fields. GB zone =
  GBP per MWh; persist that, FX-convert on the proposal row, never
  pretend GB cleared in EUR.
- Don't backfill `residual_load_mw` with `coalesce(wind_mw, 0)`
  when wind data is *missing* (vs *zero*). Flag the NULL; the row
  is unreliable and shouldn't drive a signal.
- Don't propose a spread without naming both legs in
  `analyst_dayahead_positions_proposed`. Two rows, shared
  `signal_id`. See
  [`ygg-energy-trading-analyst`](ygg-energy-trading-analyst.md#cross-zone-spread-positions-are-two-leg-one-rationale).
