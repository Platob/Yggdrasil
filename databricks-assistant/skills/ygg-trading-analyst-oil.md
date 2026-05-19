# Skill: oil trading analyst — Brent, WTI, Dubai, crack spreads, freight, refinery ops

## When to use

The user is doing analyst work on a **crude oil and refined
products** book and asks to "forecast Brent / WTI / Dubai", "compute
the crack spread", "find a curve trade in contango / backwardation",
"size a WTI–Brent arb", "evaluate a tanker arb (USGC → NWE)", "stress
the book under an OPEC+ cut", "publish the oil morning pack", "track
refinery outages". The instrument universe is **crude grades (Brent,
WTI, Dubai, ESPO, Urals, WCS), refined products (RBOB gasoline,
ULSD heating oil, gasoil ICE, jet, fuel oil), and the contracts that
settle to them** — front month, calendar strips, dated Brent,
Brent–Dubai EFS, RBOB / ULSD cracks, time spreads.

Routes off [`ygg-energy-trading-analyst`](ygg-energy-trading-analyst.md)
(shared analyst conventions — read first), uses
[`ygg-curated-views`](ygg-curated-views.md) /
[`ygg-trading-commodity`](ygg-trading-commodity.md) (curated
`<source>.brent_settle`, `<source>.wti_settle`, `<source>.dubai_assess`,
`<source>.products_settle`), and [`ygg-mlops`](ygg-mlops.md) for the
forecast model.

## Oil-desk inputs (curated tables you read)

| Curated table | Carries | Refresh |
| --- | --- | --- |
| `main.ice.brent_settle` | ICE Brent prompt + curve, USD/bbl | Daily 19:30 BST |
| `main.nymex.wti_settle` | NYMEX WTI prompt + curve, USD/bbl | Daily 14:30 EST |
| `main.platts.dubai_assess` | Platts Dubai window assessment, USD/bbl | Daily |
| `main.ice.gasoil_settle` | ICE Low Sulphur Gasoil, USD/t | Daily |
| `main.nymex.rbob_settle` | RBOB gasoline, USD/gal | Daily |
| `main.nymex.ulsd_settle` | ULSD heating oil, USD/gal | Daily |
| `main.eia.weekly_petroleum` | US crude / products inventory, refinery utilisation, imports | Wednesday 10:30 ET |
| `main.iea.monthly_oil_stocks` | OECD oil stocks by region | First week of month |
| `main.opec.production` | OPEC+ monthly + secondary-source production | Monthly |
| `main.baltic.tanker` | BDTI / BCTI dirty + clean tanker rates, Worldscale | Daily |
| `main.kpler.crude_flows` | Cargo-tracked crude flows, origin → destination, ETA | Continuous |
| `main.iir.outages` | Refinery / pipeline / upstream outages with capacity impact | Event |
| `main.weather.<provider>.hurricane` | NHC hurricane tracks / cones | 6-hourly during season |
| `main.enex.events` | Sanctions, OPEC+ announcements, geopolitical | Event |

Note on currency: oil is overwhelmingly USD-denominated. The FX
exposure on an oil book is on the **product margin** side — a French
refiner's gasoil-vs-Brent crack is realised in EUR (refinery sells
EUR-denominated diesel) even though both legs print in USD. See
[FX in product cracks](#fx-in-product-cracks) below.

## The dominant oil features

### Crack spreads — the refinery economics signal

A simple crack spread expresses refinery margin per barrel:

```
gasoline_crack_usd_bbl = RBOB_usd_gal * 42  - WTI_usd_bbl
diesel_crack_usd_bbl   = ULSD_usd_gal * 42  - WTI_usd_bbl
gasoil_crack_usd_bbl   = ICE_gasoil_usd_t / 7.45  - Brent_usd_bbl   -- ~7.45 bbl/t
```

The "3-2-1" crack — a US-refining proxy — assumes 3 bbl crude → 2 bbl
gasoline + 1 bbl distillate:

```
crack_321_usd_bbl = (2 * RBOB_usd_gal * 42 + ULSD_usd_gal * 42) / 3
                  - WTI_usd_bbl
```

Build `main.oil.dash_cracks` with one row per
`(observation_utc, region)` where `region ∈ {'USGC', 'NWE', 'SING', 'MED'}`
and columns for each crack you trade. Region matters: NWE cracks
trade gasoil (not ULSD) against Brent (not WTI).

Signal — refinery outage drives a regional crack:

```sql
WITH outage_capacity_offline AS (
  SELECT
      date(event_utc)                  AS d,
      country_iso,
      sum(affected_capacity_kbd)       AS kbd_offline
  FROM main.iir.outages
  WHERE event_type = 'refinery_outage'
    AND event_utc >= current_date() - INTERVAL 5 DAYS
    AND severity >= 3
  GROUP BY date(event_utc), country_iso
)
SELECT
    c.observation_utc,
    c.region,
    c.gasoline_crack_usd_bbl,
    avg(c.gasoline_crack_usd_bbl) OVER w_30d        AS crack_mean_30d,
    stddev(c.gasoline_crack_usd_bbl) OVER w_30d     AS crack_std_30d,
    (c.gasoline_crack_usd_bbl - avg(c.gasoline_crack_usd_bbl) OVER w_30d)
      / nullif(stddev(c.gasoline_crack_usd_bbl) OVER w_30d, 0) AS crack_zscore,
    o.kbd_offline                                    AS regional_offline_kbd,
    CASE
      WHEN o.kbd_offline > 500
       AND (c.gasoline_crack_usd_bbl - avg(c.gasoline_crack_usd_bbl) OVER w_30d)
             / nullif(stddev(c.gasoline_crack_usd_bbl) OVER w_30d, 0) < 1.0
      THEN 'long_crack_outage_underpriced'
      ELSE NULL
    END                                              AS rule_name
FROM main.oil.dash_cracks c
LEFT JOIN outage_capacity_offline o
    ON o.d = date(c.observation_utc)
   AND o.country_iso = c.region_country_iso
WHERE c.observation_utc >= current_date() - INTERVAL 60 DAYS
WINDOW w_30d AS (
  PARTITION BY c.region
  ORDER BY c.observation_utc
  ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
)
```

The signal trades a regional crack-spread long when the outage hasn't
already lifted the crack. Both legs (long product, short crude) land
in `analyst_crack_positions_proposed` with a shared `signal_id`.

### Forward-curve shape — contango vs backwardation

The shape of the Brent / WTI curve carries macro and inventory
information no single front-month does:

```
m1_m2_spread = front_month - second_month   -- positive = backwardation
m1_m12_spread = front_month - month_12      -- structural shape
```

Build `main.oil.dash_curve_shape` per `(grade, observation_utc)`
with `front_back_eur_bbl_<n>` columns for `n ∈ {1,3,6,12}`. The
signal patterns:

- **Sudden flip from contango to backwardation** in WTI front-back
  → tight US balance signal → long prompt.
- **Wider contango** in Brent with stable inventories → storage-cost
  trade isn't paying; signal the opposite leg (sell the storage
  arb).
- **Cross-grade curve divergence** (Brent backwardated, Dubai
  contango) → trade the Brent–Dubai EFS.

The carry economics gate the signal: trade the contango as long as
`m1_m12_spread > storage_cost_per_year + funding_cost`. The storage
cost number lives in `main.oil.analyst_config` as `storage_cost_usd_bbl_year`
per region.

### Inter-grade differentials

The classic three:

```
brent_wti_spread = Brent_usd_bbl - WTI_usd_bbl
                  -- driven by US export economics, Cushing inventory
brent_dubai_efs  = Brent_usd_bbl - Dubai_usd_bbl
                  -- East-West arb gate, OPEC vs non-OPEC light/sweet
wti_wcs_diff     = WTI_usd_bbl - WCS_canada_usd_bbl
                  -- US heavy vs light, pipeline constraint
```

Each lives on `main.oil.dash_grade_spreads`. Each has a slow-moving
fundamental level (cost of moving barrels from one to the other —
pipeline tariff, tanker freight, refinery yield) and a fast-moving
mean-reverting deviation. The signal is z-score deviation from a
*regime-conditioned* mean (i.e., reset the rolling window when an
OPEC+ announcement or pipeline-expansion event lands — see
[`#enex-events-as-regime-breaks`](#enex-events-as-regime-breaks)).

### Crude tanker arb — the freight-gated cross-region trade

```
arb_USGC_to_NWE_usd_bbl = Brent_usd_bbl
                        - (WTI_usd_bbl + USGC_export_premium_usd_bbl
                           + tanker_freight_USGC_NWE_usd_bbl)

arb_USGC_to_SING_usd_bbl = Dubai_usd_bbl
                         - (WTI_usd_bbl + USGC_export_premium_usd_bbl
                            + tanker_freight_USGC_SING_usd_bbl)
```

Freight conversion: `BDTI` is published in **Worldscale (WS)** points;
the analyst needs `usd_per_bbl` for the route, which is
`WS_route_points * flat_rate_route / 100 * 1 / 7.33` (≈ 7.33 bbl /
tonne for crude). The conversion lives in
`main.iso.shipping_route(flat_rate_usd_t, bbl_per_tonne)` —
**don't recompute it in signal SQL**. See
[`ygg-energy-trading-analyst#freight--shipping`](ygg-energy-trading-analyst.md#freight--shipping--the-cross-hub-arbitrage-gate)
for the wider pattern.

Build `main.oil.dash_tanker_arb` with rows per
`(origin_port, destination_port, observation_utc)`. The
**signal-actionable** filter on top:

| Check | Source |
| --- | --- |
| `arb_usd_bbl > breakeven_cost` (route-specific) | dash_tanker_arb |
| Vessel utilisation `< 90%` for VLCC / Suezmax | `main.baltic.tanker` |
| Destination port has discharge slots | `main.flexport.ais_dwell` |
| No active sanctions on origin grade | `main.enex.events` |

The trade is two legs (long destination grade, short origin grade or
WTI) — same two-row convention.

## ENEX events as regime breaks

OPEC+ announcements, sanctions, pipeline expansions, major outages
fundamentally reset the level of every spread on this desk. Don't
let a z-score signal run a stale window over the event boundary.

`main.enex.events.event_type IN ('opec_announcement',
'sanctions_change', 'major_pipeline_event', 'major_supply_disruption')`
triggers a *regime reset*: the analyst job re-computes rolling
statistics from `event_utc` forward, not from the trailing 30d
window. Persist `regime_start_utc` per grade / region on
`main.oil.dash_curve_shape` so signal SQL can `WINDOW … RANGE
BETWEEN regime_start_utc AND current_row`.

Two consequences:

- **Signal cool-off.** For 5 trading days post-event, conviction
  on all mean-reversion signals on the affected grade is haircut
  by 50% — the new equilibrium isn't established yet.
- **Event-only trades.** Some events are themselves the alpha (an
  OPEC+ surprise cut → long prompt for 3 sessions). Those go in
  `analyst_event_signals` (separate task), `rule_name='event_<id>'`.

## FX in product cracks

Crude is USD. Refined products in Europe / UK / Asia clear in local
currency (gasoil at NWE clears EUR via barge prices; UK gasoil
clears GBP; Singapore gasoil clears USD but Indian / Japanese
downstream is INR / JPY).

A French refiner's `gasoil_crack_eur_bbl` is the realised P&L driver:

```
gasoil_crack_eur_bbl = (gasoil_eur_t / 7.45)            -- product leg in EUR
                     - (Brent_usd_bbl * fx_usd_eur)     -- crude leg in EUR
```

`fx_usd_eur` from `yggdrasil.fxrate.FxRate`; see
[`ygg-energy-trading-analyst#fx-risk`](ygg-energy-trading-analyst.md#fx-risk--always-route-through-yggdrasilfxrate).
Persist the rate on the proposal row (`fx_rate_used`,
`fx_pair_iso='USDEUR'`). When the desk is EUR-reporting and signals
a crack rally, that's *also* implicitly long USD against EUR — flag
the embedded FX position in the risk snapshot's `currency_iso`
dimension.

## Cross-geo risk on an oil book

Oil's geographic dimensions:

- **`origin_country_iso`** — crude grade origin. WTI = US, Brent =
  North Sea (UK / NO), Dubai = blend, ESPO = RU. Sanctions hit by
  origin.
- **`grade`** — light-sweet / medium-sour / heavy-sour. Not strictly
  geographic but covaries with origin and refinery economics.
- **`route_choke_point`** — Suez, Hormuz, Malacca, Panama. A long
  Brent / short Dubai trade routed via Suez is exposed to Suez
  disruption even though the legs aren't.
- **`refining_region`** — for product cracks: USGC / NWE / SING /
  MED. Concentration limits go here.
- **`destination_country_iso`** — for cargo-by-cargo signals.

Aggregations follow the same pattern as
[`ygg-energy-trading-analyst#cross-geo-zone-risk`](ygg-energy-trading-analyst.md#cross-geo-zone-risk),
joining `main.iso.crude_grade` and `main.iso.refining_region`.

Geopolitical concentration check: when ≥ 25% of the book's gross
notional sits in a single `route_choke_point`, the risk snapshot
flags it (`concentration_pct > 0.25` on `dimension='route_choke_point'`).

## Hurricane / weather risk — short-window, high-impact

USGC hurricane season (Jun–Nov) reliably moves WTI, RBOB cracks,
and Henry Hub gas. Curated `main.weather.<provider>.hurricane`
carries NHC cone polygons (`forecast_cone_geojson`); the analyst
job intersects them with `main.iso.refinery(lat, lon)` and 
`main.iso.gulf_platform(lat, lon)` to compute the % of USGC 
capacity in the 72-hour cone.

Signal: ≥ 15% of refining capacity in cone → long RBOB crack
(supply-shock priced in); ≥ 20% of gas platforms in cone → long
Henry Hub prompt + long gasoil crack (gas-fired-power feedback +
distillate demand for backup generation). Always carry the cone
polygon on the proposal row's `rationale` payload so the trader
can see the spatial argument, not just a number.

## Forecast model — typical feature set (Brent prompt-month D+5)

| Feature | Source |
| --- | --- |
| `lag_1d`, `lag_5d`, `lag_30d` price | curated |
| `realised_vol_5d`, `realised_vol_30d` | dash_vol |
| `m1_m2_spread`, `m1_m12_spread` | dash_curve_shape |
| `cushing_inventory_kb` (WTI specific), `padd_3_crude_kb` | EIA weekly |
| `opec_production_anomaly` | OPEC monthly vs target |
| `usd_dxy`, `usd_eur_fx` | macro + FxRate |
| `usd_2y`, `usd_10y_real` | macro |
| `tanker_arb_USGC_NWE` (lag 1) | dash_tanker_arb |
| `crack_321_usgc`, `crack_gasoil_nwe` | dash_cracks (cross-leg feedback) |
| `event_severity_max_5d` | enex.events |
| `hurricane_capacity_in_cone_pct` | dash_hurricane |
| `holiday_us`, `dow` | calendar |

Quantile LightGBM with regime-aware feature engineering — the same
window-resetting on `regime_start_utc` applied to the training data.
Standard MLflow pipeline via [`ygg-mlops`](ygg-mlops.md).

## Morning oil pack — the dash table

`main.oil.dash_morning_oil`:

| Column | Source |
| --- | --- |
| `price_brent_usd_bbl`, `price_wti_usd_bbl`, `price_dubai_usd_bbl` | curated |
| `m1_m2_spread_brent`, `m1_m2_spread_wti` | curve shape |
| `brent_wti_spread`, `brent_dubai_efs` | grade spreads |
| `crack_321_usgc`, `crack_gasoil_nwe`, `crack_gasoil_sing` | dash_cracks |
| `cushing_inventory_kb`, `padd_3_crude_kb` | EIA weekly |
| `arb_USGC_NWE_usd_bbl`, `arb_USGC_SING_usd_bbl` | dash_tanker_arb |
| `refinery_offline_kbd_global` | iir.outages aggregate |
| `flagged_signals` | array<string> |
| `regime_start_utc` per grade | dash_curve_shape |
| `top_rationale` | denormalised one-liner |

Refreshed by the analyst job after NYMEX settle.

## Don'ts (oil-specific — also see the shared skill)

- Don't run a z-score signal through an event boundary. OPEC+ /
  sanctions / pipeline events reset the regime; respect
  `regime_start_utc` on the dash.
- Don't compute a tanker arb without Worldscale → USD/bbl conversion
  via `main.iso.shipping_route`. Hand-rolled `WS * 0.01 * something`
  in signal SQL is the standard way to get the arb wrong by 30%.
- Don't trade a "WTI-Brent looks rich" signal without checking
  Cushing inventory and US Gulf export capacity. The spread is
  bounded by export economics.
- Don't model crude with the same template as products. Cracks
  follow product-balance dynamics (regional demand + refinery
  utilisation); crude follows global supply-demand + OPEC behaviour.
- Don't ignore embedded FX on EUR-reporting product books. The
  USD crude leg makes a "crack rally" a long-USD position; flag it
  on the risk snapshot.
- Don't aggregate refinery exposure by company name. Aggregate by
  `refining_region`; a single owner across PADDs is a different
  risk than the same notional concentrated in PADD 3.
- Don't roll a crude-arb signal without checking destination grade
  acceptance — a refinery configured for medium-sour can't process
  light-sweet at the same yield. `main.iso.refinery.acceptable_grades`
  is the gate.
- Don't ignore the hurricane cone during USGC season. A signal
  computed off "yesterday's price" misses what the cone has already
  done to today's open.
