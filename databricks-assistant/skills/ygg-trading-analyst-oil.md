# Skill: oil trading analyst — Brent, WTI, Dubai, crack spreads, freight, refinery ops

## When to use

The user is doing analyst work on a **crude oil and refined
products** book and asks to "forecast Brent / WTI / Dubai", "compute
the crack spread", "find a curve trade in contango / backwardation",
"size a WTI–Brent arb", "evaluate a tanker arb (USGC → NWE)",
"stress the book under an OPEC+ cut", "publish the oil morning
pack", "track refinery outages". The instrument universe is the
**major crude grades (Brent, WTI, Dubai, ESPO, Urals, WCS)** and
the **refined products (RBOB gasoline, ULSD heating oil, ICE gasoil,
jet, fuel oil)**, and the contracts that settle to them — front
month, calendar strips, dated Brent, Brent–Dubai EFS, RBOB / ULSD
cracks, time spreads.

Routes off [`ygg-energy-trading-analyst`](ygg-energy-trading-analyst.md)
(shared analyst conventions — **read first**, especially the "don't
invent things" rule), uses [`ygg-curated-views`](ygg-curated-views.md) /
[`ygg-trading-commodity`](ygg-trading-commodity.md) (curated crude
settles, products settles, freight, inventory), and
[`ygg-modelist`](ygg-modelist.md) for the forecast model.

## Don't invent tables — applies here too

Every example below uses placeholder names — confirm the actual
table names with the data engineer before writing executable SQL.
Crack-spread formulas, Worldscale math, curve-shape definitions
are real reference conventions and travel; vendor-specific source
table names do not.

## Grade and market universe — refined notions

Real reference points you can quote with confidence:

| Grade | Currency | Native unit | API ° (approx) | Listing |
| --- | --- | --- | --- | --- |
| Brent | USD | USD/bbl | 38 | ICE Brent (MIC `IFEU`) |
| WTI | USD | USD/bbl | 40 | NYMEX (MIC `XNYM`) |
| Dubai | USD | USD/bbl | 31 | Platts assessment |
| ESPO | USD | USD/bbl | 35 | Platts assessment |
| Urals | USD | USD/bbl | 32 | Platts assessment |
| WCS | USD | USD/bbl | 21 | NYMEX (MIC `XNYM`) |

Refined products:

| Product | Listing | Native unit |
| --- | --- | --- |
| RBOB gasoline | NYMEX | USD/gal |
| ULSD heating oil | NYMEX | USD/gal |
| ICE Low Sulphur Gasoil | ICE | USD/t |
| Jet fuel | Platts assessment | USD/bbl |
| Fuel oil 3.5 % / 0.5 % | Platts assessment | USD/t |

Note on currency: oil is overwhelmingly USD-denominated. The FX
exposure on an oil book is on the **product margin** side — a
French refiner's gasoil-vs-Brent crack is realised in EUR (refinery
sells EUR-denominated diesel) even though both legs print in USD.
See [FX in product cracks](#fx-in-product-cracks) below.

## The dominant oil features

### Crack spreads — the refinery economics signal

A simple crack spread expresses refinery margin per barrel:

```
gasoline_crack_usd_bbl = RBOB_usd_gal * 42  - WTI_usd_bbl
diesel_crack_usd_bbl   = ULSD_usd_gal * 42  - WTI_usd_bbl
gasoil_crack_usd_bbl   = ICE_gasoil_usd_t / 7.45  - Brent_usd_bbl
                            -- ~7.45 bbl/t for gasoil
```

The "3-2-1" crack — a US-refining proxy — assumes 3 bbl crude → 2
bbl gasoline + 1 bbl distillate:

```
crack_321_usd_bbl = (2 * RBOB_usd_gal * 42 + ULSD_usd_gal * 42) / 3
                  - WTI_usd_bbl
```

Per refining region the crack benchmark differs: USGC = 3-2-1 vs
WTI / LLS; NWE = gasoil vs Brent; SING = gasoil + jet vs Dubai;
MED = gasoil vs Urals (pre-sanctions baseline). Build the
`<dash_cracks>` view with one row per `(observation_utc, region)`
and the regional crack benchmarks as columns.

Signal — refinery outage drives a regional crack (placeholder
table names):

```sql
WITH outage_capacity_offline AS (
  SELECT
      date(event_utc)            AS d,
      country_iso,
      sum(affected_capacity_kbd) AS kbd_offline
  FROM <curated_events>
  WHERE event_type = 'refinery_outage'
    AND event_utc >= current_date() - INTERVAL 5 DAYS
    AND severity >= 3
  GROUP BY date(event_utc), country_iso
)
SELECT
    c.observation_utc,
    c.region,
    c.gasoline_crack_usd_bbl,
    avg(c.gasoline_crack_usd_bbl) OVER w_30d AS crack_mean_30d,
    stddev(c.gasoline_crack_usd_bbl) OVER w_30d AS crack_std_30d,
    (c.gasoline_crack_usd_bbl - avg(c.gasoline_crack_usd_bbl) OVER w_30d)
      / nullif(stddev(c.gasoline_crack_usd_bbl) OVER w_30d, 0) AS crack_zscore,
    o.kbd_offline AS regional_offline_kbd,
    CASE
      WHEN o.kbd_offline > 500
       AND (c.gasoline_crack_usd_bbl - avg(c.gasoline_crack_usd_bbl) OVER w_30d)
             / nullif(stddev(c.gasoline_crack_usd_bbl) OVER w_30d, 0) < 1.0
      THEN 'long_crack_outage_underpriced'
      ELSE NULL
    END AS rule_name
FROM <dash_cracks> c
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

The signal trades a regional crack-spread long when the outage
hasn't already lifted the crack. Two-leg in
`analyst_crack_positions_proposed`.

### Forward-curve shape — contango vs backwardation

The shape of the Brent / WTI curve carries macro and inventory
information no single front-month does:

```
m1_m2_spread = front_month - second_month   -- positive = backwardation
m1_m12_spread = front_month - month_12      -- structural shape
```

Build `<dash_curve_shape>` per `(grade, observation_utc)` with
`front_back_<n>` columns for `n ∈ {1, 3, 6, 12}`. Signal patterns:

- **Sudden flip from contango to backwardation** in WTI front-back
  → tight US balance signal → long prompt.
- **Wider contango** in Brent with stable inventories → storage-cost
  trade isn't paying; signal the opposite leg.
- **Cross-grade curve divergence** (Brent backwardated, Dubai
  contango) → trade the Brent–Dubai EFS.

The carry economics gate the signal: trade the contango as long as
`m1_m12_spread > storage_cost_per_year + funding_cost`. The
storage-cost value is a desk-config constant per region (typically
expressed as USD/bbl/yr).

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

Each has a slow-moving fundamental level (cost of moving barrels
from one to the other — pipeline tariff, tanker freight, refinery
yield) and a fast-moving mean-reverting deviation. The signal is
z-score deviation from a *regime-conditioned* mean (reset on
`regime_start_utc` — see
[#enex-events-as-regime-breaks](#events-as-regime-breaks)).

### Crude tanker arb — the freight-gated cross-region trade

```
arb_origin_to_dest_usd_bbl
  = dest_grade_usd_bbl
  - (origin_grade_usd_bbl + export_premium_usd_bbl
     + tanker_freight_origin_dest_usd_bbl)
```

Freight conversion (refer to
[`ygg-energy-trading-analyst#freight--shipping`](ygg-energy-trading-analyst.md#freight--shipping--the-cross-hub-arbitrage-gate)
for the wider pattern):

```
tanker_freight_usd_bbl = WS_route_points * flat_rate_usd_t / 100 / bbl_per_tonne
                       -- bbl_per_tonne ≈ 7.33 for crude
```

`WS` is the Worldscale points published daily; `flat_rate_usd_t` is
the annual Worldscale-Association reference rate per route. Build
the converter into the curated freight view or a shared shipping
lookup — **don't recompute in signal SQL**.

Actionability filters:

- `arb_usd_bbl > breakeven_cost` (route-specific).
- Vessel utilisation `< 90 %` for VLCC / Suezmax.
- Destination port has discharge slots.
- No active sanctions on origin grade.
- Destination refinery acceptance of the grade (medium-sour vs
  light-sweet yield).

Two-leg in `analyst_tanker_arb_positions_proposed`.

## Events as regime breaks

OPEC+ announcements, sanctions, pipeline expansions, major outages
fundamentally reset the level of every spread on this desk. Don't
let a z-score signal run a stale window over the event boundary.

When the events feed (negotiate the table name with the data
engineer) flags `event_type IN ('opec_announcement',
'sanctions_change', 'major_pipeline_event', 'major_supply_disruption')`,
the analyst job re-computes rolling statistics from `event_utc`
forward, not from the trailing 30d window. Persist
`regime_start_utc` per grade / region on the curve-shape view so
signal SQL can window from it.

Two consequences:

- **Signal cool-off.** For 5 trading days post-event, conviction on
  all mean-reversion signals on the affected grade is haircut by
  50 %.
- **Event-only trades.** Some events are themselves the alpha (an
  OPEC+ surprise cut → long prompt for 3 sessions). Those go in
  `analyst_event_signals` (separate task), `rule_name='event_<id>'`.

## FX in product cracks

Crude is USD. Refined products in Europe / UK / Asia clear in local
currency (gasoil at NWE clears EUR via barge prices; UK gasoil
clears GBP; Singapore gasoil clears USD but Indian / Japanese
downstream is INR / JPY).

A French refiner's `gasoil_crack_eur_bbl`:

```
gasoil_crack_eur_bbl = (gasoil_eur_t / 7.45)            -- product leg in EUR
                     - (Brent_usd_bbl * fx_usd_eur)     -- crude leg in EUR
```

`fx_usd_eur` from `yggdrasil.fxrate.FxRate`. Persist the rate on the
proposal row. When the desk is EUR-reporting and signals a crack
rally, that's *also* implicitly long USD against EUR — flag the
embedded FX position in the risk snapshot's `currency_iso`
dimension.

## Cross-geo risk on an oil book

Geographic dimensions worth bucketing:

- **`origin_country_iso`** — crude grade origin. WTI = US, Brent =
  North Sea (UK / NO), Dubai = blend, ESPO = RU. Sanctions hit by
  origin.
- **`grade_class`** — light-sweet / medium-sour / heavy-sour.
  Covaries with origin and refinery economics.
- **`route_choke_point`** — Suez, Hormuz, Malacca, Panama,
  Bab-el-Mandeb.
- **`refining_region`** — for product cracks: USGC / NWE / SING /
  MED.
- **`destination_country_iso`** — for cargo-by-cargo signals.

## Hurricane / weather risk — short-window, high-impact

USGC hurricane season (Jun–Nov) reliably moves WTI, RBOB cracks,
and Henry Hub gas. The hurricane track / cone is supplied via NHC
(National Hurricane Center) feeds — the data engineer lands the
cone polygons (`forecast_cone_geojson`); the analyst intersects
them with refinery / platform location dims (lat/lon-bearing, see
[`ygg-energy-trading-analyst#cross-geo-zone-risk`](ygg-energy-trading-analyst.md#cross-geo-zone-risk))
to compute the % of USGC capacity in the 72-hour cone.

Signal: ≥ 15 % of refining capacity in cone → long RBOB crack;
≥ 20 % of gas platforms in cone → long Henry Hub prompt + long
gasoil crack. Carry the cone polygon on the proposal row's rationale
payload so the trader sees the spatial argument.

## Forecast model — typical feature set (Brent prompt D+5)

Conceptual features the modelist materialises:

| Feature | Shape |
| --- | --- |
| `lag_1d`, `lag_5d`, `lag_30d` price | curated settle |
| `realised_vol_5d`, `realised_vol_30d` | derived |
| `m1_m2_spread`, `m1_m12_spread` | curve shape |
| `cushing_inventory_kb` (WTI-specific), `padd_3_crude_kb` | EIA weekly |
| `opec_production_anomaly` | OPEC monthly vs target |
| `usd_dxy`, `usd_eur_fx` | macro + `FxRate` |
| `usd_2y`, `usd_10y_real` | macro |
| `tanker_arb_usgc_nwe` (lag 1) | tanker arb dash |
| `crack_321_usgc`, `crack_gasoil_nwe` | cracks dash |
| `event_severity_max_5d` | events feed |
| `hurricane_capacity_in_cone_pct` | hurricane intersection |
| `holiday_us`, `dow` | calendar |

Default: quantile LightGBM with regime-aware feature engineering
(window-resetting on `regime_start_utc` applied to the training
data). Via the modelist's multi-candidate loop.

## Don'ts (oil-specific — also see the shared skill)

- Don't run a z-score signal through an event boundary. OPEC+ /
  sanctions / pipeline events reset the regime; respect
  `regime_start_utc`.
- Don't compute a tanker arb without Worldscale → USD/bbl
  conversion. Hand-rolled `WS * 0.01 * something` in signal SQL
  is the standard way to get the arb wrong by 30 %.
- Don't trade a "WTI-Brent looks rich" signal without checking
  Cushing inventory and US Gulf export capacity.
- Don't model crude with the same template as products. Cracks
  follow product-balance dynamics; crude follows global supply-
  demand + OPEC behaviour.
- Don't ignore embedded FX on EUR-reporting product books.
- Don't aggregate refinery exposure by company name. Aggregate by
  `refining_region`.
- Don't roll a crude-arb signal without checking destination
  refinery acceptance of the grade.
- Don't ignore the hurricane cone during USGC season.
