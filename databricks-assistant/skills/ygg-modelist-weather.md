# Skill: weather modelist — temperature, photovoltaic, wind power, precipitation forecasts

## When to use

The user is acting as a **weather modelist** — the modelling
specialist who turns gridded numerical-weather-prediction (NWP)
output and observed weather into the per-zone, demand-weighted,
capacity-weighted features the energy desks consume. They ask to
"forecast tomorrow's wind power in DE", "build a per-zone HDD/CDD
table", "produce a PV (photovoltaic) output forecast", "model the
inflow into a hydro basin from rainfall", "blend two NWP runs",
"benchmark ECMWF HRES vs GFS for wind", "publish per-zone weather
features to the modelist's feature store".

Specialised sibling of [`ygg-modelist`](ygg-modelist.md) — read it
first for the general training / simulation / KPI / explainability
contract. This skill adds the **weather-specific feature shapes,
physical conversions, and decision-KPIs** the energy desks depend
on.

The output the weather modelist owns is consumed by:

- The power desk's residual-load + dayahead / intraday models.
- The gas desk's HDD/CDD and storage-draw signals.
- The oil desk's distillate demand (heating oil) and hurricane
  refinery / platform intersection.
- The other modelist tasks via the curated feature store.

## Don't invent things — mirror of the shared rule

Same rule as
[`ygg-energy-trading-analyst#dont-invent-things`](ygg-energy-trading-analyst.md#dont-invent-things--ask-or-use-a-placeholder):

- **Don't reference a vendor NWP feed, a gridded dataset, or a
  curated table by name** unless the user confirmed it exists or
  yggdrasil ships it. Use placeholders (`<nwp_hres_gridded>`,
  `<observed_station_temps>`) and ask.
- **Refined specialised notions are fair game.** Temperature
  scales (°C, °F), HDD/CDD basis (65 °F / 18.3 °C), wind hub
  heights (typically 100 m onshore, 120–150 m offshore), Betz
  limit (0.593), solar irradiance components (GHI, DNI, DHI),
  performance ratio (PR), basin runoff coefficients, NWP run
  cycles (00 / 06 / 12 / 18 UTC), forecast horizons (T+1h …
  T+240h), `GeoZoneCatalog` lat/lon.
- **Always be able to explain it.** Cite the NWP vendor docs
  (ECMWF, NOAA NCEP, DWD ICON, Météo-France ARPEGE / AROME), the
  IEC / IEA standard you're using (IEC 61400-1 for wind, IEC
  61724 for PV performance, ASHRAE for HDD basis), or the
  yggdrasil module.

## What the weather modelist owns

```
<gridded NWP / observed weather>     ← data engineer (HTTP / S3 / OPeNDAP)
└── <zone-weighted weather features>  ← WEATHER MODELIST (this skill)
    ├── per-zone temperature features
    ├── per-zone wind power features
    ├── per-zone PV output features
    ├── per-basin / per-zone precipitation features
    ├── candidate model runs            ← same modelist contract
    ├── KPI matrix + explainability
    └── dashboard / notebook surface
        └── consumed by other modelists → energy desks
```

The data engineer lands the raw gridded NWP (HRES, GFS, ICON,
AROME) and station observations. The weather modelist:

1. **Spatially aggregates** grid points to the energy desk's
   bidding-zone / demand-region / basin polygons, weighted by the
   physical driver (population for temp, installed wind capacity
   for wind, installed PV capacity for solar, basin area for
   precip).
2. **Converts** raw grid quantities to energy-relevant quantities
   (temperature → HDD/CDD; wind at 10 m → wind power at hub height;
   GHI + temp → PV output).
3. **Blends** multiple NWP runs into an ensemble forecast (mean +
   spread, optionally bias-corrected).
4. **Trains and benchmarks** candidate post-processing models that
   correct NWP forecasts using recent observations (statistical
   downscaling, neural post-processing).
5. **Publishes** the per-zone, per-hour features to the modelist
   feature store via `Schema.from_fields([...])` + `Table.insert`.

## Feature class 1 — temperature → HDD/CDD

The strongest single feature for gas + power demand. The convention
is U.S.-rooted but is the trading standard:

```
HDD_t  = max(0, 65 °F - daily_avg_temp_f)   -- 18.3 °C basis
CDD_t  = max(0, daily_avg_temp_f - 65 °F)
```

Per-zone features the desks expect:

| Column | Refined notion |
| --- | --- |
| `temp_c_pop_weighted` | Population-polygon-weighted 2-m temperature. Use a recent year of the actual population grid; the energy desk weighting matters more than a decade-old census. |
| `temp_anomaly_c` | Temp − 30-year rolling normal for the same `(zone, doy, hod)`. Seasonality-adjusted demand driver. |
| `hdd_18_3c`, `cdd_18_3c` | Per-day, per-zone aggregates. |
| `hdd_15c`, `cdd_22c` | Alternate bases the desk may need (UK gas industry uses 15 °C; aircon load uses 22 °C). |
| `temp_extreme_flag` | `temp < q5` or `temp > q95` of the 30-y rolling distribution. Driver of tail prompt-gas / prompt-power moves. |
| `dewpoint_c`, `humidity_pct` | Cooling load is humidity-sensitive — not temperature alone. |
| `forecast_horizon_h` | T+1 .. T+240 typically. |
| `forecast_age_hours` | Timestamp of last NWP run; older runs get downweighted in ensembles. |
| `provider`, `model_run_utc` | Which NWP cycle this came from. |

Population polygon source: the data engineer's `<population_grid>`
(GPWv4, WorldPop, or NUTS-level census) joined to the energy desk's
`<bidding_zone_polygon>` / `<gas_demand_region_polygon>`. Confirm
the actual polygon table names before writing the aggregation SQL.

## Feature class 2 — wind power

Wind speed at 10 m (the typical NWP output level) is not what
turbines see. The energy desk wants **per-zone wind power output**,
not wind speed. Two-step conversion:

### Step 1 — extrapolate to hub height

Use the power law (or log law if you have surface-roughness data):

```
wind_speed_hub_ms = wind_speed_10m_ms * (hub_height_m / 10) ** alpha
                  -- alpha ≈ 0.14 onshore (flat / mixed terrain)
                  --        ≈ 0.11 offshore (smooth water)
```

Hub heights vary by fleet vintage; per-zone use the
installed-capacity-weighted average hub height (typically 90–110 m
onshore, 120–150 m offshore). Persist as a desk-config so it
updates as new turbines commission.

### Step 2 — apply the power curve

The relationship between wind speed and turbine output is the
**power curve** — characteristic of the turbine model, not a
clean cubic. Real curves have:

- **Cut-in** speed (~3 m/s): below = zero output.
- **Cubic region** (~3–12 m/s): power ∝ wind_speed³ × Cp, capped
  by the Betz limit (0.593) in theory, ~0.45 in practice.
- **Rated** plateau (~12–25 m/s): power = rated.
- **Cut-out** (~25 m/s, ~28 m/s storm-mode): zero output to
  protect the turbine.

For zone-level forecasting, the fleet-average power curve is what
matters (one mid-range turbine curve like IEC Class II, or a
weighted blend of the deployed models). Persist the curve as a
lookup `(wind_speed_ms, capacity_fraction)` desk-config table; the
weather modelist's feature build does the lookup vectorised, not
in a Python loop (see CLAUDE.md → "No Python `for` loops over
data").

### Per-zone wind features

| Column | Refined notion |
| --- | --- |
| `wind_speed_hub_ms_capacity_weighted` | After both polygon + hub-height conversion. |
| `wind_power_mw_capacity_weighted` | Power curve applied per grid point, then capacity-weighted aggregated. |
| `wind_power_capacity_factor` | `wind_power_mw / installed_capacity_mw`. Dimensionless; comparable across zones. |
| `wind_direction_deg` | Less common but matters for offshore wake effects. |
| `wind_temperature_offset` | Cold-weather de-rate region (~-15 °C): some turbines park. |
| `forecast_horizon_h`, `provider`, `model_run_utc` | Same as temp. |

The capacity-weighting polygon is the **installed wind capacity
grid** by zone — not the population grid. Per-zone wind capacity
typically comes from ENTSO-E generation registry or national TSO
publishings; the data engineer's job to land. Use
`GeoZoneCatalog` for zone → polygon lookup.

## Feature class 3 — photovoltaic (PV) output

PV power isn't just irradiance — it's irradiance, panel temperature,
soiling, performance ratio, and module orientation.

### Irradiance decomposition

NWP usually ships **GHI** (global horizontal irradiance, W/m²);
some models ship **DNI** (direct normal) and **DHI** (diffuse
horizontal) explicitly. The relationships:

```
GHI = DNI * cos(zenith) + DHI                     -- physical identity
```

For fixed-tilt rooftop / utility-scale arrays, plane-of-array (POA)
irradiance is what hits the panels:

```
POA = DNI * cos(angle_of_incidence)
    + DHI * sky_diffuse_factor(tilt)
    + GHI * albedo * ground_diffuse_factor(tilt)
```

For a zone-level forecast, build a **fleet-average tilt and
azimuth** (typically tilt ≈ latitude, azimuth ≈ 180° south in
northern hemisphere); the data engineer should land the per-zone
installed-PV-capacity-weighted tilt/azimuth as desk-config.

### PV power model — quick reference

```
P_dc = POA * panel_area * efficiency_stc
       * (1 - temp_coeff * (panel_temp_c - 25))

panel_temp_c = ambient_temp_c + (POA / 800) * NOCT_offset
            -- NOCT_offset typically 20-25 °C

P_ac = P_dc * inverter_efficiency * (1 - soiling_loss) * (1 - shading_loss)
performance_ratio = P_ac / (POA * panel_area * efficiency_stc)
                 -- typically 0.75-0.85 for utility-scale
```

For zone forecasting, the simpler shape is:

```
pv_power_capacity_factor = POA / 1000 * temp_derate_factor * PR
                         -- 1000 W/m² is the STC reference
```

`temp_derate_factor = max(0, 1 - 0.004 * (panel_temp_c - 25))`
gives a usable approximation for crystalline silicon; refine with
the actual fleet's temp coefficient if known.

### Per-zone PV features

| Column | Refined notion |
| --- | --- |
| `ghi_wm2_capacity_weighted` | Capacity-polygon-weighted aggregate. |
| `dni_wm2_capacity_weighted` | When the NWP provides it. |
| `poa_wm2_capacity_weighted` | After tilt / azimuth projection. |
| `cloud_cover_pct` | Sometimes a better feature than GHI for short-horizon corrections — clouds are the residual NWP misses. |
| `pv_power_capacity_factor` | After all derates applied. |
| `pv_power_mw` | Capacity factor × installed PV capacity (zone-level). |
| `panel_temp_c` | Derived from ambient temp + POA. |
| `solar_zenith_deg` | Deterministic from `(lat, lon, observation_utc)` — cheap to compute and a strong feature for sub-hour models. |
| `provider`, `model_run_utc`, `forecast_horizon_h` | Standard. |

Solar zenith is *deterministic* — not from NWP. Use a library-free
formula or `astropy`-style calculation; it's a free feature.

## Feature class 4 — precipitation and basin runoff

Power-relevant only for hydro-heavy markets (Nordic, Alpine, parts
of South America). Two-stage conversion:

```
basin_precip_mm = sum_grid(precip_mm_per_cell * cell_area_in_basin)
                  / basin_area
                  -- spatial average over the hydrological basin

basin_runoff_gwh = basin_precip_mm * basin_area_km2
                 * runoff_coefficient
                 * head_efficiency_ratio
                 -- runoff_coefficient varies by basin geology + snowpack state
                 -- head_efficiency is the dam / plant chain conversion
```

`runoff_coefficient` is highly basin-specific and seasonally
varying (snowpack-loaded basins release in spring melt). Treat as
a per-basin desk-config that the weather modelist tunes from
observed inflows.

### Per-basin features

| Column | Refined notion |
| --- | --- |
| `precip_mm_basin` | Basin-averaged accumulation per period. |
| `snow_water_equivalent_mm_basin` | When the NWP provides SWE — predicts spring melt. |
| `basin_soil_moisture_pct` | Saturated soil → higher runoff coefficient; tracks via the moisture grid. |
| `runoff_forecast_mm` | Conceptual; the actual implementation is a small physics or learned model on top of `precip + SWE + soil_moisture`. |
| `reservoir_inflow_gwh_forecast` | Net forecast for the reservoir. Cross-checked against TSO-published reservoir levels. |

## NWP providers — refined notions

You can reference these by name with confidence (they ship publicly,
have stable conventions, and the energy desks know them):

| Provider | Run cycles (UTC) | Horizons | Notes |
| --- | --- | --- | --- |
| ECMWF HRES (IFS) | 00 / 06 / 12 / 18 | T+1h .. T+240h | Highest skill globally for most variables; commercial license. |
| ECMWF ENS (ensemble) | 00 / 06 / 12 / 18 | 50 members | For probabilistic / uncertainty signals. |
| NOAA GFS | 00 / 06 / 12 / 18 | T+1h .. T+384h | Public-domain, US-anchored skill. |
| NOAA HRRR | hourly | T+1h .. T+48h | High-res CONUS; great for US wind / solar. |
| DWD ICON | 00 / 06 / 12 / 18 | T+1h .. T+180h | EU-focused. |
| Météo-France ARPEGE / AROME | 00 / 06 / 12 / 18 | T+1h .. T+102h (ARPEGE), T+1h .. T+51h (AROME) | High-res France / Western EU. |
| UK Met Office UM | 00 / 06 / 12 / 18 | T+1h .. T+144h | High-res UK. |

The data engineer is responsible for the actual ingestion (S3 /
HTTP / OPeNDAP / GRIB2 decoding) — the weather modelist consumes
the curated gridded output. See
[`ygg-trading-commodity#raw_-schema-templates`](ygg-trading-commodity.md#raw_-schema-templates)
for the raw landing shape.

## Post-processing models — what the weather modelist trains

NWP forecasts are biased and have systematic errors (under-prediction
of wind ramps, over-prediction of summer afternoon clouds in
specific regions). The weather modelist trains **statistical
post-processors** that correct NWP using recent observations.

| Candidate | When |
| --- | --- |
| **Naive persistence** (`y_hat = last_observed`) | Baseline for short horizons. |
| **NWP raw** (no correction) | Baseline for long horizons. |
| **Linear bias correction** (per-zone, per-horizon, per-hour-of-day) | Cheap, robust, ~10–20 % MAE improvement on most variables. |
| **LightGBM with lag features** | The workhorse — corrects multi-feature interactions. |
| **NHITS / TFT** | Multi-horizon deep models; worth the cost for wind power, where ramps are the alpha. |
| **Quantile regression** at p10 / p50 / p90 | Always — the energy desks need intervals, not points. |
| **Ensemble blend** (HRES + GFS + ICON via stacked regression) | Production champion typically beats any single model. |

Same multi-candidate training + KPI-matrix contract as
[`ygg-modelist`](ygg-modelist.md#many-simulation-runs--the-modelists-loop)
— nested MLflow runs, walk-forward back-tests, long-format KPI
matrix in `ml_<task>_run_metrics`. Same explainability persistence
in `ml_<task>_run_features`.

## Weather-modelist KPIs — model + decision

Standard model KPIs apply (`rmse`, `mae`, `mape`, `pinball_p10/p50/p90`,
`crps`, `coverage_80`). Add decision KPIs that map to the consumer
desk:

| KPI | What it tells the downstream model |
| --- | --- |
| `wind_ramp_detect_f1` | F1 score on detecting ramps > X MW / h. Wind-power signal alpha lives in ramps. |
| `pv_clearsky_index_mae` | MAE on the clear-sky-normalised index, removes seasonality. |
| `hdd_z_score_directional_acc` | Sign-correct rate on HDD anomaly. Drives gas signal calibration. |
| `extreme_event_recall` | Recall on top / bottom 5 % temperature days. Tail moves are the alpha for prompt gas / power. |
| `basin_inflow_mae_gwh` | Reservoir-relevant inflow error. |
| `coverage_80_per_horizon` | Calibration of the p10/p90 band at each horizon. Wind drops out fast; coverage at T+24h is the trader's number. |

Slice all of these `by_zone`, `by_horizon`, `by_season`, `by_regime`
(if you've defined regime breaks — e.g. fleet expansion events).

## Notebook summary per run — weather-specific cells

Inherit the rendered-notebook contract from
[`ygg-modelist#per-run-notebook-artifact`](ygg-modelist.md#per-run-notebook-artifact);
add these weather-specific cells:

1. **Per-zone error map** — choropleth of MAE by zone for the
   target variable, rendered from `(lat, lon)` centroids. Makes
   geographic error structure immediately readable.
2. **Diurnal error curve** — MAE by hour-of-day, per season. PV
   models are typically worst at sunrise / sunset; wind models
   are worst at the dawn / dusk boundary layer transitions.
3. **Ramp detection ROC** — for wind power: ROC curve on
   ramp-vs-no-ramp classification at multiple thresholds.
4. **Forecast horizon decay** — MAE as a function of forecast
   horizon (T+1h … T+240h). Where does the model add value over
   the NWP raw baseline?
5. **Calibration plot per quantile per horizon** — heatmap of
   coverage error.
6. **Top-N largest residual events** — list the cases where the
   forecast missed worst, with NWP raw + post-processed + actual
   side-by-side. Manual review focus.

## Dashboard tile set — weather-specific

In addition to the standard modelist tiles
([`ygg-modelist#dashboard--app-tile-set`](ygg-modelist.md#dashboard--app-tile-set)):

1. **Live forecast map** — per-zone current forecast for the
   target variable, colour-coded; refreshes per NWP cycle.
2. **Ensemble spread** — the spread between candidate models
   (HRES vs GFS vs blend) as a band around the champion forecast.
   The trader uses this as a "do all models agree?" gut check.
3. **Ramp alert tile** — wind ramp > X MW / h forecast in the next
   N hours, per zone, with confidence.
4. **HDD/CDD anomaly tile** — table of zone-level anomaly z-scores
   for the next 7 days. The gas desk reads this every morning.

## DAG cadence

The weather modelist's cadence is **the NWP cycle**, not the
energy-desk analyst cycle:

```python
job = dbc.jobs.create_or_update(name="modelist_weather_<target>", tasks=[])

# Upstream — the data engineer's NWP-landing task key; confirm.

features  = job.pytask(build_weather_features,  task_key="features",
                       depends_on=["<nwp_curate_task>"]).create()
simulate  = job.pytask(simulate_candidates,     task_key="simulate",
                       depends_on=["features"]).create()
promote   = job.pytask(promote_champion,        task_key="promote",
                       depends_on=["simulate"]).create()
score     = job.pytask(score_predictions,       task_key="score",
                       depends_on=["features", "promote"]).create()
publish   = job.pytask(publish_zone_features,   task_key="publish",
                       depends_on=["score"]).create()
dash      = job.pytask(refresh_dash_ml_weather, task_key="dash",
                       depends_on=["simulate", "promote", "score"]).create()
```

`publish` is the new step the weather modelist owns: rather than
predictions sitting in `ml_<target>_predictions` for the analyst
to read, the **scored, post-processed forecast is published to
the shared weather-features table** the other modelists (power /
gas / oil) read. This makes the weather modelist's output a
first-class **input feature** for downstream models.

Cadence per NWP cycle: features at +30 min after cycle availability,
simulate weekly, score per cycle, publish per cycle, dashboard per
cycle. Use file-arrival triggers off the NWP-landing dataset.

Compute: serverless for everything (no internet egress required —
NWP comes from curated tables). Heavy NHITS / TFT training goes
on a one-shot job cluster.

## Don'ts (weather-specific — also see the shared modelist skill)

- Don't feed raw 10-m wind speed to a power-curve model. Hub-height
  conversion first.
- Don't apply a cubic wind-power law without the cut-in / cut-out /
  rated regions — you'll over-predict at high wind and miss the
  cut-out crash entirely.
- Don't use the population grid for PV / wind weighting. Wrong
  geography; the installed-capacity grid is what matters.
- Don't ignore panel temperature in PV forecasts. Hot afternoons
  in summer routinely de-rate PV output 10–15 % below what
  irradiance alone predicts.
- Don't treat GHI as PV output. Tilt / azimuth / temperature /
  inverter / soiling are real losses.
- Don't compute solar zenith from NWP — it's deterministic from
  `(lat, lon, time)`.
- Don't downscale gridded NWP to a station point and call that
  "the zone forecast". Spatial aggregation over the zone polygon
  is the correct shape; a single point is a single sample, not the
  zone's mean.
- Don't blend NWP runs with equal weights. Models have known
  per-region, per-variable skill; weight by recent observed MAE.
- Don't model precipitation → reservoir inflow without snowpack
  state. Spring melt is the dominant signal in alpine / Nordic
  basins.
- Don't ship a wind forecast without an interval. Wind is
  ramp-dominated; the p10/p90 band is the trader's confidence
  measure.
- Don't invent a NWP curated table name. Same rule as everywhere
  else — placeholder + confirm.
