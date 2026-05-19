# Skill: modelist — multi-feature forecast models, candidate runs, KPI dashboards

## When to use

The user is acting as a **modelist / quantitative modeller** — the
person who turns curated features into forecast models, runs many
candidate experiments, tracks model KPIs across runs, and exposes
the best-of-breed model + its run history through dashboards or a
Databricks frontend app. They ask to "train and compare N candidate
models", "score the models", "track model performance over time",
"build a feature store for the forecast task", "compare champion
vs challenger", "expose the model leaderboard to the trader",
"productionise this notebook as scheduled retraining", "build a
Databricks App showing the model's predictions and metrics".

The modelist sits **between** the data engineer (curated tables in)
and the analyst (signals out):

```
<source>.<entity>            ← data engineer (curated)
└── <source>.ml_<task>_*       ← MODELIST owns this layer
    │   ml_<task>_features         (feature store snapshots)
    │   ml_<task>_predictions      (model output rows)
    │   ml_<task>_runs             (one row per training run)
    │   ml_<task>_run_metrics      (per-run KPI matrix — modelist add)
    │   ml_<task>_run_features     (per-run feature contributions)
    ├── dash_ml_<task>_*           (read-only KPI dashboard tables)
    └── apps/ml_<task>_<app>       (Databricks Apps surface)
        └── consumed by analyst → analyst_<task>_signals
```

Builds on [`ygg-mlops`](ygg-mlops.md) (the MLflow + UC registry
contract; this skill extends it for multi-run KPI tracking and
dashboarding), [`ygg-energy-trading-analyst`](ygg-energy-trading-analyst.md)
(the analyst layer consuming model output), and
[`ygg-display-views`](ygg-display-views.md) (the `dash_*` shape for
the frontend).

## What separates the modelist from the analyst and the data engineer

| Role | Owns | Reads from | Writes to |
| --- | --- | --- | --- |
| **Data engineer** | `raw_*` ingestion + `<entity>` curation + shared `iso.*` dims | external sources | `raw_<entity>`, `<entity>`, `dash_<view>` |
| **Modelist** (this skill) | Feature engineering + training + model registry + KPI tracking | curated tables, `iso.*` dims | `ml_<task>_features / _predictions / _runs / _run_metrics`, `dash_ml_<task>_*`, UC registry |
| **Analyst** | Signal generation + position sizing + risk + P&L | `<entity>`, `dash_<view>`, `ml_<task>_predictions`, `dash_ml_<task>_*` | `analyst_<task>_*` |

The modelist doesn't ingest, doesn't propose trades, doesn't size
positions. They produce **calibrated predictions with quantified
uncertainty**, and **published model KPIs** that the analyst and
risk officer can audit before trusting the signal.

## The KPI matrix — `ml_<task>_run_metrics`

The standard `ml_<task>_runs` table in
[`ygg-mlops`](ygg-mlops.md#standard-ml-artifact-tables) carries one
row per training run with a nested `metrics` struct. That's enough
for a single-headline metric, **not** enough for the modelist's
job — which is to compare runs across many KPIs on many slices.

Add a per-run KPI matrix:

```python
from yggdrasil.data import Field, DataType, Schema

ML_RUN_METRICS_SCHEMA = Schema.from_fields([
    Field("run_id", DataType.string(), nullable=False,
          tags={"primary_key": True, "foreign_key": True},
          metadata={"references": "main.<source>.ml_<task>_runs(run_id)"}),
    Field("metric_name", DataType.string(), nullable=False,
          tags={"primary_key": True},
          comment="'rmse' | 'mae' | 'mape' | 'pinball_p10' | 'pinball_p90' | "
                  "'crps' | 'directional_acc' | 'sharpe_proxy' | 'coverage_80' | …"),
    Field("slice_dimension", DataType.string(), nullable=False,
          tags={"primary_key": True},
          comment="'overall' | 'by_entity_id' | 'by_horizon' | 'by_regime' | "
                  "'by_quantile' | 'by_country_iso' | …"),
    Field("slice_value", DataType.string(), nullable=False,
          tags={"primary_key": True},
          comment="The bucket within the dimension. 'overall' for the unsliced row."),
    Field("metric_value", DataType.decimal(28, 10), nullable=False),
    Field("n_samples", DataType.int64(), nullable=False,
          comment="Sample count behind this metric — < 30 = noisy, flag in dash."),
    Field("computed_at_utc", DataType.timestamp("UTC"), nullable=False,
          tags={"partition_by": True}),
    Field("_ingested_at",   DataType.timestamp("UTC"), nullable=False),
    Field("_source",        DataType.string(), nullable=False,
          comment="'modelist:<task>'"),
    Field("_payload_hash",  DataType.string(), nullable=False),
    Field("_batch_id",      DataType.string(), nullable=False),
])
```

Why long format and not a wide `(run_id, rmse, mae, mape, …)`
shape: KPIs evolve (you add `crps`, deprecate `mape`, slice by a new
dimension); adding a column would mean a schema migration and a
backfill. The long shape is the
[`dash_*` display layer](#dashml_task_kpis-the-leaderboard)'s
pivot input, not the analyst's read shape, so the cost is paid
once per dashboard refresh.

### KPI families the modelist persists for every run

| Family | Metrics | Why |
| --- | --- | --- |
| Point error | `rmse`, `mae`, `mape`, `smape` | Standard accuracy. Hold-out + walk-forward. |
| Quantile / interval | `pinball_p10`, `pinball_p50`, `pinball_p90`, `crps`, `coverage_80`, `coverage_95` | Trading needs intervals, not just points. |
| Directional | `directional_acc`, `directional_f1`, `up_recall`, `down_recall` | "Got the sign right" is often more tradable than RMSE. |
| Trading-proxy | `sharpe_proxy`, `info_ratio_proxy`, `pnl_proxy_usd_bbl_unit_notional` | Closes the loop with the analyst layer. |
| Stability | `prediction_drift_psi`, `feature_drift_psi`, `params_change_pct_vs_prev_run` | Telemetry between runs. |
| Cost | `train_seconds`, `predict_ms_per_row`, `model_size_mb` | Operational KPIs the data eng / platform care about. |

Implement each as a small pure-function in
`yggdrasil`-free `ml_<task>_metrics.py` (or library-side
`metrics/` if it generalises). The modelist computes them on
hold-out + walk-forward fold output and writes the long rows.

### Slicing dimensions worth shipping by default

| Slice | When to compute |
| --- | --- |
| `overall` | Always. |
| `by_entity_id` | When the task is multi-entity (one model per zone / hub / grade — or one model with `entity_id` as a feature). The trader needs to know "model is great on FR, terrible on ES". |
| `by_horizon` | When horizons differ (T+1h vs T+24h). The error grows; expose where. |
| `by_regime` | Pre-/post-regime-break (see `regime_start_utc` from the oil desk). Tells the analyst when the model lost track. |
| `by_quantile` | For quantile models — calibration of each predicted quantile separately. |
| `by_country_iso`, `by_eic_code` | For cross-zone risk views. |
| `by_dow`, `by_hod` | Time-of-week / hour-of-day error structure. |

The KPI long-format makes adding a slice a one-call change in the
training callable. Don't pivot at write time.

## Per-run feature contributions — `ml_<task>_run_features`

The trader (and the risk officer) wants to know *why* a model
ranks where it does — the top features and their contributions per
run. SHAP / permutation importance / linear-model coefs all reduce
to the same long shape:

```python
ML_RUN_FEATURES_SCHEMA = Schema.from_fields([
    Field("run_id", DataType.string(), nullable=False,
          tags={"primary_key": True, "foreign_key": True}),
    Field("feature_name", DataType.string(), nullable=False,
          tags={"primary_key": True}),
    Field("importance_method", DataType.string(), nullable=False,
          tags={"primary_key": True},
          comment="'shap_mean_abs' | 'permutation' | 'gain' | 'linear_coef'."),
    Field("importance_value", DataType.decimal(28, 10), nullable=False),
    Field("rank", DataType.int32(), nullable=False,
          comment="1 = most important. Within (run_id, importance_method)."),
    Field("computed_at_utc", DataType.timestamp("UTC"), nullable=False),
    # + provenance
])
```

Two analyst-facing reads off this table:

1. **Top-10 feature panel** on the model's dashboard tile — surfaces
   the model's current "story".
2. **Feature drift over runs** — when `lag_24h_price` ranks #1 for
   30 consecutive runs and then drops to #7, the model is finding
   new structure. Surface as a column on
   `dash_ml_<task>_kpis` (`top_feature_changed_since_last_run` bool).

## Multi-candidate training — the modelist's loop

The modelist doesn't ship one model — they train N candidates per
scheduled cycle, log every one to MLflow, score every one against
the KPI matrix, and let the production-promotion step pick the
champion. This is **deliberately wider than the [`ygg-mlops`](ygg-mlops.md)
single-training-callable pattern**.

```python
def train_candidates(
    task: str,
    candidates: tuple[str, ...] = ("baseline_arima", "lightgbm_quantile",
                                   "nhits", "ensemble_v1"),
    feature_set_version: str = "v3.1.0",
    training_window_iso: str = "P90D",
    folds: int = 5,
) -> list[str]:
    """Train N candidate models on the same feature snapshot.

    Each candidate gets its own MLflow run; KPIs land in
    ml_<task>_run_metrics for cross-run comparison; the function
    returns the list of run_ids. Promotion is a *separate* step.
    """
    import mlflow
    from yggdrasil.databricks import DatabricksClient

    dbc = DatabricksClient()
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    features_df = _load_features(dbc, task, feature_set_version, training_window_iso)
    folds_iter = _walk_forward_folds(features_df, k=folds)

    parent_run_name = f"{task}-candidates-{feature_set_version}"
    with mlflow.start_run(run_name=parent_run_name) as parent:
        run_ids: list[str] = []
        for candidate in candidates:
            with mlflow.start_run(run_name=f"{candidate}-{feature_set_version}",
                                  nested=True) as child:
                trainer = _resolve_trainer(candidate)        # one of the registered model classes
                model, fold_preds, fold_truths = trainer(features_df, folds_iter)

                # Log model artefact + register under UC.
                mlflow.pyfunc.log_model(
                    "model",
                    python_model=model,
                    registered_model_name=f"main.<source>.{task}",
                )

                # Compute the full KPI matrix on the walk-forward
                # output, write to ml_<task>_run_metrics.
                kpi_rows = _kpi_matrix(child.info.run_id, candidate,
                                       fold_preds, fold_truths, features_df)
                dbc.table(f"main.<source>.ml_{task}_run_metrics").insert(kpi_rows)

                # Per-run feature importance (SHAP-mean-abs by default).
                imp_rows = _feature_importance(child.info.run_id, model,
                                               features_df)
                dbc.table(f"main.<source>.ml_{task}_run_features").insert(imp_rows)

                # Write the headline run row.
                run_row = _record_training_run(child.info, candidate, kpi_rows)
                dbc.table(f"main.<source>.ml_{task}_runs").insert([run_row])
                run_ids.append(child.info.run_id)
    return run_ids
```

Pattern notes:

- **Nested MLflow runs.** One parent per `(task, feature_set_version)`
  scheduled cycle; one child per candidate. The MLflow UI surfaces
  the comparison automatically; the modelist's KPI matrix gives
  the analyst the same comparison in queryable Delta form.
- **Walk-forward folds.** k-fold cross-section on time series leaks
  the future into the past — use walk-forward (`yggdrasil`-free
  helper in `_walk_forward_folds`). Each fold's predictions feed
  into the metric computation; the metric values land at the
  candidate-level granularity, not the fold level (unless you also
  add `slice_dimension='fold'`).
- **Registered model name is shared.** All N candidates register
  under `main.<source>.<task>` — each becomes a new *version* of
  that registered model. Promotion (`@champion` / `@challenger`)
  picks a version, not a registry entry. See
  [`ygg-mlops#mlflow-on-databricks--non-obvious-notes`](ygg-mlops.md#mlflow-on-databricks--non-obvious-notes).
- **The training callable returns `run_ids`**, not "the best
  model". A downstream task (`promote_champion`) is what decides
  which version becomes `@champion`. Reading the KPI table for
  the headline metric per run is the standard logic.

## Champion / challenger promotion

A separate Job task — runs after `train_candidates`, reads
`ml_<task>_run_metrics`, picks the champion + a challenger using
a desk-config rule:

```python
def promote_champion(task: str, primary_metric: str = "pinball_p50",
                     direction: str = "minimize") -> dict[str, str]:
    """Read the latest training-cycle KPIs, set @champion + @challenger aliases."""
    from yggdrasil.databricks import DatabricksClient
    import mlflow

    dbc = DatabricksClient()
    df = dbc.sql.execute(
        f"""
        WITH latest_runs AS (
          SELECT r.run_id, r.model_version, r.model_name
          FROM main.<source>.ml_{task}_runs r
          WHERE r.started_at_utc = (
              SELECT max(started_at_utc) FROM main.<source>.ml_{task}_runs
              WHERE feature_set_version = (
                SELECT max(feature_set_version) FROM main.<source>.ml_{task}_runs
              )
          )
        )
        SELECT m.run_id, lr.model_version, m.metric_value
        FROM main.<source>.ml_{task}_run_metrics m
        JOIN latest_runs lr USING (run_id)
        WHERE m.metric_name = :metric
          AND m.slice_dimension = 'overall'
        ORDER BY m.metric_value {'ASC' if direction == 'minimize' else 'DESC'}
        LIMIT 2
        """,
        parameters={"metric": primary_metric},
    ).to_polars()

    if df.height < 2:
        raise RuntimeError(
            f"Need at least 2 candidate runs to set champion + challenger; "
            f"got {df.height}. Check whether train_candidates ran cleanly."
        )

    client = mlflow.MlflowClient()
    champion_version = df["model_version"][0]
    challenger_version = df["model_version"][1]
    name = f"main.<source>.{task}"
    client.set_registered_model_alias(name, "champion",  champion_version)
    client.set_registered_model_alias(name, "challenger", challenger_version)
    return {"champion": champion_version, "challenger": challenger_version}
```

Rules baked into the helper:

- **Two-run minimum.** Don't promote when only one candidate trained
  — likely a training failure.
- **Champion-on-tie keeps the previous version.** Persist
  `previous_champion_version` in `main.<source>.ml_<task>_promotions`
  so analyst dashboards can show "champion changed" badges.
- **Manual override slot.** The promotion table has a `manual_override`
  column the modelist / risk-officer toggles to pin a specific
  version; the scheduled promotion respects it.

## Scoring on a schedule — separating training from prediction

Training runs weekly (or daily for fast-moving regimes). Scoring
runs **every cycle the analyst needs a fresh forecast** —
hourly for intraday power, daily for prompt gas, weekly for
back-of-curve oil. Two distinct Job tasks; the scoring task loads
the `@champion` model URI dynamically:

```python
def score_predictions(task: str, entity_ids: tuple[str, ...]) -> int:
    """Pull champion model, score, write to ml_<task>_predictions."""
    import mlflow
    import datetime as dt
    from yggdrasil.databricks import DatabricksClient

    dbc = DatabricksClient()
    mlflow.set_registry_uri("databricks-uc")
    model_uri = f"models:/main.<source>.{task}@champion"
    model = mlflow.pyfunc.load_model(model_uri)

    features = dbc.sql.execute(
        f"""
        SELECT *
        FROM main.<source>.ml_{task}_features
        WHERE observation_utc = (
          SELECT max(observation_utc) FROM main.<source>.ml_{task}_features
        )
        AND entity_id IN (:ids)
        """,
        parameters={"ids": entity_ids},
    ).to_polars()

    predictions = model.predict(features.to_arrow())  # quantile model returns p10/p50/p90
    rows = _predictions_to_rows(predictions, features, model_uri,
                                predicted_at_utc=dt.datetime.now(dt.timezone.utc))
    dbc.table(f"main.<source>.ml_{task}_predictions").insert(rows)
    return len(rows)
```

Both **champion** and **challenger** score in parallel; predictions
table carries `model_uri` (which alias the row came from) so the
analyst dashboard can show `champion_prediction` vs
`challenger_prediction` side-by-side without re-running the model.

## `dash_ml_<task>_*` — the leaderboard + frontend tables

The modelist's output is consumed via two read shapes:

### `dash_ml_<task>_kpis` — the leaderboard

One row per `(run_id, metric_name, slice_dimension, slice_value)`
won't fit a single dashboard tile. Pivot to the common case at the
display layer:

```sql
CREATE OR REPLACE TABLE main.<source>.dash_ml_<task>_kpis AS
WITH overall AS (
  SELECT
    r.run_id,
    r.model_name,
    r.model_version,
    r.started_at_utc,
    r.candidate_name,                    -- e.g. 'lightgbm_quantile'
    r.feature_set_version,
    MAX(CASE WHEN m.metric_name = 'rmse'             THEN m.metric_value END) AS rmse,
    MAX(CASE WHEN m.metric_name = 'mae'              THEN m.metric_value END) AS mae,
    MAX(CASE WHEN m.metric_name = 'mape'             THEN m.metric_value END) AS mape,
    MAX(CASE WHEN m.metric_name = 'pinball_p50'      THEN m.metric_value END) AS pinball_p50,
    MAX(CASE WHEN m.metric_name = 'coverage_80'      THEN m.metric_value END) AS coverage_80,
    MAX(CASE WHEN m.metric_name = 'directional_acc'  THEN m.metric_value END) AS directional_acc,
    MAX(CASE WHEN m.metric_name = 'sharpe_proxy'     THEN m.metric_value END) AS sharpe_proxy,
    MAX(CASE WHEN m.metric_name = 'train_seconds'    THEN m.metric_value END) AS train_seconds
  FROM main.<source>.ml_<task>_runs r
  LEFT JOIN main.<source>.ml_<task>_run_metrics m
    ON m.run_id = r.run_id AND m.slice_dimension = 'overall' AND m.slice_value = 'overall'
  WHERE r.started_at_utc >= current_date() - INTERVAL 90 DAYS
  GROUP BY r.run_id, r.model_name, r.model_version, r.started_at_utc,
           r.candidate_name, r.feature_set_version
),
promotion AS (
  SELECT model_version, 'champion' AS alias FROM main.<source>.ml_<task>_promotions
  WHERE alias = 'champion' AND active_until_utc IS NULL
  UNION ALL
  SELECT model_version, 'challenger' AS alias FROM main.<source>.ml_<task>_promotions
  WHERE alias = 'challenger' AND active_until_utc IS NULL
)
SELECT o.*, p.alias
FROM overall o
LEFT JOIN promotion p USING (model_version)
```

Pivoted, one row per run, the headline KPIs are tile-ready and the
alias column drives the "currently in production" badge.

### `dash_ml_<task>_predictions_vs_actuals` — calibration over time

```sql
CREATE OR REPLACE TABLE main.<source>.dash_ml_<task>_predictions_vs_actuals AS
SELECT
    p.entity_id,
    p.observation_utc,
    p.predicted_at_utc,
    p.model_uri,
    p.prediction      AS p50,
    p.prediction_lower AS p10,
    p.prediction_upper AS p90,
    e.label           AS actual,
    e.label - p.prediction              AS residual,
    CASE
      WHEN e.label IS NULL THEN NULL
      WHEN e.label BETWEEN p.prediction_lower AND p.prediction_upper THEN 1 ELSE 0
    END AS in_80_band
FROM main.<source>.ml_<task>_predictions p
LEFT JOIN main.<source>.ml_<task>_features e
       ON e.entity_id = p.entity_id
      AND e.observation_utc = p.observation_utc
WHERE p.predicted_at_utc >= current_date() - INTERVAL 90 DAYS
```

The trader's dashboard reads this view to plot prediction bands
+ realised over time — the single most-trusted thing to show before
the analyst's signal.

### `dash_ml_<task>_feature_importance` — what the model is keying on

Pivot `ml_<task>_run_features` to one row per `(run_id, importance_method)`
with array columns for the top-20 features and their values — the
dashboard tile renders the bar chart from a single row read.

## Databricks frontend surface — Apps + AI/BI Dashboard

Two ways the modelist's outputs reach the trader / risk officer
through a Databricks-native frontend:

### Option A — AI/BI Dashboard

Workspace-level dashboard reading the `dash_ml_<task>_*` tables
directly via SQL Warehouse. Best when:

- The trader's read pattern is filter + scroll (date range, model
  version, entity).
- No write-back is needed.
- Embedding into Slack / Teams / a wiki via dashboard share-link
  is the consumption mode.

Tile layout the modelist ships by default:

1. **Leaderboard** — table tile reading `dash_ml_<task>_kpis`,
   sorted by primary metric, with champion / challenger row
   highlighted.
2. **Predictions vs actuals** — line chart from
   `dash_ml_<task>_predictions_vs_actuals`, filtered to the
   champion `model_uri`, last 30 days, with the p10/p90 band.
3. **Coverage telemetry** — rolling 30d % of actuals inside the
   80% band; a single-number tile with a threshold colour band.
4. **Feature importance** — bar chart of top-10 from
   `dash_ml_<task>_feature_importance`, champion run.
5. **Run timeline** — area chart of `rmse` / `pinball_p50` over
   training runs; the champion-change rows are annotated.
6. **Drift telemetry** — `feature_drift_psi`, `prediction_drift_psi`
   rolling 90d.

### Option B — Databricks Apps (Lakehouse Apps)

When the trader needs **interactive controls** — re-score with a
custom feature override, simulate a counterfactual ("what if EU
storage was at -2σ?"), accept / reject a candidate for promotion
— ship a Databricks App. The app is a small Streamlit / Dash /
Flask backend served by Databricks, with the same auth surface
as the workspace.

Patterns:

- **Read-only paths** hit the SQL Warehouse via
  `databricks-sql-connector` on the same `dash_ml_<task>_*` tables
  — same query the AI/BI dashboard would run.
- **Write paths** (promotion override, custom-feature scoring)
  call the modelist's *training / scoring callables* via the Jobs
  API — never re-implement the logic in the app. The app is a UI
  shell over the Job DAG, not a parallel inference path.
- **Auth** — the app runs as the user via OAuth on-behalf-of, so
  RBAC on the tables / registered model carries through. No
  service-principal keys in the app.

A reference app shape:

```python
# apps/ml_<task>_explorer/app.py — Streamlit on Databricks Apps
import streamlit as st
from yggdrasil.databricks import DatabricksClient

dbc = DatabricksClient()                                # workspace-scoped, OBO auth

st.title("Power day-ahead model — explorer")

# Tab 1 — leaderboard
tab_lb, tab_cal, tab_promote = st.tabs(["Leaderboard", "Calibration", "Promote"])

with tab_lb:
    df = dbc.sql.execute(
        "SELECT * FROM main.entsoe.dash_ml_dayahead_kpis ORDER BY started_at_utc DESC LIMIT 50"
    ).to_polars()
    st.dataframe(df.to_pandas())

with tab_cal:
    entity = st.selectbox("Entity", options=_entities())
    df = dbc.sql.execute(
        """
        SELECT observation_utc, p10, p50, p90, actual
        FROM main.entsoe.dash_ml_dayahead_predictions_vs_actuals
        WHERE entity_id = :e
        ORDER BY observation_utc DESC LIMIT 720
        """,
        parameters={"e": entity},
    ).to_polars()
    st.line_chart(df.to_pandas(), x="observation_utc",
                  y=["p10", "p50", "p90", "actual"])

with tab_promote:
    st.write("Override the scheduled champion / challenger promotion.")
    candidate = st.selectbox("Candidate version", options=_candidate_versions())
    alias = st.radio("Set alias", options=["champion", "challenger"])
    if st.button("Promote"):
        dbc.jobs.run_now(
            job_id=_promote_job_id(),
            job_parameters={"version": candidate, "alias": alias},
        )
        st.success(f"Promotion job dispatched for {candidate} → @{alias}")
```

App goes in `apps/ml_<task>_explorer/` next to the modelist's other
artefacts. App deployment is `databricks bundle deploy` — the bundle
spec carries the SQL Warehouse permissions, the app's compute, and
the OAuth scopes.

The modelist owns: the dashboard / app spec, the queries, and the
write-path Jobs they call. The platform team owns the deploy
plumbing.

## Wiring the modelist DAG

Three task groups, one Job per `(source, task)`:

```python
job = dbc.jobs.create_or_update(name=f"modelist_<task>", tasks=[])

# Feature build — runs daily / cycle, builds ml_<task>_features.
features = job.pytask(build_features, task_key="features",
                       depends_on=["curate"]).create()     # curate is the data engineer's task

# Multi-candidate training — runs weekly.
train    = job.pytask(train_candidates, task_key="train",
                       depends_on=["features"]).create()

# Champion / challenger promotion.
promote  = job.pytask(promote_champion, task_key="promote",
                       depends_on=["train"]).create()

# Scoring — runs every analyst cycle. Loads @champion + @challenger.
score    = job.pytask(score_predictions, task_key="score",
                       depends_on=["features", "promote"]).create()

# KPI dashboard refresh.
dash     = job.pytask(refresh_dash_ml,   task_key="dash",
                       depends_on=["train", "promote", "score"]).create()
```

The cadences differ — features daily, train weekly, score per
analyst cycle (hourly to daily depending on desk) — so split the
DAG into **separate Jobs by cadence** rather than one mega-Job
with mixed schedules. Cross-Job dependency via file-arrival
triggers on the latest table partition (see
[`ygg-databricks-job-workflows`](ygg-databricks-job-workflows.md)
for `FileArrivalTriggerConfiguration`).

Compute split (per
[CLAUDE.md "Pick compute by workload type"](../../CLAUDE.md)):

- Feature build, training, scoring, dashboard refresh = serverless
  (`environment_key=DEFAULT_ENVIRONMENT_KEY`).
- Heavy / GPU training (when the candidate is `NHITS`, `TFT`,
  `TemporalFusion`) = sized one-shot job cluster.
- The Databricks App's compute is its own pool — small,
  always-on for latency, separate from the modelling jobs.

## Auditability — the modelist's responsibility

Every prediction the analyst reads is reproducible from:

1. The `run_id` (joins to `ml_<task>_runs`, `_run_metrics`,
   `_run_features`).
2. The `model_uri` on the prediction row.
3. The `feature_set_version` carried on the features it read.
4. The `_payload_hash` on the prediction row, which hashes
   `(model_uri, feature_vector_hash, predicted_at_utc)`.

This is what makes the model auditable for regulators (MiFID II
algorithmic-trading documentation, RTS 6) and for internal risk
post-mortems. Don't drop the columns "to clean up" — they're the
audit primitive.

## Don'ts

- Don't write trade proposals from the modelist layer. The analyst
  owns `analyst_<task>_*`. The modelist hands over
  `ml_<task>_predictions` and the KPI tables; the analyst's signal
  task is what turns a prediction into a position.
- Don't promote champions automatically without a primary-metric +
  guard-metric rule. A model with the best RMSE but coverage_80
  at 50% is a regression — encode the guard in `promote_champion`.
- Don't pivot the KPI matrix at write time. Long-format is what
  lets you add a new slice without touching every downstream query.
- Don't store SHAP for every prediction row. The `_predictions`
  table is hot; SHAP per row blows it up. Persist at the run level
  (`ml_<task>_run_features`) plus an optional sample on prediction
  rows for the calibration dashboard.
- Don't read raw `<entity>` directly from a training callable.
  Materialise into `ml_<task>_features` first — back-tests then
  reproduce; reading from raw / curated mid-training means the
  training data drifts the moment the curated job re-runs.
- Don't bake the dashboard into the notebook the modelist developed
  on. The dashboard is `dash_ml_<task>_*` tables + an AI/BI spec or
  a Databricks App — version-controlled, deployable. Notebooks are
  scratch.
- Don't run the App against the model registry on every page-load.
  The leaderboard / calibration tabs hit `dash_*` tables (already
  refreshed by the Job DAG); only the "score with custom features"
  tab calls the model.
- Don't expose `pinball_p50` as a "trader-friendly" metric without
  also surfacing `coverage_80`. A point metric without an interval
  calibration metric tells the trader half the story.
- Don't run more than ~10 candidates per cycle. Each one costs
  compute + MLflow tracking entries; beyond that the leaderboard is
  noisier than informative. Cull candidates that haven't been on
  the leaderboard top-5 in the last 8 cycles.
