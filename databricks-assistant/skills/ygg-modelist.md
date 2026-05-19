# Skill: modelist — training, simulation, benchmarks, explainability, decision-KPIs

## When to use

The user is acting as a **modelist / quantitative modeller** — the
person who turns curated features into forecast models. They ask
to "train a forecast model", "run many simulation candidates and
compare them", "back-test on a rolling window", "produce
explainability for the latest run", "publish the model's KPIs to a
notebook / dashboard", "show me what the model is currently
keying on", "build a leaderboard of candidate models", "give the
trader the KPIs they need to decide on a position", "expose the
model through a Databricks dashboard or app".

The modelist sits **between** the data engineer (curated tables
in) and the analyst (signals out):

```
<curated entity>               ← data engineer (yggdrasil ingestion + cast)
└── <ml feature store>          ← MODELIST builds this
    ├── candidate training runs  ← many simulation runs, MLflow-tracked
    ├── per-run KPI matrix       ← model + decision KPIs
    ├── per-run explainability   ← SHAP / coefs / partial dependence
    ├── notebook summary         ← rendered post-run, queryable
    └── dashboard / app surface  ← AI/BI tile layout, Databricks App
        └── consumed by analyst → analyst_<task>_signals
```

Builds on:

- [`ygg-mlops`](ygg-mlops.md) — the MLflow + UC registry contract.
  This skill extends it with multi-candidate simulation runs, the
  KPI matrix, explainability persistence, and the notebook /
  dashboard surface.
- [`ygg-ingestion-pipeline`](ygg-ingestion-pipeline.md) /
  [`ygg-cast`](ygg-cast.md) /
  [`ygg-databricks-tables`](ygg-databricks-tables.md) — what
  yggdrasil exposes for moving curated data into the feature
  store cleanly.
- [`ygg-benchmarks`](ygg-benchmarks.md) — for any "I think this
  feature / model is faster" claim. Numbers ship; intuition does
  not.
- [`ygg-energy-trading-analyst`](ygg-energy-trading-analyst.md)
  (and the desk-specific siblings) — the analyst is the
  modelist's primary consumer.

For a desk-specialised variant focused on weather features
(temperature, photovoltaic, wind, rain), route to
[`ygg-modelist-weather`](ygg-modelist-weather.md).

## Don't invent things (mirror of the shared analyst rule)

Same rule as
[`ygg-energy-trading-analyst#dont-invent-things`](ygg-energy-trading-analyst.md#dont-invent-things--ask-or-use-a-placeholder):

- **Don't reference a curated input table by a concrete name** you
  haven't verified exists. Use a placeholder
  (`<curated_dayahead>`, `<weather_features>`) and ask the data
  engineer / the user to confirm.
- **Refined specialised notions are fair game.** Units, ISO codes,
  market conventions, yggdrasil's own surface (`DatabricksClient`,
  `Schema`, `DataField`, `convert`, `mlflow` on UC registry,
  `yggdrasil.fxrate.FxRate`) — all real, all citable.
- **Output schemas the modelist OWN are fair to specify.**
  `ml_<task>_features / _runs / _run_metrics / _run_features /
  _predictions` and the `dash_ml_<task>_*` views — those are the
  modelist's conventions, the point of the skill.
- **Always be able to explain it.** Every feature, metric, formula
  in a notebook or dashboard must trace to either a vendor docs /
  market convention link, a yggdrasil module, or a user-supplied
  spec. If you can't, drop it and ask.

## What separates the modelist from the analyst and the data engineer

| Role | Owns | Reads | Writes |
| --- | --- | --- | --- |
| **Data engineer** | Raw ingestion + curation + shared ISO dims | External sources, vendor APIs | Raw / curated / dash tables |
| **Modelist** (this skill) | Feature engineering + training + simulation + KPI tracking + explainability + UC registry | Curated tables, ISO dims, `yggdrasil.fxrate.FxRate` | `ml_<task>_features / _runs / _run_metrics / _run_features / _predictions`, `dash_ml_<task>_*`, MLflow runs, registered models |
| **Analyst** | Signal generation + position sizing + risk + P&L | Curated, dash, `ml_<task>_predictions`, `dash_ml_<task>_*` | `analyst_<task>_*` |

The modelist doesn't ingest, doesn't propose trades, doesn't size
positions. They produce **many candidate models per cycle**, score
each on a **multi-KPI matrix** including trading-decision KPIs,
and ship **high-quality explanation summaries** through notebooks
and dashboards so the analyst can trust (or reject) the signal.

## Leverage yggdrasil for ingestion + mlops — don't reinvent

Two things the modelist gets from yggdrasil that they'd otherwise
hand-roll:

### Curated → feature store via `yggdrasil.data` + `DatabricksClient`

Don't query raw / curated tables with `databricks.sdk` directly.
Use `DatabricksClient` so SQL execution, schema-aware results,
the singleton-by-config session, and the cast registry all line
up:

```python
from yggdrasil.databricks import DatabricksClient

dbc = DatabricksClient()
features = dbc.sql.execute(
    """
    SELECT *
    FROM <curated_entity>             -- confirm with data engineer
    WHERE observation_utc >= :cutoff
      AND entity_id IN (:ids)
    """,
    parameters={"cutoff": cutoff, "ids": entity_ids},
).to_polars()                          -- zero-copy Arrow → polars
```

For the feature-store write, build the schema with
`yggdrasil.data.Schema.from_fields([...])` and route through
`Table.ensure_created(schema=...)` + `Table.insert` /
`async_insert` — the registry knows about decimal precision, UTC
intent, PK / FK metadata, partition / cluster hints. See
[`ygg-databricks-tables`](ygg-databricks-tables.md).

### MLflow on UC via `ygg-mlops` conventions

`mlflow.set_tracking_uri("databricks")` +
`mlflow.set_registry_uri("databricks-uc")` at the top of every
training callable. Model name is a three-part UC identifier
(`<catalog>.<schema>.<model_name>`). Stages = aliases (`@champion`
/ `@challenger`). See [`ygg-mlops`](ygg-mlops.md) for the contract.

## Many simulation runs — the modelist's loop

The modelist trains N candidate models per scheduled cycle,
logs every one to MLflow as a nested run under a parent
"simulation" run, scores every one against the KPI matrix on a
**walk-forward back-test**, and lets a separate promotion step
pick the champion.

```python
def simulate_candidates(
    task: str,
    candidates: tuple[str, ...] = ("baseline_arima", "lightgbm_quantile",
                                   "nhits", "ensemble_v1"),
    feature_set_version: str = "v3.1.0",
    training_window_iso: str = "P90D",
    backtest_iso: str = "P30D",
    folds: int = 5,
) -> list[str]:
    """Train N candidates on the same feature snapshot, walk-forward
    back-test each, persist KPI matrix + explainability + a
    rendered notebook summary per candidate. Returns list[run_id].
    """
    import mlflow
    from yggdrasil.databricks import DatabricksClient

    dbc = DatabricksClient()
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    features_df = _load_features(dbc, task, feature_set_version, training_window_iso)
    folds_iter  = _walk_forward_folds(features_df, k=folds, holdout_iso=backtest_iso)

    parent_name = f"{task}-simulation-{feature_set_version}"
    run_ids: list[str] = []
    with mlflow.start_run(run_name=parent_name) as parent:
        mlflow.log_params({
            "task": task,
            "feature_set_version": feature_set_version,
            "training_window_iso": training_window_iso,
            "backtest_iso": backtest_iso,
            "n_candidates": len(candidates),
            "n_folds": folds,
        })
        for candidate in candidates:
            with mlflow.start_run(run_name=f"{candidate}-{feature_set_version}",
                                  nested=True) as child:
                trainer = _resolve_trainer(candidate)
                model, fold_preds, fold_truths = trainer(features_df, folds_iter)

                # 1. Log + register under UC.
                mlflow.pyfunc.log_model(
                    "model",
                    python_model=model,
                    registered_model_name=f"<catalog>.<schema>.{task}",
                )

                # 2. KPI matrix — model AND decision KPIs.
                kpi_rows = _kpi_matrix(child.info.run_id, candidate,
                                       fold_preds, fold_truths, features_df)
                dbc.table(f"<catalog>.<schema>.ml_{task}_run_metrics").insert(kpi_rows)

                # 3. Explainability — SHAP / coef / partial dependence.
                imp_rows = _feature_importance(child.info.run_id, model,
                                               features_df)
                dbc.table(f"<catalog>.<schema>.ml_{task}_run_features").insert(imp_rows)

                # 4. Rendered notebook summary for this candidate.
                notebook_path = _render_run_notebook(
                    run_id=child.info.run_id,
                    candidate=candidate,
                    features=features_df,
                    fold_preds=fold_preds,
                    fold_truths=fold_truths,
                )
                mlflow.log_artifact(notebook_path, artifact_path="explanation")

                # 5. Headline run row.
                run_row = _record_training_run(child.info, candidate, kpi_rows)
                dbc.table(f"<catalog>.<schema>.ml_{task}_runs").insert([run_row])
                run_ids.append(child.info.run_id)
    return run_ids
```

Pattern notes:

- **Nested MLflow runs.** One parent per simulation cycle; one
  child per candidate. MLflow UI surfaces the comparison; the KPI
  matrix gives the analyst the same comparison in queryable Delta.
- **Walk-forward folds, not k-fold.** k-fold on time series leaks
  the future into the past. Walk-forward predicts each fold using
  only data preceding it.
- **One registered model name shared across candidates.** All N
  candidates register under `<catalog>.<schema>.<task>` — each
  becomes a *version*. `@champion` / `@challenger` aliases pick a
  version, not a registry entry.
- **The function returns `run_ids`, not "the best model".** A
  downstream promotion task reads the KPI matrix and sets the
  alias. Don't bake promotion into the simulation.
- **Cull cheap candidates first.** Train baseline candidates
  (`AutoARIMA`, `naive_lag_24h`) ahead of expensive ones; if the
  baseline beats `lightgbm_quantile` on the primary KPI, that's
  the story to escalate — don't bury it under more candidates.

### Cadence

| Cadence | Why |
| --- | --- |
| Feature build | Daily (or per data-engineer refresh of curated). |
| Simulation cycle | Weekly default. Daily for fast-moving regimes (intraday power, prompt JKM). |
| Scoring | Per analyst cycle — hourly (intraday) to daily (back-of-curve). |
| Promotion | After every simulation cycle, with a guard rule. |

## The KPI matrix — model KPIs + trading-decision KPIs

The standard `ml_<task>_runs` row in
[`ygg-mlops`](ygg-mlops.md#standard-ml-artifact-tables) carries one
nested `metrics` struct. That's enough for a single-headline
metric, **not enough** to sharpen a trading decision. Add a
long-format matrix:

```python
from yggdrasil.data import Field, DataType, Schema

ML_RUN_METRICS_SCHEMA = Schema.from_fields([
    Field("run_id", DataType.string(), nullable=False,
          tags={"primary_key": True, "foreign_key": True}),
    Field("metric_name", DataType.string(), nullable=False,
          tags={"primary_key": True}),
    Field("slice_dimension", DataType.string(), nullable=False,
          tags={"primary_key": True},
          comment="'overall' | 'by_entity_id' | 'by_horizon' | 'by_regime' | "
                  "'by_quantile' | 'by_country_iso' | 'by_dow' | 'by_hod'."),
    Field("slice_value", DataType.string(), nullable=False,
          tags={"primary_key": True}),
    Field("metric_value", DataType.decimal(28, 10), nullable=False),
    Field("n_samples", DataType.int64(), nullable=False,
          comment="< 30 = noisy, flag in dash."),
    Field("computed_at_utc", DataType.timestamp("UTC"), nullable=False,
          tags={"partition_by": True}),
    # + provenance
])
```

### Two metric families — both required

| Family | Examples | Tells the trader … |
| --- | --- | --- |
| **Model KPIs** | `rmse`, `mae`, `mape`, `pinball_p10/p50/p90`, `crps`, `coverage_80`, `coverage_95` | "How well does the prediction match what happened?" |
| **Decision KPIs** | `directional_acc`, `directional_f1`, `up_recall`, `down_recall`, `signal_calibration`, `pnl_proxy_per_unit`, `sharpe_proxy`, `info_ratio_proxy`, `hit_rate_p80_band`, `regret_per_trade` | "If the analyst had traded the signal, would they have made money / lost money / been calibrated?" |

The decision KPIs are what **sharpen position-sizing decisions** —
they answer the question the trader actually asks ("is this signal
worth listening to?"), not the question the modeller asks ("is the
RMSE down?"). A model with the best RMSE but a `directional_acc`
of 0.49 is useless to the trader; a model with mediocre RMSE but
`directional_acc=0.62` and stable `coverage_80=0.78` is the keeper.

Compute trading-decision KPIs with the same convention the analyst
uses for sizing — the simulation IS a back-test of the analyst's
own conversion of prediction → signal → proposed position. Without
that loop you're measuring the model, not the model-in-the-pipeline.

### Slicing dimensions worth shipping by default

| Slice | When to compute |
| --- | --- |
| `overall` | Always. |
| `by_entity_id` | Multi-entity tasks (zone / hub / grade / refinery). "Great on FR, terrible on ES" is the kind of finding that changes how the analyst weights signals. |
| `by_horizon` | Multi-horizon tasks. Error grows with horizon; expose where. |
| `by_regime` | Pre-/post-regime-break (see the oil desk's `regime_start_utc`). Tells the analyst when the model lost track. |
| `by_quantile` | Quantile models — calibration of each predicted quantile. |
| `by_country_iso`, `by_eic_code` | Cross-zone risk views. |
| `by_dow`, `by_hod` | Time-of-week / hour-of-day error structure. |

Long-format makes adding a slice a one-call change in the training
callable. Don't pivot at write time.

## Per-run explainability — `ml_<task>_run_features`

The trader (and risk officer) wants to know *why* a model ranks
where it does. SHAP / permutation importance / linear-model coefs
/ partial dependence all reduce to one long shape:

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
    Field("rank", DataType.int32(), nullable=False),
    Field("partial_dependence_summary", DataType.string(), nullable=True,
          comment="One-line text — direction + monotonicity + saturation point. "
                  "Full PDP curve goes in the notebook artifact."),
    Field("computed_at_utc", DataType.timestamp("UTC"), nullable=False),
    # + provenance
])
```

Two analyst-facing reads off this table:

1. **Top-10 feature panel** on the model's dashboard — the model's
   current "story" in one tile.
2. **Feature drift over runs** — when a previously-#1 feature drops
   in rank for 8 consecutive runs, the model is finding new
   structure. Flag on the leaderboard.

## High-quality explanation summary per run

The modelist ships, per simulation cycle, **two surfaces**: a
rendered notebook artifact (per candidate, logged to MLflow) and a
dashboard / app tile set (refreshed on the same DAG).

### Per-run notebook artifact

Render a notebook *per candidate* immediately after training and
log it as an MLflow artifact under `explanation/`. The notebook is
what makes the model auditable to a non-modelist reader.

Mandatory sections in the rendered notebook:

1. **Run identity**: `run_id`, `candidate_name`, `feature_set_version`,
   `training_window`, `backtest_window`, MLflow run URL.
2. **Headline KPIs**: the row from `ml_<task>_run_metrics` filtered
   to `slice_dimension='overall'`, rendered as a small table —
   model KPIs + decision KPIs side by side.
3. **Predictions vs realised** plot, walk-forward back-test —
   line chart with the p10 / p50 / p90 band and the realised
   series overlaid; one chart per `entity_id` (sampled if more
   than 6 entities).
4. **Calibration plot** — predicted quantile vs realised exceedance
   frequency. Shows whether the p80 band is honest.
5. **Residual diagnostics** — residual histogram + Q-Q vs Normal,
   residual auto-correlation, residual vs each top feature
   (sanity check: residual should look like noise, not show a
   trend in a feature).
6. **Feature importance** — SHAP mean-abs bar chart top-20.
7. **Partial dependence plots** for the top 5 features. Direction +
   shape + saturation are what the trader reads.
8. **Slice breakdown** — KPI bars grouped by `by_entity_id` and
   `by_horizon` (the two most-asked slices).
9. **Regime sensitivity** — if the data spans a regime break, show
   the KPIs computed pre- vs post-`regime_start_utc`.
10. **What changed since last run** — single-row diff of headline
    KPIs and top-3 feature ranks vs the previous champion. Makes
    "should we promote?" obvious.

The notebook is the modelist's `rationale` artifact — same role
that the `rationale` column plays on an analyst signal. It is
**not** an interactive exploration notebook; it's a generated
post-run report, version-controlled in its template form, rendered
fresh per run. Use `nbformat` / `papermill` / `nbconvert`, or
build the cells via `DatabricksClient` workspace-file writes; the
generation is automatable.

### Dashboard / app tile set

Two reach paths to the trader / risk officer; pick by interactivity:

| Surface | When |
| --- | --- |
| **AI/BI Dashboard** (workspace dashboard reading `dash_ml_<task>_*` over SQL Warehouse) | Filter-and-scroll consumption; embeddable share link; no write-back needed. |
| **Databricks App** (Streamlit / Dash served by Lakehouse Apps) | Interactive controls — re-score with custom features, counterfactual simulation, accept/reject a candidate for promotion. |

Tile layout the modelist ships by default (same on both surfaces):

1. **Leaderboard** — table reading `dash_ml_<task>_kpis`, sorted by
   primary KPI, champion / challenger highlighted.
2. **Predictions vs actuals** — line chart from
   `dash_ml_<task>_predictions_vs_actuals`, filtered to champion
   `model_uri`, last 30 days, with p10/p90 band.
3. **Coverage telemetry** — rolling 30d % of actuals inside the
   80 % band; single-number tile + threshold colour band.
4. **Feature importance** — bar chart top-10 of champion run.
5. **Run timeline** — KPI over training runs; champion-change rows
   annotated.
6. **Drift telemetry** — `feature_drift_psi`, `prediction_drift_psi`
   rolling 90d.
7. **Decision-KPI tile** — `directional_acc`, `pnl_proxy_per_unit`,
   `hit_rate_p80_band` for the champion — the **trader's three
   numbers** for "do I listen to this signal?".

### Databricks App write-paths

When the trader needs interactivity (override promotion,
counterfactual simulation), build a Databricks App. Rules:

- **Read paths** hit the SQL Warehouse via
  `databricks-sql-connector` (or `DatabricksClient`) on the same
  `dash_ml_<task>_*` tables the dashboard reads. Don't re-implement.
- **Write paths** (override champion, counterfactual scoring) call
  the modelist's *training / scoring callables via the Jobs API* —
  never re-implement the logic in the app. The app is a UI shell
  over the Job DAG.
- **Auth** — OAuth on-behalf-of, so RBAC on the tables / registered
  model carries through. No service-principal keys in the app.

## Trading-decision KPIs — make them computable

Each decision KPI requires the modelist to simulate the analyst's
sizing rule, not just score the prediction. Standard shapes:

| KPI | Formula | What it tells the trader |
| --- | --- | --- |
| `directional_acc` | `mean(sign(prediction - last_observed) == sign(realised - last_observed))` | Sign-correct rate. |
| `pnl_proxy_per_unit` | `mean(sign(prediction - last_observed) * (realised - last_observed))` | Per-unit P&L if you'd traded the model 1:1. |
| `pnl_proxy_per_conviction` | weighted by `conviction` | Same, but weights the bigger calls. |
| `hit_rate_p80_band` | `mean(realised in [p10, p90])` | Calibration of the interval the analyst quotes. |
| `regret_per_trade` | `mean(realised_pnl - oracle_pnl)` | Distance to perfect-foresight floor. |
| `info_ratio_proxy` | `mean(signed_pnl) / stddev(signed_pnl)` | Risk-adjusted version of `pnl_proxy_per_unit`. |
| `sharpe_proxy` | annualised version of `info_ratio_proxy` | Same, annualised. |
| `signal_calibration_brier` | Brier score on `direction_predicted` vs `direction_realised` | How well-calibrated the up/down probability is. |

These are computed on the walk-forward back-test output during
simulation. Persist with `slice_dimension='overall'` and the
relevant slices.

The trader's dashboard headline tile typically pins three:
`directional_acc`, `pnl_proxy_per_unit`, `hit_rate_p80_band`.

## Champion / challenger promotion with a guard metric

A separate Job task — runs after simulation. Reads
`ml_<task>_run_metrics` for the latest simulation cycle, picks
champion + challenger using a desk-config primary + guard rule:

```python
def promote_champion(
    task: str,
    primary_metric: str = "directional_acc",
    primary_direction: str = "maximize",
    guard_metric: str = "coverage_80",
    guard_min: float = 0.75,
) -> dict[str, str]:
    """Pick the candidate that wins `primary_metric` AND clears
    `guard_metric >= guard_min`. Refuses to promote if no candidate
    clears both — the previous champion stays.
    """
    # … reads ml_<task>_run_metrics, applies the rule, calls
    # mlflow.MlflowClient.set_registered_model_alias(name, 'champion', version).
```

Rules baked in:

- **Two-candidate minimum.** Don't promote when only one candidate
  trained — likely a training failure.
- **Guard never overruled by primary.** A 0.65 `directional_acc`
  with `coverage_80 = 0.50` is a worse model, not a better one —
  the band is dishonest.
- **Manual override slot.** A `manual_override` table the modelist
  / risk officer toggles to pin a specific version; the scheduled
  promotion respects it.
- **Champion-on-tie keeps the previous version.** Persist
  `previous_champion_version` so the dashboard can render a
  "champion changed" badge.

## Wiring the modelist DAG

Three task groups, one Job per `(source, task)`:

```python
job = dbc.jobs.create_or_update(name=f"modelist_<task>", tasks=[])

# Upstream — the data engineer's `<curate_task>` task key; confirm.

features = job.pytask(build_features,        task_key="features",
                       depends_on=["<curate_task>"]).create()
simulate = job.pytask(simulate_candidates,   task_key="simulate",
                       depends_on=["features"]).create()
promote  = job.pytask(promote_champion,      task_key="promote",
                       depends_on=["simulate"]).create()
score    = job.pytask(score_predictions,     task_key="score",
                       depends_on=["features", "promote"]).create()
dash     = job.pytask(refresh_dash_ml,       task_key="dash",
                       depends_on=["simulate", "promote", "score"]).create()
```

Cadence rule: features + score run on the analyst's cadence;
simulate + promote run weekly (or per-regime-break). Use a
**separate Job per cadence** instead of one mega-Job with mixed
schedules; cross-Job dependency via file-arrival triggers (see
[`ygg-databricks-job-workflows`](ygg-databricks-job-workflows.md)).

Compute split: all five tasks are UC-Delta workloads (no public
internet egress required) → **serverless**
(`environment_key=DEFAULT_ENVIRONMENT_KEY`). Heavy / GPU training
(when the candidate is `NHITS`, `TFT`, `TemporalFusion`) goes on a
sized one-shot job cluster. The Databricks App's compute is its
own small always-on pool.

## Auditability — the modelist's responsibility

Every prediction the analyst reads is reproducible from:

1. The `run_id` (joins to `ml_<task>_runs`, `_run_metrics`,
   `_run_features`).
2. The `model_uri` on the prediction row.
3. The `feature_set_version` carried on the features it read.
4. The `_payload_hash` on the prediction row, hashing
   `(model_uri, feature_vector_hash, predicted_at_utc)`.
5. The per-run notebook artifact under MLflow's `explanation/`.

This is what makes the model auditable for regulators (MiFID II
RTS 6, algorithmic-trading documentation) and for internal risk
post-mortems. Don't drop the columns "to clean up".

## Don'ts

- Don't write trade proposals from the modelist layer. The analyst
  owns `analyst_<task>_*`. Modelist hands over `ml_<task>_predictions`
  and the KPI tables; analyst's signal task turns them into a
  position.
- Don't promote champions automatically without a primary + guard
  rule.
- Don't pivot the KPI matrix at write time. Long-format lets you
  add slices without touching downstream queries.
- Don't store SHAP for every prediction row. Persist at run level
  (`ml_<task>_run_features`) plus an optional sample on prediction
  rows for the calibration dashboard.
- Don't read raw / curated directly from a training callable.
  Materialise into `ml_<task>_features` first — back-tests then
  reproduce.
- Don't bake the dashboard into the notebook the modelist developed
  on. The dashboard is `dash_ml_<task>_*` tables + an AI/BI spec
  or a Databricks App, version-controlled.
- Don't run the App against the model registry on every page-load.
  Read paths hit `dash_*` tables (refreshed by the DAG); only
  interactive write-paths call the model.
- Don't expose point metrics without coverage / calibration. A
  trader using `pinball_p50` without `coverage_80` sees half the
  story.
- Don't ship more than ~10 candidates per cycle. Cull candidates
  that haven't been on the leaderboard top-5 in the last 8 cycles.
- Don't reuse k-fold cross-validation on time series. Walk-forward,
  always.
- Don't claim a model "is faster" without a [bench](ygg-benchmarks.md).
- Don't invent a curated table name to make example code run. See
  [§ Don't invent things](#dont-invent-things-mirror-of-the-shared-analyst-rule).
