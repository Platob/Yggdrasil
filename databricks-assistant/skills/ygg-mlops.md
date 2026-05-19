# Skill: MLOps on curated yggdrasil tables — MLflow, autoML, scheduled retraining

## When to use

The user asks to "train a model", "build a forecast", "auto-train a
model on this table", "register the model", "deploy to serving",
"schedule retraining", "track experiments", "predict next-day
prices", "build an anomaly detector", "auto-build models on every
relevant data source". Builds on
[`ygg-curated-views`](ygg-curated-views.md) (clean curated tables
are the only valid model input) and
[`ygg-databricks-job-workflows`](ygg-databricks-job-workflows.md)
(retraining = scheduled `Job.pytask`).

## Heads-up — there is no `yggdrasil.databricks.mlops` *yet*

This skill describes the **pattern and conventions** for MLflow-on-
Databricks workflows that yggdrasil consumers should follow. A
dedicated `yggdrasil.databricks.mlops` service (singleton-by-
config, `client.mlops.train(...)`, `client.mlops.registry`) is on
the roadmap but **not built yet** — the existing
`DatabricksClient.workspace_client().mlflow` SDK surface is the
correct path until it lands.

When the dedicated service ships, the conventions below are the
contract it will implement. Build pipelines that follow them now
so the migration is a rename, not a rewrite.

## Inputs — only curated tables

A model trains on **curated** tables, never `raw_<entity>`. Curated
guarantees UTC timestamps, decimal numerics, ISO codes, stable
column names — drift in any of those silently breaks training.

```
main.<source>.<entity>        ← curated, model input
└── main.<source>.ml_<task>    ← ML artifacts schema (see below)
```

`ml_<task>` is the per-task schema for features, predictions, and
training metadata. Examples: `ml_price_forecast`,
`ml_volatility_predict`, `ml_anomaly_score`.

## Standard ML-artifact tables

Build these three tables per modelled `<task>`:

```python
# 1. Feature snapshots — one row per (entity, observation_utc).
ML_FEATURES_SCHEMA = Schema.from_fields([
    Field("entity_id", DataType.string(), nullable=False,
          tags={"primary_key": True}),
    Field("observation_utc", DataType.timestamp("UTC"), nullable=False,
          tags={"primary_key": True, "partition_by": True}),
    Field("feature_set_version", DataType.string(), nullable=False,
          tags={"primary_key": True},
          comment="Semver of the feature-engineering code. Bumped on schema change."),
    Field("features", DataType.struct([...]),  nullable=False,
          comment="Nested struct of numeric / categorical features. Stable per feature_set_version."),
    Field("label", DataType.decimal(28, 10), nullable=True,
          comment="Forecast horizon target. NULL until horizon elapses."),
    Field("label_horizon_iso", DataType.string(), nullable=False,
          comment="ISO duration ('PT1H', 'P1D'). Documented per task."),
    # …+ provenance.
])

# 2. Predictions — one row per scoring run.
ML_PREDICTIONS_SCHEMA = Schema.from_fields([
    Field("entity_id", DataType.string(), nullable=False,
          tags={"primary_key": True}),
    Field("observation_utc", DataType.timestamp("UTC"), nullable=False,
          tags={"primary_key": True, "partition_by": True}),
    Field("predicted_at_utc", DataType.timestamp("UTC"), nullable=False,
          tags={"primary_key": True}),
    Field("model_uri", DataType.string(), nullable=False,
          comment="MLflow URI: 'models:/<name>/<version>' or 'runs:/<run_id>/model'."),
    Field("prediction", DataType.decimal(28, 10), nullable=False),
    Field("prediction_lower", DataType.decimal(28, 10), nullable=True,
          comment="Lower bound of the predicted interval (e.g. p5)."),
    Field("prediction_upper", DataType.decimal(28, 10), nullable=True),
    Field("confidence", DataType.decimal(5, 4), nullable=True,
          comment="0..1 confidence score. Optional."),
    # …+ provenance.
])

# 3. Training runs — one row per training job.
ML_RUNS_SCHEMA = Schema.from_fields([
    Field("run_id", DataType.string(), nullable=False,
          tags={"primary_key": True},
          comment="MLflow run id."),
    Field("started_at_utc", DataType.timestamp("UTC"), nullable=False,
          tags={"partition_by": True}),
    Field("finished_at_utc", DataType.timestamp("UTC"), nullable=True),
    Field("status", DataType.string(), nullable=False,
          comment="'RUNNING' | 'SUCCEEDED' | 'FAILED'."),
    Field("model_name", DataType.string(), nullable=False),
    Field("model_version", DataType.string(), nullable=True),
    Field("feature_set_version", DataType.string(), nullable=False),
    Field("training_window_iso", DataType.string(), nullable=False,
          comment="ISO duration window ('P30D' last 30 days)."),
    Field("metrics", DataType.struct([...]),  nullable=True,
          comment="Validation metrics struct (rmse, mae, mape, sharpe, …)."),
    # …+ provenance.
])
```

`metrics` and `features` are nested `struct` columns — keep them
stable per `feature_set_version`; a schema change = a version bump.

## Training callable (the unit you stage as a Job task)

```python
def train_price_forecast(
    entity_ids: tuple[str, ...],
    training_window_iso: str = "P30D",
    feature_set_version: str = "v1.0.0",
) -> str:
    """Train one model per `entity_id`, log to MLflow, return the model URI."""
    import datetime as dt
    import mlflow
    from yggdrasil.databricks import DatabricksClient

    dbc = DatabricksClient()
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")        # Unity Catalog model registry

    # 1. Pull features from curated. Always Arrow → polars / pandas via the
    #    zero-copy bridge — no row loops.
    window_seconds = parse_iso_duration_seconds(training_window_iso)
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=window_seconds)
    features = dbc.sql.execute(
        f"""
        SELECT *
        FROM main.<source>.ml_<task>_features
        WHERE feature_set_version = :v
          AND observation_utc >= :cutoff
          AND entity_id IN (:ids)
        """,
        parameters={"v": feature_set_version, "cutoff": cutoff, "ids": entity_ids},
    ).to_polars()

    # 2. Train. Pick the lib your task needs (sklearn, statsforecast,
    #    neuralforecast, prophet, lightgbm, ...) — `[mlops]` extra will
    #    pull mlflow + sklearn when that ships; for now declare via
    #    `JobTask(extra_dependencies=("statsforecast==1.7.4",))`.
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    sf = StatsForecast(models=[AutoARIMA()], freq="H")
    pdf = features.to_pandas()                # required by statsforecast's pandas API
    sf.fit(pdf[["unique_id", "ds", "y"]])

    # 3. Log + register. One MLflow run per `train_price_forecast` call.
    with mlflow.start_run(run_name=f"price_forecast-{feature_set_version}") as run:
        mlflow.log_params({
            "feature_set_version": feature_set_version,
            "training_window_iso": training_window_iso,
            "model_class": "AutoARIMA",
        })
        mlflow.statsforecast.log_model(sf, artifact_path="model",
                                       registered_model_name="main.<source>.price_forecast")
        mlflow.log_metric("validation_mape",
                          compute_validation_mape(sf, pdf))
        model_uri = f"runs:/{run.info.run_id}/model"

    # 4. Record the run in ml_runs alongside the MLflow tracking entry —
    #    so analysts can join on `run_id` from a Delta query without
    #    needing the MLflow API.
    dbc.table("main.<source>.ml_<task>_runs").insert(
        record_training_run(run.info, model_uri),
    )
    return model_uri
```

Each training callable:

- Takes its inputs as **literals** (entity ids, window, version) so
  `JobTask.from_callable` can stage them in the rendered runner.
- Goes through `DatabricksClient` for SQL / table writes — never raw
  `databricks.sdk`.
- Uses **Unity Catalog model registry** (`mlflow.set_registry_uri("databricks-uc")`)
  for the model name (`<catalog>.<schema>.<model_name>`). Workspace-
  scoped MLflow registries are deprecated in 2026.
- Returns the model URI so a downstream task can promote to a stage
  / serving endpoint.

## Auto-create models for relevant data sources

A data source is "relevant" for autoML when it has:

1. A curated table with a UTC timestamp column,
2. At least one numeric column to forecast / classify,
3. ≥ N rows (`N >= 200` for forecast, ≥ 1000 for classification),
4. A documented PK identifying the entity dimension.

Discovery shape (run as a scheduled job):

```python
def discover_ml_candidates() -> list[dict]:
    """Walk curated schemas, return tables that look modellable."""
    dbc = DatabricksClient()
    candidates = []
    for schema in dbc.schemas.list(catalog="main"):
        if schema.name.startswith(("iso", "_meta")):
            continue
        for tbl in dbc.tables.list(catalog="main", schema=schema.name):
            if tbl.name.startswith(("raw_", "ml_")):
                continue
            info = tbl.read_info()
            # Heuristics: needs a UTC ts + a decimal/float column + PK.
            ts_cols = [f for f in info.fields
                       if f.dtype.is_timestamp and "utc" in (f.name or "").lower()]
            num_cols = [f for f in info.fields if f.dtype.is_decimal or f.dtype.is_float]
            pks = info.primary_keys
            if not ts_cols or not num_cols or not pks:
                continue
            candidates.append({
                "table": tbl.full_name(),
                "ts_col": ts_cols[0].name,
                "target_candidates": [f.name for f in num_cols],
                "entity_pk": [f.name for f in pks if f.name not in {ts_cols[0].name}],
            })
    return candidates
```

Then for each candidate, `dbc.jobs.create_or_update(name=f"automl_{table}", …)`
with a `CronSchedule` that retrains nightly / weekly. The
`train_<task>` callable parameterises on the discovered columns.

## MLflow on Databricks — non-obvious notes

- **Tracking URI** = `"databricks"`. Registry URI = `"databricks-uc"`
  (Unity Catalog). Set both at the top of every training callable;
  the AST walker handles the `mlflow` dep automatically.
- **Model name** is a UC three-part identifier when registry URI is
  `databricks-uc`: `"main.<source>.<model_name>"`. Same governance
  surface as tables.
- **Stages → aliases.** UC dropped `Staging` / `Production` stages
  for **aliases** (`@champion`, `@challenger`). Set via
  `client.set_registered_model_alias(name, alias, version)`.
- **Inference at scale.** Serve via Databricks Model Serving
  (REST endpoint) when latency matters; use `mlflow.pyfunc.spark_udf`
  + `predictions = table.with_column(pyfunc(*features))` for batch.
- **Drift detection.** Score the curated `<entity>` table on a
  schedule, log the prediction → `ml_<task>_predictions`, then a
  separate job joins it back to the realised label and computes
  rolling MAPE / Brier / KS. Bind it to a `CronSchedule`; alert via
  `ErrorNotifyingHTTPSession` notifier when the metric crosses a
  threshold.

## Trading / commodity ML tasks — task templates

| Task | Curated input | Target | Default model |
| --- | --- | --- | --- |
| Day-ahead price forecast | `main.entsoe.dayahead` | `price` (per `eic_code`) | `statsforecast.AutoARIMA` or `AutoETS` |
| Intraday volatility | `main.<exchange>.ohlcv_1m` | rolling 15-min realised vol | `lightgbm` on lagged returns |
| Anomaly on order flow | `main.<exchange>.trades` | binary anomaly | `pyod.iforest.IForest` |
| Cross-instrument spread mean-reversion | `main.<exchange>.ohlcv_1m` joined | spread z-score → trade signal | rolling window + threshold |
| Energy load forecast | `main.entsoe.load` | next-24h load | `neuralforecast.NHITS` |

For each: build a `train_<task>(entity_ids, window, version)` callable
matching the template above, schedule via `Job.pytask`, write the run
into `ml_<task>_runs`, register the model under UC.

## Don'ts

- Don't train on `raw_<entity>` tables. Curated only — the standard-
  isation contract is what makes the model reproducible.
- Don't pickle a `DatabricksClient` workspace handle into the model.
  Re-build inside the predict function via singleton-by-config; the
  Spark UDF / serving endpoint will instantiate fresh.
- Don't use the workspace MLflow registry on new code — UC registry
  is the only forward-compatible target in 2026.
- Don't write a custom hyperparameter loop without `mlflow.start_run`;
  every trial deserves a tracked run, even when it loses.
- Don't ship a model without a `ml_<task>_runs` row — that table is
  what makes the rollback story tractable.
- Don't auto-deploy to a `@champion` alias from a training callable.
  Promotion is a separate, reviewed step (manual or governance-gated).
- Don't poll a vendor API for the training input — pull from the
  curated Delta table. The ingestion pipeline (see
  [`ygg-ingestion-pipeline`](ygg-ingestion-pipeline.md)) is the only
  thing that talks to the wire.
- Don't roll a parallel `yggdrasil.mlops` module until the dedicated
  service is on the roadmap — the existing `dbc.workspace_client().mlflow`
  surface plus this skill's conventions is the canonical path until
  then.
