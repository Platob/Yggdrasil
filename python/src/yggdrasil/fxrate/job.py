"""Schedule the FX ingestion as a Databricks Job.

:func:`deploy_scheduled_fxrate_job` is the one-call factory for the
standard pattern: "every <cron>, pull the last N days of FX rates
for these pairs and append to a Delta table". It wires:

* a self-contained entrypoint (:func:`fxrate_ingestion_entrypoint`)
  that runs on the Databricks worker — pulls the FX frame via
  :class:`FxRate`, adds the canonical ``_ingested_at`` / ``_source``
  / ``_source_url`` provenance columns (see CLAUDE.md
  "``raw_<entity>`` carries provenance"), and appends to the target
  Delta table (creating it on first run);
* a :class:`Job` upserted via :meth:`Jobs.create_or_update`, with
  the Quartz cron schedule, optional compute pin
  (``existing_cluster_id`` for ingestion that needs outbound HTTPS
  per CLAUDE.md "Pick compute by workload type"), and a staged
  :class:`JobTask` whose Python script bakes the deploy-time args
  as literals (re-deploy with different pairs / schedule → the
  staged ``.py`` file changes content digest, the job-side script
  pointer updates in one round trip).

Idempotent: re-calling ``deploy_scheduled_fxrate_job(name=...)``
with the same *name* upserts in place. Defaults are
``pause_status="PAUSED"`` so the first deploy never auto-fires
before the operator has reviewed the staged Job in the UI.
"""
from __future__ import annotations

import dataclasses as _dc
import json
import logging
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Union

from .session import PairLike, _coerce_pair

if TYPE_CHECKING:  # pragma: no cover - typing only
    from databricks.sdk.service.jobs import CronSchedule, PauseStatus
    from yggdrasil.databricks.client import DatabricksClient
    from yggdrasil.databricks.jobs import Job


__all__ = [
    "deploy_scheduled_fxrate_job",
    "fxrate_ingestion_entrypoint",
    "dash_fxrate_entrypoint",
    "FXRATE_INGESTION_PROVENANCE_COLUMNS",
    "DEFAULT_DASH_TARGETS",
]


#: Default per-currency standardised targets for the ``dash_fxrate``
#: downstream task — aligned with
#: :data:`yggdrasil.databricks.standardize.DEFAULT_CURRENCY_TARGETS`.
#: Override per deploy by passing ``dash_targets=(...)``.
DEFAULT_DASH_TARGETS: tuple[str, ...] = ("EUR", "USD", "CHF")


LOGGER = logging.getLogger(__name__)


#: Provenance columns the entrypoint appends to every row before
#: insertion. Aligned with CLAUDE.md's standard ``raw_<entity>``
#: shape; downstream curated tables can drop them or carry them
#: through depending on whether they need source attribution.
FXRATE_INGESTION_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "_ingested_at",
    "_source",
    "_source_url",
)


# ---------------------------------------------------------------------------
# Worker-side entrypoint
# ---------------------------------------------------------------------------


def fxrate_ingestion_entrypoint(
    target_table: str,
    pairs_json: str,
    lookback_days: int = 7,
    sampling: str = "1d",
    geo: bool = False,
) -> int:
    """Pull the last *lookback_days* of FX rates and append to *target_table*.

    Runs on the Databricks worker as a single Python task. Imports
    happen inside the function body so :meth:`JobTask.from_callable`'s
    AST walker picks them up as pip dependencies for the staged
    serverless environment.

    Args:
        target_table: Three-part Unity Catalog name
            ``"<catalog>.<schema>.<table>"``. Created on first run
            with the canonical FX schema + provenance columns; on
            subsequent runs the rows are appended.
        pairs_json: JSON-serialised list of ``[source, target]``
            pairs. Pairs ride through ``json.dumps`` rather than as a
            literal ``list`` argument so the staged script's
            ``repr``'d invocation stays a single readable line.
        lookback_days: Window depth fetched each run. The window is
            ``[now - lookback_days, now]``; daily ingestion typically
            uses 7 to absorb upstream republishes of recent rates.
        sampling: Sampling cadence string. Passes through to the
            ``sampling`` column of the FX frame.
        geo: When ``True`` the inserted rows carry geography columns
            (``source_country_iso`` / ``source_lat`` / ``source_lon``
            and the target equivalents).

    Returns:
        The number of rows inserted (useful for the Databricks Job
        UI's task-output panel).
    """
    import datetime as _dt
    import logging as _logging

    import polars as _pl

    from yggdrasil.databricks.client import DatabricksClient
    from yggdrasil.data import DataType
    from yggdrasil.data.schema import Schema, field as _field
    from yggdrasil.fxrate import FxRate

    _logger = _logging.getLogger("yggdrasil.fxrate.job")

    pairs = json.loads(pairs_json)
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(
            f"fxrate_ingestion_entrypoint: pairs_json must decode to a "
            f"non-empty list of [source, target] pairs; got {pairs!r}."
        )

    end = _dt.datetime.now(_dt.timezone.utc)
    start = end - _dt.timedelta(days=int(lookback_days))

    fx = FxRate()
    df = fx.fetch(pairs=pairs, start=start, end=end, sampling=sampling, geo=geo)
    _logger.info(
        "Fetched FX rows=%d window=[%s,%s] target=%r",
        df.height, start.isoformat(), end.isoformat(), target_table,
    )

    now_utc = _dt.datetime.now(_dt.timezone.utc)
    backend_names = ",".join(b.name for b in fx.backends)
    df = df.with_columns([
        _pl.lit(now_utc).cast(_pl.Datetime("us", time_zone="UTC")).alias("_ingested_at"),
        _pl.lit(backend_names).alias("_source"),
        _pl.lit(target_table).alias("_source_url"),
    ])

    parts = target_table.split(".")
    if len(parts) != 3:
        raise ValueError(
            f"target_table must be a three-part 'catalog.schema.table' "
            f"name; got {target_table!r}."
        )
    catalog_name, schema_name, table_name = parts

    client = DatabricksClient.current()
    table = client.tables.table(
        catalog_name=catalog_name,
        schema_name=schema_name,
        table_name=table_name,
    )

    # Build the canonical Delta schema (FX columns + optional geo +
    # provenance). ``ensure_created`` is idempotent — first run
    # creates the table, subsequent runs no-op when the schema
    # matches.
    fields = [
        _field("source",         DataType.from_str("string"),                  nullable=False),
        _field("target",         DataType.from_str("string"),                  nullable=False),
        _field("from_timestamp", DataType.from_str("timestamp[us, tz=UTC]"),   nullable=False),
        _field("to_timestamp",   DataType.from_str("timestamp[us, tz=UTC]"),   nullable=False),
        _field("sampling",       DataType.from_str("string"),                  nullable=False),
        _field("value",          DataType.from_str("float64"),                 nullable=False),
    ]
    if geo:
        fields.extend([
            _field("source_country_iso", DataType.from_str("string"),  nullable=True),
            _field("source_lat",         DataType.from_str("float64"), nullable=True),
            _field("source_lon",         DataType.from_str("float64"), nullable=True),
            _field("target_country_iso", DataType.from_str("string"),  nullable=True),
            _field("target_lat",         DataType.from_str("float64"), nullable=True),
            _field("target_lon",         DataType.from_str("float64"), nullable=True),
        ])
    fields.extend([
        _field("_ingested_at",  DataType.from_str("timestamp[us, tz=UTC]"), nullable=False),
        _field("_source",       DataType.from_str("string"),                nullable=False),
        _field("_source_url",   DataType.from_str("string"),                nullable=False),
    ])
    table.ensure_created(definition=Schema.from_fields(fields))

    table.insert(df, mode="APPEND")
    _logger.info("Inserted %d FX rows into %r", df.height, target_table)
    return df.height


# ---------------------------------------------------------------------------
# Dash worker-side entrypoint
# ---------------------------------------------------------------------------


def dash_fxrate_entrypoint(
    source_table: str,
    target_table: str,
    targets_json: str = '["EUR", "USD", "CHF"]',
    lookback_days: int = 30,
) -> int:
    """Refresh ``dash_fxrate`` (wide-form) from ``raw_fxrate`` (long-form).

    For each ``(source_currency, from_timestamp)`` in the raw FX table
    within the lookback window, emits one row carrying the standardised
    equivalent columns (``value_eur`` / ``value_usd`` / ``value_chf`` by
    default — override via *targets_json*). Same-currency rows fill
    ``1.0`` for their own target column so downstream consumers needn't
    self-join to recover the diagonal.

    Demonstrates the project's "source value + standardised equivalent"
    curated/dash pattern using
    :func:`yggdrasil.databricks.standardize.standardized_column_name`
    for column naming. The pivot uses portable ``MAX(CASE WHEN …)`` SQL
    per CLAUDE.md "Long → wide pivots use portable ``MAX(CASE WHEN …)``".

    Args:
        source_table: Three-part Unity Catalog name of the long-form
            ``raw_fxrate`` table written by
            :func:`fxrate_ingestion_entrypoint`.
        target_table: Three-part Unity Catalog name to refresh.
            Typically ``"<catalog>.<source>.dash_fxrate"``.
        targets_json: JSON-encoded list of target currency ISO 4217
            codes (``["EUR","USD","CHF"]``). Validated through
            :meth:`Currency.parse` so typos fail at deploy time.
        lookback_days: Refresh window depth. Defaults to 30 days —
            enough to absorb late upstream republishes the raw
            ingestion job picks up.

    Returns:
        Row count of the refreshed dash table — surfaces in the
        Databricks task-output panel.
    """
    import datetime as _dt
    import json as _json
    import logging as _logging

    from yggdrasil.databricks.client import DatabricksClient
    from yggdrasil.data import DataType
    from yggdrasil.data.data_field import field as _field
    from yggdrasil.data.enums.currency import Currency
    from yggdrasil.data.schema import Schema
    from yggdrasil.databricks.standardize import standardized_column_name

    _logger = _logging.getLogger("yggdrasil.fxrate.dash")

    targets = [Currency.parse(t) for t in _json.loads(targets_json)]
    if not targets:
        raise ValueError(
            "dash_fxrate_entrypoint: targets_json must decode to a non-empty "
            "list of ISO 4217 currency codes (e.g. '[\"EUR\",\"USD\",\"CHF\"]')."
        )

    for name, label in ((source_table, "source_table"), (target_table, "target_table")):
        if len(name.split(".")) != 3:
            raise ValueError(
                f"{label} must be a three-part 'catalog.schema.table' name; "
                f"got {name!r}."
            )

    catalog_name, schema_name, table_name = target_table.split(".")

    client = DatabricksClient.current()

    # Build the schema first so the table exists before the INSERT
    # OVERWRITE runs. ``ensure_created`` is idempotent — first refresh
    # creates, subsequent refreshes no-op when the schema matches.
    fields = [
        _field("currency",       DataType.from_str("string"),                nullable=False),
        _field("from_timestamp", DataType.from_str("timestamp[us, tz=UTC]"), nullable=False),
    ]
    for tgt in targets:
        col = standardized_column_name("value", tgt)
        fields.append(_field(col, DataType.from_str("float64"), nullable=True))
    fields.append(
        _field("_refreshed_at", DataType.from_str("timestamp[us, tz=UTC]"), nullable=False)
    )
    table = client.tables.table(
        catalog_name=catalog_name, schema_name=schema_name, table_name=table_name,
    )
    table.ensure_created(definition=Schema.from_fields(fields))

    # Portable wide-form pivot via MAX(CASE WHEN …). Currency codes are
    # ISO 4217 alpha-3 (validated by ``Currency.parse``), so the inline
    # interpolation is safe; table names came from the deploy-time call
    # site, same authority as every other SQL in this codebase.
    cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=int(lookback_days))
    cutoff_sql = cutoff.strftime("%Y-%m-%dT%H:%M:%S")
    now_sql = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    select_parts = ["source AS currency", "from_timestamp"]
    for tgt in targets:
        col = standardized_column_name("value", tgt)
        select_parts.append(
            f"CASE WHEN source = '{tgt.code}' "
            f"THEN CAST(1.0 AS DOUBLE) "
            f"ELSE MAX(CASE WHEN target = '{tgt.code}' THEN value END) "
            f"END AS {col}"
        )
    select_parts.append(f"TIMESTAMP'{now_sql}+00:00' AS _refreshed_at")
    select_sql = ",\n           ".join(select_parts)

    sql = (
        f"INSERT OVERWRITE {target_table}\n"
        f"SELECT {select_sql}\n"
        f"FROM {source_table}\n"
        f"WHERE from_timestamp >= TIMESTAMP'{cutoff_sql}+00:00'\n"
        f"GROUP BY source, from_timestamp"
    )
    _logger.info(
        "Refreshing dash_fxrate %r from %r (lookback=%d days, targets=%r)",
        target_table, source_table, int(lookback_days),
        [t.code for t in targets],
    )
    client.sql.execute(sql)

    count_arrow = client.sql.execute(
        f"SELECT COUNT(*) AS n FROM {target_table}"
    ).read_arrow_table()
    n_rows = int(count_arrow.column(0)[0].as_py())
    _logger.info(
        "Refreshed dash_fxrate rows=%d source=%r target=%r",
        n_rows, source_table, target_table,
    )
    return n_rows


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _coerce_cron_schedule(
    schedule: Any,
    *,
    timezone: str,
    pause_status: Any,
) -> "CronSchedule":
    """Accept a Quartz cron string or a pre-built :class:`CronSchedule`.

    Mirrors :func:`yggdrasil.databricks.workflow.flow._coerce_schedule`
    so callers learn one shape across the codebase.
    """
    from databricks.sdk.service.jobs import CronSchedule, PauseStatus

    if isinstance(schedule, CronSchedule):
        return schedule
    if isinstance(schedule, str) and schedule.strip():
        resolved_pause = pause_status
        if isinstance(resolved_pause, str):
            resolved_pause = PauseStatus(resolved_pause.upper())
        return CronSchedule(
            quartz_cron_expression=schedule,
            timezone_id=timezone,
            pause_status=resolved_pause,
        )
    raise TypeError(
        f"deploy_scheduled_fxrate_job(schedule={schedule!r}): pass a Quartz "
        f"cron string (e.g. '0 0 6 * * ?' for daily at 06:00) or a "
        f"databricks.sdk.service.jobs.CronSchedule instance."
    )


def _pairs_to_json(pairs: Iterable[PairLike]) -> str:
    """Normalise *pairs* and JSON-encode for the staged script.

    Coerces every side through :func:`_coerce_pair` (the same routine
    :meth:`FxRate.fetch` uses) so a typo in the call site fails at
    deploy time rather than on the first scheduled run.
    """
    normalised: list[list[str]] = []
    for p in pairs:
        src, tgt = _coerce_pair(p)
        normalised.append([src.code, tgt.code])
    if not normalised:
        raise ValueError(
            "deploy_scheduled_fxrate_job(pairs=[]): pass at least one "
            "(source, target) pair."
        )
    return json.dumps(normalised, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def deploy_scheduled_fxrate_job(
    *,
    target_table: str,
    pairs: Iterable[PairLike],
    schedule: Union[str, "CronSchedule"],
    name: str = "ygg-fxrate-ingestion",
    timezone: str = "UTC",
    pause_status: Union[str, "PauseStatus"] = "PAUSED",
    lookback_days: int = 7,
    sampling: str = "1d",
    geo: bool = False,
    existing_cluster_id: Optional[str] = None,
    job_cluster_key: Optional[str] = None,
    new_cluster: Optional[Mapping[str, Any]] = None,
    environment_key: Optional[str] = None,
    auto_create_cluster: bool = False,
    cluster_name: Optional[str] = None,
    cluster_node_type_id: Optional[str] = None,
    cluster_extra_libraries: Optional[Iterable[str]] = None,
    cluster_spec: Optional[Mapping[str, Any]] = None,
    tags: Optional[Mapping[str, str]] = None,
    dash_table: Optional[str] = None,
    dash_targets: Iterable[str] = DEFAULT_DASH_TARGETS,
    dash_lookback_days: int = 30,
    client: Optional["DatabricksClient"] = None,
) -> "Job":
    """Upsert a scheduled Databricks Job that ingests FX rates.

    The deployed Job runs :func:`fxrate_ingestion_entrypoint` on the
    given cron. First scheduled run creates the target Delta table
    with the canonical FX schema + provenance columns; subsequent
    runs append the latest *lookback_days* window. Re-calling
    ``deploy_scheduled_fxrate_job(..., name=<same>)`` is idempotent
    — the existing Job is upserted in place via
    :meth:`Jobs.create_or_update`.

    Args:
        target_table: Three-part Unity Catalog name
            ``"<catalog>.<schema>.<table>"``. Per CLAUDE.md's
            ``raw_<entity>`` convention the table is typically named
            ``raw_fxrate`` or ``raw_fxrate_<source>``.
        pairs: Iterable of ``(source, target)`` couples — same shape
            :meth:`FxRate.fetch` accepts.
        schedule: Quartz cron expression (e.g. ``"0 0 6 * * ?"`` for
            daily at 06:00) or a pre-built :class:`CronSchedule`.
        name: Deployed Job name. Defaults to
            ``"ygg-fxrate-ingestion"`` so the job lands consistently
            in the workspace UI; pass a distinct value per
            environment when running multiple ingestion jobs.
        timezone: IANA timezone applied to *schedule* when it's a
            Quartz string.
        pause_status: Deployed-state pause flag. Defaults to
            ``"PAUSED"`` — safe shape for an initial deploy. Flip to
            ``"UNPAUSED"`` once a manual ``job.run()`` lands the
            right rows.
        lookback_days / sampling / geo: Forwarded to the entrypoint,
            baked into the staged ``.py`` as ``repr``'d literals.
            Change them by re-deploying.
        existing_cluster_id / job_cluster_key / new_cluster /
        environment_key:
            Explicit compute pin (mutually exclusive — pass at most
            one, and not together with *auto_create_cluster*).
            Per CLAUDE.md "Pick compute by workload type" the FX
            fetch needs outbound internet, so an all-purpose cluster
            (``existing_cluster_id``) is usually the right choice on
            workspaces whose serverless tier lacks internet egress.
            Leave all four ``None`` and *auto_create_cluster* false
            to let the staged task fall back to the serverless
            ``DEFAULT_ENVIRONMENT_KEY`` env.
        auto_create_cluster: When ``True`` resolve (or create) an
            all-purpose cluster via
            :meth:`Clusters.all_purpose_cluster`, pre-installed with
            the latest matching ``ygg[http,data,databricks]`` minor
            release, and pin the resulting cluster id on the staged
            task. Idempotent: the helper reuses an existing
            workspace cluster matching *cluster_name* if one is
            already there.
        cluster_name: When *auto_create_cluster* is on, the cluster
            name to resolve / create. Defaults to a per-user
            ``"All Purpose-<user>"`` name (the
            :meth:`Clusters.all_purpose_cluster` default).
        cluster_node_type_id: Worker / driver instance type for the
            auto-created cluster (e.g. ``"Standard_D4ds_v5"`` on
            Azure, ``"m5d.large"`` on AWS). When ``None`` the
            workspace's default node type is picked by the SDK.
        cluster_extra_libraries: Additional PyPI specs to install on
            the auto-created cluster alongside the default
            ``ygg[http,data,databricks]==<latest-minor>``
            preinstall. Useful for vendor SDKs needed by a custom
            FxRate backend.
        cluster_spec: Free-form passthrough to
            :meth:`Clusters.all_purpose_cluster`'s ``**cluster_spec``
            (``num_workers``, ``autoscale``, ``custom_tags``,
            ``policy_id``, …). Caller wins on key collisions with
            the deploy-function defaults.
        tags: Optional Databricks job tags — useful for cost
            attribution / ownership in the Databricks UI.
        dash_table: Three-part UC name of the curated wide-form
            ``dash_fxrate`` table. When set, a downstream task is
            staged that depends on the ingestion task, refreshing the
            dash view via :func:`dash_fxrate_entrypoint`. When ``None``
            (default) the dash task is omitted — pure raw-only
            ingestion. Typically
            ``"<catalog>.<source>.dash_fxrate"``.
        dash_targets: Iterable of ISO 4217 currency codes that the
            dash view materialises as standardised equivalent columns.
            Defaults to :data:`DEFAULT_DASH_TARGETS` (EUR/USD/CHF —
            same default as
            :data:`yggdrasil.databricks.standardize.DEFAULT_CURRENCY_TARGETS`).
        dash_lookback_days: Lookback window for the dash refresh.
            Default 30 days — wider than the ingestion default so the
            dash carries a stable rolling window even when an upstream
            republish lands.
        client: Workspace :class:`DatabricksClient`. Defaults to
            :meth:`DatabricksClient.current`.

    Returns:
        The deployed :class:`Job`. Trigger an immediate run with
        ``job.run()``; re-deploy with the same call to update.
    """
    from yggdrasil.databricks.client import DatabricksClient
    from yggdrasil.databricks.jobs.task import JobTask

    # Mutually-exclusive compute pins — guard at the boundary so the
    # error fires here, not deep in the Databricks SDK with a less
    # actionable trace. ``auto_create_cluster`` lands as its own
    # pin (it produces an ``existing_cluster_id`` after the helper
    # call), so it counts toward the "at most one" budget.
    compute_pins = [
        ("existing_cluster_id", existing_cluster_id is not None),
        ("job_cluster_key", job_cluster_key is not None),
        ("new_cluster", new_cluster is not None),
        ("environment_key", environment_key is not None),
        ("auto_create_cluster", bool(auto_create_cluster)),
    ]
    set_pin_names = [k for k, v in compute_pins if v]
    if len(set_pin_names) > 1:
        raise ValueError(
            f"deploy_scheduled_fxrate_job: pass at most one compute pin; "
            f"got {set_pin_names!r}. Pick auto_create_cluster=True "
            f"(easiest), existing_cluster_id, job_cluster_key, "
            f"new_cluster, or environment_key."
        )

    if client is None:
        client = DatabricksClient.current()
    service = client.jobs

    cron = _coerce_cron_schedule(
        schedule, timezone=timezone, pause_status=pause_status,
    )
    pairs_json = _pairs_to_json(pairs)
    pair_count = len(json.loads(pairs_json))

    # Resolve auto-create early so a cluster-side failure surfaces
    # before we touch the Jobs API (saves the user a half-deployed
    # job sitting paused in the workspace).
    if auto_create_cluster:
        existing_cluster_id = _resolve_or_create_cluster(
            client=client,
            cluster_name=cluster_name,
            node_type_id=cluster_node_type_id,
            extra_libraries=cluster_extra_libraries,
            cluster_spec=cluster_spec,
        )

    description = (
        f"Scheduled FX ingestion into {target_table!r} on "
        f"{cron.quartz_cron_expression!r} ({cron.timezone_id}). "
        f"Generated by yggdrasil.fxrate.deploy_scheduled_fxrate_job."
    )

    LOGGER.debug(
        "Deploying FX ingestion Job %r target=%r pairs=%d schedule=%r "
        "compute=%r",
        name, target_table, pair_count, cron.quartz_cron_expression,
        set_pin_names[0] if set_pin_names else "serverless-default",
    )

    # Phase 1: upsert the bare Job shell with the schedule, tags, and
    # description. Tasks are added in phase 2 so the staged file path
    # can use the freshly-allocated job's client without an extra
    # round-trip.
    job = service.create_or_update(
        name=name,
        tasks=[],
        tags=dict(tags) if tags else None,
        description=description,
        schedule=cron,
    )

    # Phase 2: stage the entrypoint as a Python task with all knobs
    # baked into the rendered script as ``repr``'d literals. Positional
    # ``target_table`` / ``pairs_json`` plus kw ``lookback_days`` /
    # ``sampling`` / ``geo`` line up with the entrypoint signature.
    task = JobTask.from_callable(
        job,
        fxrate_ingestion_entrypoint,
        target_table,
        pairs_json,
        lookback_days=int(lookback_days),
        sampling=str(sampling),
        geo=bool(geo),
        task_key="fxrate_ingestion",
    )

    # Phase 3: patch the compute pin onto the staged Task spec. The
    # default :meth:`JobTask.from_callable` stamps
    # ``environment_key=DEFAULT_ENVIRONMENT_KEY`` for serverless; an
    # explicit pin (or the auto-resolved cluster_id) overrides it.
    using_cluster_pin = (
        existing_cluster_id is not None
        or job_cluster_key is not None
        or new_cluster is not None
    )
    if using_cluster_pin or environment_key is not None:
        patches: dict[str, Any] = {
            "existing_cluster_id": existing_cluster_id,
            "job_cluster_key": job_cluster_key,
            "new_cluster": dict(new_cluster) if new_cluster else None,
            "environment_key": environment_key,
        }
        # Cluster-bound pins exclude the serverless ``environment_key``
        # default — strip it so the spec stays internally consistent.
        if using_cluster_pin:
            patches["environment_key"] = None
        details = task._details
        if details is None:
            raise RuntimeError(
                "JobTask.from_callable did not produce a Task details "
                "object — cannot apply compute pins."
            )
        task._details = _dc.replace(
            details,
            **{k: v for k, v in patches.items() if v is not None or k == "environment_key"},
        )

    task.create()

    # Phase 4: optional downstream dash_fxrate task. Wired with a
    # ``TaskDependency`` so the dash refresh only runs after the
    # ingestion task succeeds; failed ingestion leaves the dash table
    # untouched (its previous snapshot stays current). Same compute
    # pin shape as the ingestion task — the dash workload is internal
    # (UC SQL only, no outbound HTTP), so serverless is fine even when
    # ingestion is pinned to an all-purpose cluster.
    if dash_table is not None:
        dash_targets_normalised = _normalise_dash_targets(dash_targets)
        dash_targets_json = json.dumps(dash_targets_normalised, separators=(",", ":"))
        dash_task = JobTask.from_callable(
            job,
            dash_fxrate_entrypoint,
            target_table,            # source for the dash task
            dash_table,              # target for the dash task
            targets_json=dash_targets_json,
            lookback_days=int(dash_lookback_days),
            task_key="dash_fxrate",
        )
        # Stamp the depends_on link onto the staged Task spec — the
        # SDK's ``TaskDependency`` carries just the upstream task_key.
        from databricks.sdk.service.jobs import TaskDependency
        details = dash_task._details
        if details is None:
            raise RuntimeError(
                "JobTask.from_callable did not produce a Task details "
                "object for the dash task — cannot wire depends_on."
            )
        dash_task._details = _dc.replace(
            details,
            depends_on=[TaskDependency(task_key=task.task_key)],
        )
        # Mirror the ingestion task's compute pin onto the dash task
        # by default — same workspace constraints, same auth, same
        # cluster (when one is pinned). The dash workload itself
        # would be happy on serverless, but staying on the same
        # compute keeps the deploy story uniform per CLAUDE.md
        # "Pick compute by workload type".
        if using_cluster_pin or environment_key is not None:
            dash_task._details = _dc.replace(
                dash_task._details,
                **{k: v for k, v in patches.items() if v is not None or k == "environment_key"},
            )
        dash_task.create()
        LOGGER.info(
            "Deployed FX dash task %r (job_id=%s) source=%r target=%r targets=%r",
            "dash_fxrate", getattr(job, "job_id", None),
            target_table, dash_table, dash_targets_normalised,
        )

    LOGGER.info(
        "Deployed FX ingestion Job %r (job_id=%s) schedule=%r dash=%r",
        name, getattr(job, "job_id", None), cron.quartz_cron_expression,
        dash_table,
    )
    return job


def _normalise_dash_targets(targets: Iterable[str]) -> list[str]:
    """Normalise *targets* through :meth:`Currency.parse` and dedupe.

    Validates at deploy time so a typo doesn't ride into a scheduled
    run; preserves the caller's order so the dash table's column order
    matches the caller's expectation (``["EUR","USD","CHF"]`` →
    ``value_eur, value_usd, value_chf``).
    """
    from yggdrasil.data.enums.currency import Currency

    seen: set[str] = set()
    out: list[str] = []
    for t in targets:
        code = Currency.parse(t).code
        if code in seen:
            continue
        seen.add(code)
        out.append(code)
    if not out:
        raise ValueError(
            "deploy_scheduled_fxrate_job(dash_targets=()): pass at least one "
            "ISO 4217 currency code (default: EUR/USD/CHF)."
        )
    return out


def _resolve_or_create_cluster(
    *,
    client: "DatabricksClient",
    cluster_name: Optional[str],
    node_type_id: Optional[str],
    extra_libraries: Optional[Iterable[str]],
    cluster_spec: Optional[Mapping[str, Any]],
) -> str:
    """Get-or-create an FX-ingestion cluster, return its ``cluster_id``.

    Routes through :meth:`Clusters.all_purpose_cluster` which already
    handles the idempotent shape (returns the existing cluster
    matching *cluster_name*, creates a fresh one otherwise) and
    pre-installs ``ygg[http,data,databricks]`` pinned to the latest
    minor matching the local ygg install. All-purpose compute is
    what the CLAUDE.md "Pick compute by workload type" rule calls
    for on internet-bound ingestion.
    """
    libraries: list[str] = list(extra_libraries or ())

    spec: dict[str, Any] = {}
    if node_type_id is not None:
        spec["node_type_id"] = node_type_id
    if cluster_spec:
        # Caller wins on collision so they can override the defaults
        # the helper picks (autotermination, num_workers, …).
        spec.update(dict(cluster_spec))

    LOGGER.debug(
        "Resolving FX ingestion cluster name=%r node_type=%r extra_libs=%r",
        cluster_name, node_type_id, libraries,
    )
    cluster = client.compute.clusters.all_purpose_cluster(
        name=cluster_name,
        libraries=libraries or None,
        **spec,
    )
    if cluster.cluster_id is None:
        raise RuntimeError(
            f"all_purpose_cluster returned a Cluster with no cluster_id "
            f"(name={cluster_name!r}). Likely a workspace-side error."
        )
    LOGGER.info(
        "Resolved FX ingestion cluster %r -> cluster_id=%s",
        cluster_name or cluster.cluster_name, cluster.cluster_id,
    )
    return cluster.cluster_id
