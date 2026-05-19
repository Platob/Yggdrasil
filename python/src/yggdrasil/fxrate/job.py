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
    "FXRATE_INGESTION_PROVENANCE_COLUMNS",
]


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
    tags: Optional[Mapping[str, str]] = None,
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
            Compute pin (mutually exclusive — pass at most one).
            Per CLAUDE.md "Pick compute by workload type" the FX
            fetch needs outbound internet, so an all-purpose cluster
            (``existing_cluster_id``) is usually the right choice on
            workspaces whose serverless tier lacks internet egress.
            Leave all four ``None`` to let
            :meth:`JobTask.from_callable` pick the default
            (``DEFAULT_ENVIRONMENT_KEY`` serverless env).
        tags: Optional Databricks job tags — useful for cost
            attribution / ownership in the Databricks UI.
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
    # actionable trace.
    compute_pins = [
        ("existing_cluster_id", existing_cluster_id),
        ("job_cluster_key", job_cluster_key),
        ("new_cluster", new_cluster),
        ("environment_key", environment_key),
    ]
    set_pins = [k for k, v in compute_pins if v is not None]
    if len(set_pins) > 1:
        raise ValueError(
            f"deploy_scheduled_fxrate_job: pass at most one compute pin; "
            f"got {set_pins!r}. Pick existing_cluster_id (recommended for "
            f"internet-bound ingestion), job_cluster_key, new_cluster, or "
            f"environment_key."
        )

    if client is None:
        client = DatabricksClient.current()
    service = client.jobs

    cron = _coerce_cron_schedule(
        schedule, timezone=timezone, pause_status=pause_status,
    )
    pairs_json = _pairs_to_json(pairs)
    pair_count = len(json.loads(pairs_json))

    description = (
        f"Scheduled FX ingestion into {target_table!r} on "
        f"{cron.quartz_cron_expression!r} ({cron.timezone_id}). "
        f"Generated by yggdrasil.fxrate.deploy_scheduled_fxrate_job."
    )

    LOGGER.debug(
        "Deploying FX ingestion Job %r target=%r pairs=%d schedule=%r",
        name, target_table, pair_count, cron.quartz_cron_expression,
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
    # explicit pin overrides whichever knob the caller chose.
    if set_pins:
        patches: dict[str, Any] = {
            "existing_cluster_id": existing_cluster_id,
            "job_cluster_key": job_cluster_key,
            "new_cluster": dict(new_cluster) if new_cluster else None,
            "environment_key": environment_key,
        }
        # Cluster-bound pins exclude the serverless ``environment_key``
        # default — strip it so the spec stays internally consistent.
        if any(
            patches[k] is not None
            for k in ("existing_cluster_id", "job_cluster_key", "new_cluster")
        ):
            patches.setdefault("environment_key", None)
            if environment_key is None:
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
    LOGGER.info(
        "Deployed FX ingestion Job %r (job_id=%s) schedule=%r",
        name, getattr(job, "job_id", None), cron.quartz_cron_expression,
    )
    return job
