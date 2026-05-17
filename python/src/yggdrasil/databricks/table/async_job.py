"""Per-table applier-job settings for staged async inserts.

A staged async insert lands a Parquet payload plus a JSON metadata
file under the target table's ``stg_<table>/.sql/async/insert/``
folder; a downstream Databricks Job drains them back into the
target. :meth:`AsyncInsertJob.settings` returns the full
:class:`JobSettings`-shaped kwargs dict for the per-table applier
job — splat into :meth:`Jobs.get_or_create` /
:meth:`Jobs.create_or_update` / :meth:`Job.deploy`. The matching
:meth:`Table.async_job` helper drives the lifecycle in one call.

Identity is keyed off ``(catalog_name, schema_name, table_name)`` —
one job per target table, watching the table's own staging
``data/`` folder via a file-arrival trigger so newly staged
payloads kick the applier without a cron schedule.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import logging
import re
import time
from dataclasses import replace as _dc_replace
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
)

#: Applier-task flavours :meth:`AsyncInsertJob.settings` will stage —
#: ``"notebook"`` (default) renders cells so each step's logs surface
#: under its own cell in the Databricks UI; ``"spark"`` renders a flat
#: ``.py`` script wired as a :class:`SparkPythonTask` (cheaper to
#: load, single-stream logs).
AsyncApplierTaskType = Literal["notebook", "spark"]

from databricks.sdk.service.jobs import (
    CronSchedule,
    FileArrivalTriggerConfiguration,
    JobParameterDefinition,
    NotebookTask,
    PauseStatus,
    Task,
    TriggerSettings,
)

if TYPE_CHECKING:
    from yggdrasil.databricks.client import DatabricksClient
    from yggdrasil.databricks.fs import VolumePath

    from .async_write import AsyncInsert
    from .table import Table


__all__ = ["AsyncInsertJob", "AsyncApplierTaskType"]

LOGGER = logging.getLogger(__name__)


#: Sanitize an identifier (catalog / schema / table) into the
#: ``[A-Za-z0-9_]`` shape Databricks accepts for ``task_key`` —
#: dots, hyphens, spaces collapse to underscores; everything else
#: outside the alphabet drops. Lower-cased so jobs spanning
#: case-different identifiers still land at the same key.
_TASK_KEY_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_]+")


def _slug_for_task_key(*parts: str) -> str:
    """Join *parts* with ``_`` after sanitising each into ``[A-Za-z0-9_]``."""
    return "_".join(
        _TASK_KEY_SANITIZE_RE.sub("_", part).strip("_").lower()
        for part in parts
    )


class AsyncInsertJob:
    """Namespace for the per-table async-insert applier job spec.

    Not instantiated. :meth:`settings` returns the full kwargs dict
    for the workspace Job; :meth:`load` reads the staged
    :class:`AsyncInsert` records back from the table's staging
    folder.
    """

    #: Prefix for every auto-generated artefact name (job name, task
    #: key, staging folder). Two bracketed tokens land first so the
    #: Databricks UI job list and the workspace tree make
    #: ``yggdrasil``-deployed jobs visually scannable and identify
    #: their flavour at a glance: ``[YGG]`` tags the deployer,
    #: ``[ASYNC]`` tags the role (async-insert applier) — the eye
    #: can group on both without expanding the entry, where the old
    #: ``ygg-async-insert-`` slug-style prefix vanished into the rest
    #: of the name.
    JOB_NAME_PREFIX: ClassVar[str] = "[YGG][ASYNC]"
    # ``stage_async_insert`` writes the Parquet payload under ``data/``
    # **first**, then the JSON metadata under ``logs/``. The file-arrival
    # trigger watches ``logs/`` (not ``data/``) so a fire can only happen
    # after the JSON sibling exists — :meth:`AsyncInsertJob.load` reads
    # the metadata files and would otherwise miss a record whose JSON
    # hasn't landed yet.
    TRIGGER_SUBDIR: ClassVar[str] = "logs"
    LOCK_FILENAME: ClassVar[str] = ".lock"
    DEFAULT_MIN_TIME_BETWEEN_TRIGGERS_SECONDS: ClassVar[int] = 60
    DEFAULT_WAIT_AFTER_LAST_CHANGE_SECONDS: ClassVar[int] = 60
    DEFAULT_LOCK_TIMEOUT_SECONDS: ClassVar[float] = 600.0
    DEFAULT_LOCK_POLL_SECONDS: ClassVar[float] = 2.0

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise TypeError(
            "AsyncInsertJob is a namespace, not a class — call "
            "AsyncInsertJob.settings(table, ...) or table.async_job() "
            "to get a Job."
        )

    # ------------------------------------------------------------------ #
    # Job spec
    # ------------------------------------------------------------------ #
    @staticmethod
    def job_name(table: "Table") -> str:
        """Canonical applier-job name for *table*.

        Format: ``[YGG][ASYNC] Maintain <catalog>.<schema>.<table>``.
        ``[YGG]`` tags the deployer (yggdrasil), ``[ASYNC]`` tags the
        role (async-insert applier), and the verb-first body reads
        as a sentence so an operator scanning the Databricks UI's
        job list can tell what each entry actually does without
        expanding it.
        """
        cat, sch, tbl = AsyncInsertJob._identity(table)
        return f"{AsyncInsertJob.JOB_NAME_PREFIX} Maintain {cat}.{sch}.{tbl}"

    @staticmethod
    def task_key(table: "Table") -> str:
        """Canonical applier-task key for *table*.

        ``task_key`` lives in the Job spec and must match Databricks'
        identifier shape (alphanumeric + ``_-``, no spaces / dots),
        so this is the slug equivalent of :meth:`job_name`:
        ``maintain__<catalog>_<schema>_<table>``. Encoding the table
        triple into the key gives the Databricks UI's per-task
        sub-tree a human-readable label and keeps the staged
        workspace folder
        (``/Workspace/Shared/.ygg/jobs/<task_key>/main-<digest>.py``)
        identifiable to a specific table — which the old
        ``apply_records`` key collapsed across every applier.
        """
        cat, sch, tbl = AsyncInsertJob._identity(table)
        slug = _slug_for_task_key(cat, sch, tbl)
        return f"maintain__{slug}"

    @staticmethod
    def task_description(table: "Table") -> str:
        """Human-readable task description: what this task does, on which table."""
        cat, sch, tbl = AsyncInsertJob._identity(table)
        return (
            f"Maintain {cat}.{sch}.{tbl} — apply staged async-insert records "
            f"into the target table."
        )

    @staticmethod
    def trigger_folder(table: "Table") -> "VolumePath":
        """Staging ``logs/`` folder watched by the file-arrival trigger.

        Watching the metadata-log folder (not the Parquet data folder)
        closes the write-order race: :func:`stage_async_insert` writes
        the Parquet first and the JSON metadata second, so by the time
        a ``logs/<op>.json`` lands the matching Parquet is already on
        disk — the applier won't see a metadata-less Parquet or vice
        versa.
        """
        return table.staging_folder(temporary=False, async_write=True).joinpath(
            AsyncInsertJob.TRIGGER_SUBDIR,
        )

    @staticmethod
    def trigger_url(table: "Table") -> str:
        """File-arrival URL — ``/Volumes/<cat>/<sch>/<vol>/...logs/``."""
        path = AsyncInsertJob.trigger_folder(table).full_path()
        if not path.endswith("/"):
            path = path + "/"
        return path

    @staticmethod
    def settings(
        table: "Table",
        *,
        task: Any = None,
        notebook_path: Optional[str] = None,
        notebook_warehouse_id: Optional[str] = None,
        notebook_base_parameters: Optional[Mapping[str, str]] = None,
        applier: Any = ...,
        task_type: AsyncApplierTaskType = "notebook",
        schedule: Any = None,
        schedule_timezone: str = "UTC",
        schedule_pause_status: Any = None,
        file_arrival_trigger: bool = True,
        min_time_between_triggers_seconds: int = (
            DEFAULT_MIN_TIME_BETWEEN_TRIGGERS_SECONDS
        ),
        wait_after_last_change_seconds: int = (
            DEFAULT_WAIT_AFTER_LAST_CHANGE_SECONDS
        ),
        trigger_pause_status: Any = None,
        parameters: Optional[Mapping[str, str]] = None,
        tags: Optional[Mapping[str, str]] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the full kwargs dict for the per-table applier Job.

        Splat into :meth:`Jobs.get_or_create` / :meth:`Jobs.create_or_update`
        / :meth:`Job.deploy`. Every settable :class:`JobSettings` field
        is resolved off *table* (name, parameters, description) or the
        caller's overrides. Cron strings are coerced to
        :class:`CronSchedule`; file-arrival trigger watches the table's
        own staging ``data/`` folder unless ``file_arrival_trigger=False``.

        Task wiring follows the first of:

        - ``task=...`` — caller-built :class:`Task` / iterable of tasks
          drops in verbatim.
        - ``notebook_path=...`` — wraps a :class:`NotebookTask` carrying
          the table identity as ``base_parameters``.
        - ``applier=<callable>`` (default
          :func:`AsyncInsertJob.apply_records`) — stages the callable
          via :func:`stage_python_notebook_callable` (default) or
          :func:`stage_python_callable`. ``task_type`` picks the
          flavour:

          * ``"notebook"`` (default) — Databricks-format ``.py``
            notebook with cells (imports + metadata, captured locals,
            the ``@checkargs``-wrapped function, and a
            widget-driven invocation) wired as a :class:`NotebookTask`.
            The UI surfaces stdout / ``LOGGER`` lines under the cell
            that produced them — much cleaner for diagnosing an
            applier run.
          * ``"spark"`` — flat ``.py`` script wired as a
            :class:`SparkPythonTask`. Unbound parameters are plumbed
            via ``SparkPythonTask.parameters``
            (``{{job.parameters.<name>}}`` substitution) into
            ``sys.argv`` reads, so the function still gets its
            ``catalog_name`` / ``schema_name`` / ``table_name`` at
            run time. No widget surface, single-stream logs — pick
            this when notebook task overhead isn't wanted.

          The resulting ``environments`` entry lands on the returned
          dict so a direct ``Jobs.create_or_update(**settings)`` call
          is sufficient (no follow-up :meth:`Job.update`). Pass
          ``applier=None`` to opt out — the returned ``tasks=[]`` is
          left empty.
        """
        cat, sch, tbl = AsyncInsertJob._identity(table)

        tasks, environments = AsyncInsertJob._resolve_tasks(
            table=table,
            task=task,
            notebook_path=notebook_path,
            notebook_warehouse_id=notebook_warehouse_id,
            notebook_base_parameters=notebook_base_parameters,
            applier=applier,
            task_type=task_type,
            cat=cat, sch=sch, tbl=tbl,
        )

        out: Dict[str, Any] = {
            "name": AsyncInsertJob.job_name(table),
            "tasks": tasks,
            "schedule": AsyncInsertJob._resolve_schedule(
                schedule=schedule,
                timezone_id=schedule_timezone,
                pause_status=schedule_pause_status,
            ),
            "parameters": AsyncInsertJob._resolve_parameters(
                cat=cat, sch=sch, tbl=tbl, overrides=parameters,
            ),
            "description": (
                description
                or AsyncInsertJob.task_description(table)
            ),
            "tags": dict(tags) if tags else None,
        }
        if environments:
            out["environments"] = environments
        if file_arrival_trigger:
            out["trigger"] = TriggerSettings(
                file_arrival=FileArrivalTriggerConfiguration(
                    url=AsyncInsertJob.trigger_url(table),
                    min_time_between_triggers_seconds=min_time_between_triggers_seconds,
                    wait_after_last_change_seconds=wait_after_last_change_seconds,
                ),
                pause_status=AsyncInsertJob._coerce_pause(trigger_pause_status),
            )
        return out

    # ------------------------------------------------------------------ #
    # Read staged records (called from the applier task body)
    # ------------------------------------------------------------------ #
    @staticmethod
    def load(
        table: "Table",
        *,
        path: "VolumePath | str | None" = None,
        merge: bool = True,
        client: "DatabricksClient | None" = None,
    ) -> List["AsyncInsert"]:
        """Read the staged :class:`AsyncInsert` records under *path*.

        *path* defaults to *table*'s own
        ``stg_<table>/.sql/async/insert`` folder. With ``merge=True``
        (default) returns one merged record per target — every
        overlapping append folds into a single ``INSERT INTO`` and a
        trailing overwrite drops everything before it (see
        :meth:`AsyncInsert.merge`). Pass ``merge=False`` to get the
        raw per-file records back.
        """
        from .async_write import AsyncInsert, _iter_records

        if path is None:
            path = table.staging_folder(temporary=False, async_write=True)
        if client is None:
            client = getattr(table, "client", None)
        if merge:
            return AsyncInsert.merge(path, client=client)
        return list(_iter_records(path, client=client))

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _identity(table: "Table") -> tuple[str, str, str]:
        cat, sch, tbl = table.catalog_name, table.schema_name, table.table_name
        if not cat or not sch or not tbl:
            raise ValueError(
                f"AsyncInsertJob needs a fully-qualified table "
                f"(catalog.schema.table) — got {table!r}."
            )
        return cat, sch, tbl

    @staticmethod
    def _resolve_tasks(
        *,
        table: "Table",
        task: Any,
        notebook_path: Optional[str],
        notebook_warehouse_id: Optional[str],
        notebook_base_parameters: Optional[Mapping[str, str]],
        applier: Any,
        task_type: AsyncApplierTaskType,
        cat: str, sch: str, tbl: str,
    ) -> tuple[List[Task], List[Any]]:
        """Return ``(tasks, environments)`` for the requested wiring.

        ``environments`` is populated only when staging an applier
        callable — the matching :class:`JobEnvironment` carries the
        sniffed pip dependencies so a direct
        ``Jobs.create_or_update(**settings)`` call resolves them
        without a follow-up :meth:`Job.update`.
        """
        if task is not None:
            return ([task] if isinstance(task, Task) else list(task)), []

        if notebook_path:
            base_params: Dict[str, str] = {
                "catalog_name": cat,
                "schema_name": sch,
                "table_name": tbl,
            }
            if notebook_base_parameters:
                base_params.update(
                    {str(k): str(v) for k, v in notebook_base_parameters.items()}
                )
            return [
                Task(
                    task_key=AsyncInsertJob.task_key(table),
                    description=AsyncInsertJob.task_description(table)[:1000],
                    notebook_task=NotebookTask(
                        notebook_path=notebook_path,
                        warehouse_id=notebook_warehouse_id,
                        base_parameters=base_params,
                    ),
                )
            ], []

        if applier is None:
            return [], []

        if applier is ...:
            applier = AsyncInsertJob.apply_records

        return AsyncInsertJob._stage_applier(table, applier, task_type=task_type)

    @staticmethod
    def _stage_applier(
        table: "Table",
        applier: Any,
        *,
        task_type: AsyncApplierTaskType,
        task_key: Optional[str] = None,
        description: Optional[str] = None,
    ) -> tuple[List[Task], List[Any]]:
        """Render *applier*'s source as the requested task flavour.

        Dispatches on ``task_type``:

        * ``"notebook"`` — :func:`stage_python_notebook_callable` —
          Databricks notebook with cells; the UI surfaces stdout /
          ``LOGGER`` lines per cell. Job parameters reach the
          function body through ``dbutils.widgets.get`` reads in the
          invocation cell.
        * ``"spark"`` — :func:`stage_python_callable` — flat
          ``SparkPythonTask`` script; job parameters reach the
          function body via ``SparkPythonTask.parameters``
          (``{{job.parameters.<name>}}`` substitution) plumbed into
          ``sys.argv`` reads in the rendered invocation block.

        Returns ``([task], [job_environment])`` — the matching
        :class:`JobEnvironment` carries the sniffed pip specs so a
        direct ``Jobs.create_or_update(**settings)`` resolves
        ``yggdrasil`` imports without a follow-up
        :meth:`Job.update`.
        """
        from yggdrasil.databricks.jobs.task import (
            DEFAULT_ENVIRONMENT_DEPENDENCIES,
            DEFAULT_ENVIRONMENT_KEY,
            _default_job_environment,
            stage_python_callable,
            stage_python_notebook_callable,
        )

        client = getattr(table, "client", None)
        if client is None:
            raise ValueError(
                f"AsyncInsertJob.settings(applier=...) needs a workspace "
                f"client to stage the callable — got {table!r} with no "
                "``client`` attribute. Pass ``applier=None`` to opt out "
                "or attach a client to the table first."
            )

        if task_type == "notebook":
            stager = stage_python_notebook_callable
        elif task_type == "spark":
            stager = stage_python_callable
        else:
            raise ValueError(
                f"AsyncInsertJob.settings(task_type=...) accepts "
                f"'notebook' or 'spark' — got {task_type!r}."
            )

        # Per-table ``task_key`` lands on the Databricks UI sub-tree
        # *and* on the staged file path
        # (``/Workspace/Shared/.ygg/jobs/<task_key>/main-<digest>.py``),
        # so every table's applier lives in its own workspace folder
        # — the old ``apply_records`` key collapsed every table's
        # staged source onto one path, which then collided on
        # re-upload (notebook ↔ source format) when the staging
        # flavour changed.
        resolved_task_key = task_key or AsyncInsertJob.task_key(table)
        details, extra_deps, _ = stager(
            client, applier, task_key=resolved_task_key,
        )
        # Replace the auto-derived description (function docstring +
        # signature) with the table-aware sentence so the Databricks
        # UI's task description reads as "what does this task do, on
        # which table" — same information density as the job name.
        resolved_description = description or AsyncInsertJob.task_description(table)
        details = _dc_replace(details, description=resolved_description[:1000])
        env_key = getattr(details, "environment_key", None) or DEFAULT_ENVIRONMENT_KEY
        environment = _default_job_environment(
            env_key,
            dependencies=[*DEFAULT_ENVIRONMENT_DEPENDENCIES, *extra_deps],
        )
        return [details], [environment]

    @staticmethod
    def _resolve_schedule(
        *, schedule: Any, timezone_id: str, pause_status: Any,
    ) -> Optional[CronSchedule]:
        if schedule is None or isinstance(schedule, CronSchedule):
            return schedule
        if isinstance(schedule, str):
            return CronSchedule(
                quartz_cron_expression=schedule,
                timezone_id=timezone_id,
                pause_status=AsyncInsertJob._coerce_pause(pause_status),
            )
        raise TypeError(
            f"AsyncInsertJob: ``schedule`` must be a CronSchedule, a "
            f"Quartz cron string, or None — got {type(schedule).__name__}."
        )

    @staticmethod
    def _resolve_parameters(
        *,
        cat: str, sch: str, tbl: str,
        overrides: Optional[Mapping[str, str]],
    ) -> List[JobParameterDefinition]:
        job_params: List[JobParameterDefinition] = [
            JobParameterDefinition(name="catalog_name", default=cat),
            JobParameterDefinition(name="schema_name", default=sch),
            JobParameterDefinition(name="table_name", default=tbl),
        ]
        if overrides:
            existing = {p.name: p for p in job_params}
            for k, v in overrides.items():
                if k in existing:
                    existing[k].default = str(v)
                else:
                    job_params.append(
                        JobParameterDefinition(name=str(k), default=str(v))
                    )
        return job_params

    @staticmethod
    def _coerce_pause(pause: Any) -> Any:
        if isinstance(pause, str):
            return PauseStatus(pause.upper())
        return pause

    # ------------------------------------------------------------------ #
    # Lock — coordinate concurrent applier runs against the same table
    # ------------------------------------------------------------------ #
    @staticmethod
    @contextlib.contextmanager
    def lock(
        table: "Table",
        *,
        path: "VolumePath | str | None" = None,
        timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
        interval: float = DEFAULT_LOCK_POLL_SECONDS,
        client: "DatabricksClient | None" = None,
    ) -> Iterator["VolumePath"]:
        """Exclusive applier lock on *table*'s staging folder.

        Other processes scanning the same folder block on entry
        until any pre-existing :attr:`LOCK_FILENAME` (``.lock``) file
        is removed; then this process claims the lock by creating
        the file and yields its :class:`VolumePath`. On exit
        (success or failure) the lock is removed so the next
        process can proceed.

        ``timeout`` (seconds) caps the wait; ``0`` or negative means
        wait indefinitely. ``interval`` is the polling cadence.
        Stale locks (process crashed mid-apply) are cleared by the
        next caller after ``timeout`` elapses — the wait surfaces a
        :class:`TimeoutError` which the caller can catch and
        :meth:`force_unlock` past.
        """
        from yggdrasil.databricks.path import DatabricksPath

        if path is None:
            path = table.staging_folder(temporary=False, async_write=True)
        elif not hasattr(path, "joinpath"):
            path = DatabricksPath.from_(path, client=client)

        lock_path = path.joinpath(AsyncInsertJob.LOCK_FILENAME)

        # Wait for any existing lock to be released.
        deadline = (time.time() + timeout) if timeout and timeout > 0 else None
        first_wait = True
        while lock_path.exists():
            if first_wait:
                LOGGER.info(
                    "Waiting for applier lock %r to be released "
                    "(timeout=%.0fs interval=%.1fs)",
                    lock_path, timeout, interval,
                )
                first_wait = False
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(
                    f"Timed out after {timeout:.0f}s waiting for applier "
                    f"lock {lock_path!r} to be released."
                )
            time.sleep(interval)

        # Claim the lock. The body marker (acquire timestamp) lands
        # in the file so a stale-lock diagnosis can read when the
        # holder last started without crawling logs.
        now = _dt.datetime.now(_dt.timezone.utc).isoformat()
        LOGGER.info("Claiming applier lock %r (acquired_at=%s)", lock_path, now)
        lock_path.write_bytes(now.encode("utf-8"))
        try:
            yield lock_path
        finally:
            try:
                lock_path.remove(missing_ok=True, wait=False, recursive=False)
                LOGGER.info("Released applier lock %r", lock_path)
            except Exception:  # noqa: BLE001 — best-effort
                LOGGER.exception(
                    "Failed to release applier lock %r; manual cleanup "
                    "may be required.",
                    lock_path,
                )

    @staticmethod
    def force_unlock(
        table: "Table",
        *,
        path: "VolumePath | str | None" = None,
        client: "DatabricksClient | None" = None,
    ) -> None:
        """Drop a stale ``.lock`` file unconditionally."""
        from yggdrasil.databricks.path import DatabricksPath

        if path is None:
            path = table.staging_folder(temporary=False, async_write=True)
        elif not hasattr(path, "joinpath"):
            path = DatabricksPath.from_(path, client=client)
        lock_path = path.joinpath(AsyncInsertJob.LOCK_FILENAME)
        try:
            lock_path.remove(missing_ok=True, wait=False, recursive=False)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to force-unlock %r", lock_path)

    # ------------------------------------------------------------------ #
    # Default applier — staged onto the job by ``Table.async_job``
    # ------------------------------------------------------------------ #
    @staticmethod
    def apply_records(
        catalog_name: str,
        schema_name: str,
        table_name: str,
    ) -> None:
        """Default applier task body.

        Resolves the workspace client, looks up the target table,
        takes an exclusive :meth:`lock` on its staging folder, and
        applies every staged :class:`AsyncInsert` record by handing
        the merged records + engine to :meth:`AsyncInsert.concat`
        (the engine picks the execution path — warehouse, Spark,
        whatever it's wired to). Concurrent applier runs block on
        the lock so the staging folder is drained by at most one
        process at a time.

        Used by :meth:`Table.async_job` as the staged Python task
        when the job doesn't exist yet — ``inspect.getsource`` is
        the runtime contract here, so this body must stay
        self-contained (no module-level closures, no decorators
        beyond ``@staticmethod``).
        """
        from yggdrasil.databricks.client import DatabricksClient
        from yggdrasil.databricks.table.async_job import AsyncInsertJob
        from yggdrasil.databricks.table.async_write import AsyncInsert

        client = DatabricksClient.current()
        engine = client.sql(
            catalog_name=catalog_name, schema_name=schema_name,
        )
        table = engine.table(table_name)

        with AsyncInsertJob.lock(table):
            records = AsyncInsertJob.load(table)
            if not records:
                return
            AsyncInsert.concat(
                records,
                engine=engine,
                client=client,
                wait=True,
                raise_error=True,
            )
