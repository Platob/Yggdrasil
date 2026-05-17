"""``@flow`` decorator and :class:`Flow` class.

A :class:`Flow` wraps a Python function whose body *describes* a DAG
of :class:`WorkflowTask` calls — Prefect-style. The flow function
itself runs locally without any Databricks involvement; calling
``flow.deploy(...)`` traces the body, materialises each task call
as a Databricks :class:`Task`, and upserts the whole bundle as one
:class:`Job` via :meth:`Jobs.create_or_update`. ``flow.run(...)``
triggers the resulting Job and returns the :class:`JobRun`.

Authoring shape::

    from yggdrasil.databricks.workflow import flow, task, secret

    @task
    def extract(date: str) -> str:
        return f"/Volumes/raw/{date}"

    @task(retries=2)
    def load(path: str, api_key: str = secret("vendor", "api-key")) -> str:
        return f"loaded {path}"

    @flow(name="daily-etl", schedule="0 2 * * *")
    def daily_etl(date: str = "2025-01-01") -> None:
        p = extract(date)
        load(p)

    daily_etl.deploy()                 # upsert the Databricks Job
    run = daily_etl.run(date="2025-01-15", wait=True)

The flow body is normal Python. Calling ``daily_etl(date="…")``
directly runs the tasks locally (no workspace round-trip), which is
what makes test-driving the pipeline cheap. Calling
``daily_etl.deploy()`` opens a :class:`TraceContext`, runs the body
once with :class:`FlowParam` placeholders threaded in for the flow's
declared parameters, and stages each captured :class:`TaskNode` as a
Databricks task.
"""
from __future__ import annotations

import functools
import inspect
import logging
from typing import (
    Any, Callable, Dict, Iterable, List, Mapping, Optional,
    TYPE_CHECKING, Tuple,
)

from databricks.sdk.service.jobs import (
    CronSchedule,
    JobParameterDefinition,
    PauseStatus,
    Task,
)

from .context import TraceContext
from .nodes import FlowParam, TaskNode

if TYPE_CHECKING:
    from yggdrasil.databricks.client import DatabricksClient
    from yggdrasil.databricks.jobs import Job, JobRun, Jobs


__all__ = ["Flow", "flow"]

LOGGER = logging.getLogger(__name__)


def _resolve_jobs_service(
    service: Optional["Jobs"],
    client: Optional["DatabricksClient"],
) -> "Jobs":
    """Pick a :class:`Jobs` service from caller hints, falling back to the current client."""
    if service is not None:
        return service
    if client is None:
        from yggdrasil.databricks.client import DatabricksClient
        client = DatabricksClient.current()
    return client.jobs


def _coerce_schedule(
    schedule: Any,
    *,
    timezone: str,
    pause_status: Any,
) -> Optional[CronSchedule]:
    """Accept a :class:`CronSchedule`, a Quartz cron string, or ``None``."""
    if schedule is None or isinstance(schedule, CronSchedule):
        return schedule
    if isinstance(schedule, str):
        if isinstance(pause_status, str):
            pause_status = PauseStatus(pause_status.upper())
        return CronSchedule(
            quartz_cron_expression=schedule,
            timezone_id=timezone,
            pause_status=pause_status,
        )
    raise TypeError(
        f"flow(schedule={schedule!r}): expected a Quartz cron string, a "
        "CronSchedule instance, or None."
    )


class Flow:
    """A workflow flow — a function whose body describes a Databricks DAG.

    Instances are typically built via the :func:`flow` decorator; the
    class is exposed for the rare caller that wants to subclass or
    introspect the metadata directly.

    Parameters
    ----------
    func
        The flow function. Its parameters become :class:`JobParameterDefinition`
        entries on the deployed job (defaults preserved when present),
        and its body must consist of :class:`WorkflowTask` calls plus
        plain Python composing them. No I/O at flow-body time —
        the body runs once at deploy time to capture the DAG.
    name
        Job name. Defaults to ``func.__name__``.
    schedule
        Quartz cron string or :class:`CronSchedule`. Wired into
        :class:`JobSettings.schedule`.
    timezone
        Time zone for cron coercion. Ignored when ``schedule`` is
        already a :class:`CronSchedule`.
    pause_status
        ``"UNPAUSED"`` / ``"PAUSED"`` (or the matching :class:`PauseStatus`).
        Default ``None`` → Databricks default (unpaused).
    parameters
        Extra job parameters not derived from the flow signature.
        Merge into the auto-derived list; caller wins on collision.
    tags / permissions / job_settings
        Forwarded to :meth:`Jobs.create_or_update`. ``tags`` merge
        under :meth:`Jobs.default_tags`; ``permissions`` accepts the
        same string / :class:`JobAccessControlRequest` shapes the
        underlying service does.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: Optional[str] = None,
        schedule: Any = None,
        timezone: str = "UTC",
        pause_status: Any = None,
        parameters: Optional[Mapping[str, Any]] = None,
        tags: Optional[Mapping[str, str]] = None,
        permissions: Optional[Iterable[Any]] = None,
        **job_settings: Any,
    ) -> None:
        self.func = func
        self.__wrapped__ = func
        self.name = name or func.__name__
        self.schedule = schedule
        self.timezone = timezone
        self.pause_status = pause_status
        self.parameters = dict(parameters) if parameters else {}
        self.tags = dict(tags) if tags else None
        self.permissions = list(permissions) if permissions else None
        self.job_settings = dict(job_settings)
        self._signature = inspect.signature(func)
        functools.update_wrapper(self, func)

    # ------------------------------------------------------------------ #
    # Run mode (no trace) — the flow function behaves as plain Python
    # ------------------------------------------------------------------ #
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the flow body locally — every ``@task`` call runs in-process.

        This is the unit-test path: ``daily_etl(date="2025-01-15")``
        executes the wrapped Python like any function. Use
        :meth:`deploy` / :meth:`run` to involve a workspace.
        """
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:
        return f"Flow(name={self.name!r}, func={self.func.__qualname__!r})"

    # ------------------------------------------------------------------ #
    # Trace mode — record the DAG without executing anything
    # ------------------------------------------------------------------ #
    def trace(self, **overrides: Any) -> List[TaskNode]:
        """Run the flow body in trace mode and return captured nodes.

        Each declared flow parameter that isn't pinned in *overrides*
        is threaded into the body as a :class:`FlowParam` sentinel,
        so a task call ``step(date)`` records the binding as "task
        ``step``'s ``date`` arg is flow-parameter ``date``" instead
        of pinning the default literal at deploy time.

        Override entries that *are* concrete values get pinned (their
        literal :func:`repr` lands on the staged invocation); pass an
        explicit :class:`FlowParam` to override the default while
        keeping the binding lazy.
        """
        param_values = self._build_trace_params(overrides)
        with TraceContext() as ctx:
            self.func(**param_values)
        return list(ctx)

    def _build_trace_params(self, overrides: Mapping[str, Any]) -> Dict[str, Any]:
        """Resolve flow parameters into trace-time values.

        Overrides win; declared defaults fill in the remainder. Each
        unfilled parameter (and each unspecified default) becomes a
        :class:`FlowParam` so the DAG records it as a job-level
        binding rather than a stage-time literal.
        """
        resolved: Dict[str, Any] = {}
        for name, param in self._signature.parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            if name in overrides:
                resolved[name] = overrides[name]
                continue
            default = (
                None if param.default is inspect.Parameter.empty else param.default
            )
            resolved[name] = FlowParam(name=name, default=default)
        # Extra entries the caller passed that aren't declared on the
        # flow signature — surface them so a typo doesn't silently
        # vanish into the trace.
        unknown = set(overrides) - set(self._signature.parameters)
        if unknown:
            raise TypeError(
                f"Flow.trace: unknown parameter(s) {sorted(unknown)!r}; "
                f"flow {self.name!r} declares "
                f"{list(self._signature.parameters)!r}."
            )
        return resolved

    # ------------------------------------------------------------------ #
    # Deploy — stage every node and upsert the Job
    # ------------------------------------------------------------------ #
    def deploy(
        self,
        *,
        service: Optional["Jobs"] = None,
        client: Optional["DatabricksClient"] = None,
        userinfo_defaults: bool = False,
        trace_overrides: Optional[Mapping[str, Any]] = None,
        **extra_job_settings: Any,
    ) -> "Job":
        """Trace the flow, stage every task, and upsert the Databricks Job.

        Returns the :class:`Job` handle (singleton-cached, refreshed),
        ready for ``.run()`` / ``.refresh()`` / ``.delete()`` use.

        *service* / *client* control the workspace target — defaults
        to :meth:`DatabricksClient.current`'s ``.jobs``.

        *userinfo_defaults* mirrors :meth:`Jobs.create_for_user`'s flag:
        when truthy, the job pre-fills ``git_source`` /
        ``email_notifications`` / ``tags`` from :class:`UserInfo` (the
        caller's identity / repo). Caller-supplied values still win on
        collision.

        *trace_overrides* (rarely used) pins specific flow parameters
        to concrete values at trace time. The common case is to leave
        every parameter as a :class:`FlowParam`, so the deployed Job's
        ``parameters`` carry the defaults and each run can override
        them.
        """
        jobs = _resolve_jobs_service(service, client)
        nodes = self.trace(**(trace_overrides or {}))
        if not nodes:
            raise RuntimeError(
                f"Flow.deploy({self.name!r}): the flow body produced no tasks. "
                "A flow must call at least one @task-decorated callable."
            )

        tasks, environments = self._stage_nodes(jobs.client, nodes)
        job_parameters = self._build_job_parameters(nodes, trace_overrides or {})

        settings: Dict[str, Any] = {
            "name": self.name,
            "tasks": tasks,
            **self.job_settings,
            **extra_job_settings,
        }
        if environments:
            settings["environments"] = environments
        if job_parameters:
            settings["parameters"] = job_parameters
        schedule = _coerce_schedule(
            self.schedule, timezone=self.timezone, pause_status=self.pause_status,
        )
        if schedule is not None:
            settings["schedule"] = schedule

        merged_tags = dict(self.tags) if self.tags else None
        merged_permissions = list(self.permissions) if self.permissions else None

        LOGGER.debug(
            "Deploying flow %r (%d task(s))", self.name, len(nodes),
        )
        if userinfo_defaults:
            defaults = jobs.userinfo_defaults()
            derived_tags = defaults.pop("tags", None)
            if derived_tags:
                merged_tags = {**derived_tags, **(merged_tags or {})}
            # caller settings beat userinfo defaults
            settings = {**defaults, **settings}

        job = jobs.create_or_update(
            name=self.name,
            tasks=tasks,
            tags=merged_tags,
            permissions=merged_permissions,
            **{k: v for k, v in settings.items() if k not in (
                "name", "tasks", "tags", "permissions",
            )},
        )
        LOGGER.info("Deployed flow %r as %r", self.name, job)
        return job

    def _stage_nodes(
        self, client: "DatabricksClient", nodes: Iterable[TaskNode],
    ) -> Tuple[List[Task], List[Any]]:
        """Render every node as a :class:`Task`; collect needed environments.

        Each :class:`WorkflowTask` does the per-node staging via its
        :meth:`WorkflowTask.stage`. We dedupe the matching
        :class:`JobEnvironment` entries here so tasks sharing a key
        don't push duplicates onto the job spec.
        """
        from yggdrasil.databricks.jobs.task import (
            DEFAULT_ENVIRONMENT_DEPENDENCIES,
            DEFAULT_ENVIRONMENT_KEY,
            _default_job_environment,
        )

        tasks: List[Task] = []
        env_keys_seen: set[str] = set()
        environments: List[Any] = []

        for node in nodes:
            details = node.spec.stage(client, node)
            tasks.append(details)
            env_key = getattr(details, "environment_key", None)
            if env_key and env_key not in env_keys_seen:
                env_keys_seen.add(env_key)
                environments.append(
                    _default_job_environment(
                        env_key,
                        dependencies=list(DEFAULT_ENVIRONMENT_DEPENDENCIES),
                    )
                )
        # If any task asked for the default key but didn't surface one,
        # add it once so the job spec parses on serverless workspaces.
        if DEFAULT_ENVIRONMENT_KEY not in env_keys_seen and any(
            getattr(t, "environment_key", None) == DEFAULT_ENVIRONMENT_KEY
            for t in tasks
        ):
            environments.append(
                _default_job_environment(
                    DEFAULT_ENVIRONMENT_KEY,
                    dependencies=list(DEFAULT_ENVIRONMENT_DEPENDENCIES),
                )
            )
        return tasks, environments

    def _build_job_parameters(
        self,
        nodes: Iterable[TaskNode],
        overrides: Mapping[str, Any],
    ) -> List[JobParameterDefinition]:
        """Build :class:`JobParameterDefinition` from the flow signature.

        Every declared flow parameter that *wasn't* pinned in
        ``trace_overrides`` becomes a job-level parameter the user can
        set at run time. Defaults come from the flow signature; the
        caller's ``parameters`` mapping is layered on top for entries
        the signature didn't declare.
        """
        params: List[JobParameterDefinition] = []
        seen: set[str] = set()
        for name, param in self._signature.parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            if name in overrides:
                continue
            default = (
                "" if param.default is inspect.Parameter.empty
                else str(param.default) if param.default is not None
                else ""
            )
            params.append(JobParameterDefinition(name=name, default=default))
            seen.add(name)
        for name, value in self.parameters.items():
            if name in seen:
                # caller's explicit override → patch the default in place
                for p in params:
                    if p.name == name:
                        p.default = str(value)
                        break
                continue
            params.append(JobParameterDefinition(name=str(name), default=str(value)))
            seen.add(name)
        return params

    # ------------------------------------------------------------------ #
    # Run / introspection
    # ------------------------------------------------------------------ #
    def run(
        self,
        *,
        service: Optional["Jobs"] = None,
        client: Optional["DatabricksClient"] = None,
        wait: Any = False,
        deploy_if_missing: bool = True,
        **job_parameters: Any,
    ) -> "JobRun":
        """Trigger a run of the deployed Job.

        When the Job hasn't been deployed yet (or is absent from the
        workspace), :meth:`deploy` runs first by default. Pass
        ``deploy_if_missing=False`` to surface the missing-job error
        instead — useful when the workflow ships pre-deployed via CI
        and you want a guard rail.

        ``**job_parameters`` land on
        :class:`databricks.sdk.service.jobs.JobsAPI.run_now`'s
        ``job_parameters`` map.
        """
        jobs = _resolve_jobs_service(service, client)
        job = jobs.find(name=self.name)
        if job is None:
            if not deploy_if_missing:
                raise RuntimeError(
                    f"Flow.run({self.name!r}): job not found in workspace "
                    f"{jobs.client.base_url.to_string()!r} and "
                    "deploy_if_missing=False — call .deploy() first."
                )
            job = self.deploy(service=jobs)
        return job.run(
            job_parameters={k: str(v) for k, v in job_parameters.items()} or None,
            wait=wait,
        )


def flow(
    func: Optional[Callable[..., Any]] = None,
    /,
    *,
    name: Optional[str] = None,
    schedule: Any = None,
    timezone: str = "UTC",
    pause_status: Any = None,
    parameters: Optional[Mapping[str, Any]] = None,
    tags: Optional[Mapping[str, str]] = None,
    permissions: Optional[Iterable[Any]] = None,
    **job_settings: Any,
) -> Any:
    """Decorate a Python function as a Databricks workflow flow.

    Bare or parametrised, both forms work::

        @flow
        def my_flow(date: str = "2025-01-01"): ...

        @flow(name="my-flow", schedule="0 2 * * *", timezone="Europe/Paris")
        def my_flow(date: str = "2025-01-01"): ...

    The flow body runs locally — calling ``my_flow()`` invokes the
    wrapped function as plain Python, so tests don't need a workspace.
    Calling ``my_flow.deploy()`` traces the body and upserts a
    Databricks Job; ``my_flow.run()`` triggers it.
    """
    def _wrap(f: Callable[..., Any]) -> Flow:
        return Flow(
            f,
            name=name,
            schedule=schedule,
            timezone=timezone,
            pause_status=pause_status,
            parameters=parameters,
            tags=tags,
            permissions=permissions,
            **job_settings,
        )

    if func is None:
        return _wrap
    return _wrap(func)
