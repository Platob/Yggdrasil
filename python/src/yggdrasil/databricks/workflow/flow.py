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
from .metadata import (
    collect_source_metadata,
    describe_metadata,
    metadata_tags,
)
from .nodes import FlowParam, TaskNode

if TYPE_CHECKING:
    from yggdrasil.databricks.client import DatabricksClient
    from yggdrasil.databricks.jobs import Job, JobRun, Jobs


__all__ = ["Flow", "flow"]

LOGGER = logging.getLogger(__name__)


def _resolve_jobs_service(
    service: Optional["Jobs"],
    client: Optional["DatabricksClient"],
    *,
    fallback_client: Optional["DatabricksClient"] = None,
) -> "Jobs":
    """Pick a :class:`Jobs` service from caller hints.

    Precedence: explicit *service* → explicit *client* →
    *fallback_client* (typically the one pinned on the decorator) →
    :meth:`DatabricksClient.current`. That order lets a caller
    override a flow's pinned workspace per-call while still respecting
    the default the decorator declared.
    """
    if service is not None:
        return service
    if client is not None:
        return client.jobs
    if fallback_client is not None:
        return fallback_client.jobs
    from yggdrasil.databricks.client import DatabricksClient
    return DatabricksClient.current().jobs


def _build_auto_prefix(func: Optional[Callable[..., Any]] = None) -> str:
    """Compose ``[YGG][<project>/<version>]`` from :meth:`UserInfo.current`.

    Resolution order, picking the *first* non-empty source for each
    field:

    1. **Project**: ``UserInfo.product`` (PEP 621 name from the nearest
       ``pyproject.toml``) → ``UserInfo.hostname`` → ``"ygg"``.
    2. **Version**: ``UserInfo.product_version`` → the *func*'s git
       commit short hash (when *func* lives in a checkout) →
       ``"ygg-<yggdrasil version>"``.

    Best-effort: any exception in :class:`UserInfo` resolution falls
    through to the hostname / yggdrasil-version fallbacks, so a deploy
    on a stripped-down image (no ``pyproject.toml``, no git remote)
    still gets a meaningful prefix instead of failing.
    """
    project: Optional[str] = None
    version: Optional[str] = None

    try:
        from yggdrasil.environ import UserInfo

        info = UserInfo.current()
        project = info.product or info.hostname
        version = info.product_version
    except Exception:  # noqa: BLE001 — best-effort, prefix isn't load-bearing
        LOGGER.debug("UserInfo lookup for auto-prefix failed", exc_info=True)

    if not version and func is not None:
        try:
            meta = collect_source_metadata(func)
        except Exception:  # noqa: BLE001
            meta = None
        if meta and meta.git_commit:
            version = meta.git_commit[:12]

    if not version:
        try:
            from yggdrasil.version import __version__ as ygg_version
            version = f"ygg-{ygg_version}"
        except Exception:  # noqa: BLE001
            version = "dev"

    project = project or "ygg"
    return f"[YGG][{project}/{version}]"


def _apply_prefix(
    policy: Any,
    name: str,
    func: Optional[Callable[..., Any]] = None,
) -> str:
    """Apply the configured prefix policy to *name*.

    * ``policy=True`` (default) — auto-derive ``[YGG][project/version]`` via
      :func:`_build_auto_prefix`.
    * ``policy=False`` / ``policy=None`` — return *name* unchanged.
    * ``policy=<str>`` — use the literal string as the prefix; a
      trailing space is added if missing so the rendered name stays
      readable.
    """
    if not policy:
        return name
    if policy is True:
        prefix = _build_auto_prefix(func)
    elif isinstance(policy, str):
        prefix = policy
    else:
        raise TypeError(
            f"@flow(prefix={policy!r}): expected True, False, None, or a "
            "string literal."
        )
    if not prefix:
        return name
    if not prefix.endswith(" "):
        prefix = prefix + " "
    return f"{prefix}{name}"


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
        prefix: Any = True,
        client: Optional["DatabricksClient"] = None,
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
        #: Name-prefix policy. ``True`` (default) auto-derives the
        #: ``[YGG][<project>/<version>] `` prefix from
        #: :meth:`UserInfo.current` so deployed jobs are scannable in
        #: the Databricks UI ("which project / version shipped this?").
        #: ``False`` opts out — :attr:`deployed_name` equals
        #: :attr:`name`. A string literal is used verbatim as the
        #: prefix (must include the trailing space if you want one).
        self.prefix = prefix
        #: Target workspace for :meth:`deploy` / :meth:`run`. When
        #: ``None``, the active :meth:`DatabricksClient.current` wins —
        #: that's the right default for code that wants to follow
        #: whatever workspace the caller has in scope (env vars,
        #: ``with DatabricksClient(...)`` block, …). Pin a specific
        #: client at ``@flow(client=…)`` time when the flow always
        #: targets one workspace ("prod-eu") regardless of how the
        #: caller is set up.
        self.client = client
        self.job_settings = dict(job_settings)
        self._signature = inspect.signature(func)
        functools.update_wrapper(self, func)

    # ------------------------------------------------------------------ #
    # Deployed name — :attr:`name` with the configured prefix applied
    # ------------------------------------------------------------------ #
    @property
    def deployed_name(self) -> str:
        """Job name as it actually lands on the Databricks workspace.

        With the default ``prefix=True`` policy this returns
        ``"[YGG][<project>/<version>] <name>"`` where ``<project>``
        and ``<version>`` come from :meth:`UserInfo.current` —
        ``UserInfo.product`` (the PEP 621 project name parsed from
        the nearest ``pyproject.toml``) falling back to
        ``UserInfo.hostname``, and ``UserInfo.product_version``
        falling back to the source-file git commit short hash and
        finally to ``ygg-<yggdrasil version>``. With ``prefix=False``
        the raw :attr:`name` is returned; with a string ``prefix=``
        the literal is prepended.
        """
        return _apply_prefix(self.prefix, self.name, self.func)

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
        jobs = _resolve_jobs_service(service, client, fallback_client=self.client)
        nodes = self.trace(**(trace_overrides or {}))
        if not nodes:
            raise RuntimeError(
                f"Flow.deploy({self.name!r}): the flow body produced no tasks. "
                "A flow must call at least one @task-decorated callable."
            )

        tasks, environments = self._stage_nodes(jobs.client, nodes)
        job_parameters = self._build_job_parameters(nodes, trace_overrides or {})

        # Auto-derive source attribution from the flow function — module,
        # source path / line, yggdrasil version, and (when the file lives
        # inside a git checkout) the current commit + an HTTPS link to the
        # exact line on the hosting provider. These land on:
        #   * job tags    — ``ygg.module`` / ``ygg.git_commit`` /
        #                    ``ygg.source_url`` etc. for UI search,
        #   * job description — appended as a human-readable footer so
        #                       operators can jump straight to the source.
        flow_metadata = collect_source_metadata(self.func)
        deployed_name = self.deployed_name

        settings: Dict[str, Any] = {
            "name": deployed_name,
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

        # Description: caller-supplied wins; otherwise lead with the
        # flow's docstring (if any) and trail with the metadata footer
        # so the Databricks UI surfaces the source link without forcing
        # the operator to crack the staged file open.
        existing_description = settings.get("description")
        metadata_footer = describe_metadata(flow_metadata, prefix="Flow source")
        if existing_description is None:
            doc_line = flow_metadata.docstring
            parts = [p for p in (doc_line, metadata_footer) if p]
            if parts:
                settings["description"] = "\n\n".join(parts)[:4096]
        elif metadata_footer and metadata_footer not in str(existing_description):
            settings["description"] = (
                f"{existing_description}\n\n{metadata_footer}"
            )[:4096]

        merged_tags: Optional[dict] = dict(self.tags) if self.tags else None
        derived_tags = metadata_tags(flow_metadata)
        derived_tags["ygg.flow"] = self.name
        if derived_tags:
            # Caller-supplied tags win on collision — auto-derived
            # values only fill in keys the caller didn't pin.
            merged_tags = {**derived_tags, **(merged_tags or {})}

        merged_permissions = list(self.permissions) if self.permissions else None

        LOGGER.debug(
            "Deploying flow %r as job name %r (%d task(s))",
            self.name, deployed_name, len(nodes),
        )
        if userinfo_defaults:
            defaults = jobs.userinfo_defaults()
            derived_tags = defaults.pop("tags", None)
            if derived_tags:
                merged_tags = {**derived_tags, **(merged_tags or {})}
            # caller settings beat userinfo defaults
            settings = {**defaults, **settings}

        job = jobs.create_or_update(
            name=deployed_name,
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
        jobs = _resolve_jobs_service(service, client, fallback_client=self.client)
        # The deployed Job lives under :attr:`deployed_name` (the
        # prefix-applied form), not the raw :attr:`name` — look it up
        # under the same key :meth:`deploy` registered.
        target_name = self.deployed_name
        job = jobs.find(name=target_name)
        if job is None:
            if not deploy_if_missing:
                raise RuntimeError(
                    f"Flow.run({self.name!r}): job {target_name!r} not found "
                    f"in workspace {jobs.client.base_url.to_string()!r} and "
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
    prefix: Any = True,
    client: Optional["DatabricksClient"] = None,
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

    ``prefix`` controls the deployed job-name prefix. The default
    (``True``) auto-derives ``[YGG][<project>/<version>] `` via
    :meth:`UserInfo.current` so deployed jobs are scannable in the
    Databricks UI by owning project. Pass ``False`` to deploy under
    the bare :attr:`Flow.name`, or a string literal to pin a custom
    prefix.

    ``client`` pins a target :class:`DatabricksClient`. When omitted
    the active :meth:`DatabricksClient.current` wins — usually
    what you want. Set it explicitly when the flow always targets
    one workspace ("prod-eu") regardless of how the caller's
    environment is configured. Per-call overrides on
    :meth:`Flow.deploy` / :meth:`Flow.run` still take precedence.
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
            prefix=prefix,
            client=client,
            **job_settings,
        )

    if func is None:
        return _wrap
    return _wrap(func)
