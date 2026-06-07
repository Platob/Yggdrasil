"""Prefect-style tasks & flows that run **transparently** on Databricks.

The :func:`task` / :func:`flow` decorators wrap a function into a **callable**
:class:`Task` / :class:`Flow` — like Prefect, but location-transparent:

- **call it** (``my_flow(x)`` / ``my_task(x)``) and it runs *where it should*:
  from your laptop it deploys + runs as a Databricks **serverless** job and
  returns the real result; already **inside** Databricks (the wheel task on the
  cluster) it runs **in-process** — so a flow body's ``fetch.map(urls)`` fans
  tasks out locally on the cluster. One call site, right place, no recursion;
- ``.local(x)`` forces in-process execution (tests, debugging);
- ``.submit(...)`` runs in the background and returns a :class:`Future`
  (``.result()`` blocks) — a flow fans tasks out with ``.submit`` / ``.map``;
- ``.deploy(client)`` registers it as a Databricks job *without* running it (for
  schedules / file-arrival triggers).

    @task(retries=2)
    def fetch(url: str) -> bytes: ...

    @flow(name="etl")
    def etl(day: date, n: int):
        return [f.result() for f in fetch.map(urls_for(day))]

    etl(date.today(), 7)   # from a laptop → runs on Databricks, returns result
    etl.local(date.today(), 7)   # force in-process
    etl.deploy(client)     # register as a job (don't run)

The deploy ships the **live** code as a wheel (built from the package on disk —
dev checkout or installed), placed in the shared workspace pypi registry, or in
a per-user folder + rebuilt when the package is an editable install. The cluster
task runs ``ygg run`` (the ``ygg`` entry point's ``run`` subcommand —
:mod:`yggdrasil.databricks.job.runner`), which imports the target, coerces
parameters to the function signature via the cast registry, runs the body, and
round-trips the result.

Class-based flows subclass :class:`Flow` and override :meth:`~Flow.run` (the
body), :attr:`~_Runnable.name`, :meth:`~Flow.parameters`, :meth:`~Flow.trigger`.
"""
from __future__ import annotations

import contextvars
import functools
import logging
import pickle
import re
import secrets
import sys
import time
from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, TypeVar

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yggdrasil.concurrent.threading import ThreadJob
    from yggdrasil.databricks.job.job import Job

logger = logging.getLogger(__name__)

#: Active while a body runs **in-process** (``.local`` / ``.submit`` / the
#: on-cluster ``ygg run`` runner). Nested task/flow calls consult it so they run
#: locally too instead of each dispatching its own Databricks job — the flow,
#: not every task, is the unit that ships to the cluster.
_LOCAL_MODE: "contextvars.ContextVar[bool]" = contextvars.ContextVar(
    "ygg_local_mode", default=False
)


def ensure_console_logging(name: str = "yggdrasil", level: int = logging.INFO) -> None:
    """Attach an INFO stdout handler to the *name* logger if it has none, so
    interactive deploys / job runs surface ygg logs (the default root config is
    WARNING-only). Idempotent and scoped — never touches the root logger."""
    lg = logging.getLogger(name)
    lg.setLevel(min(lg.level or level, level) if lg.level else level)
    if not any(isinstance(h, logging.StreamHandler) for h in lg.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        lg.addHandler(handler)


__all__ = [
    "Task",
    "Flow",
    "Future",
    "task",
    "flow",
    # legacy aliases
    "JobSkeleton",
    "TaskSkeleton",
    "CallableSkeleton",
    "job",
]

T = TypeVar("T")


def _render(value: Any) -> str:
    return value if isinstance(value, str) else str(value)


def _slug(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_") or "flow"


class Future(Generic[T]):
    """Handle to a :meth:`Task.submit` / :meth:`Flow.submit` background run."""

    __slots__ = ("_job",)

    def __init__(self, job: "ThreadJob") -> None:
        self._job = job

    def result(self, timeout: Any = None, *, raise_error: bool = True) -> T:
        """Block until the run finishes and return its result."""
        return self._job.wait(timeout, raise_error=raise_error)

    wait = result

    @property
    def done(self) -> bool:
        return self._job.is_done


class _Runnable:
    """Shared run + retry + transparent-dispatch + deploy surface for tasks/flows.

    A call routes by **location**: inside Databricks it runs in-process
    (:meth:`local`); from anywhere else it deploys + runs itself as a serverless
    job and returns the real result (:meth:`_dispatch_remote`)."""

    fn: Optional[Callable]
    name: str
    retries: int
    retry_delay_seconds: float

    # -- serverless / wheel defaults (shared by Task and Flow) -----------
    package_name: str = "ygg"          # wheel that ships the single ``ygg`` entry point
    entry_point: str = "ygg"           # the only console script; ``run`` subcommand → runner.py
    task_key: str = "run"
    #: Job-level tags (key → value) carried onto :meth:`Jobs.create_or_update`,
    #: merged on top of the client's owner/product defaults. ``None`` adds none.
    job_tags: dict[str, str] | None = None
    serverless: bool = True
    environment_key: str = "default"
    #: Serverless environment version. ``None`` resolves at deploy time to match
    #: the local Python (see :func:`serverless_environment_version`); pin a string.
    environment_version: "str | None" = None
    #: Build + ship the live package as a wheel on deploy (so the cluster runs
    #: exactly this code). Set ``False`` to instead install published ``ygg``.
    build_wheel: bool = True
    #: Project extras pulled into the built wheel's metadata (``[databricks]`` so
    #: the bundled image carries its databricks runtime deps).
    wheel_extras: "tuple[str, ...]" = ("databricks",)
    #: Fallback when not shipping a wheel — published ``ygg`` (``[databricks]``).
    dependencies: "tuple[str, ...]" = ("ygg[databricks]",)
    #: Always-installed extras on top of the wheel / :attr:`dependencies`.
    extra_dependencies: "tuple[str, ...]" = ("databricks-sdk",)
    #: Seconds to wait for a remote-dispatched run before giving up.
    remote_timeout: float = 3600.0
    #: Attach a serverless environment for **every** supported Python (keyed
    #: ``py3XX``) on deploy, not just the local-matched ``default``. The build
    #: already produces a wheel per Python; this exposes them as job environments.
    all_environments: bool = False
    #: Bundle the **whole transitive dependency closure** as wheels and ship
    #: them all, so the serverless environment installs with **zero PyPI
    #: access** ("0 pip install") — instead of the project wheel + index
    #: requirements. Trades a larger one-time upload for an offline, fully
    #: reproducible env build. Mutually exclusive with :attr:`all_environments`
    #: (bundles target the deploy host's single Python).
    bundle_dependencies: bool = False
    #: Reference a reusable, named serverless **base environment** (a
    #: ``<name>.yml`` in the workspace, the same convention ``ygg databricks
    #: seed`` writes) instead of inlining the dependency list — the ygg image is
    #: written there once (create-or-update) and the job points at it by file
    #: path, so jobs share one cached env. ``None`` keeps the classic inline
    #: environment. Any user package layers on top as extra dependencies. Ignored
    #: when :attr:`all_environments` is set (the per-Python matrix stays inline).
    #: Falls back to inline if the env can't be written.
    base_environment_name: "str | None" = None

    _wheel_paths: "tuple[str, ...]" = ()
    _ygg_wheels: "list[str] | None" = None
    _user_wheels: "list[str]" = ()
    _user_deps: "list[str]" = ()
    _base_environment_path: "str | None" = None
    _user_layer: "list[str] | None" = None
    _runner_params: "list[str] | None" = None
    _client: Any = None

    # -- execution -------------------------------------------------------
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """The body. Override it, or wrap a function with the decorator."""
        if self.fn is None:
            raise NotImplementedError(f"{type(self).__name__} has no run() body")
        return self.fn(*args, **kwargs)

    def _call(self, *args: Any, **kwargs: Any) -> Any:
        # Mark this (and everything it calls) as in-process, so a flow body's
        # nested task calls run here rather than each dispatching its own job.
        token = _LOCAL_MODE.set(True)
        try:
            attempt = 0
            while True:
                try:
                    return self.run(*args, **kwargs)
                except Exception:
                    if attempt >= self.retries:
                        raise
                    attempt += 1
                    if self.retry_delay_seconds:
                        time.sleep(self.retry_delay_seconds)
        finally:
            _LOCAL_MODE.reset(token)

    def local(self, *args: Any, **kwargs: Any) -> Any:
        """Run **in-process** (honouring :attr:`retries`), skipping the remote
        routing — the escape hatch for tests and the on-cluster runner."""
        return self._call(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run where it belongs: in-process when **inside** Databricks (or under
        a :meth:`local` / :meth:`submit` body), else deploy + run on Databricks
        and return the real result."""
        from yggdrasil.databricks.client import DatabricksClient

        if _LOCAL_MODE.get() or DatabricksClient.is_in_databricks_environment():
            return self._call(*args, **kwargs)
        return self._dispatch_remote(args, kwargs)

    def submit(self, *args: Any, **kwargs: Any) -> "Future":
        """Run in the background; return a :class:`Future`."""
        from yggdrasil.concurrent.threading import Job as ThreadCall

        return Future(ThreadCall.make(self._call, *args, **kwargs).fire_and_forget())

    def map(self, iterable: Any, **kwargs: Any) -> "list[Future]":
        """:meth:`submit` once per item — Prefect-style fan-out."""
        return [self.submit(item, **kwargs) for item in iterable]

    def with_options(self, **overrides: Any) -> Any:
        """Return a copy with attributes overridden."""
        clone = self.__class__.__new__(self.__class__)
        clone.__dict__.update(self.__dict__)
        clone.__dict__.update(overrides)
        return clone

    # -- transparent remote dispatch ------------------------------------
    def _dispatch_remote(self, args: tuple, kwargs: dict) -> Any:
        """Deploy this task/flow, run it once on Databricks with *args*/*kwargs*
        marshalled through a workspace payload, block, and return the real
        result. The cluster runs ``ygg run`` against :meth:`_target_ref`."""
        from yggdrasil.databricks.client import DatabricksClient
        from yggdrasil.databricks.path import DatabricksPath

        client = self._client or DatabricksClient()
        user = client.workspace_client().current_user.me().user_name
        base = f"/Workspace/Users/{user}/.ygg/runs/{_slug(self.name)}/{secrets.token_hex(6)}"
        payload_path, result_path = f"{base}/payload.pkl", f"{base}/result.pkl"

        payload = DatabricksPath.from_(payload_path, client=client)
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_bytes(pickle.dumps((tuple(args), dict(kwargs))))

        self._runner_params = [
            self._target_ref(), "--payload", payload_path, "--result", result_path,
        ]
        try:
            job = self.deploy(client)
            job.run(wait=self.remote_timeout, raise_error=True)
        finally:
            self._runner_params = None
        return pickle.loads(DatabricksPath.from_(result_path, client=client).read_bytes())

    def _target_ref(self) -> str:
        """``module.path:qualname`` the on-cluster runner imports to get back
        this object — the wrapped function, or the class for a class-based flow."""
        target = self.fn if self.fn is not None else type(self)
        return f"{target.__module__}:{target.__qualname__}"

    # -- wheel / environment --------------------------------------------
    def wheel_package(self) -> str:
        """The top-level import package to wheel — where this task/flow is
        defined, so the deploy adapts to any project (the wheel is built from
        this package's *live* files on disk)."""
        target = self.fn if self.fn is not None else type(self)
        return target.__module__.split(".")[0]

    def _scheduled_params(self) -> list[str]:
        """Positional string params a scheduled/triggered run passes (the runner
        coerces them to the signature). Flows carry ``@flow(parameters=...)``."""
        return []

    def _runner_parameters(self) -> list[str]:
        """The wheel-task parameters passed to the ``ygg`` entry point: a leading
        ``run`` subcommand, then the per-call payload form when dispatching, else
        ``[target, *scheduled-params]`` for a deploy."""
        if self._runner_params is not None:
            base = list(self._runner_params)
        else:
            base = [self._target_ref(), *self._scheduled_params()]
        return ["run", *base]

    def _serverless_dependencies(self, client: Any) -> list[str]:
        """Build + upload the serverless wheels **for every Python version** and
        return the flat dependency list for the *matched* (default) environment.

        Ships the ygg runner image (wheel by path + runtime deps from the index)
        plus — when this task/flow lives in a *different* package — that package's
        wheel + its declared deps, so the runner target is importable on the
        cluster. Editable installs build to a per-user folder and rebuild each
        deploy; published ones reuse the shared registry. The per-Python wheel
        sets are stashed for :meth:`environments` to compose the full matrix when
        :attr:`all_environments` is set.

        Splits the result into the **ygg image** deps (the shared base) and the
        **user-package layer** so :attr:`base_environment_name`, when set, can
        write the (stable) image into a reusable named env and layer the user
        package on top. Returns the flat union (the inline fallback)."""
        from yggdrasil.databricks.job import wheel as W

        editable_ygg = W.is_editable_install("ygg")
        pkg = self.wheel_package()
        dist = W.distribution_for(pkg)
        user_pkg = W._norm(dist) != "ygg"

        if self.bundle_dependencies:
            # 0 pip install: ship the whole dependency closure as wheels — the
            # env installs entirely from them, no PyPI. Bundle ygg + (when the
            # target lives elsewhere) the user package, both with their deps.
            base = W.user_pypi_dir(client) if editable_ygg else W.WORKSPACE_PYPI_DIR
            ygg_base = W.ensure_bundle(client, "ygg", workspace_dir=base, rebuild=editable_ygg)
            user_layer: list[str] = []
            if user_pkg:
                ud = W.is_editable_install(dist)
                ubase = W.user_pypi_dir(client) if ud else W.WORKSPACE_PYPI_DIR
                user_layer = W.ensure_bundle(
                    client, pkg, extras=self.wheel_extras, workspace_dir=ubase, rebuild=ud)
            self._ygg_wheels, self._user_wheels, self._user_deps = ygg_base, user_layer, []
        else:
            ygg_wheels = W.ensure_ygg_wheels(
                client,
                workspace_dir=(W.user_pypi_dir(client) if editable_ygg else W.WORKSPACE_PYPI_DIR),
                rebuild=editable_ygg,
            )
            # The shared image is the local-matched ygg wheel + its runtime
            # (index) deps; the user package (if any) is the layer on top.
            ygg_base = [W.wheel_for_python(ygg_wheels)] + W.ygg_runtime_dependencies()
            user_wheels: list[str] = []
            user_deps: list[str] = []
            if user_pkg:
                editable = W.is_editable_install(dist)
                base = W.user_pypi_dir(client) if editable else W.WORKSPACE_PYPI_DIR
                user_wheels = W.ensure_wheels(client, pkg, workspace_dir=f"{base}/{_slug(dist)}")
                user_deps = W._project_dependencies(dist, set(self.wheel_extras))
            user_layer = ([W.wheel_for_python(user_wheels)] + user_deps) if user_wheels else []
            self._ygg_wheels, self._user_wheels, self._user_deps = ygg_wheels, user_wheels, user_deps

        # Reusable named base environment: write the (stable) ygg image once and
        # have the job reference it by path; the user package layers on inline.
        # Falls back to the classic inline env if it can't be written.
        self._base_environment_path, self._user_layer = None, None
        if self.base_environment_name and not self.all_environments:
            try:
                self._base_environment_path = W.ensure_named_environment(
                    client, W.environment_folder_of(self.base_environment_name),
                    dependencies=ygg_base,
                    environment_version=self.environment_version,
                    # Match the seed's project-folder layout: a job whose
                    # base_environment_name is the canonical version-tagged stem
                    # ``ygg-<version>-py3XX`` writes the very file the seed does —
                    # ``environment/ygg/ygg-<version>-py3XX.yml`` — by folding the
                    # stem to its project folder and keeping it as the filename.
                    filename=f"{self.base_environment_name}.yml",
                )
                self._user_layer = list(user_layer)
            except Exception:  # noqa: BLE001 — degrade to inline deps on any failure
                logger.warning(
                    "could not write base environment %r — inlining dependencies",
                    self.base_environment_name, exc_info=True,
                )
                self._base_environment_path = None

        seen: set[str] = set()
        return [d for d in (ygg_base + user_layer) if not (d in seen or seen.add(d))]

    def _python_dependencies(self, python: str) -> list[str]:
        """The dependency list for a specific *python* — the wheels matching it
        (by path) + runtime/user index deps (composed from the stashed matrix)."""
        from yggdrasil.databricks.job import wheel as W

        deps = [W.wheel_for_python(self._ygg_wheels, python)] + W.ygg_runtime_dependencies()
        if self._user_wheels:
            deps += [W.wheel_for_python(self._user_wheels, python)] + list(self._user_deps)
        seen: set[str] = set()
        return [d for d in deps if not (d in seen or seen.add(d))]

    def effective_dependencies(self) -> list[str]:
        """Shipped wheels once :meth:`deploy` has composed them, else the
        published :attr:`dependencies` (``ygg`` pinned to the running version) +
        :attr:`extra_dependencies`."""
        if getattr(self, "_wheel_paths", None):
            return list(self._wheel_paths)
        return [self._pin(d) for d in self.dependencies] + list(self.extra_dependencies)

    @staticmethod
    def _pin(dependency: str) -> str:
        """Pin a bare ``ygg`` / ``ygg[...]`` requirement to the running version
        so the deployed job installs the same code."""
        if dependency == "ygg" or dependency.startswith("ygg["):
            from yggdrasil.version import __version__

            return f"{dependency}=={__version__}"
        return dependency

    def environments(self) -> Optional[list]:
        """Serverless environment list, or ``None`` when not serverless.

        Always carries a ``"default"`` env (the local-matched runtime +
        :meth:`effective_dependencies`). When :attr:`all_environments` is set and
        the per-Python wheels have been built (post-:meth:`deploy`), also appends
        one env per :data:`~yggdrasil.databricks.job.wheel.SUPPORTED_PYTHONS`
        (keyed ``py3XX``) so a task can run under any Python by environment key.

        When :meth:`_serverless_dependencies` wrote a reusable named base
        environment (:attr:`base_environment_name`), the default env references
        it by file path (``base_environment``) and only layers the user package
        on top — one shared, cached env instead of an inlined dependency list."""
        if not self.serverless:
            return None
        from databricks.sdk.service.compute import Environment
        from databricks.sdk.service.jobs import JobEnvironment
        from yggdrasil.databricks.job import wheel as W

        def env(key: str, deps: list, python: "str | None") -> Any:
            return JobEnvironment(
                environment_key=key,
                spec=Environment(
                    environment_version=(
                        self.environment_version or W.serverless_environment_version(python)
                    ),
                    dependencies=deps,
                ),
            )

        if getattr(self, "_base_environment_path", None):
            # Reference the shared, version-pinned ygg base env; layer the user
            # package (empty for ygg-only jobs) on top. base_environment carries
            # the environment_version, so it isn't set alongside it.
            return [
                JobEnvironment(
                    environment_key=self.environment_key,
                    spec=Environment(
                        base_environment=self._base_environment_path,
                        dependencies=(self._user_layer or None),
                    ),
                )
            ]

        envs = [env(self.environment_key, self.effective_dependencies(), None)]
        if self.all_environments and getattr(self, "_ygg_wheels", None):
            envs += [
                env(W.environment_key_for(v), self._python_dependencies(v), v)
                for v in W.SUPPORTED_PYTHONS
            ]
        return envs

    def tasks(self) -> list:
        """The single serverless python-wheel task that runs ``ygg run``."""
        from databricks.sdk.service.jobs import PythonWheelTask, Task as DBTask

        return [
            DBTask(
                task_key=self.task_key,
                environment_key=(self.environment_key if self.serverless else None),
                python_wheel_task=PythonWheelTask(
                    package_name=self.package_name,
                    entry_point=self.entry_point,
                    parameters=self._runner_parameters(),
                ),
            )
        ]

    def definition(self) -> dict:
        """Render the :meth:`Jobs.create_or_update` kwargs for this task/flow."""
        spec: dict[str, Any] = {"name": self.name, "tasks": self.tasks()}
        environments = self.environments()
        if environments is not None:
            spec["environments"] = environments
        if self.job_tags:
            spec["tags"] = dict(self.job_tags)
        return spec

    def deploy(self, client: Any) -> "Job":
        """Get-or-create the live :class:`Job` from :meth:`definition` (without
        running it). When :attr:`build_wheel` is set, ships the live package as
        wheels (:meth:`_serverless_dependencies`) so the cluster runs this code."""
        ensure_console_logging()  # so the deploy CRUD is visible interactively
        logger.info("deploying %s %r", type(self).__name__.lower(), self.name)
        if self.build_wheel and self.serverless:
            self._wheel_paths = tuple(self._serverless_dependencies(client))
        spec = self.definition()
        logger.info("create-or-update job %r", self.name)
        job = client.jobs.create_or_update(name=spec.pop("name"), **spec)
        logger.info("deployed job %r (id=%s)", self.name, getattr(job, "job_id", None))
        return job


class Task(_Runnable, Generic[T]):
    """A callable unit of work; also deployable as one databricks ``Task``."""

    def __init__(
        self,
        fn: Callable[..., T],
        *,
        name: Optional[str] = None,
        retries: int = 0,
        retry_delay_seconds: float = 0.0,
        tags: "tuple[str, ...]" = (),
        key: Optional[str] = None,
        depends_on: "tuple[str, ...] | list[str]" = (),
        entry_point: Optional[str] = None,
        package_name: Optional[str] = None,
        **task_options: Any,
    ) -> None:
        self.fn = fn
        self.name = name or fn.__name__
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds
        self.tags = tuple(tags)
        self.task_key = key or self.name
        self.depends_on = tuple(depends_on)
        self.task_options = task_options
        if entry_point is not None:
            self.entry_point = entry_point
        if package_name is not None:
            self.package_name = package_name
        functools.update_wrapper(self, fn)

    def to_task(self, parameters: list[str] | None = None) -> Any:
        """Render a databricks ``Task`` (python-wheel) with explicit *parameters*
        and dependency edges — for hand-built multi-task job DAGs (the transparent
        single-task dispatch uses :meth:`tasks`). *parameters* are the arguments
        the runner receives; the ``run`` subcommand the ``ygg`` entry point needs
        is prepended automatically."""
        from databricks.sdk.service.jobs import (
            PythonWheelTask,
            Task as DBTask,
            TaskDependency,
        )

        return DBTask(
            task_key=self.task_key,
            depends_on=([TaskDependency(task_key=d) for d in self.depends_on] or None),
            environment_key=(self.environment_key if self.serverless else None),
            python_wheel_task=PythonWheelTask(
                package_name=self.package_name,
                entry_point=self.entry_point,
                parameters=["run", *(parameters or [])],
            ),
            **self.task_options,
        )


class Flow(_Runnable):
    """A callable flow; deploys as a Databricks **serverless** job."""

    def __init__(
        self,
        fn: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        trigger: Any = None,
        retries: int = 0,
        retry_delay_seconds: float = 0.0,
        parameters: "tuple[str, ...] | list[str]" = (),
        entry_point: Optional[str] = None,
        package_name: Optional[str] = None,
    ) -> None:
        self.fn = fn
        self.name = name or (fn.__name__ if fn is not None else type(self).__name__)
        self._trigger = trigger
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds
        self._parameters = tuple(parameters)
        self._wheel_paths = ()
        if entry_point is not None:
            self.entry_point = entry_point
        if package_name is not None:
            self.package_name = package_name
        if fn is not None:
            functools.update_wrapper(self, fn)

    # -- deploy surface (override in class-based flows) -----------------
    def parameters(self) -> list[str]:
        """Positional string parameters a scheduled/triggered run passes."""
        return [_render(p) for p in self._parameters]

    _scheduled_params = parameters

    def trigger(self) -> Any:
        """The databricks ``TriggerSettings`` (file-arrival / schedule), or
        ``None``. Function-built flows carry the ``@flow(trigger=...)`` value."""
        return self._trigger

    def definition(self) -> dict:
        """:class:`_Runnable.definition` plus the schedule/file-arrival trigger."""
        spec = super().definition()
        trigger = self.trigger()
        if trigger is not None:
            spec["trigger"] = trigger
        return spec


# ---------------------------------------------------------------------------
# Decorators — wrap a function into a callable Task / Flow
# ---------------------------------------------------------------------------


def task(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    retries: int = 0,
    retry_delay_seconds: float = 0.0,
    key: Optional[str] = None,
    depends_on: "tuple[str, ...] | list[str]" = (),
    entry_point: Optional[str] = None,
    package_name: Optional[str] = None,
    **task_options: Any,
) -> Any:
    """Turn a function into a callable :class:`Task` (Prefect-style)."""

    def deco(f: Callable) -> Task:
        return Task(
            f,
            name=name,
            retries=retries,
            retry_delay_seconds=retry_delay_seconds,
            key=key,
            depends_on=depends_on,
            entry_point=entry_point,
            package_name=package_name,
            **task_options,
        )

    return deco(func) if callable(func) else deco


def flow(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    trigger: Any = None,
    retries: int = 0,
    retry_delay_seconds: float = 0.0,
    parameters: "tuple[str, ...] | list[str]" = (),
    entry_point: Optional[str] = None,
    package_name: Optional[str] = None,
) -> Any:
    """Turn a function into a callable :class:`Flow` (Prefect-style)."""

    def deco(f: Callable) -> Flow:
        return Flow(
            f,
            name=name,
            trigger=trigger,
            retries=retries,
            retry_delay_seconds=retry_delay_seconds,
            parameters=parameters,
            entry_point=entry_point,
            package_name=package_name,
        )

    return deco(func) if callable(func) else deco


# Legacy names kept so existing imports keep working.
JobSkeleton = Flow
TaskSkeleton = Task
CallableSkeleton = _Runnable
job = flow
