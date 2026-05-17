"""``@task`` decorator and :class:`WorkflowTask` spec.

A :class:`WorkflowTask` wraps a plain Python callable with the
metadata the workflow layer needs to render it as a Databricks
:class:`Task`:

* function source (extracted lazily via the existing
  :func:`stage_python_callable` / :func:`stage_python_notebook_callable`
  staging path),
* task-level overrides — task-key, retry count, environment-key,
  cluster spec, condition / on-failure routing,
* the rendering flavour (``"spark"`` for a flat ``.py`` script,
  ``"notebook"`` for a cell-split Databricks notebook).

When called *outside* a :class:`TraceContext`, a :class:`WorkflowTask`
behaves exactly like the original function — same args / return value
— so unit tests of the function body don't need a mock workspace.
When called *inside* a trace, the call short-circuits into a
:class:`TaskNode` registered against the active trace, capturing
the upstream futures the flow body wired up. The flow's
:meth:`Flow.deploy` then walks the trace to stage every node as
a Databricks task.
"""
from __future__ import annotations

import concurrent.futures as cf
import functools
import inspect
import logging
from typing import Any, Callable, Iterable, List, Mapping, Optional, TYPE_CHECKING

from databricks.sdk.service.jobs import TaskDependency

from contextlib import nullcontext

from .context import current_trace
from .metadata import collect_source_metadata, describe_metadata
from .nodes import TaskNode
from .resources import SecretRef

if TYPE_CHECKING:
    from databricks.sdk.service.jobs import Task

    from yggdrasil.databricks.client import DatabricksClient
    from .resources import SecretRef


__all__ = ["WorkflowTask", "task"]

#: Recognised pool flavours for :attr:`WorkflowTask.pool` / :meth:`WorkflowTask.map`.
#: ``"thread"`` → :class:`concurrent.futures.ThreadPoolExecutor` (default; best
#: for I/O-bound task bodies — Databricks SDK calls, HTTP, SQL warehouse round
#: trips). ``"process"`` → :class:`concurrent.futures.ProcessPoolExecutor`
#: (CPU-bound work; requires the task function and arguments to be picklable).
_POOL_EXECUTORS: dict[str, type[cf.Executor]] = {
    "thread": cf.ThreadPoolExecutor,
    "process": cf.ProcessPoolExecutor,
}


def _resolve_executor_cls(pool: Optional[str]) -> type[cf.Executor]:
    """Map a ``"thread"`` / ``"process"`` token to the matching executor class."""
    if pool is None:
        return cf.ThreadPoolExecutor
    try:
        return _POOL_EXECUTORS[pool]
    except KeyError as exc:
        raise ValueError(
            f"WorkflowTask.map(pool={pool!r}): expected 'thread' or 'process'. "
            "'thread' uses a ThreadPoolExecutor (best for I/O-bound work — "
            "Databricks SDK calls, HTTP, SQL warehouses); 'process' uses a "
            "ProcessPoolExecutor (CPU-bound work; task fn + args must be "
            "picklable)."
        ) from exc

LOGGER = logging.getLogger(__name__)


class WorkflowTask:
    """Decoration spec for one workflow task.

    Constructed by the :func:`task` decorator. Two responsibilities:

    1. Stand in for the original callable so the flow body can call
       it like a normal function (locally) or capture a
       :class:`TaskNode` (in trace mode).
    2. Carry the task-level overrides (``retries``, ``environment_key``,
       …) that :func:`Flow.deploy` consumes when staging the node as
       a Databricks :class:`Task`.

    Parameters
    ----------
    func
        The wrapped callable. Must live in an importable source file
        — :func:`inspect.getsource` runs against it at staging time.
    task_key
        Databricks task key. Defaults to ``func.__name__``;
        :class:`TraceContext` adds ``_2`` / ``_3`` / … suffixes when
        the same task is called multiple times in one flow.
    task_type
        ``"spark"`` (default) — render as a flat ``.py`` :class:`SparkPythonTask`.
        ``"notebook"`` — render as a Databricks-format ``.py`` notebook
        with per-cell logs, wired as a :class:`NotebookTask`.
    retries
        Forwarded to :attr:`Task.max_retries`. ``None`` keeps the
        Databricks default (no retry).
    environment_key
        Serverless environment key. Defaults to
        :data:`yggdrasil.databricks.jobs.task.DEFAULT_ENVIRONMENT_KEY`
        ("ygg-default") so the parent job's auto-attached
        ``ygg[data,databricks]`` environment is used.
    existing_cluster_id / job_cluster_key / new_cluster
        Compute bindings. At most one should be set per task; clears
        ``environment_key`` if set so the task runs on classic
        compute rather than serverless.
    pool
        Default parallel-pool flavour for :meth:`map`. ``"thread"``
        (default) → :class:`concurrent.futures.ThreadPoolExecutor`,
        which fits I/O-bound task bodies (Databricks SDK calls, HTTP,
        SQL warehouses). ``"process"`` →
        :class:`concurrent.futures.ProcessPoolExecutor` for CPU-bound
        work; the task function and arguments must be picklable. The
        per-call ``WorkflowTask.map(pool=...)`` override wins.
    max_workers
        Default worker count for :meth:`map`'s local-mode pool.
        ``None`` (default) defers to the executor's default
        (``min(32, os.cpu_count() + 4)`` for threads,
        ``os.cpu_count()`` for processes). Per-call
        ``WorkflowTask.map(max_workers=...)`` overrides this.
    task_fields
        Catch-all forwarded to :class:`databricks.sdk.service.jobs.Task`
        — ``description``, ``timeout_seconds``, ``run_if``,
        ``email_notifications``, etc.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        task_key: Optional[str] = None,
        task_type: str = "spark",
        retries: Optional[int] = None,
        environment_key: Optional[str] = ...,
        existing_cluster_id: Optional[str] = None,
        job_cluster_key: Optional[str] = None,
        new_cluster: Any = None,
        pool: Optional[str] = None,
        max_workers: Optional[int] = None,
        client: Optional["DatabricksClient"] = None,
        **task_fields: Any,
    ) -> None:
        if task_type not in ("spark", "notebook"):
            raise ValueError(
                f"WorkflowTask(task_type={task_type!r}): expected 'spark' or "
                f"'notebook'. 'spark' renders the function as a flat .py "
                "SparkPythonTask; 'notebook' renders it as a cell-split "
                "Databricks notebook NotebookTask."
            )

        self.func = func
        self.__wrapped__ = func
        self.task_key = task_key or func.__name__
        self.task_type = task_type
        self.retries = retries
        # Default ``environment_key`` to the workspace serverless env unless
        # the caller pinned a classic-compute binding — those two are
        # mutually exclusive on the Databricks side.
        if environment_key is ...:
            self.environment_key = None if (
                existing_cluster_id or job_cluster_key or new_cluster
            ) else None  # filled in by stage_python_callable / notebook
        else:
            self.environment_key = environment_key
        self.existing_cluster_id = existing_cluster_id
        self.job_cluster_key = job_cluster_key
        self.new_cluster = new_cluster
        # Validate the pool flavour up-front so a typo at decoration
        # time fails loudly with a helpful message instead of much later
        # at :meth:`map` call time.
        if pool is not None:
            _resolve_executor_cls(pool)
        #: Default executor flavour for :meth:`map` — ``"thread"`` /
        #: ``"process"`` / ``None``. ``None`` means "fall back to
        #: ``ThreadPoolExecutor`` unless :meth:`map` overrides it".
        self.pool = pool
        #: Default ``max_workers`` passed to the local-mode executor in
        #: :meth:`map`. ``None`` defers to the executor's own default.
        self.max_workers = max_workers
        #: Override the staging workspace for this task. When set,
        #: :meth:`stage` writes the staged ``.py`` to *this* client's
        #: workspace instead of the parent flow's. Useful for tasks
        #: whose source must live alongside a different workspace's
        #: secrets / volumes — the deployed Job still runs on the
        #: parent flow's workspace cluster, so that cluster must be
        #: able to read the staging path.
        self.client = client
        self.task_fields = dict(task_fields)
        functools.update_wrapper(self, func)

    # ------------------------------------------------------------------ #
    # Call surface
    # ------------------------------------------------------------------ #
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run *func* locally, or register a :class:`TaskNode` in trace mode.

        Outside a :class:`TraceContext` this resolves any
        :class:`SecretRef` arguments / defaults to their actual values
        via :meth:`DatabricksClient.current` and then runs the
        underlying function — so a flow can be unit-tested locally
        ("``daily_etl()``") or executed against a real workspace via
        Databricks Connect ("``daily_etl(date='2025-01-15')``") with
        the same call-shape, no separate adapter. Inside a trace the
        call becomes a future the flow's :meth:`Flow.deploy` walks
        later — secrets stay as :class:`SecretRef` placeholders until
        the cluster resolves them at run time.
        """
        trace = current_trace()
        if trace is None:
            resolved_args, resolved_kwargs = self._resolve_local_secrets(
                args, kwargs,
            )
            return self.func(*resolved_args, **resolved_kwargs)
        node = TaskNode(
            spec=self,
            task_key=self.task_key,
            args=tuple(args),
            kwargs=dict(kwargs),
        )
        return trace.register(node)

    # ------------------------------------------------------------------ #
    # Parallel fan-out — Prefect-style ``.map`` over an iterable
    # ------------------------------------------------------------------ #
    def map(
        self,
        iterable: Iterable[Any],
        *constants: Any,
        pool: Optional[str] = None,
        max_workers: Optional[int] = None,
        executor: Optional[cf.Executor] = None,
        **kw_constants: Any,
    ) -> List[Any]:
        """Fan the task out across *iterable* in parallel.

        Prefect-style mapping. The task is applied once per element of
        *iterable*; the remaining ``*constants`` / ``**kw_constants`` are
        passed through unchanged on every call. The behaviour depends on
        whether a :class:`TraceContext` is active:

        * **Local mode** (no active trace). The function runs in-process
          across a :class:`concurrent.futures.ThreadPoolExecutor` (or
          :class:`~concurrent.futures.ProcessPoolExecutor` if
          ``pool="process"``), preserving input order. Returns a
          ``list`` of results — same shape as ``[self(item, *constants,
          **kw_constants) for item in iterable]`` but executed in
          parallel. :class:`SecretRef` args/defaults resolve the same
          way :meth:`__call__` resolves them, so unit tests need no
          workspace mock.

        * **Trace mode** (inside ``Flow.trace`` / ``Flow.deploy``). One
          :class:`TaskNode` is registered per element, with the auto-
          suffix collision logic from :class:`TraceContext` producing
          unique task keys (``step``, ``step_2``, ``step_3``, …). The
          Databricks scheduler runs the resulting tasks in parallel
          (subject to the job's ``max_concurrent_runs`` and the
          cluster's slot capacity). Returns a ``list`` of
          :class:`TaskNode` futures — pass it to a downstream task
          (``reduce(results)``) to build a fan-in.

        Parameters
        ----------
        iterable
            The per-call dimension. Materialised into a list up-front
            so the length is known at submission time; pass a generator
            only when the body fits in memory.
        *constants
            Positional args passed unchanged on every call. Typically
            configuration the per-item body needs (target paths, table
            names, etc.).
        pool
            Override the task's default :attr:`pool` flavour.
            ``"thread"`` (default) for I/O-bound bodies, ``"process"``
            for CPU-bound bodies. Ignored in trace mode — Databricks'
            own scheduler handles the parallelism there.
        max_workers
            Override the task's default :attr:`max_workers`. ``None``
            defers to the executor's own default. Ignored in trace mode.
        executor
            Pre-built executor to submit into. When supplied, the
            decorator does not shut it down on exit — caller owns the
            lifetime. Mutually exclusive with ``pool`` / ``max_workers``
            (those would build a fresh executor; passing one here means
            "reuse this one instead"). Ignored in trace mode.
        **kw_constants
            Keyword args passed unchanged on every call.

        Raises
        ------
        TypeError
            *iterable* is not iterable.
        ValueError
            ``pool`` isn't ``"thread"`` / ``"process"``.

        Examples
        --------
        Local fan-out over an in-memory list::

            @task(pool="thread", max_workers=8)
            def fetch(url: str) -> bytes:
                import urllib.request
                return urllib.request.urlopen(url).read()

            bodies = fetch.map(["https://a", "https://b", "https://c"])

        Inside a flow — one Databricks task per element::

            @task
            def process(date: str, table: str) -> int: ...

            @task
            def reduce(counts: list[int]) -> int:
                return sum(counts)

            @flow(name="backfill")
            def backfill(target: str = "main.dim.events"):
                dates = ["2025-01-01", "2025-01-02", "2025-01-03"]
                counts = process.map(dates, table=target)
                reduce(counts)
        """
        items = list(iterable)
        trace = current_trace()

        if trace is None:
            return self._map_local(
                items,
                constants,
                kw_constants,
                pool=pool,
                max_workers=max_workers,
                executor=executor,
            )

        # Trace mode: one TaskNode per element. The trace context
        # auto-suffixes colliding ``task_key``s (``step``, ``step_2``,
        # …) so the staged Databricks tasks land at unique keys.
        nodes: List[TaskNode] = []
        for item in items:
            node = TaskNode(
                spec=self,
                task_key=self.task_key,
                args=(item, *constants),
                kwargs=dict(kw_constants),
            )
            nodes.append(trace.register(node))
        return nodes

    def _map_local(
        self,
        items: List[Any],
        constants: tuple,
        kw_constants: dict,
        *,
        pool: Optional[str],
        max_workers: Optional[int],
        executor: Optional[cf.Executor],
    ) -> List[Any]:
        """Run the task body across *items* using a local pool.

        Resolves :class:`SecretRef` args/defaults once (same as
        :meth:`__call__`) so the per-item submission pays only the
        function call, not a per-item secret round-trip. Submits
        :attr:`func` directly to the executor — no local-closure
        wrapper, so ``ProcessPoolExecutor`` can pickle the call
        target as long as the task function and arguments are
        themselves picklable. Returns results in input order — same
        shape as a list comprehension but parallelised through the
        configured executor.
        """
        # Resolve any SecretRef in the broadcast args once; per-item
        # submissions can then reuse the materialised values without
        # paying a Databricks Secrets round-trip per call.
        resolved_constants, resolved_kw = self._resolve_local_secrets(
            constants, kw_constants,
        )

        pool_flavour = pool if pool is not None else self.pool
        executor_cls = _resolve_executor_cls(pool_flavour)
        workers = max_workers if max_workers is not None else self.max_workers

        # Caller-supplied executor: reuse it without shutting it down.
        # No caller-supplied executor: build a fresh one and tear it
        # down on the way out.
        if executor is not None:
            ex_ctx: Any = nullcontext(executor)
        else:
            ex_ctx = executor_cls(max_workers=workers)

        with ex_ctx as ex:
            futures = [
                ex.submit(self.func, item, *resolved_constants, **resolved_kw)
                for item in items
            ]
            return [f.result() for f in futures]

    # ------------------------------------------------------------------ #
    # Local secret resolution — feeds both __call__ and tests
    # ------------------------------------------------------------------ #
    def _resolve_local_secrets(
        self,
        args: tuple,
        kwargs: Mapping[str, Any],
    ) -> tuple[tuple, dict]:
        """Materialise any :class:`SecretRef` args / defaults to live values.

        Walks every positional / keyword argument and replaces a
        :class:`SecretRef` with the cleartext value fetched through
        :meth:`DatabricksClient.current`'s secrets service. Defaults
        in the function signature that the caller didn't override get
        the same treatment, so a task whose signature declares
        ``api_key: str = secret("vendor", "key")`` receives a real
        string when called locally — same shape the staged invocation
        sees on a Databricks cluster.

        Raises :class:`RuntimeError` (via :func:`ygg.secret`) when
        no :class:`DatabricksClient` can be resolved — that's the
        signal to wire one in (``DatabricksClient(host=…, token=…)``
        as a context manager) or to mock the secret in tests by
        passing it explicitly.
        """
        resolved_args = tuple(_materialise_secret(a) for a in args)
        resolved_kwargs = {
            name: _materialise_secret(value) for name, value in kwargs.items()
        }

        sig = inspect.signature(self.func)
        params = list(sig.parameters.values())
        pos_bound = 0
        for param in params:
            if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                if pos_bound < len(resolved_args):
                    pos_bound += 1
                    continue
            if param.kind == param.VAR_POSITIONAL:
                continue
            if param.kind == param.VAR_KEYWORD:
                continue
            if param.name in resolved_kwargs:
                continue
            if isinstance(param.default, SecretRef):
                resolved_kwargs[param.name] = _materialise_secret(param.default)

        return resolved_args, resolved_kwargs

    def __repr__(self) -> str:
        return f"WorkflowTask(task_key={self.task_key!r}, func={self.func.__qualname__!r})"

    # ------------------------------------------------------------------ #
    # Compose ``depends_on`` edges from upstream tasks / nodes
    # ------------------------------------------------------------------ #
    def after(self, *upstreams: Any) -> Callable[..., Any]:
        """Decorator-style helper: ``@task; @other.after(self)`` adds a hard edge.

        The :meth:`__call__` path already infers dependencies from
        :class:`TaskNode` values passed as arguments, so most flows
        never need this. Use it when a downstream task should depend
        on an upstream task that *doesn't* feed it a value (e.g.
        side-effect-only ordering — drain a folder before reading
        from it). Returns a wrapper that, when called inside a
        trace, attaches the additional ``upstreams`` to the resulting
        node's ``depends_on`` list.
        """
        def _wrap(f: Callable[..., Any]) -> Callable[..., Any]:
            # We must wrap the *decorated* WorkflowTask so the
            # additional edges land on every call. ``@task @other.after(...)``
            # ordering: outermost decorator wins, which is ``@task`` here.
            inner: WorkflowTask = f if isinstance(f, WorkflowTask) else task(f)

            @functools.wraps(inner.func)
            def _call(*args: Any, **kwargs: Any) -> Any:
                result = inner(*args, **kwargs)
                if isinstance(result, TaskNode):
                    for up in upstreams:
                        if isinstance(up, TaskNode) and up not in result.depends_on:
                            result.depends_on.append(up)
                return result

            _call.task = inner  # type: ignore[attr-defined]
            return _call

        return _wrap

    # ------------------------------------------------------------------ #
    # Staging
    # ------------------------------------------------------------------ #
    def stage(
        self,
        client: "DatabricksClient",
        node: TaskNode,
    ) -> "Task":
        """Render *node* as a Databricks :class:`Task`.

        Wires in:

        - the staged ``.py`` (script or notebook flavour, per
          :attr:`task_type`) via the existing
          :func:`stage_python_callable` /
          :func:`stage_python_notebook_callable` pipeline,
        - ``depends_on`` derived from the node's upstream
          :class:`TaskNode` references,
        - task-level overrides (``max_retries``, ``environment_key``,
          ``existing_cluster_id`` / ``job_cluster_key`` / ``new_cluster``,
          and any ``task_fields`` the decorator pinned),
        - the trailing ``publish_return`` wrap so downstream tasks can
          read this task's return value via ``dbutils.jobs.taskValues``.

        Returns the assembled :class:`Task`; the caller is responsible
        for the ``environments`` extension (handled by :meth:`Flow.deploy`
        in batch).
        """
        from dataclasses import replace as _dc_replace

        from yggdrasil.databricks.jobs.task import (
            stage_python_callable,
            stage_python_notebook_callable,
        )

        from .nodes import filter_trace_values

        # FlowParam args strip out → unbound parameters; TaskNode + literal
        # + SecretRef args pass through (their ``__repr__`` renders the
        # right runtime expression).
        bound_args, bound_kwargs, _unbound = filter_trace_values(
            node.args, node.kwargs,
        )

        # Promote SecretRef defaults to explicit kwargs so the staged
        # invocation reads ``api_key=ygg.secret('scope', 'key')``
        # instead of relying on the function default. The default
        # ``api_key: str = secret(...)`` would otherwise be a SecretRef
        # at call time, which the @checkargs wrap would reject before
        # the body runs — promoting up-front keeps the runtime path
        # identical to the local-call path resolved in __call__.
        sig = inspect.signature(self.func)
        pos_bound = len(bound_args)
        positional_index = 0
        for param in sig.parameters.values():
            if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                if positional_index < pos_bound:
                    positional_index += 1
                    continue
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            if param.name in bound_kwargs:
                continue
            if isinstance(param.default, SecretRef):
                bound_kwargs[param.name] = param.default

        if self.task_type == "notebook":
            stage_fn = stage_python_notebook_callable
        else:
            stage_fn = stage_python_callable

        # Per-task client override wins; otherwise the flow's
        # workspace receives the staged source.
        staging_client = self.client if self.client is not None else client

        details, _deps, _env_names = stage_fn(
            staging_client,
            self.func,
            *bound_args,
            task_key=node.task_key,
            workspace_pypi=True,
            **bound_kwargs,
        )

        overrides: dict[str, Any] = dict(self.task_fields)
        if self.retries is not None:
            overrides["max_retries"] = self.retries
        if self.environment_key is not None:
            overrides["environment_key"] = self.environment_key
        if self.existing_cluster_id:
            overrides["existing_cluster_id"] = self.existing_cluster_id
            overrides["environment_key"] = None
        if self.job_cluster_key:
            overrides["job_cluster_key"] = self.job_cluster_key
            overrides["environment_key"] = None
        if self.new_cluster is not None:
            overrides["new_cluster"] = self.new_cluster
            overrides["environment_key"] = None

        if node.depends_on:
            overrides["depends_on"] = [
                TaskDependency(task_key=upstream.task_key)
                for upstream in node.depends_on
            ]

        # Append a source-attribution footer to whatever description
        # ``stage_python_callable`` derived from the function signature.
        # Operators reading the Databricks UI's task list get the
        # GitHub link, git commit, and module path inline — no need to
        # crack the staged file open.
        task_metadata = collect_source_metadata(self.func)
        metadata_footer = describe_metadata(task_metadata, prefix="Task source")
        if metadata_footer:
            current_desc = overrides.get("description") or getattr(
                details, "description", None,
            ) or ""
            if metadata_footer not in current_desc:
                joined = (
                    f"{current_desc}\n\n{metadata_footer}"
                    if current_desc else metadata_footer
                )
                # Databricks Task.description has a 1000-char hard cap.
                overrides["description"] = joined[:1000]

        if overrides:
            details = _dc_replace(details, **{
                k: v for k, v in overrides.items() if v is not None
            })

        return details


def task(
    func: Optional[Callable[..., Any]] = None,
    /,
    *,
    task_key: Optional[str] = None,
    task_type: str = "spark",
    retries: Optional[int] = None,
    environment_key: Optional[str] = ...,
    existing_cluster_id: Optional[str] = None,
    job_cluster_key: Optional[str] = None,
    new_cluster: Any = None,
    pool: Optional[str] = None,
    max_workers: Optional[int] = None,
    client: Optional["DatabricksClient"] = None,
    **task_fields: Any,
) -> Any:
    """Decorate a Python callable as a Databricks workflow task.

    Usable bare or parametrised — both forms work::

        @task
        def step(x: int) -> str: ...

        @task(retries=2, task_type="notebook", environment_key="ygg-default")
        def big_step(x: int) -> str: ...

    The decorated callable runs unchanged outside a flow trace, so
    unit tests don't need a workspace. Inside :meth:`Flow.deploy` the
    same call records a :class:`TaskNode` future that's staged as a
    Databricks :class:`Task` with the metadata supplied here.

    The kwargs map 1:1 onto :class:`Task` overrides — ``retries`` →
    ``max_retries``, ``environment_key`` →
    :class:`JobEnvironment` binding, ``existing_cluster_id`` /
    ``job_cluster_key`` / ``new_cluster`` → cluster placement (mutually
    exclusive with ``environment_key``; setting any of them clears
    the default serverless env). Any unrecognised kwarg falls through
    to ``task_fields`` and lands on the :class:`Task` directly
    (``description``, ``timeout_seconds``, ``run_if``, …).

    ``pool`` / ``max_workers`` configure :meth:`WorkflowTask.map`'s
    local-mode executor — ``"thread"`` (default) for I/O-bound bodies
    that fan out via HTTP / SQL / Databricks SDK calls, ``"process"``
    for CPU-bound bodies. Defaults are overridable per-call on
    ``.map(pool=..., max_workers=...)``. The trace-mode behaviour of
    ``.map`` (one Databricks task per element) is unaffected by these
    kwargs — Databricks' own scheduler handles the parallelism there.
    """
    def _wrap(f: Callable[..., Any]) -> WorkflowTask:
        return WorkflowTask(
            f,
            task_key=task_key,
            task_type=task_type,
            retries=retries,
            environment_key=environment_key,
            existing_cluster_id=existing_cluster_id,
            job_cluster_key=job_cluster_key,
            new_cluster=new_cluster,
            pool=pool,
            max_workers=max_workers,
            client=client,
            **task_fields,
        )

    if func is None:
        return _wrap
    return _wrap(func)


def _materialise_secret(value: Any) -> Any:
    """Return *value*, replacing a :class:`SecretRef` with its cleartext.

    Routes through :func:`ygg.secret` so the same code path runs
    locally and on the cluster — the cluster picks the in-process
    Databricks-injected ``DatabricksClient.current()``; locally the
    caller supplies one via environment variables, a profile, or
    Databricks Connect.
    """
    if isinstance(value, SecretRef):
        from . import ygg

        return ygg.secret(value.scope, value.key)
    return value
