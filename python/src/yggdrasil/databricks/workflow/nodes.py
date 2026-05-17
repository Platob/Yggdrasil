"""Trace-time nodes that record the DAG a ``@flow`` body builds.

Prefect-style flows describe the DAG by *calling* tasks inside the
flow body. To capture that graph without actually executing the
tasks, every ``@task``-decorated callable, when invoked inside an
active :class:`TraceContext`, returns a :class:`TaskNode` future
instead of running. The future records:

* the underlying :class:`~yggdrasil.databricks.workflow.task.WorkflowTask`
  spec (which carries the function source, retries, environment
  bindings, …),
* the positional / keyword arguments the caller passed,
* the task-key under which it'll land in the Databricks Job (with
  collision suffixing handled by the :class:`TraceContext`),
* the upstream :class:`TaskNode` dependencies, inferred from any
  arg that is itself a :class:`TaskNode`.

:class:`FlowParam` is the matching sentinel for flow-level inputs.
The :func:`Flow.deploy` path threads each declared flow parameter
through the trace as a :class:`FlowParam`, so a task call that uses
it (``step(date)``) reads as "this task takes ``date`` from the
Databricks Job's parameters at run time" — the staging path leaves
that parameter unbound and wires the matching
``{{job.parameters.<name>}}`` placeholder.

Both are deliberately tiny / pickle-safe dataclasses with no
back-references to live SDK handles, so traces survive across
fork / spawn boundaries (Spark workers in particular).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .task import WorkflowTask

__all__ = ["FlowParam", "TaskNode"]


@dataclass(frozen=True, slots=True)
class FlowParam:
    """Trace-time sentinel for a flow-level parameter.

    Holds the parameter name and its declared default. ``__repr__``
    renders the literal Python the staged invocation reaches for when
    *this* parameter feeds a task directly — but the more common
    path is that :func:`Flow.deploy` drops :class:`FlowParam` args
    *before* staging, so the parameter stays unbound and the existing
    ``{{job.parameters.<name>}}`` plumbing in
    :func:`stage_python_callable` resolves it at runtime.
    """

    name: str
    default: Any = None

    def __repr__(self) -> str:
        # Mirrors :class:`SecretRef.__repr__`: the staged script imports
        # ``_ygg_runtime`` and ``task_value`` doubles as the generic
        # "read a binding by name" helper outside the unbound-parameter
        # plumbing path. Realistically this repr is rarely emitted —
        # :func:`Flow._stage_node_args` strips :class:`FlowParam` values
        # so they flow through SparkPythonTask.parameters — but the
        # fallback is here for completeness.
        return f"_ygg_runtime.task_value({self.name!r})"


@dataclass(slots=True)
class TaskNode:
    """A captured task call inside a :func:`Flow` trace.

    Attributes:
        spec: The :class:`WorkflowTask` decoration the call came from.
        task_key: Databricks task key the call will land under. May be
            suffixed with ``_2`` / ``_3`` / … by the
            :class:`TraceContext` when the same task is called multiple
            times in one flow.
        args: Positional args the caller passed (including any upstream
            :class:`TaskNode` futures and :class:`FlowParam` sentinels).
        kwargs: Keyword args the caller passed, same shape as *args*.
        depends_on: Upstream :class:`TaskNode` futures referenced by
            *args* / *kwargs*. Populated by :meth:`_resolve_deps`
            (called in ``__post_init__``); the order matches first
            appearance in ``(*args, *kwargs.values())``.
    """

    spec: "WorkflowTask"
    task_key: str
    args: Tuple[Any, ...] = ()
    kwargs: dict = field(default_factory=dict)
    depends_on: List["TaskNode"] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.depends_on:
            self.depends_on = self._resolve_deps()

    def _resolve_deps(self) -> List["TaskNode"]:
        """Extract upstream :class:`TaskNode` references from args/kwargs.

        Walks every positional and keyword value, deduplicates by
        instance id (a node referenced by both ``args`` and ``kwargs``
        only registers one edge), and returns them in first-appearance
        order — handy when the resulting :class:`TaskDependency` list
        is rendered in the Databricks UI.
        """
        out: List[TaskNode] = []
        seen: set[int] = set()
        for value in (*self.args, *self.kwargs.values()):
            if isinstance(value, TaskNode) and id(value) not in seen:
                seen.add(id(value))
                out.append(value)
        return out

    def __repr__(self) -> str:
        # Same trick as :class:`SecretRef`: the staged script imports
        # ``_ygg_runtime``, so embedding the literal call here lets the
        # existing :func:`_classify_invocation_params` path render
        # downstream invocations without a special case. Downstream
        # tasks that take this node as an argument see
        # ``_ygg_runtime.task_value('<task_key>')`` at the call site.
        return f"_ygg_runtime.task_value({self.task_key!r})"

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


def filter_trace_values(
    args: Tuple[Any, ...],
    kwargs: dict,
) -> Tuple[Tuple[Any, ...], dict, List[str]]:
    """Split staging args into ``(bound_args, bound_kwargs, unbound_names)``.

    :class:`FlowParam` values are *removed* from the staging payload
    and their parameter names are returned in *unbound_names*, so the
    staging path treats them as parameters the caller didn't supply —
    triggering the existing ``{{job.parameters.<name>}}`` plumbing
    inside :func:`stage_python_callable`.

    :class:`TaskNode` values are *kept* — their ``__repr__`` renders
    the runtime ``task_value(...)`` call at the staging site, so the
    Databricks side sees a literal binding. Plain values (literals,
    :class:`SecretRef`) pass through untouched.

    Returns the filtered ``args`` tuple, the filtered ``kwargs``
    mapping, and the list of stripped flow-parameter names so the
    caller can include them on the parent ``JobSettings.parameters``
    list if they aren't already declared there.
    """
    bound_args: List[Any] = []
    unbound_names: List[str] = []

    for value in args:
        if isinstance(value, FlowParam):
            unbound_names.append(value.name)
            continue
        bound_args.append(value)

    bound_kwargs: dict = {}
    for name, value in kwargs.items():
        if isinstance(value, FlowParam):
            unbound_names.append(value.name)
            continue
        bound_kwargs[name] = value

    return tuple(bound_args), bound_kwargs, unbound_names
