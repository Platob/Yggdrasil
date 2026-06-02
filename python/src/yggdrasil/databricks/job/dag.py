"""Task-graph (DAG) view of a Databricks job / run.

Both a persisted :class:`~yggdrasil.databricks.job.job.Job` (its
``settings.tasks``) and a live :class:`~yggdrasil.databricks.job.run.JobRun`
(its ``run.tasks``) are a set of tasks wired by ``depends_on`` edges. This
module turns either into a small, printable :class:`JobDag` — nodes in
topological order, each with its upstream keys and (for a run) the live
:class:`~yggdrasil.enums.state.State` — so callers inspect / render the graph
without reaching into the SDK task shapes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from yggdrasil.enums.state import State

__all__ = ["JobDag", "JobDagNode"]


@dataclass(frozen=True)
class JobDagNode:
    """One task in the graph: its key, the upstream keys it waits on, and —
    for a run — its live state (``None`` for a static job definition)."""

    key: str
    depends_on: tuple[str, ...] = ()
    state: Optional[State] = None


@dataclass
class JobDag:
    """A job/run task graph. Build with :meth:`from_tasks` from either SDK
    ``Task`` (job settings) or ``RunTask`` (a run) objects — both expose
    ``task_key`` and a ``depends_on`` list of ``TaskDependency``."""

    nodes: list[JobDagNode]

    @classmethod
    def from_tasks(
        cls,
        tasks: Iterable[Any],
        *,
        state_of: Optional[Callable[[Any], Optional[State]]] = None,
    ) -> "JobDag":
        # A node is identified by its task key. A run can list the same key more
        # than once (a retried task is a second attempt) — collapse to one node,
        # keeping the latest attempt's state, so the graph stays a true DAG.
        by_key: dict[str, JobDagNode] = {}
        for t in tasks or ():
            deps = tuple(
                d.task_key
                for d in (getattr(t, "depends_on", None) or ())
                if getattr(d, "task_key", None)
            )
            by_key[t.task_key] = JobDagNode(
                key=t.task_key,
                depends_on=deps,
                state=state_of(t) if state_of is not None else None,
            )
        return cls(list(by_key.values()))

    def __bool__(self) -> bool:
        return bool(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    @property
    def keys(self) -> list[str]:
        return [n.key for n in self.nodes]

    def node(self, key: str) -> Optional[JobDagNode]:
        for n in self.nodes:
            if n.key == key:
                return n
        return None

    def roots(self) -> list[str]:
        """Keys with no upstream dependency — the entry tasks."""
        return [n.key for n in self.nodes if not n.depends_on]

    def leaves(self) -> list[str]:
        """Keys nothing else depends on — the terminal tasks."""
        depended = {d for n in self.nodes for d in n.depends_on}
        return [n.key for n in self.nodes if n.key not in depended]

    def edges(self) -> list[tuple[str, str]]:
        """``(upstream, downstream)`` pairs for every dependency edge."""
        return [(d, n.key) for n in self.nodes for d in n.depends_on]

    def topological_order(self) -> list[JobDagNode]:
        """Nodes in dependency order (Kahn). Any node left over by a cycle or a
        dangling dependency is appended last in input order, so rendering never
        silently drops a task."""
        by_key = {n.key: n for n in self.nodes}
        indeg = {
            n.key: sum(1 for d in n.depends_on if d in by_key) for n in self.nodes
        }
        ready = [n.key for n in self.nodes if indeg[n.key] == 0]
        out: list[str] = []
        seen: set[str] = set()
        while ready:
            k = ready.pop(0)
            if k in seen:
                continue
            seen.add(k)
            out.append(k)
            for n in self.nodes:
                if k in n.depends_on and n.key not in seen:
                    indeg[n.key] -= 1
                    if indeg[n.key] <= 0:
                        ready.append(n.key)
        out.extend(n.key for n in self.nodes if n.key not in seen)
        return [by_key[k] for k in out]

    def render(self) -> str:
        """A compact text rendering — tasks in topological order, each with its
        upstream keys and (when known) live state."""
        count = len(self.nodes)
        lines = [f"JobDag ({count} task{'' if count == 1 else 's'})"]
        rows: list[tuple[str, Optional[State]]] = []
        for node in self.topological_order():
            left = node.key
            if node.depends_on:
                left += "  ← " + ", ".join(node.depends_on)
            rows.append((left, node.state))
        width = max((len(left) for left, _ in rows), default=0)
        for left, state in rows:
            suffix = f"  [{state.name}]" if state is not None else ""
            lines.append(f"  {left.ljust(width)}{suffix}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()
