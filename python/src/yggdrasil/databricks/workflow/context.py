"""Trace-time context that records the DAG a ``@flow`` body produces.

A flow body is *normal Python* — when called directly it just runs.
The workflow layer uses a context manager to switch ``@task``-decorated
callables from "run the function" mode to "record a
:class:`TaskNode`" mode for the duration of a deploy / introspection
pass. The mechanism is a :class:`contextvars.ContextVar` so concurrent
deploys / coroutines don't trample each other's traces.

The context is opened by :meth:`Flow._trace` and closed when control
leaves the ``with`` block. ``@task`` wrappers consult
:func:`current_trace` on every call; ``None`` means "execute normally",
otherwise the call becomes a :class:`TaskNode` registered against
the active :class:`TraceContext`.
"""
from __future__ import annotations

import contextvars
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .nodes import TaskNode

__all__ = ["TraceContext", "current_trace"]


_TRACE_CTX: contextvars.ContextVar[Optional["TraceContext"]] = contextvars.ContextVar(
    "yggdrasil.databricks.workflow.trace",
    default=None,
)


def current_trace() -> Optional["TraceContext"]:
    """Return the active :class:`TraceContext`, or ``None`` outside trace mode."""
    return _TRACE_CTX.get()


class TraceContext:
    """In-process record of a flow's captured DAG.

    Instances are short-lived — one per :meth:`Flow._trace` call.
    Pass through the ``with`` block to expose ``self`` as the active
    trace; downstream task calls register their :class:`TaskNode`
    via :meth:`register`. After exit the recorded nodes live on
    :attr:`nodes` (insertion order = construction order = topological
    by construction because tasks must construct upstreams before
    referencing them).
    """

    __slots__ = ("nodes", "_keys", "_token")

    def __init__(self) -> None:
        self.nodes: List["TaskNode"] = []
        #: Maps the desired ``task_key`` → count of nodes that wanted
        #: it. Used to suffix duplicates with ``_2`` / ``_3`` / …
        #: while keeping the first occurrence at its natural key.
        self._keys: Dict[str, int] = {}
        self._token: Optional[contextvars.Token] = None

    # ------------------------------------------------------------------ #
    # Context-manager protocol
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "TraceContext":
        self._token = _TRACE_CTX.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _TRACE_CTX.reset(self._token)
            self._token = None

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #
    def register(self, node: "TaskNode") -> "TaskNode":
        """Add *node* to the trace, suffixing its key on collision.

        First occurrence of a key lands at its natural value; the
        second gets ``_2``, the third ``_3``, and so on. The suffix
        scheme matches what an operator would type by hand when
        debugging a flow that calls the same task twice — no random
        token, easy to reproduce.
        """
        base = node.task_key
        count = self._keys.get(base, 0)
        if count > 0:
            node.task_key = f"{base}_{count + 1}"
        self._keys[base] = count + 1
        self.nodes.append(node)
        return node

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        keys = [n.task_key for n in self.nodes]
        return f"TraceContext(nodes={keys!r})"
