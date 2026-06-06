"""Agent replication — spawn child agent **processes** locally and monitor them.

A Loki agent replicates by forking copies of itself into separate local
processes, each running a behavior in parallel. :class:`Replica` is the
handle to one such child: start it, poll its :attr:`status`, and collect its
:class:`~yggdrasil.loki.engine.AgentResponse` with :meth:`result`. This is
in-process-tree parallelism on the current machine — not remote jobs.
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Any, Optional

from .engine import AgentResponse

__all__ = ["Replica"]


def _ctx():
    """Prefer a ``fork`` context so a child inherits the live agent — its
    registered behaviors, detected backends, and loaded state — i.e. a true
    copy of the current agent. Falls back to the default where fork is absent.
    """
    try:
        return mp.get_context("fork")
    except ValueError:  # pragma: no cover - non-fork platforms (Windows/macOS)
        return mp.get_context()


def _child_main(agent_ref: str, behavior: str, kwargs: dict, queue: mp.Queue) -> None:
    """Run one behavior in the child and ship back the result (or traceback)."""
    try:
        import importlib

        module_name, _, cls_name = agent_ref.partition(":")
        klass = getattr(importlib.import_module(module_name), cls_name)
        agent = klass.current() if hasattr(klass, "current") else klass()
        result = agent.run(behavior, **kwargs)
        queue.put(("ok", AgentResponse.from_(result)))
    except Exception:  # noqa: BLE001 — ship the failure back, don't crash silently
        import traceback

        queue.put(("err", traceback.format_exc()))


class Replica:
    """A child process running one behavior on a copy of the agent."""

    def __init__(self, agent_ref: str, behavior: str, kwargs: dict) -> None:
        self.agent_ref = agent_ref
        self.behavior = behavior
        self.kwargs = kwargs
        ctx = _ctx()
        self._queue: mp.Queue = ctx.Queue()
        self._proc = ctx.Process(
            target=_child_main,
            args=(agent_ref, behavior, kwargs, self._queue),
            daemon=True,
        )
        self._started = False
        self._outcome: Optional[tuple[str, Any]] = None

    def start(self) -> Replica:
        if not self._started:
            self._proc.start()
            self._started = True
        return self

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid

    def is_alive(self) -> bool:
        return self._started and self._proc.is_alive()

    def _collect(self, timeout: Optional[float] = None) -> tuple[str, Any]:
        if self._outcome is None:
            # The child puts exactly one message before exiting; read it (with
            # the timeout) then join so the process is reaped.
            self._outcome = self._queue.get(timeout=timeout)
            self._proc.join(timeout=1)
        return self._outcome

    @property
    def status(self) -> str:
        """``pending`` → ``running`` → ``done`` / ``failed``."""
        if not self._started:
            return "pending"
        if self._outcome is not None:
            return "done" if self._outcome[0] == "ok" else "failed"
        if self._proc.is_alive() and self._queue.empty():
            return "running"
        kind, _ = self._collect(timeout=5)
        return "done" if kind == "ok" else "failed"

    def result(self, timeout: Optional[float] = None) -> AgentResponse:
        """Wait for and return the child's :class:`AgentResponse` (raises on failure)."""
        kind, payload = self._collect(timeout=timeout)
        if kind == "err":
            raise RuntimeError(f"replica {self.behavior!r} failed:\n{payload}")
        return payload

    # ``wait`` reads naturally alongside Future-like handles.
    wait = result

    def terminate(self) -> None:
        if self._started and self._proc.is_alive():
            self._proc.terminate()

    def __repr__(self) -> str:
        return f"Replica(behavior={self.behavior!r}, pid={self.pid}, status={self.status!r})"
