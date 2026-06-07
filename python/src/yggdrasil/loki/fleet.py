"""Loki fleet — delegate tasks to background *process* agents and monitor them.

Loki's autonomy multiplier: instead of doing every task itself, the agent can
fan a set of independent tasks out to **separate Loki processes** — each a
``ygg loki do <task> --json`` subprocess running its own confined act loop — and
watch them run to completion. A :class:`Fleet` spawns the agents (output streamed
to temp files so a chatty agent never deadlocks on a full pipe), polls their
status, and drives them to done; the CLI renders the live dashboard and the
``delegate`` skill exposes it programmatically.

Pure orchestration: :class:`Fleet` spawns whatever command it's given, so it's
unit tested with a trivial ``python -c`` stand-in agent — no engine required.
:meth:`Fleet.do_command` builds the real ``ygg loki do`` invocation.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

__all__ = ["AgentHandle", "Fleet", "FleetKPIs", "LOCAL_ENGINES"]


@dataclass(frozen=True, slots=True)
class FleetKPIs:
    """A fleet's live rollup as typed fields (not a loose dict)."""

    total: int = 0
    running: int = 0
    done: int = 0
    failed: int = 0
    queued: int = 0
    validated: int = 0          # agents that passed a smoke checkpoint
    steps: int = 0
    tokens: int = 0
    cost: float = 0.0
    elapsed: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

#: Engines that run on the single local accelerator — only one such agent should
#: run at a time (the GPU/NPU is one resource), while remote agents fan out.
LOCAL_ENGINES = frozenset({"ollama", "transformers", "openvino"})


class AgentHandle:
    """One spawned process agent — its task, process, and (when done) result."""

    def __init__(self, agent_id: int, task: str, cmd: list[str],
                 proc: subprocess.Popen, out_path: str, err_path: str,
                 engine: Optional[str] = None) -> None:
        self.id = agent_id
        self.task = task
        self.cmd = cmd
        self.engine = engine          # resolved engine name (for local-concurrency)
        self.local = engine in LOCAL_ENGINES
        self.proc = proc
        self._out_path = out_path
        self._err_path = err_path
        self.started_at = time.monotonic()
        self.ended_at: Optional[float] = None
        #: ``running`` → ``done`` / ``failed`` / ``timeout`` / ``cancelled``.
        self.status = "running"
        self.returncode: Optional[int] = None
        self.result: Optional[dict[str, Any]] = None   # parsed ``do --json`` transcript
        self.stderr_tail = ""

    @property
    def running(self) -> bool:
        return self.status == "running"

    @property
    def ok(self) -> bool:
        return self.status == "done"

    @property
    def elapsed(self) -> float:
        return (self.ended_at or time.monotonic()) - self.started_at

    @property
    def answer(self) -> str:
        return (self.result or {}).get("answer", "") if self.result else ""

    @property
    def files_changed(self) -> list[str]:
        return list((self.result or {}).get("files_changed", []))

    @property
    def steps(self) -> int:
        return len((self.result or {}).get("steps", []))

    @property
    def validated(self) -> "Optional[bool]":
        """Whether the agent ran a passing ``smoke`` checkpoint — ``True`` (all
        smokes green), ``False`` (one failed), or ``None`` (no smoke run). Lets
        the dashboard show which delegated work was actually self-validated."""
        steps = (self.result or {}).get("steps", [])
        smokes = [s for s in steps if isinstance(s, dict) and s.get("tool") == "smoke"]
        if not smokes:
            return None
        return all("exit=0" in str(s.get("observation", "")) for s in smokes)

    @property
    def tokens(self) -> int:
        return int((self.result or {}).get("usage", {}).get("total_tokens", 0)) if self.result else 0

    @property
    def cost(self) -> float:
        return float((self.result or {}).get("usage", {}).get("cost_usd", 0.0)) if self.result else 0.0

    def _finish(self) -> None:
        """The process exited — read its output, parse the JSON, set the status."""
        self.returncode = self.proc.returncode
        self.ended_at = time.monotonic()
        out = err = ""
        try:
            with open(self._out_path, encoding="utf-8", errors="replace") as fh:
                out = fh.read()
            with open(self._err_path, encoding="utf-8", errors="replace") as fh:
                err = fh.read()
        except OSError:
            pass
        for path in (self._out_path, self._err_path):
            try:
                os.unlink(path)
            except OSError:
                pass
        self.stderr_tail = err.strip().splitlines()[-1] if err.strip() else ""
        try:
            self.result = json.loads(out) if out.strip() else None
        except json.JSONDecodeError:
            self.result = None
        completed = self.result.get("completed", True) if isinstance(self.result, dict) else True
        self.status = "done" if (self.returncode == 0 and completed) else "failed"

    def cancel(self, status: str = "cancelled") -> None:
        if self.running:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.returncode = self.proc.returncode
            self.ended_at = time.monotonic()
            self.status = status


class Fleet:
    """Spawn and monitor a set of background process agents."""

    def __init__(self, *, python: Optional[str] = None, max_parallel: Optional[int] = None,
                 max_local: int = 1, cost_cap: Optional[float] = None,
                 per_agent_budget: Optional[float] = None, mesh_dir: Optional[str] = None) -> None:
        self.agents: list[AgentHandle] = []
        self._python = python or sys.executable
        #: Cap on concurrently-running agents (``None`` = unbounded). Extra tasks
        #: queue and launch as running slots free — so a big swarm doesn't
        #: fork-bomb the box.
        self.max_parallel = max_parallel
        #: Concurrent **local-model** agents (the GPU/NPU is one resource): keep
        #: them serialized while remote agents fan out — the mesh's local-model
        #: optimization. 1 = one local agent at a time.
        self.max_local = max_local
        #: Aggregate USD cap across the fleet; when reached with work still queued
        #: the monitor asks (``on_cap``) before going further.
        self.cost_cap = cost_cap
        #: Per-agent USD budget passed through as ``ygg loki do --budget``.
        self.per_agent_budget = per_agent_budget
        #: Shared workspace the agents read/write — the mesh's shared files. The
        #: live roster is mirrored to ``mesh.json`` there so agents can see peers.
        self.mesh_dir = mesh_dir
        self._pending: list[tuple[str, dict]] = []

    def do_command(
        self,
        task: str,
        *,
        root: str = ".",
        engine: Optional[str] = None,
        tier: Optional[str] = None,
        max_steps: int = 8,
        read_only: bool = False,
        allow_shell: bool = False,
        allow_web: bool = False,
    ) -> list[str]:
        """The ``ygg loki do`` argv for *task* — one isolated autonomous agent."""
        cmd = [self._python, "-m", "yggdrasil.loki.cli", "do", task,
               "--root", self.mesh_dir or root, "--json", "--max-steps", str(max_steps)]
        if engine:
            cmd += ["--engine", engine]
        if tier:
            cmd += ["--tier", tier]
        if self.per_agent_budget is not None:
            cmd += ["--budget", str(self.per_agent_budget)]
        if read_only:
            cmd.append("--read-only")
        if allow_shell:
            cmd.append("--allow-shell")
        if allow_web:
            cmd.append("--allow-web")
        return cmd

    def spawn(self, task: str, *, cmd: Optional[list[str]] = None,
              env: Optional[dict[str, str]] = None, **kw: Any) -> AgentHandle:
        """Launch a process agent for *task* (or an explicit *cmd*) → its handle.

        Output goes to temp files (not pipes), so a verbose agent can't deadlock
        the parent on a full OS pipe buffer while others are still running.
        """
        command = cmd or self.do_command(task, **kw)
        out = tempfile.NamedTemporaryFile(prefix="loki-agent-", suffix=".out", delete=False)
        err = tempfile.NamedTemporaryFile(prefix="loki-agent-", suffix=".err", delete=False)
        out.close()
        err.close()
        proc = subprocess.Popen(
            command,
            stdout=open(out.name, "w", encoding="utf-8"),
            stderr=open(err.name, "w", encoding="utf-8"),
            env={**os.environ, **(env or {})},
        )
        handle = AgentHandle(len(self.agents) + 1, task, command, proc, out.name, err.name,
                             engine=kw.get("engine"))
        self.agents.append(handle)
        self._write_mesh()
        return handle

    def spawn_all(self, tasks: list[str], **kw: Any) -> list[AgentHandle]:
        """Queue *tasks* and launch up to ``max_parallel`` of them now.

        Returns the handles started immediately; the rest launch from
        :meth:`poll` as running slots free.
        """
        self._pending.extend((t, kw) for t in tasks)
        return self._launch_pending()

    def _launch_pending(self) -> list[AgentHandle]:
        """Launch queued tasks that pass every gate: overall concurrency,
        local-model concurrency (one accelerator), and the aggregate cost cap."""
        started: list[AgentHandle] = []
        while self._pending:
            if self.cost_cap is not None and self.spent() >= self.cost_cap:
                break                                    # paused — monitor asks via on_cap
            if self.max_parallel is not None and len(self.running()) >= self.max_parallel:
                break
            idx = self._next_launchable()
            if idx is None:
                break
            task, kw = self._pending.pop(idx)
            started.append(self.spawn(task, **kw))
        return started

    def _next_launchable(self) -> Optional[int]:
        """Index of the next queued task allowed to start now — skipping local
        tasks while the local-model slot(s) are busy."""
        local_busy = sum(1 for h in self.running() if h.local) >= self.max_local
        for i, (_, kw) in enumerate(self._pending):
            if local_busy and kw.get("engine") in LOCAL_ENGINES:
                continue
            return i
        return None

    def spent(self) -> float:
        return round(sum(h.cost for h in self.agents), 6)

    def queued(self) -> int:
        return len(self._pending)

    def poll(self) -> list[AgentHandle]:
        """Refresh statuses: finalize exited agents, then fill freed slots."""
        changed = False
        for h in self.agents:
            if h.running and h.proc.poll() is not None:
                h._finish()
                changed = True
        self._launch_pending()
        if changed:
            self._write_mesh()
        return self.agents

    def _write_mesh(self) -> None:
        """Mirror the live roster into ``mesh.json`` in the shared workspace."""
        if not self.mesh_dir:
            return
        try:
            os.makedirs(self.mesh_dir, exist_ok=True)
            with open(os.path.join(self.mesh_dir, "mesh.json"), "w", encoding="utf-8") as fh:
                json.dump({"agents": self.summary(), "kpis": self.kpis().to_dict()}, fh, indent=2)
        except OSError:
            pass

    def running(self) -> list[AgentHandle]:
        return [h for h in self.agents if h.running]

    def all_done(self) -> bool:
        return not self.running() and not self._pending

    def monitor(
        self,
        on_update: Optional[Callable[[list[AgentHandle]], None]] = None,
        *,
        interval: float = 0.2,
        timeout: Optional[float] = None,
        on_cap: Optional[Callable[[dict[str, Any]], Optional[float]]] = None,
    ) -> list[AgentHandle]:
        """Drive the fleet to completion, calling *on_update* each tick.

        Returns when every agent has finished (or *timeout* elapses — survivors
        are cancelled and marked ``timeout``). When the aggregate cost cap is hit
        with work still queued, ``on_cap(kpis)`` is asked for a **new cap** to go
        further; returning ``None`` drops the queue and finishes the running ones.
        """
        start = time.monotonic()
        while True:
            self.poll()
            if on_update is not None:
                on_update(self.agents)
            # Cost cap reached but tasks still waiting → ask whether to go further.
            if self.cost_cap is not None and self._pending and self.spent() >= self.cost_cap:
                new_cap = on_cap(self.kpis()) if on_cap is not None else None
                if new_cap is not None:
                    self.cost_cap = float(new_cap)
                    self._launch_pending()
                else:
                    self._pending.clear()              # stop here; let runners finish
            if self.all_done():
                break
            if timeout is not None and time.monotonic() - start > timeout:
                self.cancel_all(status="timeout")
                if on_update is not None:
                    on_update(self.agents)
                break
            time.sleep(interval)
        return self.agents

    def cancel_all(self, *, status: str = "cancelled") -> None:
        for h in self.running():
            h.cancel(status=status)

    def kpis(self) -> FleetKPIs:
        """Live rollup across the fleet — counts, steps, tokens, cost, wall time."""
        return FleetKPIs(
            total=len(self.agents) + self.queued(),
            running=len(self.running()),
            done=sum(1 for h in self.agents if h.ok),
            failed=sum(1 for h in self.agents if not h.running and not h.ok),
            queued=self.queued(),
            validated=sum(1 for h in self.agents if h.validated is True),
            steps=sum(h.steps for h in self.agents),
            tokens=sum(h.tokens for h in self.agents),
            cost=round(sum(h.cost for h in self.agents), 6),
            elapsed=round(max((h.elapsed for h in self.agents), default=0.0), 1),
        )

    def summary(self) -> list[dict[str, Any]]:
        """A JSON-able rollup — one row per agent (for ``--json`` / the skill)."""
        return [
            {"id": h.id, "task": h.task, "status": h.status,
             "elapsed": round(h.elapsed, 2), "steps": h.steps,
             "answer": h.answer, "files_changed": h.files_changed,
             "error": h.stderr_tail if not h.ok else ""}
            for h in self.agents
        ]
