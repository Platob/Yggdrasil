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
from typing import Any, Callable, Optional

__all__ = ["AgentHandle", "Fleet"]


class AgentHandle:
    """One spawned process agent — its task, process, and (when done) result."""

    def __init__(self, agent_id: int, task: str, cmd: list[str],
                 proc: subprocess.Popen, out_path: str, err_path: str) -> None:
        self.id = agent_id
        self.task = task
        self.cmd = cmd
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

    def __init__(self, *, python: Optional[str] = None) -> None:
        self.agents: list[AgentHandle] = []
        self._python = python or sys.executable

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
               "--root", root, "--json", "--max-steps", str(max_steps)]
        if engine:
            cmd += ["--engine", engine]
        if tier:
            cmd += ["--tier", tier]
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
        handle = AgentHandle(len(self.agents) + 1, task, command, proc, out.name, err.name)
        self.agents.append(handle)
        return handle

    def spawn_all(self, tasks: list[str], **kw: Any) -> list[AgentHandle]:
        return [self.spawn(t, **kw) for t in tasks]

    def poll(self) -> list[AgentHandle]:
        """Refresh statuses: finalize any agent whose process has exited."""
        for h in self.agents:
            if h.running and h.proc.poll() is not None:
                h._finish()
        return self.agents

    def running(self) -> list[AgentHandle]:
        return [h for h in self.agents if h.running]

    def all_done(self) -> bool:
        return not self.running()

    def monitor(
        self,
        on_update: Optional[Callable[[list[AgentHandle]], None]] = None,
        *,
        interval: float = 0.2,
        timeout: Optional[float] = None,
    ) -> list[AgentHandle]:
        """Drive the fleet to completion, calling *on_update* each tick.

        Returns when every agent has finished (or *timeout* elapses — survivors
        are cancelled and marked ``timeout``). *on_update* receives the full
        agent list so a caller can render a live dashboard.
        """
        start = time.monotonic()
        while True:
            self.poll()
            if on_update is not None:
                on_update(self.agents)
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

    def summary(self) -> list[dict[str, Any]]:
        """A JSON-able rollup — one row per agent (for ``--json`` / the skill)."""
        return [
            {"id": h.id, "task": h.task, "status": h.status,
             "elapsed": round(h.elapsed, 2), "steps": h.steps,
             "answer": h.answer, "files_changed": h.files_changed,
             "error": h.stderr_tail if not h.ok else ""}
            for h in self.agents
        ]
