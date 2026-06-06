"""Built-in Loki behaviors.

The behavior catalog. Two patterns live here:

- :class:`GenieBehavior` — guard on a detected backend, then drive a
  Databricks service endpoint through Loki's token provider
  (``agent.databricks``).
- :class:`PythonProjectBehavior` — a *local* behavior: Loki scaffolds a
  Python project, writes code into it (provided, or reasoned from a task via
  the agent's engine), and runs it — the agent authoring and executing code.

Replication, inter-agent messaging, HTTP ingestion and serving land here next.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Any, Optional

from .behavior import LokiBehavior, register

if TYPE_CHECKING:
    from .agent import Loki

__all__ = ["AgentBehavior", "GenieBehavior", "PythonProjectBehavior"]


@register
class AgentBehavior(LokiBehavior):
    """Pursue a task autonomously inside a file tree — Loki's agentic loop.

    The headline "act on its own + modify files" behavior. Given a ``task``,
    Loki reasons against a confined toolbox (list/read/find/grep, plus
    write/edit unless ``read_only``, plus a shell when ``allow_shell``),
    taking one tool call per turn until it's done — discovering the project
    and changing files itself. Runs anywhere an engine is reachable; thin
    wrapper over :meth:`Loki.act` so code and CLI share one implementation.
    """

    name = "agent"
    description = "Autonomously discover and modify files to accomplish a task."

    def run(
        self,
        agent: Loki,
        *,
        task: str,
        root: str = ".",
        engine: Optional[str] = None,
        tier: Optional[str] = None,
        max_steps: int = 12,
        read_only: bool = False,
        allow_shell: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        return agent.act(
            task,
            root=root,
            engine=engine,
            tier=tier,
            max_steps=max_steps,
            read_only=read_only,
            allow_shell=allow_shell,
        )


@register
class GenieBehavior(LokiBehavior):
    """Ask a Databricks Genie space a question and return its answer."""

    name = "genie"
    description = "Ask a Databricks AI/BI Genie space a question (text + SQL + rows)."
    requires = "databricks"

    def run(
        self,
        agent: Loki,
        *,
        question: str,
        space: Optional[str] = None,
        rows: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        client = agent.databricks
        if client is None:  # available() already guards, belt-and-suspenders
            raise RuntimeError("no Databricks session")

        # Autonomy: when no space is named, reason against the first space the
        # current user can reach.
        if space is None:
            spaces = client.genie.spaces()
            if not spaces:
                raise RuntimeError("no Genie spaces are accessible to this user")
            target = spaces[0]
        else:
            target = client.genie.space(space)

        answer = target.ask(question)
        out: dict[str, Any] = {
            "space_id": target.space_id,
            "conversation_id": answer.conversation_id,
            "text": answer.text,
            "query": answer.query,
            "statement_id": answer.statement_id,
        }
        if rows and answer.query:
            out["rows"] = answer.to_polars()
        return out


@register
class PythonProjectBehavior(LokiBehavior):
    """Scaffold a small Python project, write code into it, and execute it.

    Runs anywhere (no backend required). Either pass ``code`` directly, or a
    ``task`` description that the agent reasons into a script via its engine
    (``agent.reason``). Loki then writes a minimal project (``pyproject.toml``
    + a package with a ``main`` module), runs ``main.py`` in a subprocess, and
    returns where it landed plus the captured output — the agent authoring and
    executing code end-to-end.
    """

    name = "python_project"
    description = "Create a Python project, write code (given or reasoned), and run it."

    def run(
        self,
        agent: Loki,
        *,
        project: str = "ygg_demo",
        task: Optional[str] = None,
        code: Optional[str] = None,
        base_dir: Optional[str] = None,
        run: bool = True,
        timeout: float = 60.0,
        **_: Any,
    ) -> dict[str, Any]:
        # Reason the code from the task when none is supplied (needs an engine).
        if code is None and task:
            code = agent.reason(
                f"Write a single self-contained Python script that: {task}. "
                "Print its result to stdout. Output only the code — no prose, "
                "no markdown fences.",
                system="You are a senior Python engineer. Return runnable code only.",
            )
        if code is None:
            raise ValueError("provide `code=` directly or a `task=` to reason it from")
        # Strip markdown fences a reasoned reply may wrap the code in.
        code = re.sub(r"\A\s*```(?:python)?\n|\n```\s*\Z", "", code).strip() + "\n"

        pkg = re.sub(r"[^0-9A-Za-z_]+", "_", project).strip("_").lower() or "app"
        root = (
            pathlib.Path(base_dir)
            if base_dir
            else pathlib.Path(tempfile.mkdtemp(prefix="ygg-loki-"))
        )
        project = root / pkg
        package = project / pkg
        package.mkdir(parents=True, exist_ok=True)

        (project / "pyproject.toml").write_text(
            f'[project]\nname = "{pkg}"\nversion = "0.1.0"\n'
            f'requires-python = ">=3.9"\n\n'
            f'[project.scripts]\n{pkg} = "{pkg}.main:main"\n'
        )
        (project / "README.md").write_text(f"# {pkg}\n\nScaffolded by Loki.\n")
        (package / "__init__.py").write_text('__all__ = ["main"]\n')
        (package / "main.py").write_text(code)

        files = sorted(
            str(p.relative_to(project)) for p in project.rglob("*") if p.is_file()
        )
        result: dict[str, Any] = {
            "project_dir": str(project),
            "package": pkg,
            "files": files,
        }
        if run:
            proc = subprocess.run(
                [sys.executable, str(package / "main.py")],
                cwd=str(project),
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONPATH": str(project)},
            )
            result.update(
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        return result
