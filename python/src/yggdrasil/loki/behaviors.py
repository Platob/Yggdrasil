"""Built-in Loki behaviors.

The seed of the behavior catalog — currently the Databricks **Genie**
behavior, which shows the pattern: guard on a detected backend, then drive
a Databricks service endpoint through Loki's token provider
(``agent.databricks``). Replication, inter-agent messaging, HTTP ingestion
and serving land here next.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .behavior import LokiBehavior, register
from .engine import AgentResponse

if TYPE_CHECKING:
    from .agent import Loki

__all__ = ["GenieBehavior", "CodeProjectBehavior"]


def _strip_code_fences(text: str) -> str:
    """Pull raw code out of a ```python fenced block when the model adds one."""
    import re

    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return (match.group(1) if match else text).strip()


@register
class CodeProjectBehavior(LokiBehavior):
    """Generate a small Python project from a spec, write it, and run it."""

    name = "code-project"
    description = "Reason out a Python program from a spec, write the project, run it."

    def run(
        self,
        agent: Loki,
        *,
        spec: str,
        name: str = "loki_project",
        run: bool = True,
        workdir: Optional[str] = None,
        timeout: float = 60.0,
        **_: Any,
    ) -> AgentResponse:
        import pathlib
        import subprocess
        import sys
        import tempfile

        # Reason out the code. Low complexity by default — codegen of a small
        # script doesn't need the most expensive model.
        reply = agent.reason(
            f"Write a single self-contained Python program that does the following:\n{spec}",
            system=(
                "You are a senior Python engineer. Output ONLY raw, runnable "
                "Python source for one module — no markdown fences, no prose, "
                "no explanation."
            ),
            complexity="low",
        )
        code = _strip_code_fences(reply.text)

        root = pathlib.Path(workdir or tempfile.mkdtemp(prefix="loki-")) / name
        root.mkdir(parents=True, exist_ok=True)
        entry = root / "main.py"
        entry.write_text(code)
        meta: dict[str, Any] = {"project": str(root), "entry": str(entry)}

        if not run:
            return AgentResponse(text=code, data={"code": code}, meta=meta)

        proc = subprocess.run(
            [sys.executable, str(entry)],
            capture_output=True, text=True, timeout=timeout, cwd=str(root),
        )
        meta["returncode"] = proc.returncode
        if proc.stderr.strip():
            meta["stderr"] = proc.stderr.strip()[:2000]
        return AgentResponse(
            text=proc.stdout.strip(),
            data={"code": code, "stdout": proc.stdout, "stderr": proc.stderr,
                  "returncode": proc.returncode},
            meta=meta,
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
        rows: bool = True,
        **_: Any,
    ) -> AgentResponse:
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
        meta = {
            "space_id": target.space_id,
            "conversation_id": answer.conversation_id,
            "query": answer.query,
            "statement_id": answer.statement_id,
        }
        # When Genie answered with SQL, the result is tabular — attach it.
        tabular = None
        if rows and answer.query:
            try:
                tabular = answer.to_polars()
            except Exception:  # noqa: BLE001 — keep the narrative even if the read fails
                tabular = None
        return AgentResponse(text=answer.text, data=answer, tabular=tabular, meta=meta)
