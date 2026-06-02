"""Continuous repository optimizer — a propose-only Databricks AI agent.

:class:`RepoOptimizer` walks a repository in the Databricks workspace
(default ``/Workspace/Shared/monteleq``), asks a Foundation Model serving
endpoint to optimize each source file, and writes the suggestions to a
side ``proposals/`` folder. It **never** rewrites the source in place —
the optimized files and a ``REPORT.md`` land under
``<repo>/.optimizer/proposals/<run-id>/`` for a human to review and apply.

Deployed as a periodic Databricks job it becomes a *continuous* optimizer;
run it once locally with :meth:`RepoOptimizer.run`.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Optional

from yggdrasil.databricks.service import DatabricksService

from ..serving.resources import DEFAULT_SERVING_ENDPOINT

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yggdrasil.path import Path


__all__ = [
    "OptimizerConfig",
    "FileProposal",
    "OptimizationReport",
    "RepoOptimizer",
]


LOGGER = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a meticulous senior software engineer doing continuous code "
    "optimization. Given one source file, propose an optimized version that "
    "improves performance, readability, and correctness WITHOUT changing the "
    "file's observable behavior or public API. Preserve language, imports that "
    "are still needed, comments that carry intent, and formatting conventions. "
    "If the file is already good, say so and return it unchanged.\n\n"
    "Respond with ONLY a single JSON object (no markdown fences, no prose) of "
    'the shape: {"summary": str, "findings": [str, ...], '
    '"optimized_code": str, "changed": bool}. "optimized_code" is the FULL '
    "file contents after your changes; set \"changed\" to false (and echo the "
    "original code) when no improvement is warranted."
)

#: File extensions the optimizer treats as reviewable source.
DEFAULT_SUFFIXES = (
    ".py", ".sql", ".scala", ".r", ".sh", ".ts", ".tsx", ".js", ".jsx",
    ".java", ".go", ".rs", ".rb", ".c", ".cc", ".cpp", ".h", ".hpp",
)

#: Directory names never descended into (build noise + the optimizer's own output).
DEFAULT_EXCLUDE_DIRS = (
    ".optimizer", ".git", "__pycache__", ".ipynb_checkpoints",
    "node_modules", ".venv", "venv", "dist", "build", ".mypy_cache",
)


@dataclass
class OptimizerConfig:
    """Everything the optimizer needs for one repository."""

    repo_path: str = "/Workspace/Shared/monteleq"
    endpoint_name: str = DEFAULT_SERVING_ENDPOINT
    #: Where proposals land. ``None`` → ``<repo>/.optimizer/proposals/<run-id>``.
    proposals_path: Optional[str] = None
    suffixes: tuple = DEFAULT_SUFFIXES
    exclude_dirs: tuple = DEFAULT_EXCLUDE_DIRS
    max_file_bytes: int = 96_000
    max_files: int = 50
    max_tokens: int = 8192
    temperature: float = 0.0
    system_prompt: str = SYSTEM_PROMPT


@dataclass
class FileProposal:
    """The optimizer's verdict on a single file."""

    rel_path: str
    summary: str = ""
    findings: list = field(default_factory=list)
    changed: bool = False
    optimized_code: Optional[str] = None
    skipped: bool = False
    reason: str = ""


@dataclass
class OptimizationReport:
    """Outcome of one optimizer pass over a repository."""

    repo_path: str
    proposals_path: str
    run_id: str
    files_scanned: int = 0
    proposals: list = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def changed(self) -> list:
        return [p for p in self.proposals if p.changed and not p.skipped]


class RepoOptimizer(DatabricksService):
    """Propose-only AI optimizer for a workspace repository."""

    def __init__(self, client=None, config: Optional[OptimizerConfig] = None):
        super().__init__(client=client)
        self.config = config if config is not None else OptimizerConfig()

    # ------------------------------------------------------------------ #
    # Resources
    # ------------------------------------------------------------------ #
    @property
    def serving(self):
        """The model-serving service backing this optimizer."""
        return self.client.ai.serving

    def repo(self) -> "Path":
        """The repository root as a workspace :class:`Path`."""
        return self.client.path(self.config.repo_path)

    def iter_source_files(self) -> Iterator["Path"]:
        """Yield reviewable source files under the repo, newest-relevant first.

        Skips directories in :attr:`OptimizerConfig.exclude_dirs`, files whose
        suffix isn't in :attr:`OptimizerConfig.suffixes`, and anything larger
        than :attr:`OptimizerConfig.max_file_bytes`.
        """
        root = self.repo().full_path().rstrip("/")
        for child in self.repo().ls(recursive=True):
            if not child.is_file():
                continue
            if child.suffix.lower() not in self.config.suffixes:
                continue
            full = child.full_path()
            rel = full[len(root):].lstrip("/")
            if any(part in self.config.exclude_dirs for part in rel.split("/")):
                continue
            if child.size > self.config.max_file_bytes:
                LOGGER.info("optimizer: skipping %s (%d bytes > limit)", rel, child.size)
                continue
            yield child

    # ------------------------------------------------------------------ #
    # Per-file optimization
    # ------------------------------------------------------------------ #
    def optimize_file(self, path: "Path") -> FileProposal:
        """Ask the serving endpoint to optimize one file, parse its verdict."""
        root = self.repo().full_path().rstrip("/")
        rel = path.full_path()[len(root):].lstrip("/")
        try:
            source = path.read_text()
        except Exception as exc:  # unreadable / binary — record and move on
            return FileProposal(rel_path=rel, skipped=True, reason=f"unreadable: {exc}")

        prompt = (
            f"File: {rel}\nLanguage (by extension): {path.suffix.lstrip('.') or 'unknown'}\n"
            f"```\n{source}\n```"
        )
        result = self.serving.chat(
            [{"role": "user", "content": prompt}],
            endpoint_name=self.config.endpoint_name,
            system=self.config.system_prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        payload = _parse_json_object(result.content)
        if payload is None:
            return FileProposal(
                rel_path=rel,
                skipped=True,
                reason="model did not return parseable JSON",
                summary=result.content[:500],
            )
        optimized = payload.get("optimized_code")
        changed = bool(payload.get("changed")) and isinstance(optimized, str) and optimized != source
        return FileProposal(
            rel_path=rel,
            summary=str(payload.get("summary", "")),
            findings=list(payload.get("findings", []) or []),
            changed=changed,
            optimized_code=optimized if changed else None,
        )

    # ------------------------------------------------------------------ #
    # Full pass
    # ------------------------------------------------------------------ #
    def run(self) -> OptimizationReport:
        """Optimize every eligible file and write proposals + a report.

        Returns the :class:`OptimizationReport`. Source files are never
        modified — all output goes under the proposals folder.
        """
        run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        proposals_root = (
            self.config.proposals_path
            or f"{self.config.repo_path.rstrip('/')}/.optimizer/proposals/{run_id}"
        )
        report = OptimizationReport(
            repo_path=self.config.repo_path,
            proposals_path=proposals_root,
            run_id=run_id,
            started_at=time.time(),
        )
        LOGGER.info("optimizer: pass %s over %s → %s", run_id, self.config.repo_path, proposals_root)

        files_dir = self.client.path(f"{proposals_root}/files")
        for path in self.iter_source_files():
            if report.files_scanned >= self.config.max_files:
                LOGGER.info("optimizer: hit max_files=%d, stopping pass", self.config.max_files)
                break
            report.files_scanned += 1
            try:
                proposal = self.optimize_file(path)
            except Exception as exc:
                root = self.repo().full_path().rstrip("/")
                rel = path.full_path()[len(root):].lstrip("/")
                LOGGER.warning("optimizer: %s failed: %s", rel, exc)
                proposal = FileProposal(rel_path=rel, skipped=True, reason=str(exc))
            report.proposals.append(proposal)

            if proposal.changed and proposal.optimized_code is not None:
                dest = files_dir.joinpath(proposal.rel_path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(proposal.optimized_code)
                LOGGER.info("optimizer: proposed changes for %s", proposal.rel_path)

        report.finished_at = time.time()
        self.client.path(f"{proposals_root}/REPORT.md").write_text(self._render_report(report))
        LOGGER.info(
            "optimizer: pass %s done — %d scanned, %d with proposed changes",
            run_id, report.files_scanned, len(report.changed),
        )
        return report

    # ------------------------------------------------------------------ #
    # Report rendering
    # ------------------------------------------------------------------ #
    def _render_report(self, report: OptimizationReport) -> str:
        lines = [
            f"# Optimization report — {report.run_id}",
            "",
            f"- Repository: `{report.repo_path}`",
            f"- Endpoint: `{self.config.endpoint_name}`",
            f"- Files scanned: {report.files_scanned}",
            f"- Files with proposed changes: {len(report.changed)}",
            f"- Optimized files written under: `{report.proposals_path}/files/`",
            "",
            "> Propose-only: source files were not modified. Review the files "
            "above and apply what you agree with.",
            "",
            "## Files",
            "",
        ]
        for p in report.proposals:
            status = "skipped" if p.skipped else ("changed" if p.changed else "no change")
            lines.append(f"### `{p.rel_path}` — {status}")
            if p.reason:
                lines.append(f"- _{p.reason}_")
            if p.summary:
                lines.append(f"- {p.summary}")
            for finding in p.findings:
                lines.append(f"  - {finding}")
            lines.append("")
        return "\n".join(lines)


def _parse_json_object(text: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from a model reply.

    Strips ```` ```json ```` fences, then falls back to slicing the first
    ``{`` … last ``}`` span — Foundation Models occasionally wrap or pad
    their JSON despite instructions.
    """
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```", 2)[1] if stripped.count("```") >= 2 else stripped
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip("`").strip()
    try:
        obj = json.loads(stripped)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(stripped[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None
