"""Workspace file sync — upload local files to the Databricks workspace."""
from __future__ import annotations

import fnmatch
import logging
from pathlib import Path
from typing import Any, Sequence

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.fs.workspace_path import WorkspacePath

LOGGER = logging.getLogger(__name__)


def sync_files(
    client: DatabricksClient,
    bundle_root: Path,
    target_cfg: dict[str, Any],
    resolved: dict[str, Any],
) -> list[str]:
    """Sync local files to the workspace per the target's sync config.

    Returns a list of workspace paths that were uploaded.
    """
    sync = target_cfg.get("sync") or {}
    sync_paths: list[str] = sync.get("paths") or ["."]
    include_patterns: list[str] = sync.get("include") or []
    exclude_patterns: list[str] = sync.get("exclude") or []

    notebook_root = _resolve_notebook_root(resolved)
    if notebook_root is None:
        LOGGER.warning("No notebook_root variable found — skipping file sync")
        return []

    uploaded: list[str] = []

    for base in sync_paths:
        base_path = bundle_root / base if base != "." else bundle_root
        if not base_path.is_dir():
            continue

        files = _collect_files(base_path, include_patterns, exclude_patterns)

        for local_file in files:
            rel = local_file.relative_to(base_path)
            workspace_dest = f"{notebook_root}/{rel}"

            LOGGER.info("Uploading %s → %s", local_file, workspace_dest)
            wp = WorkspacePath.from_(workspace_dest, service=client.workspaces)
            wp.parent.mkdir(parents=True, exist_ok=True)
            wp.write_bytes(local_file.read_bytes(), overwrite=True)
            uploaded.append(workspace_dest)

    return uploaded


def _resolve_notebook_root(resolved: dict[str, Any]) -> str | None:
    """Extract the notebook root from resolved variables or task notebook paths."""
    variables = resolved.get("variables") or {}
    for name, defn in variables.items():
        if name == "notebook_root":
            if isinstance(defn, dict):
                return defn.get("default")
            return str(defn) if defn is not None else None

    resources = resolved.get("resources") or {}
    for _job_key, job_cfg in (resources.get("jobs") or {}).items():
        for task_cfg in job_cfg.get("tasks") or []:
            nb = task_cfg.get("notebook_task") or {}
            nb_path = nb.get("notebook_path")
            if nb_path and "/" in nb_path:
                parts = nb_path.rsplit("/", 1)
                return parts[0]

    return None


def _collect_files(
    root: Path,
    include: Sequence[str],
    exclude: Sequence[str],
) -> list[Path]:
    """Collect files under *root* matching include/exclude patterns."""
    if not include:
        candidates = [p for p in root.rglob("*") if p.is_file()]
    else:
        seen: set[Path] = set()
        candidates = []
        for pattern in include:
            for p in root.glob(pattern):
                if p.is_file() and p not in seen:
                    seen.add(p)
                    candidates.append(p)

    if exclude:
        candidates = [
            p for p in candidates
            if not any(
                fnmatch.fnmatch(p.name, ex) or fnmatch.fnmatch(str(p.relative_to(root)), ex)
                for ex in exclude
            )
        ]

    return sorted(candidates)
