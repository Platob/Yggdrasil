"""Databricks Assistant ("Genie") configuration for ``ygg[databricks]``.

The in-product Databricks Assistant — the **Genie** that writes and runs
code in notebooks for the user — works best when it routes every operation
through the ygg Python API instead of a terminal. Serverless notebooks and
jobs cannot shell out (no ``%sh`` / ``!cmd`` / ``ygg``/``databricks`` CLI),
and ygg is already on the **pre-built serverless image** seeded by
``ygg databricks deploy``. These files teach the Assistant exactly that.

This package is the **single source of truth** for the three Assistant
artifacts:

- **assistant guidance** — :func:`workspace_instructions` (workspace-wide)
  and :func:`user_instructions` (per-user preferences),
- **skills** — :func:`skills`, the per-task markdown the Assistant routes to.

``ygg databricks deploy`` deploys them into a workspace via :func:`deploy`:
the workspace bundle (guidance + skills) under ``/Workspace/Shared`` and a
per-user bundle under ``/Workspace/Users/<me>``, plus a best-effort attempt
at any live Assistant-settings API the SDK exposes.
"""
from __future__ import annotations

import importlib.resources as ir
from typing import Any

__all__ = [
    "skills",
    "skill_names",
    "workspace_instructions",
    "user_instructions",
    "deploy",
    "WORKSPACE_ASSISTANT_DIR",
    "USER_ASSISTANT_DIR",
]

# Where ``deploy`` lands the bundles. The Assistant reads these folders when
# pointed at them in Settings; keeping them under a single ``.ygg/assistant``
# root makes the whole bundle easy to find, diff, and re-seed.
WORKSPACE_ASSISTANT_DIR = "/Workspace/Shared/.ygg/assistant"
USER_ASSISTANT_DIR = "/Workspace/Users/<me>/.ygg/assistant"


def _read(rel: str) -> str:
    return (ir.files(__name__) / rel).read_text(encoding="utf-8")


def skill_names() -> list[str]:
    """Sorted ``*.md`` filenames under ``skills/``."""
    skills_dir = ir.files(__name__) / "skills"
    return sorted(p.name for p in skills_dir.iterdir() if p.name.endswith(".md"))


def skills() -> dict[str, str]:
    """Map ``<skill>.md`` → markdown body for every packaged skill."""
    skills_dir = ir.files(__name__) / "skills"
    return {
        name: (skills_dir / name).read_text(encoding="utf-8")
        for name in skill_names()
    }


def workspace_instructions() -> str:
    """Workspace-wide Assistant guidance (paste into workspace instructions)."""
    return _read("workspace_instructions.md")


def user_instructions() -> str:
    """Per-user Assistant preferences (paste into the user instructions slot)."""
    return _read("user_instructions.md")


def _bundle() -> "list[tuple[str, str]]":
    """``(workspace_relative_path, markdown)`` for the whole bundle.

    The skills ship to **both** the workspace and the user skill folders so
    the Assistant finds them whichever scope it routes through; the two
    guidance files go to their respective scopes.
    """
    sk = skills()
    out: "list[tuple[str, str]]" = [
        (f"{WORKSPACE_ASSISTANT_DIR}/workspace_instructions.md", workspace_instructions()),
        (f"{USER_ASSISTANT_DIR}/user_instructions.md", user_instructions()),
    ]
    for root in (WORKSPACE_ASSISTANT_DIR, USER_ASSISTANT_DIR):
        out.extend((f"{root}/skills/{name}", body) for name, body in sk.items())
    return out


def _try_assistant_api(client: Any) -> str:
    """Best-effort push of the guidance to a live Assistant-settings API.

    There is no stable, public API to set the Databricks Assistant's custom
    instructions as of this build, so this stays guarded and never raises —
    it returns a human-readable status the seed prints. If a future SDK
    grows the surface, wire it here.
    """
    try:
        w = client.workspace_client()
    except Exception as exc:  # pragma: no cover - connectivity already gated
        return f"skipped ({exc})"
    # Probe for a plausible custom-instructions surface without inventing a
    # key: if the SDK never grows one we report a clean skip rather than fail.
    for attr in ("assistant", "ai_assistant", "genie_settings"):
        svc = getattr(w, attr, None)
        setter = getattr(svc, "update_custom_instructions", None) or getattr(
            svc, "set_instructions", None,
        )
        if callable(setter):
            try:
                setter(workspace_instructions())
                return f"updated via workspace.{attr}"
            except Exception as exc:
                return f"attempted workspace.{attr}, skipped ({exc})"
    return "skipped (no public Assistant-settings API in this SDK)"


def deploy(client: Any, *, check: bool = False, overwrite: bool = True) -> "dict[str, Any]":
    """Deploy (or, with ``check``, verify) the Assistant bundle in a workspace.

    Writes the workspace guidance + skills under
    :data:`WORKSPACE_ASSISTANT_DIR` and the per-user guidance + skills under
    :data:`USER_ASSISTANT_DIR`, then attempts a live Assistant-settings push.

    With ``overwrite=False`` (append semantics) a file that already exists is
    left untouched — only missing files are written — and is reported under
    ``skipped`` rather than ``uploaded``.

    Returns ``{"uploaded": [paths], "missing": [paths], "skipped": [paths], "api": status}``.
    """
    result: "dict[str, Any]" = {"uploaded": [], "missing": [], "skipped": [], "api": None}
    for remote, body in _bundle():
        p = client.path(remote)
        if check:
            (result["uploaded"] if p.exists() else result["missing"]).append(
                p.full_path()
            )
            continue
        if not overwrite and p.exists():
            result["skipped"].append(p.full_path())
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
        result["uploaded"].append(p.full_path())
    if not check:
        result["api"] = _try_assistant_api(client)
    return result
