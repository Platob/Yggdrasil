"""Bundle config parser — load and resolve ``databricks.yml``."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

_VAR_RE = re.compile(r"\$\{var\.(\w+)\}")


def load_bundle(path: Path) -> dict[str, Any]:
    """Load a ``databricks.yml`` file and return the raw dict."""
    import yaml

    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text)


def resolve_target(
    raw: dict[str, Any],
    target_name: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve a target from the bundle config.

    Returns ``(target_config, resolved_bundle)`` where variable
    references (``${var.xxx}``) have been substituted throughout.
    """
    targets = raw.get("targets") or {}

    if target_name is None:
        for name, cfg in targets.items():
            if cfg.get("default"):
                target_name = name
                break
        if target_name is None and len(targets) == 1:
            target_name = next(iter(targets))
        if target_name is None and not targets:
            target_name = "__default__"

    target_cfg = targets.get(target_name, {})
    if target_cfg is None:
        target_cfg = {}

    variables = _collect_variables(raw, target_cfg)
    resolved = _substitute_vars(raw, variables)
    resolved_target = _substitute_vars(target_cfg, variables)

    return resolved_target, resolved


def _collect_variables(
    raw: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, str]:
    """Merge bundle-level variable defaults with target-level overrides and env vars."""
    variables: dict[str, str] = {}

    for name, defn in (raw.get("variables") or {}).items():
        if isinstance(defn, dict):
            val = defn.get("default")
        else:
            val = defn
        if val is not None:
            variables[name] = str(val)

    for name, defn in (target.get("variables") or {}).items():
        if isinstance(defn, dict):
            val = defn.get("default", defn.get("value"))
        else:
            val = defn
        if val is not None:
            variables[name] = str(val)

    for name in list(variables):
        env_key = f"BUNDLE_VAR_{name}"
        env_val = os.environ.get(env_key)
        if env_val is not None:
            variables[name] = env_val

    return variables


def _substitute_vars(obj: Any, variables: dict[str, str]) -> Any:
    """Recursively substitute ``${var.xxx}`` in strings."""
    if isinstance(obj, str):
        return _VAR_RE.sub(lambda m: variables.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, dict):
        return {k: _substitute_vars(v, variables) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_vars(item, variables) for item in obj]
    return obj
