"""Bundle deploy — resolve config, sync files, deploy all resources."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from yggdrasil.databricks.client import DatabricksClient

from .config import load_bundle, resolve_target
from .resources import deploy_all_resources
from .sync import sync_files

LOGGER = logging.getLogger(__name__)


def deploy(
    bundle_path: Path,
    target_name: str | None = None,
    *,
    client: DatabricksClient | None = None,
) -> int:
    """Deploy a Databricks Asset Bundle.

    Parses the bundle YAML, resolves the target, syncs workspace files,
    and upserts every resource defined under ``resources:``.

    Returns an exit code (0 on success).
    """
    raw = load_bundle(bundle_path)
    bundle_name = (raw.get("bundle") or {}).get("name", bundle_path.parent.name)
    target_cfg, resolved = resolve_target(raw, target_name)

    workspace_cfg = target_cfg.get("workspace") or {}
    host = workspace_cfg.get("host")

    if client is None:
        kwargs: dict[str, Any] = {}
        if host:
            kwargs["host"] = host
        client = DatabricksClient(**kwargs)

    sys.stderr.write(
        f"Deploying bundle {bundle_name!r}"
        f" to {client.base_url.to_string()}"
        f" (target={target_name or 'default'})\n"
    )

    bundle_root = bundle_path.parent

    uploaded = sync_files(client, bundle_root, target_cfg, resolved)
    if uploaded:
        sys.stderr.write(f"  Synced {len(uploaded)} file(s) to workspace\n")

    resources = resolved.get("resources") or {}
    deployed = deploy_all_resources(client, resources)

    total = sum(len(v) for v in deployed.values())
    sys.stderr.write(f"Deploy complete — {total} resource(s) deployed.\n")
    return 0
