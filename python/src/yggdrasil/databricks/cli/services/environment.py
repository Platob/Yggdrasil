"""``ygg databricks environment`` — build / get-or-install the reusable ygg base
environment(s) from wheels in the workspace.

The environment machinery in :mod:`yggdrasil.databricks.job.wheel` builds ygg's
whole transitive dependency closure as wheels and persists, **per Python**, a
base environment under ``/Workspace/Shared/environment``::

    <proj>/<proj>-<version>-py3XX.yml               serverless base_environment
    <proj>/<proj>-<version>-py3XX.requirements.txt   classic-cluster requirements

listing wheels from the shared pypi registry (``/Workspace/Shared/pypi``), so
ygg jobs and clusters install with zero PyPI access. This command surfaces
that on its own — the same step ``ygg databricks seed`` runs::

    ygg databricks environment                 # all supported Pythons (3.10–3.13)
    ygg databricks environment --current       # only the local interpreter's Python
    ygg databricks environment --rebuild       # force a fresh wheel-closure build
    ygg databricks environment list            # list the deployed environment files

**Get-or-install:** an existing environment is reused as-is; ``--rebuild`` forces
a fresh wheel closure + rewrite.
"""
from __future__ import annotations

from typing import Any


class EnvironmentCommand:

    @classmethod
    def register(cls, subparsers: Any) -> None:
        parser = subparsers.add_parser(
            "environment", aliases=["env", "environments"],
            help="Build / get-or-install the reusable ygg base environment(s) from wheels.",
        )
        parser.add_argument("--workspace-dir", dest="workspace_dir", default=None,
                            help="Environment root (default: /Workspace/Shared/environment).")
        parser.add_argument("--rebuild", action="store_true",
                            help="Force a fresh wheel-closure build + rewrite even if present.")
        parser.add_argument("--current", dest="current", action="store_true",
                            help="Only the local interpreter's Python (default: all supported, 3.10–3.13).")
        sub = parser.add_subparsers(dest="environment_action")

        ls = sub.add_parser("list", help="List the deployed base environment files.")
        ls.add_argument("--workspace-dir", dest="workspace_dir", default=None,
                        help="Environment root (default: /Workspace/Shared/environment).")
        ls.set_defaults(handler=cls._list)

        parser.set_defaults(handler=cls._ensure)

    # -- handlers --------------------------------------------------------
    @classmethod
    def _ensure(cls, args: Any, build_client: Any) -> int:
        """Bare ``environment`` — get-or-install the base environment(s) from
        wheels (build the closure once, reuse it next time; ``--rebuild`` forces)."""
        from yggdrasil.cli import style
        from yggdrasil.databricks.job import wheel as whl

        client = build_client(args)
        workspace_dir = args.workspace_dir or whl.WORKSPACE_ENV_DIR
        pythons = [None] if args.current else list(whl.SUPPORTED_PYTHONS)
        plural = "s" if len(pythons) > 1 else ""
        with style.Spinner(
            f"building {len(pythons)} base environment{plural} (wheel closure into shared pypi)…"
        ):
            envs = whl.ensure_environments(
                client, versions=pythons, workspace_dir=workspace_dir, rebuild=args.rebuild,
            )
        for env in envs:
            style.out(
                f"    {style.dim(env['env_name'])}  {env['n_wheels']} wheels  "
                f"{style.dim('dir')} {env['env_dir']}\n"
            )
            style.out(f"          {style.dim('serverless')} {env['serverless']}\n")
            style.out(f"          {style.dim('cluster')}    {env['cluster']}\n")
        style.ok(
            f"{len(envs)} base environment(s) ready (serverless + cluster, zero-PyPI)"
        )
        return 0

    @classmethod
    def _list(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style
        from yggdrasil.databricks.job import wheel as whl

        client = build_client(args)
        workspace_dir = args.workspace_dir or whl.WORKSPACE_ENV_DIR
        paths = whl.deployed_environments(client, workspace_dir=workspace_dir)
        if not paths:
            style.warn(f"no base environment files under {workspace_dir}")
            return 1
        for path in paths:
            style.out(f"    {path}\n")
        style.ok(f"{len(paths)} environment file(s)")
        return 0
