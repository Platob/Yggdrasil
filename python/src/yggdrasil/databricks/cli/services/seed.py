"""``ygg databricks seed`` — check (and provision) the workspace prerequisites
for running ygg on Databricks.

One command to answer "is this workspace ready, and if not, make it ready":

    ygg databricks seed            # provision anything missing, then report
    ygg databricks seed --check    # read-only readiness report (CI gate)

It walks four areas:

- **config**      — connectivity, host, current user, default catalog/schema.
- **wheels**      — the versioned ygg image wheel in the workspace registry.
- **environments**— the reusable ``yellow`` base environment ygg jobs run
  under, persisted to the workspace as ``yellow.env.yaml`` (serverless
  ``base_environment``) and ``yellow.requirements.txt`` (classic-cluster
  ``Library(requirements=...)``).
- **warehouses**  — a default SQL warehouse to execute statements against.

In the default (seed) mode it builds/uploads the wheel, assembles and writes
the environment files, and ensures a default warehouse exists. With ``--check``
it touches nothing and exits non-zero when something is missing — so a
pipeline can gate on ``ygg databricks seed --check``.
"""
from __future__ import annotations

from typing import Any


class SeedCommand:

    @classmethod
    def register(cls, subparsers: Any) -> None:
        parser = subparsers.add_parser(
            "seed",
            help="Check (and provision) workspace prerequisites: wheels, environments, default warehouse, config.",
        )
        parser.add_argument("--check", action="store_true",
                            help="Read-only: report readiness, create/upload nothing (exits 1 if anything is missing).")
        parser.add_argument("--workspace-dir", dest="workspace_dir", default=None,
                            help="PyPI-like registry root (default: /Workspace/Shared/pypi).")
        parser.add_argument("--rebuild", action="store_true",
                            help="Force a fresh wheel build even if the version is already deployed.")
        parser.add_argument("--all-versions", dest="all_versions", action="store_true",
                            help="Seed a wheel + environment for every supported Python (3.10–3.13).")
        parser.set_defaults(handler=cls._seed)

    @classmethod
    def _seed(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        check = args.check
        client = build_client(args)
        style.out(f"\n  {style.bold('ygg databricks seed')}  "
                  f"{style.dim('(' + ('check' if check else 'provision') + ' mode)')}\n\n")

        ok = True  # flips false on any missing prerequisite or step error

        # -- config ------------------------------------------------------
        # Connectivity is the gate: nothing else is reachable without it.
        style.info("config")
        try:
            user = client.workspace_client().current_user.me().user_name
            style.out(f"    {style.dim('host')}      {client.base_url}\n")
            style.out(f"    {style.dim('user')}      {user}\n")
            style.out(f"    {style.dim('catalog')}   {client.catalog_name or style.dim('(unset)')}\n")
            style.out(f"    {style.dim('schema')}    {client.schema_name or style.dim('(unset)')}\n")
            try:
                style.out(f"    {style.dim('workspace')} {client.get_workspace_id()}\n")
            except Exception:
                pass
            style.ok("config reachable")
        except Exception as exc:
            style.fail(f"cannot reach workspace: {exc}")
            return 1

        from yggdrasil.databricks.job import wheel as whl
        workspace_dir = (args.workspace_dir or whl.WORKSPACE_PYPI_DIR).rstrip("/")
        import importlib.metadata as ilmd
        version = ilmd.version("ygg")

        # -- wheels ------------------------------------------------------
        style.info("wheels")
        try:
            dist_dir = f"{workspace_dir}/ygg"
            if check:
                existing = whl.deployed_wheels(
                    client, "ygg", version, workspace_dir=dist_dir, dist_only=True,
                )
                if existing:
                    for path in existing:
                        style.out(f"    {style.dim('found')}  {path}\n")
                    style.ok(f"ygg {version} wheel deployed")
                else:
                    style.warn(f"ygg {version} wheel not deployed under {dist_dir}")
                    ok = False
            else:
                if args.all_versions:
                    paths = whl.ensure_ygg_wheels(client, workspace_dir=workspace_dir, rebuild=args.rebuild)
                else:
                    paths = whl.ensure_ygg_wheel(client, workspace_dir=workspace_dir, rebuild=args.rebuild)
                for path in paths:
                    style.out(f"    {style.dim('wheel')}  {path}\n")
                style.ok(f"ygg {version} wheel ready ({len(paths)})")
        except Exception as exc:
            style.fail(f"wheel step failed: {exc}")
            ok = False

        # -- environments ------------------------------------------------
        style.info("environments")
        try:
            if check:
                # The serverless version is chosen off the local Python and the
                # deps come from installed metadata (no build); then we verify the
                # reusable environment files were actually written (read-only).
                env_version = whl.serverless_environment_version()
                deps = whl.ygg_runtime_dependencies()
                style.out(f"    {style.dim('env version')}  {env_version} {style.dim('(matches local Python)')}\n")
                style.out(f"    {style.dim('runtime deps')} {len(deps)}\n")
                persisted = whl.deployed_environments(client)
                if persisted:
                    for path in persisted[:6]:
                        style.out(f"    {style.dim('found')}  {path}\n")
                    style.ok("base environment files present (serverless + cluster)")
                else:
                    style.warn(f"no base environment files under {whl.WORKSPACE_ENV_DIR}")
                    ok = False
            else:
                # rebuild=False: reuse the wheel just built above.
                if args.all_versions:
                    envs = whl.ygg_environments(client, workspace_dir=workspace_dir, rebuild=False)
                else:
                    envs = [whl.ygg_environment(client, workspace_dir=workspace_dir, rebuild=False)]
                for env in envs:
                    style.out(f"    {style.dim('env')}  {env.environment_key}  "
                              f"{style.dim('v' + str(env.spec.environment_version))}  "
                              f"{style.dim(str(len(env.spec.dependencies)) + ' deps')}\n")
                style.ok(f"{len(envs)} JobEnvironment(s) ready")

                # Persist the canonical reusable "yellow" base environment so jobs
                # can reference it by path (serverless ``base_environment``) and
                # classic clusters can install from it (``Library(requirements=...)``).
                # Derived from the local-matched "default" image (envs[0]).
                base = envs[0]
                env_yaml = whl.ensure_named_environment(
                    client, "yellow",
                    dependencies=base.spec.dependencies,
                    environment_version=base.spec.environment_version,
                )
                reqs = whl.ensure_cluster_requirements(
                    client, "yellow", dependencies=base.spec.dependencies,
                )
                style.out(f"    {style.dim('serverless')} {env_yaml}\n")
                style.out(f"    {style.dim('cluster')}    {reqs}\n")
                style.ok("base environment 'yellow' written (serverless + cluster)")
        except Exception as exc:
            style.fail(f"environment step failed: {exc}")
            ok = False

        # -- warehouses --------------------------------------------------
        style.info("warehouses")
        try:
            if check:
                warehouses = list(client.warehouses.list_warehouses())
                if not warehouses:
                    style.warn("no SQL warehouses in the workspace")
                    ok = False
                else:
                    for wh in warehouses[:10]:
                        style.out(f"    {style.dim('•')} {wh.warehouse_name}  {style.dim(str(wh.warehouse_id))}\n")
                    if len(warehouses) > 10:
                        style.out(f"    {style.dim('… +' + str(len(warehouses) - 10) + ' more')}\n")
                    style.ok(f"{len(warehouses)} warehouse(s) available")
            else:
                wh = client.warehouses.find_default(raise_error=False)
                if wh is None:
                    style.warn("could not resolve or create a default warehouse")
                    ok = False
                else:
                    state = getattr(wh.state, "value", None) or getattr(wh.state, "name", None) or str(wh.state)
                    style.out(f"    {style.dim('default')}  {wh.warehouse_name}  {style.dim(str(wh.warehouse_id))}\n")
                    style.ok(f"default warehouse ready ({state})")
        except Exception as exc:
            style.fail(f"warehouse step failed: {exc}")
            ok = False

        # -- summary -----------------------------------------------------
        style.out("\n")
        if check:
            if ok:
                style.ok("all prerequisites present")
                return 0
            style.warn("prerequisites missing — run `ygg databricks seed` to provision")
            return 1
        if ok:
            style.ok("workspace seeded")
            return 0
        style.warn("seed completed with warnings (see above)")
        return 1
