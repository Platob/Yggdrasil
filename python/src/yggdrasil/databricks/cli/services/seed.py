"""``ygg databricks seed`` — check (and provision) the workspace prerequisites
for running ygg on Databricks.

One command to answer "is this workspace ready, and if not, make it ready":

    ygg databricks seed            # provision anything missing, then report
    ygg databricks seed --check    # read-only readiness report (CI gate)
    ygg databricks seed --overwrite  # rebuild every wheel + the env from scratch, then end

It walks five areas:

- **config**      — connectivity, host, current user, default catalog/schema.
- **wheels**      — the versioned ygg image wheel in the workspace registry.
- **environments**— the version-pinned base environments ygg jobs run under,
  persisted under ``/Workspace/Shared/environments`` — **one pair per Python**:
  ``ygg-<version>-py3XX.yml`` (serverless ``base_environment``) and
  ``ygg-<version>-py3XX.requirements.txt`` (classic-cluster
  ``Library(requirements=...)``). Both list only **built wheels in the workspace
  pypi registry**, so the runtime installs with zero PyPI access. ``--all-versions``
  (and ``--overwrite``) writes the pair for every supported Python (3.10–3.13).
- **warehouses**  — a default SQL warehouse to execute statements against.
- **pools**       — the default Light / Medium / Heavy Yggdrasil instance pools
  (AWS r5d memory-optimized, local NVMe), each preloading the local-Python DBR
  runtime so pool-backed clusters attach warm against the seeded zero-PyPI wheel
  bundle. Lazy by default (no idle nodes → no cost until attached). Skip with
  ``--no-pools``.

In the default (seed) mode it builds/uploads the wheel, assembles and writes
the environment files, and ensures a default warehouse exists. With ``--check``
it touches nothing and exits non-zero when something is missing — so a
pipeline can gate on ``ygg databricks seed --check``. With ``--overwrite`` it
forces a fresh rebuild of every wheel (all supported Pythons + the dependency
bundle), rewrites the environment files, and **ends** — skipping the warehouse
step (a focused "rebuild the image from scratch" command).
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
        parser.add_argument("--overwrite", action="store_true",
                            help="Rebuild every wheel (all Pythons + the bundle) from scratch and rewrite "
                                 "the environment files, then end (skips the warehouse + pools steps).")
        parser.add_argument("--no-pools", dest="no_pools", action="store_true",
                            help="Skip the default Light/Medium/Heavy instance pools step.")
        parser.set_defaults(handler=cls._seed)

    @classmethod
    def _seed(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        check = args.check
        # --overwrite forces a full from-scratch rebuild (every Python + the
        # bundle) and rewrites the environment files; --check stays read-only and
        # wins if both are given.
        overwrite = args.overwrite and not check
        rebuild = overwrite or args.rebuild
        all_versions = overwrite or args.all_versions
        mode = "check" if check else ("overwrite" if overwrite else "provision")
        client = build_client(args)
        style.out(f"\n  {style.bold('ygg databricks seed')}  "
                  f"{style.dim('(' + mode + ' mode)')}\n\n")

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
                if all_versions:
                    paths = whl.ensure_ygg_wheels(client, workspace_dir=workspace_dir, rebuild=rebuild)
                else:
                    paths = whl.ensure_ygg_wheel(client, workspace_dir=workspace_dir, rebuild=rebuild)
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
                # Read-only: verify the reusable environment files were written.
                persisted = whl.deployed_environments(client)
                if persisted:
                    for path in persisted[:6]:
                        style.out(f"    {style.dim('found')}  {path}\n")
                    style.ok("base environment files present (serverless + cluster)")
                else:
                    style.warn(f"no base environment files under {whl.WORKSPACE_ENV_DIR}")
                    ok = False
            else:
                # Persist the version-pinned base environments under
                # /Workspace/Shared/environments so jobs can reference them by path
                # (serverless ``base_environment``) and classic clusters can install
                # from them (``Library(requirements=...)``). One pair of files per
                # Python — a serverless ``ygg-<version>-py3XX.yml`` and a cluster
                # ``ygg-<version>-py3XX.requirements.txt`` — each listing that
                # Python's whole transitive closure built as wheels into the
                # workspace pypi registry (ensure_bundle), so the runtime installs
                # with zero PyPI access. With --all-versions/--overwrite this covers
                # every supported Python (3.10–3.13); otherwise just the local one.
                pythons = list(whl.SUPPORTED_PYTHONS) if all_versions else [None]
                for py in pythons:
                    bundle = whl.ensure_bundle(
                        client, "ygg", python=py, workspace_dir=workspace_dir, rebuild=rebuild,
                    )
                    key = whl.environment_key_for(py)
                    env_name = f"ygg-{version}-{key}"
                    env_yaml = whl.ensure_named_environment(
                        client, env_name, dependencies=bundle,
                        environment_version=whl.serverless_environment_version(py),
                        filename=f"{env_name}.yml",
                    )
                    reqs = whl.ensure_cluster_requirements(client, env_name, dependencies=bundle)
                    style.out(
                        f"    {style.dim(key)}  {len(bundle)} wheels  "
                        f"{style.dim('serverless')} {env_yaml}\n"
                    )
                    style.out(f"          {style.dim('cluster')}    {reqs}\n")
                style.ok(
                    f"base environments written for {len(pythons)} Python version(s) "
                    f"(serverless + cluster)"
                )
        except Exception as exc:
            style.fail(f"environment step failed: {exc}")
            ok = False

        # --overwrite is a focused "rebuild the image" command: stop after the
        # wheels + environments are rewritten, before the warehouse step.
        if overwrite:
            style.out("\n")
            if ok:
                style.ok("wheels rebuilt and environment rewritten")
                return 0
            style.warn("overwrite completed with warnings (see above)")
            return 1

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

        # -- instance pools ----------------------------------------------
        # The default Light / Medium / Heavy pools (r5d, local-Python DBR
        # preloaded) so ygg's pool-backed compute attaches fast against the
        # seeded zero-PyPI wheel bundle. Lazy by default — no idle nodes, no
        # cost until a cluster attaches. ``--no-pools`` opts out.
        if not args.no_pools:
            style.info("pools")
            try:
                from yggdrasil.databricks.compute.instance_pool import DEFAULT_POOL_TIERS

                pools_svc = client.compute.instance_pools
                if check:
                    missing: list[str] = []
                    for tier in DEFAULT_POOL_TIERS:
                        name = tier.pool_name()
                        if pools_svc.find(name=name) is None:
                            style.warn(f"instance pool {name!r} not provisioned")
                            missing.append(name)
                        else:
                            style.out(f"    {style.dim('found')}  {name}  {style.dim(tier.node_type_id)}\n")
                    if missing:
                        ok = False
                    else:
                        style.ok(f"{len(DEFAULT_POOL_TIERS)} default instance pool(s) present")
                else:
                    pools = pools_svc.seed_default_pools()
                    for pool in pools:
                        style.out(
                            f"    {style.dim('pool')}   {pool.instance_pool_name}  "
                            f"{style.dim(pool.node_type_id or '')}\n"
                        )
                    style.ok(
                        f"{len(pools)} default instance pool(s) ready "
                        f"(Light/Medium/Heavy, r5d)"
                    )
            except Exception as exc:
                style.fail(f"pools step failed: {exc}")
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
