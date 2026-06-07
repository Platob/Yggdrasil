"""``ygg databricks seed`` — check (and provision) the workspace prerequisites
for running ygg on Databricks.

One command to answer "is this workspace ready, and if not, make it ready":

    ygg databricks seed                  # auto: get-or-create heavy artifacts,
                                         # create-or-update the light ones
    ygg databricks seed --check          # read-only readiness report (CI gate)
    ygg databricks seed --mode append    # add only what's missing
    ygg databricks seed --mode overwrite # rebuild every wheel + the env from scratch, then end

``--mode`` (a :class:`~yggdrasil.enums.Mode`) sets the idempotency policy:

- ``auto`` (default) — **get-or-create** the heavy artifacts (wheels reused when
  already deployed, the cluster get-or-created) but **create-or-update** the
  light/configurable ones (the env config files, warehouse, pools, assistant).
- ``append`` — add only what's **missing** (don't rewrite existing env files or
  redeploy existing assistant files).
- ``overwrite`` — rebuild every wheel (all Pythons + the bundle) from scratch
  and rewrite the environment files, then **end** (skips the later steps).

It walks six areas:

- **config**      — connectivity, host, current user, default catalog/schema.
- **wheels**      — the versioned ygg image wheel in the workspace registry.
- **environments**— the version-pinned base environments ygg jobs run under,
  persisted under ``/Workspace/Shared/environment/ygg`` — **one pair per Python**:
  ``ygg-<version>-py3XX.yml`` (serverless ``base_environment``) and
  ``ygg-<version>-py3XX.requirements.txt`` (classic-cluster
  ``Library(requirements=...)``). Both list only **built wheels in the workspace
  pypi registry**, so the runtime installs with zero PyPI access. ``--all-versions``
  (and ``--mode overwrite``) writes the pair for every supported Python (3.10–3.13).
- **warehouses**  — a default SQL warehouse to execute statements against.
- **pools**       — the default Light / Medium / Heavy Yggdrasil instance pools
  (AWS r5d memory-optimized, local NVMe), each preloading the local-Python DBR
  runtime so pool-backed clusters attach warm against the seeded zero-PyPI wheel
  bundle. Lazy by default (no idle nodes → no cost until attached). Skip with
  ``--no-pools``.
- **cluster**     — a default **single-user (dedicated)** all-purpose cluster
  owned by the current user, **attached to the Light instance pool** and running
  the seeded **generic environment** (the classic-cluster ``requirements.txt``,
  zero-PyPI) so it matches the jobs' image. Lazy: created with autotermination
  and not waited on (starts on attach, self-stops when idle → no cost until
  used). Skip with ``--no-cluster``.
- **assistant**   — the Databricks Assistant ("Genie") config bundle from
  ``yggdrasil.databricks.assistant``: the workspace + user guidance and the
  per-task Skills, deployed under ``/Workspace/Shared/.ygg/assistant`` and
  ``/Workspace/Users/<me>/.ygg/assistant`` (plus a best-effort live
  Assistant-settings push). They teach the Assistant to drive ygg in Python
  on serverless — never via the CLI. Skip with ``--no-assistant``.

In the default (``auto``) mode it builds/uploads the wheel (reusing one already
deployed), assembles and writes the environment files, and ensures a default
warehouse, pools, cluster, and assistant bundle exist. With ``--check`` it
touches nothing and exits non-zero when something is missing — so a pipeline can
gate on ``ygg databricks seed --check``. With ``--mode overwrite`` it forces a
fresh rebuild of every wheel (all supported Pythons + the dependency bundle),
rewrites the environment files, and **ends** — skipping the warehouse step (a
focused "rebuild the image from scratch" command).
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
        parser.add_argument("--mode", default="auto", choices=["auto", "append", "overwrite"],
                            help="Idempotency policy. auto (default): get-or-create the wheels + "
                                 "cluster, but create-or-update the light/configurable artifacts "
                                 "(env config files, warehouse, pools, assistant). append: add only "
                                 "what's missing. overwrite: rebuild every wheel (all Pythons + the "
                                 "bundle) from scratch and rewrite the environment files, then end "
                                 "(skips the warehouse + pools + cluster + assistant steps).")
        parser.add_argument("--no-pools", dest="no_pools", action="store_true",
                            help="Skip the default Light/Medium/Heavy instance pools step.")
        parser.add_argument("--no-cluster", dest="no_cluster", action="store_true",
                            help="Skip the default single-user (dedicated) all-purpose cluster step.")
        parser.add_argument("--no-assistant", dest="no_assistant", action="store_true",
                            help="Skip deploying the Databricks Assistant skills + guidance bundle.")
        parser.set_defaults(handler=cls._seed)

    @classmethod
    def _seed(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style
        from yggdrasil.enums.mode import Mode

        check = args.check
        deploy_mode = Mode.from_(args.mode)
        # OVERWRITE forces a full from-scratch rebuild (every Python + the bundle)
        # and rewrites the environment files, then ends. AUTO (default) is
        # get-or-create for the heavy artifacts (wheels, cluster) but create-or-
        # update for the light/configurable ones (env files, warehouse, pools,
        # assistant). APPEND adds only what's missing. --check stays read-only and
        # wins if combined with a mode.
        overwrite = deploy_mode is Mode.OVERWRITE and not check
        rebuild = overwrite or args.rebuild
        all_versions = overwrite or args.all_versions
        label = "check" if check else deploy_mode.name.lower()
        client = build_client(args)
        style.out(f"\n  {style.bold('ygg databricks seed')}  "
                  f"{style.dim('(' + label + ' mode)')}\n\n")

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
                with style.Spinner(f"building ygg {version} wheel…"):
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
                # Persist the version-pinned base environments under the project
                # folder /Workspace/Shared/environment/ygg. Each Python gets a
                # version-tagged pair in that one folder — a serverless
                # ``ygg-<version>-py3XX.yml`` (referenced by path via
                # ``base_environment``), a classic-cluster
                # ``ygg-<version>-py3XX.requirements.txt``, and a shared ``binaries/``
                # closure of that Python's whole transitive dependency set built
                # as wheels under the env itself — so the runtime installs with
                # zero PyPI access and the env is self-describing. With
                # --all-versions/--overwrite this covers every supported Python
                # (3.10–3.13); otherwise just the local one.
                pythons = list(whl.SUPPORTED_PYTHONS) if all_versions else [None]
                # Each Python's environment is a self-contained folder with its
                # own wheel binaries, so they share nothing and build in parallel.
                plural = "s" if len(pythons) > 1 else ""
                with style.Spinner(
                    f"building {len(pythons)} base environment{plural} "
                    f"(parallel, wheel bundle + binaries)…"
                ):
                    envs = whl.ensure_environments(
                        client, versions=pythons,
                        workspace_dir=whl.WORKSPACE_ENV_DIR, rebuild=rebuild,
                        mode=deploy_mode,
                    )
                for env in envs:
                    style.out(
                        f"    {style.dim(env['key'])}  {env['n_wheels']} wheels  "
                        f"{style.dim('dir')} {env['env_dir']}\n"
                    )
                    style.out(f"          {style.dim('serverless')} {env['serverless']}\n")
                    style.out(f"          {style.dim('cluster')}    {env['cluster']}\n")
                style.ok(
                    f"base environments written for {len(envs)} Python version(s) "
                    f"(serverless + cluster, binaries under each)"
                )
        except Exception as exc:
            style.fail(f"environment step failed: {exc}")
            ok = False

        # OVERWRITE is a focused "rebuild the image" command: stop after the
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
                    with style.Spinner("provisioning Light/Medium/Heavy pools…"):
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

        # -- cluster -----------------------------------------------------
        # A default single-user (dedicated) all-purpose cluster owned by the
        # current user, so interactive ygg work has compute on hand. Lazy:
        # created with autotermination and not waited on — it starts on attach
        # and self-stops when idle, so seeding it costs nothing until used.
        # ``--no-cluster`` opts out.
        if not args.no_cluster:
            style.info("cluster")
            try:
                clusters_svc = client.compute.clusters
                default_name = client.user_scoped_name("All Purpose")
                if check:
                    found = clusters_svc.find_cluster(
                        cluster_name=default_name, raise_error=False,
                    )
                    if found is None:
                        style.warn(f"default cluster {default_name!r} not provisioned")
                        ok = False
                    else:
                        style.out(
                            f"    {style.dim('found')}  {default_name}  "
                            f"{style.dim(str(found.cluster_id))}\n"
                        )
                        style.ok("default single-user cluster present")
                else:
                    # Attach to the default Light pool so the cluster starts warm
                    # against the seeded zero-PyPI bundle (and inherits the pool's
                    # node type). Falls back to a standalone node type if the pool
                    # isn't present (e.g. seeded with ``--no-pools``).
                    from yggdrasil.databricks.compute.instance_pool import (
                        DEFAULT_POOL_TIERS,
                    )

                    light_name = DEFAULT_POOL_TIERS[0].pool_name()
                    pool = client.compute.instance_pools.find(name=light_name)
                    pool_id = getattr(pool, "instance_pool_id", None) if pool else None
                    if pool_id is None:
                        style.warn(
                            f"pool {light_name!r} not found — cluster will use a "
                            f"standalone node type"
                        )
                    # Install the seeded **generic environment** (the classic-cluster
                    # ``requirements.txt`` written by the environments step above) so
                    # the cluster runs the same zero-PyPI ygg image as the jobs,
                    # instead of resolving ``ygg[…]`` from PyPI.
                    env_name = whl.ygg_base_environment_name()
                    env_requirements = (
                        f"{whl.WORKSPACE_ENV_DIR}/{whl.environment_folder('ygg')}/"
                        f"{env_name}.requirements.txt"
                    )
                    with style.Spinner("provisioning default single-user cluster…"):
                        # ``single_user_name`` flips the cluster to dedicated
                        # (single-user) access mode for the current user;
                        # ``instance_pool_id`` attaches it to the Light pool;
                        # ``environment`` installs the generic env (zero-PyPI);
                        # ``wait=False`` returns without blocking on start-up.
                        cluster = clusters_svc.all_purpose_cluster(
                            single_user_name=user,
                            instance_pool_id=pool_id,
                            environment=env_requirements,
                            wait=False,
                        )
                    details = None
                    try:
                        details = cluster.details
                    except Exception:
                        pass
                    mode = getattr(
                        getattr(details, "data_security_mode", None), "value", None,
                    )
                    style.out(
                        f"    {style.dim('cluster')} {cluster.cluster_name}  "
                        f"{style.dim(str(cluster.cluster_id))}\n"
                    )
                    style.out(
                        f"    {style.dim('access')}  "
                        f"{mode or 'dedicated (single user)'}  "
                        f"{style.dim('single_user=' + str(user))}\n"
                    )
                    style.out(
                        f"    {style.dim('pool')}    "
                        f"{light_name if pool_id else style.dim('(standalone)')}\n"
                    )
                    style.out(f"    {style.dim('env')}     {env_name}\n")
                    style.ok(
                        "default single-user cluster ready "
                        "(dedicated, pool-backed, generic env, autoterminating)"
                    )
            except Exception as exc:
                style.fail(f"cluster step failed: {exc}")
                ok = False

        # -- assistant ---------------------------------------------------
        # Deploy the Databricks Assistant ("Genie") config bundle — the
        # workspace + user guidance and the per-task Skills — so the
        # in-product Assistant drives ygg in Python on serverless instead of
        # reaching for the CLI it can't run. Upload is the reliable half; the
        # live Assistant-settings push is best-effort. ``--no-assistant`` opts
        # out.
        if not args.no_assistant:
            style.info("assistant")
            try:
                from yggdrasil.databricks import assistant as ax

                if check:
                    res = ax.deploy(client, check=True)
                    for path in res["uploaded"][:6]:
                        style.out(f"    {style.dim('found')}  {path}\n")
                    if res["missing"]:
                        style.warn(
                            f"{len(res['missing'])} assistant file(s) not deployed "
                            f"(run `ygg databricks seed` to deploy)"
                        )
                        ok = False
                    else:
                        style.ok(
                            f"assistant bundle present "
                            f"({len(res['uploaded'])} files: skills + guidance)"
                        )
                else:
                    # APPEND only writes assistant files that don't exist yet;
                    # AUTO/OVERWRITE refresh them (create-or-update).
                    with style.Spinner("deploying assistant skills + guidance…"):
                        res = ax.deploy(
                            client, check=False, overwrite=deploy_mode is not Mode.APPEND,
                        )
                    for path in res["uploaded"][:8]:
                        style.out(f"    {style.dim('file')}   {path}\n")
                    if len(res["uploaded"]) > 8:
                        style.out(
                            f"    {style.dim('… +' + str(len(res['uploaded']) - 8) + ' more')}\n"
                        )
                    style.out(f"    {style.dim('api')}    {res['api']}\n")
                    style.ok(
                        f"deployed {len(res['uploaded'])} assistant file(s) "
                        f"(workspace + user skills + guidance)"
                    )
            except Exception as exc:
                style.fail(f"assistant step failed: {exc}")
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
