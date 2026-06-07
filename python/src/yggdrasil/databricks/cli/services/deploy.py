"""``ygg databricks deploy`` — deploy the current project to Databricks.

``ygg databricks deploy [path]`` discovers the nearest ``pyproject.toml`` (from
*path* or the cwd), builds the project's wheel, uploads it into the workspace's
PyPI-like registry (``/Workspace/Shared/pypi/<dist>/``), writes a serverless
base environment + classic-cluster requirements named for the project
(``<name>-<version>``), and provisions the project's **default warehouse and
cluster** wired to that env config — so a user's project runs on Databricks
with one command.

    ygg databricks deploy                 # deploy the project under the cwd
    ygg databricks deploy ./my-app        # deploy the project under ./my-app
    ygg databricks deploy --mode overwrite # rebuild + update everything
    ygg databricks deploy --bundle --no-cluster  # zero-PyPI env, no cluster

``--mode`` sets the idempotency policy: ``overwrite`` (rebuild + update all),
``append`` (add only what's missing), or ``auto`` (get-or-create wheels but
overwrite the env config files; the default). ``--no-cluster`` / ``--no-warehouse``
skip provisioning the respective compute.

(The ygg image and arbitrary-package wheels/environments live under the
dedicated ``ygg databricks wheel`` and ``ygg databricks environment`` commands.)
"""
from __future__ import annotations

from typing import Any


class DeployCommand:

    @classmethod
    def register(cls, subparsers: Any) -> None:
        parser = subparsers.add_parser(
            "deploy",
            help="Deploy the current project (wheel + environment + warehouse + cluster) to Databricks.",
        )
        parser.add_argument("path", nargs="?", default=None,
                            help="Project dir or pyproject.toml (default: discover from the cwd).")
        parser.add_argument("--workspace-dir", dest="workspace_dir", default=None,
                            help="PyPI-like registry root (default: /Workspace/Shared/pypi).")
        parser.add_argument("--extra", action="append", default=None,
                            help="Optional-dependency extra to include in the env (repeatable).")
        parser.add_argument("--bundle", action="store_true",
                            help="Bundle the dependency closure as wheels (zero-PyPI install).")
        parser.add_argument("--mode", default="auto", choices=["auto", "append", "overwrite"],
                            help="Idempotency policy: overwrite (rebuild + update all), "
                                 "append (add only what's missing), auto (get-or-create wheels, "
                                 "overwrite env config files). Default: auto.")
        parser.add_argument("--no-cluster", dest="no_cluster", action="store_true",
                            help="Don't provision the project's default single-user cluster.")
        parser.add_argument("--no-warehouse", dest="no_warehouse", action="store_true",
                            help="Don't provision the project's default serverless SQL warehouse.")
        parser.add_argument("--single-user", dest="single_user_name", default=None,
                            help="Single-user owner for the cluster (default: the current user).")
        parser.set_defaults(handler=cls._deploy)

    @classmethod
    def _deploy(cls, args: Any, build_client: Any) -> int:
        """Discover the project's ``pyproject.toml``, build its wheel +
        environment, and provision the project's default warehouse and cluster
        wired to that env config."""
        from yggdrasil.cli import style
        from yggdrasil.enums.mode import Mode

        client = build_client(args)
        extras = tuple(args.extra or ())
        mode = Mode.from_(args.mode)

        with style.Spinner("building project wheel + environment…"):
            env = client.environments.deploy_project(
                args.path, extras=extras, bundle=args.bundle,
                mode=mode, pypi_dir=args.workspace_dir or client.wheels.default_dir,
            )
        project = env.project
        style.ok(f"deployed project {style.brand(project)} {env.version}")
        style.out(f"    {style.dim('mode')}        {mode.name.lower()}\n")
        style.out(f"    {style.dim('env')}         {env.name}\n")
        style.out(f"    {style.dim('serverless')}  {env.serverless}\n")
        style.out(f"    {style.dim('cluster cfg')} {env.cluster}\n")
        style.out(f"    {style.dim('deps')}        {len(env.dependencies)} entr(y/ies)\n")

        if not args.no_warehouse:
            # A default serverless SQL warehouse named for the project — its
            # entry point for SQL/Genie work. (Warehouses run SQL, not wheels;
            # the env config wheels go on the cluster below.)
            with style.Spinner(f"provisioning default warehouse {project!r}…"):
                wh = client.warehouses.create_or_update(
                    name=project, enable_serverless_compute=True,
                )
            style.ok(f"default warehouse {wh.warehouse_name!r} ready (serverless)")

        if not args.no_cluster:
            # A default single-user cluster that installs the project's env
            # config — the classic-cluster requirements file written above
            # (project wheel + dependencies), via Library(requirements=…).
            # OVERWRITE updates an existing cluster's libraries; AUTO/APPEND
            # get-or-create it.
            user = client.workspace_client().current_user.me().user_name
            single_user = args.single_user_name or user
            clusters = client.compute.clusters
            libraries = [env.cluster, "uv", "dill"]
            with style.Spinner(f"provisioning default cluster {project!r}…"):
                existing = (
                    clusters.find_cluster(cluster_name=project, raise_error=False)
                    if mode is Mode.OVERWRITE else None
                )
                if existing is not None:
                    cluster = existing.update(
                        libraries=libraries, single_user_name=single_user, wait=False,
                    )
                else:
                    cluster = clusters.all_purpose_cluster(
                        name=project, single_user_name=single_user,
                        environment=env.cluster, wait=False,
                    )
            style.ok(
                f"default cluster {cluster.cluster_name!r} ready "
                f"(project deps, single-user, autoterminating)"
            )
        return 0
