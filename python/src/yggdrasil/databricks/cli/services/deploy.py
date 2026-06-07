"""``ygg databricks deploy`` — the single way to deploy a project to Databricks.

``ygg databricks deploy [path]`` builds the project at *path* (**the current
working directory by default**, or a PyPI name) — its wheel + whole dependency
closure as wheels (zero-PyPI) — writes the serverless base environment +
classic-cluster requirements named for the project (``<name>-<version>``), and
provisions the project's **default warehouse and cluster** wired to that env
config. One command takes a project, ``ygg`` itself included, to Databricks.

    ygg databricks deploy                 # deploy the project in the cwd
    ygg databricks deploy ./my-app        # deploy the project under ./my-app
    ygg databricks deploy ygg             # deploy a published project by name
    ygg databricks deploy --python 3.11 --python 3.12   # build for two Pythons
    ygg databricks deploy --rebuild --no-cluster

The environment bundles the project wheel + its whole dependency closure as
wheels (zero-PyPI). ``--python`` builds an environment per Python version
(repeatable; defaults to the interpreter running the CLI). ``--rebuild`` forces
a fresh build; ``--no-cluster`` / ``--no-warehouse`` skip provisioning the
respective compute.

Resource provisioning is **fire-and-forget** — the warehouse and cluster are
created without blocking on them to reach a running state (they spin up in the
background), so the command returns as soon as the spec is submitted. The client
is bound to the deployed project + version first, so the default warehouse and
cluster are named for the project (its capitalized display name).

(Lower-level wheel/environment CRUD lives under the ``ygg databricks wheel`` and
``ygg databricks environment`` commands; this is the one-shot project deploy.)
"""
from __future__ import annotations

from typing import Any


class DeployCommand:

    @classmethod
    def register(cls, subparsers: Any) -> None:
        parser = subparsers.add_parser(
            "deploy",
            help="Deploy the current project (environment + warehouse + cluster) to Databricks.",
        )
        parser.add_argument("path", nargs="?", default=".",
                            help="Project dir or pyproject.toml (default: the current working directory).")
        parser.add_argument("--workspace-dir", dest="workspace_dir", default=None,
                            help="Environment root (default: /Workspace/Shared/environment).")
        parser.add_argument("--extra", action="append", default=None,
                            help="Optional-dependency extra to include in the env (repeatable).")
        parser.add_argument("--python", dest="pythons", action="append", default=None,
                            metavar="X.Y",
                            help="Python version to build the environment for, e.g. 3.11 "
                                 "(repeatable; default: the version running this CLI).")
        parser.add_argument("--rebuild", action="store_true",
                            help="Force a fresh build of the wheel closure + env config.")
        parser.add_argument("--no-cluster", dest="no_cluster", action="store_true",
                            help="Don't provision the project's default single-user cluster.")
        parser.add_argument("--no-warehouse", dest="no_warehouse", action="store_true",
                            help="Don't provision the project's default serverless SQL warehouse.")
        parser.add_argument("--single-user", dest="single_user_name", default=None,
                            help="Single-user owner for the cluster (default: the current user).")
        parser.set_defaults(handler=cls._deploy)

    @classmethod
    def _deploy(cls, args: Any, build_client: Any) -> int:
        """Build the project's base environment(s) (wheel closure, zero-PyPI) and
        provision its default warehouse and cluster wired to that env config."""
        from yggdrasil.cli import style

        from yggdrasil.databricks.wheels.service import (
            find_pyproject,
            project_display_name,
            read_pyproject,
        )

        client = build_client(args)
        extras = tuple(args.extra or ())
        # One environment per requested Python; ``None`` builds for the
        # interpreter running this CLI (the default when --python is omitted).
        pythons = args.pythons or [None]

        # Bind the client to the project (+ version) up front when discoverable
        # from a local pyproject, so the whole deploy — wheel upload, env write,
        # provisioning — runs under the right project context (user-agent, tags,
        # default-resource names). The PyPI-name path has no local pyproject; it's
        # reconciled from the built environment below.
        pyproject = find_pyproject(args.path)
        if pyproject is not None:
            try:
                meta = read_pyproject(pyproject)
                client.project = meta["name"]
                if meta.get("version"):
                    client.product_version = str(meta["version"])
            except Exception:  # noqa: BLE001 — best-effort; reconciled post-build
                pass

        envs = []
        for py in pythons:
            label = f" (py{py})" if py else ""
            with style.Spinner(f"building project wheel + environment{label}…"):
                env = client.environments.create(
                    args.path, python=py, extras=extras,
                    workspace_dir=args.workspace_dir, rebuild=args.rebuild,
                )
            envs.append(env)
            style.ok(f"built environment {style.brand(env.name)}")
            style.out(f"    {style.dim('serverless')}  {env.serverless}\n")
            style.out(f"    {style.dim('cluster cfg')} {env.cluster}\n")
            style.out(f"    {style.dim('deps')}        {len(env.dependencies)} entr(y/ies)\n")

        # The cluster runs a single Python, so it installs the first env built;
        # serverless picks the matching env per job at runtime.
        primary = envs[0]
        project = primary.project
        # Reconcile the client with the authoritative project + version from the
        # built environment (covers the PyPI-name path with no local pyproject).
        # The default warehouse + cluster are named for the project's capitalized
        # display name — what `find_default` / `default_names()` resolve to once
        # the client carries the project.
        client.project = project
        if primary.version is not None:
            client.product_version = str(primary.version)
        name = project_display_name(project)
        style.ok(f"deployed project {style.brand(project)} {primary.version} "
                 f"· {len(envs)} environment(s)")

        if not args.no_warehouse:
            # The project's **default SQL warehouse** — the project's nice display
            # name (serverless), the same name `find_default` resolves to when
            # this project is the running client project. Fire-and-forget
            # (``wait=False``): don't block on it reaching a running state.
            with style.Spinner(f"provisioning default warehouse {name!r}…"):
                wh = client.warehouses.create_or_update(
                    name=name, enable_serverless_compute=True, wait=False,
                )
            style.ok(f"default warehouse {wh.warehouse_name!r} provisioning (serverless)")

        if not args.no_cluster:
            # A default single-user cluster, named for the project, that installs
            # the project's env config — the classic-cluster requirements file
            # (project wheel + dependencies). Fire-and-forget (``wait=False``).
            user = client.workspace_client().current_user.me().user_name
            single_user = args.single_user_name or user
            clusters = client.compute.clusters
            libraries = [primary.cluster, "uv", "dill"]
            with style.Spinner(f"provisioning default cluster {name!r}…"):
                existing = (
                    clusters.find_cluster(cluster_name=name, raise_error=False)
                    if args.rebuild else None
                )
                if existing is not None:
                    cluster = existing.update(
                        libraries=libraries, single_user_name=single_user, wait=False,
                    )
                else:
                    cluster = clusters.all_purpose_cluster(
                        name=name, single_user_name=single_user,
                        environment=primary.cluster, wait=False,
                    )
            style.ok(
                f"default cluster {cluster.cluster_name!r} provisioning "
                f"(project deps, single-user, autoterminating)"
            )
        return 0
