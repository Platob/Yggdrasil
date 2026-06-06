"""``ygg databricks deploy`` — ship wheels and serverless environment configs.

The deploy machinery in :mod:`yggdrasil.databricks.job.wheel` builds a wheel from
the **live** package on disk, uploads it into the workspace's PyPI-like registry
(``/Workspace/Shared/pypi/<dist>/``), and assembles the serverless
``JobEnvironment`` that installs it. This command surfaces that machinery:

    ygg databricks deploy                 # ygg image: wheel(s) + JobEnvironment JSON
    ygg databricks deploy ygg             # get-or-build the versioned ygg wheel(s)
    ygg databricks deploy wheel <package> # build + upload any package's wheel(s)
    ygg databricks deploy environment     # print the serverless JobEnvironment(s)
    ygg databricks deploy project [path]  # discover a pyproject.toml → wheel +
                                          # environment + a default cluster

``--all-versions`` builds/keys a wheel + environment for every supported Python
(3.10–3.13); without it the deploy targets the local interpreter's Python.

``project`` discovers the nearest ``pyproject.toml`` (from *path* or the cwd),
builds the **project's own wheel**, writes a serverless base environment +
classic-cluster requirements named for the project (``<name>``; the version
stays in the wheel, so redeploys upsert one stable environment), and
get-or-creates a default single-user cluster that installs the project's
dependencies — so a user's project runs on Databricks with one command.
``--mode`` sets the idempotency policy: ``overwrite`` (rebuild + update all),
``append`` (add only what's missing), or ``auto`` (get-or-create wheels but
overwrite the env config files; the default).
"""
from __future__ import annotations

import json
import sys
from typing import Any


class DeployCommand:

    @classmethod
    def register(cls, subparsers: Any) -> None:
        parser = subparsers.add_parser(
            "deploy", help="Ship wheels and serverless environment configs to the workspace."
        )
        # Bare ``deploy`` ships the ygg image — wheel(s) + JobEnvironment JSON.
        parser.add_argument("--workspace-dir", dest="workspace_dir", default=None,
                            help="PyPI-like registry root (default: /Workspace/Shared/pypi).")
        parser.add_argument("--rebuild", action="store_true",
                            help="Force a fresh build even if the version is already deployed.")
        parser.add_argument("--all-versions", dest="all_versions", action="store_true",
                            help="A wheel + environment for every supported Python (3.10–3.13).")
        sub = parser.add_subparsers(dest="deploy_action")

        ygg = sub.add_parser("ygg", help="Build + upload the versioned ygg image wheel(s).")
        ygg.add_argument("--workspace-dir", dest="workspace_dir", default=None,
                         help="PyPI-like registry root (default: /Workspace/Shared/pypi).")
        ygg.add_argument("--rebuild", action="store_true",
                         help="Force a fresh build even if the version is already deployed.")
        ygg.add_argument("--all-versions", dest="all_versions", action="store_true",
                         help="A wheel for every supported Python (3.10–3.13).")
        ygg.set_defaults(handler=cls._ygg)

        wheel = sub.add_parser("wheel", help="Build + upload an arbitrary package's wheel(s).")
        wheel.add_argument("package", help="Import or distribution name (e.g. yggdrasil / ygg).")
        wheel.add_argument("--workspace-dir", dest="workspace_dir", default=None,
                           help="PyPI-like registry root (default: /Workspace/Shared/pypi).")
        wheel.add_argument("--extra", action="append", default=None,
                           help="Optional-dependency extra to fold in (repeatable).")
        wheel.add_argument("-r", "--requirement", dest="requirement", action="append", default=None,
                           help="Extra requirement to bundle alongside the package (repeatable).")
        wheel.add_argument("--no-deps", dest="no_deps", action="store_true",
                           help="Pure-python project wheel only; deps resolve on the cluster.")
        wheel.add_argument("--all-versions", dest="all_versions", action="store_true",
                           help="A wheel for every supported Python (3.10–3.13).")
        wheel.set_defaults(handler=cls._wheel)

        env = sub.add_parser("environment", aliases=["env"],
                             help="Build the ygg wheel(s) and print the serverless JobEnvironment(s) as JSON.")
        env.add_argument("--workspace-dir", dest="workspace_dir", default=None,
                         help="PyPI-like registry root (default: /Workspace/Shared/pypi).")
        env.add_argument("--key", dest="environment_key", default="default",
                         help="environment_key for the config (default: default).")
        env.add_argument("--env-version", dest="environment_version", default=None,
                         help="Serverless environment version (default: matched to the local Python).")
        env.add_argument("--rebuild", action="store_true",
                         help="Force a fresh wheel build before assembling the environment.")
        env.add_argument("--all-versions", dest="all_versions", action="store_true",
                         help="One JobEnvironment per supported Python (3.10–3.13) plus a default.")
        env.set_defaults(handler=cls._environment)

        proj = sub.add_parser(
            "project",
            help="Discover a pyproject.toml → build its wheel + environment + a default cluster.",
        )
        proj.add_argument("path", nargs="?", default=None,
                          help="Project dir or pyproject.toml (default: discover from the cwd).")
        proj.add_argument("--workspace-dir", dest="workspace_dir", default=None,
                          help="PyPI-like registry root (default: /Workspace/Shared/pypi).")
        proj.add_argument("--extra", action="append", default=None,
                          help="optional-dependency extra to include in the env (repeatable).")
        proj.add_argument("--bundle", action="store_true",
                          help="Bundle the dependency closure as wheels (zero-PyPI install).")
        proj.add_argument("--mode", default="auto", choices=["auto", "append", "overwrite"],
                          help="Idempotency policy: overwrite (rebuild + update all), "
                               "append (add only what's missing), auto (get-or-create wheels, "
                               "overwrite env config files). Default: auto.")
        proj.add_argument("--no-cluster", dest="no_cluster", action="store_true",
                          help="Build the wheel + environment only; don't create the default cluster.")
        proj.add_argument("--single-user", dest="single_user_name", default=None,
                          help="Single-user owner for the cluster (default: the current user).")
        proj.set_defaults(handler=cls._project)

        parser.set_defaults(handler=cls._default)

    # -- handlers --------------------------------------------------------
    @classmethod
    def _ygg(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style
        from yggdrasil.databricks.job.wheel import (
            WORKSPACE_PYPI_DIR, ensure_ygg_wheel, ensure_ygg_wheels,
        )

        client = build_client(args)
        workspace_dir = args.workspace_dir or WORKSPACE_PYPI_DIR
        if args.all_versions:
            paths = ensure_ygg_wheels(client, workspace_dir=workspace_dir, rebuild=args.rebuild)
        else:
            paths = ensure_ygg_wheel(client, workspace_dir=workspace_dir, rebuild=args.rebuild)
        for path in paths:
            style.ok(f"deployed {style.brand(path)}")
        return 0

    @classmethod
    def _wheel(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style
        from yggdrasil.databricks.job.wheel import (
            WORKSPACE_PYPI_DIR, ensure_wheel, ensure_wheels,
        )

        client = build_client(args)
        workspace_dir = args.workspace_dir or WORKSPACE_PYPI_DIR
        extras = tuple(args.extra or ())
        if args.all_versions:
            paths = ensure_wheels(client, args.package, workspace_dir=workspace_dir, extras=extras)
        else:
            paths = ensure_wheel(
                client, args.package,
                workspace_dir=workspace_dir,
                extras=extras,
                requirements=tuple(args.requirement or ()),
                no_deps=args.no_deps,
            )
        for path in paths:
            style.ok(f"deployed {style.brand(path)}")
        return 0

    @classmethod
    def _environment(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.databricks.job.wheel import (
            WORKSPACE_PYPI_DIR, ygg_environment, ygg_environments,
        )

        client = build_client(args)
        workspace_dir = args.workspace_dir or WORKSPACE_PYPI_DIR
        if args.all_versions:
            envs = ygg_environments(client, workspace_dir=workspace_dir, rebuild=args.rebuild)
            payload: Any = [env.as_dict() for env in envs]
        else:
            env = ygg_environment(
                client,
                environment_key=args.environment_key,
                environment_version=args.environment_version,
                rebuild=args.rebuild,
                workspace_dir=workspace_dir,
            )
            payload = env.as_dict()
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return 0

    @classmethod
    def _project(cls, args: Any, build_client: Any) -> int:
        """Discover a pyproject.toml, build its wheel + environment (named for the
        project), and get-or-create a default cluster installing its deps."""
        from yggdrasil.cli import style
        from yggdrasil.databricks.job import wheel as whl
        from yggdrasil.enums.mode import Mode

        client = build_client(args)
        pypi_dir = args.workspace_dir or whl.WORKSPACE_PYPI_DIR
        extras = tuple(args.extra or ())
        mode = Mode.from_(args.mode)

        with style.Spinner("building project wheel + environment…"):
            info = whl.ensure_project_environment(
                client, args.path, extras=extras, bundle=args.bundle,
                mode=mode, pypi_dir=pypi_dir,
            )
        style.ok(f"deployed project {style.brand(info['name'])} {info['version']}")
        style.out(f"    {style.dim('mode')}       {mode.name.lower()}\n")
        style.out(f"    {style.dim('env')}        {info['env_name']}\n")
        style.out(f"    {style.dim('serverless')} {info['serverless']}\n")
        style.out(f"    {style.dim('cluster')}    {info['cluster']}\n")
        style.out(f"    {style.dim('deps')}       {info['n_wheels']} entr(y/ies)\n")

        if args.no_cluster:
            return 0

        # A default single-user cluster that installs the project's deps — the
        # classic-cluster requirements file written above (project wheel +
        # dependencies), via Library(requirements=…). OVERWRITE updates an
        # existing cluster's libraries; AUTO/APPEND get-or-create it.
        user = client.workspace_client().current_user.me().user_name
        single_user = args.single_user_name or user
        clusters = client.compute.clusters
        libraries = [info["cluster"], "uv", "dill"]
        with style.Spinner(f"provisioning default cluster {info['name']!r}…"):
            existing = (
                clusters.find_cluster(cluster_name=info["name"], raise_error=False)
                if mode is Mode.OVERWRITE else None
            )
            if existing is not None:
                cluster = existing.update(
                    libraries=libraries, single_user_name=single_user, wait=False,
                )
            else:
                cluster = clusters.all_purpose_cluster(
                    name=info["name"], single_user_name=single_user,
                    environment=info["cluster"], wait=False,
                )
        style.out(
            f"    {style.dim('cluster')}    {cluster.cluster_name}  "
            f"{style.dim(str(cluster.cluster_id))}\n"
        )
        style.ok(
            f"default cluster {cluster.cluster_name!r} ready "
            f"(project deps, single-user, autoterminating)"
        )
        return 0

    @classmethod
    def _default(cls, args: Any, build_client: Any) -> int:
        """Bare ``deploy`` — ship the ygg image: wheel(s) then the JobEnvironment(s)."""
        from yggdrasil.cli import style
        from yggdrasil.databricks.job.wheel import (
            WORKSPACE_PYPI_DIR, ensure_ygg_wheel, ensure_ygg_wheels,
            ygg_environment, ygg_environments,
        )

        client = build_client(args)
        workspace_dir = args.workspace_dir or WORKSPACE_PYPI_DIR
        # Build (and possibly rebuild) the wheel once, then assemble the
        # environment off that fresh build — no second rebuild.
        if args.all_versions:
            paths = ensure_ygg_wheels(client, workspace_dir=workspace_dir, rebuild=args.rebuild)
            for path in paths:
                style.ok(f"deployed {style.brand(path)}")
            envs = ygg_environments(client, workspace_dir=workspace_dir, rebuild=False)
            payload: Any = [env.as_dict() for env in envs]
        else:
            paths = ensure_ygg_wheel(client, workspace_dir=workspace_dir, rebuild=args.rebuild)
            for path in paths:
                style.ok(f"deployed {style.brand(path)}")
            env = ygg_environment(client, workspace_dir=workspace_dir, rebuild=False)
            payload = env.as_dict()
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return 0
