"""``ygg databricks deploy`` — take a project (or the ygg image) to the workspace.

The deploy machinery in :mod:`yggdrasil.databricks.job.wheel` builds a wheel from
the package on disk, uploads it into the workspace's PyPI-like registry
(``/Workspace/Shared/pypi/<dist>/``), and assembles the serverless
``JobEnvironment`` that installs it. This command surfaces that machinery:

    ygg databricks deploy                 # alias for ``deploy project`` (cwd)
    ygg databricks deploy project [path]  # discover a pyproject.toml → wheel +
                                          # environment + a default cluster
    ygg databricks deploy ygg             # get-or-build the versioned ygg wheel(s)
    ygg databricks deploy wheel <package> # build + upload any package's wheel(s)
    ygg databricks deploy environment     # print the serverless JobEnvironment(s)

Bare ``deploy`` is an **alias for ``deploy project``** — it discovers the nearest
``pyproject.toml`` from the cwd and ships it. To deploy a project at another
location, pass it to the ``project`` subcommand (``deploy project <path>``).
``--all-versions`` (on ``ygg`` / ``wheel`` / ``environment``) builds/keys for
every supported Python (3.10–3.13); without it those target the local Python.

``project`` discovers the nearest ``pyproject.toml`` (from *path* or the cwd),
builds the **project's own wheel** *and its whole dependency closure* as wheels
in the shared PyPI registry (``/Workspace/Shared/pypi/<dist>/``, generic
per-Python naming), writes a serverless base environment + classic-cluster
requirements named for the project (``<name>``; the version stays in the wheel,
so redeploys upsert one stable environment) that both list those same
shared-registry wheel paths, and get-or-creates a default single-user cluster
that installs from that requirements file — so a user's project runs on
Databricks, entirely from wheels with zero PyPI access, in one command.
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
            "deploy", help="Deploy a project to the workspace (alias for `deploy project`)."
        )
        # Bare ``deploy`` is an alias for ``deploy project`` — it discovers the
        # nearest pyproject.toml from the cwd and ships it (wheel closure +
        # environment + a default cluster). The ygg-image deploys live under the
        # ``ygg`` / ``wheel`` / ``environment`` subcommands. (A specific project
        # path is given via ``deploy project <path>`` — the bare form, sharing the
        # parser with the subcommands, discovers from the cwd.)
        parser.add_argument("--workspace-dir", dest="workspace_dir", default=None,
                            help="PyPI-like registry root (default: /Workspace/Shared/pypi).")
        parser.add_argument("--extra", action="append", default=None,
                            help="optional-dependency extra to include in the env (repeatable).")
        parser.add_argument("--mode", default="auto", choices=["auto", "append", "overwrite"],
                            help="Idempotency policy: overwrite (rebuild + update all), "
                                 "append (add only what's missing), auto (get-or-create wheels, "
                                 "overwrite env config files). Default: auto.")
        parser.add_argument("--no-cluster", dest="no_cluster", action="store_true",
                            help="Build the wheel + environment only; don't create the default cluster.")
        parser.add_argument("--single-user", dest="single_user_name", default=None,
                            help="Single-user owner for the cluster (default: the current user).")
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
        proj.add_argument("--mode", default="auto", choices=["auto", "append", "overwrite"],
                          help="Idempotency policy: overwrite (rebuild + update all), "
                               "append (add only what's missing), auto (get-or-create wheels, "
                               "overwrite env config files). Default: auto.")
        proj.add_argument("--no-cluster", dest="no_cluster", action="store_true",
                          help="Build the wheel + environment only; don't create the default cluster.")
        proj.add_argument("--single-user", dest="single_user_name", default=None,
                          help="Single-user owner for the cluster (default: the current user).")
        proj.set_defaults(handler=cls._project)

        # Bare ``deploy`` → ``deploy project`` discovered from the cwd.
        parser.set_defaults(handler=cls._project, path=None)

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
                client, args.path, extras=extras, mode=mode, pypi_dir=pypi_dir,
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
        # dependency wheels, all from the shared registry), via
        # Library(requirements=…) and nothing else. OVERWRITE updates an existing
        # cluster's libraries; AUTO/APPEND get-or-create it.
        user = client.workspace_client().current_user.me().user_name
        single_user = args.single_user_name or user
        clusters = client.compute.clusters
        libraries = [info["cluster"]]
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
