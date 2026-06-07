"""``ygg databricks environment`` — uniform CRUD over base environments.

An environment is keyed by ``(project, version)``. ``create``/``find`` fetch the
project + its whole dependency closure as wheels (zero-PyPI) and write the
serverless ``.yml`` + cluster ``.requirements.txt`` under
``/Workspace/Shared/environment``. Every project, ``ygg`` included, is handled
the same way::

    ygg databricks environment create <project> [version]   # build + write
    ygg databricks environment find <project> [version]      # get, building on a miss
    ygg databricks environment get <project> [version]       # get, never builds
    ygg databricks environment update <project> [version]    # re-build + overwrite
    ygg databricks environment delete <project> [version]    # remove
    ygg databricks environment list                          # browse
"""
from __future__ import annotations

import sys
from typing import Any


class EnvironmentCommand:

    @classmethod
    def register(cls, subparsers: Any) -> None:
        parser = subparsers.add_parser(
            "environment", aliases=["env", "environments"],
            help="CRUD over base environments (create/find/update/delete/list).",
        )
        parser.add_argument("--workspace-dir", dest="workspace_dir", default=None,
                            help="Environment root (default: /Workspace/Shared/environment).")
        sub = parser.add_subparsers(dest="environment_action")

        def _proj(p, *, version=True):
            p.add_argument("project", nargs="?", default="ygg",
                           help="Project: a local path (built from source) or a PyPI name.")
            if version:
                p.add_argument("version", nargs="?", default=None, help="Version (default: latest).")
            p.add_argument("--workspace-dir", dest="workspace_dir", default=None,
                           help="Environment root (default: /Workspace/Shared/environment).")
            return p

        create = _proj(sub.add_parser("create", help="Build the closure + write the base environment."))
        create.add_argument("--python", default=None, help="Target Python (e.g. 3.11).")
        create.add_argument("--extra", action="append", default=None, help="Extra to fold in (repeatable).")
        create.add_argument("--rebuild", action="store_true", help="Force a fresh build.")
        create.set_defaults(handler=cls._create)

        update = _proj(sub.add_parser("update", help="Re-build + overwrite the base environment."))
        update.add_argument("--python", default=None, help="Target Python (e.g. 3.11).")
        update.add_argument("--extra", action="append", default=None, help="Extra to fold in (repeatable).")
        update.set_defaults(handler=cls._update)

        find = _proj(sub.add_parser("find", help="Get an environment, building it on a miss."))
        find.add_argument("--python", default=None, help="Target Python (e.g. 3.11).")
        find.add_argument("--no-install", dest="install", action="store_false",
                          help="Don't build on a miss (like `get`).")
        find.set_defaults(handler=cls._find, install=True)

        get = _proj(sub.add_parser("get", help="Get a deployed environment (never builds)."))
        get.add_argument("--python", default=None, help="Target Python (e.g. 3.11).")
        get.set_defaults(handler=cls._get)

        delete = _proj(sub.add_parser("delete", aliases=["rm"], help="Delete an environment (a version, or all)."))
        delete.set_defaults(handler=cls._delete)

        ls = sub.add_parser("list", help="List the deployed base environments.")
        ls.add_argument("--workspace-dir", dest="workspace_dir", default=None,
                        help="Environment root (default: /Workspace/Shared/environment).")
        ls.set_defaults(handler=cls._list)

        parser.set_defaults(handler=cls._list)   # bare `environment` → browse

    # -- handlers --------------------------------------------------------
    @classmethod
    def _emit(cls, style: Any, env: Any) -> None:
        style.ok(f"{style.brand(env.name)}  ({len(env.dependencies)} wheels)")
        style.out(f"    {style.dim('serverless')} {env.serverless}\n")
        style.out(f"    {style.dim('cluster')}    {env.cluster}\n")

    @classmethod
    def _create(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        client = build_client(args)
        with style.Spinner(f"building base environment for {args.project}…"):
            env = client.environments.create(
                args.project, args.version, python=args.python,
                extras=tuple(args.extra or ()), workspace_dir=args.workspace_dir,
                rebuild=args.rebuild,
            )
        cls._emit(style, env)
        return 0

    @classmethod
    def _update(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        client = build_client(args)
        with style.Spinner(f"re-building base environment for {args.project}…"):
            env = client.environments.update(
                args.project, args.version, python=args.python,
                extras=tuple(args.extra or ()), workspace_dir=args.workspace_dir,
            )
        cls._emit(style, env)
        return 0

    @classmethod
    def _find(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        client = build_client(args)
        with style.Spinner(f"resolving environment for {args.project}…"):
            env = client.environments.find(args.project, args.version, install=args.install,
                                           python=args.python, workspace_dir=args.workspace_dir)
        if env is None:
            style.warn(f"no environment for {args.project!r}")
            return 1
        cls._emit(style, env)
        return 0

    @classmethod
    def _get(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        client = build_client(args)
        env = client.environments.get(args.project, args.version, python=args.python,
                                      workspace_dir=args.workspace_dir)
        if env is None:
            style.warn(f"{args.project!r} not deployed")
            return 1
        cls._emit(style, env)
        return 0

    @classmethod
    def _delete(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        client = build_client(args)
        removed = client.environments.delete(args.project, args.version, workspace_dir=args.workspace_dir)
        for env in removed:
            style.ok(f"deleted {style.brand(env.name)}")
        if not removed:
            style.warn(f"nothing to delete for {args.project!r}")
            return 1
        return 0

    @classmethod
    def _list(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        client = build_client(args)
        workspace_dir = args.workspace_dir or client.environments.default_dir
        envs = client.environments.list(workspace_dir=workspace_dir)
        if not envs:
            style.warn(f"no base environments under {workspace_dir}")
            return 1
        for env in envs:
            for path in (env.serverless, env.cluster):
                if path:
                    sys.stdout.write(f"{path}\n")
        style.ok(f"{len(envs)} environment(s)")
        return 0
