"""``ygg databricks wheel`` — uniform CRUD over the workspace wheel registry.

A wheel is keyed by ``(project, version)``. ``create``/``find`` fetch it — a
local path (with a ``pyproject.toml``) is built from source, anything else is a
PyPI project downloaded by name — and upload it under ``/Workspace/Shared/pypi``.
Every project, ``ygg`` included, is handled the same way::

    ygg databricks wheel create <project> [version]   # fetch + upload
    ygg databricks wheel find <project> [version]      # get, building on a miss
    ygg databricks wheel get <project> [version]       # get, never builds
    ygg databricks wheel update <project> [version]    # re-fetch + overwrite
    ygg databricks wheel delete <project> [version]    # remove
    ygg databricks wheel list [project]                # browse the registry
"""
from __future__ import annotations

import sys
from typing import Any


class WheelCommand:

    @classmethod
    def register(cls, subparsers: Any) -> None:
        parser = subparsers.add_parser(
            "wheels", aliases=["wheel"],
            help="CRUD over wheels in the workspace registry (create/find/update/delete/list).",
        )
        parser.add_argument("--workspace-dir", dest="workspace_dir", default=None,
                            help="Registry root (default: /Workspace/Shared/pypi).")
        sub = parser.add_subparsers(dest="wheels_action")

        def _proj(p, *, version=True):
            p.add_argument("project", nargs="?", default="ygg",
                           help="Project: a local path (built from source) or a PyPI name.")
            if version:
                p.add_argument("version", nargs="?", default=None, help="Version (default: latest).")
            p.add_argument("--workspace-dir", dest="workspace_dir", default=None,
                           help="Registry root (default: /Workspace/Shared/pypi).")
            return p

        create = _proj(sub.add_parser("create", help="Fetch (build/download) + upload a wheel."))
        create.add_argument("--python", default=None, help="Target Python (e.g. 3.11).")
        create.add_argument("--extra", action="append", default=None, help="Extra to fold in (repeatable).")
        create.add_argument("--deps", action="store_true", help="Also upload the whole dependency closure.")
        create.add_argument("--rebuild", action="store_true", help="Force a fresh fetch.")
        create.set_defaults(handler=cls._create)

        update = _proj(sub.add_parser("update", help="Re-fetch + overwrite a wheel."))
        update.add_argument("--python", default=None, help="Target Python (e.g. 3.11).")
        update.add_argument("--extra", action="append", default=None, help="Extra to fold in (repeatable).")
        update.add_argument("--deps", action="store_true", help="Also upload the dependency closure.")
        update.set_defaults(handler=cls._update)

        find = _proj(sub.add_parser("find", help="Get a wheel, building it on a miss."))
        find.add_argument("--python", default=None, help="Target Python (e.g. 3.11).")
        find.add_argument("--no-install", dest="install", action="store_false",
                          help="Don't build on a miss (like `get`).")
        find.set_defaults(handler=cls._find, install=True)

        get = _proj(sub.add_parser("get", help="Get a deployed wheel (never builds)."))
        get.add_argument("--python", default=None, help="Target Python (e.g. 3.11).")
        get.set_defaults(handler=cls._get)

        delete = _proj(sub.add_parser("delete", aliases=["rm"], help="Delete a wheel (a version, or all)."))
        delete.set_defaults(handler=cls._delete)

        ls = sub.add_parser("list", help="List wheels (or distributions) in the registry.")
        ls.add_argument("project", nargs="?", default=None,
                        help="Project to list wheels for (default: list distributions).")
        ls.add_argument("--workspace-dir", dest="workspace_dir", default=None,
                        help="Registry root (default: /Workspace/Shared/pypi).")
        ls.set_defaults(handler=cls._list)

        parser.set_defaults(handler=cls._list)   # bare `wheel` → browse

    # -- handlers --------------------------------------------------------
    @classmethod
    def _create(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        client = build_client(args)
        with style.Spinner(f"fetching + uploading {args.project}…"):
            wheels = client.wheels.create(
                args.project, args.version, python=args.python,
                extras=tuple(args.extra or ()), deps=args.deps,
                workspace_dir=args.workspace_dir, rebuild=args.rebuild,
            )
        for wheel in wheels:
            style.ok(f"deployed {style.brand(wheel.path)}")
        return 0

    @classmethod
    def _update(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        client = build_client(args)
        with style.Spinner(f"re-fetching {args.project}…"):
            wheels = client.wheels.update(
                args.project, args.version, python=args.python,
                extras=tuple(args.extra or ()), deps=args.deps,
                workspace_dir=args.workspace_dir,
            )
        for wheel in wheels:
            style.ok(f"updated {style.brand(wheel.path)}")
        return 0

    @classmethod
    def _find(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        client = build_client(args)
        with style.Spinner(f"resolving {args.project}…"):
            wheel = client.wheels.find(args.project, args.version, install=args.install,
                                       python=args.python, workspace_dir=args.workspace_dir)
        if wheel is None:
            style.warn(f"no wheel for {args.project!r}")
            return 1
        sys.stdout.write(f"{wheel.path}\n")
        style.ok(f"{style.brand(wheel.dist)} {wheel.version}")
        return 0

    @classmethod
    def _get(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        client = build_client(args)
        wheel = client.wheels.get(args.project, args.version, python=args.python,
                                  workspace_dir=args.workspace_dir)
        if wheel is None:
            style.warn(f"{args.project!r} not deployed")
            return 1
        sys.stdout.write(f"{wheel.path}\n")
        return 0

    @classmethod
    def _delete(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        client = build_client(args)
        removed = client.wheels.delete(args.project, args.version, workspace_dir=args.workspace_dir)
        for wheel in removed:
            style.ok(f"deleted {style.brand(wheel.path)}")
        if not removed:
            style.warn(f"nothing to delete for {args.project!r}")
            return 1
        return 0

    @classmethod
    def _list(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        project = getattr(args, "project", None)
        result = client.wheels.list(project, workspace_dir=args.workspace_dir)
        if not result:
            sys.stderr.write(f"nothing under {args.workspace_dir or client.wheels.default_dir}\n")
            return 1
        for item in result:
            # A Wheel handle (has a workspace .path) vs a distribution folder name.
            sys.stdout.write(f"{item.path}\n" if hasattr(item, "path") else f"{item}/\n")
        return 0
