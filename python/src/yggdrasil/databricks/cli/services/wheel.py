"""``ygg databricks wheel`` — build, upload, and browse wheels in the
workspace's PyPI-like registry.

Where ``deploy`` ships the whole ygg image (wheel + serverless
``JobEnvironment``) in one shot, this group exposes the wheel lifecycle on
its own:

    ygg databricks wheel build <package>    # build from the live package → local .whl(s)
    ygg databricks wheel upload <wheel>...   # upload prebuilt .whl(s) to the registry
    ygg databricks wheel deploy <package>    # build + upload in one step
    ygg databricks wheel list [package]      # browse the workspace registry

``build`` is fully offline — it synthesizes a buildable project from the
*installed* package and runs ``uv``/``pip`` locally, so it needs no
Databricks connection. ``upload`` / ``deploy`` / ``list`` talk to the
workspace registry (``/Workspace/Shared/pypi`` by default).
"""
from __future__ import annotations

import sys
from typing import Any


class WheelCommand:

    @classmethod
    def register(cls, subparsers: Any) -> None:
        parser = subparsers.add_parser(
            "wheel", help="Build, upload, and browse wheels in the workspace registry."
        )
        sub = parser.add_subparsers(dest="wheel_action")

        build = sub.add_parser(
            "build", help="Build wheel(s) from the live package on disk (no upload)."
        )
        build.add_argument("package", help="Import or distribution name (e.g. yggdrasil / ygg).")
        build.add_argument("--out-dir", dest="out_dir", default=None,
                           help="Directory to write the .whl(s) into (default: a temp dir).")
        build.add_argument("--extra", action="append", default=None,
                           help="Optional-dependency extra to fold in (repeatable).")
        build.add_argument("-r", "--requirement", dest="requirement", action="append", default=None,
                           help="Extra requirement to bundle alongside (repeatable).")
        build.add_argument("--no-deps", dest="no_deps", action="store_true",
                           help="Pure-python project wheel only; deps resolve at install time.")
        build.add_argument("--all-versions", dest="all_versions", action="store_true",
                           help="A wheel for every supported Python (3.10–3.13).")
        build.set_defaults(handler=cls._build)

        upload = sub.add_parser("upload", help="Upload prebuilt .whl file(s) to the registry.")
        upload.add_argument("wheels", nargs="+", help="Local .whl file path(s).")
        upload.add_argument("--workspace-dir", dest="workspace_dir", default=None,
                           help="PyPI-like registry root (default: /Workspace/Shared/pypi).")
        upload.set_defaults(handler=cls._upload)

        deploy = sub.add_parser("deploy", help="Build the live package and upload its wheel(s).")
        deploy.add_argument("package", help="Import or distribution name (e.g. yggdrasil / ygg).")
        deploy.add_argument("--workspace-dir", dest="workspace_dir", default=None,
                           help="PyPI-like registry root (default: /Workspace/Shared/pypi).")
        deploy.add_argument("--extra", action="append", default=None,
                           help="Optional-dependency extra to fold in (repeatable).")
        deploy.add_argument("-r", "--requirement", dest="requirement", action="append", default=None,
                           help="Extra requirement to bundle alongside (repeatable).")
        deploy.add_argument("--no-deps", dest="no_deps", action="store_true",
                           help="Pure-python project wheel only; deps resolve on the cluster.")
        deploy.add_argument("--all-versions", dest="all_versions", action="store_true",
                           help="A wheel for every supported Python (3.10–3.13).")
        deploy.set_defaults(handler=cls._deploy)

        ls = sub.add_parser("list", help="List wheels (or distributions) in the registry.")
        ls.add_argument("package", nargs="?", default=None,
                       help="Distribution/import name to list wheels for (default: list distributions).")
        ls.add_argument("--workspace-dir", dest="workspace_dir", default=None,
                       help="PyPI-like registry root (default: /Workspace/Shared/pypi).")
        ls.set_defaults(handler=cls._list)

        parser.set_defaults(handler=lambda args, bc: parser.print_help() or 1)

    # -- handlers --------------------------------------------------------
    @classmethod
    def _build(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style
        from yggdrasil.databricks.job.wheel import build_wheel, build_wheels_for_versions

        extras = tuple(args.extra or ())
        if args.all_versions:
            wheels = build_wheels_for_versions(args.package, extras=extras, dest_dir=args.out_dir)
        else:
            wheels = build_wheel(
                args.package,
                extras=extras,
                requirements=tuple(args.requirement or ()),
                no_deps=args.no_deps,
                dest_dir=args.out_dir,
            )
        for wheel in wheels:
            sys.stdout.write(f"{wheel}\n")
        style.ok(f"built {len(wheels)} wheel(s)")
        return 0

    @classmethod
    def _upload(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style
        from yggdrasil.databricks.job.wheel import WORKSPACE_PYPI_DIR, upload_wheel

        client = build_client(args)
        workspace_dir = args.workspace_dir or WORKSPACE_PYPI_DIR
        for wheel in args.wheels:
            dest = upload_wheel(client, wheel, workspace_dir=workspace_dir)
            sys.stdout.write(f"{dest}\n")
            style.ok(f"uploaded {style.brand(dest)}")
        return 0

    @classmethod
    def _deploy(cls, args: Any, build_client: Any) -> int:
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
            sys.stdout.write(f"{path}\n")
            style.ok(f"deployed {style.brand(path)}")
        return 0

    @classmethod
    def _list(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.databricks.job.wheel import WORKSPACE_PYPI_DIR, distribution_for
        from yggdrasil.databricks.path import DatabricksPath

        client = build_client(args)
        workspace_dir = (args.workspace_dir or WORKSPACE_PYPI_DIR).rstrip("/")

        if args.package:
            # One distribution's folder — list its wheel files.
            dist = distribution_for(args.package)
            folder = DatabricksPath.from_(f"{workspace_dir}/{dist}", client=client)
            if not folder.exists():
                sys.stderr.write(f"no deployed wheels for {dist!r} under {workspace_dir}\n")
                return 1
            for child in folder.iterdir():
                if str(child.name).endswith(".whl"):
                    sys.stdout.write(f"{child.full_path()}\n")
            return 0

        # No package — list the distribution folders in the registry.
        root = DatabricksPath.from_(workspace_dir, client=client)
        if not root.exists():
            sys.stderr.write(f"no registry at {workspace_dir}\n")
            return 1
        for child in root.iterdir():
            if child.is_dir():
                sys.stdout.write(f"{child.name}/\n")
        return 0
