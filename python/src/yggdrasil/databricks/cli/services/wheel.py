"""``ygg databricks wheel`` — build, upload, and browse wheels in the
workspace's PyPI-like registry.

Where ``deploy`` ships the whole ygg image (wheel + serverless
``JobEnvironment``) in one shot, this group exposes the wheel lifecycle on
its own:

    ygg databricks wheel                     # build + upload the ygg wheel(s) to shared pypi
    ygg databricks wheel build [package]    # build from the live package → local .whl(s)
    ygg databricks wheel upload <wheel>...   # upload prebuilt .whl(s) to the registry
    ygg databricks wheel deploy [package]    # build + upload in one step (default package: ygg)
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
            "wheels", aliases=["wheel"],
            help="Build, upload, and browse wheels in the workspace registry.",
        )
        # Bare ``wheel`` builds + uploads the versioned ygg wheel(s) to the
        # registry (the common case); subcommands cover the rest.
        parser.add_argument("--workspace-dir", dest="workspace_dir", default=None,
                            help="PyPI-like registry root (default: /Workspace/Shared/pypi).")
        parser.add_argument("--rebuild", action="store_true",
                            help="Force a fresh build even if the version is already deployed.")
        parser.add_argument("--all-versions", dest="all_versions", action="store_true",
                            help="A wheel for every supported Python (3.10–3.13).")
        sub = parser.add_subparsers(dest="wheels_action")

        build = sub.add_parser(
            "build", help="Build wheel(s) from the live package on disk (no upload)."
        )
        build.add_argument("package", nargs="?", default="ygg",
                           help="Import or distribution name (default: ygg).")
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
        deploy.add_argument("package", nargs="?", default="ygg",
                           help="Import or distribution name (default: ygg).")
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

        parser.set_defaults(handler=cls._default)

    # -- handlers --------------------------------------------------------
    @classmethod
    def _default(cls, args: Any, build_client: Any) -> int:
        """Bare ``wheel`` — build + upload the versioned ygg wheel(s) to the
        workspace PyPI-like registry (shared pypi)."""
        from yggdrasil.cli import style

        client = build_client(args)
        with style.Spinner("building + uploading ygg wheel(s)…"):
            wheels = client.wheels.deploy_ygg(
                all_versions=getattr(args, "all_versions", False),
                rebuild=getattr(args, "rebuild", False),
                workspace_dir=args.workspace_dir,
            )
        for wheel in wheels:
            style.ok(f"deployed {style.brand(wheel.path)}")
        return 0

    @classmethod
    def _build(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        client = build_client(args)
        wheels = client.wheels.build(
            args.package,
            extras=tuple(args.extra or ()),
            requirements=tuple(args.requirement or ()),
            no_deps=args.no_deps,
            all_versions=args.all_versions,
            dest_dir=args.out_dir,
        )
        for wheel in wheels:
            sys.stdout.write(f"{wheel}\n")
        style.ok(f"built {len(wheels)} wheel(s)")
        return 0

    @classmethod
    def _upload(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        client = build_client(args)
        for wheel in args.wheels:
            uploaded = client.wheels.upload(wheel, workspace_dir=args.workspace_dir, registry=False)
            sys.stdout.write(f"{uploaded.path}\n")
            style.ok(f"uploaded {style.brand(uploaded.path)}")
        return 0

    @classmethod
    def _deploy(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        client = build_client(args)
        wheels = client.wheels.deploy(
            args.package,
            workspace_dir=args.workspace_dir,
            extras=tuple(args.extra or ()),
            requirements=tuple(args.requirement or ()),
            no_deps=args.no_deps,
            all_versions=args.all_versions,
        )
        for wheel in wheels:
            sys.stdout.write(f"{wheel.path}\n")
            style.ok(f"deployed {style.brand(wheel.path)}")
        return 0

    @classmethod
    def _list(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        workspace_dir = args.workspace_dir or client.wheels.default_dir
        result = client.wheels.list(args.package or None, workspace_dir=workspace_dir)

        if args.package:
            if not result:
                sys.stderr.write(f"no deployed wheels for {args.package!r} under {workspace_dir}\n")
                return 1
            for wheel in result:
                sys.stdout.write(f"{wheel.path}\n")
            return 0

        # No package — list the distribution folders in the registry.
        if not result:
            sys.stderr.write(f"no registry at {workspace_dir}\n")
            return 1
        for name in result:
            sys.stdout.write(f"{name}/\n")
        return 0
