"""``ygg-databricks bundle`` subcommand dispatcher."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

_BUNDLE_FILENAMES = ("databricks.yml", "databricks.yaml", "bundle.yml", "bundle.yaml")


class BundleCommand:
    """Registers and dispatches ``bundle deploy`` / ``bundle run`` sub-commands."""

    @classmethod
    def register(cls, subparsers: Any) -> None:
        """Attach the ``bundle`` sub-command tree to *subparsers*."""
        bundle_parser = subparsers.add_parser(
            "bundle",
            help="Manage Databricks Asset Bundles.",
        )
        bundle_sub = bundle_parser.add_subparsers(dest="bundle_action")

        deploy_parser = bundle_sub.add_parser(
            "deploy",
            help="Deploy a bundle to a Databricks workspace.",
        )
        cls._add_common_flags(deploy_parser)
        deploy_parser.set_defaults(handler=cls._handle_deploy)

        run_parser = bundle_sub.add_parser(
            "run",
            help="Run a job from a deployed bundle.",
        )
        cls._add_common_flags(run_parser)
        run_parser.add_argument(
            "job_key", nargs="?", default=None,
            help="Job key to run (defaults to the first job in the bundle).",
        )
        run_parser.add_argument(
            "--params", action="append", default=[],
            help="Job parameter overrides as key=value (repeatable).",
        )
        run_parser.set_defaults(handler=cls._handle_run)

        validate_parser = bundle_sub.add_parser(
            "validate",
            help="Validate a bundle config without deploying.",
        )
        cls._add_common_flags(validate_parser)
        validate_parser.set_defaults(handler=cls._handle_validate)

        bundle_parser.set_defaults(handler=lambda args: (
            bundle_parser.print_help() or 1
        ))

    @classmethod
    def _add_common_flags(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "-t", "--target",
            dest="target", default=None,
            help="Deployment target name (e.g. 'prd'). "
                 "Defaults to the target marked default: true.",
        )
        parser.add_argument(
            "-f", "--file",
            dest="bundle_file", default=None,
            help="Path to the bundle YAML file. "
                 "Defaults to databricks.yml in the current directory.",
        )
        parser.add_argument(
            "--host", default=None,
            help="Workspace URL (overrides the target's workspace.host).",
        )
        parser.add_argument(
            "--token", default=None,
            help="Personal access token (overrides env DATABRICKS_TOKEN).",
        )
        parser.add_argument(
            "--profile", default=None,
            help="Profile in ~/.databrickscfg.",
        )

    @classmethod
    def _resolve_bundle_path(cls, args: argparse.Namespace) -> Path:
        if args.bundle_file:
            path = Path(args.bundle_file)
            if not path.exists():
                raise FileNotFoundError(
                    f"Bundle file not found: {path}. "
                    f"Check the path and try again."
                )
            return path

        for name in _BUNDLE_FILENAMES:
            candidate = Path.cwd() / name
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"No bundle file found in {Path.cwd()}. "
            f"Expected one of: {', '.join(_BUNDLE_FILENAMES)}. "
            f"Use -f/--file to specify the path explicitly."
        )

    @classmethod
    def _build_client_for_bundle(
        cls, args: argparse.Namespace, build_client: Any,
    ) -> "DatabricksClient":
        from .config import load_bundle, resolve_target

        bundle_path = cls._resolve_bundle_path(args)
        target = getattr(args, "target", None)

        raw = load_bundle(bundle_path)
        target_cfg, _ = resolve_target(raw, target)

        workspace_cfg = target_cfg.get("workspace") or {}
        if workspace_cfg.get("host") and not getattr(args, "host", None):
            args.host = workspace_cfg["host"]

        return build_client(args)

    @classmethod
    def _handle_deploy(cls, args: argparse.Namespace, build_client: Any) -> int:
        from .deploy import deploy

        bundle_path = cls._resolve_bundle_path(args)
        target = getattr(args, "target", None)
        client = cls._build_client_for_bundle(args, build_client)
        return deploy(bundle_path, target, client=client)

    @classmethod
    def _handle_run(cls, args: argparse.Namespace, build_client: Any) -> int:
        import sys

        from .config import load_bundle, resolve_target

        bundle_path = cls._resolve_bundle_path(args)
        target = getattr(args, "target", None)
        client = cls._build_client_for_bundle(args, build_client)

        raw = load_bundle(bundle_path)
        _, resolved = resolve_target(raw, target)

        resources = resolved.get("resources") or {}
        jobs_cfg = resources.get("jobs") or {}

        job_key = args.job_key
        if job_key is None:
            if not jobs_cfg:
                sys.stderr.write("No jobs defined in the bundle.\n")
                return 1
            job_key = next(iter(jobs_cfg))

        if job_key not in jobs_cfg:
            sys.stderr.write(
                f"Job {job_key!r} not found in bundle. "
                f"Available: {', '.join(jobs_cfg)}.\n"
            )
            return 1

        job_name = jobs_cfg[job_key].get("name", job_key)

        job_params: dict[str, str] = {}
        for kv in (args.params or []):
            if "=" not in kv:
                sys.stderr.write(f"Invalid param format {kv!r} — expected key=value.\n")
                return 1
            k, v = kv.split("=", 1)
            job_params[k] = v

        job = client.jobs.get(name=job_name)
        sys.stderr.write(f"Running job {job_name!r} …\n")

        run = job.run(
            job_parameters=job_params if job_params else None,
            wait=False,
        )
        sys.stderr.write(f"  → Run started: {run.explore_url}\n")
        return 0

    @classmethod
    def _handle_validate(cls, args: argparse.Namespace, build_client: Any) -> int:
        import sys

        from .config import load_bundle, resolve_target
        from .resources import RESOURCE_DEPLOYERS

        bundle_path = cls._resolve_bundle_path(args)
        target = getattr(args, "target", None)

        raw = load_bundle(bundle_path)
        _, resolved = resolve_target(raw, target)

        bundle_name = (resolved.get("bundle") or {}).get("name", "unnamed")
        resources = resolved.get("resources") or {}

        sys.stderr.write(f"Bundle {bundle_name!r} is valid.\n")
        sys.stderr.write(f"  Target: {target or 'default'}\n")

        for rtype in RESOURCE_DEPLOYERS:
            entries = resources.get(rtype) or {}
            if entries:
                sys.stderr.write(f"  {rtype}:\n")
                for key, cfg in entries.items():
                    detail = ""
                    if rtype == "jobs":
                        tasks = cfg.get("tasks") or []
                        detail = f" ({len(tasks)} task(s))"
                    sys.stderr.write(f"    {key}{detail}\n")

        return 0
