"""``ygg-databricks jobs`` — manage Databricks jobs."""
from __future__ import annotations

import sys
from typing import Any


class JobsCommand:

    @classmethod
    def register(cls, subparsers: Any) -> None:
        parser = subparsers.add_parser("jobs", help="Manage Databricks jobs.")
        sub = parser.add_subparsers(dest="jobs_action")

        ls = sub.add_parser("list", help="List jobs in the workspace.")
        ls.add_argument("--name", default=None, help="Filter by job name.")
        ls.add_argument("--limit", type=int, default=None, help="Max jobs to return.")
        ls.set_defaults(handler=cls._list)

        get = sub.add_parser("get", help="Get a job by id or name.")
        get.add_argument("--id", dest="job_id", type=int, default=None)
        get.add_argument("--name", default=None)
        get.set_defaults(handler=cls._get)

        create = sub.add_parser("create", help="Create a job from a YAML file.")
        create.add_argument("-f", "--file", required=True, help="Job YAML config file.")
        create.set_defaults(handler=cls._create)

        delete = sub.add_parser("delete", help="Delete a job.")
        delete.add_argument("--id", dest="job_id", type=int, default=None)
        delete.add_argument("--name", default=None)
        delete.set_defaults(handler=cls._delete)

        run = sub.add_parser("run", help="Run a job.")
        run.add_argument("--id", dest="job_id", type=int, default=None)
        run.add_argument("--name", default=None)
        run.add_argument("--params", action="append", default=[],
                         help="Job parameters as key=value (repeatable).")
        run.add_argument("--wait", action="store_true", help="Wait for run to complete.")
        run.set_defaults(handler=cls._run)

        parser.set_defaults(handler=lambda args, bc: parser.print_help() or 1)

    @classmethod
    def _list(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        for job in client.jobs.list(
            name=getattr(args, "name", None),
            limit=getattr(args, "limit", None),
        ):
            sys.stdout.write(
                f"{job.job_id}\t{job.job_name}\t{job.explore_url}\n"
            )
        return 0

    @classmethod
    def _get(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        job = client.jobs.find(
            job_id=getattr(args, "job_id", None),
            name=getattr(args, "name", None),
            raise_error=True,
        )
        settings = job.settings
        task_count = len(settings.tasks) if settings and settings.tasks else 0
        sys.stdout.write(
            f"Job:    {job.job_name}\n"
            f"ID:     {job.job_id}\n"
            f"URL:    {job.explore_url}\n"
            f"Tasks:  {task_count}\n"
        )
        if settings and settings.schedule:
            sys.stdout.write(
                f"Schedule: {settings.schedule.quartz_cron_expression} "
                f"({settings.schedule.timezone_id})\n"
            )
        return 0

    @classmethod
    def _create(cls, args: Any, build_client: Any) -> int:
        from pathlib import Path
        from ..bundle.config import load_bundle
        from ..bundle.resources import deploy_job

        path = Path(args.file)
        if not path.exists():
            sys.stderr.write(f"File not found: {path}\n")
            return 1

        import yaml
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

        client = build_client(args)
        job_name = cfg.get("name", path.stem)
        deploy_job(client, job_name, cfg)
        return 0

    @classmethod
    def _delete(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        client.jobs.delete(
            job_id=getattr(args, "job_id", None),
            name=getattr(args, "name", None),
        )
        sys.stderr.write("Deleted.\n")
        return 0

    @classmethod
    def _run(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        job = client.jobs.find(
            job_id=getattr(args, "job_id", None),
            name=getattr(args, "name", None),
            raise_error=True,
        )

        job_params: dict[str, str] = {}
        for kv in (args.params or []):
            if "=" not in kv:
                sys.stderr.write(f"Invalid param format {kv!r} — expected key=value.\n")
                return 1
            k, v = kv.split("=", 1)
            job_params[k] = v

        run = job.run(
            job_parameters=job_params if job_params else None,
            wait=getattr(args, "wait", False),
        )
        sys.stdout.write(f"{run.run_id}\t{run.explore_url}\n")
        return 0
