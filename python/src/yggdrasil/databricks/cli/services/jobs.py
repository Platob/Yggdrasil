"""``ygg databricks job`` — manage Databricks jobs & runs via the Jobs service."""
from __future__ import annotations

import sys
from typing import Any


def _resolve_job(client: Any, target: str) -> Any:
    """Resolve a job by numeric id or by name."""
    if target.isdigit():
        return client.jobs.get(job_id=int(target))
    return client.jobs.get(name=target)


def _kv(pairs: "list[str] | None") -> "dict[str, str]":
    out: dict[str, str] = {}
    for item in pairs or []:
        key, _, val = item.partition("=")
        out[key.strip()] = val
    return out


class JobsCommand:

    @classmethod
    def register(cls, subparsers: Any) -> None:
        parser = subparsers.add_parser(
            "jobs", aliases=["job"], help="Manage Databricks jobs and runs.",
        )
        sub = parser.add_subparsers(dest="jobs_action")

        ls = sub.add_parser("list", help="List jobs in the workspace.")
        ls.add_argument("--name", default=None, help="Filter by job name.")
        ls.add_argument("--limit", type=int, default=None)
        ls.set_defaults(handler=cls._list)

        get = sub.add_parser("get", help="Show a job (id or name): tasks + DAG.")
        get.add_argument("target", help="Job id or name.")
        get.set_defaults(handler=cls._get)

        run = sub.add_parser("run", help="Trigger a job run; optionally wait for it.")
        run.add_argument("target", help="Job id or name.")
        run.add_argument("--param", action="append", default=None,
                         help="Job parameter k=v (repeatable).")
        run.add_argument("--notebook-param", dest="notebook_param", action="append",
                         default=None, help="Notebook widget k=v (repeatable).")
        run.add_argument("--python-param", dest="python_param", action="append",
                         default=None, help="Python wheel/script param (repeatable).")
        run.add_argument("--wait", action="store_true", help="Block until the run finishes.")
        run.add_argument("--timeout", type=float, default=1800.0,
                         help="Seconds to wait when --wait (default 1800).")
        run.set_defaults(handler=cls._run)

        runs = sub.add_parser("runs", help="List recent runs of a job.")
        runs.add_argument("target", help="Job id or name.")
        runs.add_argument("--limit", type=int, default=20)
        runs.add_argument("--active", action="store_true", help="Active runs only.")
        runs.set_defaults(handler=cls._runs)

        logs = sub.add_parser("logs", help="Print a run's output (stdout / errors).")
        logs.add_argument("run_id", type=int, help="Run id.")
        logs.add_argument("--task", default=None, help="Restrict to one task key.")
        logs.set_defaults(handler=cls._logs)

        cancel = sub.add_parser("cancel", help="Cancel a run.")
        cancel.add_argument("run_id", type=int, help="Run id.")
        cancel.set_defaults(handler=cls._cancel)

        repair = sub.add_parser("repair", help="Re-run failed tasks of a run.")
        repair.add_argument("run_id", type=int, help="Run id.")
        repair.add_argument("--task", action="append", default=None,
                            help="Task key to rerun (repeatable; default: all failed).")
        repair.add_argument("--wait", action="store_true")
        repair.set_defaults(handler=cls._repair)

        delete = sub.add_parser("delete", help="Delete a job.")
        delete.add_argument("target", help="Job id or name.")
        delete.set_defaults(handler=cls._delete)

        parser.set_defaults(handler=lambda args, bc: parser.print_help() or 1)

    # -- handlers --------------------------------------------------------
    @classmethod
    def _list(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        for job in client.jobs.list(name=getattr(args, "name", None), limit=args.limit):
            name = getattr(job, "name", None) or (
                job.settings.name if getattr(job, "settings", None) else ""
            )
            sys.stdout.write(f"{job.job_id}\t{name}\n")
        return 0

    @classmethod
    def _get(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        client = build_client(args)
        job = _resolve_job(client, args.target)
        if job is None:
            style.fail(f"no job matching {args.target!r}")
            return 1
        name = getattr(job, "name", None) or (
            job.settings.name if getattr(job, "settings", None) else ""
        )
        style.info(f"job {style.bold(str(job.job_id))} — {style.brand(name)}")
        url = getattr(job, "explore_url", None)
        if url:
            sys.stdout.write(f"  url:   {url}\n")
        dag = job.dag()
        sys.stdout.write(f"  tasks: {', '.join(dag.keys)}\n")
        for src, dst in dag.edges():
            sys.stdout.write(f"    {src} → {dst}\n")
        return 0

    @classmethod
    def _run(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        client = build_client(args)
        job = _resolve_job(client, args.target)
        if job is None:
            style.fail(f"no job matching {args.target!r}")
            return 1
        run = job.run(
            parameters=_kv(args.param) or None,
            notebook_params=_kv(args.notebook_param) or None,
            python_params=(args.python_param or None),
            wait=(args.timeout if args.wait else False),
            raise_error=False,
        )
        style.ok(f"run {style.bold(str(run.run_id))} started")
        if args.wait:
            if run.is_succeeded:
                style.ok(f"run {run.run_id} succeeded in {run.duration_seconds or 0:.1f}s")
            else:
                style.fail(f"run {run.run_id} {run.state} — {run.state_message or ''}")
                if run.stderr:
                    sys.stderr.write(run.stderr + "\n")
                return 1
        return 0

    @classmethod
    def _runs(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        job = _resolve_job(client, args.target)
        if job is None:
            sys.stderr.write(f"no job matching {args.target!r}\n")
            return 1
        for run in job.list_runs(active_only=args.active, limit=args.limit):
            dur = f"{run.duration_seconds:.1f}s" if run.duration_seconds else "-"
            sys.stdout.write(f"{run.run_id}\t{run.state}\t{dur}\n")
        return 0

    @classmethod
    def _logs(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        run = client.job_runs.get(run_id=args.run_id)
        if run is None:
            sys.stderr.write(f"no run {args.run_id}\n")
            return 1
        if args.task:
            sys.stdout.write((run.logs(args.task) or "") + "\n")
        else:
            sys.stdout.write(run.debug() + "\n")
        return 0

    @classmethod
    def _cancel(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        client = build_client(args)
        run = client.job_runs.get(run_id=args.run_id)
        if run is None:
            style.fail(f"no run {args.run_id}")
            return 1
        run.cancel()
        style.ok(f"cancelled run {args.run_id}")
        return 0

    @classmethod
    def _repair(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        client = build_client(args)
        run = client.job_runs.get(run_id=args.run_id)
        if run is None:
            style.fail(f"no run {args.run_id}")
            return 1
        run.repair(rerun_tasks=args.task, wait=(True if args.wait else False), raise_error=False)
        style.ok(f"repair triggered for run {args.run_id}")
        return 0

    @classmethod
    def _delete(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        client = build_client(args)
        if args.target.isdigit():
            client.jobs.delete(job_id=int(args.target))
        else:
            client.jobs.delete(name=args.target)
        style.ok(f"deleted job {args.target}")
        return 0
