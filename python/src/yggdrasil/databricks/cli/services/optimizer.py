"""``ygg databricks optimizer`` — run / deploy the propose-only repo optimizer.

``run`` executes one optimization pass now (this is also the entry point the
deployed serverless job re-enters each cycle); ``deploy`` creates the
periodically-triggered Databricks job that runs ``run`` continuously.
"""
from __future__ import annotations

import sys
from typing import Any


class OptimizerCommand:

    @classmethod
    def register(cls, subparsers: Any) -> None:
        parser = subparsers.add_parser(
            "optimizer", help="Continuous, propose-only repository optimizer agent."
        )
        sub = parser.add_subparsers(dest="optimizer_action")

        run = sub.add_parser("run", help="Run one optimization pass now.")
        run.add_argument("--repo", default="/Workspace/Shared/monteleq",
                         help="Workspace repository path to optimize.")
        run.add_argument("--endpoint", default=None, help="Serving endpoint name.")
        run.add_argument("--proposals", default=None,
                         help="Where to write proposals (default: <repo>/.optimizer/proposals/<run-id>).")
        run.add_argument("--max-files", type=int, default=None, dest="max_files")
        run.add_argument("--max-tokens", type=int, default=None, dest="max_tokens")
        run.add_argument("--temperature", type=float, default=None)
        run.set_defaults(handler=cls._run)

        deploy = sub.add_parser("deploy", help="Deploy the optimizer as a periodic job.")
        deploy.add_argument("--repo", default="/Workspace/Shared/monteleq",
                            help="Workspace repository path to optimize.")
        deploy.add_argument("--endpoint", default=None, help="Serving endpoint name.")
        deploy.add_argument("--proposals", default=None)
        deploy.add_argument("--interval", type=int, default=6,
                            help="Trigger interval (default: 6).")
        deploy.add_argument("--unit", default="HOURS", choices=["HOURS", "DAYS", "WEEKS"],
                            help="Trigger interval unit (default: HOURS).")
        deploy.add_argument("--max-files", type=int, default=None, dest="max_files")
        deploy.add_argument("--max-tokens", type=int, default=None, dest="max_tokens")
        deploy.add_argument("--name", default=None, help="Job name (default: optimize-<repo>).")
        deploy.add_argument("--paused", action="store_true", help="Create the schedule paused.")
        deploy.set_defaults(handler=cls._deploy)

        parser.set_defaults(handler=lambda args, bc: parser.print_help() or 1)

    @classmethod
    def _run(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.databricks.ai import OptimizerConfig

        client = build_client(args)
        fields: dict[str, Any] = {"repo_path": args.repo}
        if args.endpoint:
            fields["endpoint_name"] = args.endpoint
        if args.proposals:
            fields["proposals_path"] = args.proposals
        if args.max_files is not None:
            fields["max_files"] = args.max_files
        if args.max_tokens is not None:
            fields["max_tokens"] = args.max_tokens
        if args.temperature is not None:
            fields["temperature"] = args.temperature

        report = client.ai.optimizer(OptimizerConfig(**fields)).run()
        sys.stdout.write(
            f"Optimized {report.files_scanned} file(s); "
            f"{len(report.changed)} with proposed changes.\n"
            f"Proposals: {report.proposals_path}\n"
        )
        return 0

    @classmethod
    def _deploy(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.databricks.ai import RepoOptimizerFlow

        client = build_client(args)
        flow = RepoOptimizerFlow(
            repo_path=args.repo,
            endpoint_name=args.endpoint,
            proposals_path=args.proposals,
            max_files=args.max_files,
            max_tokens=args.max_tokens,
            interval=args.interval,
            unit=args.unit,
            paused=args.paused,
            name=args.name,
        )
        job = flow.deploy(client)
        sys.stdout.write(
            f"Deployed optimizer job {flow.name!r} (id={getattr(job, 'job_id', None)}) — "
            f"every {args.interval} {args.unit.lower()}.\n"
        )
        return 0
