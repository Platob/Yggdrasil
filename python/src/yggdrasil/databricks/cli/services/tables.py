"""``ygg databricks tables`` — table operations via the Tables service.

Currently exposes Auto Loader ingestion::

    ygg databricks tables autoload catalog.schema.table
    ygg databricks tables autoload my_cat.my_sch.events --no-file-arrival
    ygg databricks tables autoload c.s.t --source s3://bucket/drop/ --format json --run --wait 900

``autoload`` get-or-creates the serverless ``cloudFiles`` ingestion job built by
:meth:`yggdrasil.databricks.table.table.Table.auto_loader` (named
``[YGG][AUTOLOADER] <table>``), shipping the ygg image as a reusable named base
environment (``--environment``; default: the version-pinned ygg image
``ygg-<version>-py3XX`` the seed writes) with the whole dependency closure
bundled for a zero-pip-install cluster env.
"""
from __future__ import annotations

import sys
from typing import Any


class TablesCommand:

    @classmethod
    def register(cls, subparsers: Any) -> None:
        parser = subparsers.add_parser("tables", aliases=["table"],
                                       help="Table operations (Auto Loader ingestion).")
        sub = parser.add_subparsers(dest="tables_action")

        al = sub.add_parser("autoload", aliases=["auto-load", "auto_loader"],
                            help="Get-or-create an Auto Loader (cloudFiles) ingestion job.")
        al.add_argument("table", help="Target table: catalog.schema.table.")
        al.add_argument("--source", default=None,
                        help="Cloud path to watch (default: the table's cloud staging area).")
        al.add_argument("--name", default=None, help="Job name override.")
        al.add_argument("--format", dest="file_format", default="parquet",
                        help="cloudFiles.format (parquet/json/csv/avro/…). Default parquet.")
        al.add_argument("--checkpoint", default=None,
                        help="Checkpoint + schema location (default: derived next to the table).")
        al.add_argument("--continuous", action="store_true",
                        help="Continuous 1-minute micro-batch stream (default: one AvailableNow sweep).")
        al.add_argument("--no-file-arrival", dest="file_arrival", action="store_false",
                        help="Deploy without the default file-arrival trigger on the source.")
        al.set_defaults(file_arrival=True)
        al.add_argument("--clean-source", dest="clean_source", action="store_true",
                        help="Delete each staged file once ingested + past retention (self-cleaning).")
        al.add_argument("--clean-source-retention", dest="clean_source_retention", default="8 days",
                        help="Retention window for --clean-source (> 7 days; default '8 days').")
        al.add_argument("--environment", default=None,
                        help="Reusable serverless base environment name "
                             "(default: the version-pinned ygg image, ygg-<version>-py3XX).")
        al.add_argument("--no-environment", dest="no_environment", action="store_true",
                        help="Inline the dependency list on the job instead of a named base env.")
        al.add_argument("--no-bundle", dest="no_bundle", action="store_true",
                        help="Ship only the ygg wheel; resolve deps from the index (not 0-pip-install).")
        al.add_argument("--no-deploy", dest="no_deploy", action="store_true",
                        help="Build + show the configured job without creating it.")
        al.add_argument("--run", action="store_true",
                        help="Trigger a run once deployed.")
        al.add_argument("--wait", type=float, default=0.0,
                        help="Seconds to block on --run (0 = fire-and-forget).")
        al.set_defaults(handler=cls._autoload)

        parser.set_defaults(handler=lambda args, bc: parser.print_help() or 1)

    # -- handlers --------------------------------------------------------
    @classmethod
    def _autoload(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        client = build_client(args)
        table = client.tables.table(args.table)

        if args.no_environment:
            environment = None                       # inline the deps on the job
        elif args.environment is not None:
            environment = args.environment           # explicit shared env name
        else:
            from yggdrasil.databricks.environments.service import environment_stem
            environment = environment_stem("ygg")  # canonical version-pinned ygg image
        deploy = not args.no_deploy

        style.step(
            f"auto loader for {style.brand(table.full_name())} "
            f"{style.dim('←')} {style.bold(args.source or 'staging')}"
        )
        bits = [f"format={args.file_format}",
                "continuous" if args.continuous else "available-now"]
        bits.append("file-arrival" if args.file_arrival else "no-trigger")
        bits.append(f"env={environment or 'inline'}")
        bits.append("bundle" if not args.no_bundle else "no-bundle")
        style.info(style.dim(" · ".join(bits)))

        with style.Spinner("building wheels + deploying serverless job"):
            result = table.auto_loader(
                source=args.source,
                name=args.name,
                file_format=args.file_format,
                checkpoint=args.checkpoint,
                available_now=not args.continuous,
                file_arrival=args.file_arrival,
                clean_source=args.clean_source,
                clean_source_retention=args.clean_source_retention,
                bundle_dependencies=not args.no_bundle,
                environment=environment,
                deploy=deploy,
            )

        if not deploy:
            # A configured (un-deployed) Flow — surface its shape.
            style.ok(f"configured {style.brand(getattr(result, 'name', '<flow>'))} "
                     f"{style.dim('(not deployed; --no-deploy)')}")
            return 0

        job = result
        job_id = getattr(job, "job_id", None)
        style.ok(f"deployed {style.brand(str(getattr(job, 'name', table.full_name())))} "
                 f"{style.dim('id=' + str(job_id))}")
        url = getattr(job, "explore_url", None)
        if url:
            sys.stdout.write(f"  {style.dim('url')}  {url}\n")

        if args.run:
            style.step("triggering run")
            run = job.run(wait=(args.wait or False), raise_error=False)
            run_id = getattr(run, "run_id", None)
            if args.wait:
                if getattr(run, "is_succeeded", False):
                    dur = getattr(run, "duration_seconds", None) or 0.0
                    style.ok(f"run {style.bold(str(run_id))} succeeded in {dur:.1f}s")
                else:
                    style.fail(f"run {run_id} {getattr(run, 'state', '?')} — "
                               f"{getattr(run, 'state_message', '') or ''}")
                    return 1
            else:
                style.ok(f"run {style.bold(str(run_id))} started")
        return 0
