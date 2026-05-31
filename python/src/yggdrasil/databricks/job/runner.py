"""``ygg-job`` — cluster-side entry point a deployed :class:`Flow` task invokes.

The python-wheel task runs ``ygg-job <kind> <args...>`` on the Databricks
cluster; :func:`main` dispatches to the right job body. Today the only kind is
``table-async-load`` (the :class:`~yggdrasil.databricks.table.async_job.TableJob`
loader), reconstructing the table from its full name via the runtime client.
"""
from __future__ import annotations

import logging
import sys
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = ["main", "run"]


def run(kind: str, args: Sequence[str]) -> int:
    """Dispatch a job *kind* with its positional *args*. Returns rows/count."""
    logger.info("ygg-job %s %s", kind, list(args))
    if kind == "table-async-load":
        from yggdrasil.databricks.client import DatabricksClient
        from yggdrasil.databricks.table.async_job import TableJob

        (full_name,) = args
        client = DatabricksClient()          # runtime auth on the cluster
        table = client.tables[full_name]
        processed = TableJob(table).run(wait=True)
        logger.info("table-async-load %s: %s ops", full_name, processed)
        return int(processed or 0)
    raise SystemExit(f"ygg-job: unknown job kind {kind!r}")


def _setup_logging() -> None:
    """Surface ygg INFO logs in the Databricks job output (stdout). Scoped to
    the ``yggdrasil`` logger so it doesn't clobber an existing root config."""
    ygg = logging.getLogger("yggdrasil")
    ygg.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in ygg.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        ygg.addHandler(handler)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _setup_logging()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("ygg-job: expected <kind> <args...>")
    kind, *rest = argv
    result = run(kind, rest)
    # Always surface a result line, even if logging is filtered.
    print(f"ygg-job {kind} {rest} -> {result}", flush=True)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
