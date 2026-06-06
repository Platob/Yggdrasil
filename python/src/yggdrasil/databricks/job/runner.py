"""``ygg run`` — the subcommand a deployed ``@task`` / ``@flow`` runs *as* on the cluster.

A serverless python-wheel task invokes this entry point (``package_name="ygg"``,
``entry_point="ygg"`` → ``ygg run``) with the **target** of the decorated object plus
either a pickled argument payload (the transparent ``__call__`` dispatch from a
laptop) or positional string parameters (a scheduled / triggered deploy)::

    ygg run pkg.flows:etl --payload <ws>/payload.pkl --result <ws>/result.pkl
    ygg run pkg.flows:etl 2024-01-01 7          # scheduled: string params

It imports the target, binds the arguments to the function signature — coercing
each to its annotation via :func:`yggdrasil.data.cast.convert` (so string job
parameters become ints/dates/dataclasses, and pickled args are a no-op identity)
— runs the body **in-process** (``.local`` bypasses the remote routing, since
we're already inside Databricks), and, when a result path is given, writes the
pickled return value back for the dispatching driver to read.

This is the counterpart to :meth:`yggdrasil.databricks.job.skeleton._Runnable.__call__`:
the laptop side ships args here; this side reconstructs the typed call.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import logging
import pickle
import sys
import typing
from typing import Any, Sequence

logger = logging.getLogger(__name__)


def resolve_target(target: str) -> Any:
    """Import ``module.path:qualname`` and return the referenced object — the
    decorated :class:`Task` / :class:`Flow` instance (or its class, or a plain
    function). A class-based flow is instantiated by the caller."""
    module_path, _, qualname = target.partition(":")
    obj: Any = importlib.import_module(module_path)
    for part in filter(None, qualname.split(".")):
        obj = getattr(obj, part)
    return obj


def bind_and_convert(fn: Any, args: Sequence[Any], kwargs: dict) -> inspect.BoundArguments:
    """Bind *args* / *kwargs* to ``fn``'s signature and coerce each declared
    parameter to its annotation via the cast registry. ``*args`` / ``**kwargs``
    params are left untouched. Already-typed (pickled) values convert by
    identity; string job parameters get parsed into their typed form."""
    from yggdrasil.data.cast import convert

    sig = inspect.signature(fn)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    # Resolve string annotations (``from __future__ import annotations`` / forward
    # refs) to real types; fall back to the raw annotations if resolution fails.
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # noqa: BLE001 - unresolvable hints → coerce only what we can
        hints = getattr(fn, "__annotations__", {}) or {}
    for name, param in sig.parameters.items():
        if name not in bound.arguments:
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        hint = hints.get(name)
        if isinstance(hint, type) or typing.get_origin(hint) is not None:
            bound.arguments[name] = convert(bound.arguments[name], hint)
    return bound


def main(argv: "Sequence[str] | None" = None) -> int:
    from yggdrasil.databricks.client import DatabricksClient
    from yggdrasil.databricks.job.skeleton import ensure_console_logging
    from yggdrasil.databricks.path import DatabricksPath

    parser = argparse.ArgumentParser(prog="ygg run", description="Run a deployed @task/@flow.")
    parser.add_argument("target", help="module.path:qualname of the decorated task/flow")
    parser.add_argument("--payload", help="workspace path to a pickled (args, kwargs) tuple")
    parser.add_argument("--result", help="workspace path to write the pickled return value to")
    parser.add_argument("params", nargs="*", help="positional string parameters (scheduled runs)")
    ns = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])

    ensure_console_logging()  # surface ygg logs in the job task output
    logger.info("ygg run target=%s", ns.target)

    import importlib

    # A ready-to-use client for the task: registered as the process-wide current
    # client (so DatabricksClient.current() returns it everywhere) and injected as
    # a module global ``databricks`` in the target's module, so a bare
    # ``databricks.sql(...)`` in the body just works — built once, reused.
    client = DatabricksClient()
    DatabricksClient.set_current(client)
    module = importlib.import_module(ns.target.partition(":")[0])
    if not hasattr(module, "databricks"):
        setattr(module, "databricks", client)

    obj = resolve_target(ns.target)
    if isinstance(obj, type):  # class-based flow → instantiate
        obj = obj()
    # The signature lives on the wrapped function (decorated) or the run() body
    # (class-based); the call goes through .local so retries apply but the
    # remote routing does not (we're already on the cluster).
    fn = getattr(obj, "fn", None)
    if fn is None:
        fn = getattr(obj, "run", obj)
    runner = getattr(obj, "local", obj)

    if ns.payload:
        raw = DatabricksPath.from_(ns.payload, client=client).read_bytes()
        args, kwargs = pickle.loads(raw)
    else:
        args, kwargs = tuple(ns.params), {}

    bound = bind_and_convert(fn, args, kwargs)
    result = runner(*bound.args, **bound.kwargs)

    if ns.result:
        out = DatabricksPath.from_(ns.result, client=client)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(pickle.dumps(result))
        logger.info("ygg run wrote result to %s", ns.result)
    logger.info("ygg run target=%s done", ns.target)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
