"""Runtime input readers for Databricks Python tasks.

When a staged callable wakes up on the cluster side it needs to read
its parameters from whichever channel the caller used:

* :func:`read_argv` parses ``sys.argv[1:]`` as ``--key=value`` /
  ``--key value`` pairs. That's how
  :class:`~databricks.sdk.service.jobs.SparkPythonTask` parameters
  land — the values in ``parameters`` arrive positionally on the
  command line.
* :func:`read_widgets` reads ``dbutils.widgets`` entries. That's the
  notebook channel, and also what job parameters resolve to when a
  widget of the same name is declared in the notebook.
* :func:`read_job_parameters` returns every ``getCurrentBindings()``
  entry — the union of widget values and ``{{job.parameters.*}}``
  substitutions exposed through ``dbutils.notebook.entry_point``.

Every reader returns a ``dict[str, str]`` — strings are the native
shape on the Databricks side. Splat the result into a
``@checkargs``-wrapped function and the decorator coerces each value
to the declared annotation via
:func:`yggdrasil.data.cast.convert`::

    from yggdrasil.dataclasses.safe_function import checkargs
    from yggdrasil.databricks.jobs.inputs import read_argv

    @checkargs
    def run(name: str, count: int) -> None: ...

    if __name__ == "__main__":
        run(**read_argv())
"""
from __future__ import annotations

import builtins
import logging
import sys
from typing import Any, Optional

__all__ = [
    "get_dbutils",
    "read_argv",
    "read_widgets",
    "read_job_parameters",
]

LOGGER = logging.getLogger(__name__)


def get_dbutils() -> Optional[Any]:
    """Locate a live ``dbutils`` instance, or return ``None``.

    Looks in the order Databricks itself injects:

    1. ``builtins.dbutils`` (notebook + workflow runtime injection).
    2. IPython user namespace.
    3. Walks up the caller's globals (last-resort, fragile).

    Returns ``None`` when not running inside Databricks so callers
    can fall back to ``sys.argv`` / env vars without raising.
    """
    if hasattr(builtins, "dbutils"):
        return builtins.dbutils  # type: ignore[attr-defined]
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None and getattr(ip, "user_ns", None):
            dbutils = ip.user_ns.get("dbutils")
            if dbutils is not None:
                return dbutils
    except Exception:
        pass
    import inspect as _inspect

    frame = _inspect.currentframe()
    try:
        caller = frame.f_back if frame is not None else None
        for _ in range(5):
            if caller is None:
                break
            g = getattr(caller, "f_globals", None)
            if g and "dbutils" in g:
                return g["dbutils"]
            caller = caller.f_back
    finally:
        del frame
    return None


def read_argv(argv: Optional[list[str]] = None) -> dict[str, str]:
    """Parse ``--key=value`` / ``--key value`` pairs out of *argv*.

    *argv* defaults to ``sys.argv[1:]`` — the values Databricks
    passes to a :class:`SparkPythonTask`. Recognized shapes:

    * ``--key=value`` — single token, split on first ``=``.
    * ``--key value`` — two tokens, value must not itself start
      with ``--``.
    * ``--flag`` — bare flag with no value, stored as the string
      ``"true"`` so a ``bool``-annotated parameter coerces cleanly.

    Bare positional tokens (no ``--`` prefix) are silently skipped —
    callers that want positional inputs should consume ``sys.argv``
    directly.
    """
    if argv is None:
        argv = sys.argv[1:]
    out: dict[str, str] = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if token.startswith("--"):
            tail = token[2:]
            if "=" in tail:
                key, value = tail.split("=", 1)
                out[key] = value
                i += 1
                continue
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out[tail] = argv[i + 1]
                i += 2
                continue
            out[tail] = "true"
            i += 1
        else:
            i += 1
    return out


def read_widgets(*names: str) -> dict[str, str]:
    """Read ``dbutils.widgets.get(name)`` for each *name*.

    Use this from a Databricks notebook task or a Python task with
    declared widgets. Raises :class:`RuntimeError` when ``dbutils``
    is not available (i.e. you're not running inside Databricks)
    so the failure mode is loud instead of returning silently empty.
    """
    dbutils = get_dbutils()
    if dbutils is None or not hasattr(dbutils, "widgets"):
        raise RuntimeError(
            "read_widgets: dbutils.widgets is not available — this reader "
            "only works inside a Databricks runtime. Use read_argv() for "
            "SparkPythonTask parameters or read_job_parameters() for the "
            "full bindings view."
        )
    return {name: dbutils.widgets.get(name) for name in names}


def read_job_parameters() -> dict[str, str]:
    """Return every binding exposed via ``dbutils.notebook.entry_point``.

    Wraps ``dbutils.notebook.entry_point.getCurrentBindings()`` which
    returns the union of declared widget values and
    ``{{job.parameters.*}}`` substitutions for the current run.
    Raises :class:`RuntimeError` when ``dbutils`` is not on the path.
    Each value is coerced to ``str`` because the underlying Java map
    sometimes hands back ``None`` for unset bindings.
    """
    dbutils = get_dbutils()
    if dbutils is None:
        raise RuntimeError(
            "read_job_parameters: dbutils is not available — this reader "
            "only works inside a Databricks runtime."
        )
    try:
        bindings = dbutils.notebook.entry_point.getCurrentBindings()
    except Exception as exc:
        raise RuntimeError(
            "read_job_parameters: dbutils.notebook.entry_point."
            "getCurrentBindings() failed — bindings are typically only "
            "available inside a notebook run."
        ) from exc
    # ``getCurrentBindings`` hands back a Java Map; iterate keys via the
    # py4j ``keySet`` interface and coerce values to ``str``.
    if hasattr(bindings, "keySet"):
        return {str(k): str(bindings.get(k)) for k in bindings.keySet()}
    # In-process fakes / unit tests can hand back a plain dict.
    return {str(k): str(v) for k, v in dict(bindings).items()}
