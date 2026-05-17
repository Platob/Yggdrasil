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
* :func:`task_parameters` rolls all three plus the ``DATABRICKS_*``
  process environment into a single :class:`TaskParameters` snapshot —
  one call returns ``(args, kwargs, env)`` so a staged callable can
  resolve every input channel without picking the right reader by
  hand.

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
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple, Union

__all__ = [
    "TaskParameters",
    "get_dbutils",
    "read_argv",
    "read_widgets",
    "read_job_parameters",
    "task_parameters",
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


#: Default env-var prefixes captured by :func:`task_parameters` into
#: :attr:`TaskParameters.env`. Covers the per-run identity vars
#: Databricks injects into a task's process environment
#: (``DATABRICKS_JOB_ID`` / ``DATABRICKS_RUN_ID`` /
#: ``DATABRICKS_TASK_KEY`` / …) plus anything the task spec adds via
#: ``spark_env_vars`` with the same prefix.
DEFAULT_TASK_ENV_PREFIXES: Tuple[str, ...] = ("DATABRICKS_",)


@dataclass(frozen=True)
class TaskParameters:
    """Snapshot of a Databricks task's runtime inputs across every channel.

    A staged callable receives values through up to three channels —
    positional ``sys.argv`` (``SparkPythonTask.parameters``), named
    bindings (notebook widgets + ``{{job.parameters.*}}``
    substitutions), and the task's process environment. This struct
    rolls all three into one place so callers don't have to pick the
    right reader by hand.

    Attributes:
        args: Positional tokens from ``sys.argv[1:]`` — every entry
            that isn't a ``--flag`` / ``--key=value`` pair. Preserves
            order so ``SparkPythonTask.parameters`` indexing still
            works.
        kwargs: Named bindings, layered (lowest precedence first):
            ``dbutils.notebook.entry_point.getCurrentBindings()`` (the
            union of widget values and ``{{job.parameters.*}}``
            substitutions) overlaid with ``--key=value`` / ``--key
            value`` / ``--flag`` pairs parsed from ``sys.argv``. Argv
            wins on collision — a caller who passed ``--foo=cli`` on
            the command line means it.
        env: Process environment variables whose key starts with one
            of the configured prefixes (default ``DATABRICKS_``).
    """

    args: Tuple[str, ...] = ()
    kwargs: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


def task_parameters(
    *,
    argv: Optional[list[str]] = None,
    env_prefix: Union[str, Sequence[str]] = DEFAULT_TASK_ENV_PREFIXES,
    require_dbutils: bool = False,
) -> TaskParameters:
    """Collect every input channel a Databricks task exposes into one snapshot.

    Reads, in order:

    1. ``sys.argv[1:]`` (or *argv* when supplied) — positional tokens
       land in :attr:`TaskParameters.args`, ``--key=value`` / ``--key
       value`` / ``--flag`` pairs land in :attr:`kwargs`.
    2. ``dbutils.notebook.entry_point.getCurrentBindings()`` — the
       union of widget values and ``{{job.parameters.*}}``
       substitutions. Merged into :attr:`kwargs` *under* the argv
       layer, so an explicit ``--key=cli`` overrides a job-parameter
       binding of the same name. Silently skipped when ``dbutils`` is
       not on the path (typical local re-run); set *require_dbutils*
       to raise :class:`RuntimeError` instead.
    3. ``os.environ`` — keys starting with any *env_prefix* (default
       ``DATABRICKS_``) land in :attr:`env`. Pass an empty tuple to
       skip the env capture entirely, or a custom prefix list to also
       pick up ``spark_env_vars`` your task spec sets with a different
       prefix.

    Pass the *argv* / *env_prefix* kwargs explicitly in unit tests so
    the snapshot doesn't pick up the harness's own ``sys.argv`` /
    ``os.environ``.
    """
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    args_list: list[str] = []
    kwargs: dict[str, str] = {}
    i = 0
    while i < len(raw_argv):
        token = raw_argv[i]
        if token.startswith("--"):
            tail = token[2:]
            if "=" in tail:
                key, value = tail.split("=", 1)
                kwargs[key] = value
                i += 1
                continue
            if i + 1 < len(raw_argv) and not raw_argv[i + 1].startswith("--"):
                kwargs[tail] = raw_argv[i + 1]
                i += 2
                continue
            kwargs[tail] = "true"
            i += 1
            continue
        args_list.append(token)
        i += 1

    bindings: dict[str, str] = {}
    dbutils = get_dbutils()
    if dbutils is None:
        if require_dbutils:
            raise RuntimeError(
                "task_parameters(require_dbutils=True): dbutils is not "
                "available — this snapshot only includes argv + env. Run "
                "inside a Databricks task or drop require_dbutils."
            )
    else:
        try:
            bindings = read_job_parameters()
        except RuntimeError as exc:
            # ``getCurrentBindings()`` is only wired inside a notebook
            # run; SparkPythonTask invocations have dbutils on the
            # path but no notebook entry point. Fall back to widgets
            # if any are declared, otherwise carry on with argv-only.
            LOGGER.debug(
                "task_parameters: getCurrentBindings() unavailable (%s) — "
                "falling back to argv + env only", exc,
            )
    # argv wins on collision: an explicit ``--key=cli`` is the caller
    # overriding the job parameter for this run.
    merged_kwargs = {**bindings, **kwargs}

    prefixes: Tuple[str, ...] = (
        (env_prefix,) if isinstance(env_prefix, str) else tuple(env_prefix)
    )
    if prefixes:
        env = {
            k: v for k, v in os.environ.items()
            if any(k.startswith(p) for p in prefixes)
        }
    else:
        env = {}

    return TaskParameters(
        args=tuple(args_list),
        kwargs=merged_kwargs,
        env=env,
    )
