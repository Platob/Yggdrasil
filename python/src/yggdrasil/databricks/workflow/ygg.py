"""The ``ygg`` runtime module — helpers callable from staged workflow tasks.

This module is the cluster-side surface every staged task imports as
``ygg``. The renderer in :mod:`yggdrasil.databricks.jobs.task` adds
``from yggdrasil.databricks.workflow import ygg`` to every staged
script; :func:`SecretRef.__repr__` and :func:`TaskNode.__repr__` emit
literal ``ygg.secret(...)`` / ``ygg.task_value(...)`` expressions so
the existing :func:`stage_python_callable` rendering pipeline doesn't
need a special-case hook.

Three runtime affordances live here:

* :func:`secret` resolves ``{secrets/<scope>/<key>}`` at call time
  against :meth:`DatabricksClient.current`. Used to materialise a
  :class:`~yggdrasil.databricks.workflow.resources.SecretRef` default
  the moment the task body needs it, so cleartext never lives in the
  staged source on disk.
* :func:`task_value` reads a value an upstream task published via
  ``dbutils.jobs.taskValues.set``. Used to forward a
  :class:`~yggdrasil.databricks.workflow.nodes.TaskNode` return value
  from one task to the next; falls back to ``default`` outside a
  Databricks task host so a local re-run still parses.
* :func:`publish_return` is the inverse — wraps the staged ``func(...)``
  invocation so its return value lands on the run's task-values map
  for downstream tasks.

Users can also import the same surface inside a task body for explicit
use::

    from yggdrasil.databricks.workflow import ygg

    @task
    def step():
        api_key = ygg.secret("vendor", "key")
        upstream = ygg.task_value("extract")
        ...
"""
from __future__ import annotations

import logging
from typing import Any, Optional

__all__ = [
    "RETURN_VALUE_KEY",
    "publish_return",
    "secret",
    "task_value",
]

LOGGER = logging.getLogger(__name__)

#: Default task-value key under which :func:`publish_return` stores
#: the staged function's return value. Downstream tasks read the same
#: key via :func:`task_value` — kept private (double-underscore prefix)
#: so it can't collide with caller-set task values.
RETURN_VALUE_KEY = "__ygg_return__"


def secret(scope: str, key: str, *, host: Optional[str] = None) -> str:
    """Resolve ``{secrets/<scope>/<key>}`` against the current workspace client.

    Returns the decoded string value. Raises :class:`RuntimeError`
    when no :class:`DatabricksClient` is bound to the current process
    (the staged task ran outside a Databricks cluster and no client
    was explicitly registered).

    ``host`` (optional) targets a workspace other than the
    process-default :meth:`DatabricksClient.current`. A fresh
    :class:`DatabricksClient` is built for that host on the spot,
    inheriting authentication from the environment (token, OAuth
    client credentials, profile) the same way the default client
    does. Used to make a single staged task resolve secrets from a
    workspace different from the one it's running in.
    """
    from yggdrasil.databricks.client import DatabricksClient

    if host:
        client = DatabricksClient(host=host)
    else:
        client = DatabricksClient.current()
    if client is None:
        raise RuntimeError(
            f"yggdrasil.databricks.workflow.ygg.secret({scope!r}, {key!r}): "
            "no DatabricksClient is bound to the current process — staged "
            "tasks resolve secrets via DatabricksClient.current(). Run inside "
            "a Databricks task or wrap the call in a DatabricksClient context."
        )
    return client.secrets[f"{scope}/{key}"].svalue()


def task_value(
    task_key: str,
    value_key: str = RETURN_VALUE_KEY,
    *,
    default: Any = None,
) -> Any:
    """Read a value an upstream task published via ``dbutils.jobs.taskValues``.

    Returns *default* when ``dbutils`` is not reachable (typical for a
    local ``python <script>`` re-run) or the key was never set —
    matches the Databricks ``debugValue`` semantics.
    """
    from yggdrasil.databricks.jobs.inputs import get_dbutils

    dbutils = get_dbutils()
    if dbutils is None:
        return default
    try:
        return dbutils.jobs.taskValues.get(
            taskKey=task_key, key=value_key, default=default,
        )
    except Exception:  # noqa: BLE001 — be permissive at runtime
        LOGGER.debug(
            "task_value(taskKey=%r, key=%r): dbutils.jobs.taskValues.get "
            "raised — returning default",
            task_key, value_key, exc_info=True,
        )
        return default


def publish_return(
    value: Any,
    *,
    value_key: str = RETURN_VALUE_KEY,
) -> Any:
    """Publish *value* on the current task's task-values map and return it.

    Silently no-ops when ``dbutils`` isn't reachable so a local
    ``python <script>`` re-run of the staged source still terminates
    normally. The original *value* is always returned so the wrapped
    invocation reads naturally:

        _ygg_result = ygg.publish_return(func(...))
    """
    from yggdrasil.databricks.jobs.inputs import get_dbutils

    dbutils = get_dbutils()
    if dbutils is None:
        return value
    try:
        dbutils.jobs.taskValues.set(key=value_key, value=value)
    except Exception:  # noqa: BLE001 — be permissive at runtime
        LOGGER.debug(
            "publish_return(value_key=%r): dbutils.jobs.taskValues.set "
            "raised — return value will not be visible to downstream tasks",
            value_key, exc_info=True,
        )
    return value
