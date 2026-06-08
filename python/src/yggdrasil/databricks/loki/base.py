"""Shared base for the per-service Databricks Loki skills.

Each Databricks service module (``sql``, ``table``, ``volume``, ``job``, …)
ships its own ``loki.py`` defining the Loki skill(s) for that service — close
to the code they drive, for isolation. They all build on
:class:`DatabricksServiceSkill` here: it guards the workspace session, hands
the authenticated client over (``agent.databricks`` — Loki as token provider),
carries a precise domain **preprompt**, and offers the two small helpers every
service skill needs (``_names`` to summarize SDK objects, ``_tabular`` to take
a row set as a frame through the project's Tabular-convertible objects).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from yggdrasil.loki.skill import LokiSkill

if TYPE_CHECKING:
    from yggdrasil.loki import Loki

__all__ = ["DatabricksServiceSkill", "names", "tabular"]


def names(items: Any, *, attrs: tuple[str, ...] = ("name", "id", "full_name")) -> list[Any]:
    """Summarize an iterable of SDK/resource objects to their identifying field."""
    out: list[Any] = []
    for it in list(items or [])[:200]:
        for a in attrs:
            v = getattr(it, a, None)
            if v is not None:
                out.append(v)
                break
        else:
            out.append(str(it))
    return out


def tabular(result: Any) -> Any:
    """Take a row set as a polars frame **via the object's own conversion**.

    A Databricks statement result, a Genie answer, an :class:`IO` leaf — they
    are all Tabular-convertible, so we don't reshape data by hand: ask the
    object for the representation we want (``to_polars`` preferred, then pandas,
    then records). Returns the object unchanged when it isn't tabular.
    """
    for method in ("to_polars", "to_pandas", "to_pylist"):
        fn = getattr(result, method, None)
        if callable(fn):
            return fn()
    return result


class DatabricksServiceSkill(LokiSkill):
    """Base for the Databricks service skills — guards the session for all of them.

    The shared preprompt below is the *package-level* guideline; each service
    skill overrides :attr:`preprompt` with its own precise guidance so a model
    reasoning in that service's context reaches for the right ``dbc`` accessor.
    """

    requires = "databricks"
    preprompt = (
        "You are a Databricks expert operating through yggdrasil's DatabricksClient "
        "(the dbc.<service> accessors). Prefer serverless compute for inner I/O, "
        "Unity Catalog three-level names (catalog.schema.object), Arrow-returning "
        "SQL (results are Tabular — convert with to_polars/to_pylist), and the "
        "seeded ygg wheel environments (never %pip install per run). Be precise, "
        "least-privilege, and explicit about anything destructive or cost-bearing."
    )

    def _client(self, agent: "Loki") -> Any:
        client = agent.databricks
        if client is None:  # available() already guards; belt-and-suspenders
            raise RuntimeError("no Databricks session — run `ygg databricks configure`")
        return client
