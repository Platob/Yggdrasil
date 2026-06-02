"""Databricks Genie service.

``client.genie.ask("which region sold the most last quarter?")`` is the
headline. The service holds a :class:`GenieDefaults` so callers set the
default ``space_id`` once and stop repeating it::

    from dataclasses import replace
    client.genie.defaults = replace(client.genie.defaults, space_id="01ef…")

    answer = client.genie.ask("top 5 customers by revenue this year")
    print(answer.text)            # natural-language summary
    print(answer.sql)             # the SQL Genie generated
    df = answer.to_polars()       # the result as a polars DataFrame

    # Drive a multi-turn conversation
    conv, first = client.genie.space().start_conversation("revenue by month")
    nxt = conv.ask("now just for EMEA")

    # Or let the agent act on its own
    run = client.genie.agent().run("explain the Q3 revenue dip")
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator, Optional

from yggdrasil.databricks.service import DatabricksService
from yggdrasil.dataclasses import WaitingConfig, WaitingConfigArg

from .resources import (
    DEFAULT_GENIE_WAIT,
    GenieAnswer,
    GenieConversation,
    GenieDefaults,
    GenieSpace,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from databricks.sdk.service.dashboards import GenieAPI

    from .agent import GenieAgent


__all__ = ["Genie"]


LOGGER = logging.getLogger(__name__)


class Genie(DatabricksService):
    """High-level wrapper around the Databricks Genie API.

    Attributes
    ----------
    defaults
        :class:`GenieDefaults` — service-wide configuration. Replace via
        ``client.genie.defaults = replace(...)`` the same way the other
        Databricks services do.
    """

    def __init__(self, client=None, defaults: Optional[GenieDefaults] = None):
        super().__init__(client=client)
        self.defaults: GenieDefaults = defaults if defaults is not None else GenieDefaults()

    # ------------------------------------------------------------------ #
    # SDK boundary
    # ------------------------------------------------------------------ #
    @property
    def api(self) -> "GenieAPI":
        return self.client.workspace_client().genie

    # ------------------------------------------------------------------ #
    # Space resolution
    # ------------------------------------------------------------------ #
    def space(self, space_id: Optional[str] = None) -> GenieSpace:
        """Return a :class:`GenieSpace` handle.

        ``space_id`` defaults to :attr:`GenieDefaults.space_id`.
        """
        sid = space_id or self.defaults.space_id
        if not sid:
            raise ValueError(
                "No space_id given and Genie.defaults.space_id is unset. "
                "Pass space_id=... or set the default."
            )
        return GenieSpace(service=self, space_id=sid)

    def list_spaces(self) -> Iterator[GenieSpace]:
        """Iterate over Genie spaces visible to the current principal."""
        token: Optional[str] = None
        while True:
            resp = self.api.list_spaces(page_token=token)
            for info in getattr(resp, "spaces", None) or []:
                sid = getattr(info, "space_id", None)
                if not sid:
                    continue
                yield GenieSpace(service=self, space_id=sid, details=info)
            token = getattr(resp, "next_page_token", None)
            if not token:
                break

    def find_space(self, *, title: str) -> Optional[GenieSpace]:
        """Return the first space whose title matches (case-insensitive)."""
        target = title.strip().lower()
        for space in self.list_spaces():
            if (space.title or "").strip().lower() == target:
                return space
        return None

    # ------------------------------------------------------------------ #
    # One-shot ask
    # ------------------------------------------------------------------ #
    def ask(
        self,
        question: str,
        *,
        space_id: Optional[str] = None,
        wait: WaitingConfigArg = None,
    ) -> GenieAnswer:
        """Ask a one-shot question against a space (starts a fresh conversation)."""
        return self.space(space_id).ask(question, wait=wait)

    def conversation(
        self, conversation_id: str, *, space_id: Optional[str] = None,
    ) -> GenieConversation:
        """Return a handle to an existing conversation in a space."""
        return self.space(space_id).conversation(conversation_id)

    # ------------------------------------------------------------------ #
    # Agent
    # ------------------------------------------------------------------ #
    def agent(self, *, space_id: Optional[str] = None, **kwargs) -> "GenieAgent":
        """Return a :class:`~.agent.GenieAgent` bound to a space."""
        from .agent import GenieAgent

        return GenieAgent(space=self.space(space_id), **kwargs)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _resolve_wait(self, override: WaitingConfigArg) -> WaitingConfig:
        if override is None or override is True:
            return self.defaults.wait or DEFAULT_GENIE_WAIT
        return WaitingConfig.from_(override)
