"""Databricks Genie service.

``client.genie.ask("How many orders last month?")`` is the entire API surface
most callers need. The service auto-resolves a default space (via
:attr:`Genie.defaults`), waits for Genie to finish, and returns a rich
:class:`GenieAnswer` that can produce the SQL result as Arrow / Polars / pandas.

Higher-level helpers live on the returned resources:

- ``answer.ask(follow_up)`` continues the conversation.
- ``space = client.genie.space("…")`` gives direct access to one space.
- ``space.start_conversation(q)`` returns ``(GenieConversation, GenieAnswer)``
  so the thread can be continued explicitly.

Defaults
--------
Tweak the service-level :class:`GenieDefaults` once and every subsequent call
inherits them::

    from dataclasses import replace
    from yggdrasil.dataclasses import WaitingConfig
    client.genie.defaults = replace(
        client.genie.defaults,
        space_id="01ef...",
        warehouse_id="abc123",
        wait=WaitingConfig.from_(300),
    )
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator, Optional

from yggdrasil.databricks.client import DatabricksService
from yggdrasil.dataclasses import WaitingConfig, WaitingConfigArg

from .resources import (
    DEFAULT_POLL_INTERVAL_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_WAIT,
    GenieAnswer,
    GenieConversation,
    GenieDefaults,
    GenieSpace,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from databricks.sdk.service.dashboards import GenieAPI, GenieMessage
    from yggdrasil.databricks.warehouse.warehouse import SQLWarehouse


__all__ = ["Genie"]

LOGGER = logging.getLogger(__name__)


class Genie(DatabricksService):
    """High-level wrapper around Databricks Workspace Genie APIs.

    The simplest path is::

        client.genie.defaults = replace(client.genie.defaults, space_id="…")
        answer = client.genie.ask("How many orders last month?")
        print(answer.text)
        df = answer.polars()  # SQL result, if Genie ran one

    Without a configured default the service can still auto-pick a space when
    :attr:`GenieDefaults.auto_pick_space` is on (the default). Set
    :attr:`GenieDefaults.space_name` to bias the pick by title.

    Attributes
    ----------
    defaults
        :class:`GenieDefaults` — service-wide configuration. Replace in place
        via ``client.genie.defaults = replace(client.genie.defaults, …)``.
    """

    def __init__(
        self,
        client=None,
        defaults: Optional[GenieDefaults] = None,
    ):
        super().__init__(client=client)
        self.defaults: GenieDefaults = defaults if defaults is not None else GenieDefaults()

    # ------------------------------------------------------------------ #
    # SDK boundary
    # ------------------------------------------------------------------ #
    @property
    def api(self) -> "GenieAPI":
        return self.client.workspace_client().genie

    # ------------------------------------------------------------------ #
    # The one-shot ask
    # ------------------------------------------------------------------ #
    def ask(
        self,
        question: str,
        *,
        space_id: str | None = None,
        conversation_id: str | None = None,
        wait: WaitingConfigArg = None,
    ) -> GenieAnswer:
        """Send a question to Genie and return the completed reply.

        Parameters
        ----------
        question
            The user's natural-language question.
        space_id
            Genie space id. Falls back to :attr:`GenieDefaults.space_id`, then
            to :meth:`default_space` (when ``auto_pick_space`` is enabled).
        conversation_id
            Continue an existing conversation. When ``None``, a new one is
            started.
        wait
            Per-call wait budget. Anything :meth:`WaitingConfig.from_`
            accepts works — a number of seconds, a :class:`datetime.timedelta`,
            a deadline ``datetime``, an options dict, or a full
            :class:`WaitingConfig`. When ``None`` the
            :attr:`GenieDefaults.wait` budget is used.
        """
        space_id = self._resolve_space_id(space_id)
        wait_cfg = self._resolve_wait(wait)
        timeout = wait_cfg.timeout_timedelta

        if conversation_id:
            LOGGER.debug(
                "Genie.create_message space=%s conversation=%s len=%d",
                space_id, conversation_id, len(question),
            )
            waiter = self.api.create_message(
                space_id=space_id,
                conversation_id=conversation_id,
                content=question,
            )
            message = waiter.result(timeout=timeout)
            resolved_conversation_id = conversation_id
        else:
            LOGGER.debug(
                "Genie.start_conversation space=%s len=%d",
                space_id, len(question),
            )
            waiter = self.api.start_conversation(
                space_id=space_id,
                content=question,
            )
            # Wait.bind() exposes the kwargs captured at submission time
            # (conversation_id, message_id, space_id). We pull conversation_id
            # before .result() blocks so callers can build follow-ups even
            # when the wait times out before completion.
            resolved_conversation_id = waiter.bind().get("conversation_id") or ""
            message = waiter.result(timeout=timeout)

        # GenieMessage carries the canonical conversation_id once Genie has
        # processed the request — prefer it over the bind() snapshot.
        message_conv_id = getattr(message, "conversation_id", None)
        if message_conv_id:
            resolved_conversation_id = message_conv_id

        return GenieAnswer(
            service=self,
            space_id=getattr(message, "space_id", None) or space_id,
            conversation_id=resolved_conversation_id,
            message=message,
        )

    # ------------------------------------------------------------------ #
    # Space resolution
    # ------------------------------------------------------------------ #
    def space(self, space_id: str | None = None) -> GenieSpace:
        """Return a :class:`GenieSpace` handle.

        ``space_id`` defaults to :attr:`GenieDefaults.space_id`; if neither is
        set and :attr:`GenieDefaults.auto_pick_space` is on, a space is picked
        from :meth:`list_spaces` (optionally filtered by
        :attr:`GenieDefaults.space_name`).
        """
        resolved = self._resolve_space_id(space_id)
        return GenieSpace(service=self, space_id=resolved)

    def default_space(self) -> GenieSpace:
        """Return the default :class:`GenieSpace`."""
        return self.space(None)

    def find_space(self, *, name: str) -> Optional[GenieSpace]:
        """Return the first space whose title matches ``name``, or ``None``."""
        for space in self.list_spaces():
            title = getattr(space.details, "title", None) if space._details is not None else None
            if title is None:
                title = getattr(self.api.get_space(space_id=space.space_id), "title", None)
            if title == name:
                return space
        return None

    def list_spaces(self, *, page_size: int | None = None) -> Iterator[GenieSpace]:
        """Iterate over Genie spaces accessible to the current identity."""
        response = self.api.list_spaces(page_size=page_size)
        for entry in getattr(response, "spaces", None) or []:
            entry_id = getattr(entry, "space_id", None) or getattr(entry, "id", None)
            if entry_id is None:
                continue
            yield GenieSpace(service=self, space_id=entry_id, details=entry)

    # ------------------------------------------------------------------ #
    # Space lifecycle (thin SDK passthrough — kept on the service for
    # discoverability via ``client.genie.create_space(...)``)
    # ------------------------------------------------------------------ #
    def create_space(
        self,
        *,
        warehouse_id: str | None = None,
        serialized_space: str,
        title: str | None = None,
        description: str | None = None,
        parent_path: str | None = None,
    ) -> GenieSpace:
        """Create a new Genie space."""
        warehouse_id = warehouse_id or self.defaults.warehouse_id
        if not warehouse_id:
            raise ValueError(
                "warehouse_id is required to create a Genie space; "
                "pass warehouse_id=... or set Genie.defaults.warehouse_id."
            )
        space = self.api.create_space(
            warehouse_id=warehouse_id,
            serialized_space=serialized_space,
            title=title,
            description=description,
            parent_path=parent_path,
        )
        space_id = getattr(space, "space_id", None) or getattr(space, "id", None)
        if not space_id:
            raise ValueError(f"Genie API returned no space id: {space!r}")
        return GenieSpace(service=self, space_id=space_id, details=space)

    def delete_space(self, space_id: str) -> None:
        """Move a Genie space to trash."""
        self.api.trash_space(space_id=space_id)

    # ------------------------------------------------------------------ #
    # Conversation passthroughs
    # ------------------------------------------------------------------ #
    def conversation(self, *, space_id: str, conversation_id: str) -> GenieConversation:
        return GenieConversation(
            service=self,
            space_id=space_id,
            conversation_id=conversation_id,
        )

    # ------------------------------------------------------------------ #
    # Warehouse resolution (for query-result materialisation)
    # ------------------------------------------------------------------ #
    def resolve_warehouse(self) -> "SQLWarehouse":
        """Return the warehouse used to wrap Genie query results.

        Resolution order: :attr:`GenieDefaults.warehouse_id` →
        ``client.sql.warehouse()`` (the workspace default).
        """
        if self.defaults.warehouse_id:
            return self.client.warehouses.find_warehouse(
                warehouse_id=self.defaults.warehouse_id,
            )
        return self.client.sql.warehouse()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _resolve_wait(self, override: WaitingConfigArg) -> WaitingConfig:
        if override is None:
            return self.defaults.wait or DEFAULT_WAIT
        return WaitingConfig.from_(override)

    def _resolve_space_id(self, space_id: str | None) -> str:
        if space_id:
            return space_id
        if self.defaults.space_id:
            return self.defaults.space_id
        if not self.defaults.auto_pick_space:
            raise ValueError(
                "No Genie space_id provided and auto_pick_space is disabled. "
                "Pass space_id=... or set Genie.defaults.space_id."
            )
        picked = self._pick_space_id()
        if not picked:
            raise ValueError(
                "Could not auto-pick a Genie space. Either set "
                "Genie.defaults.space_id, pass space_id=..., or "
                "create a Genie space in this workspace."
            )
        return picked

    def _pick_space_id(self) -> Optional[str]:
        wanted_title = self.defaults.space_name
        for space in self.list_spaces():
            if wanted_title is None:
                return space.space_id
            title = getattr(space._details, "title", None) if space._details else None
            if title == wanted_title:
                return space.space_id
        return None
