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

from dataclasses import replace as _dc_replace

from .resources import (
    DEFAULT_MANAGED_SPACE_TITLE,
    DEFAULT_WAIT,
    GenieAnswer,
    GenieConversation,
    GenieDefaults,
    GenieSpace,
    build_serialized_space,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from databricks.sdk.service.dashboards import GenieAPI
    from yggdrasil.databricks.warehouse.warehouse import SQLWarehouse

    from .agent import GenieAgent
    from .autonomous import AutonomousAgent


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
        self.defaults: GenieDefaults = (
            defaults if defaults is not None else GenieDefaults()
        )
        self._agent: "Optional[GenieAgent]" = None
        self._autonomous_agent: "Optional[AutonomousAgent]" = None

    # ------------------------------------------------------------------ #
    # SDK boundary
    # ------------------------------------------------------------------ #
    @property
    def api(self) -> "GenieAPI":
        return self.client.workspace_client().genie

    # ------------------------------------------------------------------ #
    # Local agent
    # ------------------------------------------------------------------ #
    @property
    def agent(self) -> "GenieAgent":
        """Lazily-built :class:`GenieAgent` for local orchestration.

        One instance per :class:`Genie` service — its history and
        registered tools persist across calls within a single process.
        Construct directly with ``GenieAgent(genie_service)`` if you
        need a separate session.
        """
        cached = self._agent
        if cached is None:
            from .agent import GenieAgent

            cached = GenieAgent(service=self)
            self._agent = cached
        return cached

    @property
    def autonomous_agent(self) -> "AutonomousAgent":
        """Lazily-built :class:`AutonomousAgent` for autonomous orchestration.

        One instance per :class:`Genie` service — history, children, and
        registered tools persist across calls.  Construct directly with
        ``AutonomousAgent(genie_service)`` if you need a separate session.
        """
        cached = self._autonomous_agent
        if cached is None:
            from .autonomous import AutonomousAgent

            cached = AutonomousAgent(service=self)
            self._autonomous_agent = cached
        return cached

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
                "Creating Genie message in space %s on conversation %s (len=%d)",
                space_id,
                conversation_id,
                len(question),
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
                "Starting Genie conversation in space %s (len=%d)",
                space_id,
                len(question),
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
            title = (
                getattr(space.details, "title", None)
                if space._details is not None
                else None
            )
            if title is None:
                title = getattr(
                    self.api.get_space(space_id=space.space_id), "title", None
                )
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
    # Ensure / cleanup — the "easiest interaction" path
    # ------------------------------------------------------------------ #
    def ensure_space(self) -> GenieSpace:
        """Resolve, or create, the default :class:`GenieSpace`.

        Resolution order:

        1. :attr:`GenieDefaults.space_id`, when set, returns immediately.
        2. :meth:`_pick_space_id` walks :meth:`list_spaces` (filtered by
           :attr:`GenieDefaults.space_name` when set; otherwise matched by
           :attr:`GenieDefaults.managed_space_title` first, then the first
           listed space).
        3. When neither step resolves an id and
           :attr:`GenieDefaults.auto_create_space` is ``True``, a fresh
           space is created via :meth:`create_space` using the
           ``managed_space_*`` fields on the defaults.

        On a successful resolve / create, the discovered ``space_id`` is
        cached back onto :attr:`defaults` so subsequent :meth:`ask` calls
        skip the listing round trip. When
        :attr:`GenieDefaults.cleanup_dead_spaces` is also ``True``,
        duplicates are trashed before the resolved id is returned.
        """
        if self.defaults.space_id:
            space = GenieSpace(service=self, space_id=self.defaults.space_id)
            if self.defaults.cleanup_dead_spaces:
                self.cleanup_dead_spaces()
            return space

        picked = self._pick_space_id()
        if picked is None:
            if not self.defaults.auto_create_space:
                raise ValueError(
                    "Could not resolve a Genie space. Either pass space_id=..., "
                    "set Genie.defaults.space_id, or enable "
                    "Genie.defaults.auto_create_space=True (and configure "
                    "managed_space_tables / warehouse_id) to create one."
                )
            picked = self._create_managed_space().space_id

        # Cache the resolved id on the frozen defaults so the next ask()
        # bypasses list_spaces entirely.
        self.defaults = _dc_replace(self.defaults, space_id=picked)

        if self.defaults.cleanup_dead_spaces:
            self.cleanup_dead_spaces()
        return GenieSpace(service=self, space_id=picked)

    def cleanup_dead_spaces(self) -> list[str]:
        """Trash duplicate managed-title Genie spaces.

        Finds every space whose ``title`` matches
        :attr:`GenieDefaults.managed_space_title`, keeps the one whose id
        matches :attr:`GenieDefaults.space_id` (or, when that's unset, the
        first one listed), and trashes the rest. Returns the list of
        trashed ``space_id`` values so callers can log / report what was
        removed.

        Idempotent: running again after a clean workspace returns ``[]``.
        """
        title = self.defaults.managed_space_title
        if not title:
            return []

        keepers = self.defaults.space_id
        matches: list[tuple[str, Optional[str]]] = []
        for space in self.list_spaces():
            entry_title = (
                getattr(space._details, "title", None)
                if space._details is not None
                else None
            )
            if entry_title == title:
                matches.append((space.space_id, entry_title))

        if len(matches) <= 1:
            return []

        # Pick the survivor: the configured default if present, else the
        # first listed (Genie's list ordering is stable per call).
        if keepers and any(sid == keepers for sid, _ in matches):
            survivor = keepers
        else:
            survivor = matches[0][0]

        trashed: list[str] = []
        for sid, _ in matches:
            if sid == survivor:
                continue
            LOGGER.info(
                "Trashing dead Genie space %r (duplicate of survivor %r, title=%r)",
                sid,
                survivor,
                title,
            )
            self.api.trash_space(space_id=sid)
            trashed.append(sid)
        return trashed

    def _resolve_managed_title(self) -> str:
        """Build the managed space title, scoped to the current user.

        Returns ``"Yggdrasil Genie — <username>"`` so each user gets
        their own auto-created space.  Falls back to the plain default
        when the identity can't be resolved.
        """
        base = self.defaults.managed_space_title or DEFAULT_MANAGED_SPACE_TITLE
        try:
            slug = self.client.user_scoped_name("", separator="").strip("-")
            if slug:
                return f"{base} — {slug}"
        except Exception:
            pass
        return base

    def _create_managed_space(self) -> GenieSpace:
        """Create a Genie space using the ``managed_space_*`` defaults.

        When ``warehouse_id`` is unset, falls back to the workspace
        default warehouse via :meth:`resolve_warehouse`.
        """
        if not self.defaults.managed_space_tables:
            raise ValueError(
                "Cannot auto-create a Genie space without "
                "Genie.defaults.managed_space_tables — Genie requires at "
                "least one fully-qualified `catalog.schema.table` to expose."
            )
        wh_id = self.defaults.warehouse_id
        if not wh_id:
            wh_id = self.resolve_warehouse().warehouse_id
        serialized = build_serialized_space(
            tables=self.defaults.managed_space_tables,
            text_instructions=self.defaults.managed_space_instructions,
        )
        title = self._resolve_managed_title()
        LOGGER.info(
            "Creating Genie space %r (tables=%r, warehouse_id=%r)",
            title,
            tuple(self.defaults.managed_space_tables),
            wh_id,
        )
        return self.create_space(
            warehouse_id=wh_id,
            serialized_space=serialized,
            title=title,
            description=self.defaults.managed_space_description,
            parent_path=self.defaults.managed_space_parent_path,
        )

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
        if picked:
            # Cache so the next ask() skips the list round trip.
            self.defaults = _dc_replace(self.defaults, space_id=picked)
            return picked

        if self.defaults.auto_create_space:
            created = self._create_managed_space()
            self.defaults = _dc_replace(self.defaults, space_id=created.space_id)
            return created.space_id

        raise ValueError(
            "Could not auto-pick a Genie space. Either set "
            "Genie.defaults.space_id, pass space_id=..., enable "
            "Genie.defaults.auto_create_space=True (with "
            "managed_space_tables + warehouse_id), or create a Genie "
            "space in this workspace."
        )

    def _pick_space_id(self) -> Optional[str]:
        # Resolution priority:
        # 1. Exact match on ``defaults.space_name`` (explicit caller intent).
        # 2. Exact match on the user-scoped managed title (re-pick the
        #    space we previously auto-created for this user).
        # 3. Exact match on the base managed title (fallback for spaces
        #    created before user-scoping was added).
        # 4. First listed space — only when neither bias is set, so we
        #    never silently switch a configured workspace.
        wanted_title = self.defaults.space_name
        user_title = self._resolve_managed_title()
        base_title = self.defaults.managed_space_title
        first_id: Optional[str] = None
        user_managed_id: Optional[str] = None
        base_managed_id: Optional[str] = None

        for space in self.list_spaces():
            title = getattr(space._details, "title", None) if space._details else None
            if wanted_title is not None:
                if title == wanted_title:
                    return space.space_id
                continue
            if first_id is None:
                first_id = space.space_id
            if user_managed_id is None and title == user_title:
                user_managed_id = space.space_id
            if base_managed_id is None and base_title and title == base_title:
                base_managed_id = space.space_id

        if wanted_title is not None:
            return None
        return user_managed_id or base_managed_id or first_id
