"""Genie space resource — a single AI/BI Genie space, manipulated by code.

A :class:`GenieSpace` is the resource handle the :class:`Genie` service
hands back: it carries its bound client + ``space_id``, lazily fetches the
SDK ``GenieSpace`` info, and exposes the space lifecycle (get / update /
trash) plus conversation entry points (``ask`` / ``follow_up`` /
``conversations`` / ``messages``).
"""
from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Iterator, Optional

from .answer import GenieAnswer

if TYPE_CHECKING:
    from databricks.sdk.service.dashboards import (
        GenieConversation,
        GenieMessage,
        GenieSpace as SDKGenieSpace,
    )

    from ..client import DatabricksClient

__all__ = ["GenieSpace"]

# Genie completes a turn server-side; the SDK's *_and_wait helpers poll for
# us. Twenty minutes matches the SDK default and the Statement Execution
# window the result re-attach relies on.
_DEFAULT_TIMEOUT = datetime.timedelta(minutes=20)


class GenieSpace:
    """A single Genie space — resource-style lifecycle + Q&A."""

    def __init__(
        self,
        space_id: str,
        *,
        client: "DatabricksClient",
        info: "Optional[SDKGenieSpace]" = None,
    ):
        self.space_id = space_id
        self.client = client
        self._info = info

    @property
    def _api(self):
        return self.client.workspace_client().genie

    # -- info --------------------------------------------------------------

    @property
    def info(self) -> "SDKGenieSpace":
        """The SDK ``GenieSpace`` metadata (fetched once, then cached)."""
        if self._info is None:
            self._info = self._api.get_space(self.space_id)
        return self._info

    def refresh(self) -> "GenieSpace":
        """Re-fetch the space metadata from the workspace."""
        self._info = self._api.get_space(self.space_id)
        return self

    @property
    def title(self) -> Optional[str]:
        return self.info.title

    @property
    def description(self) -> Optional[str]:
        return self.info.description

    @property
    def warehouse_id(self) -> Optional[str]:
        return self.info.warehouse_id

    # -- lifecycle ---------------------------------------------------------

    def update(
        self,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        warehouse_id: Optional[str] = None,
        serialized_space: Optional[str] = None,
        parent_path: Optional[str] = None,
    ) -> "GenieSpace":
        """Update the space's settings; refreshes the cached info."""
        self._info = self._api.update_space(
            self.space_id,
            title=title,
            description=description,
            warehouse_id=warehouse_id,
            serialized_space=serialized_space,
            parent_path=parent_path,
            etag=getattr(self._info, "etag", None),
        )
        return self

    def trash(self) -> None:
        """Move the space to trash (delete)."""
        self._api.trash_space(self.space_id)

    # Alias for the conventional resource verb.
    delete = trash

    # -- conversations -----------------------------------------------------

    def ask(
        self,
        question: str,
        *,
        timeout: datetime.timedelta = _DEFAULT_TIMEOUT,
    ) -> GenieAnswer:
        """Start a new conversation, ask *question*, and wait for the answer."""
        message = self._api.start_conversation_and_wait(
            self.space_id, question, timeout=timeout,
        )
        return self._answer(message)

    def follow_up(
        self,
        conversation_id: str,
        question: str,
        *,
        timeout: datetime.timedelta = _DEFAULT_TIMEOUT,
    ) -> GenieAnswer:
        """Ask a follow-up *question* in an existing conversation."""
        message = self._api.create_message_and_wait(
            self.space_id, conversation_id, question, timeout=timeout,
        )
        return self._answer(message)

    def conversations(self) -> "Iterator[GenieConversation]":
        """Iterate the space's conversations (most recent first)."""
        return iter(self._api.list_conversations(self.space_id).conversations or [])

    def messages(self, conversation_id: str) -> "Iterator[GenieMessage]":
        """Iterate the messages of one conversation."""
        return iter(
            self._api.list_conversation_messages(
                self.space_id, conversation_id,
            ).messages or []
        )

    # -- internal ----------------------------------------------------------

    def _answer(self, message: "GenieMessage") -> GenieAnswer:
        return GenieAnswer(
            message,
            client=self.client,
            space_id=self.space_id,
            warehouse_id=self.warehouse_id,
        )

    def __repr__(self) -> str:
        title = self._info.title if self._info is not None else "?"
        return f"GenieSpace(space_id={self.space_id!r}, title={title!r})"
