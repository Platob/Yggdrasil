"""Genie resources and default configuration.

This module models Genie's conceptual hierarchy as Yggdrasil resources:

- :class:`GenieDefaults` — frozen config dataclass attached to the
  :class:`Genie` service via ``service.defaults``. Holds the workspace's
  default space id, polling cadence, timeout, and behavior toggles so
  callers can write ``client.genie.ask("…")`` once they have set it.
- :class:`GenieSpace` — a Genie *space* (the chatbot definition + warehouse
  binding). Exposes :meth:`GenieSpace.ask` (the one-shot path) and
  :meth:`GenieSpace.start_conversation`.
- :class:`GenieConversation` — a multi-turn conversation in a space.
  Supports :meth:`GenieConversation.ask` to continue and
  :meth:`GenieConversation.messages` to iterate history.
- :class:`GenieAnswer` — a single Genie reply. Carries the natural-language
  text, the SQL Genie generated (if any), and convenience methods to fetch
  the query result as Arrow / Polars / pandas via the existing warehouse
  statement-result plumbing.

The shape mirrors how Databricks itself documents Genie
(`spaces → conversations → messages`) so callers familiar with the UI find
the API immediately approachable.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Optional, Union

from yggdrasil.databricks.client import DatabricksResource
from yggdrasil.dataclasses import WaitingConfig
from yggdrasil.io.url import URL

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from databricks.sdk.service.dashboards import (
        GenieAttachment,
        GenieMessage,
        MessageStatus,
    )
    from databricks.sdk.service.sql import StatementResponse

    from yggdrasil.databricks.warehouse.statement import WarehouseStatementResult

    from .service import Genie


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "DEFAULT_WAIT",
    "DEFAULT_MANAGED_SPACE_TITLE",
    "DEFAULT_SERIALIZED_SPACE_VERSION",
    "GenieDefaults",
    "GenieAnswer",
    "GenieConversation",
    "GenieSpace",
    "GENIE_TERMINAL_STATUSES",
    "build_serialized_space",
]


LOGGER = logging.getLogger(__name__)

#: Default budget for Genie to finish ``ASKING_AI`` / ``EXECUTING_QUERY`` /
#: ``FETCHING_METADATA``. Matches Databricks's own SDK default (20 minutes)
#: but exposed as a knob in :class:`GenieDefaults` so users with chattier
#: spaces can lower it.
DEFAULT_TIMEOUT_SECONDS: float = 1200.0

#: Default polling cadence when the SDK has not finished a message yet. The
#: SDK's own waiter polls every second; we keep the same default.
DEFAULT_POLL_INTERVAL_SECONDS: float = 1.0

#: Canonical :class:`WaitingConfig` used by :class:`GenieDefaults`. Pre-built
#: so the dataclass default is a real singleton — replace the whole field via
#: ``replace(defaults, wait=WaitingConfig.from_(60))`` to tweak per call site.
DEFAULT_WAIT: WaitingConfig = WaitingConfig(
    timeout=DEFAULT_TIMEOUT_SECONDS,
    interval=DEFAULT_POLL_INTERVAL_SECONDS,
)

#: Non-pending Genie message statuses. When :meth:`Genie.ask` polls and
#: observes one of these, the response is considered final.
GENIE_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"COMPLETED", "FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED"}
)

#: Title used when :class:`Genie` auto-creates a default space. Also acts as
#: the marker that :meth:`Genie.cleanup_dead_spaces` uses to identify
#: yggdrasil-managed spaces — anything outside this title is left alone.
DEFAULT_MANAGED_SPACE_TITLE: str = "Yggdrasil Genie"

#: ``serialized_space.version`` baked by :func:`build_serialized_space`.
#: The Genie API has accepted ``1`` and ``2``; the v1 shape is the smaller,
#: more permissive payload and what every minimal example in the docs uses.
DEFAULT_SERIALIZED_SPACE_VERSION: int = 1


def build_serialized_space(
    *,
    tables: "tuple[str, ...] | list[str]" = (),
    text_instructions: "tuple[str, ...] | list[str]" = (),
    version: int = DEFAULT_SERIALIZED_SPACE_VERSION,
) -> str:
    """Build a minimal ``serialized_space`` JSON payload for ``create_space``.

    Genie's ``create_space`` requires a JSON-encoded definition of the space's
    data sources and instructions. This helper produces the smallest payload
    the Genie API will accept so callers configuring
    :attr:`GenieDefaults.managed_space_tables` don't have to hand-roll the
    schema.

    Parameters
    ----------
    tables
        Fully qualified ``catalog.schema.table`` identifiers to expose in
        the space. At least one is required by Genie itself; the helper
        forwards whatever is passed and lets the API raise on an empty list.
    text_instructions
        Free-text guidance the LLM should follow inside the space. Optional.
    version
        Schema version. Defaults to :data:`DEFAULT_SERIALIZED_SPACE_VERSION`.

    Returns
    -------
    str
        A JSON string ready to pass to :meth:`Genie.create_space`.
    """
    from uuid import uuid4

    from yggdrasil.pickle import json as ygg_json

    body: dict[str, Any] = {
        "version": version,
        "data_sources": {
            "tables": [{"identifier": ident} for ident in tables],
        },
    }
    if text_instructions:
        body["instructions"] = {
            "text_instructions": [
                {"id": uuid4().hex, "content": [text]}
                for text in text_instructions
            ],
        }
    return ygg_json.dumps(body, to_bytes=False)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenieDefaults:
    """Default configuration for :class:`Genie`.

    Set once on the service, every :meth:`Genie.ask` / :meth:`GenieSpace.ask`
    call inherits these values unless overridden inline::

        from dataclasses import replace
        client.genie.defaults = replace(
            client.genie.defaults,
            space_id="01ef...your-space-id...",
            warehouse_id="abc123",
        )
        answer = client.genie.ask("How many orders last month?")
        print(answer.text)
        print(answer.polars())  # SQL result as a Polars frame, if Genie ran one

    Attributes
    ----------
    space_id
        Default Genie space id. When set, :meth:`Genie.ask` can be called
        without a ``space_id`` argument.
    space_name
        Human-readable filter used by :meth:`Genie.default_space` when
        ``space_id`` is not set and ``auto_pick_space`` is true. The first
        space whose ``title`` matches wins.
    warehouse_id
        Optional warehouse id used to materialise Genie's SQL results when
        :meth:`GenieAnswer.arrow_table` / ``polars`` / ``pandas`` is called.
        Defaults to the workspace's default warehouse when unset.
    wait
        :class:`~yggdrasil.dataclasses.WaitingConfig` carrying the maximum
        time to wait for Genie to finish a message (``wait.timeout``) and the
        SDK polling cadence (``wait.interval``). Defaults to
        :data:`DEFAULT_WAIT` (20 minutes / 1 second). Override per-call by
        passing ``wait=`` to :meth:`Genie.ask` — anything
        :meth:`WaitingConfig.from_` accepts (seconds, ``timedelta``, deadline,
        dict, full ``WaitingConfig``) works.
    auto_execute_query
        When ``True``, :meth:`GenieAnswer.arrow_table` /
        :meth:`GenieAnswer.polars` / :meth:`GenieAnswer.pandas` will
        eagerly re-execute the attached query if no statement result is
        already cached on the message.
    auto_pick_space
        When :attr:`space_id` is unset, :meth:`Genie.default_space` may pick
        a space from :meth:`Genie.list_spaces` (filtered by :attr:`space_name`
        if set). Turn off to require an explicit id.
    auto_create_space
        When :attr:`space_id` is unset *and* :meth:`Genie._pick_space_id`
        could not resolve one (no matching space exists),
        :meth:`Genie.ensure_space` will create a fresh space using
        :attr:`managed_space_title`, :attr:`managed_space_tables`, and
        :attr:`warehouse_id`. Off by default — opt-in because space creation
        has workspace-visible side effects.
    cleanup_dead_spaces
        When ``True``, :meth:`Genie.cleanup_dead_spaces` (called
        automatically from :meth:`Genie.ensure_space` when the flag is on)
        trashes duplicate managed-title spaces, keeping only the active one
        identified by :attr:`space_id`. "Dead" means: same title as
        :attr:`managed_space_title`, not the active id. Off by default.
    managed_space_title
        Title applied when auto-creating a Genie space, also used as the
        marker for :meth:`Genie.cleanup_dead_spaces`. Defaults to
        :data:`DEFAULT_MANAGED_SPACE_TITLE`.
    managed_space_description
        Description applied to auto-created spaces. ``None`` skips the
        field.
    managed_space_tables
        Fully-qualified ``catalog.schema.table`` identifiers exposed in the
        auto-created space's data sources. Genie requires at least one.
    managed_space_parent_path
        Workspace folder path the auto-created space is filed under.
        ``None`` lets Genie pick its default location.
    managed_space_instructions
        Free-text instructions baked into the auto-created space's
        ``serialized_space``.
    agent_output_dir
        Root directory the :class:`GenieAgent` writes artifacts under.
        When ``None`` (the default), :attr:`GenieAgent.output_dir` resolves
        to ``$XDG_CACHE_HOME/yggdrasil/genie`` (falling back to
        ``~/.cache/yggdrasil/genie``).
    agent_auto_save
        When ``True``, :meth:`GenieAgent.run` saves the SQL result of every
        answer that carries a query attachment.
    agent_auto_save_format
        File format used by auto-save. One of ``"parquet"``, ``"csv"``,
        ``"arrow"``, ``"json"``, ``"text"``. Parquet is the default.
    agent_max_steps
        Soft step budget honored by :meth:`GenieAgent.chat`. The agent
        stops accepting new questions once this many turns have completed
        on a single :meth:`chat` call.
    """

    space_id: Optional[str] = None
    space_name: Optional[str] = None
    warehouse_id: Optional[str] = None
    wait: WaitingConfig = DEFAULT_WAIT
    auto_execute_query: bool = True
    auto_pick_space: bool = True
    auto_create_space: bool = True
    cleanup_dead_spaces: bool = False
    managed_space_title: str = DEFAULT_MANAGED_SPACE_TITLE
    managed_space_description: Optional[str] = None
    managed_space_tables: tuple[str, ...] = ()
    managed_space_parent_path: Optional[str] = None
    managed_space_instructions: tuple[str, ...] = ()
    agent_output_dir: Optional[str] = None
    agent_auto_save: bool = False
    agent_auto_save_format: str = "parquet"
    agent_max_steps: int = 8

    @property
    def timeout(self) -> dt.timedelta:
        """The wait budget expressed as a ``timedelta`` (SDK shorthand)."""
        return self.wait.timeout_timedelta


# ---------------------------------------------------------------------------
# GenieAnswer
# ---------------------------------------------------------------------------


class GenieAnswer(DatabricksResource):
    """A single Genie reply.

    Wraps a Databricks ``GenieMessage`` and exposes:

    - the natural-language reply (:attr:`text`) and any SQL Genie generated
      (:attr:`query`, :attr:`statement_id`);
    - convenience accessors that materialise the SQL result as Arrow /
      Polars / pandas (:meth:`arrow_table`, :meth:`polars`, :meth:`pandas`);
    - :meth:`ask` to continue the conversation, :meth:`refresh` to re-poll
      and :meth:`feedback` to rate the response.

    Instances should not be constructed directly — they come back from
    :meth:`Genie.ask` / :meth:`GenieSpace.ask` / :meth:`GenieConversation.ask`.
    """

    def __init__(
        self,
        service: "Genie",
        *,
        space_id: str,
        conversation_id: str,
        message: "GenieMessage",
    ):
        super().__init__(service=service)
        self.service: "Genie" = service
        self.space_id = space_id
        self.conversation_id = conversation_id
        self._message: "GenieMessage" = message
        self._statement_result_cache: Optional["WarehouseStatementResult"] = None

    # ------------------------------------------------------------------ #
    # Identity / debug
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(space_id={self.space_id!r}, "
            f"conversation_id={self.conversation_id!r}, "
            f"message_id={self.message_id!r}, status={self.status!r})"
        )

    def url(self) -> URL:
        """Workspace UI URL for the underlying conversation."""
        return URL.from_str(
            f"{self.client.base_url.to_string().rstrip('/')}"
            f"/genie/rooms/{self.space_id}"
            f"#conversation/{self.conversation_id}"
        )

    # ------------------------------------------------------------------ #
    # SDK passthrough
    # ------------------------------------------------------------------ #
    @property
    def raw(self) -> "GenieMessage":
        """The underlying ``GenieMessage`` returned by the SDK."""
        return self._message

    @property
    def message_id(self) -> str:
        return self._message.id or ""

    @property
    def status(self) -> "Optional[MessageStatus]":
        return self._message.status

    @property
    def is_completed(self) -> bool:
        from databricks.sdk.service.dashboards import MessageStatus
        return self._message.status == MessageStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        from databricks.sdk.service.dashboards import MessageStatus
        return self._message.status in {MessageStatus.FAILED, MessageStatus.CANCELLED}

    @property
    def error(self) -> Optional[str]:
        err = getattr(self._message, "error", None)
        if err is None:
            return None
        # SDK wraps errors in a small struct; surface the message string when present
        return getattr(err, "error", None) or str(err)

    # ------------------------------------------------------------------ #
    # Attachment shortcuts
    # ------------------------------------------------------------------ #
    @property
    def attachments(self) -> list["GenieAttachment"]:
        return list(self._message.attachments or [])

    def _first_attachment(self) -> "Optional[GenieAttachment]":
        attachments = self._message.attachments or []
        return attachments[0] if attachments else None

    @property
    def attachment_id(self) -> Optional[str]:
        att = self._first_attachment()
        return getattr(att, "attachment_id", None) if att else None

    @property
    def text(self) -> Optional[str]:
        """The natural-language reply, if any."""
        att = self._first_attachment()
        if att is None:
            return None
        text_att = getattr(att, "text", None)
        if text_att is not None:
            return getattr(text_att, "content", None)
        return None

    @property
    def query(self) -> Optional[str]:
        """The SQL Genie generated for this answer, if any."""
        att = self._first_attachment()
        if att is None:
            return None
        query_att = getattr(att, "query", None)
        return getattr(query_att, "query", None) if query_att else None

    @property
    def statement_id(self) -> Optional[str]:
        att = self._first_attachment()
        if att is None:
            return None
        query_att = getattr(att, "query", None)
        return getattr(query_att, "statement_id", None) if query_att else None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def refresh(self) -> "GenieAnswer":
        """Re-fetch this message from the Genie API."""
        self._message = self.service.api.get_message(
            space_id=self.space_id,
            conversation_id=self.conversation_id,
            message_id=self.message_id,
        )
        self._statement_result_cache = None
        return self

    def ask(self, question: str, **kwargs: Any) -> "GenieAnswer":
        """Continue the conversation with a follow-up question."""
        return self.service.ask(
            question,
            space_id=self.space_id,
            conversation_id=self.conversation_id,
            **kwargs,
        )

    def feedback(self, rating: Union[str, Any], *, comment: Optional[str] = None) -> "GenieAnswer":
        """Send :class:`GenieFeedbackRating` feedback for this message.

        ``rating`` accepts the SDK enum or its string name (``"POSITIVE"`` /
        ``"NEGATIVE"`` / ``"NONE"``).
        """
        from databricks.sdk.service.dashboards import GenieFeedbackRating

        if isinstance(rating, str):
            rating = GenieFeedbackRating(rating.upper())

        self.service.api.send_message_feedback(
            space_id=self.space_id,
            conversation_id=self.conversation_id,
            message_id=self.message_id,
            rating=rating,
            comment=comment,
        )
        return self

    def delete(self) -> None:
        """Delete this message from the conversation."""
        self.service.api.delete_conversation_message(
            space_id=self.space_id,
            conversation_id=self.conversation_id,
            message_id=self.message_id,
        )

    # ------------------------------------------------------------------ #
    # Query result materialisation
    # ------------------------------------------------------------------ #
    def fetch_query_result(self) -> "Optional[StatementResponse]":
        """Return the raw SQL ``StatementResponse`` for this answer.

        Returns ``None`` when Genie did not generate a query attachment.
        Cached after the first call so repeated ``arrow_table()`` / ``polars()``
        calls do not re-hit the API.
        """
        attachment_id = self.attachment_id
        if attachment_id is None or self.query is None:
            return None

        cached = self._statement_result_cache
        if cached is not None and cached._response is not None:
            return cached._response

        response = self.service.api.get_message_attachment_query_result(
            space_id=self.space_id,
            conversation_id=self.conversation_id,
            message_id=self.message_id,
            attachment_id=attachment_id,
        )
        statement_response = getattr(response, "statement_response", None)
        if statement_response is None:
            return None

        self._statement_result_cache = self._wrap_statement(statement_response)
        return statement_response

    def execute_query(self) -> "Optional[WarehouseStatementResult]":
        """Re-execute the attached query against the warehouse and return the result.

        Useful when the cached Genie result has expired
        (``MessageStatus.QUERY_RESULT_EXPIRED``).
        """
        attachment_id = self.attachment_id
        if attachment_id is None:
            return None

        response = self.service.api.execute_message_attachment_query(
            space_id=self.space_id,
            conversation_id=self.conversation_id,
            message_id=self.message_id,
            attachment_id=attachment_id,
        )
        statement_response = getattr(response, "statement_response", None)
        if statement_response is None:
            return None

        self._statement_result_cache = self._wrap_statement(statement_response)
        return self._statement_result_cache

    def statement_result(self) -> "Optional[WarehouseStatementResult]":
        """Return the result as a :class:`WarehouseStatementResult` for engine use."""
        if self._statement_result_cache is not None:
            return self._statement_result_cache
        if self.fetch_query_result() is None:
            return None
        return self._statement_result_cache

    def arrow_table(self) -> "Optional[pa.Table]":
        """Materialise the query result as a :class:`pyarrow.Table`."""
        result = self.statement_result()
        return None if result is None else result.read_arrow_table()

    def polars(self) -> "Optional[pl.DataFrame]":
        """Materialise the query result as a :class:`polars.DataFrame`."""
        from yggdrasil.lazy_imports import polars as pl
        table = self.arrow_table()
        return None if table is None else pl.from_arrow(table)

    def pandas(self) -> "Optional[pd.DataFrame]":
        """Materialise the query result as a :class:`pandas.DataFrame`."""
        table = self.arrow_table()
        return None if table is None else table.to_pandas()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _wrap_statement(self, response: "StatementResponse") -> "WarehouseStatementResult":
        """Wrap a raw ``StatementResponse`` in a :class:`WarehouseStatementResult`.

        Reuses the workspace's default warehouse purely as a back-pointer so
        the result can compose with the broader yggdrasil SQL surface (cast,
        chunk fetch, arrow batching). No new statement is submitted.
        """
        warehouse = self.service.resolve_warehouse()
        from yggdrasil.databricks.warehouse.statement import (
            WarehousePreparedStatement,
            WarehouseStatementResult,
        )

        statement = WarehousePreparedStatement(
            executor=warehouse,
            statement_text=self.query or "",
        )
        result = WarehouseStatementResult(
            executor=warehouse,
            statement=statement,
        )
        return result.set_api_response(response)


# ---------------------------------------------------------------------------
# GenieConversation
# ---------------------------------------------------------------------------


class GenieConversation(DatabricksResource):
    """An ongoing Genie conversation in a single space.

    Conversations are multi-turn — each new question becomes a new
    :class:`GenieAnswer`. Use :meth:`ask` to send follow-ups and
    :meth:`messages` to iterate history.
    """

    def __init__(
        self,
        service: "Genie",
        *,
        space_id: str,
        conversation_id: str,
    ):
        super().__init__(service=service)
        self.service: "Genie" = service
        self.space_id = space_id
        self.conversation_id = conversation_id

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(space_id={self.space_id!r}, "
            f"conversation_id={self.conversation_id!r})"
        )

    def url(self) -> URL:
        return URL.from_str(
            f"{self.client.base_url.to_string().rstrip('/')}"
            f"/genie/rooms/{self.space_id}#conversation/{self.conversation_id}"
        )

    def ask(self, question: str, **kwargs: Any) -> GenieAnswer:
        """Send another message in this conversation."""
        return self.service.ask(
            question,
            space_id=self.space_id,
            conversation_id=self.conversation_id,
            **kwargs,
        )

    def messages(self, *, page_size: int | None = None) -> Iterator[GenieAnswer]:
        """Iterate over messages in this conversation as :class:`GenieAnswer` objects.

        Note that ``GenieAnswer.text`` / ``query`` etc. read from each message's
        attachments, which only populate once Genie finishes processing.
        """
        response = self.service.api.list_conversation_messages(
            space_id=self.space_id,
            conversation_id=self.conversation_id,
            page_size=page_size,
        )
        for message in getattr(response, "messages", None) or []:
            yield GenieAnswer(
                service=self.service,
                space_id=self.space_id,
                conversation_id=self.conversation_id,
                message=message,
            )

    def delete(self) -> None:
        """Delete this conversation."""
        self.service.api.delete_conversation(
            space_id=self.space_id,
            conversation_id=self.conversation_id,
        )


# ---------------------------------------------------------------------------
# GenieSpace
# ---------------------------------------------------------------------------


class GenieSpace(DatabricksResource):
    """A Databricks Genie space — the conversational interface to a dataset."""

    def __init__(
        self,
        service: "Genie",
        space_id: str,
        *,
        details: Any = None,
    ):
        super().__init__(service=service)
        self.service: "Genie" = service
        self.space_id = space_id
        self._details = details

    def __repr__(self) -> str:
        title = getattr(self._details, "title", None)
        if title:
            return f"{self.__class__.__name__}(space_id={self.space_id!r}, title={title!r})"
        return f"{self.__class__.__name__}(space_id={self.space_id!r})"

    def url(self) -> URL:
        return URL.from_str(
            f"{self.client.base_url.to_string().rstrip('/')}/genie/rooms/{self.space_id}"
        )

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    @property
    def details(self) -> Any:
        """Cached space metadata. Lazily fetched on first access."""
        if self._details is None:
            self._details = self.service.api.get_space(space_id=self.space_id)
        return self._details

    def refresh(self) -> "GenieSpace":
        self._details = self.service.api.get_space(space_id=self.space_id)
        return self

    @property
    def title(self) -> Optional[str]:
        return getattr(self.details, "title", None)

    @property
    def description(self) -> Optional[str]:
        return getattr(self.details, "description", None)

    @property
    def warehouse_id(self) -> Optional[str]:
        return getattr(self.details, "warehouse_id", None)

    # ------------------------------------------------------------------ #
    # Asking
    # ------------------------------------------------------------------ #
    def ask(self, question: str, **kwargs: Any) -> GenieAnswer:
        """Start a new conversation and return Genie's first reply."""
        return self.service.ask(question, space_id=self.space_id, **kwargs)

    def start_conversation(self, question: str, **kwargs: Any) -> tuple[GenieConversation, GenieAnswer]:
        """Start a new conversation and return both the conversation and the first answer.

        Use this when you want to continue the thread after the first reply::

            conv, answer = space.start_conversation("Show orders by region")
            follow_up = conv.ask("Filter to last quarter")
        """
        answer = self.service.ask(question, space_id=self.space_id, **kwargs)
        conversation = GenieConversation(
            service=self.service,
            space_id=self.space_id,
            conversation_id=answer.conversation_id,
        )
        return conversation, answer

    def conversation(self, conversation_id: str) -> GenieConversation:
        """Return a :class:`GenieConversation` handle for an existing thread."""
        return GenieConversation(
            service=self.service,
            space_id=self.space_id,
            conversation_id=conversation_id,
        )

    def list_conversations(self, *, page_size: int | None = None) -> Iterator[GenieConversation]:
        """Iterate over conversations in this space."""
        response = self.service.api.list_conversations(
            space_id=self.space_id,
            page_size=page_size,
        )
        for conv in getattr(response, "conversations", None) or []:
            cid = getattr(conv, "id", None)
            if cid is None:
                continue
            yield GenieConversation(
                service=self.service,
                space_id=self.space_id,
                conversation_id=cid,
            )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def update(
        self,
        *,
        title: str | None = None,
        description: str | None = None,
        serialized_space: str | None = None,
        warehouse_id: str | None = None,
    ) -> "GenieSpace":
        self.service.api.update_space(
            space_id=self.space_id,
            title=title,
            description=description,
            serialized_space=serialized_space,
            warehouse_id=warehouse_id,
        )
        return self.refresh()

    def delete(self) -> None:
        """Move the space to trash."""
        self.service.api.trash_space(space_id=self.space_id)
