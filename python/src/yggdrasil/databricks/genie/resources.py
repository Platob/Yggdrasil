"""Genie resources and default configuration.

Mirrors the Databricks Genie (AI/BI conversational analytics) hierarchy
as Yggdrasil resources:

- :class:`GenieDefaults` — frozen config attached to the
  :class:`~.service.Genie` service (``service.defaults``). Carries the
  default space id, an optional warehouse override for result
  materialisation, the polling budget for long-running ``ask`` calls,
  and the row cap applied when a query result is pulled into Arrow.
- :class:`GenieSpace` — a single Genie *space* (a curated room scoped to
  a set of tables + instructions). Exposes :meth:`ask` (one-shot:
  start a conversation and return the answer), :meth:`start_conversation`
  (returns the live :class:`GenieConversation`), plus :attr:`infos` /
  :attr:`title` / :attr:`warehouse_id` and conversation listing.
- :class:`GenieConversation` — a live conversation thread. :meth:`ask`
  posts a follow-up turn and waits for the answer; :meth:`messages`
  lists the thread.
- :class:`GenieAnswer` — wrapper around a single ``GenieMessage``: the
  natural-language :attr:`text`, the generated :attr:`sql`, the
  suggested follow-up :attr:`questions`, lifecycle state, and the query
  result materialised as Arrow / Polars / pandas through the shared
  :class:`~yggdrasil.databricks.warehouse.statement.WarehouseStatementResult`
  (so external-link streaming, typed casts, and row endpoints all come
  for free).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Optional

from yggdrasil.databricks.resource import DatabricksResource
from yggdrasil.dataclasses import WaitingConfig, WaitingConfigArg
from yggdrasil.url import URL

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from databricks.sdk.service.dashboards import (
        GenieAPI,
        GenieAttachment,
        GenieConversation as SdkGenieConversation,
        GenieMessage,
        GenieSpace as SdkGenieSpace,
    )
    from databricks.sdk.service.sql import StatementResponse

    from .service import Genie


__all__ = [
    "DEFAULT_GENIE_WAIT",
    "GenieDefaults",
    "GenieSpace",
    "GenieConversation",
    "GenieAnswer",
]


LOGGER = logging.getLogger(__name__)

#: Default wait budget for a Genie ``ask`` round-trip. Genie embeds the
#: question, plans a SQL query, runs it on the space's warehouse, and
#: summarises the result — a cold warehouse plus a non-trivial query
#: routinely takes a minute or two, so the SDK's own 20-minute ceiling is
#: kept, with a 2-second poll cadence.
DEFAULT_GENIE_WAIT: WaitingConfig = WaitingConfig(timeout=1200.0, interval=2.0)


# Genie query results come back INLINE — the data rides the
# ``statement_response.result.data_array`` as ``List[List[str]]`` with the
# column types in ``manifest.schema.columns[*].type_name``. Map the SQL
# type tokens onto Arrow builders so the result casts in one C-bridge hop
# (the same trick the vector-search result wrapper uses).
_PA_BY_SQL_TYPE_NAME = {
    "STRING": "string",
    "CHAR": "string",
    "BOOLEAN": "bool_",
    "BYTE": "int8",
    "SHORT": "int16",
    "INT": "int32",
    "LONG": "int64",
    "FLOAT": "float32",
    "DOUBLE": "float64",
    "BINARY": "binary",
    "DATE": "string",       # ISO date text — left as string to preserve bytes
    "TIMESTAMP": "string",  # ISO timestamp text — same
}


def _statement_response_to_arrow(response: "StatementResponse") -> "pa.Table":
    """Build a :class:`pyarrow.Table` from an inline Genie statement response.

    Reads the column manifest + the ``List[List[str]]`` ``data_array`` and
    casts each column through ``pyarrow.compute.cast``. Unknown / complex
    types stay as strings so the original byte payload is preserved —
    exactly the contract the vector-search result wrapper follows.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    manifest = getattr(response, "manifest", None)
    schema = getattr(manifest, "schema", None) if manifest is not None else None
    columns = list(getattr(schema, "columns", None) or [])
    result = getattr(response, "result", None)
    rows = list(getattr(result, "data_array", None) or []) if result is not None else []

    if not columns:
        return pa.table({})

    col_cells: list[list[Any]] = [[] for _ in columns]
    for row in rows:
        for i, _ in enumerate(columns):
            col_cells[i].append(row[i] if i < len(row) else None)

    arrays: list[pa.Array] = []
    names: list[str] = []
    for col, cells in zip(columns, col_cells):
        type_name = getattr(col, "type_name", None)
        token = type_name.value if type_name is not None and hasattr(type_name, "value") else type_name
        builder = _PA_BY_SQL_TYPE_NAME.get(token or "")
        string_arr = pa.array(cells, type=pa.string())
        if builder is None or builder == "string":
            arr = string_arr
        elif token == "DECIMAL":
            arr = string_arr  # keep precision/scale text intact
        else:
            try:
                arr = pc.cast(string_arr, target_type=getattr(pa, builder)())
            except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
                arr = string_arr
        arrays.append(arr)
        names.append(getattr(col, "name", None) or "")
    return pa.table(arrays, names=names)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenieDefaults:
    """Default configuration for :class:`~.service.Genie`.

    Set once on the service and every subsequent call inherits these
    values unless overridden inline::

        from dataclasses import replace
        client.genie.defaults = replace(
            client.genie.defaults, space_id="01ef…", max_result_rows=5000,
        )

    Attributes
    ----------
    space_id
        Default Genie space used by :meth:`Genie.space` / :meth:`Genie.ask`
        when none is passed inline.
    warehouse_id
        Optional warehouse override used to materialise query results.
        ``None`` (default) uses the warehouse the space itself runs on —
        the right answer almost always, since Genie already executed the
        query there.
    max_result_rows
        Row cap applied when a :class:`GenieAnswer` query result is pulled
        into Arrow. ``None`` (default) leaves it to the result's own
        truncation. Guards against an unbounded ``SELECT *`` landing in
        memory.
    wait
        :class:`~yggdrasil.dataclasses.WaitingConfig` budget for a Genie
        ``ask`` round-trip. Defaults to :data:`DEFAULT_GENIE_WAIT`
        (20 minutes / 2 seconds). Override per-call by passing ``wait=`` —
        anything :meth:`WaitingConfig.from_` accepts works.
    """

    space_id: Optional[str] = None
    warehouse_id: Optional[str] = None
    max_result_rows: Optional[int] = None
    default_space_title: str = "Yggdrasil Genie"
    wait: WaitingConfig = DEFAULT_GENIE_WAIT


# ---------------------------------------------------------------------------
# GenieAnswer
# ---------------------------------------------------------------------------


class GenieAnswer:
    """A single Genie message — the unit Genie hands back from a turn.

    A Genie answer is a small bundle: a natural-language summary
    (:attr:`text`), an optional generated SQL query (:attr:`sql`) with a
    one-line :attr:`description`, optional follow-up :attr:`questions`,
    and — when the message ran a query — a tabular result reachable via
    :meth:`to_arrow` / :meth:`to_polars` / :meth:`to_pandas` /
    :meth:`rows`.
    """

    def __init__(self, conversation: "GenieConversation", message: "GenieMessage"):
        self.conversation = conversation
        self.message = message
        self._statement_response: "Optional[StatementResponse]" = None
        self._table: "Optional[pa.Table]" = None

    def __repr__(self) -> str:
        head = (self.text or "").strip().replace("\n", " ")
        if len(head) > 60:
            head = head[:57] + "…"
        return (
            f"{type(self).__name__}(status={self.status!r}, "
            f"has_query={self.has_query}, text={head!r})"
        )

    # ------------------------------------------------------------------ #
    # Identity / lifecycle
    # ------------------------------------------------------------------ #
    @property
    def space(self) -> "GenieSpace":
        return self.conversation.space

    @property
    def service(self) -> "Genie":
        return self.conversation.service

    @property
    def message_id(self) -> Optional[str]:
        return getattr(self.message, "message_id", None) or getattr(self.message, "id", None)

    @property
    def conversation_id(self) -> Optional[str]:
        return getattr(self.message, "conversation_id", None) or self.conversation.conversation_id

    @property
    def status(self) -> Optional[str]:
        st = getattr(self.message, "status", None)
        return st.value if st is not None and hasattr(st, "value") else st

    @property
    def is_complete(self) -> bool:
        return self.status == "COMPLETED"

    @property
    def failed(self) -> bool:
        return self.status in ("FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED")

    @property
    def error(self) -> Optional[str]:
        err = getattr(self.message, "error", None)
        if err is None:
            return None
        return getattr(err, "error", None) or str(err)

    # ------------------------------------------------------------------ #
    # Attachments
    # ------------------------------------------------------------------ #
    @property
    def attachments(self) -> tuple["GenieAttachment", ...]:
        return tuple(getattr(self.message, "attachments", None) or ())

    @property
    def text(self) -> Optional[str]:
        """Concatenated natural-language text across text attachments.

        Genie returns its prose summary as one or more ``text``
        attachments; the top-level ``content`` field carries the original
        question, so the answer text is assembled from the attachments.
        """
        parts: list[str] = []
        for att in self.attachments:
            text_att = getattr(att, "text", None)
            content = getattr(text_att, "content", None) if text_att is not None else None
            if content:
                parts.append(content)
        return "\n".join(parts) if parts else None

    @property
    def _query_attachment(self) -> "Optional[Any]":
        for att in self.attachments:
            q = getattr(att, "query", None)
            if q is not None:
                return q
        return None

    @property
    def has_query(self) -> bool:
        return self._query_attachment is not None

    @property
    def sql(self) -> Optional[str]:
        """The SQL query Genie generated for this turn, if any."""
        q = self._query_attachment
        return getattr(q, "query", None) if q is not None else None

    @property
    def description(self) -> Optional[str]:
        """Genie's one-line description of the generated query."""
        q = self._query_attachment
        return getattr(q, "description", None) if q is not None else None

    @property
    def attachment_id(self) -> Optional[str]:
        for att in self.attachments:
            if getattr(att, "query", None) is not None:
                return getattr(att, "attachment_id", None)
        return None

    @property
    def questions(self) -> tuple[str, ...]:
        """Suggested follow-up questions Genie offered, if any."""
        for att in self.attachments:
            sq = getattr(att, "suggested_questions", None)
            qs = getattr(sq, "questions", None) if sq is not None else None
            if qs:
                # Each entry is a SuggestedQuestion with a ``question`` field
                # in newer SDKs, or a bare string in older ones.
                return tuple(getattr(q, "question", None) or str(q) for q in qs)
        return ()

    # ------------------------------------------------------------------ #
    # Query result materialisation
    # ------------------------------------------------------------------ #
    @property
    def statement_response(self) -> "Optional[StatementResponse]":
        """The raw SDK statement response behind the query, fetched on demand.

        Genie executes the query on the space's warehouse and returns the
        result *inline*; this fetches that response (cached). ``None`` for
        a text-only answer.
        """
        if not self.has_query:
            return None
        cached = self._statement_response
        if cached is not None:
            return cached

        space_id = self.space.space_id
        conv_id = self.conversation_id
        msg_id = self.message_id
        att_id = self.attachment_id
        LOGGER.debug(
            "Fetching Genie query result (space=%s, conv=%s, msg=%s, att=%s)",
            space_id, conv_id, msg_id, att_id,
        )
        if att_id is not None:
            resp = self.service.api.get_message_attachment_query_result(
                space_id=space_id,
                conversation_id=conv_id,
                message_id=msg_id,
                attachment_id=att_id,
            )
        else:
            resp = self.service.api.get_message_query_result(
                space_id=space_id, conversation_id=conv_id, message_id=msg_id,
            )
        self._statement_response = getattr(resp, "statement_response", None)
        return self._statement_response

    @property
    def row_count(self) -> Optional[int]:
        """Row count reported by the query result manifest, if any."""
        resp = self.statement_response
        manifest = getattr(resp, "manifest", None) if resp is not None else None
        return getattr(manifest, "total_row_count", None) if manifest is not None else None

    def to_arrow(self) -> "Optional[pa.Table]":
        """Materialise the query result as a :class:`pyarrow.Table` (or ``None``).

        Reuses the project's inline-result → Arrow projection (typed casts,
        unknown types preserved as strings). Cached after the first call.
        """
        if not self.has_query:
            return None
        if self._table is not None:
            return self._table
        resp = self.statement_response
        if resp is None:
            return None
        self._table = _statement_response_to_arrow(resp)
        return self._table

    def to_polars(self) -> "Optional[pl.DataFrame]":
        """Materialise the query result as a :class:`polars.DataFrame` (or ``None``)."""
        table = self.to_arrow()
        if table is None:
            return None
        import polars as pl

        return pl.from_arrow(table)

    def to_pandas(self) -> "Optional[pd.DataFrame]":
        """Materialise the query result as a :class:`pandas.DataFrame` (or ``None``)."""
        table = self.to_arrow()
        return table.to_pandas() if table is not None else None

    def rows(self) -> list[dict[str, Any]]:
        """Return the query result as a list of dict rows (empty for text-only)."""
        table = self.to_arrow()
        return table.to_pylist() if table is not None else []

    # ------------------------------------------------------------------ #
    # Feedback
    # ------------------------------------------------------------------ #
    def thumbs_up(self, comment: Optional[str] = None) -> "GenieAnswer":
        """Send positive feedback on this answer."""
        return self._feedback("POSITIVE", comment)

    def thumbs_down(self, comment: Optional[str] = None) -> "GenieAnswer":
        """Send negative feedback on this answer."""
        return self._feedback("NEGATIVE", comment)

    def _feedback(self, rating: str, comment: Optional[str]) -> "GenieAnswer":
        from databricks.sdk.service.dashboards import GenieFeedbackRating

        self.service.api.send_message_feedback(
            space_id=self.space.space_id,
            conversation_id=self.conversation_id,
            message_id=self.message_id,
            rating=GenieFeedbackRating(rating),
            comment=comment,
        )
        return self


# ---------------------------------------------------------------------------
# GenieConversation
# ---------------------------------------------------------------------------


class GenieConversation(DatabricksResource):
    """A live Genie conversation thread within a space."""

    def __init__(
        self,
        service: "Genie",
        space: "GenieSpace",
        conversation_id: str,
        *,
        details: "Optional[SdkGenieConversation]" = None,
    ):
        super().__init__(service=service)
        self.service: "Genie" = service
        self.space = space
        self.conversation_id = conversation_id
        self._details = details

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(space_id={self.space.space_id!r}, "
            f"conversation_id={self.conversation_id!r})"
        )

    @property
    def api(self) -> "GenieAPI":
        return self.service.api

    @property
    def explore_url(self) -> URL:
        return URL.from_str(
            f"{self.client.base_url.to_string().rstrip('/')}"
            f"/genie/rooms/{self.space.space_id}"
            f"?conversation={self.conversation_id}"
        )

    def ask(self, content: str, *, wait: WaitingConfigArg = None) -> GenieAnswer:
        """Post a follow-up turn and wait for the answer."""
        wait_cfg = self.service._resolve_wait(wait)
        LOGGER.debug("Asking Genie follow-up in %r: %s", self, content)
        message = self.api.create_message_and_wait(
            space_id=self.space.space_id,
            conversation_id=self.conversation_id,
            content=content,
            timeout=wait_cfg.timeout_timedelta,
        )
        return GenieAnswer(conversation=self, message=message)

    def message(self, message_id: str) -> GenieAnswer:
        """Fetch a single message in this conversation as a :class:`GenieAnswer`."""
        msg = self.api.get_message(
            space_id=self.space.space_id,
            conversation_id=self.conversation_id,
            message_id=message_id,
        )
        return GenieAnswer(conversation=self, message=msg)

    def messages(self) -> Iterator[GenieAnswer]:
        """Iterate over the messages in this conversation."""
        resp = self.api.list_conversation_messages(
            space_id=self.space.space_id,
            conversation_id=self.conversation_id,
        )
        for msg in getattr(resp, "messages", None) or []:
            yield GenieAnswer(conversation=self, message=msg)

    def delete(self) -> None:
        """Delete this conversation."""
        self.api.delete_conversation(
            space_id=self.space.space_id, conversation_id=self.conversation_id,
        )


# ---------------------------------------------------------------------------
# GenieSpace
# ---------------------------------------------------------------------------


class GenieSpace(DatabricksResource):
    """A Databricks Genie space — a curated conversational-analytics room."""

    def __init__(
        self,
        service: "Genie",
        space_id: str,
        *,
        details: "Optional[SdkGenieSpace]" = None,
    ):
        super().__init__(service=service)
        self.service: "Genie" = service
        self.space_id = space_id
        self._details = details

    def __repr__(self) -> str:
        return f"{type(self).__name__}(space_id={self.space_id!r})"

    @property
    def api(self) -> "GenieAPI":
        return self.service.api

    @property
    def explore_url(self) -> URL:
        return URL.from_str(
            f"{self.client.base_url.to_string().rstrip('/')}/genie/rooms/{self.space_id}"
        )

    # ------------------------------------------------------------------ #
    # Identity / cached infos
    # ------------------------------------------------------------------ #
    @property
    def infos(self) -> "SdkGenieSpace":
        """Cached :class:`GenieSpace` SDK info — fetched on first access."""
        cached = self._details
        if cached is not None:
            return cached
        LOGGER.debug("Fetching Genie space %r from remote", self)
        info = self.api.get_space(space_id=self.space_id)
        self._details = info
        return info

    def refresh(self) -> "GenieSpace":
        self._details = self.api.get_space(space_id=self.space_id)
        return self

    def exists(self) -> bool:
        from databricks.sdk.errors import NotFound

        try:
            _ = self.infos
            return True
        except NotFound:
            return False

    @property
    def title(self) -> Optional[str]:
        return getattr(self.infos, "title", None)

    @property
    def description(self) -> Optional[str]:
        return getattr(self.infos, "description", None)

    @property
    def warehouse_id(self) -> Optional[str]:
        return getattr(self.infos, "warehouse_id", None)

    # ------------------------------------------------------------------ #
    # Conversations
    # ------------------------------------------------------------------ #
    def start_conversation(
        self, content: str, *, wait: WaitingConfigArg = None,
    ) -> tuple[GenieConversation, GenieAnswer]:
        """Start a new conversation with an opening question.

        Returns the live :class:`GenieConversation` (for follow-ups) and
        the first :class:`GenieAnswer`.
        """
        wait_cfg = self.service._resolve_wait(wait)
        LOGGER.debug("Starting Genie conversation in %r: %s", self, content)
        message = self.api.start_conversation_and_wait(
            space_id=self.space_id, content=content, timeout=wait_cfg.timeout_timedelta,
        )
        conv = GenieConversation(
            service=self.service, space=self, conversation_id=message.conversation_id,
        )
        return conv, GenieAnswer(conversation=conv, message=message)

    def ask(self, content: str, *, wait: WaitingConfigArg = None) -> GenieAnswer:
        """One-shot: start a conversation, return just the answer."""
        _conv, answer = self.start_conversation(content, wait=wait)
        return answer

    def conversation(self, conversation_id: str) -> GenieConversation:
        """Return a handle to an existing conversation."""
        return GenieConversation(
            service=self.service, space=self, conversation_id=conversation_id,
        )

    def conversations(self) -> Iterator[GenieConversation]:
        """Iterate over conversations in this space."""
        resp = self.api.list_conversations(space_id=self.space_id)
        for summary in getattr(resp, "conversations", None) or []:
            cid = getattr(summary, "conversation_id", None)
            if not cid:
                continue
            yield GenieConversation(service=self.service, space=self, conversation_id=cid)

    def agent(self, **kwargs: Any):
        """Return a :class:`~.agent.GenieAgent` bound to this space."""
        from .agent import GenieAgent

        return GenieAgent(space=self, **kwargs)

    def trash(self, *, missing_ok: bool = False) -> None:
        """Move this space to the trash (the SDK's delete for spaces)."""
        from databricks.sdk.errors import NotFound

        LOGGER.debug("Trashing Genie space %r", self)
        try:
            self.api.trash_space(space_id=self.space_id)
        except NotFound:
            if not missing_ok:
                raise
            LOGGER.debug("Genie space %r already gone", self)
        self._details = None
