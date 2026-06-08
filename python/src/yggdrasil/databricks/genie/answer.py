"""Genie answer — the materialized result of asking a Genie space a question.

A :class:`GenieAnswer` wraps the SDK ``GenieMessage`` returned by a
completed Genie turn and exposes it the ygg way:

- the natural-language ``text`` Genie replied with,
- the ``query`` (SQL) it generated, if any, and its ``statement_id``,
- the tabular result, re-attached to the space's warehouse through the
  same :class:`~yggdrasil.data.statement.StatementResult` the SQL engine
  returns — so ``to_polars()`` / ``to_arrow_table()`` / ``to_pandas()``
  work identically.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from databricks.sdk.service.dashboards import GenieMessage

    from ..client import DatabricksClient

__all__ = ["GenieAnswer"]


class GenieAnswer:
    """The completed result of a Genie turn — text + generated SQL + rows."""

    def __init__(
        self,
        message: "GenieMessage",
        *,
        client: "DatabricksClient",
        space_id: str,
        warehouse_id: Optional[str] = None,
    ):
        self.message = message
        self.client = client
        self.space_id = space_id
        # The space's own warehouse runs the generated SQL; when unknown we
        # let the SQL engine fall back to its default warehouse.
        self.warehouse_id = warehouse_id

    # -- identity ----------------------------------------------------------

    @property
    def conversation_id(self) -> Optional[str]:
        return self.message.conversation_id

    @property
    def message_id(self) -> Optional[str]:
        return self.message.message_id or self.message.id

    @property
    def status(self) -> Optional[str]:
        st = self.message.status
        return getattr(st, "value", None) or (str(st) if st else None)

    @property
    def error(self) -> Optional[str]:
        err = self.message.error
        return getattr(err, "error", None) or (str(err) if err else None)

    # -- content -----------------------------------------------------------

    @property
    def attachments(self) -> list:
        return list(self.message.attachments or [])

    @property
    def text(self) -> str:
        """Genie's natural-language reply (all text attachments joined)."""
        parts = [
            a.text.content
            for a in self.attachments
            if a.text is not None and a.text.content
        ]
        return "\n\n".join(parts)

    @property
    def _query_attachment(self):
        for a in self.attachments:
            if a.query is not None:
                return a.query
        return None

    @property
    def query(self) -> Optional[str]:
        """The SQL Genie generated for this turn, if it answered with a query."""
        qa = self._query_attachment
        return qa.query if qa is not None else None

    @property
    def statement_id(self) -> Optional[str]:
        qa = self._query_attachment
        return qa.statement_id if qa is not None else None

    # -- tabular result ----------------------------------------------------

    def result(self):
        """Re-attach to the generated query's result as a ``StatementResult``.

        Reuses the SQL engine's statement re-attach path, so the rows are
        read on the space's warehouse without re-running the query. Raises
        when the turn produced no query.
        """
        statement_id = self.statement_id
        if not statement_id:
            raise ValueError(
                "this Genie answer has no SQL result to read "
                f"(status={self.status!r}, text={self.text[:80]!r})"
            )
        return self.client.sql.statement_result(
            statement_id, warehouse_id=self.warehouse_id,
        )

    def to_arrow_table(self):
        return self.result().to_arrow_table()

    def to_polars(self):
        return self.result().to_polars()

    def to_pandas(self):
        return self.result().to_pandas()

    def to_pylist(self) -> list[dict[str, Any]]:
        return self.result().to_pylist()

    def __repr__(self) -> str:
        return (
            f"GenieAnswer(space_id={self.space_id!r}, "
            f"conversation_id={self.conversation_id!r}, "
            f"status={self.status!r}, has_query={self.query is not None})"
        )
