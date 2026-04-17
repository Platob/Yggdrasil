"""Pre-execution container for a SQL statement and its bound arguments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, List, Optional

from databricks.sdk.service.sql import StatementParameterListItem

__all__ = ["Statement"]


@dataclass(frozen=True, slots=True)
class Statement:
    """A SQL query string paired with arguments, prepared for execution.

    Holds the text of a SQL statement together with any arguments that
    should be bound at execution time.  Instances are immutable; use
    :meth:`bind` or :meth:`with_temporary_tables` to derive new statements
    with additional arguments.

    Attributes
    ----------
    text:
        Raw SQL query text.  Named parameters are referenced with the
        Databricks ``:name`` placeholder syntax.  Temporary-table aliases
        are referenced as ``{alias}``.
    parameters:
        Mapping of parameter name to Python value.  Values are rendered to
        strings when converted to Databricks
        :class:`~databricks.sdk.service.sql.StatementParameterListItem`
        entries via :meth:`to_parameter_list`.
    temporary_tables:
        Mapping of alias to :class:`~yggdrasil.databricks.sql.staging.StagingPath`
        or tabular data that should be staged before execution.
    """

    text: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    temporary_tables: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def prepare(
        cls,
        statement: "Statement | str",
        *,
        parameters: Mapping[str, Any] | None = None,
        temporary_tables: Mapping[str, Any] | None = None,
    ) -> "Statement":
        """Coerce ``statement`` into a :class:`Statement`, merging extra args.

        When ``statement`` is already a :class:`Statement`, any additional
        ``parameters`` or ``temporary_tables`` are merged on top of its
        existing values.  When it is a string, a new instance is built from
        the supplied arguments.
        """
        if isinstance(statement, cls):
            prepared = statement
            if parameters:
                prepared = prepared.bind(**parameters)
            if temporary_tables:
                prepared = prepared.with_temporary_tables(**temporary_tables)
            return prepared

        return cls(
            text=str(statement),
            parameters=dict(parameters) if parameters else {},
            temporary_tables=dict(temporary_tables) if temporary_tables else {},
        )

    def bind(self, **parameters: Any) -> "Statement":
        """Return a new Statement with additional named parameters bound."""
        if not parameters:
            return self
        return replace(
            self,
            parameters={**self.parameters, **parameters},
        )

    def with_temporary_tables(self, **tables: Any) -> "Statement":
        """Return a new Statement with additional temporary tables registered."""
        if not tables:
            return self
        return replace(
            self,
            temporary_tables={**self.temporary_tables, **tables},
        )

    def clear(self) -> "Statement":
        """Return a new Statement with text and all bound arguments cleared."""
        return replace(
            self,
            text="",
            parameters={},
            temporary_tables={},
        )

    def to_parameter_list(self) -> Optional[List[StatementParameterListItem]]:
        """Render bound parameters as Databricks ``StatementParameterListItem``.

        Returns ``None`` when no parameters are bound so the result can be
        passed directly to the Databricks statement execution API.
        """
        if not self.parameters:
            return None
        return [
            StatementParameterListItem(
                name=str(name),
                value=None if value is None else str(value),
            )
            for name, value in self.parameters.items()
        ]
