"""Abstract Unity table — :class:`ExecutionResource` mixed with :class:`Tabular`.

A table is both a *resource* (it has metadata, a lifecycle, an
existence probe) and a *Tabular* (it yields and consumes Arrow record
batches). :class:`ExecutionTable` composes both so caller code reads:

::

    table = engine["main"]["default"]["sales"]
    arrow = table.read_arrow_table()
    table.write_arrow_batches(batches)

Backends override the two :class:`Tabular` abstract hooks
(:meth:`_read_arrow_batches` / :meth:`_write_arrow_batches`) plus the
resource lifecycle. Schema lookup short-circuits through the stored
:class:`TableInfo` so :meth:`collect_schema` never touches the data.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.io.tabular.base import Tabular
from yggdrasil.unity.base import ExecutionResource
from yggdrasil.unity.info import TableInfo

if TYPE_CHECKING:
    from yggdrasil.unity.schema import ExecutionSchema


__all__ = ["ExecutionTable"]


logger = logging.getLogger(__name__)


class ExecutionTable(ExecutionResource, Tabular[CastOptions]):
    """Abstract managed table — resource lifecycle + :class:`Tabular` IO."""

    def __init__(self) -> None:
        Tabular.__init__(self)

    # ── identity ────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def schema_handle(self) -> "ExecutionSchema":
        """The :class:`ExecutionSchema` owning this table.

        Named ``schema_handle`` so it doesn't clash with the
        :class:`Schema` payload exposed below or with
        :meth:`pa.Table.schema` ergonomics on the same surface.
        """

    @property
    def catalog_name(self) -> str:
        return self.schema_handle.catalog_name

    @property
    def schema_name(self) -> str:
        return self.schema_handle.name

    @property
    def full_name(self) -> str:
        return f"{self.catalog_name}.{self.schema_name}.{self.name}"

    # ── info / schema short-circuit ─────────────────────────────────────

    @abstractmethod
    def _read_info(self) -> TableInfo: ...

    @property
    def info(self) -> TableInfo:  # type: ignore[override]
        return super().info  # type: ignore[return-value]

    @property
    def schema(self) -> Schema:
        """Stored :class:`Schema` for this table — zero-IO."""
        return self.info.schema

    def _collect_schema(self, options: CastOptions) -> Schema:
        """Skip the Arrow-batch sniff: the stored schema IS the truth."""
        return self.schema

    # ── data IO (Tabular contract) ──────────────────────────────────────

    @abstractmethod
    def _read_arrow_batches(self, options: CastOptions):
        """Yield Arrow record batches from the table's data files."""

    @abstractmethod
    def _write_arrow_batches(self, batches, options: CastOptions) -> None:
        """Persist Arrow record batches into the table's data files."""

    # ── lifecycle hooks (override to inject schema + format kwargs) ─────

    @abstractmethod
    def create(
        self,
        schema: Any = ...,
        *,
        if_not_exists: bool = True,
        **kwargs: Any,
    ) -> "ExecutionTable":
        """Create the table. *schema* accepts anything
        :class:`Schema.from_` handles; backends persist it into the
        :class:`TableInfo`."""
