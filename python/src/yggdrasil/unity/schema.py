"""Abstract schema — owns the table and view leaves under a catalog.

Schemas are the leaf containers of the ``catalog.schema.table`` tree:
they hold both tables (materialised :class:`Tabular` data) and views
(read-only references / SQL projections). ``schema["sales"]`` resolves
either by name, with table-before-view precedence so the common case
"give me the rows under this name" doesn't surprise.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Iterator

from yggdrasil.unity.base import ExecutionResource

if TYPE_CHECKING:
    from yggdrasil.unity.catalog import ExecutionCatalog
    from yggdrasil.unity.table import ExecutionTable
    from yggdrasil.unity.view import ExecutionView


__all__ = ["ExecutionSchema"]


logger = logging.getLogger(__name__)


class ExecutionSchema(ExecutionResource):
    """Abstract schema — child tables and views plus the inherited lifecycle."""

    @property
    @abstractmethod
    def catalog(self) -> "ExecutionCatalog":
        """The :class:`ExecutionCatalog` owning this schema."""

    @property
    def catalog_name(self) -> str:
        return self.catalog.name

    @property
    def full_name(self) -> str:
        return f"{self.catalog_name}.{self.name}"

    # ── table / view navigation ─────────────────────────────────────────

    @abstractmethod
    def table(self, name: str) -> "ExecutionTable":
        """Return a :class:`ExecutionTable` bound to *name* (no existence check)."""

    @abstractmethod
    def tables(self) -> Iterator["ExecutionTable"]:
        """Iterate over every table in this schema."""

    @abstractmethod
    def view(self, name: str) -> "ExecutionView":
        """Return a :class:`ExecutionView` bound to *name* (no existence check)."""

    @abstractmethod
    def views(self) -> Iterator["ExecutionView"]:
        """Iterate over every view in this schema."""

    @abstractmethod
    def create_table(
        self,
        name: str,
        schema: "object",
        *,
        if_not_exists: bool = True,
        **kwargs,
    ) -> "ExecutionTable":
        """Create a managed table under this schema and return its handle.

        ``schema`` is anything :meth:`yggdrasil.data.Schema.from_` accepts —
        a :class:`Schema`, a :class:`pa.Schema`, a list of :class:`Field`
        instances, a polars / pandas / spark schema.
        """

    @abstractmethod
    def create_view(
        self,
        name: str,
        source: "str | ExecutionTable | ExecutionView",
        *,
        definition: str | None = None,
        if_not_exists: bool = True,
        **kwargs,
    ) -> "ExecutionView":
        """Create a view under this schema.

        ``source`` accepts a dotted ``catalog.schema.name`` string or a
        live :class:`ExecutionTable` / :class:`ExecutionView` (its ``full_name``
        is recorded). ``definition`` is an optional SQL projection
        applied at read time.
        """

    # ── default surface ─────────────────────────────────────────────────

    def __getitem__(self, name: str) -> "ExecutionResource":
        """Table by name first, view as a fallback.

        Tables and views share a namespace; matching tables first keeps
        the common "give me the rows" path short. Raises
        :class:`KeyError` listing both kinds when neither resolves.
        """
        tbl = self.table(name)
        if tbl.exists:
            return tbl
        vw = self.view(name)
        if vw.exists:
            return vw
        tables = sorted(t.name for t in self.tables())
        views = sorted(v.name for v in self.views())
        raise KeyError(
            f"No table or view named {name!r} in schema {self.full_name!r}. "
            f"Tables: {tables!r}, views: {views!r}."
        )

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return self.table(name).exists or self.view(name).exists

    def __iter__(self) -> Iterator["ExecutionResource"]:
        yield from self.tables()
        yield from self.views()
