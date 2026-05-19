"""Statement vocabulary for :class:`ExecutionEngine` execution plans.

Every catalog-level operation (create / drop a resource, insert rows,
read rows back, list a level of the namespace) is modelled as a
:class:`PreparedStatement` subclass. The engine submits one through
:meth:`ExecutionEngine.send` / :meth:`ExecutionEngine.execute` and gets a
:class:`ExecutionStatementResult` whose ``output`` carries the operation's
return value (a :class:`ExecutionResource`, a row count, a :class:`Tabular`).

The statements are intentionally polymorphic — each one knows how to
apply itself to a :class:`ExecutionEngine` through the :meth:`apply` hook,
so the engine doesn't have to fan out on ``isinstance``. New backends
plug in by overriding the engine's high-level resource methods
(``create_catalog`` / ``schema.table`` / …); the statement layer flows
straight through.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Mapping

from yggdrasil.data.statement import PreparedStatement
from yggdrasil.unity.plan_type import PlanCategory, PlanTypeId

if TYPE_CHECKING:
    import pyarrow as pa

    from yggdrasil.io.tabular.base import Tabular
    from yggdrasil.unity.base import ExecutionResource
    from yggdrasil.unity.engine import ExecutionEngine


__all__ = [
    "ExecutionStatement",
    "CreateCatalog",
    "CreateSchema",
    "CreateTable",
    "CreateView",
    "DropCatalog",
    "DropSchema",
    "DropTable",
    "DropView",
    "Insert",
    "Select",
    "ShowCatalogs",
    "ShowSchemas",
    "ShowTables",
    "ShowViews",
]


logger = logging.getLogger(__name__)


def _quote(value: Any) -> str:
    """Best-effort SQL-ish rendering of *value* for statement text."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    return f"'{str(value).replace(chr(39), chr(39) + chr(39))}'"


def _render_options(options: "Mapping[str, Any] | None") -> str:
    if not options:
        return ""
    inner = ", ".join(f"{k}={_quote(v)}" for k, v in options.items() if v is not None)
    return f" OPTIONS ({inner})" if inner else ""


def _resolve_resource_full_name(value: Any) -> str:
    """Turn *value* into a dotted ``catalog.schema.name`` identifier."""
    from yggdrasil.unity.base import ExecutionResource

    if isinstance(value, ExecutionResource):
        return value.full_name
    if isinstance(value, str):
        return value
    raise TypeError(
        f"Expected an ExecutionResource or dotted identifier; got "
        f"{type(value).__name__}: {value!r}."
    )


# ── base ────────────────────────────────────────────────────────────────


class ExecutionStatement(PreparedStatement):
    """Base class for every :class:`ExecutionEngine` statement.

    Subclasses store the operation's structured payload as attributes,
    declare a :attr:`plan_type_id` ClassVar (used by
    :class:`ExecutionPlan` and its specialised sub-plans for
    classification and validation), and implement :meth:`apply` —
    invoked by :meth:`ExecutionStatementResult.start`. The :attr:`text`
    slot is a SQL-ish projection of the payload, used by the standard
    logging / repr path inherited from :class:`PreparedStatement`.
    """

    #: Stable :class:`PlanTypeId` for this concrete statement shape.
    #: The abstract base intentionally leaves it unset — every
    #: instantiable subclass pins it.
    plan_type_id: ClassVar[PlanTypeId]

    def __init__(self, *, key: "str | None" = None) -> None:
        super().__init__(text=self._render_text(), key=key)

    @property
    def category(self) -> PlanCategory:
        """Coarse classification derived from :attr:`plan_type_id`."""
        return self.plan_type_id.category

    @property
    def is_mutation(self) -> bool:
        """``True`` when this statement changes backend state (DDL / DML)."""
        return self.plan_type_id.is_mutation

    @property
    def is_query(self) -> bool:
        """``True`` when this statement is read-only (DQL / META)."""
        return self.plan_type_id.is_query

    @abstractmethod
    def _render_text(self) -> str:
        """Return the SQL-ish text representation of this statement."""

    @abstractmethod
    def apply(self, engine: "ExecutionEngine") -> Any:
        """Run the operation against *engine* and return its output.

        The output is whatever the operation produces — a
        :class:`ExecutionResource` for create/drop, an integer row count
        for inserts, a :class:`Tabular` for selects/shows. The engine
        wraps the return value into a :class:`ExecutionStatementResult`.
        """


# ── create ──────────────────────────────────────────────────────────────


class CreateCatalog(ExecutionStatement):
    plan_type_id: ClassVar[PlanTypeId] = PlanTypeId.CREATE_CATALOG

    def __init__(
        self,
        name: str,
        *,
        comment: "str | None" = None,
        owner: "str | None" = None,
        properties: "Mapping[str, str] | None" = None,
        if_not_exists: bool = True,
        key: "str | None" = None,
    ) -> None:
        self.name = name
        self.comment = comment
        self.owner = owner
        self.properties = dict(properties) if properties else {}
        self.if_not_exists = if_not_exists
        super().__init__(key=key)

    def _render_text(self) -> str:
        ine = " IF NOT EXISTS" if self.if_not_exists else ""
        return f"CREATE CATALOG{ine} {self.name}{_render_options(self.properties)}"

    def apply(self, engine: "ExecutionEngine") -> "ExecutionResource":
        return engine.create_catalog(
            self.name,
            comment=self.comment,
            owner=self.owner,
            properties=self.properties or None,
            if_not_exists=self.if_not_exists,
        )


class CreateSchema(ExecutionStatement):
    plan_type_id: ClassVar[PlanTypeId] = PlanTypeId.CREATE_SCHEMA

    def __init__(
        self,
        catalog_name: str,
        name: str,
        *,
        comment: "str | None" = None,
        owner: "str | None" = None,
        properties: "Mapping[str, str] | None" = None,
        if_not_exists: bool = True,
        key: "str | None" = None,
    ) -> None:
        self.catalog_name = catalog_name
        self.name = name
        self.comment = comment
        self.owner = owner
        self.properties = dict(properties) if properties else {}
        self.if_not_exists = if_not_exists
        super().__init__(key=key)

    def _render_text(self) -> str:
        ine = " IF NOT EXISTS" if self.if_not_exists else ""
        return (
            f"CREATE SCHEMA{ine} {self.catalog_name}.{self.name}"
            f"{_render_options(self.properties)}"
        )

    def apply(self, engine: "ExecutionEngine") -> "ExecutionResource":
        catalog = engine.catalog(self.catalog_name)
        if not catalog.exists:
            raise FileNotFoundError(
                f"Catalog {self.catalog_name!r} does not exist on {engine!r}."
            )
        return catalog.create_schema(
            self.name,
            comment=self.comment,
            owner=self.owner,
            properties=self.properties or None,
            if_not_exists=self.if_not_exists,
        )


class CreateTable(ExecutionStatement):
    plan_type_id: ClassVar[PlanTypeId] = PlanTypeId.CREATE_TABLE

    def __init__(
        self,
        catalog_name: str,
        schema_name: str,
        name: str,
        schema: Any,
        *,
        format: Any = ...,
        partition_by: "tuple[str, ...] | list[str] | None" = None,
        comment: "str | None" = None,
        owner: "str | None" = None,
        properties: "Mapping[str, str] | None" = None,
        if_not_exists: bool = True,
        key: "str | None" = None,
    ) -> None:
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.name = name
        self.schema = schema
        self.format = format
        self.partition_by = tuple(partition_by or ())
        self.comment = comment
        self.owner = owner
        self.properties = dict(properties) if properties else {}
        self.if_not_exists = if_not_exists
        super().__init__(key=key)

    def _render_text(self) -> str:
        ine = " IF NOT EXISTS" if self.if_not_exists else ""
        part = (
            f" PARTITIONED BY ({', '.join(self.partition_by)})"
            if self.partition_by else ""
        )
        fmt = f" USING {self.format}" if self.format is not ... else ""
        return (
            f"CREATE TABLE{ine} {self.catalog_name}.{self.schema_name}.{self.name}"
            f"{fmt}{part}{_render_options(self.properties)}"
        )

    def apply(self, engine: "ExecutionEngine") -> "ExecutionResource":
        catalog = engine.catalog(self.catalog_name)
        if not catalog.exists:
            raise FileNotFoundError(
                f"Catalog {self.catalog_name!r} does not exist on {engine!r}."
            )
        schema_handle = catalog.schema(self.schema_name)
        if not schema_handle.exists:
            raise FileNotFoundError(
                f"Schema {self.catalog_name}.{self.schema_name!r} does not exist."
            )
        kwargs: dict[str, Any] = dict(
            schema=self.schema,
            partition_by=self.partition_by or None,
            comment=self.comment,
            owner=self.owner,
            properties=self.properties or None,
            if_not_exists=self.if_not_exists,
        )
        if self.format is not ...:
            kwargs["format"] = self.format
        return schema_handle.create_table(self.name, **kwargs)


class CreateView(ExecutionStatement):
    plan_type_id: ClassVar[PlanTypeId] = PlanTypeId.CREATE_VIEW

    def __init__(
        self,
        catalog_name: str,
        schema_name: str,
        name: str,
        source: Any,
        *,
        definition: "str | None" = None,
        comment: "str | None" = None,
        owner: "str | None" = None,
        properties: "Mapping[str, str] | None" = None,
        if_not_exists: bool = True,
        key: "str | None" = None,
    ) -> None:
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.name = name
        self.source_full_name = _resolve_resource_full_name(source)
        self.definition = definition
        self.comment = comment
        self.owner = owner
        self.properties = dict(properties) if properties else {}
        self.if_not_exists = if_not_exists
        super().__init__(key=key)

    def _render_text(self) -> str:
        ine = " IF NOT EXISTS" if self.if_not_exists else ""
        body = self.definition or f"SELECT * FROM {self.source_full_name}"
        return (
            f"CREATE VIEW{ine} {self.catalog_name}.{self.schema_name}.{self.name} "
            f"AS {body}"
        )

    def apply(self, engine: "ExecutionEngine") -> "ExecutionResource":
        catalog = engine.catalog(self.catalog_name)
        if not catalog.exists:
            raise FileNotFoundError(
                f"Catalog {self.catalog_name!r} does not exist on {engine!r}."
            )
        schema_handle = catalog.schema(self.schema_name)
        if not schema_handle.exists:
            raise FileNotFoundError(
                f"Schema {self.catalog_name}.{self.schema_name!r} does not exist."
            )
        return schema_handle.create_view(
            self.name,
            source=self.source_full_name,
            definition=self.definition,
            comment=self.comment,
            owner=self.owner,
            properties=self.properties or None,
            if_not_exists=self.if_not_exists,
        )


# ── drop ────────────────────────────────────────────────────────────────


class DropCatalog(ExecutionStatement):
    plan_type_id: ClassVar[PlanTypeId] = PlanTypeId.DROP_CATALOG

    def __init__(
        self,
        name: str,
        *,
        recursive: bool = False,
        missing_ok: bool = True,
        key: "str | None" = None,
    ) -> None:
        self.name = name
        self.recursive = recursive
        self.missing_ok = missing_ok
        super().__init__(key=key)

    def _render_text(self) -> str:
        ie = " IF EXISTS" if self.missing_ok else ""
        cascade = " CASCADE" if self.recursive else ""
        return f"DROP CATALOG{ie} {self.name}{cascade}"

    def apply(self, engine: "ExecutionEngine") -> "ExecutionResource":
        catalog = engine.catalog(self.name)
        catalog.delete(recursive=self.recursive, missing_ok=self.missing_ok)
        return catalog


class DropSchema(ExecutionStatement):
    plan_type_id: ClassVar[PlanTypeId] = PlanTypeId.DROP_SCHEMA

    def __init__(
        self,
        catalog_name: str,
        name: str,
        *,
        recursive: bool = False,
        missing_ok: bool = True,
        key: "str | None" = None,
    ) -> None:
        self.catalog_name = catalog_name
        self.name = name
        self.recursive = recursive
        self.missing_ok = missing_ok
        super().__init__(key=key)

    def _render_text(self) -> str:
        ie = " IF EXISTS" if self.missing_ok else ""
        cascade = " CASCADE" if self.recursive else ""
        return f"DROP SCHEMA{ie} {self.catalog_name}.{self.name}{cascade}"

    def apply(self, engine: "ExecutionEngine") -> "ExecutionResource":
        catalog = engine.catalog(self.catalog_name)
        schema_handle = catalog.schema(self.name)
        schema_handle.delete(recursive=self.recursive, missing_ok=self.missing_ok)
        return schema_handle


class DropTable(ExecutionStatement):
    plan_type_id: ClassVar[PlanTypeId] = PlanTypeId.DROP_TABLE

    def __init__(
        self,
        catalog_name: str,
        schema_name: str,
        name: str,
        *,
        purge_data: bool = True,
        missing_ok: bool = True,
        key: "str | None" = None,
    ) -> None:
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.name = name
        self.purge_data = purge_data
        self.missing_ok = missing_ok
        super().__init__(key=key)

    def _render_text(self) -> str:
        ie = " IF EXISTS" if self.missing_ok else ""
        return f"DROP TABLE{ie} {self.catalog_name}.{self.schema_name}.{self.name}"

    def apply(self, engine: "ExecutionEngine") -> "ExecutionResource":
        table = (
            engine.catalog(self.catalog_name)
            .schema(self.schema_name)
            .table(self.name)
        )
        table.delete(purge_data=self.purge_data, missing_ok=self.missing_ok)
        return table


class DropView(ExecutionStatement):
    plan_type_id: ClassVar[PlanTypeId] = PlanTypeId.DROP_VIEW

    def __init__(
        self,
        catalog_name: str,
        schema_name: str,
        name: str,
        *,
        missing_ok: bool = True,
        key: "str | None" = None,
    ) -> None:
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.name = name
        self.missing_ok = missing_ok
        super().__init__(key=key)

    def _render_text(self) -> str:
        ie = " IF EXISTS" if self.missing_ok else ""
        return f"DROP VIEW{ie} {self.catalog_name}.{self.schema_name}.{self.name}"

    def apply(self, engine: "ExecutionEngine") -> "ExecutionResource":
        view = (
            engine.catalog(self.catalog_name)
            .schema(self.schema_name)
            .view(self.name)
        )
        view.delete(missing_ok=self.missing_ok)
        return view


# ── data movement ───────────────────────────────────────────────────────


class Insert(ExecutionStatement):
    """Write Arrow record batches into a table.

    *data* is anything the table's :meth:`ExecutionTable.write_table` /
    :meth:`ExecutionTable.write_arrow_batches` surface accepts — a
    :class:`pa.Table`, a list of :class:`pa.RecordBatch`, an iterable
    of batches, or another :class:`Tabular`.
    """

    plan_type_id: ClassVar[PlanTypeId] = PlanTypeId.INSERT

    def __init__(
        self,
        catalog_name: str,
        schema_name: str,
        table_name: str,
        data: Any,
        *,
        mode: Any = ...,
        match_by: "list[str] | None" = None,
        key: "str | None" = None,
    ) -> None:
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.table_name = table_name
        self.data = data
        self.mode = mode
        self.match_by = list(match_by) if match_by else None
        super().__init__(key=key)

    @property
    def target_full_name(self) -> str:
        """Dotted ``catalog.schema.table`` identifier of the insert target."""
        return f"{self.catalog_name}.{self.schema_name}.{self.table_name}"

    def _render_text(self) -> str:
        m = f" /* mode={self.mode!r} */" if self.mode is not ... else ""
        return (
            f"INSERT INTO {self.target_full_name}"
            f"{m}"
        )

    def apply(self, engine: "ExecutionEngine") -> int:
        from yggdrasil.data.options import CastOptions

        table = (
            engine.catalog(self.catalog_name)
            .schema(self.schema_name)
            .table(self.table_name)
        )
        if not table.exists:
            raise FileNotFoundError(
                f"Table {table.full_name!r} does not exist on {engine!r}."
            )
        options_kwargs: dict[str, Any] = {}
        if self.mode is not ...:
            options_kwargs["mode"] = self.mode
        if self.match_by:
            options_kwargs["match_by"] = self.match_by
        options = CastOptions(**options_kwargs) if options_kwargs else None
        # Mutable container so the generator can publish the running
        # total back to the caller after the writer has drained it.
        total = [0]

        def _tally() -> Iterable["pa.RecordBatch"]:
            for batch in _coerce_to_batches(self.data):
                total[0] += batch.num_rows
                yield batch

        table.write_arrow_batches(_tally(), options=options)
        return total[0]


class Select(ExecutionStatement):
    plan_type_id: ClassVar[PlanTypeId] = PlanTypeId.SELECT

    """Read rows from a table or view as a :class:`Tabular`.

    ``apply`` returns the live :class:`ExecutionTable` / :class:`ExecutionView`
    handle — every downstream read method (``read_arrow_table``,
    ``read_polars_frame``, ``read_pandas_frame``, …) flows through the
    :class:`ExecutionStatementResult` because it forwards Tabular hooks to
    its output.
    """

    def __init__(
        self,
        catalog_name: str,
        schema_name: str,
        name: str,
        *,
        options: Any = None,
        key: "str | None" = None,
    ) -> None:
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.name = name
        self.options = options
        super().__init__(key=key)

    @property
    def target_full_name(self) -> str:
        """Dotted ``catalog.schema.name`` identifier of the read target."""
        return f"{self.catalog_name}.{self.schema_name}.{self.name}"

    def _render_text(self) -> str:
        return (
            f"SELECT * FROM {self.target_full_name}"
        )

    def apply(self, engine: "ExecutionEngine") -> "Tabular":
        schema_handle = engine.catalog(self.catalog_name).schema(self.schema_name)
        # Tables before views — same precedence as ``schema[name]``.
        table = schema_handle.table(self.name)
        if table.exists:
            return table
        view = schema_handle.view(self.name)
        if view.exists:
            return view
        raise FileNotFoundError(
            f"{self.catalog_name}.{self.schema_name}.{self.name!r} resolves to "
            "neither a table nor a view."
        )


# ── show / list ─────────────────────────────────────────────────────────


class ShowCatalogs(ExecutionStatement):
    plan_type_id: ClassVar[PlanTypeId] = PlanTypeId.SHOW_CATALOGS

    def _render_text(self) -> str:
        return "SHOW CATALOGS"

    def apply(self, engine: "ExecutionEngine") -> "list[str]":
        return sorted(c.name for c in engine.catalogs())


class ShowSchemas(ExecutionStatement):
    plan_type_id: ClassVar[PlanTypeId] = PlanTypeId.SHOW_SCHEMAS

    def __init__(
        self,
        catalog_name: str,
        *,
        key: "str | None" = None,
    ) -> None:
        self.catalog_name = catalog_name
        super().__init__(key=key)

    def _render_text(self) -> str:
        return f"SHOW SCHEMAS IN {self.catalog_name}"

    def apply(self, engine: "ExecutionEngine") -> "list[str]":
        catalog = engine.catalog(self.catalog_name)
        if not catalog.exists:
            raise FileNotFoundError(
                f"Catalog {self.catalog_name!r} does not exist on {engine!r}."
            )
        return sorted(s.name for s in catalog.schemas())


class ShowTables(ExecutionStatement):
    plan_type_id: ClassVar[PlanTypeId] = PlanTypeId.SHOW_TABLES

    def __init__(
        self,
        catalog_name: str,
        schema_name: str,
        *,
        key: "str | None" = None,
    ) -> None:
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        super().__init__(key=key)

    def _render_text(self) -> str:
        return f"SHOW TABLES IN {self.catalog_name}.{self.schema_name}"

    def apply(self, engine: "ExecutionEngine") -> "list[str]":
        schema_handle = engine.catalog(self.catalog_name).schema(self.schema_name)
        if not schema_handle.exists:
            raise FileNotFoundError(
                f"Schema {self.catalog_name}.{self.schema_name!r} does not exist."
            )
        return sorted(t.name for t in schema_handle.tables())


class ShowViews(ExecutionStatement):
    plan_type_id: ClassVar[PlanTypeId] = PlanTypeId.SHOW_VIEWS

    def __init__(
        self,
        catalog_name: str,
        schema_name: str,
        *,
        key: "str | None" = None,
    ) -> None:
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        super().__init__(key=key)

    def _render_text(self) -> str:
        return f"SHOW VIEWS IN {self.catalog_name}.{self.schema_name}"

    def apply(self, engine: "ExecutionEngine") -> "list[str]":
        schema_handle = engine.catalog(self.catalog_name).schema(self.schema_name)
        if not schema_handle.exists:
            raise FileNotFoundError(
                f"Schema {self.catalog_name}.{self.schema_name!r} does not exist."
            )
        return sorted(v.name for v in schema_handle.views())


# ── helpers ─────────────────────────────────────────────────────────────


def _coerce_to_batches(data: Any) -> "Iterable[pa.RecordBatch]":
    """Project *data* into an iterable of :class:`pa.RecordBatch`.

    Handles the common shapes :class:`Insert` callers reach for —
    :class:`pa.Table` (drained via :meth:`pa.Table.to_batches`),
    :class:`pa.RecordBatch` (wrapped into a one-element list), an
    iterable of either, or another :class:`Tabular` (drained via
    :meth:`Tabular.read_arrow_batches`).
    """
    import pyarrow as pa  # already a hard dep; local for clarity

    from yggdrasil.io.tabular.base import Tabular

    if isinstance(data, pa.Table):
        return data.to_batches()
    if isinstance(data, pa.RecordBatch):
        return [data]
    if isinstance(data, Tabular):
        return data.read_arrow_batches()
    # Generic iterable of batches / tables.
    return data


