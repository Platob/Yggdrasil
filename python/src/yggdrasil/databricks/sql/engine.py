"""
Databricks SQL engine utilities.

A thin execution and table-management layer that routes SQL through
either:

- The Spark session (via :class:`SparkSQL`) when a
  :class:`pyspark.sql.SparkSession` is reachable (Databricks Connect).
- The warehouse API (via :class:`SQLWarehouse`) otherwise.

The engine itself owns no statement logic. It picks which inner path
handles a given call and delegates statement preparation to
:meth:`WarehousePreparedStatement.prepare` / :class:`SparkSQL`.

Insert paths
------------
The DML write logic (arrow / spark / sql) lives on :class:`Table`
(see :mod:`yggdrasil.databricks.table.table`).  The engine's
:meth:`insert_into` / :meth:`arrow_insert_into` /
:meth:`spark_insert_into` / :meth:`sql_insert_into` resolve a target
:class:`Table` from the caller's parameters and forward to the matching
``Table`` method.
"""

from __future__ import annotations

import logging
import time
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Union,
)

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.io.tabular import Tabular
from yggdrasil.execution.expr import Predicate
from yggdrasil.databricks.fs import VolumePath
from yggdrasil.databricks.warehouse import (
    DatabricksSQL,
    ExternalStatementData,
    SQLWarehouse,
    WarehousePreparedStatement,
    WarehouseStatementBatch,
)
from yggdrasil.databricks.warehouse.wh_utils import DEFAULT_ALL_PURPOSE_SERVERLESS_NAME
from yggdrasil.dataclasses import WaitingConfig, WaitingConfigArg
from yggdrasil.enums import Mode
from yggdrasil.databricks.sql.sql_utils import looks_like_query
from yggdrasil.databricks.catalog.catalogs import Catalogs
from yggdrasil.databricks.schema.schemas import Schemas
from yggdrasil.databricks.table.table import Table
from yggdrasil.databricks.table.tables import Tables
from ..client import DatabricksService

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import pandas
    import polars
    import pyspark
    from pyspark.sql import SparkSession

    from yggdrasil.spark.tabular import SparkDataset

__all__ = ["SQLEngine"]


_DEFAULT_WAREHOUSE_RECHECK_S = 30


def _coerce_external_data_for_spark(
    external_data: Optional[Mapping[str, Any]],
) -> Optional[dict[str, ExternalStatementData]]:
    """Normalize engine-level ``external_data`` for the Spark path.

    Mirrors :meth:`WarehousePreparedStatement.check_external_data`'s
    permissiveness — accepts the same value shapes the warehouse path
    does so a single ``external_data=`` argument works in either
    mode without coercion at the call site.
    """
    if not external_data:
        return None

    from yggdrasil.io.tabular import ArrowTabular
    from yggdrasil.arrow.cast import any_to_arrow_table

    out: dict[str, ExternalStatementData] = {}
    for alias, value in external_data.items():
        if isinstance(value, ExternalStatementData):
            entry = value
            if entry.text_key != alias:
                entry = ExternalStatementData(
                    alias,
                    tabular=entry.tabular,
                    text_value=entry.text_value,
                )
            out[alias] = entry
            continue
        if isinstance(value, VolumePath):
            out[alias] = ExternalStatementData(
                alias,
                text_value=WarehousePreparedStatement.volume_path_text_value(value),
            )
            continue
        if isinstance(value, Tabular):
            out[alias] = ExternalStatementData(alias, tabular=value)
            continue
        if isinstance(value, str):
            out[alias] = ExternalStatementData(alias, text_value=value)
            continue
        if isinstance(value, tuple) and len(value) == 2:
            tabular, text_value = value
            out[alias] = ExternalStatementData(
                alias,
                tabular=tabular,
                text_value=text_value,
            )
            continue

        # Raw frame: wrap so the Spark register-views step has a real
        # :class:`Tabular`.
        try:
            arrow = any_to_arrow_table(value)
        except Exception as e:
            raise TypeError(
                f"external_data[{alias!r}]: cannot bind {type(value).__name__} "
                f"to Spark — accepts VolumePath, ExternalStatementData, Tabular, "
                f"str, (tabular, text_value), or any frame any_to_arrow_table "
                f"can convert: {e}"
            ) from e
        out[alias] = ExternalStatementData(alias, tabular=ArrowTabular(arrow))

    return out or None


def _apply_retry_to_warehouse_statement(
    stmt: WarehousePreparedStatement,
    retry: Optional[WaitingConfigArg],
) -> None:
    """Install ``retry`` on a warehouse statement, in place."""
    if retry is None:
        return
    if retry is False:
        stmt.retry = None
        return
    stmt.retry = WaitingConfig.from_(retry)


# ---------------------------------------------------------------------------
# SQLEngine
# ---------------------------------------------------------------------------


class SQLEngine(DatabricksService):
    """Unified SQL execution and Delta-table write engine for Databricks.

    Routes SQL through two paths:

    - Spark — via :class:`SparkSQL` when a :class:`pyspark.sql.SparkSession`
      is reachable (Databricks Connect or local).
    - Warehouse API — via :class:`SQLWarehouse`, the Databricks SQL
      warehouse HTTP path.

    Routing (:meth:`_pick_engine`): explicit override -> caller-supplied
    session -> environment session -> fall back to warehouse API.
    """

    _SINGLETON_TTL: ClassVar[Any] = None

    @classmethod
    def _singleton_key(
        cls,
        client=None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        default_warehouse: Optional[SQLWarehouse] = None,
        **kwargs: Any,
    ) -> Any:
        return (cls, client, catalog_name, schema_name, default_warehouse)

    def __init__(
        self,
        client=None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        default_warehouse: Optional[SQLWarehouse] = None,
        **kwargs: Any,
    ):
        if getattr(self, "_initialized", False):
            return
        super().__init__(client=client)
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.default_warehouse = default_warehouse
        self._last_default_wh_check = 0.0
        self._initialized = True

    # ------------------------------------------------------------------
    # Spark session helpers
    # ------------------------------------------------------------------

    def _resolve_spark_session(self, create: bool = False) -> "SparkSession | None":
        """Try to find an active Spark session."""
        try:
            from pyspark.sql import SparkSession as _SS
            active = _SS.getActiveSession()
            if active is not None:
                return active
            if create:
                return self.client.spark()
        except ImportError:
            pass
        return None

    def _has_spark_session(self) -> bool:
        """True if a Spark session is reachable without creating one."""
        return self._resolve_spark_session(create=False) is not None

    # ------------------------------------------------------------------
    # Sub-services
    # ------------------------------------------------------------------

    @property
    def catalogs(self) -> "Catalogs":
        return Catalogs(client=self.client)

    @property
    def schemas(self) -> "Schemas":
        return Schemas(client=self.client, catalog_name=self.catalog_name)

    @property
    def tables(self) -> Tables:
        return Tables(
            client=self.client,
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
        )

    def __call__(
        self,
        obj: str | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        warehouse: Optional[SQLWarehouse | str] = None,
    ) -> Union["SQLEngine", DatabricksSQL]:
        """Return a re-scoped engine or execute a SQL query."""
        if obj is not None:
            catalog_name = catalog_name or self.catalog_name
            schema_name = schema_name or self.schema_name

            if looks_like_query(obj):
                return self.execute(
                    obj, catalog_name=catalog_name, schema_name=schema_name
                )
            else:
                raise TypeError(
                    f"SQLEngine.__call__ only accepts SQL strings, not {type(obj).__name__}."
                )

        if catalog_name is None and schema_name is None and warehouse is None:
            return self
        if (
            catalog_name == self.catalog_name
            and schema_name == self.schema_name
            and warehouse == self.default_warehouse
        ):
            return self
        if isinstance(warehouse, str):
            warehouse = self.warehouses.find_warehouse(warehouse_name=warehouse)
        return SQLEngine(
            client=self.client,
            catalog_name=catalog_name,
            schema_name=schema_name,
            default_warehouse=warehouse,
        )

    def warehouse(
        self,
        warehouse_id: str | None = None,
        warehouse_name: str | None = None,
    ) -> SQLWarehouse:
        """Resolve the warehouse used by this engine."""
        if self.default_warehouse is None:
            self._last_default_wh_check = time.time()
            self.default_warehouse = self.warehouses.find_warehouse(
                warehouse_id=warehouse_id,
                warehouse_name=warehouse_name,
                find_default=True,
            )
            return self.default_warehouse

        if warehouse_id or warehouse_name:
            return self.warehouses.find_warehouse(
                warehouse_id=warehouse_id,
                warehouse_name=warehouse_name,
            )

        # Periodic re-check for the all-purpose default warehouse.
        if self.default_warehouse.warehouse_name == DEFAULT_ALL_PURPOSE_SERVERLESS_NAME:
            now_s = time.time()
            if (now_s - self._last_default_wh_check) > _DEFAULT_WAREHOUSE_RECHECK_S:
                self.default_warehouse = self.warehouses.find_default()
                self._last_default_wh_check = now_s

        return self.default_warehouse

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _pick_engine(
        self,
        explicit: Optional[Literal["spark", "api"]],
        spark_session: Optional["SparkSession"],
    ) -> Literal["spark", "api"]:
        """Decide whether a statement runs on Spark or the warehouse API."""
        if explicit:
            return explicit
        if spark_session is not None:
            return "spark"
        if self._has_spark_session():
            return "spark"
        return "api"

    # ------------------------------------------------------------------
    # Public execution surface
    # ------------------------------------------------------------------

    def execute(
        self,
        statement: "str | WarehousePreparedStatement | DatabricksSQL",
        *,
        row_limit: int | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        engine: Optional[Literal["spark", "api"]] = None,
        warehouse_id: str | None = None,
        warehouse_name: str | None = None,
        byte_limit: int | None = None,
        spark_session: Optional["SparkSession"] = None,
        external_data: Mapping[str, "VolumePath | Any"] | None = None,
        parameters: Mapping[str, Any] | None = None,
        retry: Optional[WaitingConfigArg] = None,
    ) -> "DatabricksSQL | Tabular":
        """Execute a SQL statement through Spark or the Databricks SQL API."""
        # Already-started result with no rebinding requested -> just wait.
        if isinstance(statement, DatabricksSQL):
            already_running = (
                external_data is None
                and parameters is None
                and statement.started
            )
            if already_running:
                return statement.wait(wait=wait, raise_error=raise_error)
            statement = statement.statement

        catalog_name = catalog_name or self.catalog_name
        schema_name = schema_name or self.schema_name
        engine_choice = self._pick_engine(engine, spark_session)

        if engine_choice == "spark":
            from yggdrasil.spark.sql import SparkSQL

            session = spark_session or self._resolve_spark_session(create=True)
            text = statement.text if isinstance(statement, WarehousePreparedStatement) else str(statement).strip()

            if retry is not None:
                logger.debug(
                    "Ignoring retry config for Spark statement — "
                    "driver-side retry applies"
                )

            result = SparkSQL(query=text, spark=session)
            result.wait(wait=wait, raise_error=raise_error)
            return result
        else:
            prepared = WarehousePreparedStatement.prepare(
                statement,
                client=self.client,
                parameters=parameters,
                external_data=external_data,
                catalog_name=catalog_name,
                schema_name=schema_name,
                warehouse_id=warehouse_id,
                warehouse_name=warehouse_name,
                byte_limit=byte_limit,
                row_limit=row_limit,
                retry=retry,
            )
            wh = self.warehouse(
                warehouse_id=prepared.warehouse_id,
                warehouse_name=prepared.warehouse_name,
            )
            result = wh.send(prepared)
            result.wait(wait=wait, raise_error=raise_error)
            return result

    def execute_many(
        self,
        statements: (
            Iterable["str | WarehousePreparedStatement"]
            | Mapping[str, "str | WarehousePreparedStatement"]
        ),
        *,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        parallel: Optional[int] = None,
        engine: Optional[Literal["spark", "api"]] = None,
        warehouse_id: str | None = None,
        warehouse_name: str | None = None,
        spark_session: Optional["SparkSession"] = None,
        retry: Optional[WaitingConfigArg] = None,
    ) -> WarehouseStatementBatch:
        """Run a collection of statements; return per-statement results in order."""
        engine_choice = self._pick_engine(engine, spark_session)

        if engine_choice == "spark":
            if retry is not None:
                logger.debug(
                    "Ignoring retry config for Spark statement — "
                    "driver-side retry applies"
                )
            # For Spark, execute each statement sequentially as SparkSQL
            from yggdrasil.spark.sql import SparkSQL

            session = spark_session or self._resolve_spark_session(create=True)
            stmt_list = list(statements.values() if isinstance(statements, Mapping) else statements)
            # Wrap in a warehouse batch for a uniform return type.
            wh = self.warehouse(warehouse_id=warehouse_id, warehouse_name=warehouse_name)
            batch = WarehouseStatementBatch(
                executor=wh,
                statements=stmt_list,
                parallel=parallel or 1,
            )
            batch.wait(wait=wait, raise_error=raise_error)
            return batch

        # Warehouse path — broadcast retry config onto warehouse statements.
        if retry is not None:
            statements = self._broadcast_retry(statements, retry)

        return self.warehouse(
            warehouse_id=warehouse_id,
            warehouse_name=warehouse_name,
        ).execute_many(
            statements,
            wait=wait,
            raise_error=raise_error,
            parallel=parallel,
        )

    @staticmethod
    def _broadcast_retry(
        statements: (
            Iterable["str | WarehousePreparedStatement"] | Mapping[str, Any]
        ),
        retry: Optional[WaitingConfigArg],
    ) -> Iterable[Any]:
        """Apply ``retry`` to every warehouse statement in ``statements``."""
        if isinstance(statements, Mapping):
            out: dict[str, Any] = {}
            for key, stmt in statements.items():
                out[key] = SQLEngine._broadcast_retry_one(stmt, retry)
            return out

        return [SQLEngine._broadcast_retry_one(s, retry) for s in statements]

    @staticmethod
    def _broadcast_retry_one(
        stmt: Any,
        retry: Optional[WaitingConfigArg],
    ) -> Any:
        """Best-effort: apply ``retry`` when ``stmt`` is warehouse-typed."""
        if isinstance(stmt, str):
            stmt = WarehousePreparedStatement(text=stmt)
        elif isinstance(stmt, DatabricksSQL):
            stmt = stmt.statement
        if isinstance(stmt, WarehousePreparedStatement):
            _apply_retry_to_warehouse_statement(stmt, retry)
        return stmt

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def table(
        self,
        location: str | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
    ) -> Table:
        """Resolve a table handle."""
        return self.tables.table(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
        )

    # ------------------------------------------------------------------
    # insert_into — thin wrappers over :class:`Table`
    # ------------------------------------------------------------------

    def _resolve_target(
        self,
        *,
        location: str | None,
        catalog_name: str | None,
        schema_name: str | None,
        table_name: str | None,
        table: Optional[Table],
    ) -> Table:
        return Table.from_(
            obj=location if table is None else table,
            catalog_name=catalog_name or self.catalog_name,
            schema_name=schema_name or self.schema_name,
            table_name=table_name,
            service=self.tables,
        )

    def insert_into(
        self,
        data: Union[
            pa.Table,
            pa.RecordBatch,
            pa.RecordBatchReader,
            dict,
            list,
            str,
            WarehousePreparedStatement,
            DatabricksSQL,
            "pandas.DataFrame",
            "polars.DataFrame",
            "pyspark.sql.DataFrame",
        ],
        *,
        mode: Mode | str | None = None,
        schema_mode: Mode | str | None = None,
        location: str | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        cast_options: Optional[CastOptions] = None,
        overwrite_schema: bool | None = None,
        match_by: Optional[list[str]] = None,
        update_column_names: Optional[list[str]] = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        zorder_by: Optional[list[str]] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,
        spark_session: Optional["pyspark.sql.SparkSession"] = None,
        spark_options: Optional[Dict[str, Any]] = None,
        table: Optional[Table] = None,
        where: Predicate | None = None,
        prune_by: list[str] | str | None = None,
        prune_values: dict[str, tuple[Any]] | None = None,
        retry: Optional[WaitingConfigArg] = None,
    ) -> "WarehouseStatementBatch | None":
        """Resolve the target :class:`Table` and call :meth:`Table.insert_into`."""
        target = self._resolve_target(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            table=table,
        )
        return target.insert_into(
            data,
            mode=mode,
            schema_mode=schema_mode,
            cast_options=cast_options,
            overwrite_schema=overwrite_schema,
            match_by=match_by,
            update_column_names=update_column_names,
            wait=wait,
            raise_error=raise_error,
            zorder_by=zorder_by,
            optimize_after_merge=optimize_after_merge,
            vacuum_hours=vacuum_hours,
            spark_session=spark_session,
            spark_options=spark_options,
            where=where,
            prune_by=prune_by,
            prune_values=prune_values,
            retry=retry,
        )

    def arrow_insert_into(
        self,
        data,
        location: str | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        mode: Mode | str | None = None,
        schema_mode: Mode | str | None = None,
        cast_options: Optional[CastOptions] = None,
        overwrite_schema: bool | None = None,
        match_by: Optional[list[str]] = None,
        update_column_names: Optional[list[str]] = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        zorder_by: Optional[list[str]] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,
        table: Optional[Table] = None,
        where: Predicate | None = None,
        prune_by: list[str] | str | None = None,
        prune_values: Mapping[str, list[Any]] | None = None,
        retry: Optional[WaitingConfigArg] = None,
    ) -> "WarehouseStatementBatch | None":
        """Resolve target and forward to :meth:`Table.arrow_insert`."""
        target = self._resolve_target(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            table=table,
        )
        return target.arrow_insert(
            data,
            mode=mode,
            schema_mode=schema_mode,
            cast_options=cast_options,
            overwrite_schema=overwrite_schema,
            match_by=match_by,
            update_column_names=update_column_names,
            wait=wait,
            raise_error=raise_error,
            zorder_by=zorder_by,
            optimize_after_merge=optimize_after_merge,
            vacuum_hours=vacuum_hours,
            where=where,
            prune_by=prune_by,
            prune_values=prune_values,
            retry=retry,
        )

    def spark_insert_into(
        self,
        data: Any,
        *,
        mode: Mode | str | None = None,
        schema_mode: Mode | str | None = None,
        location: str | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        cast_options: Optional[CastOptions] = None,
        overwrite_schema: bool | None = None,
        match_by: Optional[list[str]] = None,
        update_column_names: Optional[list[str]] = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        zorder_by: Optional[list[str]] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,
        spark_options: Optional[Dict[str, Any]] = None,
        table: Optional[Table] = None,
        where: Predicate | None = None,
        prune_by: list[str] | str | None = None,
        prune_values: dict[str, tuple[Any, ...]] | None = None,
        spark_session: Optional["pyspark.sql.SparkSession"] = None,
        retry: Optional[WaitingConfigArg] = None,
    ) -> "WarehouseStatementBatch | None":
        """Resolve target and forward to :meth:`Table.spark_insert`."""
        target = self._resolve_target(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            table=table,
        )
        return target.spark_insert(
            data,
            mode=mode,
            schema_mode=schema_mode,
            cast_options=cast_options,
            overwrite_schema=overwrite_schema,
            match_by=match_by,
            update_column_names=update_column_names,
            wait=wait,
            raise_error=raise_error,
            zorder_by=zorder_by,
            optimize_after_merge=optimize_after_merge,
            vacuum_hours=vacuum_hours,
            spark_options=spark_options,
            where=where,
            prune_by=prune_by,
            prune_values=prune_values,
            spark_session=spark_session,
            retry=retry,
        )

    def sql_insert_into(
        self,
        statement: "WarehousePreparedStatement | DatabricksSQL | str",
        *,
        mode: Mode | str | None = None,
        location: str | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        match_by: Optional[list[str]] = None,
        update_column_names: Optional[list[str]] = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        zorder_by: Optional[list[str]] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,
        table: Optional[Table] = None,
        spark_session: Optional["pyspark.sql.SparkSession"] = None,
        where: Predicate | None = None,
        prune_by: list[str] | str | None = None,
        prune_values: dict[str, tuple[Any]] | None = None,
        retry: Optional[WaitingConfigArg] = None,
    ) -> "WarehouseStatementBatch | None":
        """Resolve target and forward to :meth:`Table.sql_insert`."""
        target = self._resolve_target(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            table=table,
        )
        return target.sql_insert(
            statement,
            mode=mode,
            match_by=match_by,
            update_column_names=update_column_names,
            wait=wait,
            raise_error=raise_error,
            zorder_by=zorder_by,
            optimize_after_merge=optimize_after_merge,
            vacuum_hours=vacuum_hours,
            spark_session=spark_session,
            where=where,
            prune_by=prune_by,
            prune_values=prune_values,
            retry=retry,
        )

    # ------------------------------------------------------------------
    # Drop / create
    # ------------------------------------------------------------------

    def drop_table(
        self,
        location: str | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> None:
        """Drop a table if it exists."""
        return self.table(
            location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
        ).delete(wait=wait, missing_ok=not raise_error)

    def create_table(
        self,
        definition: Union[pa.Field, pa.Schema, Any],
        *,
        full_name: str | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        **kwargs,
    ) -> Table:
        """Create a table if it does not already exist."""
        target = self.table(
            location=full_name,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
        )
        return target.create(definition=definition, missing_ok=True, **kwargs)

    # ------------------------------------------------------------------
    # SparkTabular (Dataset) convenience
    # ------------------------------------------------------------------

    def dataset(
        self,
        sql_or_table: str,
        *,
        schema: Any = None,
    ) -> "SparkDataset":
        """Return a :class:`Dataset` from a SQL query or table name."""
        from yggdrasil.spark.tabular import SparkDataset

        session = self._resolve_spark_session(create=True)
        if looks_like_query(sql_or_table):
            return SparkDataset.from_sql(
                sql_or_table,
                spark_session=session,
                schema=schema,
            )
        return SparkDataset.from_table(
            sql_or_table,
            spark_session=session,
            schema=schema,
        )

    def parallelize(
        self,
        inputs: "Iterable",
        function: "Callable | None" = None,
        *,
        schema: Any = None,
        byte_size: int = 128 * 1024 * 1024,
    ) -> "SparkDataset":
        """Distribute *function* over *inputs* via Spark executors."""
        from yggdrasil.spark.tabular import SparkDataset

        session = self._resolve_spark_session(create=True)
        return SparkDataset.parallelize(
            inputs,
            function,
            schema=schema,
            spark_session=session,
            byte_size=byte_size,
        )
