"""
Databricks SQL engine utilities.

A thin execution and table-management layer over two inner
:class:`StatementExecutor` instances, composed via *has-a*:

- :class:`SparkStatementExecutor` (held on ``spark``) — used when a
  :class:`pyspark.sql.SparkSession` is reachable. Defaults to a
  :class:`DatabricksSparkStatementExecutor` so a missing session is
  built via :meth:`DatabricksClient.spark` (Databricks Connect).
- :class:`SQLWarehouse` (resolved via :meth:`warehouse`) — the Databricks
  SQL warehouse API path.

The engine itself owns no statement logic.  It picks which inner executor
handles a given call and delegates statement preparation to
:meth:`WarehousePreparedStatement.prepare` /
:meth:`SparkPreparedStatement.from_`.

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
from yggdrasil.data.executor import StatementExecutor
from yggdrasil.io.tabular import Tabular
from yggdrasil.execution.expr import Predicate
from yggdrasil.data.statement import (
    ExternalStatementData,
    PreparedStatement,
    StatementResult,
    StatementBatch,
)
from yggdrasil.databricks.fs import VolumePath
from yggdrasil.databricks.warehouse import (
    SQLWarehouse,
    WarehousePreparedStatement,
    WarehouseStatementResult,
)
from yggdrasil.databricks.warehouse.wh_utils import DEFAULT_ALL_PURPOSE_SERVERLESS_NAME
from yggdrasil.dataclasses import WaitingConfig, WaitingConfigArg
from yggdrasil.data.enums import Mode
from yggdrasil.spark.executor import SparkStatementExecutor
from yggdrasil.spark.statement import SparkPreparedStatement, SparkStatementResult
from .spark_executor import DatabricksSparkStatementExecutor
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

    from yggdrasil.spark.tabular import Dataset

__all__ = ["SQLEngine"]


_DEFAULT_WAREHOUSE_RECHECK_S = 30


def _coerce_external_data_for_spark(
    external_data: Optional[Mapping[str, Any]],
) -> Optional[dict[str, ExternalStatementData]]:
    """Normalize engine-level ``external_data`` for the Spark path.

    Mirrors :meth:`WarehousePreparedStatement.check_external_data`'s
    permissiveness — accepts the same value shapes the warehouse path
    does so a single ``external_data=`` argument works in either
    mode without coercion at the call site:

    - :class:`VolumePath` → text-substituted as
      ``parquet.\\`<full_path>\\``` (Spark reads parquet by path).
    - :class:`ExternalStatementData` → passed through (key is taken
      from the mapping key on collisions).
    - :class:`Tabular` → bound for temp-view registration at
      :meth:`SparkStatementResult.start` time.
    - ``str`` → ``text_value`` only (caller already staged it
      somewhere; we just substitute).
    - ``(tabular, text_value)`` tuple → both fields set.
    - Anything else (``pa.Table``, polars / pandas / pyspark frames,
      list-of-dicts, ...) → wrapped in :class:`ArrowTabular` so the
      Spark side has a real :class:`Tabular` to register, matching the
      warehouse side which would have staged the same value as
      Parquet.
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
        # :class:`Tabular`. ``any_to_arrow_table`` handles pa.Table,
        # polars / pandas / pyspark frames, list-of-dicts, etc.
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
    """Install ``retry`` (a :class:`WaitingConfig` arg) on a warehouse
    statement, in place.

    ``None`` is a no-op (don't override).  ``False`` clears any existing
    policy.  Anything else is normalized via :meth:`WaitingConfig.from_`.
    """
    if retry is None:
        return
    if retry is False:
        stmt.retry = None
        return
    stmt.retry = WaitingConfig.from_(retry)


# ---------------------------------------------------------------------------
# SQLEngine
# ---------------------------------------------------------------------------


class SQLEngine(DatabricksService, StatementExecutor):
    """Unified SQL execution and Delta-table write engine for Databricks.

    Composes two inner executors:

    - ``spark`` — :class:`SparkStatementExecutor` for Spark-side execution.
    - the warehouse handle resolved by :meth:`warehouse` —
      :class:`SQLWarehouse`, also a :class:`StatementExecutor`.

    Routing (:meth:`_pick_engine`): explicit override → caller-supplied
    session → executor's own session → environment session → fall back to
    warehouse API.

    Singleton-cached by ``(client, catalog_name, schema_name,
    default_warehouse)`` so two callers asking for the same scope share
    the same Spark sub-executor, the same lazy ``default_warehouse``
    resolution, and the same sub-service caches (catalogs / schemas /
    tables route through ``self.client``, which is itself singleton-
    cached). The ``spark`` field doesn't participate in identity —
    callers that pass a custom Spark executor onto an existing engine
    re-bind it in place.
    """

    # Cache scoped engines for the process lifetime. ``client`` already
    # carries the workspace identity, so two callers asking for the same
    # ``(client, catalog, schema)`` collapse to one engine.
    _SINGLETON_TTL: ClassVar[Any] = None

    @classmethod
    def _singleton_key(
        cls,
        client=None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        default_warehouse: Optional[SQLWarehouse] = None,
        spark: Optional[SparkStatementExecutor] = None,
        **kwargs: Any,
    ) -> Any:
        # ``spark`` is rebindable; the warehouse is identity-bearing
        # (different warehouses ⇒ different engine). ``client`` carries
        # the workspace identity.
        return (cls, client, catalog_name, schema_name, default_warehouse)

    def __init__(
        self,
        client=None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        default_warehouse: Optional[SQLWarehouse] = None,
        spark: Optional[SparkStatementExecutor] = None,
    ):
        if getattr(self, "_initialized", False):
            return
        super().__init__(client=client)
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.default_warehouse = default_warehouse
        # Default to a Databricks Connect-backed executor so a missing
        # session is built via ``client.spark()`` rather than PyEnv's
        # local PySpark bootstrap. Callers that want a custom Spark
        # backend keep passing ``spark=`` explicitly.
        self.spark = (
            spark
            if spark is not None
            else DatabricksSparkStatementExecutor(client=self.client)
        )
        self._last_default_wh_check = 0.0
        self._initialized = True

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
    ) -> Union[SQLEngine, StatementResult]:
        """Return a re-scoped engine, sharing the same inner Spark executor."""
        if obj is not None:
            catalog_name = catalog_name or self.catalog_name
            schema_name = schema_name or self.schema_name

            if PreparedStatement.looks_like_query(obj) or isinstance(
                obj, PreparedStatement
            ):
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
            spark=self.spark,
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
        if self.spark.has_session():
            return "spark"
        return "api"

    # ------------------------------------------------------------------
    # Executor contract — pure delegation
    # ------------------------------------------------------------------

    def _submit_statement(
        self,
        statement: WarehousePreparedStatement | SparkPreparedStatement,
        start: bool = True,
    ) -> WarehouseStatementResult | SparkStatementResult:
        """Dispatch by concrete statement type to the matching inner executor.

        Spark statements go to :attr:`spark`; warehouse statements get
        routed to whichever warehouse their ``warehouse_id`` /
        ``warehouse_name`` resolves to.
        """
        if isinstance(statement, SparkPreparedStatement):
            return self.spark.send(statement, start=start)

        if not isinstance(statement, WarehousePreparedStatement):
            statement = WarehousePreparedStatement.from_(statement)

        wh = self.warehouse(
            warehouse_id=statement.warehouse_id,
            warehouse_name=statement.warehouse_name,
        )
        return wh.send(statement, start=start)

    # ------------------------------------------------------------------
    # Public execution surface
    # ------------------------------------------------------------------

    def execute(
        self,
        statement: str | PreparedStatement | StatementResult,
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
    ) -> StatementResult:
        """Execute a SQL statement through Spark or the Databricks SQL API.

        ``retry`` controls *result-level* retry on the warehouse path
        (what :meth:`StatementResult.retry` does after a terminal
        failure).  Has no effect on the Spark path here.
        """
        # Already-started result with no rebinding requested → just wait.
        if isinstance(statement, StatementResult):
            already_running = (
                external_data is None
                and parameters is None
                and getattr(statement, "started", statement.done)
            )
            if already_running:
                return statement.wait(wait=wait, raise_error=raise_error)
            statement = statement.statement

        catalog_name = catalog_name or self.catalog_name
        schema_name = schema_name or self.schema_name
        engine_choice = self._pick_engine(engine, spark_session)

        if engine_choice == "spark":
            session = spark_session or self.spark.resolve_session(create=True)

            # Carry forward any ``external_data`` already on the input
            # statement so a caller-prepared SparkPreparedStatement keeps
            # its bindings; ``external_data`` from this call layers on
            # top (last write wins on alias collisions).
            if isinstance(statement, PreparedStatement):
                text = statement.text
                merged: dict[str, ExternalStatementData] = (
                    dict(statement.external_data) if statement.external_data else {}
                )
            else:
                text = str(statement).strip()
                merged = {}

            new_external = _coerce_external_data_for_spark(external_data)
            if new_external:
                merged.update(new_external)

            prepared = SparkPreparedStatement(
                text=text,
                spark_session=session,
                row_limit=row_limit,
                external_data=merged or None,
            )
            if retry is not None:
                logger.debug(
                    "Ignoring retry on Spark execution path; Spark statements "
                    "use driver-side retry, not StatementResult.retry()."
                )
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
        return super().execute(prepared, wait=wait, raise_error=raise_error)

    def execute_many(
        self,
        statements: (
            Iterable[str | PreparedStatement | StatementResult]
            | Mapping[str, str | PreparedStatement | StatementResult]
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
    ) -> StatementBatch:
        """Run a collection of statements; return per-statement results in order.

        Statements that already carry per-statement
        ``external_volume_paths`` get their ``{alias}`` substitution from
        the warehouse-batch coercer at submit time — the engine doesn't
        manage a parallel registry.

        ``retry`` is broadcast onto each warehouse statement before
        submission (Spark statements pass through untouched).  ``None``
        leaves whatever the statement already says intact; ``False``
        explicitly clears any existing policy.
        """
        engine_choice = self._pick_engine(engine, spark_session)

        if engine_choice == "spark":
            if retry is not None:
                logger.debug(
                    "Ignoring retry on Spark execution path; Spark statements "
                    "use driver-side retry, not StatementResult.retry()."
                )
            return self.spark.execute_many(
                statements,
                wait=wait,
                raise_error=raise_error,
                parallel=parallel,
            )

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
            Iterable[str | PreparedStatement | StatementResult] | Mapping[str, Any]
        ),
        retry: Optional[WaitingConfigArg],
    ) -> Iterable[Any]:
        """Apply ``retry`` to every warehouse statement in ``statements``.

        Strings get coerced to :class:`WarehousePreparedStatement` so the
        config sticks; non-warehouse statements (Spark, etc.) pass through
        untouched.  Returns a list to preserve the input cardinality
        (matters for mappings — keys must align with statements).
        """
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
        elif isinstance(stmt, StatementResult):
            stmt = stmt.statement
        # Only mutate warehouse statements — Spark statements have their
        # own retry semantics and shouldn't get a warehouse-style policy.
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
    #
    # The actual DML logic lives on :class:`Table`.  These engine methods
    # resolve a target :class:`Table` from the caller's parameters
    # (``location`` / ``catalog_name`` / ``schema_name`` / ``table_name``
    # / a pre-built ``table=``) and forward to it.

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
            PreparedStatement,
            StatementResult,
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
    ) -> "StatementBatch | None":
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
    ) -> "StatementBatch | None":
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
    ) -> "StatementBatch | None":
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
        statement: "PreparedStatement | StatementResult | str",
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
    ) -> "StatementBatch | None":
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
    ) -> "Dataset":
        """Return a :class:`Dataset` from a SQL query or table name.

        Auto-detects SQL (``SELECT``, ``WITH``, …) vs table name.
        Session resolved through Databricks Connect.
        """
        from yggdrasil.spark.tabular import Dataset

        session = self.spark.resolve_session(create=True)
        if PreparedStatement.looks_like_query(sql_or_table):
            return Dataset.from_sql(
                sql_or_table,
                spark_session=session,
                schema=schema,
            )
        return Dataset.from_table(
            sql_or_table,
            spark_session=session,
            schema=schema,
        )

    def parallelize(
        self,
        inputs: "Callable | Iterable",
        inputs_or_schema: "Iterable | None" = None,
        *,
        schema: Any = None,
        byte_size: int = 128 * 1024 * 1024,
    ) -> "Dataset":
        """Distribute *function* over *inputs* via Spark executors, or
        create a Dataset directly from *inputs* when no function is given."""
        from yggdrasil.spark.tabular import Dataset

        session = self.spark.resolve_session(create=True)
        return Dataset.parallelize(
            inputs,
            inputs_or_schema,
            schema=schema,
            spark_session=session,
            byte_size=byte_size,
        )
