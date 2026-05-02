"""
Databricks SQL engine utilities.

A thin execution and table-management layer over two inner
:class:`StatementExecutor` instances, composed via *has-a*:

- :class:`SparkStatementExecutor` (held on ``spark``) — used when a
  :class:`pyspark.sql.SparkSession` is reachable.
- :class:`SQLWarehouse` (resolved via :meth:`warehouse`) — the Databricks
  SQL warehouse API path.

The engine itself owns no statement logic.  It:

- picks which inner executor handles a given call
- delegates statement preparation to
  :meth:`WarehousePreparedStatement.prepare` /
  :meth:`SparkPreparedStatement.from_` (no parallel impl in the engine)
- generates DML SQL (INSERT/MERGE/DELETE/OPTIMIZE/VACUUM) once via
  :func:`_build_dml_statements` and runs it through whichever inner
  executor matches the source shape

Insert paths
------------
All three (:meth:`arrow_insert_into`, :meth:`spark_insert_into`,
:meth:`sql_insert_into`) feed the same DML generator.  They differ only in
how the *source* is prepared:

============= ========================== =============================
Path          Source preparation          Source reference
============= ========================== =============================
arrow         stage Parquet to Volume     ``{__tmpsrc__}`` external arg
spark         register temp view          ``\\`tmp_view_xxx\\```
sql           wrap caller's query + CAST  ``(SELECT ... FROM (q))``
============= ========================== =============================

Save modes
----------
- ``append`` — insert-only.  With ``match_by``: only non-matching rows.
- ``overwrite`` — drop, then insert/append.
- ``truncate`` — in-place wipe + insert.  With ``match_by``: targeted
  ``DELETE`` of matching keys instead of full wipe; schema preserved.
- ``auto`` — default; with ``match_by`` performs upsert.

Merge ``ON`` is built with ``<=>`` (null-safe) so ``NULL`` matches ``NULL``.

Merge pruning
-------------
When ``prune_by`` is supplied (or ``"auto"`` to use the table's partition
fields), the engine collects distinct values per prune key from the source
and appends ``T.<col> IN (...)`` predicates to merge ``ON`` / DELETE
clauses.  Delta's planner uses these target-side predicates to skip whole
files / partitions.  Combined with :meth:`Expr.in_`'s automatic compaction
of contiguous integer/date runs into ``BETWEEN`` clauses, the predicate
stays cheap even for thousands of distinct partition values.

Staging
-------
Only the arrow path stages to Volume.  The Spark path registers an
in-memory temp view and runs the same SQL via ``spark.sql(...)``.  The
sql path doesn't stage at all — it wraps the caller's query.

Per-statement ``external_volume_paths`` carries the staged source
reference; the warehouse statement-batch coercer rewrites ``{alias}``
tokens against that map at submit time.  No batch-wide registry lives on
the engine itself.

Retry semantics for DML
-----------------------
Insert paths apply caller-supplied ``retry`` config (a
:class:`WaitingConfig` arg) only to DML statements (INSERT / MERGE /
DELETE).  TRUNCATE, OPTIMIZE, and VACUUM stay non-retryable —
re-running TRUNCATE is dangerous after an INSERT has succeeded in the
same batch, OPTIMIZE/VACUUM are best-effort maintenance and a re-run
on transient failure costs more than it saves.  See
:func:`_classify_dml` for the classification.
"""


from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Union,
)

import pyarrow as pa

from yggdrasil.data.cast.options import CastOptions
from yggdrasil.data.executor import StatementExecutor
from yggdrasil.data.expr import Expr, Predicate
from yggdrasil.data.statement import PreparedStatement, StatementResult, StatementBatch
from yggdrasil.databricks.fs import VolumePath
from yggdrasil.databricks.sql.sql_utils import quote_ident
from yggdrasil.databricks.warehouse import (
    SQLWarehouse,
    WarehousePreparedStatement,
    WarehouseStatementResult,
)
from yggdrasil.databricks.warehouse.wh_utils import DEFAULT_ALL_PURPOSE_SERVERLESS_NAME
from yggdrasil.dataclasses import WaitingConfig, WaitingConfigArg
from yggdrasil.environ import PyEnv
from yggdrasil.io.buffer.primitive import ParquetIO
from yggdrasil.io.enums import Mode
from yggdrasil.spark.executor import SparkStatementExecutor
from yggdrasil.spark.statement import SparkPreparedStatement, SparkStatementResult
from .catalogs import Catalogs
from .schemas import Schemas
from .table import Table
from .tables import Tables
from ..client import DatabricksService

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas
    import polars
    import pyspark
    from pyspark.sql import SparkSession

__all__ = ["SQLEngine"]


_ALIAS_TMPSRC = "__tmpsrc__"
_DEFAULT_WAREHOUSE_RECHECK_S = 30


# Statement leading keyword classifier — strips comments/whitespace then
# pulls the first word.  Used by :func:`_classify_dml` to decide whether
# to apply caller-supplied retry config.
_DML_HEAD_RE = re.compile(
    r"\A(?:\s+|--[^\n]*\n|--[^\n]*\Z|/\*.*?\*/)*"
    r"(?P<kw>[A-Za-z]+)",
    re.DOTALL,
)
_DML_KEYWORDS: frozenset[str] = frozenset({"INSERT", "MERGE", "DELETE", "UPDATE"})


def _classify_dml(sql: str) -> bool:
    """Return True when ``sql`` looks like an INSERT/MERGE/DELETE/UPDATE.

    Used to decide whether to broadcast retry config onto a generated
    statement.  TRUNCATE / OPTIMIZE / VACUUM / CREATE / ALTER all fall
    through as False.
    """
    if not sql:
        return False
    m = _DML_HEAD_RE.match(sql)
    if not m:
        return False
    return m.group("kw").upper() in _DML_KEYWORDS


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
# Shared SQL helpers
# ---------------------------------------------------------------------------


def _build_match_condition(
    match_by: list[str],
    *,
    left_alias: str,
    right_alias: str,
    null_safe: bool = True,
    extra_predicates: Optional[Iterable[str]] = None,
) -> str:
    """Build a merge ``ON`` expression from key columns and optional extras."""
    op = "<=>" if null_safe else "="
    clauses = [
        f"{left_alias}.{quote_ident(k)} {op} {right_alias}.{quote_ident(k)}"
        for k in match_by
    ]
    if extra_predicates:
        clauses.extend(p for p in extra_predicates if p)
    return " AND ".join(clauses)


def _build_prune_predicates(
    prune_values: Mapping[str, Iterable[Any]],
    *,
    target_alias: str,
) -> list[str]:
    """Convert ``{column: [values]}`` into target-side ``IN`` predicates.

    Compound predicates (top-level ``OR`` from NULL-aware expansion or
    multi-run compaction) are wrapped in parens so they bind correctly
    when stitched into a merge ``ON`` joined by ``AND``.
    """
    predicates: list[str] = []
    for col, vals in prune_values.items():
        materialized = tuple(vals)
        if not materialized:
            continue
        pred = Expr(col, flavor="databricks", alias=target_alias).in_(materialized)
        sql = pred.to_sql()
        if pred.kind != "leaf":
            sql = f"({sql})"
        predicates.append(sql)
    return predicates


def _wrap_user_predicate(pred: Predicate, *, target_alias: str) -> str:
    """Render a user ``where=`` predicate aliased to ``target_alias``,
    parenthesizing if compound (so it binds correctly in an ``AND`` chain).
    """
    aliased = pred.with_table_alias(target_alias)
    sql = aliased.to_sql()
    if aliased.kind != "leaf":
        sql = f"({sql})"
    return sql


def _collect_prune_values_polars(
    buffer: ParquetIO,
    prune_by: list[str],
) -> dict[str, tuple[Any, ...]]:
    """Single-pass distinct over a staged Parquet buffer."""
    df = buffer.scan_polars().select(*prune_by).unique().collect()
    return {col: tuple(df.get_column(col).to_list()) for col in prune_by}


def _collect_prune_values_spark(
    data_df: Any,
    prune_by: list[str],
) -> dict[str, tuple[Any, ...]]:
    """Spark-side equivalent of :func:`_collect_prune_values_polars`."""
    rows = data_df.select(*prune_by).distinct().collect()
    return {col: tuple(row[col] for row in rows) for col in prune_by}


def _resolve_prune_by(
    prune_by: list[str] | str | None,
    fallback_partition_fields: Iterable[Any],
) -> Optional[list[str]]:
    """Normalize ``prune_by``: ``"auto"`` → partition field names, else as-is."""
    if prune_by == "auto":
        return [f.name for f in fallback_partition_fields] or None
    if prune_by:
        return list(prune_by)
    return None


# ---------------------------------------------------------------------------
# Unified DML statement generator
# ---------------------------------------------------------------------------


def _build_dml_statements(
    *,
    target_location: str,
    source_sql: str,
    columns: list[str],
    mode: Mode,
    match_by: Optional[list[str]],
    update_cols: Optional[list[str]],
    prune_predicates: list[str],
    zorder_by: Optional[list[str]] = None,
    optimize_after_merge: bool = False,
    vacuum_hours: Optional[int] = None,
) -> list[str]:
    """Generate INSERT / MERGE / DELETE / OPTIMIZE / VACUUM SQL.

    Path-agnostic — feeds from any source-SQL fragment that names the
    rows to merge.  ``prune_predicates`` are pre-rendered & parenthesized;
    they get AND-stitched onto ``ON`` clauses but NOT plain INSERT.
    """
    cols_quoted = ", ".join(quote_ident(c) for c in columns)
    statements: list[str] = []

    if mode in (Mode.TRUNCATE, Mode.OVERWRITE):
        if mode == Mode.TRUNCATE and match_by:
            # Targeted DELETE then INSERT — preserves schema.
            key_cols = ", ".join(quote_ident(k) for k in match_by)
            on_condition = _build_match_condition(
                match_by, left_alias="T", right_alias="S",
                null_safe=True, extra_predicates=prune_predicates,
            )
            statements.extend([
                (
                    f"DELETE FROM {target_location} AS T\n"
                    f"USING (\n"
                    f"  SELECT DISTINCT {key_cols} FROM ({source_sql}) AS src\n"
                    f") AS S\n"
                    f"ON {on_condition}"
                ),
                f"INSERT INTO {target_location} ({cols_quoted})\n{source_sql}",
            ])
        elif mode == Mode.TRUNCATE:
            statements.extend([
                f"TRUNCATE TABLE {target_location}",
                f"INSERT INTO {target_location} ({cols_quoted})\n{source_sql}",
            ])
        else:
            # OVERWRITE — caller already issued table.delete(); fall through to INSERT.
            statements.append(
                f"INSERT INTO {target_location} ({cols_quoted})\n{source_sql}"
            )

    elif match_by:
        on_condition = _build_match_condition(
            match_by, left_alias="T", right_alias="S",
            null_safe=True, extra_predicates=prune_predicates,
        )
        insert_clause = (
            f"WHEN NOT MATCHED THEN INSERT ({cols_quoted}) "
            f"VALUES ({', '.join(f'S.{quote_ident(c)}' for c in columns)})"
        )

        if mode == Mode.APPEND:
            # Append-with-keys: INSERT only the rows that aren't already there.
            statements.append(
                f"MERGE INTO {target_location} AS T\n"
                f"USING (\n{source_sql}\n) AS S\n"
                f"ON {on_condition}\n"
                f"{insert_clause}"
            )
        else:
            update_cols_effective = (
                update_cols
                if update_cols is not None
                else [c for c in columns if c not in match_by]
            )
            update_clause = ""
            if update_cols_effective:
                update_set = ", ".join(
                    f"T.{quote_ident(c)} = S.{quote_ident(c)}"
                    for c in update_cols_effective
                )
                update_clause = f"WHEN MATCHED THEN UPDATE SET {update_set}\n"

            statements.append(
                f"MERGE INTO {target_location} AS T\n"
                f"USING (\n{source_sql}\n) AS S\n"
                f"ON {on_condition}\n"
                f"{update_clause}"
                f"{insert_clause}"
            )
    else:
        statements.append(
            f"INSERT INTO {target_location} ({cols_quoted})\n{source_sql}"
        )

    # Maintenance tail.
    if zorder_by:
        zorder_cols = ", ".join(quote_ident(c) for c in zorder_by)
        statements.append(f"OPTIMIZE {target_location} ZORDER BY ({zorder_cols})")
    if optimize_after_merge and match_by:
        statements.append(f"OPTIMIZE {target_location}")
    if vacuum_hours is not None:
        statements.append(f"VACUUM {target_location} RETAIN {int(vacuum_hours)} HOURS")

    return statements


# ---------------------------------------------------------------------------
# SQLEngine
# ---------------------------------------------------------------------------


@dataclass
class SQLEngine(DatabricksService, StatementExecutor):
    """Unified SQL execution and Delta-table write engine for Databricks.

    Composes two inner executors:

    - ``spark`` — :class:`SparkStatementExecutor` for Spark-side execution.
    - the warehouse handle resolved by :meth:`warehouse` —
      :class:`SQLWarehouse`, also a :class:`StatementExecutor`.

    Routing (:meth:`_pick_engine`): explicit override → caller-supplied
    session → executor's own session → environment session → fall back to
    warehouse API.
    """

    catalog_name: str | None = None
    schema_name: str | None = None
    default_warehouse: Optional[SQLWarehouse] = field(
        default=None, repr=False, hash=False, compare=False,
    )
    spark: SparkStatementExecutor = field(
        default_factory=SparkStatementExecutor, repr=False, compare=False,
    )
    _last_default_wh_check: float = field(
        default=0.0, init=False, repr=False, hash=False, compare=False,
    )

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
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        warehouse: Optional[SQLWarehouse | str] = None,
    ) -> SQLEngine:
        """Return a re-scoped engine, sharing the same inner Spark executor."""
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
                raise_error=True,
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
    ) -> WarehouseStatementResult | SparkStatementResult:
        """Dispatch by concrete statement type to the matching inner executor.

        Spark statements go to :attr:`spark`; warehouse statements get
        routed to whichever warehouse their ``warehouse_id`` /
        ``warehouse_name`` resolves to.
        """
        if isinstance(statement, SparkPreparedStatement):
            return self.spark._submit_statement(statement)

        if not isinstance(statement, WarehousePreparedStatement):
            statement = WarehousePreparedStatement.from_(statement)

        wh = self.warehouse(
            warehouse_id=statement.warehouse_id,
            warehouse_name=statement.warehouse_name,
        )
        return wh._submit_statement(statement)

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
        external_tables: Mapping[str, "VolumePath | Any"] | None = None,
        parameters: Mapping[str, Any] | None = None,
        # Result-level retry policy — forwarded to
        # WarehousePreparedStatement.prepare on the warehouse path.
        # Ignored on the Spark path (Spark statements don't go through
        # the WarehouseStatementResult.retry loop in the engine layer).
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
                external_tables is None
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
            prepared = SparkPreparedStatement(
                text=str(statement.text if hasattr(statement, "text") else statement).strip(),
                spark_session=session,
                row_limit=row_limit,
            )
            if retry is not None:
                logger.debug(
                    "Ignoring retry on Spark execution path — Spark statements "
                    "use driver-side retry, not StatementResult.retry()."
                )
        else:
            prepared = WarehousePreparedStatement.prepare(
                statement,
                parameters=parameters,
                external_data=external_tables,
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
        statements: Iterable[str | PreparedStatement | StatementResult] | Mapping[str, str | PreparedStatement | StatementResult],
        *,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        parallel: Optional[int] = None,
        engine: Optional[Literal["spark", "api"]] = None,
        warehouse_id: str | None = None,
        warehouse_name: str | None = None,
        spark_session: Optional["SparkSession"] = None,
        # Result-level retry policy.  On the warehouse path, broadcast
        # onto each WarehousePreparedStatement before submission.
        # Ignored on the Spark path.
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
                    "Ignoring retry on Spark execution path — Spark statements "
                    "use driver-side retry, not StatementResult.retry()."
                )
            return self.spark.execute_many(
                statements, wait=wait, raise_error=raise_error, parallel=parallel,
            )

        # Warehouse path — broadcast retry config onto warehouse statements.
        if retry is not None:
            statements = self._broadcast_retry(statements, retry)

        return self.warehouse(
            warehouse_id=warehouse_id, warehouse_name=warehouse_name,
        ).execute_many(
            statements, wait=wait, raise_error=raise_error, parallel=parallel,
        )

    @staticmethod
    def _broadcast_retry(
        statements: Iterable[str | PreparedStatement | StatementResult] | Mapping[str, Any],
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
    # insert_into — top-level dispatcher
    # ------------------------------------------------------------------

    def insert_into(
        self,
        data: Union[
            pa.Table, pa.RecordBatch, pa.RecordBatchReader,
            dict, list, str,
            PreparedStatement, StatementResult,
            "pandas.DataFrame", "polars.DataFrame", "pyspark.sql.DataFrame",
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
        update_cols: Optional[list[str]] = None,
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
        # Result-level retry config.  Applied only to DML statements
        # (INSERT/MERGE/DELETE/UPDATE) on the warehouse path.  TRUNCATE,
        # OPTIMIZE, VACUUM stay non-retryable to avoid double-applying
        # destructive ops or wasting time on best-effort maintenance.
        retry: Optional[WaitingConfigArg] = None,
    ) -> None:
        """Insert data into a Delta table using the most appropriate backend.

        Routing:

        - Query-shaped sources (str, ``PreparedStatement``,
          ``StatementResult``) → :meth:`sql_insert_into`
        - Spark DataFrame (or anything when a ``SparkSession`` is reachable)
          → :meth:`spark_insert_into`
        - Otherwise → :meth:`arrow_insert_into` (warehouse path with
          Volume staging)

        ``retry`` is forwarded to whichever path handles the source.
        Only DML statements get the retry config; maintenance statements
        (TRUNCATE/OPTIMIZE/VACUUM) stay non-retryable.
        """
        common = dict(
            mode=mode, location=location,
            catalog_name=catalog_name, schema_name=schema_name, table_name=table_name,
            match_by=match_by, update_cols=update_cols,
            wait=wait, raise_error=raise_error,
            zorder_by=zorder_by, optimize_after_merge=optimize_after_merge,
            vacuum_hours=vacuum_hours,
            table=table, where=where, prune_by=prune_by, prune_values=prune_values,
            retry=retry,
        )

        if isinstance(data, (PreparedStatement, StatementResult)) or PreparedStatement.looks_like_query(data):
            return self.sql_insert_into(data, spark_session=spark_session, **common)

        # Pull a session off a Spark DataFrame if the caller didn't pass one.
        if spark_session is None:
            session_attr = getattr(data, "sparkSession", None)
            spark_session = session_attr if session_attr is not None else self.spark.resolve_session(create=False)

        if spark_session is not None:
            return self.spark_insert_into(
                data=data,
                schema_mode=schema_mode,
                cast_options=cast_options,
                overwrite_schema=overwrite_schema,
                spark_options=spark_options,
                spark_session=spark_session,
                **common,
            )

        return self.arrow_insert_into(
            data=data,
            schema_mode=schema_mode,
            cast_options=cast_options,
            overwrite_schema=overwrite_schema,
            **common,
        )

    # ------------------------------------------------------------------
    # arrow_insert_into — warehouse path, Volume staging
    # ------------------------------------------------------------------

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
        update_cols: Optional[list[str]] = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        zorder_by: Optional[list[str]] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,
        table: Optional[Table] = None,
        where: Predicate | None = None,
        prune_by: list[str] | str | None = None,
        prune_values: Mapping[str, list[Any]] | None = None,
        # Retry config — applied only to DML statements.
        retry: Optional[WaitingConfigArg] = None,
    ) -> None:
        """Insert through the warehouse SQL path with staged Parquet."""
        if isinstance(data, (PreparedStatement, StatementResult)) or PreparedStatement.looks_like_query(data):
            return self.sql_insert_into(
                data,
                mode=mode, location=location,
                catalog_name=catalog_name, schema_name=schema_name, table_name=table_name,
                match_by=match_by, update_cols=update_cols,
                wait=wait, raise_error=raise_error,
                zorder_by=zorder_by, optimize_after_merge=optimize_after_merge,
                vacuum_hours=vacuum_hours,
                table=table, where=where, prune_by=prune_by,
                retry=retry,
            )

        mode_enum = Mode.from_(mode, default=Mode.AUTO)
        catalog_name = catalog_name or self.catalog_name
        schema_name = schema_name or self.schema_name

        target = Table.from_(
            obj=location if table is None else table,
            catalog_name=catalog_name, schema_name=schema_name, table_name=table_name,
            service=self.tables,
        )

        if mode_enum == Mode.OVERWRITE and not match_by:
            target.delete(wait=True, raise_error=False)

        target = target.create(data, mode=schema_mode)
        target_location = target.full_name(safe=True)
        existing_schema = target.collect_schema()
        cast_options = CastOptions.check(options=cast_options).check_target(
            existing_schema.to_field(),
        )

        if match_by == "auto":
            match_by = [f.name for f in existing_schema.primary_fields] or None
        prune_by = _resolve_prune_by(prune_by, existing_schema.partition_fields)

        wait_cfg = WaitingConfig.from_(wait)

        # Stage Parquet to a Volume; harvest prune values in the same pass.
        staging = VolumePath.staging_path(
            client=self.client,
            catalog_name=target.catalog_name,
            schema_name=target.schema_name,
            resource_name=target.table_name,
            max_lifetime=3600,
            temporary=bool(wait_cfg),
        )

        prune_values = prune_values or {}
        with ParquetIO() as buffer:
            buffer.write_table(data, cast_options)
            buffer.seek(0)
            if prune_by:
                prune_values = _collect_prune_values_polars(buffer, prune_by)
                logger.debug(
                    "Arrow pruning %s -> %s",
                    prune_by, {k: len(v) for k, v in prune_values.items()},
                )
            buffer.seek(0)
            staging.write_stream(buffer)

        buffer.clear()
        prune_predicates = _build_prune_predicates(prune_values, target_alias="T") if prune_values else []
        if where is not None:
            prune_predicates.append(_wrap_user_predicate(where, target_alias="T"))

        columns = list(existing_schema.field_names())
        cols_quoted = ", ".join(quote_ident(c) for c in columns)
        source_sql = f"SELECT {cols_quoted} FROM {{{_ALIAS_TMPSRC}}}"

        sql_texts = _build_dml_statements(
            target_location=target_location,
            source_sql=source_sql,
            columns=columns,
            mode=mode_enum,
            match_by=match_by,
            update_cols=update_cols,
            prune_predicates=prune_predicates,
            zorder_by=zorder_by,
            optimize_after_merge=optimize_after_merge,
            vacuum_hours=vacuum_hours,
        )

        # Per-statement external_volume_paths: only the statements that
        # reference {__tmpsrc__} carry the alias mapping.  The warehouse
        # batch coercer rewrites those references at submit time.
        retry_active = retry is not None
        prepared: list[WarehousePreparedStatement] = []
        for sql in sql_texts:
            external_data = (
                {_ALIAS_TMPSRC: staging}
                if (f"{{{_ALIAS_TMPSRC}}}" in sql)
                else None
            )
            stmt = WarehousePreparedStatement.prepare(
                sql,
                external_data=external_data,
                catalog_name=target.catalog_name,
                schema_name=target.schema_name,
            )
            # Apply retry config only to DML — TRUNCATE/OPTIMIZE/VACUUM
            # stay non-retryable on purpose.
            if retry_active and _classify_dml(sql):
                _apply_retry_to_warehouse_statement(stmt, retry)
            prepared.append(stmt)

        logger.debug(
            "Arrow insert -> %s | mode=%s match_by=%s prune_by=%s statements=%d retry=%s",
            target_location, mode_enum, match_by, prune_by, len(prepared),
            retry_active,
        )

        # Force the warehouse path: arrow_insert_into is *the* warehouse
        # entry point — don't redirect to Spark just because a session exists.
        return self.execute_many(prepared, wait=wait_cfg, raise_error=raise_error, engine="api")

    # ------------------------------------------------------------------
    # spark_insert_into — Spark path, temp-view source
    # ------------------------------------------------------------------

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
        update_cols: Optional[list[str]] = None,
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
        # Accepted for API symmetry with arrow_insert_into; ignored here
        # because Spark uses driver-side retry, not the
        # StatementResult.retry() loop.
        retry: Optional[WaitingConfigArg] = None,
    ) -> None:
        """Insert into a Delta table using Spark.

        Casts the input to a Spark DataFrame, registers it as a unique
        temp view, and runs the SAME SQL the warehouse path would emit
        through :meth:`SparkStatementExecutor.execute_many`.

        No Volume staging, no Parquet round-trip.  Same merge / pruning /
        maintenance semantics as the warehouse path; the only environmental
        difference is the source reference (a temp view name vs. an
        external-table parameter).

        ``retry`` is accepted for API symmetry with the other insert
        paths but ignored — Spark already retries Delta concurrent-append
        internally on the driver.  If you need the warehouse-style result
        retry, route through :meth:`arrow_insert_into` instead.
        """
        if isinstance(data, (PreparedStatement, StatementResult)) or PreparedStatement.looks_like_query(data):
            return self.sql_insert_into(
                data,
                mode=mode, location=location,
                catalog_name=catalog_name, schema_name=schema_name, table_name=table_name,
                match_by=match_by, update_cols=update_cols,
                wait=wait, raise_error=raise_error,
                zorder_by=zorder_by, optimize_after_merge=optimize_after_merge,
                vacuum_hours=vacuum_hours,
                table=table, where=where, prune_by=prune_by,
                spark_session=spark_session,
                retry=retry,
            )

        if retry is not None:
            logger.debug(
                "Ignoring retry on spark_insert_into — Spark statements use "
                "driver-side retry."
            )

        from yggdrasil.spark.cast import any_to_spark_dataframe

        mode_enum = Mode.from_(mode, default=Mode.AUTO)
        catalog_name = catalog_name or self.catalog_name
        schema_name = schema_name or self.schema_name

        if table is None:
            table = self.table(
                location=location,
                catalog_name=catalog_name, schema_name=schema_name,
                table_name=table_name,
            )

        # TODO: Fix async databricks notebook.
        wait = True if PyEnv.in_databricks() else wait

        if mode_enum == Mode.OVERWRITE and not match_by:
            table.delete(wait=True, raise_error=False)

        table = table.create(data, mode=schema_mode)
        target_location = table.full_name(safe=True)
        existing_schema = table.collect_schema()
        cast_options = CastOptions.check(options=cast_options).check_target(
            table.collect_data_field(),
        )

        session = spark_session or self.spark.resolve_session(create=True)
        data_df = any_to_spark_dataframe(data, cast_options)

        if match_by == "auto":
            match_by = [f.name for f in existing_schema.primary_fields] or None
        prune_by = _resolve_prune_by(prune_by, existing_schema.partition_fields)

        # Collect prune values from the DataFrame in a single distinct() pass.
        prune_values = prune_values or {}
        if prune_by:
            prune_values = _collect_prune_values_spark(data_df, prune_by)
            logger.debug(
                "Spark pruning %s -> %s",
                prune_by, {k: len(v) for k, v in prune_values.items()},
            )

        prune_predicates = _build_prune_predicates(prune_values, target_alias="T") if prune_values else []
        if where is not None:
            prune_predicates.append(_wrap_user_predicate(where, target_alias="T"))

        # Register the DataFrame as a unique temp view so SQL can reference it.
        view_name = f"_yg_src_{uuid.uuid4().hex}"
        data_df.createOrReplaceTempView(view_name)

        columns = list(existing_schema.field_names())
        cols_quoted = ", ".join(quote_ident(c) for c in columns)
        source_sql = f"SELECT {cols_quoted} FROM {quote_ident(view_name)}"

        sql_texts = _build_dml_statements(
            target_location=target_location,
            source_sql=source_sql,
            columns=columns,
            mode=mode_enum,
            match_by=match_by,
            update_cols=update_cols,
            prune_predicates=prune_predicates,
            zorder_by=zorder_by,
            optimize_after_merge=optimize_after_merge,
            vacuum_hours=vacuum_hours,
        )

        prepared = [SparkPreparedStatement(text=sql, spark_session=session) for sql in sql_texts]

        logger.info(
            "Spark insert -> %s | mode=%s match_by=%s prune_by=%s statements=%d",
            target_location, mode_enum, match_by, prune_by, len(prepared),
        )

        # spark_options that affect SQL behaviour are set as session conf
        # for the duration of the call; the finally block restores them.
        applied_conf = _delta_conf_for(overwrite_schema, spark_options)

        try:
            with self.spark.scoped_spark_conf(session, applied_conf):
                return self.execute_many(prepared, wait=wait, raise_error=raise_error, engine="spark")
        finally:
            try:
                session.catalog.dropTempView(view_name)
            except Exception:
                logger.debug("Failed to drop temp view %r; continuing.", view_name, exc_info=True)

    # ------------------------------------------------------------------
    # sql_insert_into — query source, no staging
    # ------------------------------------------------------------------

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
        update_cols: Optional[list[str]] = None,
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
        # Retry config — applied to DML on the warehouse fallback;
        # forwarded along on the cached / Spark fast paths.
        retry: Optional[WaitingConfigArg] = None,
    ) -> None:
        """Insert into a Delta table from a SQL source query.

        Smart dispatch
        --------------
        1. Cached :class:`StatementResult` → reuse the materialized frame
           via :meth:`insert_into` (no re-execution).
        2. SparkSession reachable → run via :meth:`spark_insert_into` —
           the resulting DataFrame becomes the source.
        3. Otherwise → warehouse-side ``INSERT ... SELECT`` /
           ``MERGE ... USING (q)`` / ``DELETE ... USING (q)`` with a CAST
           projection aligning the user's query schema to the target.

        ``prune_by`` requires a materialized source to harvest distinct
        values.  Honored on the cached + Spark routes; the warehouse
        fallback applies only the user's ``where=`` predicate (re-running
        the source query just to harvest partition values isn't worth it).
        """
        common = dict(
            mode=mode, location=location,
            catalog_name=catalog_name, schema_name=schema_name, table_name=table_name,
            match_by=match_by, update_cols=update_cols,
            wait=wait, raise_error=raise_error,
            zorder_by=zorder_by, optimize_after_merge=optimize_after_merge,
            vacuum_hours=vacuum_hours,
            table=table, where=where, prune_by=prune_by, prune_values=prune_values,
            retry=retry,
        )

        # ---- Fast path 1: cached StatementResult ----
        if isinstance(statement, StatementResult) and statement.cached:
            spark_df = getattr(statement, "spark_dataframe", None)
            cached = spark_df if spark_df is not None else statement.to_arrow_table()
            return self.insert_into(data=cached, spark_session=spark_session, **common)

        # ---- Fast path 2: run in Spark ----
        if spark_session is None:
            spark_session = self.spark.resolve_session(create=False)
        if spark_session is not None:
            text = (
                statement.statement.text
                if isinstance(statement, StatementResult)
                else (statement.text if isinstance(statement, PreparedStatement) else str(statement))
            )
            df = spark_session.sql(text)
            return self.spark_insert_into(data=df, spark_session=spark_session, **common)

        # ---- Fallback: warehouse-side SQL with CAST projection ----
        return self._sql_insert_warehouse_fallback(statement, **common)

    def _sql_insert_warehouse_fallback(
        self,
        statement: "PreparedStatement | StatementResult | str",
        *,
        mode: Mode | str | None,
        location: str | None,
        catalog_name: str | None,
        schema_name: str | None,
        table_name: str | None,
        match_by: Optional[list[str]],
        update_cols: Optional[list[str]],
        wait: WaitingConfigArg,
        raise_error: bool,
        zorder_by: Optional[list[str]],
        optimize_after_merge: bool,
        vacuum_hours: int | None,
        table: Optional[Table],
        where: Predicate | None,
        prune_by: list[str] | str | None,
        prune_values: dict[str, tuple[Any]] | None = None,
        retry: Optional[WaitingConfigArg] = None,
    ) -> None:
        """Warehouse path of :meth:`sql_insert_into`.

        Source is the caller's query wrapped in a CAST projection.  No
        prune-value harvesting (would require re-executing the query).
        Each generated SQL statement keeps the source statement's
        parameters / external_volume_paths so binding still works.

        Retry config is applied only to DML statements
        (INSERT/MERGE/DELETE/UPDATE); maintenance statements
        (TRUNCATE/OPTIMIZE/VACUUM) stay non-retryable.
        """
        # Carry parameters / external volumes from the source statement onto
        # every generated statement (OPTIMIZE/VACUUM don't strictly need them
        # but it's harmless and keeps the code simple).
        base = statement.statement if isinstance(statement, StatementResult) else statement
        source_prepared = WarehousePreparedStatement.from_(base)

        mode_enum = Mode.from_(mode, default=Mode.AUTO)
        catalog_name = catalog_name or self.catalog_name
        schema_name = schema_name or self.schema_name

        target = Table.from_(
            obj=location if table is None else table,
            catalog_name=catalog_name, schema_name=schema_name, table_name=table_name,
            service=self.tables,
        )

        if mode_enum == Mode.OVERWRITE and not match_by:
            target.delete(wait=True, raise_error=False)

        if not target.exists:
            raise ValueError(
                "sql_insert_into requires the target table to exist; "
                f"{target.full_name()!r} was not found."
            )

        target_location = target.full_name(safe=True)
        existing_schema = target.collect_schema()
        fields = list(existing_schema.fields)
        columns = [f.name for f in fields]

        if match_by == "auto":
            match_by = [f.name for f in existing_schema.primary_fields] or None

        cast_projection = ", ".join(
            (
                f"CAST(raw_src.{quote_ident(f.name)} AS "
                f"{f.to_databricks_ddl(put_name=False, put_not_null=False, put_comment=False)})"
                f" AS {quote_ident(f.name)}"
            )
            for f in fields
        )
        source_sql = (
            f"SELECT {cast_projection} FROM (\n{source_prepared.text}\n) AS raw_src"
        )

        prune_predicates: list[str] = []
        if where is not None:
            prune_predicates.append(_wrap_user_predicate(where, target_alias="T"))
        if prune_by:
            logger.debug(
                "prune_by %s ignored on warehouse-fallback sql_insert_into "
                "(would require re-executing source query)", prune_by,
            )

        sql_texts = _build_dml_statements(
            target_location=target_location,
            source_sql=source_sql,
            columns=columns,
            mode=mode_enum,
            match_by=match_by,
            update_cols=update_cols,
            prune_predicates=prune_predicates,
            zorder_by=zorder_by,
            optimize_after_merge=optimize_after_merge,
            vacuum_hours=vacuum_hours,
        )

        retry_active = retry is not None

        # Each generated statement inherits the source statement's parameters
        # and external_volume_paths so any bindings in the user's query
        # carry through to the rewritten INSERT/MERGE/DELETE.
        prepared: list[WarehousePreparedStatement] = []
        for sql in sql_texts:
            stmt = WarehousePreparedStatement.prepare(
                sql,
                parameters=source_prepared.parameters,
                external_volume_paths=source_prepared.external_volume_paths,
                catalog_name=catalog_name,
                schema_name=schema_name,
            )
            if retry_active and _classify_dml(sql):
                _apply_retry_to_warehouse_statement(stmt, retry)
            prepared.append(stmt)

        logger.info(
            "SQL insert -> %s | mode=%s match_by=%s statements=%d retry=%s",
            target_location, mode_enum, match_by, len(prepared), retry_active,
        )

        if prepared:
            self.execute_many(prepared, wait=wait, raise_error=raise_error, engine="api")

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
            catalog_name=catalog_name, schema_name=schema_name, table_name=table_name,
        ).delete(wait=wait, raise_error=raise_error)

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
            catalog_name=catalog_name, schema_name=schema_name, table_name=table_name,
        )
        return target.create(definition=definition, if_not_exists=True, **kwargs)


# ---------------------------------------------------------------------------
# Spark-conf scoping helpers
# ---------------------------------------------------------------------------


def _delta_conf_for(
    overwrite_schema: bool | None,
    spark_options: Optional[Dict[str, Any]],
) -> dict[str, str]:
    """Translate caller-facing knobs into Spark session conf keys.

    Currently only ``overwrite_schema`` / ``spark_options["overwriteSchema"]``
    map to ``spark.databricks.delta.schema.autoMerge.enabled``.
    """
    out: dict[str, str] = {}
    if overwrite_schema or (spark_options and spark_options.get("overwriteSchema")):
        out["spark.databricks.delta.schema.autoMerge.enabled"] = "true"
    return out