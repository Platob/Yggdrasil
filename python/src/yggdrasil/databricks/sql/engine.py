"""
Databricks SQL engine utilities.

This module provides a thin execution and table-management layer across two
Databricks runtimes:

- Spark SQL / Delta Lake, when a SparkSession is available
- Databricks SQL PreparedStatement Execution API, when running outside Spark

It is designed to keep read and write behavior as consistent as possible
between both paths, especially for Delta insert and merge workflows.

Core capabilities
-----------------
- Resolve catalog/schema-scoped table handles
- Execute SQL through Spark or Databricks SQL warehouses
- Insert Arrow, Spark, pandas, polars, dict/list, and similar tabular inputs
  into Delta tables
- Support append, overwrite, and merge-style upsert semantics
- Create tables from schemas or input data when needed

Execution model
---------------
The engine chooses the execution backend as follows:

1. If `engine="spark"` is explicitly requested, Spark SQL is used
2. If `engine="api"` is explicitly requested, the Databricks SQL API is used
3. If no engine is provided:
   - Spark is used when an active SparkSession is available
   - otherwise the Databricks SQL warehouse API is used

Write paths
-----------
Two write strategies are supported:

Spark path
    Used when a SparkSession is available or when the input is already a Spark
    DataFrame. Data is written directly into Delta tables and MERGE semantics
    are implemented through Delta APIs.

Warehouse SQL path
    Used when Spark is not available. Input data is converted to Parquet,
    staged into a temporary Databricks volume, and then inserted or merged into
    the target Delta table through SQL.

Save modes
----------
- `append`
    Insert-only behavior. When `match_by` is provided, only non-matching rows
    are inserted.

- `overwrite`
    Full replacement behavior. The target table is dropped first, then data is
    written back with plain insert/append logic.

- `truncate`
    In-place replacement behavior.  The table structure is preserved.

    * Without ``match_by``: ``TRUNCATE TABLE`` empties the table, then all
      rows from the input are inserted.
    * With ``match_by``: a targeted ``DELETE`` removes every existing row
      whose key appears in the input, then all rows from the input are
      inserted.  This avoids a full table scan while keeping the schema intact.

- `auto`
    Default behavior. When `match_by` is provided, matching rows are updated
    and new rows are inserted.

Merge semantics
---------------
When `match_by` is provided, merge conditions are built using Databricks
null-safe equality (`<=>`) by default so NULL matches NULL.

- `append` + `match_by`
    Insert-only merge

- `auto` + `match_by`
    Upsert merge

- `overwrite`
    Table is dropped first, so merge logic is skipped

Safety and consistency
----------------------
This module is intended to be safe by default:

- SQL identifiers are quoted
- merge conditions are generated from explicit key columns
- schemas are aligned through cast options before writing
- Spark and SQL paths follow the same overwrite and merge rules
"""


from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Literal,
    Mapping,
    Optional,
    OrderedDict,
    TypeVar,
    Union,
)

import pyarrow as pa
from databricks.sdk.service.sql import Disposition

from yggdrasil.concurrent.threading import Job
from yggdrasil.data.cast import CastOptions
from yggdrasil.data.statement import PreparedStatement, StatementBatch
from yggdrasil.databricks.sql.sql_utils import quote_ident, sql_literal
from yggdrasil.dataclasses import ExpiringDict, WaitingConfig, WaitingConfigArg
from yggdrasil.environ import PyEnv
from yggdrasil.io.enums import SaveMode
from .catalogs import Catalogs
from .grants import Grants
from .schemas import Schemas
from .service import DEFAULT_ALL_PURPOSE_SERVERLESS_NAME
from .staging import StagingPath
from .statement import StatementResult
from .table import Table
from .tables import Tables
from .types import PrimaryKeySpec, ForeignKeySpec
from .warehouse import SQLWarehouse
from ..client import DatabricksService

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas
    import polars
    import pyspark
    from pyspark.sql import SparkSession

__all__ = [
    "SQLEngine",
]

R = TypeVar("R")


def _build_match_condition(
    match_by: list[str],
    *,
    left_alias: str,
    right_alias: str,
    null_safe: bool = True,
    extra_predicates: Optional[Iterable[str]] = None,
) -> str:
    """Build a merge ``ON`` expression from key columns and optional extras.

    ``extra_predicates`` are already-rendered SQL fragments (e.g. partition or
    cluster-column scope filters). They are AND-ed onto the equality clause so
    Delta can prune files and narrow its OCC read set — the fix for
    ``DELTA_CONCURRENT_APPEND.WHOLE_TABLE_READ`` conflicts.
    """
    op = "<=>" if null_safe else "="
    clauses = [
        f"{left_alias}.{quote_ident(k)} {op} {right_alias}.{quote_ident(k)}"
        for k in match_by
    ]
    if extra_predicates:
        clauses.extend(p for p in extra_predicates if p)
    return " AND ".join(clauses)


# Delta's DELTA_CONCURRENT_APPEND conflict surfaces under both the Spark and
# warehouse paths. We match on the stable error-code + class-name tokens so the
# detection works across SDK/PyPI versions.
_CONCURRENT_APPEND_TOKENS = (
    "DELTA_CONCURRENT_APPEND",
    "ConcurrentAppendException",
)


def _is_concurrent_append(exc: BaseException) -> bool:
    """Return ``True`` when *exc* is a Delta concurrent-append conflict."""
    pieces = [type(exc).__name__, str(exc)]
    for attr in ("error_code", "message"):
        value = getattr(exc, attr, None)
        if value is not None:
            pieces.append(str(value))
    blob = "\n".join(pieces)
    return any(token in blob for token in _CONCURRENT_APPEND_TOKENS)


def _delta_partition_columns(table: Table) -> list[str]:
    """Target-side partition columns, in partition order.

    Pulled straight from :class:`~databricks.sdk.service.catalog.TableInfo`
    (``columns[*].partition_index``) so no extra round-trip is needed beyond
    the TTL-cached ``table.infos`` fetch.
    """
    try:
        cols = [
            c for c in (table.infos.columns or [])
            if getattr(c, "partition_index", None) is not None
        ]
    except Exception:
        logger.debug(
            "Failed to read partition columns for %s", table.full_name(),
            exc_info=True,
        )
        return []
    cols.sort(key=lambda c: c.partition_index)
    return [c.name for c in cols if getattr(c, "name", None)]


def _delta_cluster_columns(table: Table) -> list[str]:
    """Target-side liquid-cluster columns, read from ``TBLPROPERTIES``.

    Databricks stores them as ``clusteringColumns`` — a JSON list of
    ``[column_path]`` entries. We return the leaf names in declared order.
    """
    try:
        props = table.infos.properties or {}
    except Exception:
        logger.debug(
            "Failed to read cluster columns for %s", table.full_name(),
            exc_info=True,
        )
        return []

    raw = props.get("clusteringColumns")
    if not raw:
        return []

    import json
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return []

    cols: list[str] = []
    for item in parsed or ():
        if isinstance(item, (list, tuple)) and item:
            cols.append(str(item[0]))
        elif isinstance(item, str):
            cols.append(item)
    return [c for c in cols if c]


def _narrow_target_columns(
    table: Table,
    *,
    match_by: Iterable[str] | None,
) -> list[str]:
    """Columns to narrow the merge scan on — partitions first, then cluster keys.

    Columns already in ``match_by`` are dropped (equality already narrows
    them), and duplicates are removed preserving order.
    """
    used: set[str] = set(match_by or ())
    out: list[str] = []
    for col in (*_delta_partition_columns(table), *_delta_cluster_columns(table)):
        if col and col not in used:
            used.add(col)
            out.append(col)
    return out


def _narrowing_predicates_via_subquery(
    cols: Iterable[str],
    *,
    target_alias: str,
    source_expr: str,
) -> list[str]:
    """Render ``alias.col IN (SELECT DISTINCT col FROM <source>)`` predicates.

    ``source_expr`` is inlined — for the warehouse path it can be an
    ``{src}`` staging alias (substituted later by
    :func:`_apply_external_table_aliases`) or a full ``(SELECT ...) AS src``
    subquery. Lets Delta prune target files without us pre-scanning the batch.
    """
    return [
        f"{target_alias}.{quote_ident(col)} IN "
        f"(SELECT DISTINCT {quote_ident(col)} FROM {source_expr})"
        for col in cols
    ]


def _narrowing_predicates_from_spark(
    data_df: Any,
    cols: Iterable[str],
    *,
    target_alias: str,
    max_in_values: int = 500,
) -> list[str]:
    """Collect distinct narrowing values from a Spark DataFrame.

    Falls back to ``BETWEEN min AND max`` when a column's cardinality exceeds
    ``max_in_values`` (keeps the generated SQL from blowing past parser
    limits on wide partition spaces). NULLs are re-introduced with an
    explicit ``OR col IS NULL`` when present in the batch.

    Returns an empty list on any failure — the caller falls back to the
    plain key-only merge predicate.
    """
    cols = [c for c in cols if c in getattr(data_df, "columns", [])]
    if not cols:
        return []

    import pyspark.sql.functions as F

    predicates: list[str] = []
    for col in cols:
        try:
            rows = (
                data_df.select(F.col(col))
                .distinct()
                .limit(max_in_values + 1)
                .collect()
            )
        except Exception:
            logger.debug(
                "Failed to collect distinct %r for merge narrowing", col,
                exc_info=True,
            )
            continue

        values = [r[0] for r in rows]
        if not values:
            continue

        has_null = any(v is None for v in values)
        non_null = [v for v in values if v is not None]
        qcol = f"{target_alias}.{quote_ident(col)}"

        if not non_null:
            predicates.append(f"{qcol} IS NULL")
            continue

        if len(non_null) <= max_in_values:
            in_list = ", ".join(sql_literal(v) for v in non_null)
            pred = f"{qcol} IN ({in_list})"
        else:
            try:
                mm = data_df.agg(F.min(col), F.max(col)).collect()[0]
                lo, hi = mm[0], mm[1]
            except Exception:
                logger.debug(
                    "Failed to collect min/max for %r", col, exc_info=True,
                )
                continue
            if lo is None or hi is None:
                continue
            pred = f"{qcol} BETWEEN {sql_literal(lo)} AND {sql_literal(hi)}"

        if has_null:
            pred = f"({pred} OR {qcol} IS NULL)"
        predicates.append(pred)

    return predicates


def _retry_concurrent_append(
    fn: Callable[[], R],
    *,
    attempts: int,
    base_delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 30.0,
    op: str = "delta merge",
) -> R:
    """Retry ``fn`` when Delta reports a concurrent-append conflict.

    Other exceptions propagate immediately — we only retry the specific
    ``DELTA_CONCURRENT_APPEND`` case, which is safe because an OCC abort
    means no target changes were committed.
    """
    if attempts < 1:
        attempts = 1
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — routed by _is_concurrent_append
            if not _is_concurrent_append(exc):
                raise
            last_exc = exc
            if attempt >= attempts:
                break
            delay = min(max_delay, base_delay * (backoff ** (attempt - 1)))
            import random as _random
            jittered = delay * (0.5 + _random.random())
            logger.warning(
                "%s hit DELTA_CONCURRENT_APPEND (attempt %d/%d); "
                "retrying in %.1fs", op, attempt, attempts, jittered,
            )
            time.sleep(jittered)
    assert last_exc is not None
    raise last_exc


def _staging_parquet_ref(path: StagingPath) -> str:
    """Inline ``parquet.`<path>``` source clause for a staging path."""
    return f"parquet.{quote_ident(str(path.path))}"


def _apply_external_table_aliases(
    statement: str,
    substitutions: Mapping[str, str],
) -> str:
    """Replace ``{alias}`` occurrences in ``statement`` with staging references."""
    if not substitutions:
        return statement
    rendered = statement
    for alias, replacement in substitutions.items():
        rendered = rendered.replace("{" + alias + "}", replacement)
    return rendered


@dataclass(frozen=True)
class SQLEngine(DatabricksService):
    """
    Unified SQL execution and Delta table write engine for Databricks.

    `SQLEngine` provides a single interface for:

    - executing SQL through Spark or Databricks SQL warehouses
    - resolving catalog/schema-scoped tables
    - inserting tabular data into Delta tables
    - applying append, overwrite, or merge/upsert semantics
    - performing optional post-write maintenance such as OPTIMIZE or VACUUM

    The engine prefers Spark execution when a SparkSession is available and
    falls back to the Databricks SQL PreparedStatement Execution API otherwise.

    Scope
    -----
    An engine instance may be bound to a default catalog, schema, and warehouse.
    The instance is lightweight and can be re-scoped by calling it:

        engine(catalog_name="main", schema_name="analytics")

    This returns a new engine sharing the same client and query cache.

    Notes
    -----
    - Spark and warehouse paths are intended to behave consistently
    - overwrite mode always drops the table before reinserting
    - merge behavior is enabled only when `match_by` is provided
    """

    catalog_name: str | None = None
    schema_name: str | None = None
    default_warehouse: Optional[SQLWarehouse] = field(
        default=None,
        repr=False,
        hash=False,
        compare=False,
    )

    _last_default_wh_check: int = field(
        default=0,
        init=False,
        repr=False,
        hash=False,
        compare=False,
    )
    _cached_queries: Optional[ExpiringDict[str, StatementResult]] = field(
        default=ExpiringDict,
        init=False,
        repr=False,
        hash=False,
        compare=False,
    )

    @property
    def catalogs(self) -> "Catalogs":
        """
        Return the `Catalogs` service scoped to this engine's catalog and schema.
        """
        return Catalogs(
            client=self.client,
        )

    @property
    def schemas(self) -> "Schemas":
        """
        Return the `Schemas` service scoped to this engine's catalog and schema.
        """
        return Schemas(
            client=self.client,
            catalog_name=self.catalog_name,
        )

    @property
    def tables(self) -> Tables:
        """
        Return the `Tables` service scoped to this engine's catalog and schema.
        """
        return Tables(
            client=self.client,
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
        )

    @property
    def grants(self) -> Grants:
        """
        Return the `Grants` service scoped to this engine's catalog and schema.
        """
        return Grants(
            client=self.client,
        )

    def __call__(
        self,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        warehouse: Optional[SQLWarehouse | str] = None,
    ) -> SQLEngine:
        """
        Return a re-scoped engine.

        If no scope changes are provided, the current instance is returned.
        If the requested scope is identical to the current scope, the current
        instance is also returned.

        Args:
            catalog_name:
                Catalog override for the returned engine.
            schema_name:
                Schema override for the returned engine.
            warehouse:
                Default warehouse override. May be a warehouse instance or a
                warehouse name.

        Returns:
            A new `SQLEngine` sharing the same client and query cache, or the
            current instance if no change is needed.
        """
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

        built = SQLEngine(
            client=self.client,
            catalog_name=catalog_name,
            schema_name=schema_name,
            default_warehouse=warehouse,
        )

        object.__setattr__(built, "_cached_queries", self._cached_queries)
        return built

    def warehouse(
        self,
        warehouse_id: str | None = None,
        warehouse_name: str | None = None,
    ) -> SQLWarehouse:
        """
        Resolve the warehouse used by this engine.

        Behavior:
        - If no default warehouse is cached, resolve one and cache it
        - If a warehouse override is provided, resolve that warehouse directly
        - If the cached default is the all-purpose serverless placeholder,
          periodically refresh it

        Args:
            warehouse_id:
                Explicit warehouse ID to resolve.
            warehouse_name:
                Explicit warehouse name to resolve.

        Returns:
            A resolved `SQLWarehouse` instance.
        """
        if self.default_warehouse is None:
            object.__setattr__(self, "_last_default_wh_check", time.time())
            object.__setattr__(
                self,
                "default_warehouse",
                self.warehouses.find_warehouse(
                    warehouse_id=warehouse_id,
                    warehouse_name=warehouse_name,
                    find_default=True,
                    raise_error=True,
                ),
            )
            return self.default_warehouse

        if warehouse_id or warehouse_name:
            return self.warehouses.find_warehouse(
                warehouse_id=warehouse_id,
                warehouse_name=warehouse_name,
            )

        if self.default_warehouse.warehouse_name == DEFAULT_ALL_PURPOSE_SERVERLESS_NAME:
            now_s = time.time()
            if (now_s - self._last_default_wh_check) > 30:
                object.__setattr__(
                    self,
                    "default_warehouse",
                    self.warehouses.find_default(),
                )
                object.__setattr__(self, "_last_default_wh_check", now_s)

        return self.default_warehouse

    def _stage_external_tables(
        self,
        external_tables: Mapping[str, Any] | None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
    ) -> tuple[Dict[str, str], list[StagingPath]]:
        """Resolve ``external_tables`` into SQL substitutions and owned staging paths.

        For each entry:

        - If the value is already a :class:`StagingPath`, its path is
          substituted into the SQL.  When the staging path reports
          ``owned=True`` it is also returned for cleanup handoff (caller
          explicitly asked the engine to manage it); ``owned=False`` paths
          are used read-only and never cleaned up.
        - Otherwise the value is treated as tabular data, written to a fresh
          :class:`StagingPath` as Parquet, and the engine takes ownership of
          the new staging path so it can be cleaned up lazily when the
          resulting :class:`StatementResult` is done.

        Returns:
            A tuple ``(substitutions, owned_paths)`` where ``substitutions``
            maps each alias to the ``parquet.`<path>``` SQL source clause and
            ``owned_paths`` lists every staging path the engine is responsible
            for cleaning up.
        """
        if not external_tables:
            return {}, []

        effective_catalog = catalog_name or self.catalog_name
        effective_schema = schema_name or self.schema_name

        substitutions: Dict[str, str] = {}
        owned: list[StagingPath] = []

        for alias, value in external_tables.items():
            if isinstance(value, StagingPath):
                substitutions[alias] = _staging_parquet_ref(value)
                if value.owned:
                    owned.append(value)
                continue

            if not effective_catalog or not effective_schema:
                raise ValueError(
                    "external_tables requires catalog_name and schema_name "
                    "to be resolvable on the engine or provided explicitly; "
                    f"cannot stage alias {alias!r}."
                )

            staging = StagingPath.for_table(
                client=self.client,
                catalog_name=effective_catalog,
                schema_name=effective_schema,
                table_name=alias,
                max_lifetime=3600,
            )
            staging.register_shutdown_cleanup()
            staging.write_table(value)

            owned.append(staging)
            substitutions[alias] = _staging_parquet_ref(staging)

        return substitutions, owned

    def execute_many(
        self,
        statements: "Iterable[str | PreparedStatement | StatementResult] | Mapping[str, str | PreparedStatement | StatementResult]",
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
        cache_for: WaitingConfigArg = None,
        spark_session: Optional["SparkSession"] = None,
        parallel: int | bool = False,
        external_tables: Mapping[str, "StagingPath | Any"] | None = None,
    ) -> StatementBatch:
        """
        Execute multiple SQL statements.

        Behavior
        --------
        - Statements are normalized with ``strip()``
        - Empty statements are ignored
        - Execution order is preserved in the returned batch
        - When ``parallel`` is ``False`` (default):
          - all statements except the last are executed with ``wait=True``
          - the final statement uses the caller-provided ``wait`` value
        - When ``parallel`` is truthy (``True`` or an ``int >= 2``):
          - statements are submitted to a bounded thread pool so at most
            ``parallel`` statements are executing on the warehouse at once
            (``True`` maps to the default pool size of 4)
          - the batch returns once every statement reaches a terminal state
          - if any statement fails, the remaining pooled statements are
            cancelled and their staging resources are cleaned up

        Args:
            statements:
                SQL statements to execute.

                Accepts either:
                - an iterable of SQL strings or :class:`PreparedStatement` objects,
                  keyed as ``"0"``, ``"1"``, ...
                - a mapping of ``{name: statement}``, preserving mapping order,
                  where each value is a SQL string or :class:`PreparedStatement`
            row_limit:
                Optional row limit forwarded to each statement execution.
            catalog_name:
                Catalog override for warehouse API execution context.
            schema_name:
                Schema override for warehouse API execution context.
            wait:
                Waiting configuration.
            raise_error:
                Whether execution errors should be raised.
            engine:
                Explicit engine override: ``"spark"`` or ``"api"``.
            warehouse_id:
                Warehouse ID override for API execution.
            warehouse_name:
                Warehouse name override for API execution.
            byte_limit:
                Optional response byte limit for API execution.
            cache_for:
                Optional TTL for statement result caching.
            spark_session:
                Explicit SparkSession override.
            parallel:
                Pool sizing for parallel execution.  ``False`` (default) runs
                sequentially.  ``True`` uses a bounded pool of 4.  An
                ``int >= 2`` caps concurrency to that many statements running
                on the warehouse at once.  Inner polling is managed by the
                batch — callers do not need to handle futures or join threads.
            external_tables:
                Optional mapping of alias → :class:`StagingPath` or tabular
                data. Aliases referenced in statements as ``{alias}`` are
                replaced with the corresponding ``parquet.`<path>``` source
                clause. Tabular values are materialized to a fresh staging
                path via Parquet. Engine-owned staging paths are attached to
                every result in the batch and cleaned up lazily once those
                results reach a terminal state.  Merged on top of any
                per-statement external tables already carried by
                :class:`PreparedStatement` inputs.

        Returns:
            A :class:`StatementBatch` containing results in input order.

        Raises:
            ValueError:
                If no non-empty SQL statements were provided.
        """
        items: OrderedDict[str, PreparedStatement] = OrderedDict()

        def _add(key: str, raw: "str | PreparedStatement | StatementResult") -> None:
            # Accept plain SQL, a PreparedStatement config, or a pre-built StatementResult.
            if isinstance(raw, StatementResult):
                cfg = raw.statement
            else:
                cfg = PreparedStatement.prepare(raw)
            stripped = cfg.text.strip()
            if not stripped:
                return
            items[key] = cfg.with_text(stripped)

        if isinstance(statements, Mapping):
            for key, raw in statements.items():
                _add(str(key), raw)
        else:
            for i, raw in enumerate(statements):
                _add(str(i), raw)

        if not items:
            raise ValueError("No non-empty SQL statements were provided.")

        substitutions, owned_staging = self._stage_external_tables(
            external_tables,
            catalog_name=catalog_name,
            schema_name=schema_name,
        )

        if substitutions:
            items = OrderedDict(
                (
                    key,
                    cfg.with_text(
                        _apply_external_table_aliases(cfg.text, substitutions),
                    ),
                )
                for key, cfg in items.items()
            )

        batch = StatementBatch.from_statements(items, engine=self)

        # Build a runner only when the caller overrode one of the engine's
        # per-call defaults.  Otherwise the batch's default runner
        # (``engine.execute(result, wait=False)``) already does the right
        # thing and we avoid the closure.
        overrides = {
            name: value
            for name, value in (
                ("row_limit", row_limit),
                ("catalog_name", catalog_name),
                ("schema_name", schema_name),
                ("engine", engine),
                ("warehouse_id", warehouse_id),
                ("warehouse_name", warehouse_name),
                ("byte_limit", byte_limit),
                ("cache_for", cache_for),
                ("spark_session", spark_session),
            )
            if value is not None
        }
        runner = None
        if overrides:
            def runner(result: StatementResult) -> StatementResult:
                return self.execute(
                    result, wait=False, raise_error=raise_error, **overrides,
                )

        # ``batch.start`` owns the full lifecycle — it submits, drives
        # polling, and cancels everything on failure before re-raising.
        batch.start(
            parallel=parallel,
            wait=wait,
            raise_error=raise_error,
            runner=runner,
        )

        if owned_staging:
            for result in batch.results.values():
                result.attach_external_tables(owned_staging)
                result._maybe_cleanup_external_tables()

        return batch

    def statement_result(
        self,
        statement: "str | PreparedStatement | StatementResult" = "",
        *,
        parameters: Mapping[str, Any] | None = None,
        external_tables: Mapping[str, "StagingPath | Any"] | None = None,
    ) -> StatementResult:
        """Build an unstarted :class:`StatementResult` bound to this engine.

        Accepts a string, a :class:`PreparedStatement` config, or an
        existing :class:`StatementResult` (in which case extra ``parameters``
        / ``external_tables`` are merged onto its config).  Used as the
        default factory for :class:`StatementBatch`.
        """
        return StatementResult.prepare(
            statement,
            parameters=parameters,
            external_tables=external_tables,
        )

    def prepare(
        self,
        statement: "str | PreparedStatement | StatementResult",
        *,
        parameters: Mapping[str, Any] | None = None,
        external_tables: Mapping[str, "StagingPath | Any"] | None = None,
    ) -> PreparedStatement:
        """Build a :class:`PreparedStatement` config from a string or existing statement.

        Extra ``parameters`` and ``external_tables`` are merged on top of
        any values carried by ``statement``.  The returned :class:`PreparedStatement`
        config can be passed to :meth:`execute` later.
        """
        base = statement.statement if isinstance(statement, StatementResult) else statement
        return PreparedStatement.prepare(
            base,
            parameters=parameters,
            external_tables=external_tables,
        )

    def execute(
        self,
        statement: "str | PreparedStatement | StatementResult",
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
        cache_for: WaitingConfigArg = None,
        spark_session: Optional["SparkSession"] = None,
        external_tables: Mapping[str, "StagingPath | Any"] | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> StatementResult:
        """
        Execute a SQL statement through Spark or the Databricks SQL API.

        Engine selection
        ----------------
        - `engine="spark"` forces Spark SQL
        - `engine="api"` forces warehouse API execution
        - `engine=None` auto-selects Spark when a SparkSession is available,
          otherwise falls back to the warehouse API

        Cache behavior
        --------------
        When `cache_for` is provided, results are cached by the normalized SQL
        statement text for the specified TTL.

        Args:
            statement:
                SQL text to execute, or a :class:`PreparedStatement` carrying both
                the text and its bound arguments.  Strings are coerced via
                :meth:`PreparedStatement.prepare` together with any ``parameters``
                or ``external_tables`` passed to this call.
            row_limit:
                Optional row limit. Applied through `limit()` on Spark results
                or forwarded to the SQL API.
            catalog_name:
                Catalog override for warehouse API execution context.
            schema_name:
                Schema override for warehouse API execution context.
            wait:
                Waiting configuration for API execution.
            raise_error:
                Whether execution errors should be raised.
            engine:
                Explicit engine override: `"spark"` or `"api"`.
            warehouse_id:
                Warehouse ID override for API execution.
            warehouse_name:
                Warehouse name override for API execution.
            byte_limit:
                Optional response byte limit for API execution.
            cache_for:
                Optional TTL for statement result caching.
            spark_session:
                Explicit SparkSession override.
            external_tables:
                Optional mapping of alias → :class:`StagingPath` or tabular
                data. Aliases referenced in ``statement`` as ``{alias}`` are
                replaced with the corresponding ``parquet.`<path>``` source
                clause. Tabular values are materialized to a fresh staging
                path via Parquet. Engine-owned staging paths are attached to
                the returned ``PreparedStatement`` and cleaned up lazily once
                it reaches a terminal state.  Merged on top of any external
                tables already carried by ``statement``.
            parameters:
                Optional named parameters bound to ``:name`` placeholders in
                the query text.  Merged on top of any parameters already
                carried by ``statement``.

        Returns:
            A `PreparedStatement` wrapping either a Spark result or a warehouse API
            statement execution result.

        Raises:
            ValueError:
                If Spark execution is requested and no SparkSession can be
                resolved.
        """
        prepared = StatementResult.prepare(
            statement,
            parameters=parameters,
            external_tables=external_tables,
        )

        catalog_name = catalog_name or self.catalog_name
        schema_name = schema_name or self.schema_name

        substitutions, owned_staging = self._stage_external_tables(
            prepared.statement.external_tables,
            catalog_name=catalog_name,
            schema_name=schema_name,
        )
        if substitutions:
            prepared = prepared.with_text(
                _apply_external_table_aliases(prepared.statement.text, substitutions),
            )

        if not engine:
            spark_session = (
                PyEnv.spark_session(
                    create=False,
                    import_error=False,
                    install_spark=False,
                )
                if spark_session is None
                else spark_session
            )

            if spark_session is not None:
                engine = "spark"
            else:
                engine = "api"

        if spark_session is not None:
            engine = "spark"

        prepared = prepared.with_text(prepared.statement.text.strip())

        if cache_for is not None:
            cache_for = WaitingConfig.check_arg(cache_for)
            existing = self._cached_queries.get(prepared.statement.text)
            if existing is not None:
                return existing

        if engine == "spark":
            spark_session = (
                PyEnv.spark_session(
                    create=True,
                    install_spark=False,
                    import_error=True,
                )
                if spark_session is None
                else spark_session
            )

            df = spark_session.sql(prepared.statement.text)
            if row_limit:
                df = df.limit(row_limit)

            object.__setattr__(prepared, "service", self.client.statements)
            object.__setattr__(prepared, "warehouse_id", "SparkSQL")
            object.__setattr__(prepared, "statement_id", "SparkSQL")
            object.__setattr__(prepared, "disposition", Disposition.EXTERNAL_LINKS)
            result = prepared
            if owned_staging:
                # Spark is lazy; materialize to Arrow before the staging
                # parquet files get cleaned up, otherwise the DataFrame
                # would read from files that no longer exist.
                result.persist(mode="arrow", data=df.toArrow())
            else:
                result.persist(data=df)
        else:
            wh = self.warehouse(
                warehouse_id=warehouse_id,
                warehouse_name=warehouse_name,
            )

            result = wh.execute(
                statement=prepared,
                warehouse_id=warehouse_id,
                warehouse_name=warehouse_name,
                byte_limit=byte_limit,
                catalog_name=catalog_name,
                schema_name=schema_name,
                wait=wait,
                raise_error=raise_error,
                row_limit=row_limit,
            )

        if owned_staging:
            result.attach_external_tables(owned_staging)
            result._maybe_cleanup_external_tables()

        if cache_for is not None:
            self._cached_queries.set(
                key=prepared.statement.text,
                value=result,
                ttl=cache_for.timeout_total_seconds,
            )

        return result

    def table(
        self,
        location: str | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
    ) -> Table:
        """
        Resolve a table handle.

        Args:
            location:
                Fully qualified table name.
            catalog_name:
                Catalog override when `location` is not fully specified.
            schema_name:
                Schema override when `location` is not fully specified.
            table_name:
                Table name override when `location` is not provided.

        Returns:
            A `Table` handle.
        """
        return self.tables.table(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
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
        mode: SaveMode | str | None = None,
        schema_mode: SaveMode | str | None = None,
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
        primary_keys: "list[str] | PrimaryKeySpec | None" = None,
        foreign_keys: "list[ForeignKeySpec] | dict[str, str] | None" = None,
        narrow_merge: bool = True,
        concurrent_append_retries: int = 3,
    ) -> None:
        """
        Insert data into a Delta table using the most appropriate backend.

        Routing behavior
        ----------------
        - If ``data`` is a :class:`PreparedStatement`, :class:`StatementResult`,
          or SQL-like string, dispatch to :meth:`sql_insert_into` which
          smart-routes between a cached :class:`StatementResult`, the Spark
          SQL path, and the warehouse SQL path.
        - Else if a SparkSession is available, use :meth:`spark_insert_into`.
        - Otherwise, use :meth:`arrow_insert_into` (warehouse SQL with
          staged Parquet).

        Args:
            data:
                Input data. Supported inputs include Arrow objects, Spark
                DataFrames, pandas/polars DataFrames, dict/list tabular values,
                and other project-supported convertible types.
            mode:
                Save mode controlling append / overwrite / merge semantics.
            schema_mode:
                Schema mode to merge with current target schema
            location:
                Fully qualified destination table.
            catalog_name:
                Catalog override.
            schema_name:
                Schema override.
            table_name:
                Table name override.
            cast_options:
                Casting rules used to align input data to the destination schema.
            overwrite_schema:
                Spark writer option. When True, writes with
                `overwriteSchema=true`.
            match_by:
                Merge key columns enabling key-based insert or upsert behavior.
            update_cols:
                Columns to update when a merge key match is found (``mode=auto``
                only).  ``None`` updates all non-key columns (default).
                An empty list disables the UPDATE clause entirely (insert-only
                for matched rows).
            wait:
                Waiting configuration for the warehouse SQL path.
            raise_error:
                Whether write errors should be raised on the warehouse SQL path.
            zorder_by:
                Optional ZORDER columns for post-write optimization.
            optimize_after_merge:
                Whether to run optimize after merge-related writes.
            vacuum_hours:
                Optional retention window for VACUUM.
            spark_session:
                Explicit SparkSession override.
            spark_options:
                Additional Spark DataFrameWriter options.
            table:
                Optional pre-resolved table handle.
            primary_keys:
                Column name(s) to set as primary key when the table is created
                by this call.  Composite keys: ``["trade_date", "instrument_id"]``.
                ``None`` reads from field metadata automatically.
            foreign_keys:
                FK constraints to apply when the table is created.
                Accepts a ``{col: "cat.sch.tbl.col"}`` dict or a list of
                :class:`~yggdrasil.databricks.sql.types.ForeignKeySpec`.

        Returns:
            None.
        """
        if isinstance(data, (PreparedStatement, StatementResult)) or PreparedStatement.looks_like_query(data):
            return self.sql_insert_into(
                data,
                mode=mode,
                location=location,
                catalog_name=catalog_name,
                schema_name=schema_name,
                table_name=table_name,
                match_by=match_by,
                update_cols=update_cols,
                wait=wait,
                raise_error=raise_error,
                zorder_by=zorder_by,
                optimize_after_merge=optimize_after_merge,
                vacuum_hours=vacuum_hours,
                table=table,
                spark_session=spark_session,
                narrow_merge=narrow_merge,
                concurrent_append_retries=concurrent_append_retries,
            )

        if spark_session is None:
            if hasattr(data, "sparkSession"):
                spark_session = data.sparkSession
            else:
                spark_session = PyEnv.spark_session(
                    create=False,
                    import_error=False,
                    install_spark=False,
                )

        if spark_session is not None:
            return self.spark_insert_into(
                data=data,
                location=location,
                catalog_name=catalog_name,
                schema_name=schema_name,
                table_name=table_name,
                mode=mode,
                schema_mode=schema_mode,
                cast_options=cast_options,
                overwrite_schema=overwrite_schema,
                match_by=match_by,
                update_cols=update_cols,
                wait=wait,
                zorder_by=zorder_by,
                optimize_after_merge=optimize_after_merge,
                vacuum_hours=vacuum_hours,
                spark_options=spark_options,
                table=table,
                primary_keys=primary_keys,
                foreign_keys=foreign_keys,
                narrow_merge=narrow_merge,
                concurrent_append_retries=concurrent_append_retries,
            )

        return self.arrow_insert_into(
            data=data,
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            mode=mode,
            schema_mode=schema_mode,
            cast_options=cast_options,
            overwrite_schema=overwrite_schema,
            match_by=match_by,
            update_cols=update_cols,
            wait=wait,
            raise_error=raise_error,
            zorder_by=zorder_by,
            optimize_after_merge=optimize_after_merge,
            vacuum_hours=vacuum_hours,
            table=table,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            narrow_merge=narrow_merge,
            concurrent_append_retries=concurrent_append_retries,
        )

    def arrow_insert_into(
        self,
        data,
        location: str | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        mode: SaveMode | str | None = None,
        schema_mode: SaveMode | str | None = None,
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
        temp_volume_path=None,
        primary_keys: "list[str] | PrimaryKeySpec | None" = None,
        foreign_keys: "list[ForeignKeySpec] | dict[str, str] | None" = None,
        narrow_merge: bool = True,
        concurrent_append_retries: int = 3,
    ) -> None:
        """
        Insert data through the warehouse SQL path.

        Implementation
        --------------
        - Ensure the destination table exists
        - Convert input data to Parquet
        - Stage the Parquet file to a temporary Databricks volume
        - Execute INSERT INTO or MERGE INTO against the target table
        - Optionally run OPTIMIZE, ZORDER, and VACUUM

        Behavior
        --------
        - `overwrite` drops the target table before inserting
        - `truncate` (no ``match_by``): ``TRUNCATE TABLE`` then ``INSERT INTO``
        - `truncate` (with ``match_by``): ``DELETE`` rows whose keys appear in
          the input, then ``INSERT INTO`` — keeps the schema intact
        - `match_by` without overwrite/truncate enables MERGE semantics:
          ``append`` → insert-only; ``auto`` → upsert
        - rows with NULL in match columns are expected to follow the same merge
          behavior as the Spark path

        Args:
            data:
                Arrow or Arrow-convertible tabular data.
            location:
                Fully qualified destination table.
            catalog_name:
                Catalog override.
            schema_name:
                Schema override.
            table_name:
                Table name override.
            mode:
                Save mode controlling append / overwrite / merge semantics.
            schema_mode:
                Schema mode to merge with current target schema
            cast_options:
                Casting rules used to align staged data to the destination
                schema.
            overwrite_schema:
                Reserved for API parity with the Spark path.
            match_by:
                Merge key columns.
            update_cols:
                Columns to update when a merge key match is found (``mode=auto``
                only).  ``None`` updates all non-key columns (default).
                An empty list disables the UPDATE clause entirely.
            wait:
                Waiting configuration for statement execution.
            raise_error:
                Whether statement execution errors should be raised.
            zorder_by:
                Columns used for `OPTIMIZE ... ZORDER BY (...)`.
            optimize_after_merge:
                Whether to run `OPTIMIZE` after merge operations.
            vacuum_hours:
                Optional retention window for `VACUUM`.
            table:
                Optional pre-resolved table handle.
            temp_volume_path:
                Optional explicit staging volume path.

        Returns:
            None.
        """
        if isinstance(data, (PreparedStatement, StatementResult)) or PreparedStatement.looks_like_query(data):
            return self.sql_insert_into(
                data,
                mode=mode,
                location=location,
                catalog_name=catalog_name,
                schema_name=schema_name,
                table_name=table_name,
                match_by=match_by,
                update_cols=update_cols,
                wait=wait,
                raise_error=raise_error,
                zorder_by=zorder_by,
                optimize_after_merge=optimize_after_merge,
                vacuum_hours=vacuum_hours,
                table=table,
            )

        mode = SaveMode.parse(mode, default=SaveMode.AUTO)
        catalog_name = catalog_name or self.catalog_name
        schema_name = schema_name or self.schema_name

        table = Table.parse(
            obj=location if table is None else table,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            service=self.tables,
        )

        if mode == SaveMode.OVERWRITE:
            table.delete(wait=True, raise_error=False)

        if mode == SaveMode.OVERWRITE:
            table.delete(wait=True, raise_error=False)

        table = table.create(
            data,
            mode=schema_mode,
            primary_keys=primary_keys, foreign_keys=foreign_keys
        )
        location = table.full_name(safe=True)
        cast_options = CastOptions.check(options=cast_options).check_target(table.data_field)
        existing_schema = table.data_schema

        logger.debug("Inserting %s into %s", type(data), location)

        if temp_volume_path is None:
            staging = StagingPath.for_table(
                client=self.client,
                catalog_name=table.catalog_name,
                schema_name=table.schema_name,
                table_name=table.table_name,
                max_lifetime=3600,
            )
            staging.register_shutdown_cleanup()
        else:
            staging = StagingPath.from_volume(
                temp_volume_path, client=self.client, owned=False,
            )
        staging.write_table(data, cast_options=cast_options)

        columns = list(existing_schema.field_names())
        cols_quoted = ", ".join(quote_ident(c) for c in columns)

        statements: list[str] = []

        # SQL uses the ``{src}`` alias — ``execute_many`` substitutes it
        # with ``parquet.`<staging path>``` via the external_tables kwarg,
        # and hands the owned staging off to result-lifecycle cleanup.
        source_sql = f"SELECT {cols_quoted} FROM {{src}}"

        # Narrow the MERGE/DELETE scope using partition + liquid-cluster
        # columns from the target — keeps Delta OCC from taking a whole-table
        # read and throwing DELTA_CONCURRENT_APPEND.WHOLE_TABLE_READ.
        narrow_cols = _narrow_target_columns(table, match_by=match_by) if narrow_merge else []
        narrow_cols = [c for c in narrow_cols if c in columns]
        scope_predicates = _narrowing_predicates_via_subquery(
            narrow_cols, target_alias="T", source_expr="{src}",
        ) if narrow_cols else []
        if scope_predicates:
            logger.debug(
                "Narrowing MERGE scope on %s by %s", location, narrow_cols,
            )

        if mode == SaveMode.TRUNCATE:
            insert_sql = (
                f"INSERT INTO {location} ({cols_quoted})\n{source_sql}"
            )

            if match_by:
                # Delete every existing row whose key appears in the
                # incoming batch, then insert all rows from that batch.
                key_cols = ", ".join(quote_ident(k) for k in match_by)
                on_condition = _build_match_condition(
                    match_by,
                    left_alias="T",
                    right_alias="S",
                    null_safe=True,
                    extra_predicates=scope_predicates,
                )
                delete_sql = (
                    f"DELETE FROM {location} AS T\n"
                    f"USING (\n"
                    f"SELECT DISTINCT {key_cols}\nFROM {{src}}\n"
                    f") AS S\n"
                    f"ON {on_condition}"
                )
                statements.extend([delete_sql, insert_sql])
            else:
                # Wipe the table in-place (schema kept), then insert all rows.
                statements.extend([
                    f"TRUNCATE TABLE {location}",
                    insert_sql,
                ])

        elif match_by and mode != SaveMode.OVERWRITE:
            on_condition = _build_match_condition(
                match_by,
                left_alias="T",
                right_alias="S",
                null_safe=True,
                extra_predicates=scope_predicates,
            )
            insert_clause = (
                f"WHEN NOT MATCHED THEN INSERT ({cols_quoted}) "
                f"VALUES ({', '.join(f'S.{quote_ident(c)}' for c in columns)})"
            )

            if mode == SaveMode.APPEND:
                merge_sql = (
                    f"MERGE INTO {location} AS T\n"
                    f"USING (\n{source_sql}\n) AS S\n"
                    f"ON {on_condition}\n"
                    f"{insert_clause}"
                )
                statements.append(merge_sql)
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
                    update_clause = f"WHEN MATCHED THEN UPDATE SET {update_set}"

                merge_sql = (
                    f"MERGE INTO {location} AS T\n"
                    f"USING (\n{source_sql}\n) AS S\n"
                    f"ON {on_condition}\n"
                    f"{update_clause}\n"
                    f"{insert_clause}"
                )
                statements.append(merge_sql)
        else:
            statements.append(
                f"INSERT INTO {location} ({cols_quoted})\n{source_sql}"
            )

        # Chain DML + post-write maintenance in a single sequential batch so
        # the wait-all-but-last logic covers every step.  Only the final
        # statement respects the caller's ``wait``.
        if zorder_by:
            zorder_cols = ", ".join(quote_ident(c) for c in zorder_by)
            statements.append(f"OPTIMIZE {location} ZORDER BY ({zorder_cols})")
        if optimize_after_merge and match_by:
            statements.append(f"OPTIMIZE {location}")
        if vacuum_hours is not None:
            statements.append(f"VACUUM {location} RETAIN {int(vacuum_hours)} HOURS")

        # ``external_tables`` carries the staging into ``execute_many`` —
        # if ``staging.owned`` is True the engine will attach it to every
        # resulting :class:`StatementResult` so cleanup fires lazily when
        # each one reaches a terminal state.  ``owned=False`` staging paths
        # supplied by the caller are substituted but never deleted.
        if statements:
            _retry_concurrent_append(
                lambda: self.execute_many(
                    statements,
                    wait=wait,
                    raise_error=raise_error,
                    external_tables={"src": staging},
                ),
                attempts=max(1, concurrent_append_retries + 1),
                op=f"arrow merge into {location}",
            )

        logger.info("Arrow inserted into %s", location)
        return None

    def sql_insert_into(
        self,
        statement: "PreparedStatement | StatementResult | str",
        *,
        mode: SaveMode | str | None = None,
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
        narrow_merge: bool = True,
        concurrent_append_retries: int = 3,
    ) -> None:
        """Insert into a Delta table from a SQL source.

        Smart dispatch
        --------------
        1. If ``statement`` is a :class:`StatementResult` that has already
           been materialized (``persisted=True``), the cached Spark
           DataFrame or Arrow table is reused via :meth:`insert_into` — the
           query is not re-executed.
        2. Otherwise, if a SparkSession is available, the source SQL is
           executed in Spark and the resulting DataFrame is handed to
           :meth:`spark_insert_into`, which uses native Delta
           ``MERGE`` / append / overwrite APIs.
        3. Otherwise, the query runs on the warehouse and its output is
           written into the target table via ``INSERT INTO ... SELECT``,
           ``MERGE INTO ... USING (<query>)``, or ``DELETE`` + ``INSERT``
           depending on ``mode`` and ``match_by``.

        The target table must already exist; its column list drives the
        target columns and the projection over the source subquery.
        """
        # ---- Fast path 1: cached StatementResult ----
        if isinstance(statement, StatementResult) and statement.persisted:
            cached = (
                statement._spark_df
                if statement._spark_df is not None
                else statement.to_arrow_table()
            )
            return self.insert_into(
                data=cached,
                mode=mode,
                location=location,
                catalog_name=catalog_name,
                schema_name=schema_name,
                table_name=table_name,
                match_by=match_by,
                update_cols=update_cols,
                wait=wait,
                raise_error=raise_error,
                zorder_by=zorder_by,
                optimize_after_merge=optimize_after_merge,
                vacuum_hours=vacuum_hours,
                spark_session=spark_session,
                table=table,
            )

        # ---- Fast path 2: run in Spark for native Delta writes ----
        if spark_session is None:
            spark_session = PyEnv.spark_session(
                create=False,
                import_error=False,
                install_spark=False,
            )
        if spark_session is not None:
            cfg = (
                statement.statement
                if isinstance(statement, StatementResult)
                else PreparedStatement.prepare(statement)
            )
            df = spark_session.sql(cfg.text)
            return self.spark_insert_into(
                data=df,
                mode=mode,
                location=location,
                catalog_name=catalog_name,
                schema_name=schema_name,
                table_name=table_name,
                match_by=match_by,
                update_cols=update_cols,
                wait=wait,
                zorder_by=zorder_by,
                optimize_after_merge=optimize_after_merge,
                vacuum_hours=vacuum_hours,
                table=table,
                narrow_merge=narrow_merge,
                concurrent_append_retries=concurrent_append_retries,
            )

        # ---- Fallback: warehouse-side SQL merge ----
        base = statement.statement if isinstance(statement, StatementResult) else statement
        prepared = PreparedStatement.prepare(base)
        mode = SaveMode.parse(mode, default=SaveMode.AUTO)
        catalog_name = catalog_name or self.catalog_name
        schema_name = schema_name or self.schema_name

        table = Table.parse(
            obj=location if table is None else table,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            service=self.tables,
        )

        if mode == SaveMode.OVERWRITE:
            table.delete(wait=True, raise_error=False)

        if not table.exists:
            raise ValueError(
                "sql_insert_into requires the target table to exist; "
                f"{table.full_name()!r} was not found."
            )

        location = table.full_name(safe=True)
        fields = list(table.data_schema.fields)
        columns = [f.name for f in fields]
        cols_quoted = ", ".join(quote_ident(c) for c in columns)
        cast_projection = ", ".join(
            (
                f"CAST(raw_src.{quote_ident(f.name)} AS "
                f"{f.to_databricks_ddl(put_name=False, put_not_null=False, put_comment=False)})"
                f" AS {quote_ident(f.name)}"
            )
            for f in fields
        )
        src_cols = ", ".join(f"src.{quote_ident(c)}" for c in columns)
        source_sql = (
            f"(\nSELECT {cast_projection} FROM (\n{prepared.text}\n) AS raw_src\n)"
        )

        logger.debug("Inserting query into %s", location)

        statements: list["PreparedStatement | str"] = []  # each entry is PreparedStatement config or raw SQL

        # Narrow the merge scope by partition + cluster columns against the
        # source subquery — keeps Delta from a whole-table read.
        narrow_cols = _narrow_target_columns(table, match_by=match_by) if narrow_merge else []
        narrow_cols = [c for c in narrow_cols if c in columns]
        scope_predicates = _narrowing_predicates_via_subquery(
            narrow_cols, target_alias="T", source_expr=f"{source_sql} AS src",
        ) if narrow_cols else []
        if scope_predicates:
            logger.debug(
                "Narrowing SQL merge scope on %s by %s", location, narrow_cols,
            )

        if mode == SaveMode.TRUNCATE:
            insert_sql = (
                f"INSERT INTO {location} ({cols_quoted})\n"
                f"SELECT {src_cols} FROM {source_sql} AS src"
            )
            if match_by:
                key_cols = ", ".join(quote_ident(k) for k in match_by)
                on_condition = _build_match_condition(
                    match_by,
                    left_alias="T",
                    right_alias="S",
                    null_safe=True,
                    extra_predicates=scope_predicates,
                )
                delete_sql = (
                    f"DELETE FROM {location} AS T\n"
                    f"USING (\nSELECT DISTINCT {key_cols} FROM {source_sql} AS src\n) AS S\n"
                    f"ON {on_condition}"
                )
                statements.extend([
                    replace(prepared, text=delete_sql),
                    replace(prepared, text=insert_sql),
                ])
            else:
                statements.append(f"TRUNCATE TABLE {location}")
                statements.append(replace(prepared, text=insert_sql))

        elif match_by and mode != SaveMode.OVERWRITE:
            on_condition = _build_match_condition(
                match_by,
                left_alias="T",
                right_alias="S",
                null_safe=True,
                extra_predicates=scope_predicates,
            )
            insert_clause = (
                f"WHEN NOT MATCHED THEN INSERT ({cols_quoted}) "
                f"VALUES ({', '.join(f'S.{quote_ident(c)}' for c in columns)})"
            )

            if mode == SaveMode.APPEND:
                merge_sql = (
                    f"MERGE INTO {location} AS T\n"
                    f"USING {source_sql} AS S\n"
                    f"ON {on_condition}\n"
                    f"{insert_clause}"
                )
                statements.append(replace(prepared, text=merge_sql))
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
                    update_clause = f"WHEN MATCHED THEN UPDATE SET {update_set}"

                merge_sql = (
                    f"MERGE INTO {location} AS T\n"
                    f"USING {source_sql} AS S\n"
                    f"ON {on_condition}\n"
                    f"{update_clause}\n"
                    f"{insert_clause}"
                )
                statements.append(replace(prepared, text=merge_sql))

        else:
            insert_sql = (
                f"INSERT INTO {location} ({cols_quoted})\n"
                f"SELECT {src_cols} FROM {source_sql} AS src"
            )
            statements.append(replace(prepared, text=insert_sql))

        if zorder_by:
            zorder_cols = ", ".join(quote_ident(c) for c in zorder_by)
            statements.append(f"OPTIMIZE {location} ZORDER BY ({zorder_cols})")
        if optimize_after_merge and match_by:
            statements.append(f"OPTIMIZE {location}")
        if vacuum_hours is not None:
            statements.append(f"VACUUM {location} RETAIN {int(vacuum_hours)} HOURS")

        if statements:
            _retry_concurrent_append(
                lambda: self.execute_many(
                    statements, wait=wait, raise_error=raise_error,
                ),
                attempts=max(1, concurrent_append_retries + 1),
                op=f"sql merge into {location}",
            )

        logger.info("SQL inserted into %s", location)
        return None

    def spark_insert_into(
        self,
        data: Any,
        *,
        mode: SaveMode | str | None = None,
        schema_mode: SaveMode | str | None = None,
        location: str | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        cast_options: Optional[CastOptions] = None,
        overwrite_schema: bool | None = None,
        match_by: Optional[list[str]] = None,
        update_cols: Optional[list[str]] = None,
        wait: WaitingConfigArg = True,
        zorder_by: Optional[list[str]] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,
        spark_options: Optional[Dict[str, Any]] = None,
        table: Optional[Table] = None,
        primary_keys: "list[str] | PrimaryKeySpec | None" = None,
        foreign_keys: "list[ForeignKeySpec] | dict[str, str] | None" = None,
        narrow_merge: bool = True,
        narrow_max_in_values: int = 500,
        concurrent_append_retries: int = 3,
    ) -> None:
        """
        Insert data into a Delta table using Spark.

        Implementation
        --------------
        - Convert input to a Spark DataFrame
        - Ensure the destination table exists
        - Write using append or Delta MERGE semantics
        - Optionally run optimize / ZORDER / vacuum maintenance

        Behavior
        --------
        - `overwrite` drops the target table before writing
        - `truncate` (no ``match_by``): ``DeltaTable.delete()`` empties the
          table in-place, then data is appended
        - `truncate` (with ``match_by``): Delta MERGE deletes rows whose keys
          appear in the input, then data is appended
        - `append` + `match_by` performs insert-only merge
        - `auto` + `match_by` performs upsert merge; ``update_cols`` controls
          which columns are updated (default: all non-key columns)
        - without `match_by`, data is appended directly
        - ``wait=False`` submits the entire write to a background thread and
          returns immediately; errors are logged but not re-raised

        Args:
            data:
                Input data convertible to a Spark DataFrame.
            mode:
                Save mode controlling append / overwrite / merge semantics.
            schema_mode:
                Schema mode to merge with current target schema
            location:
                Fully qualified destination table.
            catalog_name:
                Catalog override.
            schema_name:
                Schema override.
            table_name:
                Table name override.
            cast_options:
                Casting rules used to align input data to the destination
                schema.
            overwrite_schema:
                When True, passes `overwriteSchema=true` to the Spark writer.
            match_by:
                Merge key columns.
            update_cols:
                Columns to update when a merge key match is found (``mode=auto``
                only).  ``None`` updates all non-key columns (default).
                An empty list disables the UPDATE clause entirely.
            wait:
                When ``True`` (default) the call blocks until the write
                completes.  When ``False`` the write is submitted to a
                background thread and the method returns immediately.
            zorder_by:
                Columns used for Delta ZORDER optimization.
            optimize_after_merge:
                Whether to run optimize after merge-style writes.
            vacuum_hours:
                Optional retention window for Delta vacuum.
            spark_options:
                Additional Spark writer options.
            table:
                Optional pre-resolved table handle.

        Returns:
            None.
        """
        if isinstance(data, (PreparedStatement, StatementResult)) or PreparedStatement.looks_like_query(data):
            return self.sql_insert_into(
                data,
                mode=mode,
                location=location,
                catalog_name=catalog_name,
                schema_name=schema_name,
                table_name=table_name,
                match_by=match_by,
                update_cols=update_cols,
                wait=wait,
                zorder_by=zorder_by,
                optimize_after_merge=optimize_after_merge,
                vacuum_hours=vacuum_hours,
                table=table,
                narrow_merge=narrow_merge,
                concurrent_append_retries=concurrent_append_retries,
            )

        from yggdrasil.spark.cast import any_to_spark_dataframe
        import pyspark.sql.functions as F

        logger.info(
            "Spark insert into %s (mode=%s, match_by=%s, overwrite_schema=%s, wait=%s)",
            location,
            mode,
            match_by,
            overwrite_schema,
            wait,
        )

        mode = SaveMode.parse(mode, default=SaveMode.AUTO)
        catalog_name = catalog_name or self.catalog_name
        schema_name = schema_name or self.schema_name

        if table is None:
            table = self.table(
                location=location,
                catalog_name=catalog_name,
                schema_name=schema_name,
                table_name=table_name,
            )

        # TODO: Fix async databricks notebook
        wait = True if PyEnv.in_databricks() else wait

        # OVERWRITE: drop before the background thread so the slot is freed
        # synchronously and callers can re-create the table immediately.
        if mode == SaveMode.OVERWRITE:
            table.delete(wait=True, raise_error=False)

        table = table.ensure_created(
            data,
            schema_mode=schema_mode,
            primary_keys=primary_keys, foreign_keys=foreign_keys
        )
        cast_options = CastOptions.check(options=cast_options).check_target(table.data_field)
        data_df = any_to_spark_dataframe(data, cast_options)
        target = table.delta_spark()

        _spark_options = spark_options if spark_options else {}
        if overwrite_schema:
            _spark_options["overwriteSchema"] = "true"

        # Narrow the merge scope by the target's partition + cluster columns
        # so Delta OCC reads a small file set instead of the whole table.
        # Computed once before ``_run`` so retries don't re-scan the batch.
        narrow_cols: list[str] = []
        scope_predicates: list[str] = []
        if narrow_merge and match_by:
            narrow_cols = _narrow_target_columns(table, match_by=match_by)
            narrow_cols = [c for c in narrow_cols if c in data_df.columns]
            if narrow_cols:
                scope_predicates = _narrowing_predicates_from_spark(
                    data_df,
                    narrow_cols,
                    target_alias="t",
                    max_in_values=narrow_max_in_values,
                )
                if scope_predicates:
                    logger.debug(
                        "Narrowing Spark merge scope on %s by %s",
                        table.full_name(), narrow_cols,
                    )

        def _run() -> None:
            if mode == SaveMode.TRUNCATE:
                cond = _build_match_condition(
                    match_by,
                    left_alias="t",
                    right_alias="s",
                    null_safe=True,
                    extra_predicates=scope_predicates,
                ) if match_by else None

                if match_by:
                    logger.info(
                        "Spark truncate (match_by=%s): Delta delete matching keys", match_by
                    )
                    distinct_keys = data_df.select(list(match_by)).distinct()
                    _retry_concurrent_append(
                        lambda: (
                            target.alias("t")
                            .merge(distinct_keys.alias("s"), cond)
                            .whenMatchedDelete()
                            .execute()
                        ),
                        attempts=max(1, concurrent_append_retries + 1),
                        op=f"spark truncate merge into {table.full_name()}",
                    )
                else:
                    logger.info("Spark truncate: Delta delete all rows")
                    target.delete()

                logger.info("Spark write saveAsTable mode=append (after truncate)")
                (
                    data_df.write
                    .format("delta")
                    .mode("append")
                    .options(**_spark_options)
                    .saveAsTable(table.full_name())
                )

            elif match_by and mode != SaveMode.OVERWRITE:
                cond = _build_match_condition(
                    match_by,
                    left_alias="t",
                    right_alias="s",
                    null_safe=True,
                    extra_predicates=scope_predicates,
                )

                if mode == SaveMode.APPEND:
                    _retry_concurrent_append(
                        lambda: (
                            target.alias("t")
                            .merge(data_df.alias("s"), cond)
                            .whenNotMatchedInsertAll()
                            .execute()
                        ),
                        attempts=max(1, concurrent_append_retries + 1),
                        op=f"spark append merge into {table.full_name()}",
                    )
                else:
                    update_cols_effective = (
                        update_cols
                        if update_cols is not None
                        else [c for c in data_df.columns if c not in match_by]
                    )
                    set_expr = {
                        c: F.expr(f"s.{quote_ident(c)}")
                        for c in update_cols_effective
                    }

                    def _upsert() -> None:
                        builder = target.alias("t").merge(data_df.alias("s"), cond)
                        if set_expr:
                            builder = builder.whenMatchedUpdate(set=set_expr)
                        builder.whenNotMatchedInsertAll().execute()

                    _retry_concurrent_append(
                        _upsert,
                        attempts=max(1, concurrent_append_retries + 1),
                        op=f"spark upsert merge into {table.full_name()}",
                    )
            else:
                logger.info("Spark write saveAsTable mode=append")
                (
                    data_df.write
                    .format("delta")
                    .mode("append")
                    .options(**_spark_options)
                    .saveAsTable(table.full_name())
                )

            if optimize_after_merge and zorder_by:
                logger.info("Delta optimize + zorder (%s)", zorder_by)
                target.optimize().executeZOrderBy(*zorder_by)

            if vacuum_hours is not None:
                logger.info("Delta vacuum retain=%s hours", vacuum_hours)
                target.vacuum(vacuum_hours)

        if wait:
            _run()
        else:
            Job.make(_run).fire_and_forget()

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
        """
        Drop a table if it exists.

        Args:
            location:
                Fully qualified destination table.
            catalog_name:
                Catalog override when `location` is not fully specified.
            schema_name:
                Schema override when `location` is not fully specified.
            table_name:
                Table name override when `location` is not provided.
            wait:
                Waiting configuration for the drop operation.
            raise_error:
                Whether drop errors should be raised.

        Returns:
            None.
        """
        return self.table(
            location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
        ).delete(wait=wait, raise_error=raise_error)

    def create_table(
        self,
        definition: Union[pa.Field, pa.Schema, Any],
        *,
        full_name: str | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        primary_keys: "list[str] | PrimaryKeySpec | None" = None,
        foreign_keys: "list[ForeignKeySpec] | dict[str, str] | None" = None,
        **kwargs,
    ) -> Table:
        """
        Create a table if it does not already exist.

        Primary and foreign key constraints are applied after the table is
        created via ``ALTER TABLE``, in the order PK → FK.

        Constraint sources (in priority order):

        1. Explicit ``primary_keys`` / ``foreign_keys`` parameters.
        2. Field-level metadata tags ``t:primary_key`` / ``t:foreign_key``
           on the supplied *definition* schema.

        Args:
            definition:
                Table definition — an Arrow field, Arrow schema, or any
                project-supported schema-like object.
            full_name:
                Fully qualified destination table (``catalog.schema.table``).
            catalog_name:
                Catalog override.
            schema_name:
                Schema override.
            table_name:
                Table name override.
            primary_keys:
                Column name(s) to mark as primary key, or a full
                :class:`~yggdrasil.databricks.sql.types.PrimaryKeySpec`.
                Composite keys are expressed as a list of column names:
                ``primary_keys=["trade_date", "instrument_id"]``.
                When ``None``, fields carrying ``t:primary_key`` metadata are
                used automatically.
            foreign_keys:
                A list of
                :class:`~yggdrasil.databricks.sql.types.ForeignKeySpec`
                objects, or a ``{col_name: "cat.sch.tbl.col"}`` dict.
                When ``None``, fields carrying ``t:foreign_key`` metadata are
                used automatically.
            **kwargs:
                Additional arguments forwarded to :meth:`Table.create`.

        Returns:
            The created or existing :class:`Table` handle.
        """
        table = self.table(
            location=full_name,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
        )

        return table.create(
            definition=definition,
            if_not_exists=True,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            **kwargs,
        )
