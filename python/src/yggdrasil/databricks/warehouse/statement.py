"""Databricks SQL warehouse-backed statements.

Three concrete types layered on the abstractions in
``yggdrasil.data.statement``:

- :class:`WarehousePreparedStatement` — adds typed routing
  (``warehouse_id`` / ``warehouse_name``), wire format, server-side wait,
  result caps, parameter bindings, and external-table aliases.
- :class:`WarehouseStatementResult` — tracks a single submission against
  the Databricks Statement Execution API: ``statement_id``, response
  caching, polling, and external-link fetch for Arrow streams.
- :class:`WarehouseStatementBatch` — re-uses the base batch contract; per-
  statement external-table aliases are resolved at coerce time, batch-wide
  scratch is cleaned up at teardown.

A few invariants the cleanup pass enforces:

- Each statement carries its own ``external_volume_paths``.  The batch
  doesn't maintain a parallel registry — the alias-substitution rewriter
  reads the per-statement field directly.  Single source of truth.
- ``_coerce`` returns the prepared statement (the previous version
  silently dropped it on the floor).
- Alias substitution doesn't mutate the input statement — it returns a
  rewritten copy, so re-submitting the same batch is safe.
"""

from __future__ import annotations

import copy as copy_mod
import logging
import re
from typing import (
    Any,
    ClassVar,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    TYPE_CHECKING, Literal,
)

import pyarrow as pa
import pyarrow.ipc as pipc
from databricks.sdk.service.sql import (
    Disposition,
    ExecuteStatementRequestOnWaitTimeout,
    ExternalLink,
    Format,
    StatementParameterListItem,
    StatementResponse,
    StatementState,
    StatementStatus,
)

from yggdrasil.concurrent.threading import Job, JobPoolExecutor
from yggdrasil.data import Schema
from yggdrasil.data.enums import MimeType, MimeTypes, Mode
from yggdrasil.data.enums.media_type import MediaTypes
from yggdrasil.data.enums.state import State
from yggdrasil.data.options import CastOptions
from yggdrasil.data.statement import (
    ExternalStatementData,
    PreparedStatement,
    StatementResult,
    StatementBatch,
)
from yggdrasil.databricks.sql.exceptions import SQLError
from yggdrasil.dataclasses import WaitingConfig, WaitingConfigArg
from yggdrasil.io.tabular import Tabular
from ..fs import VolumePath, DatabricksPath
from ..sql.types import parse_databricks_field

if TYPE_CHECKING:
    from yggdrasil.databricks.client import DatabricksClient
    from yggdrasil.databricks.warehouse.warehouse import SQLWarehouse

__all__ = [
    "WarehousePreparedStatement",
    "WarehouseStatementResult",
    "WarehouseStatementBatch",
]

logger = logging.getLogger(__name__)


_DEFAULT_BYTE_SIZE = 32 * 1024 * 1024
# Module-level — single source of truth for retryable error codes.
_RETRYABLE_ERROR_CODES: frozenset[str] = frozenset({
    "DELTA_CONCURRENT_APPEND.ROW_LEVEL_CHANGES",
})

_RETRYABLE_ELAPSED_LIMIT: float = 120.0
_RETRYABLE_ITERATION_LIMIT: int = 3

def _empty_arrow_batches(arrow_schema: pa.Schema) -> Iterator[pa.RecordBatch]:
    """Yield a single zero-row :class:`pa.RecordBatch` matching *arrow_schema*.

    Used to keep schema information flowing through ``_read_arrow_batches``
    when a warehouse statement returns no rows — without this, the
    iterator stays empty and ``read_arrow_table`` collapses to a
    schema-less empty table.
    """
    yield pa.RecordBatch.from_arrays(
        [pa.array([], type=f.type) for f in arrow_schema],
        schema=arrow_schema,
    )

DONE_STATES = {
    StatementState.CANCELED,
    StatementState.CLOSED,
    StatementState.FAILED,
    StatementState.SUCCEEDED,
}

FAILED_STATES = {
    StatementState.FAILED,
    StatementState.CANCELED,
}


# Map the Databricks SDK ``StatementState`` to the unified
# :class:`yggdrasil.data.enums.State`. ``CLOSED`` is "result already
# fetched / TTL elapsed" — terminal but not an error, so it buckets with
# ``SUCCEEDED`` (matches the legacy ``done and not failed`` behavior).
_SDK_TO_STATE: dict[StatementState, State] = {
    StatementState.PENDING:   State.PENDING,
    StatementState.RUNNING:   State.RUNNING,
    StatementState.SUCCEEDED: State.SUCCEEDED,
    StatementState.CLOSED:    State.SUCCEEDED,
    StatementState.FAILED:    State.FAILED,
    StatementState.CANCELED:  State.CANCELED,
}


# Aliases must survive ``{name}``-style substitution.
_VALID_ALIAS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ---------------------------------------------------------------------------
# WarehousePreparedStatement
# ---------------------------------------------------------------------------


class WarehousePreparedStatement(PreparedStatement):
    """Typed Databricks SQL prepared statement.

    Carries everything :meth:`SQLWarehouse._submit_statement` reads — no
    sentinel attributes, no ``getattr`` fallbacks.

    Routing & scope
    ---------------
    ``warehouse_id`` / ``warehouse_name`` are routing hints: when set,
    :meth:`SQLWarehouse.execute` redirects submission to the matching
    warehouse rather than ``self``.  ``catalog_name`` / ``schema_name``
    set the per-statement context.

    Wire format
    -----------
    ``disposition`` / ``format`` control how Databricks returns results.
    Defaults are applied by :class:`SQLWarehouse` when these are ``None``;
    CSV / ARROW_STREAM force ``EXTERNAL_LINKS`` because INLINE only
    supports JSON_ARRAY.

    Parameters & external tables
    ----------------------------
    ``parameters`` is the SDK-typed list (use :meth:`with_parameters` for
    a Mapping-friendly builder).  ``external_volume_paths`` maps query-
    text aliases to staged :class:`VolumePath` instances; :meth:`prepare`
    auto-stages tabular ``external_data`` into Parquet on a fresh
    :class:`VolumePath`.

    Retry
    -----
    Inherits ``retry`` (a :class:`WaitingConfig`) from
    :class:`PreparedStatement`.  Default is ``None`` (not retryable).
    Pass ``retry=WaitingConfig(...)``, ``retry=True`` for the standard
    default policy, or ``retry={"timeout": 60, "retries": 3}`` for a
    dict-shaped config; see :meth:`WaitingConfig.from_`.
    """

    # ---- Routing ----
    warehouse_id: Optional[str] = None
    warehouse_name: Optional[str] = None

    # ---- Scope ----
    catalog_name: Optional[str] = None
    schema_name: Optional[str] = None

    # ---- Wire format ----
    disposition: Optional[Disposition] = None
    format: Optional[Format] = None

    # ---- Server-side wait ----
    on_wait_timeout: Optional[ExecuteStatementRequestOnWaitTimeout] = None
    wait_timeout: Optional[str] = None

    # ---- Result caps ----
    byte_limit: Optional[int] = None
    row_limit: Optional[int] = None

    # ---- Bindings ----
    parameters: Optional[List[StatementParameterListItem]] = None
    external_volume_paths: Optional[dict[str, VolumePath]] = None

    def __init__(
        self,
        text: str = "",
        *,
        key: Optional[str] = None,
        retry: Optional[WaitingConfigArg] = None,
        warehouse_id: Optional[str] = None,
        warehouse_name: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        disposition: Optional[Disposition] = None,
        format: Optional[Format] = None,
        on_wait_timeout: Optional[ExecuteStatementRequestOnWaitTimeout] = None,
        wait_timeout: Optional[str] = None,
        byte_limit: Optional[int] = None,
        row_limit: Optional[int] = None,
        parameters: Optional[List[StatementParameterListItem]] = None,
        external_volume_paths: Optional[dict[str, VolumePath]] = None,
        external_data: Optional[
            Mapping[str, "ExternalStatementData | Tabular | str | tuple"]
        ] = None,
        **kwargs: Any,
    ):
        super().__init__(text, key=key, retry=retry, external_data=external_data)
        self.warehouse_id = warehouse_id
        self.warehouse_name = warehouse_name
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.disposition = disposition
        self.format = format
        self.on_wait_timeout = on_wait_timeout
        self.wait_timeout = wait_timeout
        self.byte_limit = byte_limit
        self.row_limit = row_limit
        self.parameters = parameters
        self.external_volume_paths = external_volume_paths

    # ------------------------------------------------------------------
    # External data validation / staging
    # ------------------------------------------------------------------

    @classmethod
    def check_external_data(
        cls,
        external_data: Optional[Mapping[str, Any]] = None,
        *,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        resource_name: Optional[str] = None,
        temporary: bool = True,
        client: "DatabricksClient | None" = None,
    ) -> dict[str, VolumePath]:
        """Validate ``external_data`` and stage tabular values to Parquet volumes.

        Each entry maps a query-text alias (used as ``{alias}`` in the
        statement text) to one of:

        - an existing :class:`VolumePath` — passed through.
        - tabular data (Arrow / polars / pandas / list / dict) — staged as
          Parquet onto a fresh :meth:`Table.insert_volume_path` under the
          supplied ``catalog_name`` / ``schema_name`` (alias becomes the
          table-name segment when ``resource_name`` is unset).
        - an :class:`ExternalStatementData` wrapping either of the above —
          the underlying value is unwrapped and processed as if passed
          directly.  ``text_value``-only entries (no tabular bound) are
          skipped here; they're substituted via
          :attr:`PreparedStatement.external_data` by the batch coercer.

        Returns a fresh ``dict[str, VolumePath]``; never mutates input.
        Empty / ``None`` input returns ``{}``.
        """
        if external_data is None:
            return {}
        if not isinstance(external_data, Mapping):
            raise TypeError(
                f"external_data must be a mapping; got {type(external_data).__name__}"
            )
        if not external_data:
            return {}

        out: dict[str, VolumePath] = {}
        for alias, value in external_data.items():
            cls._validate_alias(alias)

            # Unwrap ExternalStatementData: ``text_value``-only entries
            # need no staging here; entries with a tabular bound get
            # processed exactly like a raw tabular value.
            if isinstance(value, ExternalStatementData):
                if value.tabular is None:
                    continue
                value = value.tabular

            if isinstance(value, VolumePath):
                out[alias] = value
                continue
            if isinstance(value, DatabricksPath):
                raise TypeError(
                    f"external_data[{alias!r}]: only VolumePath is supported, "
                    f"got {type(value).__name__}; stage to a Volume first"
                )
            if value is None:
                raise ValueError(
                    f"external_data[{alias!r}]: value is None; "
                    f"pass a VolumePath or tabular data"
                )

            out[alias] = cls._stage_external_value(
                alias=alias,
                value=value,
                catalog_name=catalog_name,
                schema_name=schema_name,
                resource_name=resource_name,
                temporary=temporary,
                client=client
            )

        return out

    @staticmethod
    def volume_path_text_value(path: VolumePath) -> str:
        """Build the SQL fragment that references *path* in a query.

        Single source of truth for the ``parquet.\\`<full>\\``` form used
        by :meth:`WarehouseStatementBatch._coerce`; reusing this helper
        keeps text-substitution and cleanup pointing at the same string.
        """
        return f"parquet.`{path.full_path()}`"

    @staticmethod
    def _validate_alias(alias: Any) -> None:
        if not isinstance(alias, str) or not alias:
            raise ValueError(
                f"external_data alias must be a non-empty string; got {alias!r}"
            )
        if not _VALID_ALIAS_RE.match(alias):
            raise ValueError(
                f"external_data alias {alias!r} is not a valid identifier; "
                f"must match [A-Za-z_][A-Za-z0-9_]*"
            )

    @classmethod
    def _stage_external_value(
        cls,
        *,
        alias: str,
        value: Any,
        catalog_name: Optional[str],
        schema_name: Optional[str],
        resource_name: Optional[str],
        temporary: bool,
        client: "DatabricksClient"
    ) -> VolumePath:
        """Stage tabular ``value`` to a fresh Parquet volume.  Override
        in subclasses for custom file formats / staging policies.

        Mints the staging path through :meth:`Table.insert_volume_path`
        on a transient :class:`Table` keyed by ``(catalog, schema,
        resource_name or alias)`` — same per-table ``<table>``
        volume layout the warehouse insert path uses — then writes the
        Parquet payload, unlinking the path on write failure when
        ``temporary=True``.
        """
        from yggdrasil.databricks.table.table import Table

        if not catalog_name or not schema_name:
            raise ValueError(
                f"external_data[{alias!r}]: staging tabular value requires "
                f"catalog_name and schema_name; got catalog_name={catalog_name!r}, "
                f"schema_name={schema_name!r}"
            )

        table = Table(
            service=client.tables,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=resource_name or alias,
        )
        staged = table.insert_volume_path(temporary=temporary)
        try:
            staged.as_media(media_type=MediaTypes.PARQUET).write_table(
                value, mode=Mode.OVERWRITE,
            )
        except Exception as e:
            if staged.temporary:
                staged.clear()
            raise RuntimeError(
                f"Failed to stage external_data[{alias!r}] "
                f"({type(value).__name__}) as Parquet: {e}"
            ) from e
        return staged

    # ------------------------------------------------------------------
    # Coercion / preparation
    # ------------------------------------------------------------------

    @classmethod
    def prepare(
        cls,
        statement: "WarehousePreparedStatement | PreparedStatement | str",
        *,
        client: "DatabricksClient | None" = None,
        parameters: Optional[Mapping[str, Any] | List[StatementParameterListItem]] = None,
        external_data: Optional[Mapping[str, Any]] = None,
        external_volume_paths: Optional[dict[str, VolumePath]] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        resource_name: Optional[str] = None,
        disposition: Optional[Disposition] = None,
        format: Optional[Format] = None,
        temporary: bool = True,
        retry: Optional[WaitingConfigArg] = None,
        **kwargs: Any,
    ) -> "WarehousePreparedStatement":
        """Coerce + bind parameters + stage external data.

        Merge precedence on alias collisions:
        existing volumes < caller-supplied < just-staged.

        ``retry`` is applied only when non-None — preserves whatever was
        already on the statement when ``prepare`` is given an existing
        :class:`WarehousePreparedStatement`.  Pass ``retry=False`` to
        explicitly clear an existing retry policy.
        """
        prepared = cls.from_(statement)

        # ---- External data: validate + stage ----
        staged = cls.check_external_data(
            external_data,
            catalog_name=catalog_name,
            schema_name=schema_name,
            resource_name=resource_name,
            temporary=temporary,
            client=client
        )
        ext_paths: dict[str, VolumePath] = dict(prepared.external_volume_paths or {})
        if external_volume_paths:
            ext_paths.update(external_volume_paths)
        ext_paths.update(staged)

        # ---- Generic external-data registry ----
        # Mirror every staged / supplied path into the base
        # ``external_data`` map with a pre-baked ``text_value``, and merge
        # any caller-supplied :class:`ExternalStatementData` entries (which
        # may carry their own ``text_value`` for already-baked SQL
        # fragments — e.g. an existing table reference).  Substitution at
        # batch coerce time reads ``external_data`` first, then falls back
        # to ``external_volume_paths`` for entries that didn't get mirrored.
        ext_data: dict[str, ExternalStatementData] = dict(prepared.external_data or {})
        if external_data:
            for alias, value in external_data.items():
                if isinstance(value, ExternalStatementData):
                    # Caller-built entry: preserve text_value as-is when
                    # set; otherwise materialize from the staged path
                    # below.
                    if value.text_value:
                        ext_data[alias] = ExternalStatementData(
                            alias,
                            tabular=value.tabular,
                            text_value=value.text_value,
                        )
                    elif value.tabular is None:
                        raise ValueError(
                            f"external_data[{alias!r}]: ExternalStatementData "
                            f"has neither tabular nor text_value"
                        )
        for alias, path in ext_paths.items():
            ext_data[alias] = ExternalStatementData(
                alias,
                tabular=ext_data.get(alias).tabular if alias in ext_data else None,
                text_value=cls.volume_path_text_value(path),
            )

        # ---- Parameters: list or mapping ----
        new_params: list[StatementParameterListItem] = list(prepared.parameters or [])
        if parameters:
            if isinstance(parameters, Mapping):
                new_params.extend(_mapping_to_parameter_list(parameters))
            else:
                new_params.extend(parameters)

        prepared = prepared.with_parameters(new_params)

        # ---- Wire format defaults ----
        if format is not None:
            prepared.format = format
        elif prepared.format is None:
            prepared.format = Format.ARROW_STREAM

        if prepared.format is Format.JSON_ARRAY:
            prepared.disposition = disposition or prepared.disposition or Disposition.INLINE
        else:
            # CSV / ARROW_STREAM only support EXTERNAL_LINKS.
            prepared.disposition = Disposition.EXTERNAL_LINKS

        # ---- Retry config: only override when caller asked.
        # ``False`` explicitly disables; any other non-None value passes
        # through WaitingConfig.from_.
        if retry is not None:
            if retry is False:
                prepared.retry = None
            else:
                prepared.retry = WaitingConfig.from_(retry)

        # Apply any extra typed-field kwargs in one shot.
        if catalog_name is not None:
            prepared.catalog_name = catalog_name
        if schema_name is not None:
            prepared.schema_name = schema_name
        for k, v in kwargs.items():
            if hasattr(prepared, k):
                setattr(prepared, k, v)

        prepared.external_volume_paths = ext_paths or None
        prepared.external_data = ext_data or None

        # Bake placeholder substitution into the text now so the
        # single-statement submit path (which doesn't go through
        # WarehouseStatementBatch._coerce) reaches the SDK with valid
        # SQL.  Idempotent: re-prepare on an already-substituted
        # statement is a no-op since the placeholders are gone.
        if ext_data:
            prepared.text = cls.apply_external_substitution(
                prepared.text, ext_data,
            )
        return prepared

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def with_warehouse(
        self,
        warehouse_id: Optional[str] = None,
        warehouse_name: Optional[str] = None,
    ) -> "WarehousePreparedStatement":
        copied = copy_mod.copy(self)
        copied.warehouse_id = warehouse_id if warehouse_id is not None else self.warehouse_id
        copied.warehouse_name = warehouse_name if warehouse_name is not None else self.warehouse_name
        return copied

    def with_parameters(
        self,
        parameters: "Mapping[str, Any] | List[StatementParameterListItem] | None",
        *,
        merge: bool = True,
        copy: bool = True,
    ) -> "WarehousePreparedStatement":
        """Return (or update in place) a copy with ``parameters`` set.

        ``parameters`` may be an SDK-typed list, a ``{name: value}``
        mapping, or ``None`` to clear.  ``merge=True`` (default) appends
        to existing parameters; ``copy=False`` mutates ``self`` in place.
        """
        if parameters is None:
            new_params: Optional[List[StatementParameterListItem]] = None
        elif isinstance(parameters, Mapping):
            new_params = _mapping_to_parameter_list(parameters)
        else:
            new_params = list(parameters)

        if new_params and merge and self.parameters:
            new_params = list(self.parameters) + new_params

        target = copy_mod.copy(self) if copy else self
        target.parameters = new_params or None
        return target

    def to_parameter_list(self) -> Optional[List[StatementParameterListItem]]:
        """Single override point for richer parameter representations."""
        return self.parameters

    # ------------------------------------------------------------------
    # Per-statement scratch cleanup
    # ------------------------------------------------------------------

    def clear_temporary_resources(self) -> None:
        """Unlink any temporary staged volumes and clear the registry.

        Idempotent: safe to call repeatedly.  Mirrors the per-batch
        contract — the legacy ``external_volume_paths`` is the single
        source of truth for paths to unlink; ``external_data`` is only
        cleared to release tabular references so a re-prepared statement
        re-stages cleanly.
        """
        if self.external_volume_paths:
            for alias, path in self.external_volume_paths.items():
                if getattr(path, "temporary", False):
                    try:
                        Job.make(path.unlink, missing_ok=True).fire_and_forget()
                    except Exception:
                        logger.exception(
                            "Failed to unlink temporary staged volume %r (alias=%r); continuing.",
                            path, alias,
                        )
            self.external_volume_paths = None

        if self.external_data:
            self.external_data = None


def _mapping_to_parameter_list(
    parameters: Mapping[str, Any],
) -> List[StatementParameterListItem]:
    """``{name: value}`` -> SDK ``StatementParameterListItem`` list.

    Already-typed values pass through; everything else goes as a stringified
    ``value`` and the SDK infers the type — matches Databricks' auto-coercion
    on untyped parameters.
    """
    out: List[StatementParameterListItem] = []
    for name, value in parameters.items():
        if isinstance(value, StatementParameterListItem):
            out.append(value)
            continue
        out.append(
            StatementParameterListItem(
                name=name,
                value=None if value is None else str(value),
            )
        )
    return out


# ---------------------------------------------------------------------------
# WarehouseStatementResult
# ---------------------------------------------------------------------------


class WarehouseStatementResult(StatementResult):
    """Databricks-backed :class:`StatementResult`.

    Wraps a :class:`WarehousePreparedStatement` plus per-execution state
    (``statement_id``, cached :class:`StatementResponse`).  Configuration
    (text, parameters, external tables, routing) lives on
    ``self.statement``.

    The ``warehouse_id`` field is the *resolved* warehouse the statement
    actually ran on (set after submission); ``self.statement.warehouse_id``
    is the *requested* routing hint (set by the caller before submission).

    Retry semantics are inherited from :class:`StatementResult`: the
    looping ``retry()`` method drives ``start(reset=True)`` per attempt,
    sleeping per ``self.statement.retry`` (a :class:`WaitingConfig`)
    between tries.  ``retryable`` is a derived property — non-retryable
    when ``self.statement.retry is None`` or the attempt budget is
    exhausted.
    """

    _PREPARED_CLASS = WarehousePreparedStatement
    _FINAL_TABULAR_IO: ClassVar[bool] = True

    @classmethod
    def default_media_type(cls) -> "MimeType | None":
        return MimeTypes.DATABRICKS_STATEMENT_RESULT

    executor: "SQLWarehouse"
    statement: WarehousePreparedStatement
    statement_id: Optional[str] = None

    def __init__(
        self,
        executor: "SQLWarehouse",
        statement: Optional[WarehousePreparedStatement] = None,
        *,
        statement_id: Optional[str] = None,
        _response: Optional[StatementResponse] = None,
        **kwargs: Any,
    ):
        self.statement_id = statement_id
        self._response = _response
        super().__init__(statement=statement, executor=executor, **kwargs)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def retryable(self) -> bool:
        """Result-level retry predicate.

        Drives the base :meth:`StatementResult.retry` loop, which calls
        :meth:`start` with ``reset=True`` and sleeps per
        ``self.statement.retry`` between attempts.  Returns ``True`` only
        when:

        - the statement ran and failed (``_compute_error`` returned an error),
        - the failure code is one we know is transient
          (``DELTA_CONCURRENT_APPEND.ROW_LEVEL_CHANGES`` today),
        - we haven't blown the elapsed budget (5 min) or the attempt
          budget (2 retries).

        Submission-level failures (cold/busy warehouse, transport) aren't
        seen here — those propagate from :meth:`send` and the caller
        owns the retry decision.
        """
        if self.iteration >= _RETRYABLE_ITERATION_LIMIT:
            return False

        elapsed = self.elapsed_timestamp
        if elapsed and elapsed > _RETRYABLE_ELAPSED_LIMIT:
            return False

        error = self._compute_error()
        if error is None:
            return False

        msg = getattr(error, "message", None) or ""
        return any(code in msg for code in _RETRYABLE_ERROR_CODES)

    @property
    def started(self) -> bool:
        """True once the statement has been submitted (``statement_id`` present).

        Doesn't read :attr:`state` (which would refresh) — the
        ``statement_id`` is set synchronously by :meth:`start` and is
        the cheapest started-flag the warehouse path has.
        """
        return bool(self.statement_id) and self.statement_id != "unknown"

    @property
    def cached(self) -> bool:
        """True when the statement is in a terminal state (response is final)."""
        return self.done

    @property
    def client(self):
        return self.executor.client

    def persist(
        self,
        engine: Literal["arrow", "polars", "spark", "auto"] = "auto",
        *,
        data: Any | None = None,
    ) -> "WarehouseStatementResult":
        """No-op for warehouse results — backend already caches the response."""
        return self

    def unpersist(self) -> None:
        """No-op."""
        pass

    def set_api_response(self, response: StatementResponse) -> "WarehouseStatementResult":
        """Test hook: stuff a fully-formed API response into the result."""
        self._response = response

        if isinstance(self._response, StatementResponse):
            self.statement_id = response.statement_id

        return self

    def start(
        self,
        reset: bool = False,
        *,
        wait: WaitingConfigArg = False,
        raise_error: bool = False,
        **kwargs: Any,
    ) -> "WarehouseStatementResult":
        """Submit the statement.  Idempotent on already-started results.

        ``reset=True`` cancels the existing submission (when not already
        terminal) and clears local state before resubmitting — this is
        the path :meth:`StatementResult.retry` drives.

        Caller kwargs override anything carried on ``self.statement`` for
        this submission only — the underlying statement's hints stay put.
        """
        if self.started:
            if not reset:
                logger.debug(
                    "Skipping start for statement %r — already started", self,
                )
                return self

            logger.debug(
                "Resetting statement %r before resubmit (iteration=%d)",
                self, self.iteration,
            )
            self.cancel(wait=False)

            self.statement_id = None
            self._response = None
            self._unpersist_schema()

        logger.debug("Submitting statement on %r", self.executor)
        submitted = self.executor.send(self.statement)

        self.statement = submitted.statement
        self.set_api_response(submitted._response)
        self.iteration = self.iteration + 1
        logger.debug(
            "Submitted statement %r (iteration=%d, state=%s)",
            self, self.iteration,
            submitted._response.status.state if submitted._response else None,
        )

        return self

    def cancel(self, wait: WaitingConfigArg = False, **kwargs) -> "WarehouseStatementResult":
        """Cancel the running statement.  No-op when not started or already terminal."""
        if not self.started:
            return self
        # Use the cached SDK response directly here — we don't want to
        # trigger ``state``'s refresh just to short-circuit on an already
        # terminal statement, and the cached response is authoritative
        # for this check.
        if self._response is not None and self._response.status.state in DONE_STATES:
            return self

        wait = WaitingConfig.from_(wait)

        if wait:
            logger.debug("Cancelling statement %r", self)
            try:
                self.client.workspace_client().statement_execution.cancel_execution(
                    statement_id=self.statement_id,
                )
            except Exception:
                logger.exception("Failed to cancel statement %r", self.key)
        else:
            logger.debug("Cancelling statement %r (no-wait)", self)
            Job.make(self.client.workspace_client().statement_execution.cancel_execution, statement_id=self.statement_id).fire_and_forget()

        self._response = None
        return self

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.started:
            return f"WarehouseStatementResult({self.monitoring_url!r})"
        return f"WarehouseStatementResult(text={self.statement.text!r})"

    def __str__(self) -> str:
        return self.monitoring_url if self.started else self.statement.text

    @property
    def monitoring_url(self) -> str:
        """Databricks UI monitoring URL for this statement execution."""
        return "%ssql/warehouses/%s/monitoring?queryId=%s" % (
            self.client.base_url.to_string(),
            self.executor.warehouse_id,
            self.statement_id,
        )

    # ------------------------------------------------------------------
    # State / status
    # ------------------------------------------------------------------

    @property
    def response(self) -> StatementResponse:
        """Latest statement response (auto-refreshes until terminal)."""
        self.refresh_status()
        return self._response

    def api_result_data_at_index(self, chunk_index: int):
        """Fetch a specific result chunk by index via the SDK."""
        return self.client.workspace_client().statement_execution.get_statement_result_chunk_n(
            statement_id=self.statement_id,
            chunk_index=chunk_index,
        )

    def refresh_status(self) -> "WarehouseStatementResult":
        """Pull the latest :class:`StatementResponse` from the SDK.

        Pure status read — no side effects beyond updating ``self._response``.
        Terminal responses are sticky (the SDK won't change them), so we
        short-circuit and return ``self`` without a round-trip.  Auto-retry
        of retryable failures lives in :meth:`retry` / the base
        :class:`StatementResult` retry loop, not here — keeping this method
        pure means ``wait()`` loops and the ``state`` snapshot cache hold
        their invariants.
        """
        if not self.statement_id:
            self._response = StatementResponse(
                statement_id="unknown",
                status=StatementStatus(state=StatementState.PENDING),
            )
            return self

        if self._response is not None and self._response.status.state in DONE_STATES:
            return self

        cached_state = (
            self._response.status.state if self._response is not None else None
        )
        statement_execution = self.client.workspace_client().statement_execution
        response = statement_execution.get_statement(self.statement_id)
        new_state = response.status.state

        if new_state in DONE_STATES:
            logger.info(
                "Statement %r finished in state %s", self, new_state,
            )
        elif cached_state != new_state:
            logger.debug(
                "Polled statement %r (state=%s, prev=%s)",
                self, new_state, cached_state,
            )
        else:
            logger.debug("Polled statement %r (state=%s)", self, new_state)
        self.set_api_response(response)
        return self

    @property
    def status(self) -> StatementStatus:
        return self.response.status

    @property
    def sdk_state(self) -> StatementState:
        """Backend-typed :class:`StatementState` (Databricks SDK).

        Use :attr:`state` for the unified :class:`State` enum that
        ``done`` / ``failed`` / ``started`` derive from; reach for
        ``sdk_state`` only when a caller specifically needs the SDK
        type (e.g. for pattern matching against ``StatementState``
        members the unified enum collapses, like ``CLOSED``).
        """
        return self.status.state

    def _compute_state(self) -> State:
        """Refresh and map the SDK state onto the unified :class:`State`.

        Single source of truth for ``done`` / ``failed`` / ``started``
        on the warehouse path; the base ``state`` property caches this
        for the duration of any :meth:`state_snapshot` block so a code
        path that checks several state-derived predicates only hits the
        warehouse status endpoint once.
        """
        return _SDK_TO_STATE.get(self.sdk_state, State.PENDING)

    def _compute_error(self) -> Optional[SQLError]:
        """Build a :class:`SQLError` from the SDK response, or ``None`` on success.

        Only meaningful in terminal states — :meth:`SQLError.from_statement`
        inspects ``status.error``, which the SDK only populates on FAILED
        / CANCELED.  Called from both :attr:`retryable` (no-refresh, uses
        cached state) and :meth:`_raise_for_status` (after ``wait()`` has
        forced termination).
        """
        if self.sdk_state not in FAILED_STATES:
            return None
        return SQLError.from_statement(self)

    def _raise_for_status(self) -> None:
        error = self._compute_error()
        if error is not None:
            raise error

    # ------------------------------------------------------------------
    # Manifest / schema
    # ------------------------------------------------------------------

    @property
    def manifest(self):
        return self.response.manifest

    @property
    def result(self):
        return self.response.result

    @property
    def disposition(self) -> Optional[Disposition]:
        return self.statement.disposition

    def _collect_schema(self, options) -> Schema:
        if options.target:
            return options.target

        self.wait()
        manifest = self.manifest
        metadata = {
            b"engine": b"databricks-sql",
            b"warehouse_id": (self.executor.warehouse_id or "").encode(),
            b"statement_id": (self.statement_id or "").encode(),
        }
        result_schema = getattr(manifest, "schema", None) if manifest is not None else None
        columns = getattr(result_schema, "columns", None) if result_schema is not None else None

        if columns is None:
            return Schema.empty(metadata=metadata)

        schema = Schema.from_fields(
            [parse_databricks_field(c) for c in (columns or [])],
            name=self.key,
            metadata=metadata,
        )
        self._persist_schema(schema)
        return schema

    # ------------------------------------------------------------------
    # External links
    # ------------------------------------------------------------------

    def external_links(self) -> Iterator[ExternalLink]:
        """Yield external result links for ``Disposition.EXTERNAL_LINKS``."""
        if self.disposition != Disposition.EXTERNAL_LINKS:
            raise RuntimeError(
                f"Cannot get external links from {self}; disposition is "
                f"{self.disposition!r}, not EXTERNAL_LINKS"
            )

        self.wait()
        result_data = self.result
        wsdk = self.client.workspace_client()

        while True:
            links = result_data.external_links or []
            if not links:
                return
            yield from links

            next_internal = getattr(links[-1], "next_chunk_internal_link", None)
            if not next_internal:
                return

            try:
                chunk_index = int(next_internal.rstrip("/").split("/")[-1])
            except Exception as e:
                raise ValueError(f"Bad next_chunk_internal_link {next_internal!r}: {e}") from e

            try:
                result_data = wsdk.statement_execution.get_statement_result_chunk_n(
                    statement_id=self.statement_id,
                    chunk_index=chunk_index,
                )
            except Exception as e:
                raise ValueError(f"Cannot retrieve data batch from {next_internal!r}: {e}") from e

    # ------------------------------------------------------------------
    # Arrow conversions
    # ------------------------------------------------------------------

    def _read_arrow_batches(self, options: CastOptions) -> Iterator[pa.RecordBatch]:
        options = options.check_target(self.collect_schema)

        max_workers = 8
        max_in_flight = max_workers * 2

        # Pool is owned by the warehouse executor — its lifetime matches
        # the warehouse handle, so we don't leak TCP connections in a
        # long-running process that creates and discards warehouses, and
        # we still get connection reuse across every chunk fetched
        # against this warehouse.
        http = self.executor.external_link_pool(max_workers)
        options = options.with_target(self._collect_schema(options))
        byte_size = options.byte_size or _DEFAULT_BYTE_SIZE
        memory_pool = options.arrow_memory_pool
        pending: List[pa.RecordBatch] = []
        pending_bytes = 0
        yielded_any = False
        total_rows = 0
        total_batches = 0
        total_bytes = 0

        def fetch_batches(url: str) -> Iterator[pa.RecordBatch]:
            # ``preload_content=False`` keeps the session's
            # :class:`MemoryStream` lazy — bytes pull through the
            # decoder pipeline on demand instead of buffering the full
            # payload (chunks can run hundreds of MB) up front. We
            # stream straight into Arrow's IPC reader instead.
            resp = http.fetch(
                "GET", url,
                preload_content=False,
                decode_content=True,
            )
            try:
                if resp.status >= 400:
                    raise RuntimeError(f"GET {url} failed: {resp.status}")
                with pa.input_stream(resp) as src:
                    reader = pipc.open_stream(src)
                    for batch in reader:
                        yield batch
            finally:
                # Drain anything Arrow didn't consume so the connection
                # can return to the session's idle cache cleanly;
                # ``release_conn`` alone won't recycle a partially-read
                # response.
                try:
                    resp.drain_conn()
                except Exception:
                    pass
                resp.release_conn()

        def jobs() -> Iterable[Job]:
            nonlocal total_rows, total_batches, total_bytes

            for link in self.external_links():
                if link.external_link:
                    if logger.isEnabledFor(logging.INFO):
                        total_batches += 1
                        total_rows += link.row_count or 0
                        total_bytes += link.byte_count or 0
                    yield Job.make(fetch_batches, link.external_link)

        def raw_batches() -> Iterator[pa.RecordBatch]:
            with JobPoolExecutor.from_(max_workers) as ex:
                for result in ex.as_completed(
                    jobs(),
                    ordered=True,
                    max_in_flight=max_in_flight,
                    cancel_on_exit=True,
                    shutdown_on_exit=True,
                    shutdown_wait=False,
                ):
                    yield from result.result

        def flush() -> Iterator[pa.RecordBatch]:
            nonlocal pending, pending_bytes, yielded_any
            if not pending:
                return

            # Always go through `concat_batches`, even for a singleton.
            # The singleton-skip shortcut handed out a batch that could
            # alias the HTTP response buffer that backed the IPC read;
            # once `fetch_batches` returns and the response is GC'd,
            # those buffers vanish.  `concat_batches` materializes a
            # fresh batch owned by `memory_pool`, breaking the alias.
            combined = pa.concat_batches(pending, memory_pool=memory_pool)
            casted = options.cast_arrow_table(combined)
            pending = []
            pending_bytes = 0
            yielded_any = True
            yield casted

        for batch in raw_batches():
            pending.append(batch)
            pending_bytes += batch.nbytes
            if pending_bytes >= byte_size:
                yield from flush()

        yield from flush()

        if not yielded_any:
            yield from _empty_arrow_batches(options.target.to_arrow_schema())
        else:
            logger.info(
                "Statement %r streamed %d batches / %d rows / %d bytes",
                self, total_batches, total_rows, total_bytes,
            )

    def _write_arrow_batches(self, batches: Iterable[pa.RecordBatch], options: CastOptions) -> None:
        raise NotImplementedError("Cannot write to Databricks SQL")


# ---------------------------------------------------------------------------
# WarehouseStatementBatch
# ---------------------------------------------------------------------------


class WarehouseStatementBatch(StatementBatch):
    """Warehouse-backed batch of statements.

    External-table aliases are resolved at coerce-time by reading the
    per-statement ``external_volume_paths`` and rewriting ``{alias}``
    occurrences in the text to ``parquet.\\`<full_path>\\```.  Rewriting
    happens on a *copy*, so re-submitting the same batch is safe.

    The optional ``external_paths`` constructor argument is a batch-wide
    set of aliases applied on top of any per-statement registry — useful
    for shared scratch volumes that every statement in the batch reads.
    Per-statement entries take precedence on alias collisions.

    :meth:`retry` is inherited from :class:`StatementBatch` — it walks
    the result map, picks every entry that is both ``failed`` and
    ``retryable``, and reissues each via :meth:`StatementResult.retry`
    on the configured ``parallel`` thread pool.
    """

    external_volume_paths: Optional[dict[str, VolumePath]]

    def __init__(
        self,
        executor: "SQLWarehouse",
        statements: Optional[Iterable["WarehousePreparedStatement | str"]] = None,
        *,
        parallel: int = 1,
        external_paths: Optional[dict[str, VolumePath]] = None,
    ):
        super().__init__(executor=executor, statements=None, parallel=parallel)
        self.external_volume_paths = dict(external_paths) if external_paths else {}
        if statements:
            self.extend(statements)

    def _coerce(self, statement: "WarehousePreparedStatement | str") -> WarehousePreparedStatement:
        stmt = WarehousePreparedStatement.from_(statement)

        # Build the effective substitution map.  Three sources, in
        # increasing precedence:
        #   1. batch-wide ``external_volume_paths`` (legacy)
        #   2. per-statement ``external_volume_paths`` (legacy)
        #   3. per-statement ``external_data`` (new generic registry)
        # Generic entries win because :meth:`WarehousePreparedStatement.prepare`
        # already mirrors any staged path into ``external_data`` with a
        # pre-baked ``text_value``, so the two sources stay consistent;
        # callers who skipped ``prepare`` (built the statement directly
        # with ``external_volume_paths``) still get substitution.
        effective: dict[str, str] = {}
        for alias, path in (self.external_volume_paths or {}).items():
            effective[alias] = WarehousePreparedStatement.volume_path_text_value(path)
        for alias, path in (stmt.external_volume_paths or {}).items():
            effective[alias] = WarehousePreparedStatement.volume_path_text_value(path)
        for alias, entry in (stmt.external_data or {}).items():
            if entry.text_value is None:
                raise ValueError(
                    f"WarehousePreparedStatement.external_data[{alias!r}] "
                    f"has no text_value; call prepare() first or set it manually"
                )
            effective[alias] = entry.text_value

        if not effective:
            return stmt

        # Substitute on a copy — never mutate the caller's statement.
        rewritten_text = stmt.text
        for alias, text_value in effective.items():
            rewritten_text = rewritten_text.replace(
                "{%s}" % alias, text_value,
            )
        if rewritten_text == stmt.text:
            return stmt

        copied = copy_mod.copy(stmt)
        copied.text = rewritten_text
        return copied

    def clear_temporary_resources(self) -> "WarehouseStatementBatch":
        # Per-statement scratch first (each result owns its statement).
        super().clear_temporary_resources()

        # Batch-wide scratch second.  Idempotent: callers (wait() then
        # retry(), raise_for_status() then retry(), ...) may invoke this
        # more than once, so bail out once we've already cleared.
        if not self.external_volume_paths:
            return self

        for alias, path in list(self.external_volume_paths.items()):
            if getattr(path, "temporary", False):
                try:
                    Job.make(path.unlink, missing_ok=True).fire_and_forget()
                except Exception:
                    logger.exception(
                        "Failed to unlink temporary path %r (alias=%r); continuing.",
                        path, alias,
                    )

        self.external_volume_paths = None

        return self