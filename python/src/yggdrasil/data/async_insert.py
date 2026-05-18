"""Backend-agnostic ``AsyncInsert`` — a deferred, replayable table insert.

An :class:`AsyncInsert` is the canonical "stage now, apply later"
record: it points at a target (a :class:`URL` identifying the
destination table), carries an expected :class:`Schema`, and lists
the Parquet payload(s) + sibling metadata logs that materialise the
rows the target should eventually receive. The same record renders
to raw SQL via :meth:`to_sql` and to executable
:class:`PreparedStatement` objects via :meth:`to_statements`, so the
same instance flows from "metadata-on-disk" to "live in-flight
batch" without an intermediate factory.

The class lives in :mod:`yggdrasil.data` so any backend (Databricks
warehouse, Spark, DuckDB, Trino, …) can plug into the same record
shape. The backend-specific bits — which :class:`PreparedStatement`
subclass to mint, how to render a Parquet path inside SQL, and how
to attach the path to a statement so the lifecycle hook unlinks it
on success — are exposed as class-level hooks the backend overrides
in a thin subclass:

- :attr:`_PREPARED_CLASS` — :class:`PreparedStatement` subclass for
  :meth:`to_statements`. Databricks overrides to
  :class:`WarehousePreparedStatement` (carries
  ``external_volume_paths`` / ``catalog_name`` / ``schema_name``).
- :meth:`_resolve_path` — coerce a string / live path into the
  backend's live path handle. Default returns the input unchanged;
  Databricks override routes through :meth:`DatabricksPath.from_`.
- :meth:`_format_path_for_sql` — render a path string as the SQL
  fragment that appears inside ``SELECT * FROM <ref>``. Default
  emits ``parquet.\\`<path>\\``` (Spark / Databricks dialect); other
  backends override.
- :meth:`_attach_cleanup_paths` — bind staged paths to a freshly
  prepared statement so its
  :meth:`PreparedStatement.clear_temporary_resources` hook removes
  them on success. Default is a no-op (paths only carry through SQL
  text); Databricks override sets
  :attr:`WarehousePreparedStatement.external_volume_paths` and flips
  the ``temporary`` flag on each path.

File layout (when staged on a volume / filesystem) — sibling
``data/`` (Parquet payloads) and ``logs/`` (JSON metadata)
subfolders under :data:`ASYNC_INSERT_ROOT`. Sibling folders keep
the two concerns separable: appliers walk ``logs/`` for merge
candidates and walk ``data/`` for cleanup without filtering past
the other.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from typing import (
    Any,
    ClassVar,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from yggdrasil.data.schema import Field, Schema
from yggdrasil.data.statement import (
    PreparedStatement,
    StatementBatch,
)
from yggdrasil.io.url import URL
from yggdrasil.pickle import json as ygg_json

LOGGER = logging.getLogger(__name__)


__all__ = [
    "AsyncInsert",
    "AsyncWrite",
    "ASYNC_INSERT_ROOT",
    "ASYNC_INSERT_DATA_SUBDIR",
    "ASYNC_INSERT_LOGS_SUBDIR",
    "METADATA_VERSION",
    "iter_records",
    "path_for_sql",
]


# Wire-format version. Format is still POC — overwritten in place when
# the layout changes rather than gracefully version-gated.
METADATA_VERSION = 1

# Staging folder layout — Parquet payloads go under ``data/`` and the
# JSON metadata logs go under ``logs/``. Sibling folders keep the two
# concerns separable: appliers walk ``logs/`` for merge candidates and
# walk ``data/`` for cleanup without filtering past the other.
ASYNC_INSERT_ROOT: str = ".sql/async/insert"
ASYNC_INSERT_DATA_SUBDIR: str = "data"
ASYNC_INSERT_LOGS_SUBDIR: str = "logs"

# Mode tokens, normalised to lowercase. Anything else is treated as an
# append unless explicitly listed.
_APPEND_TOKENS: frozenset[str] = frozenset({"append", "", "auto", "insert"})
_OVERWRITE_TOKENS: frozenset[str] = frozenset({"overwrite"})


# Ordered list of metadata field names + defaults — drives
# :meth:`AsyncInsert.__init__`, :meth:`_replace`, :meth:`to_dict`,
# :meth:`from_dict`, equality, and the tuple-coercion set in
# :meth:`from_dict`. Kept module-level so the hot per-record walks
# don't rebuild it on every call.
#
# ``target`` (a :class:`URL`) and ``schema`` (a :class:`Schema`) live
# on the base :class:`StatementBatch` surface — every backend
# statement uses the same two slots to describe "what resource is
# this for" and "what shape do its rows have", so :class:`AsyncInsert`
# stays a thin wrapper around its operation-specific bits (Parquet
# payload paths, mode, match keys, …).
_METADATA_FIELDS: Tuple[Tuple[str, Any], ...] = (
    ("target", None),
    ("schema", None),
    ("parquet_paths", ()),
    ("metadata_paths", ()),
    ("operation_ids", ()),
    ("created_at", ""),
    ("mode", None),
    ("schema_mode", None),
    ("overwrite_schema", None),
    ("match_by", None),
    ("update_column_names", None),
    ("zorder_by", None),
    ("optimize_after_merge", False),
    ("vacuum_hours", None),
    ("where", None),
    ("prune_by", None),
    ("prune_values", None),
    ("safe_merge", False),
    ("version", METADATA_VERSION),
)


_FIELD_NAMES: frozenset[str] = frozenset(name for name, _ in _METADATA_FIELDS)
_TUPLE_FIELD_NAMES: frozenset[str] = frozenset({
    "parquet_paths",
    "metadata_paths",
    "operation_ids",
    "match_by",
    "update_column_names",
    "zorder_by",
    "prune_by",
})


def path_for_sql(path: Any) -> str:
    r"""Return the path string used inside SQL / metadata.

    Prefers the canonical filesystem-style shape from
    :meth:`full_path` (e.g. Databricks Unity ``/Volumes/...``) when
    available, falls back to the path's ``url`` attribute, finally
    coerces via ``str()``. A bare string passes through unchanged.
    """
    full_path = getattr(path, "full_path", None)
    if callable(full_path):
        try:
            return full_path()
        except Exception:
            pass
    url = getattr(path, "url", None)
    if url is not None:
        return str(url)
    return str(path)


# ---------------------------------------------------------------------------
# AsyncInsert — deferred, replayable table insert as a StatementBatch
# ---------------------------------------------------------------------------


class AsyncInsert(StatementBatch):
    """JSON-serialisable description of a deferred table insert.

    Thin wrapper around :class:`StatementBatch`: the unified
    ``target`` (a :class:`URL` pointing at the destination table)
    and ``schema`` (a :class:`Schema` describing the target columns)
    live on the base batch surface, so every statement / batch /
    result in the project speaks the same two metadata slots.
    AsyncInsert only adds the operation-specific bits (Parquet
    payload paths, sibling metadata logs, operation ids, mode, match
    keys, …) and an auto statement generator (:meth:`to_statements`)
    that materialises a :class:`PreparedStatement` from those bits
    when an executor binds.

    Constructed without an executor it sits in metadata-only mode
    (``self.executor is None``, ``self.results`` empty), suitable for
    JSON round-tripping, merging, and :meth:`to_sql` rendering. Pass
    ``executor=`` to submit straight away.

    Backend extension points
    ------------------------
    Subclasses customise four hooks:

    - :attr:`_PREPARED_CLASS` — :class:`PreparedStatement` subclass
      :meth:`to_statements` mints.
    - :meth:`_resolve_path` — coerce a path string into the backend's
      live path handle (default: passthrough).
    - :meth:`_format_path_for_sql` — render a path inside the
      ``SELECT * FROM <ref>`` clause (default: ``parquet.\\`<path>\\```).
    - :meth:`_attach_cleanup_paths` — bind staged paths to a fresh
      :class:`PreparedStatement` so its lifecycle hook unlinks them on
      success (default: no-op).

    Path serialisation contract
    ---------------------------
    ``parquet_paths`` / ``metadata_paths`` accept either :class:`str`
    URLs or live path handles; in-memory the record carries whatever
    it was handed (so backend staging helpers can stash a freshly-built
    handle without forcing a round-trip through string coercion). On
    any serialisation path — :meth:`to_dict`, :meth:`to_json_bytes`,
    pickle via :meth:`__getstate__` — path entries are dumped as
    their URL strings, ``target`` is dumped as a URL string,
    ``schema`` as a Field dict, and the live :attr:`executor` /
    :attr:`results` are stripped so the output is a clean metadata
    snapshot.
    """

    _METADATA_FIELDS: ClassVar[Tuple[Tuple[str, Any], ...]] = _METADATA_FIELDS

    # Live handles + in-flight state that don't survive pickling —
    # the bound executor, the queue of submitted prepared statements,
    # and the schema cache rebuild on the receiver. Matches the
    # ``yggdrasil.io.session.Session`` / ``yggdrasil.aws.AWSClient``
    # pickle pattern (see ``AGENTS.md`` → "Make objects picklable").
    _TRANSIENT_STATE_ATTRS: ClassVar[frozenset[str]] = frozenset({
        "executor",
        "results",
        "external_volume_paths",
        "_cached_schema",
        "start_timestamp",
    })

    # ------------------------------------------------------------------ #
    # Backend extension points — subclasses override to wire in their
    # own path / statement / SQL conventions.
    # ------------------------------------------------------------------ #

    _PREPARED_CLASS: ClassVar[type[PreparedStatement]] = PreparedStatement

    def _resolve_path(self, path: Any, *, client: Any = None) -> Any:
        """Coerce a stored path into the backend's live handle.

        Default returns the input unchanged — generic backends use
        the path string verbatim. Databricks-style subclasses route
        through :meth:`DatabricksPath.from_` so they get a singleton-
        cached path handle the lifecycle hook can unlink.
        """
        return path

    def _format_path_for_sql(self, path: Any) -> str:
        r"""Render *path* as the SQL fragment inside ``SELECT * FROM <ref>``.

        Default emits ``parquet.\`<path>\``` (Spark / Databricks
        dialect). Subclasses on other backends (DuckDB ``read_parquet``,
        Trino ``hive.<schema>.<table>``, …) override.
        """
        return f"parquet.`{path_for_sql(path)}`"

    def _attach_cleanup_paths(
        self,
        prepared: PreparedStatement,
        *,
        parquet_paths: Sequence[Any],
        metadata_paths: Sequence[Any],
        cleanup: bool,
    ) -> None:
        """Attach staged paths to *prepared* so its lifecycle hook unlinks them.

        Default is a no-op — the generic path is "SQL text references
        the file, executor handles it, caller removes via
        :meth:`cleanup` if anything is left behind". Backend subclasses
        whose :class:`PreparedStatement` has an
        ``external_volume_paths``-like cleanup slot override to attach
        each path with a ``temporary`` flag.
        """
        del prepared, parquet_paths, metadata_paths, cleanup  # default no-op

    @classmethod
    def _resolve_executor(cls, target: Any) -> Any:
        """Coerce *target* into the executor :meth:`execute` will submit on.

        ``AsyncInsert`` itself stays executor-agnostic: the generic
        :meth:`execute` / :meth:`concat` / :class:`AsyncWrite` path
        accepts whatever the caller hands them, and this hook is the
        single coercion point. The base implementation is the
        identity — pass an already-shaped :class:`StatementExecutor`
        and you get it back. Backends where the caller naturally has
        an "engine"-style object that needs an extra step to reach
        the executor (Databricks routes through
        :meth:`SQLEngine.warehouse`) override this to do that step.
        """
        return target

    # ------------------------------------------------------------------ #
    # Init
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        target: "URL | str | None" = None,
        *,
        schema: "Schema | None" = None,
        parquet_paths: Tuple[Any, ...] = (),
        metadata_paths: Tuple[Any, ...] = (),
        operation_ids: Tuple[str, ...] = (),
        created_at: str = "",
        mode: Optional[str] = None,
        schema_mode: Optional[str] = None,
        overwrite_schema: Optional[bool] = None,
        match_by: Optional[Tuple[str, ...]] = None,
        update_column_names: Optional[Tuple[str, ...]] = None,
        zorder_by: Optional[Tuple[str, ...]] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: Optional[int] = None,
        where: Optional[str] = None,
        prune_by: Optional[Tuple[str, ...]] = None,
        prune_values: Optional[Mapping[str, Tuple[Any, ...]]] = None,
        safe_merge: bool = False,
        version: int = METADATA_VERSION,
        executor: Any = None,
        parallel: int = 1,
    ):
        self.parquet_paths = parquet_paths
        self.metadata_paths = metadata_paths
        self.operation_ids = operation_ids
        self.created_at = created_at
        self.mode = mode
        self.schema_mode = schema_mode
        self.overwrite_schema = overwrite_schema
        self.match_by = match_by
        self.update_column_names = update_column_names
        self.zorder_by = zorder_by
        self.optimize_after_merge = optimize_after_merge
        self.vacuum_hours = vacuum_hours
        self.where = where
        self.prune_by = prune_by
        self.prune_values = prune_values
        self.safe_merge = safe_merge
        self.version = version

        # StatementBatch handles ``target`` / ``schema`` coercion and
        # the result-map / schema-cache plumbing. No statements are
        # eagerly submitted: rendering only happens once an executor
        # is bound (via :meth:`execute`) so the metadata-only mode
        # never touches a backend.
        super().__init__(
            executor=executor, statements=None, parallel=parallel,
            target=target, schema=schema,
        )

    # ------------------------------------------------------------------ #
    # Replace / equality / hash / repr
    # ------------------------------------------------------------------ #
    def _replace(self, **changes: Any) -> "AsyncInsert":
        """Return a copy of this record with *changes* applied."""
        kwargs = {name: getattr(self, name) for name, _ in _METADATA_FIELDS}
        kwargs.update(changes)
        return type(self)(**kwargs)

    def __eq__(self, other: object) -> bool:
        if other is self:
            return True
        if not isinstance(other, AsyncInsert):
            return NotImplemented
        for name, _ in _METADATA_FIELDS:
            if getattr(self, name) != getattr(other, name):
                return False
        return True

    def __hash__(self) -> int:
        return hash((self.target_full_name, self.operation_ids))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(target={self.target_full_name!r}, "
            f"ops={len(self.operation_ids)}, mode={self.mode!r})"
        )

    def __getstate__(self) -> dict[str, Any]:
        """Pickle the metadata snapshot — drop the live executor."""
        state = {
            k: v for k, v in self.__dict__.items()
            if k not in self._TRANSIENT_STATE_ATTRS
        }
        state["parquet_paths"] = tuple(path_for_sql(p) for p in self.parquet_paths)
        state["metadata_paths"] = tuple(path_for_sql(p) for p in self.metadata_paths)
        if self.target is not None:
            state["target"] = self.target.to_string()
        if self.schema is not None:
            state["schema"] = self.schema.to_dict()
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Restore *state* and re-init the dropped transient handles."""
        self.__dict__.update(state)
        if isinstance(self.target, str) and self.target:
            self.target = URL.from_(self.target)
        if isinstance(self.schema, Mapping):
            self.schema = Field.from_dict(self.schema)
        self.executor = None
        self.results = OrderedDict()
        self.external_volume_paths = {}
        self._cached_schema = None
        self.start_timestamp = None

    # ---- derived target identity --------------------------------------- #
    # ``target`` (a :class:`URL`) is the single source of truth for the
    # destination table. The legacy ``cat.sch.tbl`` / per-segment
    # strings stay around as derived getters so SQL rendering / logs /
    # grouping don't have to walk URL parts every time.
    @property
    def target_parts(self) -> Tuple[str, ...]:
        return tuple(self.target.parts) if self.target is not None else ()

    @property
    def target_catalog_name(self) -> Optional[str]:
        parts = self.target_parts
        return parts[0] if len(parts) >= 1 else None

    @property
    def target_schema_name(self) -> Optional[str]:
        parts = self.target_parts
        return parts[1] if len(parts) >= 2 else None

    @property
    def target_table_name(self) -> Optional[str]:
        parts = self.target_parts
        return parts[2] if len(parts) >= 3 else None

    @property
    def target_full_name(self) -> str:
        """``catalog.schema.table`` dotted form derived from :attr:`target`."""
        return ".".join(self.target_parts) if self.target is not None else ""

    @property
    def target_field_names(self) -> Optional[Tuple[str, ...]]:
        """Column names derived from :attr:`schema`."""
        return tuple(self.schema.field_names()) if self.schema is not None else None

    @property
    def operation_id(self) -> str:
        """First operation id (the primary one before any merge)."""
        return self.operation_ids[0] if self.operation_ids else ""

    @property
    def is_overwrite(self) -> bool:
        return (self.mode or "").lower() in _OVERWRITE_TOKENS

    @property
    def is_append(self) -> bool:
        mode = (self.mode or "").lower()
        return mode in _APPEND_TOKENS or not self.is_overwrite

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Return a plain-dict view suitable for JSON serialisation."""
        pv = self.prune_values
        mb = self.match_by
        ucn = self.update_column_names
        zb = self.zorder_by
        pb = self.prune_by
        return {
            "target": self.target.to_string() if self.target is not None else None,
            "schema": self.schema.to_dict() if self.schema is not None else None,
            "parquet_paths": [path_for_sql(p) for p in self.parquet_paths],
            "metadata_paths": [path_for_sql(p) for p in self.metadata_paths],
            "operation_ids": list(self.operation_ids),
            "created_at": self.created_at,
            "mode": self.mode,
            "schema_mode": self.schema_mode,
            "overwrite_schema": self.overwrite_schema,
            "match_by": list(mb) if mb else None,
            "update_column_names": list(ucn) if ucn else None,
            "zorder_by": list(zb) if zb else None,
            "optimize_after_merge": self.optimize_after_merge,
            "vacuum_hours": self.vacuum_hours,
            "where": self.where,
            "prune_by": list(pb) if pb else None,
            "prune_values": (
                {k: list(v) for k, v in pv.items()} if pv else None
            ),
            "safe_merge": self.safe_merge,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AsyncInsert":
        """Rebuild an :class:`AsyncInsert` from a JSON-loaded dict."""
        kwargs: dict[str, Any] = {}
        tuple_fields = _TUPLE_FIELD_NAMES
        for key, value in data.items():
            if key not in _FIELD_NAMES:
                continue
            if key in tuple_fields and isinstance(value, list):
                value = tuple(value)
            kwargs[key] = value

        schema_value = kwargs.get("schema")
        if isinstance(schema_value, Mapping):
            kwargs["schema"] = Field.from_dict(schema_value)

        prune_values = kwargs.get("prune_values")
        if isinstance(prune_values, Mapping):
            kwargs["prune_values"] = {
                str(k): tuple(v) if isinstance(v, list) else v
                for k, v in prune_values.items()
            }

        return cls(**kwargs)

    def to_json_bytes(self) -> bytes:
        return ygg_json.dumps(self.to_dict())

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "AsyncInsert":
        return cls.from_dict(ygg_json.loads(data))

    @classmethod
    def from_file(
        cls,
        path: Any,
        *,
        client: Any = None,
    ) -> "AsyncInsert":
        """Read a metadata JSON file and rebuild the :class:`AsyncInsert`.

        Backends that route path strings through their own filesystem
        layer override :meth:`_resolve_path` and may want to override
        this too — the default just calls ``path.read_bytes()`` when
        *path* already exposes that method, else stringifies. The
        Databricks subclass coerces strings through
        :meth:`DatabricksPath.from_` before reading.
        """
        if hasattr(path, "read_bytes"):
            return cls.from_json_bytes(path.read_bytes())
        # Subclasses that need path resolution route through
        # ``_resolve_path`` on an instance — use a sentinel record to
        # reach the hook without polluting the public surface.
        resolved = cls()._resolve_path(path, client=client)
        return cls.from_json_bytes(resolved.read_bytes())

    # ------------------------------------------------------------------ #
    # Merge
    # ------------------------------------------------------------------ #
    def merge_with(self, other: "AsyncInsert") -> "AsyncInsert":
        """Combine two records for the same target into one."""
        if self.target_full_name != other.target_full_name:
            raise ValueError(
                f"Cannot merge AsyncInsert records for different targets "
                f"({self.target_full_name!r} vs {other.target_full_name!r}); "
                "merge groups records by target_full_name before pairwise combine."
            )

        # Pairwise: one comparison beats ``sorted([self, other], key=…)``
        # by ~30% on the per-record hot path :meth:`merge` drives.
        if self.created_at <= other.created_at:
            older, newer = self, other
        else:
            older, newer = other, self

        if newer.is_overwrite:
            return newer._replace(
                metadata_paths=(
                    older.parquet_paths + older.metadata_paths
                    + newer.metadata_paths
                ),
                operation_ids=older.operation_ids + newer.operation_ids,
            )

        merged_mode = older.mode if older.is_overwrite else newer.mode or older.mode

        return older._replace(
            parquet_paths=older.parquet_paths + newer.parquet_paths,
            metadata_paths=older.metadata_paths + newer.metadata_paths,
            operation_ids=older.operation_ids + newer.operation_ids,
            created_at=newer.created_at,
            mode=merged_mode,
            schema_mode=newer.schema_mode or older.schema_mode,
            overwrite_schema=(
                older.overwrite_schema
                if newer.overwrite_schema is None
                else newer.overwrite_schema
            ),
            match_by=newer.match_by or older.match_by,
            update_column_names=newer.update_column_names or older.update_column_names,
            zorder_by=newer.zorder_by or older.zorder_by,
            optimize_after_merge=older.optimize_after_merge or newer.optimize_after_merge,
            vacuum_hours=newer.vacuum_hours or older.vacuum_hours,
            where=newer.where or older.where,
            prune_by=newer.prune_by or older.prune_by,
            prune_values=newer.prune_values or older.prune_values,
            safe_merge=older.safe_merge or newer.safe_merge,
            schema=newer.schema or older.schema,
        )

    @classmethod
    def merge(
        cls,
        source: Any,
        *,
        client: Any = None,
    ) -> List["AsyncInsert"]:
        """Collapse multiple staged operations into one per target."""
        records = list(iter_records(source, cls=cls, client=client))

        groups: dict[str, list[AsyncInsert]] = {}
        for record in records:
            groups.setdefault(record.target_full_name, []).append(record)

        merged: list[AsyncInsert] = []
        for _target, recs in groups.items():
            recs.sort(key=lambda r: r.created_at)
            last_overwrite = max(
                (i for i, r in enumerate(recs) if r.is_overwrite),
                default=-1,
            )
            kept = recs[last_overwrite:] if last_overwrite >= 0 else recs

            head = kept[0]
            for r in kept[1:]:
                head = head.merge_with(r)

            if last_overwrite > 0:
                dropped = recs[:last_overwrite]
                dropped_cleanup = (
                    tuple(p for r in dropped for p in r.parquet_paths)
                    + tuple(p for r in dropped for p in r.metadata_paths)
                )
                head = head._replace(
                    metadata_paths=dropped_cleanup + head.metadata_paths,
                    operation_ids=(
                        tuple(o for r in dropped for o in r.operation_ids)
                        + head.operation_ids
                    ),
                )
            merged.append(head)

        return merged

    # ------------------------------------------------------------------ #
    # SQL rendering
    # ------------------------------------------------------------------ #
    def _build_sql(self, parquet_refs: Sequence[str]) -> Optional[str]:
        r"""Common SQL shape used by both raw-text and prepared paths.

        Returns ``None`` when there is nothing to apply (no Parquet
        payloads or no target). Each entry in ``parquet_refs`` is the
        full table expression spliced into ``SELECT * FROM <ref>`` —
        either the result of :meth:`_format_path_for_sql` for raw SQL
        or a bare ``{alias}`` placeholder whose substituted
        ``text_value`` already carries that form. Wrapping is the
        caller's job so the prepared-statement path doesn't end up
        double-wrapping.
        """
        if not parquet_refs or not self.target_full_name:
            return None

        selects = [f"SELECT * FROM {ref}" for ref in parquet_refs]
        source = " UNION ALL ".join(selects)

        target = self.target_full_name
        field_names = self.target_field_names
        if field_names:
            cols = ", ".join(f"`{c}`" for c in field_names)
            target = f"{target} ({cols})"

        prefix = (
            f"INSERT OVERWRITE {target}"
            if self.is_overwrite
            else f"INSERT INTO {target}"
        )
        where = f" WHERE {self.where}" if self.where else ""
        return f"{prefix} {source}{where}"

    def to_sql(self) -> List[str]:
        """Render the operation as one or more raw SQL strings.

        Emits a single statement: ``INSERT INTO`` (append) or
        ``INSERT OVERWRITE`` (overwrite) whose source is the staged
        Parquet payloads read via :meth:`_format_path_for_sql`.
        Returns an empty list when no Parquet payloads are recorded.
        """
        refs = tuple(self._format_path_for_sql(p) for p in self.parquet_paths)
        sql = self._build_sql(refs)
        return [sql] if sql is not None else []

    def to_statements(
        self,
        *,
        client: Any = None,
        retry: Any = None,
        cleanup: bool = True,
    ) -> List[PreparedStatement]:
        """Render the operation as a list of :class:`PreparedStatement`.

        One statement is emitted per record (empty payload returns
        ``[]``). Parquet payloads ride as ``{__pN__}`` aliases in the
        SQL text; the substituted ``text_value`` is what
        :meth:`_format_path_for_sql` returns. Metadata files (which
        never appear in SQL) are handed to :meth:`_attach_cleanup_paths`
        so the backend can wire them into the statement lifecycle.

        Pass ``cleanup=False`` to attach the same paths without
        marking them temporary.
        """
        if not self.parquet_paths:
            return []

        # Resolve every path through the backend hook once so the
        # cleanup hook + SQL formatter both see the same handle.
        resolved_parquets = [
            self._resolve_path(p, client=client) for p in self.parquet_paths
        ]
        resolved_metas = [
            self._resolve_path(p, client=client) for p in self.metadata_paths
        ]

        # Build SQL with the SQL fragments already substituted. The
        # backend ``_format_path_for_sql`` decides the dialect
        # (``parquet.\`<path>\`` for Spark/Databricks, ``read_parquet(...)``
        # for DuckDB, …).
        refs = tuple(self._format_path_for_sql(p) for p in resolved_parquets)
        sql = self._build_sql(refs)
        if sql is None:
            return []

        stmt = self._build_prepared_statement(
            sql, retry=retry, client=client,
        )
        # Surface the unified target/schema on the statement too so
        # the executor can read them without dereferencing the batch.
        if stmt.target is None:
            stmt.target = self.target
        if stmt.schema is None:
            stmt.schema = self.schema

        # Backend cleanup hook — default no-op; Databricks-style
        # backends bind ``external_volume_paths`` so the statement
        # lifecycle unlinks the staged files on success.
        self._attach_cleanup_paths(
            stmt,
            parquet_paths=resolved_parquets,
            metadata_paths=resolved_metas,
            cleanup=cleanup,
        )
        return [stmt]

    def _build_prepared_statement(
        self,
        sql: str,
        *,
        retry: Any = None,
        client: Any = None,
    ) -> PreparedStatement:
        """Mint the :class:`PreparedStatement` for *sql*.

        Default goes through :meth:`PreparedStatement.prepare`, which
        on the base class just coerces the text. Subclasses with a
        richer ``prepare`` signature (e.g. Databricks's
        :meth:`WarehousePreparedStatement.prepare` that wants
        ``catalog_name`` / ``schema_name`` / ``client`` /
        ``external_volume_paths``) override this to thread the
        backend-specific kwargs through.
        """
        del client  # unused at the generic level
        stmt = self._PREPARED_CLASS.prepare(sql)
        if retry is not None:
            stmt = stmt.with_retry(retry)
        return stmt

    # ------------------------------------------------------------------ #
    # Concat — render a batch-execution suite across many records
    # ------------------------------------------------------------------ #
    @classmethod
    def concat(
        cls,
        source: Any,
        *,
        executor: Any = None,
        client: Any = None,
        wait: Any = True,
        raise_error: bool = True,
        cleanup: bool = True,
    ) -> Any:
        """Concatenate staged operations into the SQL suite that applies them.

        With *executor* supplied: every merged record is wrapped in
        one unified :class:`AsyncWrite` batch. Without *executor*:
        returns the bare list of SQL strings.

        *executor* is routed through :meth:`_resolve_executor` so a
        backend whose caller has an engine-style object (Databricks
        ``SQLEngine``) can coerce it to the actual executor in one
        place.
        """
        merged = cls.merge(source, client=client)
        if not merged:
            return [] if executor is None else None

        if executor is None:
            statements: list[str] = []
            for record in merged:
                statements.extend(record.to_sql())
            return statements

        return AsyncWrite.from_records(
            merged,
            executor=cls._resolve_executor(executor),
            client=client,
            cleanup=cleanup,
            wait=wait,
            raise_error=raise_error,
        )

    def __call__(
        self,
        executor: Any = None,
        *others: "AsyncInsert",
        client: Any = None,
        wait: Any = True,
        raise_error: bool = True,
        cleanup: bool = True,
    ) -> Any:
        """Shorthand for :meth:`concat` keyed off this record."""
        return type(self).concat(
            [self, *others],
            executor=executor,
            client=client,
            wait=wait,
            raise_error=raise_error,
            cleanup=cleanup,
        )

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def execute(
        self,
        executor: Any,
        *,
        wait: Any = True,
        raise_error: bool = True,
        cleanup: bool = True,
        client: Any = None,
    ) -> "AsyncInsert | None":
        """Submit this record's prepared statements through *executor*.

        Binds *executor* (after routing through :meth:`_resolve_executor`
        so backends can map an "engine"-style handle to its actual
        executor) as :attr:`executor`, renders the record via
        :meth:`to_statements`, and extends self with the resulting
        prepared statements (which submits them).

        Empty operations (no Parquet paths) return ``None`` without
        touching *executor*.
        """
        if not self.parquet_paths or not self.target_full_name:
            return None

        statements = self.to_statements(client=client, cleanup=cleanup)
        if not statements:
            return None

        self.executor = type(self)._resolve_executor(executor)
        self.extend(statements)
        self.wait(wait=wait, raise_error=raise_error)
        return self

    def cleanup(self, *, client: Any = None) -> None:
        """Force-remove every staged Parquet + metadata file recorded on this op.

        Best-effort: missing files are tolerated, individual failures
        are logged and swallowed.
        """
        for full_path in tuple(self.parquet_paths) + tuple(self.metadata_paths):
            if not full_path:
                continue
            try:
                resolved = self._resolve_path(full_path, client=client)
                remove = getattr(resolved, "remove", None)
                if callable(remove):
                    remove(missing_ok=True, wait=False, recursive=False)
            except Exception:
                LOGGER.exception(
                    "Failed to clean up staged async-insert artifact %r; continuing.",
                    full_path,
                )


# ---------------------------------------------------------------------------
# AsyncWrite — unified StatementBatch over staged AsyncInsert records
# ---------------------------------------------------------------------------


class AsyncWrite:
    """Unified :class:`StatementBatch` factory over staged async inserts.

    The single apply path for :class:`AsyncInsert` records. Submits
    one prepared statement per merged target as one batch.

    Construct via :meth:`from_records` (in-memory records) or
    :meth:`from_source` (anything :meth:`AsyncInsert.merge` accepts).
    Each classmethod returns a submitted batch.
    """

    # Subclasses can override to point at their backend-specific
    # :class:`StatementBatch` subclass (the Databricks subclass
    # routes through :class:`WarehouseStatementBatch`).
    _BATCH_CLASS: ClassVar[type[StatementBatch]] = StatementBatch
    _RECORD_CLASS: ClassVar[type[AsyncInsert]] = AsyncInsert

    def __new__(cls, *args: Any, **kwargs: Any) -> StatementBatch:
        if "source" in kwargs:
            return cls.from_source(*args, **kwargs)
        return cls.from_records(*args, **kwargs)

    @classmethod
    def from_records(
        cls,
        records: Iterable[AsyncInsert],
        *,
        executor: Any,
        client: Any = None,
        cleanup: bool = True,
        retry: Any = None,
        parallel: int = 1,
        wait: Any = True,
        raise_error: bool = True,
    ) -> Any:
        """Build, submit, and (by default) wait on the batch for *records*."""
        statements: list[Any] = []
        record_count = 0
        for record in records:
            record_count += 1
            statements.extend(
                record.to_statements(
                    client=client,
                    cleanup=cleanup,
                    retry=retry,
                )
            )

        if not statements:
            return None

        LOGGER.info(
            "Applying %d-statement async-insert batch across %d target(s)",
            len(statements), record_count,
        )

        batch = cls._BATCH_CLASS(
            executor=executor,
            statements=statements,
            parallel=parallel,
        )
        batch.wait(wait=wait, raise_error=raise_error)
        return batch

    @classmethod
    def from_source(
        cls,
        source: Any,
        *,
        executor: Any = None,
        client: Any = None,
        cleanup: bool = True,
        retry: Any = None,
        parallel: int = 1,
        wait: Any = True,
        raise_error: bool = True,
    ) -> Any:
        """Merge *source* through :meth:`AsyncInsert.merge`, then apply.

        *executor* is required and routed through
        :meth:`AsyncInsert._resolve_executor` so backends that hand
        a higher-level engine object still resolve to the actual
        executor in one place.
        """
        if executor is None:
            raise ValueError(
                "AsyncWrite.from_source needs an ``executor`` "
                "(or an engine-shaped object the record class's "
                "``_resolve_executor`` knows how to coerce)."
            )
        executor = cls._RECORD_CLASS._resolve_executor(executor)

        merged = cls._RECORD_CLASS.merge(source, client=client)
        if not merged:
            return None

        return cls.from_records(
            merged,
            executor=executor,
            client=client,
            cleanup=cleanup,
            retry=retry,
            parallel=parallel,
            wait=wait,
            raise_error=raise_error,
        )


# ---------------------------------------------------------------------------
# Iter records helper
# ---------------------------------------------------------------------------


def iter_records(
    source: Any,
    *,
    cls: type[AsyncInsert] = AsyncInsert,
    client: Any = None,
) -> Iterable[AsyncInsert]:
    """Yield :class:`AsyncInsert` records from a folder / iterable / records.

    Centralised so :meth:`AsyncInsert.merge` accepts the same shapes
    callers naturally have on hand (a folder path, a list of metadata
    files, a list of already-loaded records).

    Folder-like sources are walked for ``*.json`` entries; a
    ``logs/`` subdirectory is descended into automatically so
    callers can hand in either the staging root
    (``.sql/async/insert/``) or the explicit logs folder
    (``.sql/async/insert/logs/``) and get the same records back.
    """
    if isinstance(source, AsyncInsert):
        yield source
        return

    if hasattr(source, "ls"):
        source_name = getattr(source, "name", "") or ""
        if source_name == "insert" and hasattr(source, "joinpath"):
            try:
                logs_folder = source.joinpath(ASYNC_INSERT_LOGS_SUBDIR)
                entries = logs_folder.ls(recursive=False)
            except FileNotFoundError:
                return
            for entry in entries:
                if (getattr(entry, "name", "") or "").endswith(".json"):
                    yield cls.from_file(entry, client=client)
            return

        for entry in source.ls(recursive=False):
            name = getattr(entry, "name", "") or ""
            if name == ASYNC_INSERT_LOGS_SUBDIR and hasattr(entry, "ls"):
                yield from iter_records(entry, cls=cls, client=client)
                continue
            if name.endswith(".json"):
                yield cls.from_file(entry, client=client)
        return

    if isinstance(source, str):
        if source.endswith(".json"):
            yield cls.from_file(source, client=client)
            return
        # Subclasses route through their own path layer.
        resolved = cls()._resolve_path(source, client=client)
        if resolved is not source:
            yield from iter_records(resolved, cls=cls, client=client)
        return

    try:
        items = iter(source)
    except TypeError as exc:
        raise TypeError(
            f"AsyncInsert.merge cannot iterate {source!r} "
            f"(expected folder path, path string, or iterable of "
            f"metadata files / AsyncInsert records)."
        ) from exc

    for item in items:
        yield from iter_records(item, cls=cls, client=client)


