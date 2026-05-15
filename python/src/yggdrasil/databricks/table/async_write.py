"""Stage table inserts as Parquet + metadata for asynchronous execution.

When :meth:`Table.insert` is called with ``async_write=True``, the
caller is not waiting on a warehouse round trip — the rows are cast
to the target schema, written as Parquet next to the table's
``stg_<table>/.sql/async/insert`` staging folder, and a sibling JSON
file carries an :class:`AsyncInsert` record so a downstream applier
(typically a job-driven loop) can replay the operation against the
target table when it's convenient.

File layout under the table's async staging folder::

    .sql/async/insert/
        async-<epoch_ms>-<seed>.parquet   # rows, cast to target schema
        async-<epoch_ms>-<seed>.json      # operation metadata (orjson)

This module lives outside ``table.py`` so the latter doesn't pick up
async-specific helpers; :meth:`Table.insert` delegates to
:func:`stage_async_insert` only when ``async_write=True``.

Multiple staged operations against the same table can be folded into
one logical insert via :meth:`AsyncInsert.merge` — pure appends fold
together, an overwrite drops every earlier op for that target, and
the resulting record renders to a single ``INSERT INTO`` / ``INSERT
OVERWRITE`` statement via :meth:`AsyncInsert.to_sql`.
:meth:`AsyncInsert.execute` runs the SQL and cleans up the staged
files on success.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import time
from dataclasses import asdict, dataclass, fields, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from yggdrasil.data.enums import Mode
from yggdrasil.data.options import CastOptions
from yggdrasil.io.tabular.execution.expr import Predicate
from yggdrasil.pickle import json as ygg_json

if TYPE_CHECKING:
    from databricks.sdk.service.jobs import CronSchedule

    from yggdrasil.databricks.client import DatabricksClient
    from yggdrasil.databricks.fs import VolumePath
    from yggdrasil.databricks.jobs.job import Job
    from yggdrasil.databricks.jobs.service import Jobs
    from yggdrasil.databricks.sql.engine import SQLEngine
    from .table import Table


__all__ = [
    "AsyncInsert",
    "METADATA_VERSION",
    "stage_async_insert",
]


LOGGER = logging.getLogger(__name__)

# Wire-format version. Bump when the JSON schema changes in a way that
# breaks existing appliers so the consumer can fail loudly instead of
# mis-applying an operation.
METADATA_VERSION = 1

# Mode tokens, normalized to lowercase. Anything else is treated as an
# append unless explicitly listed.
_APPEND_TOKENS: frozenset[str] = frozenset({"append", "", "auto", "insert"})
_OVERWRITE_TOKENS: frozenset[str] = frozenset({"overwrite"})


# ---------------------------------------------------------------------------
# AsyncInsert dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AsyncInsert:
    """Frozen, JSON-serialisable description of a deferred table insert.

    Records a list of Parquet payloads (one per merged operation, all
    rooted under the same target's async staging folder) plus the
    insert-time parameters needed to replay the operation against the
    target table.

    A single record may carry more than one Parquet path after
    :meth:`merge` collapses several staged operations into one logical
    insert. Each contributing operation's id and metadata-file path
    are preserved on the record so :meth:`cleanup` can remove the
    whole set after :meth:`execute` succeeds.
    """

    # ---- identity ---------------------------------------------------------
    target_full_name: str
    parquet_paths: Tuple[str, ...] = ()
    metadata_paths: Tuple[str, ...] = ()
    operation_ids: Tuple[str, ...] = ()
    created_at: str = ""

    # ---- target detail ----------------------------------------------------
    target_catalog_name: Optional[str] = None
    target_schema_name: Optional[str] = None
    target_table_name: Optional[str] = None
    target_field_names: Optional[Tuple[str, ...]] = None

    # ---- insert spec ------------------------------------------------------
    mode: Optional[str] = None
    schema_mode: Optional[str] = None
    overwrite_schema: Optional[bool] = None
    match_by: Optional[Tuple[str, ...]] = None
    update_column_names: Optional[Tuple[str, ...]] = None
    zorder_by: Optional[Tuple[str, ...]] = None
    optimize_after_merge: bool = False
    vacuum_hours: Optional[int] = None
    where: Optional[str] = None
    prune_by: Optional[Tuple[str, ...]] = None
    prune_values: Optional[Mapping[str, Tuple[Any, ...]]] = None
    safe_merge: bool = False

    version: int = METADATA_VERSION

    # ---- derived ---------------------------------------------------------
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
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Return a plain-dict view suitable for JSON serialisation.

        Tuples are emitted as lists so a downstream reader doesn't need
        to import this module to read the file. The ``prune_values``
        mapping is shallow-copied with list-of-values entries.
        """
        out = asdict(self)
        for key in (
            "parquet_paths", "metadata_paths", "operation_ids",
            "target_field_names", "match_by", "update_column_names",
            "zorder_by", "prune_by",
        ):
            value = out.get(key)
            if value is not None:
                out[key] = list(value)
        if out.get("prune_values"):
            out["prune_values"] = {
                k: list(v) for k, v in out["prune_values"].items()
            }
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AsyncInsert":
        """Rebuild an :class:`AsyncInsert` from a JSON-loaded dict."""
        allowed = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key not in allowed:
                continue
            kwargs[key] = value

        # Lists → tuples for the tuple fields so the dataclass stays
        # hashable-shaped and frozen.
        for key in (
            "parquet_paths", "metadata_paths", "operation_ids",
            "target_field_names", "match_by", "update_column_names",
            "zorder_by", "prune_by",
        ):
            if isinstance(kwargs.get(key), list):
                kwargs[key] = tuple(kwargs[key])

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
        client: "DatabricksClient | None" = None,
    ) -> "AsyncInsert":
        """Read a metadata JSON file and rebuild the :class:`AsyncInsert`."""
        from yggdrasil.databricks.path import DatabricksPath

        if not hasattr(path, "read_bytes"):
            path = DatabricksPath.from_(path, client=client)
        return cls.from_json_bytes(path.read_bytes())

    # ------------------------------------------------------------------ #
    # Merge
    # ------------------------------------------------------------------ #
    def merge_with(self, other: "AsyncInsert") -> "AsyncInsert":
        """Combine two records for the same target into one.

        Both records must share ``target_full_name``. Semantics:

        - Both append → parquet paths are concatenated, the rest of the
          spec inherits the older record with newer non-``None`` values
          overriding (``match_by``, ``zorder_by``, …).
        - One overwrite (and it's the newer one) → drops the older
          record entirely; the newer overwrite stands alone.
        - One overwrite (and it's the older one) → the newer append's
          rows are folded into the overwrite scope so a single
          ``INSERT OVERWRITE`` carries every staged row.
        - Both overwrite → take the newer overwrite.
        """
        if self.target_full_name != other.target_full_name:
            raise ValueError(
                f"Cannot merge AsyncInsert records for different targets "
                f"({self.target_full_name!r} vs {other.target_full_name!r}); "
                "merge groups records by target_full_name before pairwise combine."
            )

        older, newer = sorted([self, other], key=lambda r: r.created_at)

        # Newer overwrite wins outright: the older record's Parquet
        # data is dropped from the SQL projection (parquet_paths stays
        # as the newer's set), but the older record's Parquet file +
        # metadata still need to be cleaned up after execute. Pack
        # them into ``_extra_cleanup_paths`` via the metadata_paths
        # list — they're listed in the cleanup walk anyway.
        if newer.is_overwrite:
            return replace(
                newer,
                # ``parquet_paths`` stays the newer's set — older's
                # Parquet is dropped from the SQL projection. We still
                # need to clean it up; record it on metadata_paths so
                # ``cleanup()`` removes it during the same sweep.
                metadata_paths=(
                    older.parquet_paths + older.metadata_paths
                    + newer.metadata_paths
                ),
                operation_ids=older.operation_ids + newer.operation_ids,
            )

        # All other cases (both append, or older overwrite + newer
        # append): combine parquets into one record. When the older is
        # an overwrite, the merged record stays an overwrite (the newer
        # append rows are pulled into the overwrite scope).
        merged_mode = older.mode if older.is_overwrite else newer.mode or older.mode

        return replace(
            older,
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
            target_field_names=newer.target_field_names or older.target_field_names,
        )

    @classmethod
    def merge(
        cls,
        source: Any,
        *,
        client: "DatabricksClient | None" = None,
    ) -> List["AsyncInsert"]:
        """Collapse multiple staged operations into one per target.

        ``source`` may be:

        - A folder-like :class:`VolumePath` (or path string) — every
          ``*.json`` entry under it is read and merged.
        - An iterable of metadata file paths (or path strings).
        - An iterable of already-loaded :class:`AsyncInsert` records.

        Returns one merged record per unique target. Within each
        target, an overwrite drops every earlier operation; the
        remaining records (appends and at most one trailing overwrite)
        fold into a single record via :meth:`merge_with`.
        """
        records = list(_iter_records(source, client=client))

        groups: dict[str, list[AsyncInsert]] = {}
        for record in records:
            groups.setdefault(record.target_full_name, []).append(record)

        merged: list[AsyncInsert] = []
        for target, recs in groups.items():
            recs.sort(key=lambda r: r.created_at)
            # The latest overwrite wipes everything before it.
            last_overwrite = max(
                (i for i, r in enumerate(recs) if r.is_overwrite),
                default=-1,
            )
            kept = recs[last_overwrite:] if last_overwrite >= 0 else recs

            head = kept[0]
            for r in kept[1:]:
                head = head.merge_with(r)

            # When earlier records were dropped by an overwrite, their
            # data is NOT part of the merged SQL projection — but their
            # staged files still have to be removed. Pile both the
            # dropped parquet and metadata paths onto ``metadata_paths``
            # so :meth:`cleanup` removes them during the same sweep.
            if last_overwrite > 0:
                dropped = recs[:last_overwrite]
                dropped_cleanup = (
                    tuple(p for r in dropped for p in r.parquet_paths)
                    + tuple(p for r in dropped for p in r.metadata_paths)
                )
                head = replace(
                    head,
                    metadata_paths=dropped_cleanup + head.metadata_paths,
                    operation_ids=(
                        tuple(o for r in dropped for o in r.operation_ids)
                        + head.operation_ids
                    ),
                )
            merged.append(head)

        return merged

    # ------------------------------------------------------------------ #
    # Job management — one Databricks Job per (catalog, schema)
    # ------------------------------------------------------------------ #
    @staticmethod
    def resolve_schema_key(target: Any) -> Tuple[str, str]:
        """Return ``(catalog_name, schema_name)`` for *target*.

        Accepts a :class:`Table`, a :class:`Schema`, a ``"cat.sch"`` /
        ``"cat.sch.tbl"`` string, or any object exposing
        ``catalog_name`` / ``schema_name`` attributes. Raises
        :class:`ValueError` when the pair can't be resolved — the
        applier job is keyed off the schema, so neither half can be
        missing.
        """
        if isinstance(target, str):
            parts = [p for p in target.split(".") if p]
            if len(parts) < 2:
                raise ValueError(
                    f"Cannot derive (catalog, schema) from {target!r}; "
                    "pass ``cat.sch`` or ``cat.sch.tbl``."
                )
            return parts[0], parts[1]

        catalog = getattr(target, "catalog_name", None)
        schema = getattr(target, "schema_name", None)
        if not catalog or not schema:
            raise ValueError(
                f"Cannot derive (catalog, schema) from {target!r}: "
                "object exposes neither (catalog_name, schema_name) nor a "
                "parseable ``cat.sch[.tbl]`` string."
            )
        return catalog, schema

    @staticmethod
    def default_job_name(target: Any) -> str:
        """Return the canonical applier-job name for *target*'s schema.

        Identity is keyed off ``(catalog, schema)`` — every table in
        the same schema shares one applier job. Accepts a
        :class:`Table`, a :class:`Schema`, a ``"cat.sch"`` /
        ``"cat.sch.tbl"`` string, or any object exposing
        ``catalog_name`` / ``schema_name`` attributes.
        """
        catalog, schema = AsyncInsert.resolve_schema_key(target)
        return f"ygg-async-insert-{catalog}-{schema}"

    @classmethod
    def ensure_job(
        cls,
        target: Any = None,
        *,
        table: "Table | None" = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        jobs: "Jobs | None" = None,
        client: "DatabricksClient | None" = None,
        job_name: str | None = None,
        task: Any = None,
        notebook_path: str | None = None,
        notebook_warehouse_id: str | None = None,
        notebook_base_parameters: Optional[Mapping[str, str]] = None,
        schedule: Any = None,
        schedule_timezone: str = "UTC",
        schedule_pause_status: Any = None,
        parameters: Optional[Mapping[str, str]] = None,
        description: str | None = None,
        permissions: Optional[List[Any]] = None,
        tags: Optional[Mapping[str, str]] = None,
        **settings: Any,
    ) -> "Job":
        """Find-or-create the Databricks Job that applies async inserts for a schema.

        Resolution
        ----------
        Identity is keyed off ``(catalog_name, schema_name)`` — every
        table in the same schema shares one applier job (named
        ``ygg-async-insert-<catalog>-<schema>``). The schema can be
        derived from any of these arguments (any one is enough):

        - ``target`` — a :class:`Table`, :class:`Schema`, or
          ``"cat.sch[.tbl]"`` string.
        - ``table`` — a :class:`Table` handle.
        - ``catalog_name`` + ``schema_name`` — explicit.

        Tasks
        -----
        Provide either:

        - ``task`` — a :class:`Task` (or list of) used verbatim.
        - ``notebook_path`` — wrapped in a default ``apply`` task with
          ``catalog_name`` / ``schema_name`` in ``base_parameters`` so
          the notebook can drain *every* table in the schema via
          :func:`apply_schema_async_inserts`.

        When neither is supplied the job is still ensured (empty task
        list), and a warning is logged — useful for callers that wire
        the task definition separately.

        Scheduling
        ----------
        ``schedule`` accepts a :class:`CronSchedule` directly or a
        Quartz-style cron string (e.g. ``"0 0 */1 * * ?"``). Passing
        ``None`` leaves the job unscheduled (trigger via
        :meth:`Job.run`).
        """
        # Resolve schema identity --------------------------------------
        resolved_catalog, resolved_schema = cls._resolve_catalog_schema(
            target=target,
            table=table,
            catalog_name=catalog_name,
            schema_name=schema_name,
        )

        # Resolve service ----------------------------------------------
        if jobs is None:
            resolved_client = client or _client_from_target(target, table=table)
            if resolved_client is None:
                raise ValueError(
                    "AsyncInsert.ensure_job needs a ``jobs`` service, a "
                    "``client``, or a ``table`` whose client exposes ``jobs``."
                )
            jobs = resolved_client.jobs

        if not job_name:
            job_name = f"ygg-async-insert-{resolved_catalog}-{resolved_schema}"

        # Tasks ---------------------------------------------------------
        resolved_tasks = cls._resolve_tasks(
            task=task,
            notebook_path=notebook_path,
            notebook_warehouse_id=notebook_warehouse_id,
            notebook_base_parameters=notebook_base_parameters,
            catalog_name=resolved_catalog,
            schema_name=resolved_schema,
        )

        # Schedule ------------------------------------------------------
        cron_schedule = cls._resolve_schedule(
            schedule=schedule,
            timezone_id=schedule_timezone,
            pause_status=schedule_pause_status,
        )

        # Job parameters carry the schema identity by default ----------
        from databricks.sdk.service.jobs import JobParameterDefinition

        job_params: list[Any] = [
            JobParameterDefinition(name="catalog_name", default=resolved_catalog),
            JobParameterDefinition(name="schema_name", default=resolved_schema),
        ]
        if parameters:
            existing = {p.name: p for p in job_params}
            for k, v in parameters.items():
                if k in existing:
                    existing[k].default = str(v)
                else:
                    job_params.append(
                        JobParameterDefinition(name=str(k), default=str(v))
                    )

        if description is None:
            description = (
                f"Apply async inserts staged across tables in "
                f"{resolved_catalog}.{resolved_schema}"
            )

        LOGGER.info(
            "Ensuring async-insert job %r for schema %s.%s (schedule=%r tasks=%d)",
            job_name, resolved_catalog, resolved_schema,
            cron_schedule.quartz_cron_expression if cron_schedule else None,
            len(resolved_tasks),
        )

        return jobs.create_or_update(
            name=job_name,
            tasks=resolved_tasks,
            schedule=cron_schedule,
            parameters=job_params,
            description=description,
            permissions=permissions,
            tags=dict(tags) if tags else None,
            **settings,
        )

    @staticmethod
    def _resolve_catalog_schema(
        *,
        target: Any,
        table: "Table | None",
        catalog_name: str | None,
        schema_name: str | None,
    ) -> Tuple[str, str]:
        """Centralised (catalog, schema) resolution for :meth:`ensure_job`."""
        if catalog_name and schema_name:
            return catalog_name, schema_name

        if table is not None:
            return AsyncInsert.resolve_schema_key(table)

        if target is not None:
            if isinstance(target, AsyncInsert):
                if target.target_catalog_name and target.target_schema_name:
                    return target.target_catalog_name, target.target_schema_name
                return AsyncInsert.resolve_schema_key(target.target_full_name)
            return AsyncInsert.resolve_schema_key(target)

        raise ValueError(
            "AsyncInsert.ensure_job needs one of: ``target``, ``table``, or "
            "explicit ``catalog_name`` + ``schema_name``."
        )

    @staticmethod
    def _resolve_tasks(
        *,
        task: Any,
        notebook_path: str | None,
        notebook_warehouse_id: str | None,
        notebook_base_parameters: Optional[Mapping[str, str]],
        catalog_name: str,
        schema_name: str,
    ) -> List[Any]:
        """Normalize the caller's task spec into a list of :class:`Task`."""
        from databricks.sdk.service.jobs import NotebookTask, Task

        if task is not None:
            if isinstance(task, Task):
                return [task]
            return list(task)

        if notebook_path:
            base_params: dict[str, str] = {
                "catalog_name": catalog_name,
                "schema_name": schema_name,
            }
            if notebook_base_parameters:
                base_params.update(
                    {str(k): str(v) for k, v in notebook_base_parameters.items()}
                )

            return [
                Task(
                    task_key="apply",
                    notebook_task=NotebookTask(
                        notebook_path=notebook_path,
                        warehouse_id=notebook_warehouse_id,
                        base_parameters=base_params,
                    ),
                )
            ]

        LOGGER.warning(
            "AsyncInsert.ensure_job called without ``task`` or ``notebook_path`` — "
            "the resulting job for %s.%s will have no tasks. Attach tasks later "
            "via ``jobs.create_or_update(name=..., tasks=[...])``.",
            catalog_name, schema_name,
        )
        return []

    @staticmethod
    def _resolve_schedule(
        *,
        schedule: Any,
        timezone_id: str,
        pause_status: Any,
    ) -> "CronSchedule | None":
        """Coerce *schedule* into a :class:`CronSchedule` (or ``None``)."""
        if schedule is None:
            return None

        from databricks.sdk.service.jobs import CronSchedule, PauseStatus

        if isinstance(schedule, CronSchedule):
            return schedule

        if isinstance(schedule, str):
            resolved_pause: Any = pause_status
            if isinstance(resolved_pause, str):
                resolved_pause = PauseStatus(resolved_pause.upper())
            return CronSchedule(
                quartz_cron_expression=schedule,
                timezone_id=timezone_id,
                pause_status=resolved_pause,
            )

        raise TypeError(
            f"AsyncInsert.ensure_job: ``schedule`` must be a CronSchedule, a "
            f"Quartz cron string, or None — got {type(schedule).__name__}."
        )

    def ensure_applier_job(
        self,
        *,
        jobs: "Jobs | None" = None,
        client: "DatabricksClient | None" = None,
        **kwargs: Any,
    ) -> "Job":
        """Instance-level shortcut for :meth:`ensure_job` keyed off this record.

        Uses the record's schema (``target_catalog_name`` /
        ``target_schema_name``) — every record for tables in the same
        schema lands on the same applier job.
        """
        return type(self).ensure_job(
            target=self,
            jobs=jobs,
            client=client,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # Schema-wide discovery + apply (used by the applier task)
    # ------------------------------------------------------------------ #
    @classmethod
    def iter_schema_staging_folders(
        cls,
        catalog_name: str,
        schema_name: str,
        *,
        client: "DatabricksClient | None" = None,
    ) -> Iterable["VolumePath"]:
        """Yield every ``.sql/async/insert`` folder under *schema*'s
        ``stg_*`` volumes.

        These are the folders :func:`stage_async_insert` writes into;
        the applier task walks them to discover staged metadata files.
        """
        client = client or _resolve_current_client()
        volumes = client.volumes.list(
            catalog_name=catalog_name, schema_name=schema_name,
        )
        for volume in volumes:
            name = getattr(volume, "volume_name", "") or ""
            if not name.startswith("stg_"):
                continue
            yield volume.path(".sql/async/insert")

    @classmethod
    def merge_schema(
        cls,
        catalog_name: str,
        schema_name: str,
        *,
        client: "DatabricksClient | None" = None,
    ) -> List["AsyncInsert"]:
        """Discover + merge every staged metadata file under *schema*.

        One :class:`AsyncInsert` is returned per target table found.
        Folders without metadata files (or that don't exist yet) are
        skipped silently.
        """
        client = client or _resolve_current_client()

        records: list[AsyncInsert] = []
        for folder in cls.iter_schema_staging_folders(
            catalog_name, schema_name, client=client,
        ):
            try:
                records.extend(_iter_records(folder, client=client))
            except FileNotFoundError:
                continue
            except Exception:
                LOGGER.exception(
                    "Failed to read async-insert metadata under %r; skipping.",
                    folder,
                )
        if not records:
            return []
        return cls.merge(records, client=client)

    @classmethod
    def apply_schema(
        cls,
        engine: "SQLEngine",
        catalog_name: str,
        schema_name: str,
        *,
        client: "DatabricksClient | None" = None,
        wait: Any = True,
        raise_error: bool = True,
        cleanup: bool = True,
    ) -> List[Any]:
        """Merge + execute every staged async insert under *schema*.

        The single entry point for a schema-level applier task — drain
        every table's staging folder, run the merged inserts, and
        clean up the staged files. Returns the list of per-target
        execute results in the order targets were drained.
        """
        records = cls.merge_schema(catalog_name, schema_name, client=client)
        if not records:
            return []

        results: list[Any] = []
        for record in records:
            results.append(
                record.execute(
                    engine,
                    wait=wait,
                    raise_error=raise_error,
                    cleanup=cleanup,
                    client=client,
                )
            )
        return results

    # ------------------------------------------------------------------ #
    # SQL rendering
    # ------------------------------------------------------------------ #
    def to_sql(self) -> List[str]:
        r"""Render the operation as one or more SQL statements.

        The current implementation emits a single statement: an
        ``INSERT INTO`` (append) or ``INSERT OVERWRITE`` (overwrite)
        whose source is the staged Parquet payloads read via
        ``parquet.\`<path>\``. Multiple paths are unioned with
        ``UNION ALL``. Returns an empty list when no Parquet payloads
        are recorded.
        """
        if not self.parquet_paths or not self.target_full_name:
            return []

        selects = [
            f"SELECT * FROM parquet.`{path}`"
            for path in self.parquet_paths
        ]
        source = " UNION ALL ".join(selects)

        target = self.target_full_name
        if self.target_field_names:
            cols = ", ".join(f"`{c}`" for c in self.target_field_names)
            target = f"{target} ({cols})"

        if self.is_overwrite:
            prefix = f"INSERT OVERWRITE {target}"
        else:
            prefix = f"INSERT INTO {target}"

        where = f" WHERE {self.where}" if self.where else ""
        return [f"{prefix} {source}{where}"]

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def execute(
        self,
        engine: "SQLEngine",
        *,
        wait: Any = True,
        raise_error: bool = True,
        cleanup: bool = True,
        client: "DatabricksClient | None" = None,
    ) -> Any:
        """Run the rendered SQL against *engine*; clean up staged files on success.

        Returns the last :class:`StatementResult` produced (or the only
        one in the typical single-statement case). Empty operations
        (no Parquet paths) return ``None`` without touching *engine*.
        Set ``cleanup=False`` to keep the staged files around (useful
        for debugging an applier).
        """
        statements = self.to_sql()
        if not statements:
            return None

        results: list[Any] = []
        for sql in statements:
            results.append(
                engine.execute(sql, wait=wait, raise_error=raise_error),
            )

        if cleanup:
            self.cleanup(client=client)

        return results[-1] if len(results) == 1 else results

    def cleanup(self, *, client: "DatabricksClient | None" = None) -> None:
        """Remove every staged Parquet + metadata file recorded on this op.

        Best-effort: missing files are tolerated, individual delete
        failures are logged and swallowed so a partial cleanup
        doesn't mask the (already-successful) execute.
        """
        from yggdrasil.databricks.path import DatabricksPath

        for full_path in tuple(self.parquet_paths) + tuple(self.metadata_paths):
            if not full_path:
                continue
            try:
                DatabricksPath.from_(full_path, client=client).remove(
                    missing_ok=True, wait=False, recursive=False,
                )
            except Exception:
                LOGGER.exception(
                    "Failed to clean up staged async-insert artifact %r; "
                    "continuing.",
                    full_path,
                )


# ---------------------------------------------------------------------------
# Iter records helper
# ---------------------------------------------------------------------------


def _iter_records(
    source: Any,
    *,
    client: "DatabricksClient | None" = None,
) -> Iterable[AsyncInsert]:
    """Yield :class:`AsyncInsert` records from a folder / iterable / records.

    Centralised so :meth:`AsyncInsert.merge` accepts the same shapes
    callers naturally have on hand (a folder path, a list of metadata
    files, a list of already-loaded records).
    """
    if isinstance(source, AsyncInsert):
        yield source
        return

    # Folder-like: walk for ``*.json`` files.
    if hasattr(source, "ls"):
        for entry in source.ls(recursive=False):
            name = getattr(entry, "name", "") or ""
            if name.endswith(".json"):
                yield AsyncInsert.from_file(entry, client=client)
        return

    # String / Path: treat as a folder if it has no extension, otherwise
    # a single file.
    if isinstance(source, str):
        if source.endswith(".json"):
            yield AsyncInsert.from_file(source, client=client)
            return
        # Fall through to DatabricksPath.from_ which dispatches to the
        # right subclass; then walk it as a folder.
        from yggdrasil.databricks.path import DatabricksPath

        path = DatabricksPath.from_(source, client=client)
        yield from _iter_records(path, client=client)
        return

    # Iterable: dispatch each item recursively.
    try:
        items = iter(source)
    except TypeError as exc:
        raise TypeError(
            f"AsyncInsert.merge cannot iterate {source!r} "
            f"(expected folder VolumePath, path string, or iterable of "
            f"metadata files / AsyncInsert records)."
        ) from exc

    for item in items:
        yield from _iter_records(item, client=client)


# ---------------------------------------------------------------------------
# Stage helper (top-level API)
# ---------------------------------------------------------------------------


def _make_operation_id() -> str:
    """Unique per-operation id, monotonic-ish on epoch ms + random seed."""
    return f"async-{int(time.time() * 1000)}-{os.urandom(4).hex()}"


def _predicate_to_sql(value: Any) -> Optional[str]:
    """Best-effort SQL rendering of a :class:`Predicate` or pass-through string."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Predicate):
        try:
            from yggdrasil.io.tabular.execution.expr.backends.sql import (
                Dialect,
                to_sql as expr_to_sql,
            )
            return expr_to_sql(value, dialect=Dialect.SPARK)
        except Exception:
            LOGGER.debug(
                "Could not render predicate %r to SQL for async insert metadata; "
                "falling back to repr.",
                value, exc_info=True,
            )
            return repr(value)
    return str(value)


def _enum_to_value(value: Any) -> Any:
    """Return ``value.value`` for enums, ``str(value)`` for unknown shapes,
    or pass primitives through unchanged."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return enum_value
    return str(value)


def _normalize_prune_by(prune_by: Any) -> Optional[Tuple[str, ...]]:
    if prune_by is None:
        return None
    if isinstance(prune_by, str):
        return (prune_by,)
    return tuple(prune_by)


def _normalize_prune_values(
    prune_values: Optional[Mapping[str, Any]],
) -> Optional[dict]:
    if not prune_values:
        return None
    out: dict[str, tuple] = {}
    for key, values in prune_values.items():
        if values is None:
            continue
        out[str(key)] = tuple(_enum_to_value(v) for v in values)
    return out or None


def stage_async_insert(
    table: "Table",
    data: Any,
    *,
    mode: Any = None,
    schema_mode: Any = None,
    cast_options: Optional[CastOptions] = None,
    overwrite_schema: bool | None = None,
    match_by: Optional[Sequence[str]] = None,
    update_column_names: Optional[Sequence[str]] = None,
    zorder_by: Optional[Sequence[str]] = None,
    optimize_after_merge: bool = False,
    vacuum_hours: int | None = None,
    where: Any = None,
    prune_by: Any = None,
    prune_values: Optional[Mapping[str, Any]] = None,
    safe_merge: bool = False,
    operation_id: str | None = None,
) -> "VolumePath":
    """Stage *data* and a sibling :class:`AsyncInsert` metadata file.

    The Parquet payload is cast to the target table's existing schema
    when one can be resolved (the usual case — the target exists).
    When the target can't be inspected (e.g. it doesn't exist yet),
    the source rows are written as-is and the metadata records the
    fact so the applier can decide whether to ``CREATE TABLE`` first.

    Returns the :class:`VolumePath` to the staged Parquet file. The
    sibling metadata file lives at the same stem with a ``.json``
    suffix.
    """
    op_id = operation_id or _make_operation_id()
    folder = table.staging_folder(temporary=False, async_write=True)

    parquet_path = folder.joinpath(f"{op_id}.parquet")
    meta_path = folder.joinpath(f"{op_id}.json")

    # Best-effort target schema resolution. ``collect_schema`` raises
    # when the table doesn't exist yet — that's fine, the applier
    # handles the cold-start case via the recorded ``schema_mode``
    # plus the source schema embedded in the Parquet.
    existing_schema = None
    try:
        existing_schema = table.collect_schema()
    except Exception:
        LOGGER.debug(
            "Target table %r has no resolvable schema; writing rows as-is.",
            table, exc_info=True,
        )

    opts = CastOptions.check(options=cast_options)
    if existing_schema is not None:
        opts = opts.with_target(existing_schema)

    parquet_path.write_table(data, opts, mode=Mode.OVERWRITE)

    record = AsyncInsert(
        target_full_name=table.full_name(safe=True),
        parquet_paths=(_path_for_sql(parquet_path),),
        metadata_paths=(_path_for_sql(meta_path),),
        operation_ids=(op_id,),
        created_at=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        target_catalog_name=table.catalog_name,
        target_schema_name=table.schema_name,
        target_table_name=table.table_name,
        target_field_names=(
            tuple(existing_schema.field_names())
            if existing_schema is not None else None
        ),
        mode=_enum_to_value(mode),
        schema_mode=_enum_to_value(schema_mode),
        overwrite_schema=overwrite_schema,
        match_by=tuple(match_by) if match_by else None,
        update_column_names=(
            tuple(update_column_names) if update_column_names else None
        ),
        zorder_by=tuple(zorder_by) if zorder_by else None,
        optimize_after_merge=bool(optimize_after_merge),
        vacuum_hours=vacuum_hours,
        where=_predicate_to_sql(where),
        prune_by=_normalize_prune_by(prune_by),
        prune_values=_normalize_prune_values(prune_values),
        safe_merge=bool(safe_merge),
    )

    meta_path.write_bytes(record.to_json_bytes())

    LOGGER.info(
        "Staged async insert %s for %r at %r",
        op_id, table, parquet_path,
    )
    return parquet_path


def _resolve_current_client() -> "DatabricksClient":
    """Lazy import + ``current()`` shortcut so the applier helpers
    don't drag :class:`DatabricksClient` into every call signature."""
    from yggdrasil.databricks.client import DatabricksClient
    return DatabricksClient.current()


def _client_from_target(
    target: Any,
    *,
    table: "Table | None" = None,
) -> "DatabricksClient | None":
    """Try to pull a :class:`DatabricksClient` out of *target* / *table*.

    Used by :meth:`AsyncInsert.ensure_job` so callers can pass just a
    :class:`Table` and have the rest (jobs service, applier wiring)
    resolved for free. Returns ``None`` when nothing usable is
    reachable.
    """
    if table is not None:
        client = getattr(table, "client", None)
        if client is not None:
            return client

    if target is None:
        return None

    client = getattr(target, "client", None)
    if client is not None:
        return client

    service = getattr(target, "service", None)
    if service is not None:
        return getattr(service, "client", None)

    return None


def _path_for_sql(path: Any) -> str:
    r"""Return the path string used inside SQL / metadata.

    Prefers the Unity-style ``/Volumes/...`` shape from
    :meth:`VolumePath.full_path` when available — that's what
    ``parquet.\`<path>\`` expects in Databricks SQL.
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
