"""Filesystem-backed :class:`ExecutionTable`.

The table's identity, schema, and partition intent live in
``<schema>/<table>/_yggdrasil/table.json``; row data lives under
``<schema>/<table>/data/`` and is read/written through a
:class:`FolderIO` bound to that directory. The folder picks up
whatever child media type the table is declared with (Parquet by
default, Arrow IPC / CSV / NDJSON / … via the :class:`MediaType`
extension registry).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping

import pyarrow as pa

from yggdrasil.data.enums import Mode
from yggdrasil.data.enums.media_type import MediaType, MediaTypes
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema, StructField
from yggdrasil.io.nested.folder_io import FolderIO, FolderOptions
from yggdrasil.io.path import Path
from yggdrasil.unity.fs import registry
from yggdrasil.unity.info import TableInfo
from yggdrasil.unity.table import ExecutionTable

if TYPE_CHECKING:
    from yggdrasil.unity.fs.engine import FSEngine
    from yggdrasil.unity.fs.schema import FSSchema


__all__ = ["FSTable"]


logger = logging.getLogger(__name__)


_UNSET = ...


def _coerce_schema(value: Any) -> Schema:
    """Coerce *value* into a :class:`StructField` schema.

    Anything :class:`Schema.from_` handles works — a :class:`Schema`,
    a :class:`pa.Schema`, a list of :class:`Field` instances, a
    polars / pandas / spark schema, an Arrow table.
    """
    if isinstance(value, StructField):
        return value
    coerced = Schema.from_(value)
    if isinstance(coerced, StructField):
        return coerced
    # Field with StructType dtype — wrap its members into a StructField.
    if hasattr(coerced, "children") and coerced.children:
        return StructField(coerced.children, name=coerced.name)
    raise TypeError(
        f"Cannot coerce {type(value).__name__} to a Schema. Pass a "
        "yggdrasil Schema, a pyarrow Schema, a list of Fields, or any "
        "frame whose schema yggdrasil.data.Schema.from_(...) accepts."
    )


def _coerce_media_type(value: Any) -> MediaType:
    if value is _UNSET or value is None:
        return MediaTypes.PARQUET
    coerced = MediaType.from_(value, default=None)
    if coerced is None:
        raise ValueError(
            f"format must coerce to a MediaType; got {value!r}. Pass "
            "MediaTypes.PARQUET / MediaTypes.ARROW_IPC / a registered "
            "extension string."
        )
    return coerced


class FSTable(ExecutionTable):
    """Managed table backed by a :class:`FolderIO` over ``<table>/data/``."""

    def __init__(self, *, schema: "FSSchema", name: str) -> None:
        ExecutionTable.__init__(self)
        self._schema_handle = schema
        self._name = name

    # ── identity ───────────────────────────────────────────────────────

    @property
    def engine(self) -> "FSEngine":
        return self._schema_handle.engine

    @property
    def schema_handle(self) -> "FSSchema":
        return self._schema_handle

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> Path:
        """Root directory of this table (``<schema>/<name>``)."""
        return self._schema_handle.child_path(self._name)

    @property
    def data_path(self) -> Path:
        """Directory holding the row-data files."""
        return self.path / registry.DATA_DIR

    # ── info ───────────────────────────────────────────────────────────

    def _read_info(self) -> TableInfo:
        return registry.read_table_info(self.path)

    # ── lifecycle ──────────────────────────────────────────────────────

    def create(
        self,
        schema: Any = _UNSET,
        *,
        format: Any = _UNSET,
        partition_by: "tuple[str, ...] | list[str] | None" = None,
        comment: str | None = None,
        owner: str | None = None,
        properties: "Mapping[str, str] | None" = None,
        if_not_exists: bool = True,
    ) -> "FSTable":
        if self.exists:
            if if_not_exists:
                logger.debug("Table %r already exists — skipping create", self)
                return self
            raise FileExistsError(
                f"Table {self.full_name!r} already exists at "
                f"{self.path.full_path()!r}."
            )
        if schema is _UNSET:
            raise ValueError(
                f"Table {self.full_name!r} requires a schema= argument on "
                "create. Pass a yggdrasil Schema, a pyarrow Schema, or any "
                "frame whose schema can be inferred."
            )
        if not self._schema_handle.exists:
            raise FileNotFoundError(
                f"Schema {self._schema_handle.full_name!r} does not exist. "
                "Create it first via catalog.create_schema(name)."
            )
        coerced_schema = _coerce_schema(schema)
        media = _coerce_media_type(format)
        partitions = tuple(partition_by or ())
        if partitions:
            field_names = {f.name for f in coerced_schema.fields}
            missing = [p for p in partitions if p not in field_names]
            if missing:
                raise ValueError(
                    f"partition_by columns {missing!r} are not in the table "
                    f"schema {sorted(field_names)!r}."
                )
        logger.debug(
            "Creating table %r (format=%r, partition_by=%r, owner=%r)",
            self, media, partitions, owner,
        )
        self.path.mkdir(parents=True, exist_ok=True)
        self.data_path.mkdir(parents=True, exist_ok=True)
        info = TableInfo(
            catalog_name=self._schema_handle.catalog_name,
            schema_name=self._schema_handle.name,
            name=self._name,
            schema=coerced_schema,
            format=media,
            partition_by=partitions,
            comment=comment,
            owner=owner,
            properties=dict(properties or {}),
        )
        registry.write_table_info(self.path, info)
        self._store_info(info)
        logger.info("Created table %r", self)
        return self

    def delete(
        self,
        *,
        purge_data: bool = True,
        missing_ok: bool = True,
    ) -> "FSTable":
        if not self.exists:
            if missing_ok:
                logger.debug("Table %r does not exist — skipping delete", self)
                return self
            raise FileNotFoundError(
                f"Table {self.full_name!r} does not exist."
            )
        logger.debug("Deleting table %r (purge_data=%s)", self, purge_data)
        if purge_data:
            self.path.remove(recursive=True, missing_ok=missing_ok)
        else:
            registry.delete_metadata(self.path, missing_ok=missing_ok)
        self._invalidate_info()
        logger.info("Deleted table %r", self)
        return self

    # ── Tabular IO — delegate to a FolderIO over data_path ────────────

    def _open_folder(self) -> FolderIO:
        """Bind a fresh :class:`FolderIO` to :attr:`data_path`.

        The folder is the canonical multi-file Arrow / Parquet / Arrow-IPC
        read/write engine; we let it own batch reading, partition discovery,
        and OVERWRITE / APPEND / UPSERT semantics. The table layer just
        threads the persisted child media type through ``FolderOptions``.
        """
        return FolderIO(path=self.data_path)

    def _folder_options(
        self,
        options: CastOptions | None,
    ) -> FolderOptions:
        """Build a :class:`FolderOptions` derived from *options* + persisted info."""
        info = self.info
        if isinstance(options, FolderOptions):
            base = options
        else:
            base = FolderOptions()
            if options is not None:
                base = base.copy(
                    source=options.source,
                    target=options.target,
                    mode=options.mode,
                    safe=options.safe,
                    row_size=options.row_size,
                    byte_size=options.byte_size,
                    row_limit=options.row_limit,
                    predicate=options.predicate,
                )
        return base.copy(child_media_type=info.format)

    def _read_arrow_batches(self, options: CastOptions) -> Iterator[pa.RecordBatch]:
        if not self.data_path.exists():
            return
        folder = self._open_folder()
        yield from folder._read_arrow_batches(self._folder_options(options))

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: CastOptions,
    ) -> None:
        self.data_path.mkdir(parents=True, exist_ok=True)
        folder = self._open_folder()
        # Default to APPEND when caller didn't pick a mode — managed
        # tables in this layer accumulate parts the same way Delta /
        # Hive partitioned tables do.
        opts = self._folder_options(options)
        if opts.mode is Mode.AUTO:
            opts = opts.copy(mode=Mode.APPEND)
        folder._write_arrow_batches(batches, opts)
        # Drop the FolderIO's stat cache so the next read sees the
        # newly-written files.
        self.data_path.invalidate_singleton(remove_global=False)
