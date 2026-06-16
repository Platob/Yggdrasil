""":class:`ParquetFolder` — an optimized :class:`Folder` over a directory of
``.parquet`` part files.

A directory of parquet files (optionally Hive-partitioned, ``col=val/…``) is a
first-class tabular: the generic :class:`Folder` can already read it
file-by-file, but parquet is footer-indexed, so a directory is best read as one
**dataset** — letting the engine discover partitions, push the projection into
each leaf, and *skip whole row groups* whose footer min/max can't satisfy the
predicate, in one native pass.

``ParquetFolder`` specializes the read path on that:

- **arrow** — a local directory scans through ``pyarrow.dataset`` with
  ``partitioning="hive"``, projection + predicate pushdown (row-group
  statistics skipping), then the target cast;
- **polars** — ``polars.scan_parquet`` over the directory glob with
  ``hive_partitioning`` does the same in the Rust engine.

A non-local (remote) holder falls back to the generic per-leaf
:class:`Folder` read, which still prunes partitions on the path and pushes the
projection into each :class:`ParquetFile`. Writes / child iteration are
inherited from :class:`Folder` (children default to parquet).

**Schema detection** is specialized too: :meth:`ParquetFolder._collect_schema`
reads the shape from the **first part file's footer** (unified with the Hive
partition columns on the path) rather than the inherited ``.ygg/schema.arrow``
sidecar — the parquet footer is the authoritative, never-stale record of what's
on disk and is present even for directories materialised by external writers
that drop no sidecar.

Mirrors :class:`~yggdrasil.io.delta.delta_folder.DeltaFolder` (a parquet
directory + a transaction log) minus the ``_delta_log`` — same pruning/pushdown
ideas against the raw file layout.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Optional

import pyarrow as pa

from yggdrasil.enums import MimeTypes
from yggdrasil.enums.media_type import MediaType, MediaTypes
from yggdrasil.lazy_imports import polars_module, pyarrow_dataset_module
from yggdrasil.path.folder import Folder, FolderOptions

if TYPE_CHECKING:  # pragma: no cover - typing only
    import polars as pl

__all__ = ["ParquetFolder", "ParquetFolderOptions"]


@dataclasses.dataclass(frozen=True, slots=True)
class ParquetFolderOptions(FolderOptions):
    """:class:`FolderOptions` whose children default to parquet."""

    child_media_type: MediaType = MediaTypes.PARQUET


class ParquetFolder(Folder):
    """:class:`Folder` over a directory of parquet part files."""

    mime_type: ClassVar[MimeTypes] = MimeTypes.PARQUET_FOLDER

    @classmethod
    def options_class(cls):
        return ParquetFolderOptions

    def __repr__(self) -> str:  # noqa: D105
        return f"ParquetFolder(path={self.path!r})"

    # ------------------------------------------------------------------
    # Local-directory detection
    # ------------------------------------------------------------------
    def _local_dir_str(self) -> "Optional[str]":
        """Filesystem path of this directory when it's local, else ``None``
        (so the native dataset / polars scanners only fire where pyarrow /
        polars can open the path directly)."""
        path = self.path
        if not getattr(path, "is_local_path", False):
            return None
        full = getattr(path, "full_path", None)
        return full() if callable(full) else None

    # ------------------------------------------------------------------
    # Schema — from the first part file's footer, not the .ygg sidecar
    # ------------------------------------------------------------------
    def _collect_schema(self, options: FolderOptions) -> Any:
        """Schema from the first parquet part file's footer, not the sidecar.

        A parquet directory carries its schema natively in every part
        file's footer, so the first part is the authoritative shape — it
        stays correct for directories materialised by external writers
        (Spark, ``pyarrow.dataset.write_dataset``, a bare
        ``pq.write_table`` loop) that never drop a ``.ygg/schema.arrow``
        sidecar, and it never goes stale the way a hand-maintained
        sidecar can. ``pyarrow.dataset`` infers the dataset schema from
        the **first** file alone (no full-tree scan) and folds in the
        Hive partition columns discovered on the directory path, so one
        footer read yields the complete folder shape.

        Resolution order (most specific wins):

        1. Ancestor ``_schema_cache`` walk (as :class:`Folder`) — a
           partition child minted mid-read reuses the root's already
           resolved schema instead of re-opening a footer.
        2. The first local part file's footer, unified with the Hive
           partition columns on the path (the part files don't carry
           them); those columns are tagged ``partition_by`` so the
           inherited write / prune machinery still recognises them.
        3. Falls back to :meth:`Folder._collect_schema` (sidecar /
           first-batch) for a remote holder or a directory with no
           readable parquet part file yet.
        """
        parent = self.tabular_parent
        while parent is not None:
            cached = getattr(parent, "_schema_cache", ...)
            if cached is not ...:
                return cached
            parent = getattr(parent, "tabular_parent", None)

        path = self._local_dir_str()
        if path is not None:
            arrow_schema = None
            physical: list[str] = []
            try:
                dataset = pyarrow_dataset_module().dataset(
                    path, format="parquet", partitioning="hive",
                )
                arrow_schema = dataset.schema
                # Hive partition columns live on the path, not in the
                # footer — diff the unified dataset schema against the
                # first fragment's physical schema to recover them.
                physical = next(dataset.get_fragments()).physical_schema.names
            except Exception:
                # No parquet files / not materialised yet — defer to the
                # sidecar / first-batch fallback below.
                arrow_schema = None
            if arrow_schema is not None:
                from yggdrasil.data.schema import Schema
                schema = Schema.from_arrow(arrow_schema)
                partition_names = [n for n in arrow_schema.names if n not in physical]
                if partition_names:
                    schema = Schema(
                        [
                            c.with_partition_by(True, inplace=False)
                            if c.name in partition_names else c
                            for c in schema.children
                        ],
                        name=schema.name,
                        metadata=schema.metadata,
                    )
                return schema

        return super()._collect_schema(options)

    # ------------------------------------------------------------------
    # Read — native dataset scan with pushdown, else generic per-leaf
    # ------------------------------------------------------------------
    def _read_arrow_batches(
        self, options: FolderOptions,
    ) -> Iterator[pa.RecordBatch]:
        path = self._local_dir_str()
        if path is None:
            # Remote: the generic Folder read still prunes partitions on the
            # path and pushes the projection into each ParquetFile leaf.
            yield from super()._read_arrow_batches(options)
            return

        pds = pyarrow_dataset_module()
        try:
            dataset = pds.dataset(path, format="parquet", partitioning="hive")
        except Exception:
            # No files / not yet materialised — defer to the generic path,
            # which emits the resolved empty-schema batch.
            yield from super()._read_arrow_batches(options)
            return

        # Projection ∪ predicate columns; ``None`` reads everything.
        wanted = options.read_columns()
        columns = None
        if wanted is not None:
            columns = [n for n in dataset.schema.names if n in wanted] or None

        # Push the predicate to the dataset for row-group statistics skipping,
        # then cast with the predicate stripped (the dataset already filtered)
        # so the row filter isn't re-applied per batch.
        arrow_filter = (
            options.predicate.to_arrow() if options.predicate is not None else None
        )
        cast_opts = (
            dataclasses.replace(options, predicate=None)
            if options.predicate is not None
            else options
        )

        yielded = False
        for batch in dataset.to_batches(columns=columns, filter=arrow_filter):
            yielded = True
            yield cast_opts.cast_arrow_batch(batch)
        if not yielded:
            # A dataset with files but no matching rows still yields a
            # schema-carrying batch above; only a genuinely empty directory
            # reaches here — fall back for the resolved empty-schema batch.
            yield from super()._read_arrow_batches(options)

    # ------------------------------------------------------------------
    # Polars — native lazy / eager scan with full pushdown
    # ------------------------------------------------------------------
    def _scan_polars_frame(self, options: FolderOptions) -> "pl.LazyFrame":
        path = self._local_dir_str()
        if path is None:
            return super()._scan_polars_frame(options)
        pl = polars_module()
        lf = pl.scan_parquet(
            f"{path.rstrip('/')}/**/*.parquet", hive_partitioning=True,
        )
        # ``cast_polars_tabular`` folds the predicate + target cast into the
        # lazy plan, so projection/predicate pushdown still reaches the files.
        return options.cast_polars_tabular(lf)

    def _read_polars_frame(self, options: FolderOptions) -> "pl.DataFrame":
        path = self._local_dir_str()
        if path is None:
            return super()._read_polars_frame(options)
        # Collect the pushed-down lazy scan — polars plans projection,
        # predicate, and row-group skipping before reading any data.
        return self._scan_polars_frame(options).collect()

    def _read_arrow_dataset(self, options: FolderOptions) -> "Any":
        """Native :class:`pyarrow.dataset.Dataset` over the directory, so a
        caller wanting the dataset handle (further pushdown, partitioning)
        gets the Hive-aware one for a local folder."""
        path = self._local_dir_str()
        if path is None:
            return super()._read_arrow_dataset(options)
        return pyarrow_dataset_module().dataset(
            path, format="parquet", partitioning="hive",
        )
