"""Delta Lake table I/O as a :class:`PathIO`.

:class:`DeltaIO` treats a Delta Lake table (a directory with a
``_delta_log/`` transaction log) as a single logical path that
:class:`yggdrasil.io.buffer.path_io.PathIO` can read from. Under the
hood it delegates to the ``deltalake`` package (delta-rs) — the
canonical reference implementation of the Delta protocol — so the
whole spec surface is supported without re-implementing any of it:

* Reader/writer versions 1 through 3/7 (table features gate).
* Table features: **deletion vectors**, **V2 checkpoints** (including
  sidecars + UUID-named files), column mapping (id / name modes),
  timestamp-ntz, row tracking, domain metadata, CDF, etc. See the
  Delta Lake protocol spec for the full list.
* Time travel by ``version`` or ``timestamp``.
* Partition filter pushdown.
* Log compaction via ``_last_checkpoint`` + ``NNNN.checkpoint.*``.

Read path
---------
``_read_arrow_batches`` resolves the active snapshot via
``deltalake.DeltaTable`` and calls :meth:`to_pyarrow_dataset`. The
returned :class:`pyarrow.dataset.Dataset` already:

1. Reconciles ``add`` / ``remove`` actions.
2. Applies deletion vectors (materializes the Roaring64 bitmaps into
   row-level keep masks before the parquet scan).
3. Translates column-mapping (physical → logical names).
4. Projects partition columns.

From that point the base :class:`PathIO` pipeline (filter → projection
→ cast) layers on top in exactly the same way it does for raw Parquet,
so ``read_arrow_table`` / ``read_arrow_batches`` / ``scanner`` /
``execute`` all "just work" with SQL pushdown for free.

Polars tightens that further: :meth:`_read_polars_frame` delegates to
``polars.scan_delta`` when available, which gives Polars full control
of projection/predicate pushdown through the scanner rather than the
Arrow layer — noticeably faster for ``SELECT … WHERE …`` statements
against large tables. The Arrow dataset path remains the fallback.

Write path
----------
Writes round-trip through :func:`deltalake.write_deltalake`. Each
:class:`yggdrasil.io.enums.SaveMode` maps to a delta-rs mode:

=========================  ===========
``SaveMode``               delta-rs
=========================  ===========
``APPEND``                 ``"append"``
``OVERWRITE`` / ``TRUNCATE``  ``"overwrite"``
``IGNORE``                 ``"ignore"``
``ERROR_IF_EXISTS``        ``"error"``
``AUTO``                   ``"append"`` if table exists, else new
=========================  ===========

``UPSERT`` is intentionally *not* supported yet — delta-rs implements
it via a separate :meth:`DeltaTable.merge` builder with its own API;
a future iteration can surface that through :meth:`write_arrow_table`.

Layout probes
-------------
The ``iter_files`` implementation yields the active data-file paths
reported by :meth:`DeltaTable.file_uris`. It exists for diagnostics
and to satisfy the :class:`PathIO` contract — the main read path does
NOT walk files; it goes straight through the delta-rs dataset so that
deletion vectors and column mapping are applied correctly. A raw
``pyarrow.dataset.dataset(list_of_paths)`` would silently miss
deletion vectors.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

from yggdrasil.arrow.lib import pyarrow as pa
from yggdrasil.io.enums import MediaType, MimeType, MimeTypes, SaveMode

from .bytes_io import BytesIO
from .path_io import PathIO, PathOptions

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pyarrow.dataset as ds

__all__ = ["DeltaOptions", "DeltaIO"]


# =====================================================================
# Options
# =====================================================================

@dataclass
class DeltaOptions(PathOptions):
    """Options for a Delta Lake table read/write.

    Parameters
    ----------
    version:
        Time-travel to this log version. Mutually exclusive with
        ``timestamp``. ``None`` means "latest".
    timestamp:
        Time-travel to this wall-clock timestamp (ISO-8601 string or
        :class:`datetime.datetime`). Mutually exclusive with ``version``.
    partitions:
        Delta-style partition filter passed through to
        :meth:`DeltaTable.to_pyarrow_dataset`. List of 3-tuples
        ``(column, operator, value)`` where ``operator`` is one of
        ``=``, ``!=``, ``<``, ``<=``, ``>``, ``>=``, ``in``, ``not in``.
        Applied before ``PathOptions.filter``; cheaper than a row-level
        filter because it prunes entire files.
    storage_options:
        Passed through to :class:`deltalake.DeltaTable` — bucket creds,
        retries, etc.
    without_files:
        Load the table metadata without enumerating files. Faster for
        schema-only queries.
    as_large_types:
        Force Arrow ``large_string`` / ``large_binary`` output to avoid
        offset overflow on very wide columns (delta-rs flag).
    partition_by:
        Write-side: list of partition columns when creating a new
        table. Ignored when appending / overwriting an existing table.
    configuration:
        Write-side: ``delta.*`` table properties to set on create.
    schema_mode:
        Write-side: ``"merge"`` to evolve the schema on append /
        overwrite, ``"overwrite"`` to replace, ``None`` to reject
        schema drift (delta-rs default).
    target_file_size:
        Write-side: approximate target bytes per output parquet file.
    """

    version: int | None = None
    timestamp: str | datetime | None = None
    partitions: Sequence[tuple[str, str, Any]] | None = None
    storage_options: Mapping[str, str] | None = None
    without_files: bool = False
    as_large_types: bool = False

    # Write-side knobs. None-safe defaults.
    partition_by: Sequence[str] | str | None = None
    configuration: Mapping[str, str | None] | None = None
    schema_mode: str | None = None
    target_file_size: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.version is not None and self.timestamp is not None:
            raise ValueError(
                "DeltaOptions.version and DeltaOptions.timestamp are "
                "mutually exclusive — pick one time-travel anchor"
            )
        if self.version is not None:
            if not isinstance(self.version, int) or isinstance(self.version, bool):
                raise TypeError(
                    f"version must be int|None, got {type(self.version).__name__}"
                )
            if self.version < 0:
                raise ValueError("version must be >= 0")

        if self.partitions is not None:
            normalized: list[tuple[str, str, Any]] = []
            for item in self.partitions:
                if (
                    not isinstance(item, (tuple, list))
                    or len(item) != 3
                    or not isinstance(item[0], str)
                    or not isinstance(item[1], str)
                ):
                    raise TypeError(
                        "partitions must be a sequence of "
                        "(column, operator, value) 3-tuples"
                    )
                normalized.append((item[0], item[1], item[2]))
            self.partitions = tuple(normalized)

        if not isinstance(self.without_files, bool):
            raise TypeError(
                f"without_files must be bool, got {type(self.without_files).__name__}"
            )
        if not isinstance(self.as_large_types, bool):
            raise TypeError(
                f"as_large_types must be bool, got {type(self.as_large_types).__name__}"
            )

        if self.schema_mode is not None and self.schema_mode not in {"merge", "overwrite"}:
            raise ValueError(
                f"schema_mode must be 'merge', 'overwrite', or None; got {self.schema_mode!r}"
            )

        if self.target_file_size is not None:
            if (
                not isinstance(self.target_file_size, int)
                or isinstance(self.target_file_size, bool)
                or self.target_file_size <= 0
            ):
                raise ValueError("target_file_size must be a positive int or None")

    @classmethod
    def resolve(cls, *, options: "DeltaOptions | None" = None, **overrides: Any) -> "DeltaOptions":
        return cls.check_parameters(options=options, **overrides)


# =====================================================================
# DeltaIO
# =====================================================================

# Delta-rs SaveMode → delta-rs mode literal.
_SAVEMODE_MAP: dict[SaveMode, str] = {
    SaveMode.APPEND: "append",
    SaveMode.OVERWRITE: "overwrite",
    SaveMode.TRUNCATE: "overwrite",
    SaveMode.IGNORE: "ignore",
    SaveMode.ERROR_IF_EXISTS: "error",
}


# Sentinel for _peek_iterator; distinct from None so that a legitimate
# leading None survives (not that pa.RecordBatch can be None, but we
# keep the helper generic).
_SENTINEL: Any = object()


def _delta_protocol_error_types() -> tuple[type[BaseException], ...]:
    """Return the delta-rs protocol-error classes we want to catch.

    Lazily resolved — ``deltalake`` is optional, and importing it at
    module top would block base installs. Falls back to a plain
    ``Exception`` filter when the package isn't around, which is safe:
    ``DeltaIO`` methods that touch delta-rs raise long before any
    ``_read_arrow_batches`` call reaches this handler in the
    no-deltalake scenario.
    """
    try:
        from deltalake.exceptions import DeltaProtocolError  # type: ignore

        return (DeltaProtocolError,)
    except Exception:
        try:
            from deltalake._internal import DeltaProtocolError  # type: ignore

            return (DeltaProtocolError,)
        except Exception:
            return ()


# Bound once at import. Empty tuple disables the fallback branch —
# which is what we want when deltalake isn't installed (the earlier
# calls would have failed with a clearer ImportError already). We
# deliberately do NOT use a bare ``Exception`` here: it would swallow
# unrelated bugs and silently reroute through the Polars fallback.
_DELTA_PROTOCOL_ERRORS: tuple[type[BaseException], ...] = _delta_protocol_error_types()


@dataclass(slots=True)
class DeltaIO(PathIO):
    """PathIO reading from a Delta Lake table directory."""

    path: Path = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.path is None:
            raise ValueError("DeltaIO requires a non-None path")
        if not isinstance(self.path, Path):
            self.path = Path(self.path)

        # Always a Delta table — pin the mime type so subsequent
        # MediaType.parse() on the directory never short-circuits to
        # OCTET_STREAM (parent's first-file inference would pick up
        # parquet data files, which is the wrong answer for a Delta
        # table root).
        if self.media_type is None:
            self.media_type = MediaType(MimeTypes.DELTA)

    # ------------------------------------------------------------------
    # Options plumbing
    # ------------------------------------------------------------------

    @classmethod
    def check_options(
        cls,
        options: Optional[DeltaOptions],
        *args,
        **kwargs,
    ) -> DeltaOptions:
        del args
        return DeltaOptions.check_parameters(options=options, **kwargs)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def make(
        cls,
        path: str | Path,
        media: MediaType | MimeType | str | None = None,
    ) -> "DeltaIO":
        """Build a :class:`DeltaIO` rooted at ``path``.

        ``path`` must be the Delta table directory (the parent of
        ``_delta_log/``). The table doesn't have to exist yet —
        writes will create it.
        """
        resolved_path = Path(path)

        if media is None:
            resolved_media: MediaType = MediaType(MimeTypes.DELTA)
        elif isinstance(media, MediaType):
            resolved_media = media
        elif isinstance(media, MimeType):
            resolved_media = MediaType(media)
        else:  # str
            resolved_media = MediaType.parse(str(media), default=MediaType(MimeTypes.DELTA))

        return cls(
            media_type=resolved_media,
            holder=BytesIO(),
            path=resolved_path,
        )

    # ------------------------------------------------------------------
    # Public Delta helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_delta_table(path: str | Path) -> bool:
        """Return ``True`` when ``path`` is a Delta table root.

        Probes for ``_delta_log/`` locally first (cheap); falls back to
        :func:`deltalake.DeltaTable.is_deltatable` for remote URIs.
        """
        p = Path(path)
        if p.is_dir() and (p / "_delta_log").is_dir():
            return True
        try:
            from yggdrasil.deltalake.lib import deltalake  # noqa: F401
            from deltalake import DeltaTable

            return bool(DeltaTable.is_deltatable(str(p)))
        except Exception:
            return False

    @property
    def table_exists(self) -> bool:
        return self.is_delta_table(self.path)

    def version(self) -> int:
        """Return the latest log version number of this table.

        Raises :class:`FileNotFoundError` if the path isn't a Delta
        table yet.
        """
        return int(self.delta_table().version())

    def protocol(self) -> Any:
        """Return the :class:`ProtocolVersions` for the current snapshot.

        Exposes ``min_reader_version``, ``min_writer_version``,
        ``reader_features``, ``writer_features`` — useful for
        diagnosing whether a table will go through the Arrow dataset
        fast path or the Polars fallback.
        """
        return self.delta_table().protocol()

    def history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return recent ``commitInfo`` entries from the transaction log."""
        return list(self.delta_table().history(limit))

    def delta_table(
        self,
        *,
        options: DeltaOptions | None = None,
        **option_kwargs: Any,
    ) -> Any:
        """Return a live :class:`deltalake.DeltaTable` for this path.

        Honors ``version`` / ``timestamp`` / ``storage_options`` /
        ``without_files`` from the options. Raises
        :class:`FileNotFoundError` when the path is not a Delta table
        yet — callers that want a metadata-less instance should use
        :func:`deltalake.DeltaTable` directly.
        """
        from deltalake import DeltaTable  # lazy: optional dep

        resolved = self.check_options(options=options, **option_kwargs)
        if not self.is_delta_table(self.path):
            raise FileNotFoundError(
                f"Not a Delta table: {self.path!s}. "
                "The directory must contain a _delta_log/ subdirectory."
            )

        dt = DeltaTable(
            str(self.path),
            version=resolved.version,
            storage_options=dict(resolved.storage_options)
            if resolved.storage_options is not None
            else None,
            without_files=resolved.without_files,
        )
        if resolved.timestamp is not None:
            dt.load_as_version(resolved.timestamp)
        return dt

    # ------------------------------------------------------------------
    # PathIO overrides
    # ------------------------------------------------------------------

    def iter_files(
        self,
        recursive: bool = True,
        *,
        include_hidden: bool = False,
        supported_only: bool = True,
        mime_type: MimeType | str | None = None,
    ) -> Iterator["DeltaIO"]:
        """Yield one :class:`DeltaIO` per active data file.

        This mirrors the :class:`LocalPathIO` contract for completeness
        and diagnostics — the main read path does NOT call this;
        :meth:`_read_arrow_batches` goes through the delta-rs dataset
        so deletion vectors and column mapping are honored.

        ``recursive`` / ``include_hidden`` / ``supported_only`` are
        ignored: the active file set is defined by the transaction
        log, not by directory walking.
        """
        del recursive, include_hidden, supported_only, mime_type

        if not self.is_delta_table(self.path):
            return

        from deltalake import DeltaTable

        dt = DeltaTable(str(self.path))
        for uri in dt.file_uris():
            yield DeltaIO(
                media_type=MediaType(MimeTypes.PARQUET),
                holder=BytesIO(),
                path=Path(uri),
            )

    # ------------------------------------------------------------------
    # Read — override PathIO to go through deltalake
    # ------------------------------------------------------------------

    def read_dataset(
        self,
        *,
        options: DeltaOptions | None = None,
        **option_kwargs: Any,
    ) -> "ds.Dataset":
        """Return a :class:`pyarrow.dataset.Dataset` over the active snapshot.

        Under the hood calls :meth:`DeltaTable.to_pyarrow_dataset`,
        which applies column mapping and partition filters before
        handing the Arrow scanner the live parquet files.

        .. note::

           As of ``deltalake`` 1.5 the Arrow dataset path refuses to
           read tables that declare the ``deletionVectors`` reader
           feature (see delta-rs upstream). For those tables use
           :meth:`read_arrow_table` / :meth:`read_arrow_batches`
           directly — they transparently fall back to a Polars
           ``scan_delta`` plan, which does apply deletion vectors.
        """
        resolved = self.check_options(options=options, **option_kwargs)
        dt = self.delta_table(options=resolved)
        return dt.to_pyarrow_dataset(
            partitions=list(resolved.partitions) if resolved.partitions else None,
            as_large_types=resolved.as_large_types,
        )

    def to_arrow_dataset(
        self,
        *,
        options: DeltaOptions | None = None,
        **option_kwargs: Any,
    ) -> "ds.Dataset":
        return self.read_dataset(options=options, **option_kwargs)

    def scanner(
        self,
        *,
        options: DeltaOptions | None = None,
        **option_kwargs: Any,
    ) -> "ds.Scanner":
        resolved = self.check_options(options=options, **option_kwargs)
        dataset = self.read_dataset(options=resolved)
        return dataset.scanner(
            columns=list(resolved.columns) if resolved.columns is not None else None,
            filter=self._normalize_filter_to_expression(resolved.filter),
            batch_size=resolved.batch_size or 131_072,
            batch_readahead=resolved.batch_readahead or 16,
            fragment_readahead=resolved.fragment_readahead or 4,
            use_threads=resolved.use_threads,
        )

    def _is_dataset_capable(self) -> bool:
        # Always true for Delta — we have a materialized pyarrow dataset.
        return True

    def _read_arrow_batches(
        self,
        options: DeltaOptions,
    ) -> Iterator["pa.RecordBatch"]:
        """Stream Arrow batches through the best available Delta reader.

        Dispatch strategy:

        1. Try the delta-rs :class:`pyarrow.dataset.Dataset`. It's the
           fastest path — Arrow scans the parquet fragments with full
           pushdown for ``options.filter`` and ``options.columns``.
        2. On :class:`DeltaProtocolError` (delta-rs still refuses
           tables with deletion vectors as of 1.5), fall back to
           :func:`polars.scan_delta` → Arrow, which *does* materialize
           deletion vectors.

        Caller-side cast is applied last via
        ``options.cast.cast_iterator`` exactly like the Parquet path.
        """
        try:
            raw_batches = self._iter_batches_via_deltalake(options)
            # Materialize the first batch eagerly so any protocol error
            # raised by the delta-rs scanner surfaces here (inside the
            # try) rather than later in the generator chain, where we
            # can no longer fall back.
            raw_batches = self._peek_iterator(raw_batches)
        except _DELTA_PROTOCOL_ERRORS:
            raw_batches = self._iter_batches_via_polars(options)

        yield from options.cast.cast_iterator(raw_batches)

    def _iter_batches_via_deltalake(
        self,
        options: DeltaOptions,
    ) -> Iterator["pa.RecordBatch"]:
        scanner = self.scanner(options=options)
        yield from scanner.to_batches()

    def _iter_batches_via_polars(
        self,
        options: DeltaOptions,
    ) -> Iterator["pa.RecordBatch"]:
        """Fallback reader for protocol features delta-rs won't scan.

        Uses :func:`polars.scan_delta` which handles deletion vectors
        correctly, applies caller filter/projection inside the Polars
        lazy plan, then hands an Arrow table to the normal batch
        pipeline.
        """
        from yggdrasil.polars.lib import polars as pl

        scan_kwargs: dict[str, Any] = {}
        if options.version is not None:
            scan_kwargs["version"] = options.version
        if options.storage_options is not None:
            scan_kwargs["storage_options"] = dict(options.storage_options)

        lf = pl.scan_delta(str(self.path), **scan_kwargs)

        if options.columns is not None:
            lf = lf.select(list(options.columns))

        expr = self._normalize_filter_to_expression(options.filter)
        if expr is not None:
            # Arrow expressions and Polars expressions are separate
            # dialects; rebuild the filter as a Polars expression via
            # pa.dataset → RecordBatchReader → Polars path. Simpler:
            # collect then filter in Arrow. Delta tables with DVs tend
            # to be large, so push the filter via Arrow after collect.
            table = lf.collect().to_arrow()
            yield from table.filter(expr).to_batches(
                max_chunksize=options.batch_size or 131_072
            )
            return

        table = lf.collect().to_arrow()
        yield from table.to_batches(
            max_chunksize=options.batch_size or 131_072
        )

    @staticmethod
    def _peek_iterator(
        it: Iterator["pa.RecordBatch"],
    ) -> Iterator["pa.RecordBatch"]:
        """Pull the first element to force generator start, then rechain.

        Lets exceptions raised at iteration-startup bubble up at the
        call site instead of being deferred until the first ``yield``
        — critical for the Arrow → Polars fallback trigger.
        """
        import itertools

        first = next(it, _SENTINEL)
        if first is _SENTINEL:
            return iter(())
        return itertools.chain((first,), it)

    def _collect_arrow_schema(self, full: bool = False) -> "pa.Schema":
        """Return the table schema via :meth:`DeltaTable.schema`.

        ``full`` is accepted for API parity with the base class; Delta
        tables always have a single authoritative schema in their
        metaData action, so the flag is a no-op.

        ``deltalake.Schema.to_arrow`` returns an ``arro3.core.Schema``
        (delta-rs uses arro3 internally). We funnel it through
        :func:`pyarrow.schema`, which consumes the Arrow C Data
        Interface capsule and produces a native :class:`pa.Schema` —
        matching the contract of every other MediaIO subclass.
        """
        del full
        if not self.is_delta_table(self.path):
            return pa.schema([])
        dt = self.delta_table()
        schema = dt.schema().to_arrow()
        if isinstance(schema, pa.Schema):
            return schema
        # arro3.core.Schema → pa.Schema via the Arrow C Data Interface.
        return pa.schema(schema)

    # ------------------------------------------------------------------
    # Optional: lazy Polars integration for SQL (execute)
    # ------------------------------------------------------------------

    def _read_polars_frame(self, options):  # noqa: ANN001 - inherits signature
        """Override to prefer :func:`polars.scan_delta` for true lazy scans.

        Polars' native Delta scanner pushes projection and predicates
        further down than the Arrow dataset path, so queries run
        through :meth:`MediaIO.execute` hit the parquet layer with the
        tightest column/row slice the SQL planner can deduce.

        Falls back to the Arrow-backed parent implementation when:

        * Polars doesn't expose ``scan_delta`` (very old versions), or
        * the Polars reader raises (protocol features it can't handle).
        """
        from yggdrasil.polars.lib import polars as pl

        want_lazy = bool(getattr(options, "lazy", False))
        batch_size = getattr(options, "batch_size", 0) or 0

        # scan_delta is a single-frame lazy plan; when batched reads are
        # requested, fall back to the Arrow batch iterator to keep the
        # chunking semantics consistent with ParquetIO/CsvIO.
        if batch_size > 0 or not hasattr(pl, "scan_delta"):
            return super()._read_polars_frame(options)

        scan_kwargs: dict[str, Any] = {}
        if options.version is not None:
            scan_kwargs["version"] = options.version
        # scan_delta takes storage_options as a dict.
        if options.storage_options is not None:
            scan_kwargs["storage_options"] = dict(options.storage_options)

        try:
            lf = pl.scan_delta(str(self.path), **scan_kwargs)
        except Exception:
            # Any scan_delta failure → fall through to the Arrow path
            # (e.g. protocol features Polars can't read directly).
            return super()._read_polars_frame(options)

        if options.columns is not None:
            lf = lf.select(list(options.columns))

        # The base-class cast contract is applied at the Arrow batch
        # layer; when a target schema is set, defer to super() so the
        # same cast pipeline runs. Polars' own cast can't honor the
        # full CastOptions surface.
        cast = getattr(options, "cast", None)
        if cast is not None and cast.target_field is not None:
            return super()._read_polars_frame(options)

        return lf if want_lazy else lf.collect()

    # ------------------------------------------------------------------
    # Write — delegate to deltalake.write_deltalake
    # ------------------------------------------------------------------

    def _write_arrow_batches(
        self,
        batches: Iterator["pa.RecordBatch"],
        options: DeltaOptions,
    ) -> None:
        """Write batches into the Delta table via :func:`write_deltalake`.

        :func:`write_deltalake` expects an Arrow-convertible input; we
        materialize the iterator into a :class:`pyarrow.Table` once
        since delta-rs itself re-chunks according to
        ``target_file_size``.
        """
        from deltalake import write_deltalake  # lazy: optional dep

        table = pa.Table.from_batches(list(batches))

        # Empty batch shouldn't create a commit unless the caller is
        # clearly asking for table creation. For OVERWRITE we still
        # want the empty write to happen so the schema gets recorded.
        if table.num_rows == 0 and options.mode not in {SaveMode.OVERWRITE, SaveMode.TRUNCATE}:
            return

        exists = self.is_delta_table(self.path)
        mode = self._resolve_write_mode(options.mode, exists=exists)

        # Guard path for SaveMode.ERROR_IF_EXISTS / IGNORE when the
        # base-class `skip_write` couldn't help (we're not buffer-
        # backed). Match ParquetIO semantics here.
        if exists and options.mode == SaveMode.IGNORE:
            return

        partition_by = options.partition_by
        if isinstance(partition_by, str):
            partition_by = [partition_by]
        elif partition_by is not None:
            partition_by = list(partition_by)

        write_kwargs: dict[str, Any] = {
            "mode": mode,
        }
        if partition_by is not None:
            write_kwargs["partition_by"] = partition_by
        if options.configuration is not None:
            write_kwargs["configuration"] = dict(options.configuration)
        if options.schema_mode is not None:
            write_kwargs["schema_mode"] = options.schema_mode
        if options.storage_options is not None:
            write_kwargs["storage_options"] = dict(options.storage_options)
        if options.target_file_size is not None:
            write_kwargs["target_file_size"] = options.target_file_size

        write_deltalake(str(self.path), table, **write_kwargs)

    @staticmethod
    def _resolve_write_mode(mode: SaveMode, *, exists: bool) -> str:
        """Map yggdrasil :class:`SaveMode` to a delta-rs mode literal."""
        if mode == SaveMode.AUTO:
            return "append" if exists else "error"
        if mode == SaveMode.UPSERT:
            raise NotImplementedError(
                "SaveMode.UPSERT is not supported by DeltaIO yet — "
                "use DeltaTable.merge() directly from deltalake"
            )
        try:
            return _SAVEMODE_MAP[mode]
        except KeyError:
            raise ValueError(f"Unsupported SaveMode for DeltaIO: {mode!r}")
