"""In-memory :class:`Tabular` holding a (mutable) Spark DataFrame.

:class:`Dataset` is the canonical Spark surface. Two interleaved
roles, both satisfied by one class:

* :class:`Tabular` contract — :meth:`_read_arrow_batches` /
  :meth:`_write_arrow_batches` plus the Spark-native
  :meth:`_read_spark_frame` / :meth:`_write_spark_frame` so the
  holder fits anywhere the IO surface expects a Tabular.
* Rich :class:`pyspark.sql.DataFrame` wrapper — schema-aware
  ``map`` / ``apply`` / ``filter`` / ``explode`` / ``cast`` over
  :meth:`pyspark.sql.DataFrame.mapInArrow`, executor module
  auto-shipping (yggdrasil + user-function deps), schema inference
  off dynamic-mode (pickled-object) frames.

The held DataFrame is mutable: writes replace it (OVERWRITE) or
union to it (APPEND). :meth:`read_spark_frame` returns the held
frame untouched (no driver collect); :meth:`read_arrow_batches`
falls back to ``df.toArrow().to_batches()`` (which does collect to
the driver — fine when the frame is small enough, but check before
reaching for it in a hot path).

Executor cache
--------------

:meth:`persist` leverages Spark's own
:meth:`pyspark.sql.DataFrame.persist` with
:data:`pyspark.StorageLevel.MEMORY_AND_DISK` so the partitions land
on the executors' memory and spill to executor-local disk under
pressure. The call is idempotent — when
:attr:`pyspark.sql.DataFrame.is_cached` is already ``True`` the
persist is skipped silently. :meth:`unpersist` mirrors the path:
it calls :meth:`pyspark.sql.DataFrame.unpersist` first (when the
frame was cached) before dropping the local reference, so an
intentional ``persist → ... → unpersist`` round trip cleans up
the executor cache too.

Dynamic vs typed
----------------

Two modes, distinguished by :attr:`schema`:

* **Dynamic** (``schema is None``) — the underlying Spark frame
  carries the single-column ``_pickle`` schema and rows are
  arbitrary pickled Python objects. Transforms (``map``, ``filter``,
  ``apply``, ``explode``) see unpickled inner objects.
* **Typed** (``schema`` set) — the underlying Spark frame matches
  ``schema.to_spark_schema()``. Transforms receive ``dict`` rows
  and outputs are cast back through the ``Schema.cast_arrow`` chain
  driven by ``mapInArrow``.

Backwards-compat aliases
------------------------

:class:`Dataset` (this module) and ``yggdrasil.spark.frame.Dataset``
are kept as aliases for :class:`Dataset` so external imports of the
legacy spellings keep resolving to the same class. ``self.df`` is a
property over the underlying ``frame`` slot so call sites using either
``frame=`` / ``df=`` spelling keep working.
"""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Iterable,
    Iterator,
    Optional,
)

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.io.tabular.base import Tabular
from yggdrasil.data.enums import MimeType, Mode

if TYPE_CHECKING:
    from pyspark import StorageLevel
    from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
    from yggdrasil.data.schema import Schema
    from yggdrasil.execution.expr import Predicate


__all__ = ["Dataset", "SparkTabular"]


logger = logging.getLogger(__name__)


class Dataset(Tabular[CastOptions]):
    """:class:`Tabular` + Spark-DataFrame surface in one class.

    The frame is the holder's only state; reads of
    :meth:`_read_spark_frame` return it as-is, writes mutate it in
    place. The Spark session is cached off the frame on construction
    (or set explicitly) so an empty buffer can still synthesize an
    empty DataFrame on read.
    """

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    @classmethod
    def default_media_type(cls) -> Optional[MimeType]:
        return None  # not registered, see ArrowTabular for the rationale

    def __init__(
        self,
        frame: Optional["SparkDataFrame"] = None,
        schema: "Schema | None" = None,
        *,
        df: Optional["SparkDataFrame"] = None,
        spark: Optional["SparkSession"] = None,
        installed_modules: "set[str] | None" = None,
    ) -> None:
        """Wrap a Spark DataFrame, optionally with a yggdrasil schema.

        Two accepted spellings: ``Dataset(frame=df)`` (the
        Tabular-style argument name) and ``Dataset(df=df)`` (the
        legacy ``Dataset`` spelling). Passing both raises.
        """
        super().__init__()
        if frame is not None and df is not None:
            raise TypeError(
                f"{type(self).__name__} got both ``frame`` and ``df``; "
                "pass exactly one (they're aliases)."
            )
        held = frame if frame is not None else df

        self._frame: Optional["SparkDataFrame"] = held
        self._spark: Optional["SparkSession"] = spark
        # Schema lives on :attr:`Tabular._schema_cache` — the base
        # class already owns the slot, the sentinel-vs-value protocol
        # (``...`` = no schema declared, anything else = the declared
        # / inferred yggdrasil Schema), and the
        # :meth:`_persist_schema` / :meth:`_unpersist_schema` writers.
        # The :attr:`schema` property below reads it back as
        # ``None`` when the sentinel is set so the legacy ``Dataset``
        # "schema is None means dynamic mode" contract survives the
        # merge without a parallel slot.
        if schema is not None:
            self._persist_schema(schema)
        # Top-level package names this frame has already declared on
        # the cluster. Auto-populated when :meth:`apply` / :meth:`map`
        # / :meth:`filter` scan a user function's globals and feed
        # them through :meth:`_ensure_installed`. Persists across
        # transforms so a chain like ``df.apply(f).map(g).filter(h)``
        # only round-trips each module once per frame lineage.
        self.installed_modules: set[str] = (
            set(installed_modules) if installed_modules else set()
        )
        if held is not None and self._spark is None:
            # Cache the session off the frame so subsequent
            # empty-frame reads / writes don't have to rediscover it.
            self._spark = getattr(held, "sparkSession", None)

    def __repr__(self) -> str:
        if self._frame is None:
            return f"{type(self).__name__}(frame=None)"
        return f"{type(self).__name__}(frame={self._frame!r})"

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def frame(self) -> Optional["SparkDataFrame"]:
        """Currently-held Spark DataFrame, or ``None`` when empty."""
        return self._frame

    @frame.setter
    def frame(self, value: Optional["SparkDataFrame"]) -> None:
        self._frame = value
        if value is not None and self._spark is None:
            self._spark = getattr(value, "sparkSession", None)

    @property
    def df(self) -> Optional["SparkDataFrame"]:
        """Alias for :attr:`frame` — keeps the legacy ``Dataset``
        spelling working without forking the API surface."""
        return self._frame

    @df.setter
    def df(self, value: Optional["SparkDataFrame"]) -> None:
        self.frame = value

    @property
    def schema(self) -> "Schema | None":
        """Yggdrasil :class:`Schema` describing the frame, when set.

        Reads through :attr:`Tabular._schema_cache` (the base-class
        slot) and surfaces the ``...`` sentinel as ``None`` so the
        legacy ``Dataset`` "``schema is None`` means dynamic mode"
        contract still holds.
        """
        cached = self._schema_cache
        return None if cached is ... else cached

    @schema.setter
    def schema(self, value: "Schema | None") -> None:
        if value is None:
            self._unpersist_schema()
        else:
            self._persist_schema(value)

    @property
    def is_dynamic(self) -> bool:
        """``True`` iff this holder is in dynamic (pickled-object) mode."""
        return self._schema_cache is ...

    @property
    def spark_schema(self):
        """The underlying Spark schema, or ``None`` when no frame held."""
        return self._frame.schema if self._frame is not None else None

    @property
    def sparkSession(self) -> "SparkSession":
        """Bound :class:`SparkSession`. Raises when no session is reachable."""
        return self._require_spark()

    @property
    def spark(self) -> Optional["SparkSession"]:
        return self._spark

    def is_empty(self) -> bool:
        return self._frame is None

    def __bool__(self) -> bool:
        return self._frame is not None

    def __iter__(self) -> Iterator[Any]:
        return self.to_local_iterator()

    def _count(self, options=None) -> int:
        if self._frame is None:
            return 0
        df = options.cast_spark_tabular(self._frame) if options is not None else self._frame
        return df.count()

    # ------------------------------------------------------------------
    # DataFrame proxy — fall through to the underlying frame for any
    # attribute we don't define ourselves. Wraps DataFrame results so
    # chained ``.select(...).groupBy(...).agg(...)`` stays inside
    # :class:`Dataset`.
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup fails. Forward to the held
        # DataFrame so users keep ``.select`` / ``.groupBy`` / ``.where``
        # without us re-implementing each one.
        if name.startswith("_") or self._frame is None:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        attr = getattr(self._frame, name)
        if callable(attr):
            return _ProxiedCallable(attr, owner=self)
        return _wrap(attr, schema=self.schema, owner=self)

    # ------------------------------------------------------------------
    # Tabular contract — cache & persist
    # ------------------------------------------------------------------

    @property
    def cached(self) -> bool:
        return self._frame is not None

    @property
    def is_cached(self) -> bool:
        """``True`` iff the underlying Spark frame is currently persisted.

        Mirrors :attr:`pyspark.sql.DataFrame.is_cached` and answers
        ``False`` whenever no frame is held — there's nothing to
        cache. :meth:`persist` skips when this is already ``True``.
        """
        frame = self._frame
        if frame is None:
            return False
        return bool(getattr(frame, "is_cached", False))

    def cache(self):
        """Cache the underlying frame on Spark executors."""
        try:
            self._frame = self._frame.cache()
        except Exception:
            # Spark Connect can reject ``cache`` on a logical-plan node
            # it can't materialize (e.g. an unbound stream). Don't let
            # a best-effort cache cripple the caller.
            logger.warning(
                "Caching %r failed; continuing without executor cache",
                self,
                exc_info=True,
            )
        return self

    def persist(
        self,
        engine: str = "auto",
        *,
        data: Any = None,
        storage_level: "StorageLevel | str | None" = None,
    ) -> "Dataset":
        """Cache the underlying frame on Spark executors.

        ``data=`` replaces the held frame first (legacy stash path
        used by the ``Statement`` materialization shims).
        ``storage_level=`` overrides the default
        :data:`pyspark.StorageLevel.MEMORY_AND_DISK` — pass
        ``"MEMORY_ONLY"`` / ``"DISK_ONLY"`` / a real ``StorageLevel``
        when the workload calls for it.

        **Skip-if-cached:** when
        :attr:`pyspark.sql.DataFrame.is_cached` already reads
        ``True`` (or no frame is held) the persist is a no-op. So a
        chain like ``df.persist().persist().persist()`` triggers one
        cache and the rest are free. Same skip path on a holder
        whose frame was persisted upstream — we don't double-stamp
        the executor-side cache.
        """
        del engine  # spark holder has only one engine
        if data is not None:
            self.frame = self._coerce_frame(data)
        frame = self._frame
        if frame is None:
            return self
        if getattr(frame, "is_cached", False):
            logger.debug(
                "Skipping persist on %r — frame already cached",
                self,
            )
            return self
        level = self._resolve_storage_level(storage_level)
        try:
            frame.persist(level) if level is not None else frame.persist()
        except Exception:
            return self.cache()
        else:
            logger.debug(
                "Persisted %r (storage_level=%r)",
                self,
                level,
            )
        return self

    def unpersist(self) -> None:
        """Drop the held frame and release any executor-side cache.

        When :attr:`is_cached` is true,
        :meth:`pyspark.sql.DataFrame.unpersist` is called first so
        the partitions are evicted from executor memory / disk.
        Failures during unpersist are swallowed (best-effort), then
        the local reference is dropped unconditionally — calling
        ``unpersist`` on an already-empty holder is fine.
        """
        frame = self._frame
        if frame is not None and getattr(frame, "is_cached", False):
            try:
                frame.unpersist()
            except Exception:
                logger.debug(
                    "Unpersist on %r failed; dropping local ref anyway",
                    self,
                    exc_info=True,
                )
        self._frame = None

    # ------------------------------------------------------------------
    # Spark read / write — no driver collect on the spark path
    # ------------------------------------------------------------------

    def _native_spark_frame(self) -> "SparkDataFrame | None":
        # Advertises the held frame so :meth:`Tabular._write_table` can
        # skip the Arrow round-trip when writing this :class:`Dataset`
        # into a Spark-aware sink (another :class:`Dataset`, a
        # :class:`databricks.table.Table` — both override
        # :meth:`_write_spark_frame` to keep the data on the executors).
        return self._frame

    def _read_spark_frame(self, options: CastOptions) -> "SparkDataFrame":
        if self._frame is None:
            spark = self._require_spark()
            schema = options.merged
            spark_schema = schema.to_spark_schema() if schema is not None else None
            return spark.createDataFrame([], schema=spark_schema)
        # Cast first (schema alignment), then apply the read-time
        # passes (resample + dedup) natively on the Spark frame so
        # the read path doesn't have to detour through
        # ``df.toArrow → arrow.ops → createDataFrame`` to honor
        # ``unique_by`` / ``time_sample_by``. The fast path runs
        # entirely inside Spark — ``groupBy + applyInArrow`` for the
        # partitioned resample, ``row_number() OVER (...)`` for dedup.
        frame = options.cast_spark_tabular(self._frame)
        return options.apply_post_read_spark_frame(frame)

    def _read_spark_dataset(self, options: CastOptions) -> "Dataset":
        # Source already speaks Spark — skip the
        # :meth:`Tabular.from_spark_frame` rewrap when no target schema
        # forces a recast. Returning ``self`` keeps the holder identity
        # stable across ``to_spark_dataset()`` round trips, which the
        # ``Dataset``-as-cache pattern relies on.
        target = options.target
        if target is None and not options.unique_by and not options.time_sample_by:
            return self
        if self._frame is None:
            frame = None
        else:
            frame = options.cast_spark_tabular(self._frame)
            frame = options.apply_post_read_spark_frame(frame)
        return type(self)(frame=frame, schema=target)

    def _write_spark_frame(
        self,
        frame: "SparkDataFrame",
        options: CastOptions,
    ) -> None:
        action = self._resolve_save_mode(options.mode)
        if action is Mode.IGNORE:
            return
        if action is Mode.OVERWRITE or self._frame is None:
            self.frame = frame
            return
        if action is Mode.APPEND:
            self.frame = self._frame.unionByName(
                frame,
                allowMissingColumns=True,
            )
            return
        raise NotImplementedError(
            f"{type(self).__name__}._write_spark_frame handles "
            f"OVERWRITE / APPEND / IGNORE; got {action!r}."
        )

    def _union(self, other: "Tabular", *, mode: "Mode" = ...) -> "Dataset":
        other_frame = other._native_spark_frame()
        if other_frame is None:
            merged = self.collect_schema().merge_with(
                other.collect_schema(), mode=mode,
            )
            other_frame = other.read_spark_frame(target=merged)
        self.frame = self._frame.unionByName(
            other_frame, allowMissingColumns=True,
        )
        return self

    # ------------------------------------------------------------------
    # Arrow read / write — collects on read, builds Spark on write
    # ------------------------------------------------------------------

    def stat(self):
        return self._stats

    def _read_arrow_batches(self, options: CastOptions) -> Iterator[pa.RecordBatch]:
        # Forces a driver-side collect via ``df.toArrow()`` (or the
        # PySpark 3.x fallback in :func:`spark_dataframe_to_arrow`).
        # Loud rather than silent — the call site is the one asking
        # for Arrow batches off a Spark holder.
        if self._frame is None:
            return
        from yggdrasil.spark.cast import spark_dataframe_to_arrow

        arrow_table = spark_dataframe_to_arrow(self._frame)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Dataset collected %d rows from Spark frame",
                arrow_table.num_rows,
            )
        for batch in arrow_table.to_batches(max_chunksize=options.row_size):
            yield options.cast_arrow_batch(batch)

    def _read_records(self, options: CastOptions) -> "Iterator[Any]":
        # Skip the Arrow round-trip — `toLocalIterator()` streams
        # rows from the executors one by one, so the driver memory
        # footprint stays bounded even for frames that wouldn't fit
        # in a single ``df.toArrow()`` collect.
        from yggdrasil.data.record import Record

        if self._frame is None:
            return
        yield from Record.from_spark_frame(self._frame)

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: CastOptions,
    ) -> None:
        # Build a Spark frame from the incoming Arrow stream, then
        # delegate to ``_write_spark_frame`` so OVERWRITE / APPEND /
        # IGNORE branching applies the same way.
        materialized = list(batches)
        if not materialized:
            # APPEND of nothing is a no-op; OVERWRITE of nothing
            # leaves the existing frame. Match the IPC writer's
            # behavior on an empty iterator.
            return
        from yggdrasil.spark.cast import any_to_spark_dataframe

        table = pa.Table.from_batches(materialized)
        frame = any_to_spark_dataframe(table, options=options)
        self._write_spark_frame(frame, options)

    # ------------------------------------------------------------------
    # Typed-argument projection / row-filter hooks
    #
    # Override the cross-engine defaults in :class:`Tabular` so the
    # returned ``Dataset`` carries over per-instance state
    # (``installed_modules``, the typed ``schema`` for filter — which
    # preserves the row layout). The base defaults would re-build a
    # bare ``Dataset(frame=...)`` and lose the executor-side module
    # install record, forcing a re-install on the next transform.
    # ------------------------------------------------------------------

    def _select(self, *, columns: list[str]) -> "Dataset":
        if self._frame is None:
            return self
        new_frame = self._frame.select(*columns)
        # Schema becomes the projection — leave it to the new Dataset
        # to read off ``frame.schema``. Preserving the typed schema
        # would require manual sub-setting that the engine just did
        # in the SQL planner.
        return type(self)(
            frame=new_frame,
            schema=None,
            installed_modules=self.installed_modules,
        )

    def _drop(self, *, columns: list[str]) -> "Dataset":
        if self._frame is None:
            return self
        present = [c for c in columns if c in self._frame.columns]
        new_frame = self._frame.drop(*present) if present else self._frame
        return type(self)(
            frame=new_frame,
            schema=None,
            installed_modules=self.installed_modules,
        )

    def _filter(self, *, predicate: "Predicate") -> "Dataset":
        if self._frame is None:
            return self
        new_frame = predicate.filter_spark_frame(self._frame)
        return type(self)(
            frame=new_frame,
            schema=self.schema,
            installed_modules=self.installed_modules,
        )

    # ==================================================================
    # ``Dataset``-style constructors
    # ==================================================================

    @classmethod
    def from_spark_frame(
        cls,
        df: "SparkDataFrame",
        schema: "Schema | None" = None,
    ) -> "Dataset":
        """Wrap a Spark frame, optionally re-casting it against ``schema``.

        ``schema=None`` infers a yggdrasil :class:`Schema` from the
        Spark frame's schema. A non-``None`` ``schema`` first runs the
        frame through :meth:`Schema.cast_spark_tabular` so the held
        frame matches the declared shape.
        """
        from yggdrasil.data.schema import Schema as _Schema

        if schema is None:
            schema = _Schema.from_(df)
        else:
            schema = _Schema.from_any(schema)
            df = schema.cast_spark_tabular(df)
        instance = cls(frame=df, schema=schema)
        logger.debug(
            "Created Spark dataset %r from Spark frame (schema=%r)",
            instance,
            schema,
        )
        return instance

    @classmethod
    def from_iterable(
        cls,
        items: "Iterable[Any]",
        schema: "Schema | None" = None,
        *,
        spark_session: Optional["SparkSession"] = None,
        byte_size: int = 128 * 1024 * 1024,
    ) -> "Dataset":
        """Build a frame from an in-memory iterable.

        ``schema=None`` pickles each element into a dynamic frame.
        ``schema=<Schema>`` casts the iterable on the driver and
        returns a typed frame whose underlying ``DataFrame`` matches
        ``schema``.
        """
        from yggdrasil.environ import PyEnv
        from yggdrasil.arrow.cast import any_to_arrow_table
        from yggdrasil.data.schema import Schema as _Schema
        from yggdrasil.pickle.ser import dumps
        from yggdrasil.spark.frame import DYNAMIC_SCHEMA

        if spark_session is None:
            spark_session = PyEnv.spark_session(
                create=True,
                install_spark=False,
                import_error=True,
            )

        if schema is None:
            cls._ensure_installed_on_session(spark_session)
            # Materialize before handing to Spark — Spark Connect's
            # ``createDataFrame`` indexes ``_data[0]`` to sniff the
            # shape, which IndexErrors on an empty generator.
            rows = [(dumps(x),) for x in items]
            logger.debug(
                "Creating dynamic Spark dataset from iterable (rows=%d)",
                len(rows),
            )
            df = spark_session.createDataFrame(
                rows,
                schema=DYNAMIC_SCHEMA.to_spark_schema(),
            )
            instance = cls(frame=df, schema=None)
            logger.info(
                "Created dynamic Spark dataset %r (rows=%d)",
                instance,
                len(rows),
            )
            return instance

        cls._ensure_installed_on_session(spark_session)
        schema = _Schema.from_any(schema)
        table = any_to_arrow_table(
            items,
            options=CastOptions(target=schema, safe=False, byte_size=byte_size),
        )
        logger.debug(
            "Creating typed Spark dataset from iterable (rows=%d, schema=%r)",
            table.num_rows,
            schema,
        )
        instance = cls(
            frame=spark_session.createDataFrame(table),
            schema=schema,
        )
        logger.info(
            "Created typed Spark dataset %r (rows=%d)",
            instance,
            table.num_rows,
        )
        return instance

    @classmethod
    def from_sql(
        cls,
        text: str,
        *,
        spark_session: Optional["SparkSession"] = None,
        schema: "Schema | None" = None,
    ) -> "Dataset":
        """Execute SQL and wrap the resulting Spark frame as a :class:`Dataset`.

        Resolves a :class:`SparkSession` through the environment when
        none is passed — in a Databricks context this picks up the
        active notebook / Connect / Job session automatically.
        """
        if spark_session is None:
            from yggdrasil.environ import PyEnv

            spark_session = PyEnv.spark_session(
                create=True,
                install_spark=False,
                import_error=True,
            )
        df = spark_session.sql(text)
        return cls.from_spark_frame(df, schema=schema)

    @classmethod
    def from_table(
        cls,
        name: str,
        *,
        spark_session: Optional["SparkSession"] = None,
        schema: "Schema | None" = None,
    ) -> "Dataset":
        """Read a table by its fully-qualified name (``catalog.schema.table``).

        Thin wrapper around ``spark.table(name)`` that auto-resolves
        the session from the environment.
        """
        if spark_session is None:
            from yggdrasil.environ import PyEnv

            spark_session = PyEnv.spark_session(
                create=True,
                install_spark=False,
                import_error=True,
            )
        df = spark_session.table(name)
        return cls.from_spark_frame(df, schema=schema)

    def to_table(
        self,
        name: str,
        *,
        mode: str = "overwrite",
        format: str = "delta",
        partition_by: "list[str] | None" = None,
    ) -> "Dataset":
        """Write the held frame to a Unity Catalog / Spark table.

        Returns ``self`` so the call can be chained::

            (Dataset.from_sql("SELECT ...")
             .map(transform)
             .to_table("main.curated.output"))
        """
        if self._frame is None:
            raise ValueError(
                "Cannot write an empty Dataset to table %r; "
                "the underlying frame is None." % name
            )
        writer = self._frame.write.format(format).mode(mode)
        if partition_by:
            writer = writer.partitionBy(*partition_by)
        writer.saveAsTable(name)
        logger.info("Wrote dataset %r to table %r (mode=%s)", self, name, mode)
        return self

    @classmethod
    def parallelize(
        cls,
        inputs: "Iterable[Any]",
        function: "Callable[..., Any] | None" = None,
        schema: "Schema | None" = None,
        *,
        spark_session: Optional["SparkSession"] = None,
        byte_size: int = 128 * 1024 * 1024,
    ) -> "Dataset":
        """Distribute ``function`` over ``inputs`` via ``mapInArrow``,
        or create a frame directly from ``inputs`` when no function is
        given.

        Two call shapes:

        * ``parallelize(inputs, function, schema=...)`` — apply
          *function* to each element of *inputs* on Spark executors.
        * ``parallelize(inputs, schema=...)`` — wrap *inputs* as a
          :class:`Dataset` (delegates to :meth:`from_iterable`).

        Per-input dispatch goes through
        :func:`yggdrasil.dataclasses.build_row_invoker` so any
        pyfunc shape is accepted (single-arg, multi-arg, ``**kwargs``,
        ``*args``); dict inputs spread as kwargs, list/tuple inputs
        spread as positional args when the function declares a
        ``*args`` catch-all. ``schema=None`` returns a dynamic frame
        of pickled outputs; ``schema=<Schema>`` casts outputs and
        returns a typed frame.
        """
        if function is None:
            return cls.from_iterable(
                inputs,
                schema=schema,
                spark_session=spark_session,
                byte_size=byte_size,
            )
        from yggdrasil.environ import PyEnv
        from yggdrasil.data.schema import Schema as _Schema
        from yggdrasil.dataclasses.safe_function import build_row_invoker
        from yggdrasil.pickle.ser import dumps
        from yggdrasil.spark.frame import (
            DYNAMIC_SCHEMA,
            _dynamic_rows,
            _emit_pickled,
            _typed_cast,
        )

        if spark_session is None:
            spark_session = PyEnv.spark_session(
                create=True,
                install_spark=False,
                import_error=True,
            )

        installed_modules = cls._ensure_installed_on_session(
            spark_session,
            function,
        )
        dumped = [(dumps(x),) for x in inputs]
        function_pickle = dumps(function)
        logger.debug(
            "Creating Spark dataset via parallelize (inputs=%d, "
            "function=%r, schema=%r)",
            len(dumped),
            function,
            schema,
        )
        input_df = spark_session.createDataFrame(
            dumped,
            schema=DYNAMIC_SCHEMA.to_spark_schema(),
            verifySchema=False,
        )

        if schema is None:

            def _runner(
                batches: "Iterator[pa.RecordBatch]",
            ) -> "Iterator[pa.RecordBatch]":
                from yggdrasil.pickle.ser import loads

                invoke = build_row_invoker(loads(function_pickle))
                yield from _emit_pickled(
                    (invoke(obj) for obj in _dynamic_rows(batches)),
                    byte_size=byte_size,
                )

            result_df = input_df.mapInArrow(
                _runner,
                schema=DYNAMIC_SCHEMA.to_spark_schema(),
            )
            return cls(
                frame=result_df,
                schema=None,
                installed_modules=installed_modules,
            )

        schema = _Schema.from_any(schema)

        def _typed_runner(
            batches: "Iterator[pa.RecordBatch]",
        ) -> "Iterator[pa.RecordBatch]":
            from yggdrasil.pickle.ser import loads

            invoke = build_row_invoker(loads(function_pickle))

            def _groups() -> "Iterator[list[Any]]":
                for batch in batches:
                    col = batch.column(0)
                    n = batch.num_rows
                    if n == 0:
                        continue
                    yield [invoke(loads(col[i].as_py())) for i in range(n)]

            return _typed_cast(_groups(), schema, byte_size=byte_size)

        result_df = input_df.mapInArrow(
            _typed_runner,
            schema=schema.to_spark_schema(),
        )
        return cls(
            frame=result_df,
            schema=schema,
            installed_modules=installed_modules,
        )

    # ==================================================================
    # Schema inference
    # ==================================================================

    def infer_schema(
        self,
        *,
        limit: "int | None" = None,
        force: bool = False,
        inplace: bool = True,
    ) -> "Schema":
        """Infer a yggdrasil :class:`Schema` from the row contents.

        Dynamic mode: each row is unpickled and shape-inferred via
        :meth:`Schema.from_`; per-partition schemas are merged in
        ``APPEND`` mode (union of fields, widening of nullability),
        then folded on the driver into the final schema.

        Typed mode: returns :attr:`schema` unchanged unless
        ``force=True``, in which case the underlying batches are
        re-inferred from row dicts — useful after a heterogeneous
        transform whose output schema is looser than the declared one.
        """
        from yggdrasil.data.schema import Schema as _Schema
        from yggdrasil.pickle.ser import dumps, loads
        from yggdrasil.spark.frame import (
            DYNAMIC_SCHEMA,
            PICKLE_COLUMN_NAME,
            _ARROW_DYNAMIC_SCHEMA,
        )

        if self._frame is None:
            raise ValueError("Cannot infer schema from an empty frame.")

        if not self.is_dynamic and not force:
            return self.schema

        # ---- sample path: drive the inference locally -----------------
        if limit is not None:
            df = self._frame.limit(limit)
            merged: "Schema | None" = None
            if self.is_dynamic:
                for row in df.toLocalIterator():
                    shape = _Schema.from_(loads(row[PICKLE_COLUMN_NAME]))
                    merged = (
                        shape
                        if merged is None
                        else merged.merge_with(
                            shape,
                            mode=Mode.APPEND,
                        )
                    )
            else:
                for row in df.toLocalIterator():
                    shape = _Schema.from_(row.asDict(recursive=True))
                    merged = (
                        shape
                        if merged is None
                        else merged.merge_with(
                            shape,
                            mode=Mode.APPEND,
                        )
                    )
            if merged is None:
                raise ValueError("Cannot infer schema from an empty frame.")
            return merged

        # ---- full-scan path: per-partition inference via mapInArrow ---
        self._ensure_installed()
        is_dynamic_in = self.is_dynamic

        def _runner(batches: "Iterator[pa.RecordBatch]") -> "Iterator[pa.RecordBatch]":
            partition_schema: "Schema | None" = None

            if is_dynamic_in:
                for batch in batches:
                    col = batch.column(0)
                    for i in range(batch.num_rows):
                        shape = _Schema.from_(loads(col[i].as_py()))
                        partition_schema = (
                            shape
                            if partition_schema is None
                            else partition_schema.merge_with(shape, mode=Mode.APPEND)
                        )
            else:
                for batch in batches:
                    for row in batch.to_pylist():
                        shape = _Schema.from_(row)
                        partition_schema = (
                            shape
                            if partition_schema is None
                            else partition_schema.merge_with(shape, mode=Mode.APPEND)
                        )

            if partition_schema is not None:
                yield pa.RecordBatch.from_pylist(
                    [{PICKLE_COLUMN_NAME: dumps(partition_schema)}],
                    schema=_ARROW_DYNAMIC_SCHEMA,
                )

        schemas_df = self._frame.mapInArrow(
            _runner,
            schema=DYNAMIC_SCHEMA.to_spark_schema(),
        )

        merged = None
        for row in schemas_df.toLocalIterator():
            partition_schema = loads(row[PICKLE_COLUMN_NAME])
            merged = (
                partition_schema
                if merged is None
                else merged.merge_with(partition_schema, mode=Mode.APPEND)
            )

        if merged is None:
            raise ValueError("Cannot infer schema from an empty frame.")

        if inplace:
            self._persist_schema(merged)

        return merged

    # ==================================================================
    # Executor dependency wiring
    # ==================================================================

    @classmethod
    def _ensure_installed_on_session(
        cls,
        session: "SparkSession",
        *functions: "Callable[..., Any]",
    ) -> "set[str]":
        """Auto-ship ygg (+ any function deps) to executors on first use."""
        from yggdrasil.spark.frame import (
            _PER_SESSION_INSTALLED_MODULES,
            _function_top_modules,
            _install_modules_on_executors,
        )

        cache = _PER_SESSION_INSTALLED_MODULES.setdefault(id(session), set())
        wanted: set[str] = {"yggdrasil"}
        for fn in functions:
            if fn is None:
                continue
            wanted.update(_function_top_modules(fn))

        new_modules = wanted - cache
        if not new_modules:
            return set(cache)

        logger.debug(
            "Installing modules on Spark executors %r (modules=%r)",
            session,
            sorted(new_modules),
        )
        try:
            installed = _install_modules_on_executors(session, new_modules)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "%s: failed to install %s on executors: %s",
                cls.__name__,
                sorted(new_modules),
                exc,
            )
            return set(cache)

        cache.update(installed)
        if installed:
            logger.info(
                "Installed modules on Spark executors %r (modules=%r)",
                session,
                sorted(installed),
            )
        return set(cache)

    def _ensure_installed(self, *functions: "Callable[..., Any]") -> "set[str]":
        """Per-frame wrapper around :meth:`_ensure_installed_on_session`."""
        installed = self._ensure_installed_on_session(
            self.sparkSession,
            *functions,
        )
        new = installed - self.installed_modules
        self.installed_modules.update(installed)
        return new

    # ==================================================================
    # Transforms (Dataset surface)
    # ==================================================================

    def map(
        self,
        function: "Callable[..., Any]",
        schema: "Schema | None" = None,
        *,
        byte_size: int = 128 * 1024 * 1024,
    ) -> "Dataset":
        """1:1 map over rows.

        Input rows are unpickled objects (dynamic mode) or row-dicts
        (typed mode). The function's signature drives the row-shape
        adaptation via :func:`yggdrasil.dataclasses.build_row_invoker`:
        single-arg functions get the row directly (or
        ``row[arg_name]`` when the arg name matches a key in a dict
        row), multi-arg / ``**kwargs`` functions get the dict spread
        as kwargs, ``*args`` catch-alls get sequence rows spread
        positionally, and annotated parameters are coerced through
        the :func:`yggdrasil.data.cast.convert` registry. Typed-mode
        batches go through
        :func:`yggdrasil.dataclasses.build_batch_invoker` so a
        single-positional-by-name function gets a vectorised column
        cast. Output schema follows ``schema`` if given, else the
        result is a dynamic frame.
        """
        from yggdrasil.data.schema import Schema as _Schema
        from yggdrasil.dataclasses.safe_function import (
            build_batch_invoker,
            build_row_invoker,
        )
        from yggdrasil.pickle.ser import dumps, loads
        from yggdrasil.spark.frame import (
            DYNAMIC_SCHEMA,
            _dynamic_rows,
            _emit_pickled,
            _typed_cast,
            _typed_rows,
        )

        self._ensure_installed(function)
        function_pickle = dumps(function)
        is_dynamic_in = self.is_dynamic

        if schema is None:

            def _runner(
                batches: "Iterator[pa.RecordBatch]",
            ) -> "Iterator[pa.RecordBatch]":
                invoke = build_row_invoker(loads(function_pickle))
                rows = _dynamic_rows(batches) if is_dynamic_in else _typed_rows(batches)
                yield from _emit_pickled(
                    (invoke(row) for row in rows),
                    byte_size=byte_size,
                )

            result_df = self._frame.mapInArrow(
                _runner,
                schema=DYNAMIC_SCHEMA.to_spark_schema(),
            )
            return type(self)(
                frame=result_df,
                schema=None,
                installed_modules=self.installed_modules,
            )

        schema = _Schema.from_any(schema)

        def _typed_runner(
            batches: "Iterator[pa.RecordBatch]",
        ) -> "Iterator[pa.RecordBatch]":
            func = loads(function_pickle)

            def _groups() -> "Iterator[list[Any]]":
                if is_dynamic_in:
                    invoke_row = build_row_invoker(func)
                    for batch in batches:
                        col = batch.column(0)
                        n = batch.num_rows
                        if n == 0:
                            continue
                        yield [invoke_row(loads(col[i].as_py())) for i in range(n)]
                else:
                    invoke_batch = build_batch_invoker(func)
                    for batch in batches:
                        if batch.num_rows == 0:
                            continue
                        yield invoke_batch(batch)

            return _typed_cast(_groups(), schema, byte_size=byte_size)

        result_df = self._frame.mapInArrow(
            _typed_runner,
            schema=schema.to_spark_schema(),
        )
        return type(self)(
            frame=result_df,
            schema=schema,
            installed_modules=self.installed_modules,
        )

    def apply(
        self,
        function: "Callable[..., Any]",
        schema: "Schema | None" = None,
        *,
        byte_size: int = 128 * 1024 * 1024,
    ) -> "Dataset":
        """Map ``function`` over each row, optionally casting against ``schema``.

        ``function`` may carry any signature — single-arg
        (``def f(row): ...``), multi-arg
        (``def f(id: int, name: str): ...``), ``**kwargs`` catch-all
        (``def f(**row): ...``), or a ``*args`` catch-all over
        tuple/list rows. Per-row dispatch is built once per
        partition via
        :func:`yggdrasil.dataclasses.build_row_invoker`, which:

        * passes the row directly when ``function`` has one
          positional parameter,
        * extracts ``row[arg_name]`` when the row is a mapping and
          the single positional arg name matches a column key —
          ``def f(id: int)`` on a ``{"id": ..., "name": ...}`` row
          gets the ``id`` value, not the whole dict,
        * spreads mapping rows as ``**kwargs`` (filtered to
          declared names, unless ``**kwargs`` catches the rest)
          when ``function`` has multiple named parameters,
        * spreads sequence rows as ``*args`` when ``function``
          declares a ``*args`` catch-all and no other positional,
        * coerces annotated parameters through
          :func:`yggdrasil.data.cast.convert` so string-shaped
          inputs reach the function as the annotated Python type.

        Typed-mode batches additionally route through
        :func:`yggdrasil.dataclasses.build_batch_invoker`, which picks
        the cheapest dispatch shape per batch:

        * **whole-batch tabular** — ``function`` annotated
          ``def f(batch: pa.RecordBatch)`` / ``pa.Table`` /
          ``pl.DataFrame`` / ``pl.LazyFrame`` / ``pd.DataFrame``
          receives the whole batch converted to that type in one
          call (zero per-row Python overhead),
        * **column-by-name vectorised cast** — single-positional
          annotated + arg name matches a column → the column is
          cast via :func:`pyarrow.compute.cast` (one C++ kernel
          call) and ``function`` runs per cell,
        * **per-row fallback** — every other shape walks
          ``batch.to_pylist()`` rows through the row invoker.

        Without a schema this is :meth:`map`. With a schema,
        ``function`` may return any tabular shape (dict, dataclass,
        list-of-rows, polars/pandas/arrow frame, ``pa.RecordBatch``);
        outputs are streamed through :func:`any_to_arrow_batch_iterator`
        in one pass.
        """
        from yggdrasil.data.schema import Schema as _Schema
        from yggdrasil.dataclasses.safe_function import (
            build_batch_invoker,
            build_row_invoker,
        )
        from yggdrasil.pickle.ser import dumps, loads
        from yggdrasil.spark.frame import _typed_cast

        if schema is None:
            return self.map(function, byte_size=byte_size)

        self._ensure_installed(function)
        schema = _Schema.from_any(schema)
        function_pickle = dumps(function)
        is_dynamic_in = self.is_dynamic

        def _runner(batches: "Iterator[pa.RecordBatch]") -> "Iterator[pa.RecordBatch]":
            func = loads(function_pickle)

            def _groups() -> "Iterator[list[Any]]":
                if is_dynamic_in:
                    # Rows are pickled bytes — unpickle each one and dispatch
                    # via the per-row invoker. ``build_batch_invoker`` would
                    # have nothing to vectorise on a single binary column.
                    invoke_row = build_row_invoker(func)
                    for batch in batches:
                        col = batch.column(0)
                        n = batch.num_rows
                        if n == 0:
                            continue
                        yield [invoke_row(loads(col[i].as_py())) for i in range(n)]
                else:
                    # Typed mode — let the batch invoker pick the column
                    # by arg name and vectorise the cast when possible.
                    invoke_batch = build_batch_invoker(func)
                    for batch in batches:
                        if batch.num_rows == 0:
                            continue
                        yield invoke_batch(batch)

            return _typed_cast(_groups(), schema, byte_size=byte_size)

        result_df = self._frame.mapInArrow(
            _runner,
            schema=schema.to_spark_schema(),
        )
        return type(self)(
            frame=result_df,
            schema=schema,
            installed_modules=self.installed_modules,
        )

    def filter(
        self,
        predicate: "Callable[[Any], bool] | Any",
        schema: "Schema | None" = None,
        *,
        byte_size: int = 128 * 1024 * 1024,
    ) -> "Dataset":
        """Drop rows where *predicate* is false.

        Two shapes, dispatched by the predicate's runtime type:

        * **Predicate-like** — a SQL string, a yggdrasil
          :class:`Expression` / :class:`Predicate` node, or a native
          engine expression (:class:`pyarrow.compute.Expression`,
          :class:`polars.Expr`, :class:`pyspark.sql.Column`). Routes
          through the typed :meth:`_filter` hook; the cross-engine
          :meth:`Tabular.filter` contract — the predicate compiles
          to a :class:`pyspark.sql.Column` and Spark's Catalyst
          plans the filter natively.
        * **Pure callable** (the legacy :class:`Dataset` shape) —
          ``predicate(row)`` evaluated per row inside a
          :meth:`mapInArrow` worker; sees unpickled objects on a
          dynamic frame or row-dicts on a typed frame. Kept for
          backwards compatibility — row-by-row callables are the
          last-resort path when no vectorised predicate fits.

        When called on a typed frame without a ``schema`` argument
        the existing schema is preserved (no re-cast needed); the
        ``byte_size`` cap only applies to the callable path.
        """
        from yggdrasil.execution.expr import Expression

        # Predicate-like inputs route to the cross-engine
        # :meth:`Tabular.filter`. Recognised: yggdrasil Expression,
        # SQL string, or native engine expression (pyarrow / polars /
        # pyspark) — the module-name sniff matches
        # :meth:`Expression.from_`. ``callable(predicate)`` is the
        # remaining signal: anything that isn't a real Python
        # callable can't be the legacy row-predicate path.
        if isinstance(predicate, (Expression, str)) or not callable(predicate):
            module = (type(predicate).__module__ or "").split(".", 1)[0]
            if isinstance(predicate, (Expression, str)) or module in {
                "pyarrow",
                "polars",
                "pyspark",
            }:
                return super().filter(predicate)

        from yggdrasil.data.schema import Schema as _Schema
        from yggdrasil.pickle.ser import dumps, loads
        from yggdrasil.spark.frame import (
            DYNAMIC_SCHEMA,
            PICKLE_COLUMN_NAME,
            _ARROW_DYNAMIC_SCHEMA,
            _typed_cast,
        )

        self._ensure_installed(predicate)
        predicate_pickle = dumps(predicate)
        is_dynamic_in = self.is_dynamic

        if is_dynamic_in and schema is None:

            def _runner(
                batches: "Iterator[pa.RecordBatch]",
            ) -> "Iterator[pa.RecordBatch]":
                pred = loads(predicate_pickle)
                out: "list[dict[str, bytes]]" = []
                out_bytes = 0
                for batch in batches:
                    col = batch.column(0)
                    for i in range(batch.num_rows):
                        ser = col[i].as_py()
                        if not pred(loads(ser)):
                            continue
                        if out and out_bytes + len(ser) > byte_size:
                            yield pa.RecordBatch.from_pylist(
                                out,
                                schema=_ARROW_DYNAMIC_SCHEMA,
                            )
                            out = []
                            out_bytes = 0
                        out.append({PICKLE_COLUMN_NAME: ser})
                        out_bytes += len(ser)
                if out:
                    yield pa.RecordBatch.from_pylist(
                        out,
                        schema=_ARROW_DYNAMIC_SCHEMA,
                    )

            result_df = self._frame.mapInArrow(
                _runner,
                schema=DYNAMIC_SCHEMA.to_spark_schema(),
            )
            return type(self)(
                frame=result_df,
                schema=None,
                installed_modules=self.installed_modules,
            )

        out_schema = _Schema.from_any(schema) if schema is not None else self.schema
        if out_schema is None:
            raise AssertionError("unreachable")

        def _typed_runner(
            batches: "Iterator[pa.RecordBatch]",
        ) -> "Iterator[pa.RecordBatch]":
            pred = loads(predicate_pickle)

            def _groups() -> "Iterator[list[Any]]":
                if is_dynamic_in:
                    for batch in batches:
                        col = batch.column(0)
                        kept = []
                        for i in range(batch.num_rows):
                            obj = loads(col[i].as_py())
                            if pred(obj):
                                kept.append(obj)
                        if kept:
                            yield kept
                else:
                    for batch in batches:
                        kept = [r for r in batch.to_pylist() if pred(r)]
                        if kept:
                            yield kept

            return _typed_cast(_groups(), out_schema, byte_size=byte_size)

        result_df = self._frame.mapInArrow(
            _typed_runner,
            schema=out_schema.to_spark_schema(),
        )
        return type(self)(
            frame=result_df,
            schema=out_schema,
            installed_modules=self.installed_modules,
        )

    def explode(
        self,
        schema: "Schema | None" = None,
        *,
        byte_size: int = 128 * 1024 * 1024,
    ) -> "Dataset":
        """Explode rows of iterables into one row per element.

        Only meaningful in dynamic mode — typed rows are dicts, not
        iterables. Pass a ``schema`` to type the flattened output.
        """
        from yggdrasil.data.schema import Schema as _Schema
        from yggdrasil.pickle.ser import loads
        from yggdrasil.spark.frame import (
            DYNAMIC_SCHEMA,
            _emit_pickled,
            _typed_cast,
        )

        if not self.is_dynamic:
            raise TypeError(
                "explode() is only defined on dynamic-mode frames; "
                "the inner objects must be iterable."
            )

        self._ensure_installed()

        if schema is None:

            def _runner(
                batches: "Iterator[pa.RecordBatch]",
            ) -> "Iterator[pa.RecordBatch]":
                def _items() -> "Iterator[Any]":
                    for batch in batches:
                        col = batch.column(0)
                        for i in range(batch.num_rows):
                            yield from loads(col[i].as_py())

                yield from _emit_pickled(_items(), byte_size=byte_size)

            result_df = self._frame.mapInArrow(
                _runner,
                schema=DYNAMIC_SCHEMA.to_spark_schema(),
            )
            return type(self)(
                frame=result_df,
                schema=None,
                installed_modules=self.installed_modules,
            )

        schema = _Schema.from_any(schema)

        def _typed_runner(
            batches: "Iterator[pa.RecordBatch]",
        ) -> "Iterator[pa.RecordBatch]":
            def _groups() -> "Iterator[list[Any]]":
                for batch in batches:
                    col = batch.column(0)
                    group: list[Any] = []
                    for i in range(batch.num_rows):
                        group.extend(loads(col[i].as_py()))
                    if group:
                        yield group

            return _typed_cast(_groups(), schema, byte_size=byte_size)

        result_df = self._frame.mapInArrow(
            _typed_runner,
            schema=schema.to_spark_schema(),
        )
        return type(self)(
            frame=result_df,
            schema=schema,
            installed_modules=self.installed_modules,
        )

    def cast(
        self,
        schema: "Schema",
        *,
        byte_size: int = 128 * 1024 * 1024,
    ) -> "Dataset":
        """Materialise rows against ``schema`` as a typed frame."""
        from yggdrasil.data.schema import Schema as _Schema
        from yggdrasil.pickle.ser import loads
        from yggdrasil.spark.frame import _typed_cast

        self._ensure_installed()
        schema = _Schema.from_any(schema)
        is_dynamic_in = self.is_dynamic

        def _runner(batches: "Iterator[pa.RecordBatch]") -> "Iterator[pa.RecordBatch]":
            def _groups() -> "Iterator[list[Any]]":
                if is_dynamic_in:
                    for batch in batches:
                        col = batch.column(0)
                        n = batch.num_rows
                        if n == 0:
                            continue
                        yield [loads(col[i].as_py()) for i in range(n)]
                else:
                    for batch in batches:
                        rows = batch.to_pylist()
                        if rows:
                            yield rows

            return _typed_cast(_groups(), schema, byte_size=byte_size)

        result_df = self._frame.mapInArrow(
            _runner,
            schema=schema.to_spark_schema(),
        )
        return type(self)(
            frame=result_df,
            schema=schema,
            installed_modules=self.installed_modules,
        )

    def to_dynamic(
        self,
        *,
        byte_size: int = 128 * 1024 * 1024,
    ) -> "Dataset":
        """Drop typing: re-pickle row-dicts back into a dynamic frame.

        No-op when already dynamic.
        """
        from yggdrasil.spark.frame import (
            DYNAMIC_SCHEMA,
            _emit_pickled,
            _typed_rows,
        )

        if self.is_dynamic:
            return self

        self._ensure_installed()

        def _runner(batches: "Iterator[pa.RecordBatch]") -> "Iterator[pa.RecordBatch]":
            yield from _emit_pickled(_typed_rows(batches), byte_size=byte_size)

        result_df = self._frame.mapInArrow(
            _runner,
            schema=DYNAMIC_SCHEMA.to_spark_schema(),
        )
        return type(self)(
            frame=result_df,
            schema=None,
            installed_modules=self.installed_modules,
        )

    # ==================================================================
    # Terminal ops
    # ==================================================================

    def collect(self) -> "list[Any]":
        from yggdrasil.pickle.ser import loads
        from yggdrasil.spark.frame import PICKLE_COLUMN_NAME

        if self._frame is None:
            return []
        if self.is_dynamic:
            return [loads(row[PICKLE_COLUMN_NAME]) for row in self._frame.collect()]
        return [row.asDict(recursive=True) for row in self._frame.collect()]

    def to_local_iterator(self) -> "Iterator[Any]":
        from yggdrasil.pickle.ser import loads
        from yggdrasil.spark.frame import PICKLE_COLUMN_NAME

        if self._frame is None:
            return
        if self.is_dynamic:
            for row in self._frame.toLocalIterator():
                yield loads(row[PICKLE_COLUMN_NAME])
        else:
            for row in self._frame.toLocalIterator():
                yield row.asDict(recursive=True)

    def toArrow(
        self,
        schema: "Schema | None" = None,
        *,
        byte_size: int = 128 * 1024 * 1024,
    ) -> pa.Table:
        from yggdrasil.arrow.cast import any_to_arrow_table

        bound = self.schema
        target = schema if schema is not None else bound
        if self.is_dynamic and target is None:
            return any_to_arrow_table(
                self.to_local_iterator(),
                options=CastOptions(byte_size=byte_size, safe=False),
            )
        from yggdrasil.spark.cast import spark_dataframe_to_arrow

        if target is not None and target is not bound:
            return spark_dataframe_to_arrow(
                self.cast(target, byte_size=byte_size)._frame,
            )
        return spark_dataframe_to_arrow(self._frame)

    def toPandas(
        self,
        schema: "Schema | None" = None,
        *,
        byte_size: int = 128 * 1024 * 1024,
    ):
        return self.toArrow(schema=schema, byte_size=byte_size).to_pandas()

    def toPolars(
        self,
        schema: "Schema | None" = None,
        *,
        byte_size: int = 128 * 1024 * 1024,
    ):
        from yggdrasil.lazy_imports import polars_module

        pl = polars_module()
        return pl.from_arrow(self.toArrow(schema=schema, byte_size=byte_size))

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _resolve_save_mode(self, mode: Any) -> Mode:
        m = Mode.from_(mode, default=Mode.AUTO)
        if m in (Mode.AUTO, Mode.OVERWRITE, Mode.TRUNCATE):
            return Mode.OVERWRITE
        if m is Mode.IGNORE:
            return Mode.IGNORE if self._frame is not None else Mode.OVERWRITE
        if m is Mode.ERROR_IF_EXISTS:
            if self._frame is not None:
                raise FileExistsError(
                    f"{type(self).__name__} write with Mode.ERROR_IF_EXISTS "
                    "but buffer is non-empty."
                )
            return Mode.OVERWRITE
        if m is Mode.APPEND:
            return Mode.APPEND
        raise ValueError(
            f"{type(self).__name__} does not support Mode.{m.name}; "
            f"valid: AUTO, OVERWRITE, TRUNCATE, APPEND, IGNORE, ERROR_IF_EXISTS."
        )

    def _require_spark(self) -> "SparkSession":
        if self._spark is None:
            if self._frame is not None:
                self._spark = getattr(self._frame, "sparkSession", None)
            if self._spark is None:
                from yggdrasil.environ import PyEnv

                self._spark = PyEnv.spark_session(
                    create=True,
                    install_spark=False,
                    import_error=True,
                )
        return self._spark

    def _coerce_frame(self, value: Any) -> "SparkDataFrame":
        from yggdrasil.spark.cast import any_to_spark_dataframe

        return any_to_spark_dataframe(value)

    def _resolve_storage_level(
        self,
        value: "StorageLevel | str | None",
    ) -> "StorageLevel | None":
        """Coerce a string / ``StorageLevel`` / ``None`` argument.

        ``None`` keeps pyspark's own default for
        :meth:`pyspark.sql.DataFrame.persist`. A string is looked up
        on :class:`pyspark.StorageLevel` so callers can pass the
        familiar ``"MEMORY_AND_DISK"`` / ``"DISK_ONLY"`` /
        ``"MEMORY_ONLY"`` shorthand.
        """
        if value is None:
            try:
                from pyspark import StorageLevel
            except ImportError:
                return None
            return StorageLevel.MEMORY_AND_DISK
        if isinstance(value, str):
            from pyspark import StorageLevel

            attr = getattr(StorageLevel, value.upper(), None)
            if attr is None:
                raise ValueError(
                    f"Unknown StorageLevel {value!r}. Try one of: "
                    "MEMORY_ONLY, MEMORY_AND_DISK, DISK_ONLY, "
                    "MEMORY_ONLY_SER, MEMORY_AND_DISK_SER, "
                    "MEMORY_AND_DISK_2, MEMORY_ONLY_2, DISK_ONLY_2."
                )
            return attr
        return value


# ---------------------------------------------------------------------------
# Proxy plumbing — keeps ``Dataset.<spark-df-method>(...)`` chains
# returning :class:`Dataset`, so the user-visible type doesn't
# disappear behind a single ``.select`` call.
# ---------------------------------------------------------------------------


def _wrap(
    value: Any,
    *,
    schema: "Schema | None" = None,
    owner: "Dataset | None" = None,
) -> Any:
    """Wrap ``DataFrame`` results as :class:`Dataset`; pass others through."""
    try:
        from pyspark.sql import DataFrame as _SparkDataFrame
    except ImportError:
        return value
    if isinstance(value, _SparkDataFrame):
        # Carry the parent's schema only when the resulting DataFrame
        # still matches it — otherwise infer fresh from the new shape.
        out_schema = schema
        try:
            if out_schema is not None and out_schema.to_spark_schema() != value.schema:
                from yggdrasil.data.schema import Schema as _Schema

                out_schema = _Schema.from_any(value.schema)
        except Exception:
            from yggdrasil.data.schema import Schema as _Schema

            out_schema = _Schema.from_any(value.schema)
        installed = owner.installed_modules if owner is not None else None
        return Dataset(
            frame=value,
            schema=out_schema,
            installed_modules=installed,
        )
    return value


class _ProxiedCallable:
    """Bound-method shim that wraps DataFrame return values.

    Returned by :meth:`Dataset.__getattr__` for callable attributes
    on the underlying ``DataFrame``. Calling it forwards args/kwargs
    and runs the result through :func:`_wrap` so chained DataFrame ops
    stay inside :class:`Dataset`.
    """

    __slots__ = ("_callable", "_owner")

    def __init__(
        self,
        fn: "Callable[..., Any]",
        *,
        owner: "Dataset | None" = None,
    ) -> None:
        self._callable = fn
        self._owner = owner

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        owner = self._owner
        schema = owner.schema if owner is not None else None
        return _wrap(
            self._callable(*args, **kwargs),
            schema=schema,
            owner=owner,
        )

    def __repr__(self) -> str:
        return f"_ProxiedCallable({self._callable!r})"


# Pre-rename spelling. External callers that did
# ``from yggdrasil.spark.tabular import SparkTabular`` (or
# ``from yggdrasil.io.tabular import SparkTabular``) keep resolving
# to the same class — the alias is the same object, not a subclass,
# so ``isinstance(x, SparkTabular)`` and ``isinstance(x, Dataset)``
# are interchangeable.
SparkTabular = Dataset
