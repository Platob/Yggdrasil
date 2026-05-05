"""Spark connector for :class:`YGGFolderIO`.

Two pipes, both Arrow-native:

- **Batch** — ``YGGFolderSparkConnector.read_batch(io, spark)``
  reuses :meth:`YGGFolderIO.read_arrow_batches` (so checkpoints,
  predicates, partition pruning, and the universal predicate
  filter all flow through automatically) and pumps the resulting
  :class:`pyarrow.RecordBatch` stream into a Spark
  :class:`DataFrame` via ``mapInArrow``. The Python function runs
  on the Spark worker, but the source iterator runs against the
  driver-visible :class:`YGGFolderIO`, so the connector is most
  useful with one-partition trigger DataFrames where the driver
  is the I/O actor.

- **Stream** — ``read_stream(io, spark)`` returns a
  :class:`pyspark.sql.DataFrame` with ``isStreaming=True``,
  built from Spark's native parquet streaming source pointed at
  the folder root. Children are parquet by construction (every
  ``YGGFolderIO`` write produces parquet files); Spark's reader
  handles all the heavy lifting, including its own predicate
  pushdown.

Optional Spark Data Source registration
---------------------------------------

PySpark 4.0+ ships :mod:`pyspark.sql.datasource`, which lets a
Python class register as a first-class data source. When that's
available, :func:`register_datasource` plugs in
``"yggfolder"`` so callers can write::

    spark.read.format("yggfolder").load("/path/to/folder")

On older PySpark this gracefully no-ops; the function-based
connector still works.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional

import pyarrow as pa


if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
    from .ygg_folder_io import YGGFolderIO


__all__ = [
    "YGGFolderSparkConnector",
    "register_datasource",
]


#: Spark conf key that carries the driver's
#: :func:`yggdrasil.io.buffer._concurrency.compute_identifier_url`
#: across to executor processes. Workers pick the same value back up
#: when they read ``$YGG_OWNER_URL`` (Spark propagates ``executorEnv``
#: bindings to executor processes at task start).
_OWNER_URL_SPARK_CONF: str = "spark.yggdrasil.owner_url"
_OWNER_URL_EXECUTOR_ENV: str = "spark.executorEnv.YGG_OWNER_URL"


def _iter_leaf_paths(io: "Any") -> "Iterator[str]":
    """Recursively flatten a NestedIO tree to its tabular leaf paths.

    ``iter_children`` yields a mix of leaf :class:`TabularIO` (one
    parquet/IPC/CSV per child) and sub-:class:`NestedIO` (recursive
    sub-folders). Spark partitions read one leaf each, so we walk
    the tree once and emit only the absolute leaf paths.
    """
    from .base import NestedIO

    for child in io.iter_children():
        if isinstance(child, NestedIO):
            with child as opened:
                yield from _iter_leaf_paths(opened)
        else:
            yield child.path.full_path()


LOGGER = logging.getLogger(__name__)


def _resolve_spark_session(
    session: "SparkSession | None",
    *,
    create: bool = True,
) -> "SparkSession":
    """Return *session*, falling back to :class:`PyEnv.spark_session`.

    Local import to avoid pulling Spark into the base import
    graph — the connector is opt-in and the base IO layer must
    keep working without PySpark installed.
    """
    if session is not None:
        return session
    from yggdrasil.environ import PyEnv

    s = PyEnv.spark_session(create=create)
    if s is None:
        raise RuntimeError(
            "No active SparkSession and create=False — pass spark=... "
            "to the connector explicitly."
        )
    return s


class YGGFolderSparkConnector:
    """Pump Arrow batches from a :class:`YGGFolderIO` into Spark.

    Construction is cheap — the connector holds no per-call state.
    Build one per IO or share one and call :meth:`read_batch` /
    :meth:`read_stream` per query.
    """

    __slots__ = ("_io",)

    def __init__(self, io: "YGGFolderIO") -> None:
        from .ygg_folder_io import YGGFolderIO

        if not isinstance(io, YGGFolderIO):
            raise TypeError(
                f"YGGFolderSparkConnector requires a YGGFolderIO; got "
                f"{type(io).__name__}."
            )
        self._io = io

    @property
    def io(self) -> "YGGFolderIO":
        return self._io

    # ==================================================================
    # Owner-URL propagation — share the driver's identifier with workers
    # ==================================================================

    @staticmethod
    def driver_owner_url() -> str:
        """Compute (or return the propagated) compute-identifier URL.

        Convenience accessor — equivalent to calling
        :func:`yggdrasil.io.buffer._concurrency.compute_identifier_url`
        directly. Useful when a caller wants to capture the URL
        on the driver before invoking a Spark write so the same
        value can be plumbed to executors.
        """
        from yggdrasil.io.buffer._concurrency import compute_identifier_url

        return compute_identifier_url()

    @classmethod
    def propagate_owner_url(
        cls,
        spark: "SparkSession",
        owner_url: "str | None" = None,
    ) -> str:
        """Plumb a shared compute-identifier URL into the Spark session.

        Captures the driver's URL (or the caller-supplied ``owner_url``)
        and pushes it onto the Spark conf so executor-side code that
        reads ``$YGG_OWNER_URL`` — including
        :func:`compute_identifier_url` itself — returns the same
        value as the driver. That way any locks taken by worker tasks
        (and any checkpoints they happen to record) carry the driver's
        attribution: same job id, same task key, same overall identity.

        Two conf keys are stamped:

        - ``spark.yggdrasil.owner_url`` — readable by driver-side code
          via ``spark.conf.get(...)``. Used by :meth:`commit_checkpoint`
          to default the ``owner`` field.
        - ``spark.executorEnv.YGG_OWNER_URL`` — Spark propagates
          ``executorEnv.*`` bindings into the OS environment of newly
          launched executors, so worker processes pick the URL up
          via :data:`OWNER_URL_ENV` without any extra wiring.

        Returns the URL that was set so callers can stash it for
        use in subsequent ``mapInArrow`` / ``foreachPartition``
        closures (executor processes already running before this call
        won't pick up the executorEnv binding — pass via closure
        instead in that case).
        """
        if owner_url is None:
            owner_url = cls.driver_owner_url()
        try:
            conf = spark.conf
        except AttributeError as exc:
            raise RuntimeError(
                "SparkSession.conf is unavailable — cannot propagate "
                "the owner URL. Pass spark=... to a real SparkSession."
            ) from exc
        try:
            conf.set(_OWNER_URL_SPARK_CONF, owner_url)
        except Exception:
            # Conf set on a stopped / RPC-detached session is a
            # warning, not a fatal — driver-side code can still use
            # the URL via the returned value.
            LOGGER.debug(
                "Failed to set %s on Spark conf; continuing.",
                _OWNER_URL_SPARK_CONF, exc_info=True,
            )
        try:
            conf.set(_OWNER_URL_EXECUTOR_ENV, owner_url)
        except Exception:
            LOGGER.debug(
                "Failed to set %s on Spark conf; executor processes "
                "started before this call will not pick it up via env.",
                _OWNER_URL_EXECUTOR_ENV, exc_info=True,
            )
        return owner_url

    # ==================================================================
    # Batch — Arrow → Spark via mapInArrow
    # ==================================================================

    def read_batch(
        self,
        spark: "SparkSession | None" = None,
        *,
        options: Any = None,
        **kwargs: Any,
    ) -> "SparkDataFrame":
        """Materialise the folder as a (non-streaming) Spark DataFrame.

        The Arrow batches come from
        :meth:`YGGFolderIO.read_arrow_batches`, so every kwarg on
        :class:`CastOptions` (predicate, row_size, byte_size,
        children_predicate, …) flows through unchanged. The batches
        are pushed into Spark via ``mapInArrow`` against a
        single-partition trigger DataFrame.
        """
        spark_session = _resolve_spark_session(spark)
        schema = self._io.collect_schema(options=options, **kwargs)
        if schema is None or len(schema) == 0:
            # Empty folder — return an empty DataFrame with the
            # zero-column schema. The mapInArrow path needs a
            # non-empty schema, so short-circuit here.
            return spark_session.createDataFrame([], schema="EMPTY STRUCT")

        spark_schema = schema.to_spark_schema()

        # Capture only what the worker needs; the IO itself isn't
        # picklable across worker boundaries (file descriptors,
        # Disposable graph). Driver-side execution is the supported
        # mode — same partition contract as the existing
        # :meth:`TabularIO._scan_spark_frame` fallback.
        io = self._io

        def pump(batches: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
            # ``batches`` is the single-row trigger; we ignore it
            # and produce our own batch stream from the IO.
            for _ in batches:
                pass
            with io as opened:
                yield from opened.read_arrow_batches(options=options, **kwargs)

        trigger = spark_session.range(1).repartition(1)
        return trigger.mapInArrow(pump, spark_schema)

    # ==================================================================
    # Write — record a checkpoint with shared driver-owner attribution
    # ==================================================================

    def commit_checkpoint(
        self,
        spark: "SparkSession | None" = None,
        *,
        message: "str | None" = None,
        owner: "str | None" = None,
        propagate: bool = True,
        **extra: Any,
    ) -> dict:
        """Record a YGG checkpoint with a Spark-aware owner attribution.

        Intended to follow a Spark write into the folder root —
        e.g.::

            df.write.mode("append").parquet(io.path.full_path())
            connector.commit_checkpoint(spark, message="batch 42")

        The driver's compute-identifier URL is captured once and
        used as the ``owner`` field of the checkpoint record so the
        single commit point attributes the entire distributed write
        (every worker that contributed parquet files) to one job /
        run / task tuple. When ``propagate`` is ``True`` (default),
        the URL is also stamped onto the Spark conf via
        :meth:`propagate_owner_url` so any new executor processes
        — and any subsequent worker-side calls to
        :func:`compute_identifier_url` — return the same value, in
        turn making any locks they take share the same id.

        ``extra`` flows through to the checkpoint record verbatim,
        same shape as :meth:`YGGFolderIO.checkpoint`.
        """
        spark_session = _resolve_spark_session(spark, create=False) if spark is not None else None
        if spark_session is None and propagate:
            try:
                spark_session = _resolve_spark_session(None, create=False)
            except RuntimeError:
                # No active session — skip propagation, still record
                # the driver-local owner.
                spark_session = None

        if owner is None:
            owner = self.driver_owner_url()

        if propagate and spark_session is not None:
            try:
                self.propagate_owner_url(spark_session, owner)
            except Exception:
                LOGGER.debug(
                    "Owner-URL propagation failed; checkpoint will "
                    "still record %r as the driver owner.", owner,
                    exc_info=True,
                )

        return self._io.checkpoint(message=message, owner=owner, **extra)

    # ==================================================================
    # Stream — Spark's native parquet streaming source
    # ==================================================================

    def read_stream(
        self,
        spark: "SparkSession | None" = None,
        *,
        options: Any = None,
        **kwargs: Any,
    ) -> "SparkDataFrame":
        """Return a streaming :class:`DataFrame` over the folder.

        Backed by ``spark.readStream.format("parquet")`` since the
        folder's data children are always parquet by construction.
        Spark monitors the directory and surfaces newly committed
        children to the streaming query — the same checkpoint /
        rename atomicity that ``YGGFolderIO`` enforces makes this
        race-free.

        Predicate pushdown happens at the Spark layer if the caller
        chains ``.filter(...)`` on the returned DataFrame; the
        in-process predicate filter on the IO doesn't apply in this
        path because the I/O is happening on Spark workers.
        """
        spark_session = _resolve_spark_session(spark)
        schema = self._io.collect_schema(options=options, **kwargs)
        if schema is None or len(schema) == 0:
            raise RuntimeError(
                f"YGGFolderIO at {self._io.path!r} has no committed "
                "schema yet — write at least one batch (or call "
                "write_stats()) before opening a stream."
            )
        spark_schema = schema.to_spark_schema()
        # ``iter_children`` yields TabularIO — including sub-folders
        # for partitioned layouts. ``recursiveFileLookup=true`` makes
        # Spark's parquet streaming source descend the same way so
        # both reader sides see the same set of leaves.
        return (
            spark_session.readStream
            .schema(spark_schema)
            .option("recursiveFileLookup", "true")
            .parquet(str(self._io.path.full_path()))
        )


# ---------------------------------------------------------------------------
# Optional Spark Data Source registration (PySpark 4.0+)
# ---------------------------------------------------------------------------


def register_datasource(
    spark: "SparkSession | None" = None,
    *,
    name: str = "yggfolder",
) -> bool:
    """Register a ``"yggfolder"`` data source on the active session.

    PySpark 4.0+ exposes :mod:`pyspark.sql.datasource`. When that
    module is importable, the call wires up a small subclass of
    :class:`pyspark.sql.datasource.DataSource` that delegates the
    read to :class:`YGGFolderSparkConnector` so callers can write::

        spark.read.format("yggfolder").load("/path/to/folder")

    Returns ``True`` when the registration succeeded, ``False``
    otherwise (no PySpark installed, no datasource API on this
    version, or registration raised). The function-based connector
    still works regardless.
    """
    try:
        from pyspark.sql.datasource import (  # type: ignore[import-not-found]
            DataSource,
            DataSourceReader,
            InputPartition,
        )
    except ImportError:
        LOGGER.debug(
            "pyspark.sql.datasource not available — skipping registration"
        )
        return False

    spark_session = _resolve_spark_session(spark, create=False)
    if spark_session is None:
        return False

    register = getattr(getattr(spark_session, "dataSource", None), "register", None)
    if register is None:
        LOGGER.debug(
            "SparkSession.dataSource.register missing — skipping registration"
        )
        return False

    class YGGFolderDataSource(DataSource):
        """Read-only :class:`DataSource` over a :class:`YGGFolderIO`.

        Options:

        - ``path`` (required) — folder root.
        - ``predicate`` (optional) — predicate string in yggdrasil
          expression DSL; parsed via :meth:`Expression.from_python`.
        """

        @classmethod
        def name(cls) -> str:  # type: ignore[override]
            return name

        def schema(self) -> str:  # type: ignore[override]
            from .ygg_folder_io import YGGFolderIO

            path = self.options.get("path")
            if not path:
                raise ValueError(
                    "yggfolder data source requires a 'path' option."
                )
            with YGGFolderIO(path=path) as io:
                schema = io.collect_schema()
            return schema.to_spark_schema().simpleString()

        def reader(self, schema):  # type: ignore[override]
            return _YGGFolderReader(self.options, schema)

    class _YGGFolderReader(DataSourceReader):
        def __init__(self, options, schema) -> None:
            self._options = options
            self._schema = schema

        def partitions(self):  # type: ignore[override]
            # One partition per *leaf* child. ``iter_children``
            # yields ``TabularIO`` — sub-folders included — so we
            # flatten to leaves only here so the worker reads
            # exactly one buffer per partition. Sub-folder leaves
            # are reachable through their absolute path.
            from .ygg_folder_io import YGGFolderIO

            path = self._options.get("path")
            with YGGFolderIO(path=path) as io:
                leaves = list(_iter_leaf_paths(io))
            return [_ChildPartition(p) for p in leaves]

        def read(self, partition):  # type: ignore[override]
            from yggdrasil.io.fs import Path
            from yggdrasil.io.buffer.base import TabularIO

            # Use the registry-aware factory so the reader picks
            # the right concrete leaf (ParquetIO, IPC, …).
            child = TabularIO.from_path(Path.from_(partition.uri))
            with child as opened:
                yield from opened.read_arrow_batches()

    class _ChildPartition(InputPartition):
        def __init__(self, uri: str) -> None:
            self.uri = uri

    try:
        register(YGGFolderDataSource)
        return True
    except Exception:
        LOGGER.debug("Failed to register yggfolder data source", exc_info=True)
        return False
