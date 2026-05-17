import logging
import sys
from typing import Any, Callable, Iterable, Iterator

import pyarrow as pa
from pyspark.sql import DataFrame, SparkSession
from yggdrasil.arrow.cast import any_to_arrow_batch_iterator, any_to_arrow_table
from yggdrasil.data import schema as schema_builder, field as field_builder, Schema
from yggdrasil.data.options import CastOptions
from yggdrasil.environ import PyEnv
# Use yggdrasil's serde wire format (orjson-backed with broad type
# coverage — datetime / UUID / Path / dataclass / namedtuple / Decimal —
# beyond what cloudpickle alone provides) for the row payloads. The
# cluster needs ygg installed for the unpickle side; that's exactly
# what ``DatabricksClient.spark`` auto-declares via
# ``DatabricksEnv.withDependencies("ygg")``.
from yggdrasil.pickle import dumps, loads

LOGGER = logging.getLogger(__name__)

# Function dependency scanning helpers — exposed via the pyspark-free
# ``yggdrasil.spark.dependencies`` module so the scan logic can be
# tested / imported in environments where ``pyspark`` is not installed
# (which is the common path for the Spark Connect client).
from yggdrasil.spark.dependencies import (
    _function_top_modules as _function_top_modules,  # re-export
    _stdlib_modules as _stdlib_modules,  # re-export
)

__all__ = [
    "DynamicFrame",
    "is_dynamic_schema",
]

from yggdrasil.data.enums import Mode
from yggdrasil.lazy_imports import polars_module

# Per-session install cache. Spark Connect sessions don't always accept
# attribute writes (``session.X = ...`` may be rejected by the proxy), so we
# can't reliably stash this on the session itself. Keying by ``id(session)``
# leaks at most one entry per session over the process lifetime — acceptable
# for the long-lived client pattern these sessions are designed for.
_PER_SESSION_INSTALLED_MODULES: "dict[int, set[str]]" = {}

PICKLE_COLUMN_NAME = "_pickle"
DYNAMIC_SCHEMA = schema_builder(
    [
        field_builder(
            name=PICKLE_COLUMN_NAME,
            arrow_type=pa.binary(),
            nullable=False,
            metadata={"format": "binary"},
            tags={"namespace": "yggdrasil.spark.frame"},
        )
    ]
)
_ARROW_DYNAMIC_SCHEMA = DYNAMIC_SCHEMA.to_arrow_schema()


def is_dynamic_schema(obj: Any) -> bool:
    schema = Schema.from_any(obj)
    if len(schema) != 1:
        return False
    first = schema.field(index=0)
    return first.name == PICKLE_COLUMN_NAME and pa.types.is_binary(first.arrow_type)


# ---------------------------------------------------------------------------
# Per-partition helpers
# ---------------------------------------------------------------------------

def _emit_pickled(
    objects: Iterator[Any],
    *,
    byte_size: int,
) -> Iterator[pa.RecordBatch]:
    """Pickle a stream of Python objects into dynamic-schema record batches."""
    out: list[dict[str, bytes]] = []
    out_bytes = 0
    for obj in objects:
        ser = dumps(obj)
        if out and out_bytes + len(ser) > byte_size:
            yield pa.RecordBatch.from_pylist(out, schema=_ARROW_DYNAMIC_SCHEMA)
            out = []
            out_bytes = 0
        out.append({PICKLE_COLUMN_NAME: ser})
        out_bytes += len(ser)
    if out:
        yield pa.RecordBatch.from_pylist(out, schema=_ARROW_DYNAMIC_SCHEMA)


def _typed_cast(
    objects_per_batch: Iterator[list[Any]],
    schema: Schema,
    *,
    byte_size: int,
) -> Iterator[pa.RecordBatch]:
    """Cast batched Python objects into Arrow batches matching ``schema``."""
    options = CastOptions(target=schema, safe=False, byte_size=byte_size)

    def _tables() -> Iterator[pa.Table]:
        for group in objects_per_batch:
            yield any_to_arrow_table(group)

    return any_to_arrow_batch_iterator(_tables(), options=options)


def _dynamic_rows(batches: Iterator[pa.RecordBatch]) -> Iterator[Any]:
    """Yield unpickled inner objects from a dynamic-schema batch stream."""
    for batch in batches:
        col = batch.column(0)
        for i in range(batch.num_rows):
            yield loads(col[i].as_py())


def _typed_rows(batches: Iterator[pa.RecordBatch]) -> Iterator[dict[str, Any]]:
    """Yield row-dicts from a typed batch stream."""
    for batch in batches:
        for row in batch.to_pylist():
            yield row


# ---------------------------------------------------------------------------
# Executor module shipping
# ---------------------------------------------------------------------------

# File extensions that mean "compiled native module" — the zip-import path
# Python uses for ``addPyFile`` archives only handles pure-Python source.
# A package shipping these has to land on disk somewhere on ``sys.path``,
# not inside a zip, so we never auto-ship them.
_NATIVE_EXTENSION_SUFFIXES = (".so", ".pyd", ".dylib")


def _module_has_native_extensions(root: "Any") -> bool:
    """``True`` iff *root* is a package directory containing ``.so`` / ``.pyd``.

    A single-file ``.whl`` or ``.zip`` is taken at face value (we trust the
    archive name); the check only rejects raw package directories whose
    on-disk contents include compiled extensions.
    """
    try:
        if not root.is_dir():
            return False
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in _NATIVE_EXTENSION_SUFFIXES:
                return True
    except Exception:
        return False
    return False


def _install_modules_on_executors(
    session: SparkSession,
    modules: "set[str]",
) -> set[str]:
    """Ship each module to Spark executors via ``addArtifacts``.

    Two backends, picked by feature detection:

    - **Spark Connect / Databricks Connect** — ``session.addArtifacts(
      path, pyfile=True)`` injects the archive into the
      session's artifact directory for every executor. When a
      :class:`DatabricksClient` is stashed on the session
      (``session.ygg_client`` — set by
      :meth:`DatabricksClient.spark`), the archive is *also*
      published to the shared workspace registry at
      ``/Workspace/.../.ygg/pypi/simple/`` so teammates on the
      same workspace can reuse it without rebuilding.
    - **Plain Spark** — falls back to
      :meth:`SparkContext.addPyFile`.

    Every module is materialized through
    :func:`build_module_archive` so the artifact is a
    deflated ``.zip`` whose top-level entry IS the package
    directory — ``addArtifacts(pyfile=True)`` then loads it
    straight onto the executor's ``sys.path``. Failures are
    logged at INFO / WARNING and swallowed: best-effort
    installs shouldn't crash an otherwise valid transform.
    """
    from yggdrasil.io.path._module_pack import (
        build_module_archive,
        resolve_module_root,
    )

    client = getattr(session, "ygg_client", None)
    registry = None
    if client is not None:
        try:
            from yggdrasil.databricks.registry import WorkspacePyPIRegistry
            registry = WorkspacePyPIRegistry(client=client)
        except Exception as exc:
            LOGGER.info(
                "Could not build WorkspacePyPIRegistry: %s", exc,
            )
            registry = None

    add_art = getattr(session, "addArtifacts", None) or getattr(
        session, "addArtifact", None,
    )
    # ``session.sparkContext`` raises ``JVM_ATTRIBUTE_NOT_SUPPORTED`` on
    # Spark Connect — never just returns ``None`` — so guard the probe.
    try:
        sc = getattr(session, "sparkContext", None)
    except Exception:
        sc = None
    add_py = getattr(sc, "addPyFile", None) if sc is not None else None

    if not callable(add_art) and not callable(add_py):
        LOGGER.info(
            "SparkSession exposes neither addArtifacts nor "
            "addPyFile — cannot install %s on executors.",
            sorted(modules),
        )
        return set()

    installed: set[str] = set()
    for module_name in sorted(modules):
        try:
            root = resolve_module_root(module_name)
        except Exception as exc:
            LOGGER.info(
                "Skipping %s — cannot resolve module root: %s",
                module_name, exc,
            )
            continue

        # Compiled extensions (``.so`` / ``.pyd``) inside a wheel-installed
        # package can't be loaded from a zip — Python's zipimporter handles
        # pure-Python source but not native ``.so`` modules. Trying to ship
        # pyarrow / numpy / polars / pandas via ``addPyFile`` puts a broken
        # copy ahead of the executor's real install in ``sys.path``. These
        # are the packages the cluster always already has; skip them.
        if _module_has_native_extensions(root):
            LOGGER.info(
                "Skipping %s — package contains compiled extensions that "
                "won't load from a zip; assuming cluster has it pre-installed.",
                module_name,
            )
            continue

        archive_path = None
        if registry is not None:
            # Workspace-cache path: publish through the registry
            # so the wheel / zip ends up at
            # ``.ygg/pypi/simple/<pkg>/...`` for sharing. The
            # registry returns a ``local:<path>`` spec and the
            # remote :class:`WorkspacePath`; we use the local
            # path for ``addArtifacts``.
            try:
                spec, _remote = registry.publish(module_name)
                if spec.startswith("local:"):
                    archive_path = spec[len("local:"):]
            except Exception as exc:
                LOGGER.info(
                    "Registry publish failed for %s; falling back "
                    "to in-place archive: %s", module_name, exc,
                )

        if archive_path is None:
            try:
                archive_path = str(
                    build_module_archive(root, dest=None),
                )
            except Exception as exc:
                LOGGER.info(
                    "Skipping %s — cannot build archive: %s",
                    module_name, exc,
                )
                continue

        try:
            if callable(add_art):
                add_art(archive_path, pyfile=True)
            else:
                add_py(archive_path)
        except Exception as exc:
            LOGGER.warning(
                "Failed to ship %s via Spark: %s", module_name, exc,
            )
            continue
        installed.add(module_name)
    return installed


# ---------------------------------------------------------------------------
# DynamicFrame
# ---------------------------------------------------------------------------

class DynamicFrame:
    """Spark DataFrame wrapper with an optional yggdrasil Schema.

    Two modes:

    * **Dynamic** (``schema is None``) — the underlying Spark frame has the
      single-column ``_pickle`` schema; rows are arbitrary pickled Python
      objects. Transforms (``map``/``filter``/``apply``/``explode``) operate
      on the unpickled inner objects.
    * **Typed** (``schema`` set) — the underlying Spark frame matches
      ``schema.to_spark_schema()``. Transforms receive ``dict`` rows and
      outputs are cast back through ``Schema.cast_arrow``-driven
      ``mapInArrow`` pipelines.

    Any attribute not defined here is proxied to the underlying ``DataFrame``
    via ``__getattr__``; ``DataFrame`` results are re-wrapped as
    ``DynamicFrame`` carrying the lifted Arrow schema.
    """

    __slots__ = ('df', 'schema', 'installed_modules')

    def __init__(
        self,
        df: DataFrame,
        schema: Schema | None = None,
        *,
        installed_modules: "set[str] | None" = None,
    ):
        self.df = df
        self.schema = schema
        # Top-level package names this frame has already declared on
        # the cluster. Auto-populated when :meth:`apply` / :meth:`map`
        # / :meth:`filter` scan a user function's globals and feed
        # them through :meth:`_ensure_installed`. Persists across
        # transforms so a chain like
        # ``df.apply(f).map(g).filter(h)`` only round-trips each
        # module once per frame lineage. Tests and call sites that
        # know exactly which modules to ship can seed this directly
        # (``DynamicFrame(df, installed_modules={"ygg", "polars"})``)
        # to bypass the scan.
        self.installed_modules: set[str] = (
            set(installed_modules) if installed_modules else set()
        )

    # ---- core --------------------------------------------------------------

    @property
    def is_dynamic(self) -> bool:
        return self.schema is None

    @property
    def spark_schema(self):
        return self.df.schema

    @property
    def sparkSession(self) -> SparkSession:
        return self.df.sparkSession

    def __iter__(self) -> Iterator[Any]:
        return self.to_local_iterator()

    def count(self) -> int:
        return self.df.count()

    # ---- proxy -------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup fails (slots + dataclass means our
        # own attributes resolve before we get here).
        attr = getattr(self.df, name)
        if callable(attr):
            return _ProxiedCallable(attr)
        return _wrap(attr)

    # ---- constructors ------------------------------------------------------

    @classmethod
    def from_spark_frame(
        cls,
        df: DataFrame,
        schema: Schema | None = None,
    ):
        if schema is None:
            schema = Schema.from_(df)
        else:
            schema = Schema.from_any(schema)
            df = schema.cast_spark_tabular(df)

        return cls(df=df, schema=schema)

    @classmethod
    def from_iterable(
        cls,
        items: Iterable[Any],
        schema: Schema | None = None,
        *,
        spark_session: SparkSession | None = None,
        byte_size: int = 128 * 1024 * 1024,
    ) -> "DynamicFrame":
        """Build a frame from an in-memory iterable.

        ``schema=None`` pickles each element into a dynamic frame.
        ``schema=<Schema>`` casts the iterable on the driver and returns a
        typed ``DynamicFrame`` whose underlying ``DataFrame`` matches
        ``schema``.
        """
        if spark_session is None:
            spark_session = PyEnv.spark_session(
                create=True, install_spark=False, import_error=True,
            )

        if schema is None:
            # Materialize before handing to Spark — Spark Connect's
            # ``createDataFrame`` indexes ``_data[0]`` to sniff the shape,
            # which IndexErrors on an empty generator. A list also lets the
            # Arrow path on Spark Connect take the fast LocalRelation route
            # when ``len(data) == 0``. Local mode tolerates either shape.
            cls._ensure_installed_on_session(spark_session)
            rows = [(dumps(x),) for x in items]
            df = spark_session.createDataFrame(
                rows,
                schema=DYNAMIC_SCHEMA.to_spark_schema(),
            )
            return cls(df=df, schema=None)

        cls._ensure_installed_on_session(spark_session)
        schema = Schema.from_any(schema)
        table = any_to_arrow_table(
            items,
            options=CastOptions(target=schema, safe=False, byte_size=byte_size),
        )
        return cls(df=spark_session.createDataFrame(table), schema=schema)

    @classmethod
    def parallelize(
        cls,
        function: Callable[[Any], Any],
        inputs: Iterable[Any],
        schema: Schema | None = None,
        *,
        spark_session: SparkSession | None = None,
        byte_size: int = 128 * 1024 * 1024,
    ) -> "DynamicFrame":
        """Distribute ``function`` over ``inputs`` via ``mapInArrow``.

        ``schema=None`` returns a dynamic frame of pickled outputs.
        ``schema=<Schema>`` casts outputs and returns a typed frame.
        """
        if spark_session is None:
            spark_session = PyEnv.spark_session(
                create=True, install_spark=False, import_error=True,
            )

        installed_modules = cls._ensure_installed_on_session(
            spark_session, function,
        )
        dumped = [(dumps(x),) for x in inputs]
        function_pickle = dumps(function)
        input_df = spark_session.createDataFrame(
            dumped,
            schema=DYNAMIC_SCHEMA.to_spark_schema(),
            verifySchema=False
        )

        if schema is None:
            def _runner(batches: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
                func = loads(function_pickle)
                yield from _emit_pickled(
                    (func(obj) for obj in _dynamic_rows(batches)),
                    byte_size=byte_size,
                )

            result_df = input_df.mapInArrow(
                _runner, schema=DYNAMIC_SCHEMA.to_spark_schema()
            )
            return cls(
                df=result_df, schema=None,
                installed_modules=installed_modules,
            )

        schema = Schema.from_any(schema)

        def _typed_runner(batches: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
            func = loads(function_pickle)

            def _groups() -> Iterator[list[Any]]:
                for batch in batches:
                    col = batch.column(0)
                    n = batch.num_rows
                    if n == 0:
                        continue
                    yield [func(loads(col[i].as_py())) for i in range(n)]

            return _typed_cast(_groups(), schema, byte_size=byte_size)

        result_df = input_df.mapInArrow(_typed_runner, schema=schema.to_spark_schema())
        return cls(
            df=result_df, schema=schema,
            installed_modules=installed_modules,
        )

    def infer_schema(
        self,
        *,
        limit: int | None = None,
        force: bool = False,
        inplace: bool = True,
    ) -> Schema:
        """Infer a yggdrasil :class:`Schema` from the row contents.

        Dynamic mode: each row is unpickled and shape-inferred via
        :meth:`Schema.from_`; per-partition schemas are merged in
        ``APPEND`` mode (union of fields, widening of nullability), then
        folded on the driver into the final schema.

        Typed mode: returns :attr:`schema` unchanged unless ``force=True``,
        in which case the underlying batches are re-inferred from row dicts
        — useful after a heterogeneous transform whose output schema is
        looser than the declared one (e.g. struct rows with optional keys).

        Parameters
        ----------
        limit:
            If given, only the first ``limit`` rows are scanned (driver-side
            ``df.limit(limit).toLocalIterator()``). Cheaper but may miss
            fields that only appear later. ``None`` (default) does a full
            distributed scan.
        force:
            Re-infer even when ``self.schema`` is already set.
        inplace:
            If ``True`` (default), update ``self.schema`` in-place.
        """
        if not self.is_dynamic and not force:
            return self.schema

        # ---- sample path: drive the inference locally ---------------------
        if limit is not None:
            df = self.df.limit(limit)
            merged: Schema | None = None
            if self.is_dynamic:
                for row in df.toLocalIterator():
                    shape = Schema.from_(loads(row[PICKLE_COLUMN_NAME]))
                    merged = shape if merged is None else merged.merge_with(
                        shape, mode=Mode.APPEND,
                    )
            else:
                for row in df.toLocalIterator():
                    shape = Schema.from_(row.asDict(recursive=True))
                    merged = shape if merged is None else merged.merge_with(
                        shape, mode=Mode.APPEND,
                    )
            if merged is None:
                raise ValueError("Cannot infer schema from an empty frame.")
            return merged

        # ---- full-scan path: per-partition inference via mapInArrow -------
        # The UDF imports ``yggdrasil.pickle`` on executors — make sure it's
        # there. No user function to scan; the seed-with-yggdrasil default
        # in :meth:`_ensure_installed_on_session` is exactly enough.
        self._ensure_installed()
        is_dynamic_in = self.is_dynamic

        def _runner(batches: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
            partition_schema: Schema | None = None

            if is_dynamic_in:
                for batch in batches:
                    col = batch.column(0)
                    for i in range(batch.num_rows):
                        shape = Schema.from_(loads(col[i].as_py()))
                        partition_schema = (
                            shape
                            if partition_schema is None
                            else partition_schema.merge_with(shape, mode=Mode.APPEND)
                        )
            else:
                for batch in batches:
                    for row in batch.to_pylist():
                        shape = Schema.from_(row)
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

        schemas_df = self.df.mapInArrow(
            _runner, schema=DYNAMIC_SCHEMA.to_spark_schema()
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
            self.schema = merged

        return merged

    # ---- executor dependency wiring ----------------------------------------

    @classmethod
    def _ensure_installed_on_session(
        cls,
        session: SparkSession,
        *functions: Callable[..., Any],
    ) -> "set[str]":
        """Auto-ship ygg (+ any function deps) to executors on first use.

        UDFs registered via :meth:`mapInArrow` unpickle ``yggdrasil.pickle``
        bytes inside the executor — which needs ``yggdrasil`` itself on the
        worker. When the cluster has an *older* ygg pre-installed (a common
        Databricks Connect setup), the local driver's pickle format may
        disagree with what the executor knows how to read, surfacing as
        ``UnpicklingError: invalid load key, 'Y'`` (stdlib pickle on YGG
        magic bytes) or ``SerializationError: Invalid magic header``. Shipping
        the driver's exact ``yggdrasil`` source via ``addArtifacts`` /
        ``addPyFile`` overrides the stale install.

        Returns the set of modules now known-installed on *session* — useful
        as the ``installed_modules`` seed for fresh :class:`DynamicFrame`
        instances so the next transform skips the re-scan.
        """
        cache = _PER_SESSION_INSTALLED_MODULES.setdefault(id(session), set())
        # Always seed with yggdrasil itself — every UDF path imports
        # ``yggdrasil.pickle`` on the executor, regardless of the user
        # function's globals.
        wanted: set[str] = {"yggdrasil"}
        for fn in functions:
            if fn is None:
                continue
            wanted.update(_function_top_modules(fn))

        new_modules = wanted - cache
        if not new_modules:
            return set(cache)

        try:
            installed = _install_modules_on_executors(session, new_modules)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning(
                "DynamicFrame: failed to install %s on executors: %s",
                sorted(new_modules), exc,
            )
            return set(cache)

        cache.update(installed)
        return set(cache)

    def _ensure_installed(self, *functions: Callable[..., Any]) -> set[str]:
        """Per-frame wrapper around :meth:`_ensure_installed_on_session`.

        Scans *functions* for module deps, ships any new ones to executors
        (alongside ``yggdrasil`` itself if the session hasn't been seeded
        yet), and folds the result into :attr:`installed_modules` so a chain
        of transforms only round-trips each module once.

        Resolution mirrors the docstring on
        :meth:`_install_modules_on_executors`: Spark Connect /
        Databricks Connect goes through ``addArtifacts(pyfile=True)``
        (and optionally :class:`WorkspacePyPIRegistry` when the session is
        bound via :meth:`DatabricksClient.spark`); plain Spark falls back to
        :meth:`SparkContext.addPyFile`. Failures are logged at WARNING and
        swallowed — a missing executor install surfaces as a UDF-time
        ``ModuleNotFoundError`` we don't want a best-effort scan to mask.
        """
        installed = self._ensure_installed_on_session(
            self.sparkSession, *functions,
        )
        new = installed - self.installed_modules
        self.installed_modules.update(installed)
        return new

    def _install_modules_on_executors(
        self, modules: "set[str]",
    ) -> set[str]:
        """Per-frame shim around the module-level
        :func:`_install_modules_on_executors`."""
        return _install_modules_on_executors(self.sparkSession, modules)

    # ---- transforms --------------------------------------------------------

    def map(
        self,
        function: Callable[[Any], Any],
        schema: Schema | None = None,
        *,
        byte_size: int = 128 * 1024 * 1024,
    ) -> "DynamicFrame":
        """1:1 map over rows.

        Input rows are unpickled objects (dynamic mode) or row-dicts
        (typed mode). Output schema follows ``schema`` if given, else
        the result is a dynamic frame.
        """
        self._ensure_installed(function)
        function_pickle = dumps(function)
        is_dynamic_in = self.is_dynamic

        if schema is None:
            def _runner(batches: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
                func = loads(function_pickle)
                rows = _dynamic_rows(batches) if is_dynamic_in else _typed_rows(batches)
                yield from _emit_pickled(
                    (func(row) for row in rows), byte_size=byte_size,
                )

            result_df = self.df.mapInArrow(
                _runner, schema=DYNAMIC_SCHEMA.to_spark_schema()
            )
            return type(self)(
                df=result_df, schema=None,
                installed_modules=self.installed_modules,
            )

        schema = Schema.from_any(schema)

        def _typed_runner(batches: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
            func = loads(function_pickle)

            def _groups() -> Iterator[list[Any]]:
                if is_dynamic_in:
                    for batch in batches:
                        col = batch.column(0)
                        n = batch.num_rows
                        if n == 0:
                            continue
                        yield [func(loads(col[i].as_py())) for i in range(n)]
                else:
                    for batch in batches:
                        rows = batch.to_pylist()
                        if not rows:
                            continue
                        yield [func(r) for r in rows]

            return _typed_cast(_groups(), schema, byte_size=byte_size)

        result_df = self.df.mapInArrow(_typed_runner, schema=schema.to_spark_schema())
        return type(self)(
            df=result_df, schema=schema,
            installed_modules=self.installed_modules,
        )

    def apply(
        self,
        function: Callable[[Any], Any],
        schema: Schema | None = None,
        *,
        byte_size: int = 128 * 1024 * 1024,
    ) -> "DynamicFrame":
        """Map ``function`` over each row, optionally casting against ``schema``.

        Without a schema this is :meth:`map`. With a schema, ``function``
        may return any tabular shape (dict, dataclass, list-of-rows,
        polars/pandas/arrow frame, ``pa.RecordBatch``); outputs are
        streamed through :func:`any_to_arrow_batch_iterator` in one pass.
        """
        if schema is None:
            return self.map(function, byte_size=byte_size)

        self._ensure_installed(function)
        schema = Schema.from_any(schema)
        function_pickle = dumps(function)
        is_dynamic_in = self.is_dynamic

        def _runner(batches: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
            func = loads(function_pickle)

            def _groups() -> Iterator[list[Any]]:
                if is_dynamic_in:
                    for batch in batches:
                        col = batch.column(0)
                        n = batch.num_rows
                        if n == 0:
                            continue
                        yield [func(loads(col[i].as_py())) for i in range(n)]
                else:
                    for batch in batches:
                        rows = batch.to_pylist()
                        if not rows:
                            continue
                        yield [func(r) for r in rows]

            return _typed_cast(_groups(), schema, byte_size=byte_size)

        result_df = self.df.mapInArrow(_runner, schema=schema.to_spark_schema())
        return type(self)(
            df=result_df, schema=schema,
            installed_modules=self.installed_modules,
        )

    def filter(
        self,
        predicate: Callable[[Any], bool],
        schema: Schema | None = None,
        *,
        byte_size: int = 128 * 1024 * 1024,
    ) -> "DynamicFrame":
        """Drop rows where ``predicate(row)`` is false.

        Predicate sees unpickled objects (dynamic mode) or row-dicts
        (typed mode). When called on a typed frame without a ``schema``
        argument, the existing schema is preserved (no re-cast needed).
        """
        self._ensure_installed(predicate)
        predicate_pickle = dumps(predicate)
        is_dynamic_in = self.is_dynamic

        if is_dynamic_in and schema is None:
            def _runner(batches: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
                pred = loads(predicate_pickle)
                out: list[dict[str, bytes]] = []
                out_bytes = 0
                for batch in batches:
                    col = batch.column(0)
                    for i in range(batch.num_rows):
                        ser = col[i].as_py()
                        if not pred(loads(ser)):
                            continue
                        if out and out_bytes + len(ser) > byte_size:
                            yield pa.RecordBatch.from_pylist(
                                out, schema=_ARROW_DYNAMIC_SCHEMA
                            )
                            out = []
                            out_bytes = 0
                        out.append({PICKLE_COLUMN_NAME: ser})
                        out_bytes += len(ser)
                if out:
                    yield pa.RecordBatch.from_pylist(out, schema=_ARROW_DYNAMIC_SCHEMA)

            result_df = self.df.mapInArrow(
                _runner, schema=DYNAMIC_SCHEMA.to_spark_schema()
            )
            return type(self)(
                df=result_df, schema=None,
                installed_modules=self.installed_modules,
            )

        # Typed input, or dynamic input + explicit output schema.
        out_schema = Schema.from_any(schema) if schema is not None else self.schema
        if out_schema is None:
            # Dynamic input, no schema given — falls through to the dynamic path
            # above; this branch is unreachable but keeps mypy quiet.
            raise AssertionError("unreachable")

        def _typed_runner(batches: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
            pred = loads(predicate_pickle)

            def _groups() -> Iterator[list[Any]]:
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

        result_df = self.df.mapInArrow(
            _typed_runner, schema=out_schema.to_spark_schema()
        )
        return type(self)(
            df=result_df, schema=out_schema,
            installed_modules=self.installed_modules,
        )

    def explode(
        self,
        schema: Schema | None = None,
        *,
        byte_size: int = 128 * 1024 * 1024,
    ) -> "DynamicFrame":
        """Explode rows of iterables into one row per element.

        Only meaningful in dynamic mode — typed rows are dicts, not
        iterables. Pass a ``schema`` to type the flattened output.
        """
        if not self.is_dynamic:
            raise TypeError(
                "explode() is only defined on dynamic-mode frames; "
                "the inner objects must be iterable."
            )

        # No user function to scan, but the UDF still imports
        # ``yggdrasil.pickle`` on the executor — seed the install.
        self._ensure_installed()

        if schema is None:
            def _runner(batches: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
                def _items() -> Iterator[Any]:
                    for batch in batches:
                        col = batch.column(0)
                        for i in range(batch.num_rows):
                            yield from loads(col[i].as_py())

                yield from _emit_pickled(_items(), byte_size=byte_size)

            result_df = self.df.mapInArrow(
                _runner, schema=DYNAMIC_SCHEMA.to_spark_schema()
            )
            return type(self)(
                df=result_df, schema=None,
                installed_modules=self.installed_modules,
            )

        schema = Schema.from_any(schema)

        def _typed_runner(batches: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
            def _groups() -> Iterator[list[Any]]:
                for batch in batches:
                    col = batch.column(0)
                    group: list[Any] = []
                    for i in range(batch.num_rows):
                        group.extend(loads(col[i].as_py()))
                    if group:
                        yield group

            return _typed_cast(_groups(), schema, byte_size=byte_size)

        result_df = self.df.mapInArrow(_typed_runner, schema=schema.to_spark_schema())
        return type(self)(
            df=result_df, schema=schema,
            installed_modules=self.installed_modules,
        )

    def cast(
        self,
        schema: Schema,
        *,
        byte_size: int = 128 * 1024 * 1024,
    ) -> "DynamicFrame":
        """Materialise rows against ``schema`` as a typed ``DynamicFrame``."""
        self._ensure_installed()
        schema = Schema.from_any(schema)
        is_dynamic_in = self.is_dynamic

        def _runner(batches: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
            def _groups() -> Iterator[list[Any]]:
                if is_dynamic_in:
                    for batch in batches:
                        col = batch.column(0)
                        n = batch.num_rows
                        if n == 0:
                            continue
                        yield [loads(col[i].as_py()) for i in range(n)]
                else:
                    # Already typed — recast batches through Schema.cast_arrow.
                    for batch in batches:
                        rows = batch.to_pylist()
                        if rows:
                            yield rows

            return _typed_cast(_groups(), schema, byte_size=byte_size)

        result_df = self.df.mapInArrow(_runner, schema=schema.to_spark_schema())
        return type(self)(
            df=result_df, schema=schema,
            installed_modules=self.installed_modules,
        )

    def to_dynamic(self, *, byte_size: int = 128 * 1024 * 1024) -> "DynamicFrame":
        """Drop typing: re-pickle row-dicts back into a dynamic frame.

        No-op when already dynamic. Useful before applying transforms
        whose output shape isn't a stable schema.
        """
        if self.is_dynamic:
            return self

        self._ensure_installed()

        def _runner(batches: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
            yield from _emit_pickled(_typed_rows(batches), byte_size=byte_size)

        result_df = self.df.mapInArrow(
            _runner, schema=DYNAMIC_SCHEMA.to_spark_schema()
        )
        return type(self)(
            df=result_df, schema=None,
            installed_modules=self.installed_modules,
        )

    # ---- terminal ops ------------------------------------------------------

    def collect(self) -> list[Any]:
        if self.is_dynamic:
            return [loads(row[PICKLE_COLUMN_NAME]) for row in self.df.collect()]
        return [row.asDict(recursive=True) for row in self.df.collect()]

    def to_local_iterator(self) -> Iterator[Any]:
        if self.is_dynamic:
            for row in self.df.toLocalIterator():
                yield loads(row[PICKLE_COLUMN_NAME])
        else:
            for row in self.df.toLocalIterator():
                yield row.asDict(recursive=True)

    def toArrow(
        self,
        schema: Schema | None = None,
        *,
        byte_size: int = 128 * 1024 * 1024,
    ) -> pa.Table:
        target = schema if schema is not None else self.schema
        if self.is_dynamic and target is None:
            return any_to_arrow_table(
                self.to_local_iterator(),
                options=CastOptions(byte_size=byte_size, safe=False),
            )
        from yggdrasil.spark.cast import spark_dataframe_to_arrow
        if target is not None and target is not self.schema:
            return spark_dataframe_to_arrow(
                self.cast(target, byte_size=byte_size).df,
            )
        return spark_dataframe_to_arrow(self.df)

    def toPandas(
        self,
        schema: Schema | None = None,
        *,
        byte_size: int = 128 * 1024 * 1024,
    ):
        return self.toArrow(schema=schema, byte_size=byte_size).to_pandas()

    def toPolars(
        self,
        schema: Schema | None = None,
        *,
        byte_size: int = 128 * 1024 * 1024,
    ):
        pl = polars_module()
        return pl.from_arrow(self.toArrow(schema=schema, byte_size=byte_size))


# ---------------------------------------------------------------------------
# Proxy plumbing
# ---------------------------------------------------------------------------

def _wrap(value: Any) -> Any:
    """Wrap ``DataFrame`` results as ``DynamicFrame``; pass others through."""
    if isinstance(value, DataFrame):
        return DynamicFrame(df=value, schema=Schema.from_any(value.schema))
    return value


class _ProxiedCallable:
    """Bound-method shim that wraps DataFrame return values.

    Returned by :meth:`DynamicFrame.__getattr__` for callable attributes
    on the underlying ``DataFrame``. Calling it forwards args/kwargs and
    runs the result through :func:`_wrap` so chained DataFrame ops stay
    inside ``DynamicFrame``. Nested attribute access (e.g.
    ``df.groupBy("x").agg(...)``) works because intermediate non-DF
    objects (``GroupedData``, ``Column``) pass through unchanged and
    their methods aren't proxied — only the final ``DataFrame`` they
    return gets wrapped, which happens via this same shim if the
    intermediate is itself accessed through ``__getattr__``.
    """

    __slots__ = ("_callable",)

    def __init__(self, fn: Callable[..., Any]) -> None:
        self._callable = fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _wrap(self._callable(*args, **kwargs))

    def __repr__(self) -> str:
        return f"_ProxiedCallable({self._callable!r})"