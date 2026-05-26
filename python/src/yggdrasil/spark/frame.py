import logging
from typing import Any, Iterator

import pyarrow as pa
from pyspark.sql import SparkSession
from yggdrasil.arrow.cast import any_to_arrow_batch_iterator, any_to_arrow_table
from yggdrasil.data import schema as schema_builder, field as field_builder, Schema
from yggdrasil.data.options import CastOptions
# Use yggdrasil's serde wire format (orjson-backed with broad type
# coverage — datetime / UUID / Path / dataclass / namedtuple / Decimal —
# beyond what cloudpickle alone provides) for the row payloads. The
# cluster needs ygg installed for the unpickle side; that's exactly
# what ``DatabricksClient.spark`` auto-declares via
# ``DatabricksEnv.withDependencies("ygg")``.
from yggdrasil.pickle.ser import dumps, loads

# Function dependency scanning helpers — exposed via the pyspark-free
# ``yggdrasil.spark.dependencies`` module so the scan logic can be
# tested / imported in environments where ``pyspark`` is not installed
# (which is the common path for the Spark Connect client).
from yggdrasil.spark.dependencies import (
    _function_top_modules as _function_top_modules,  # re-export
    _stdlib_modules as _stdlib_modules,  # re-export
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "Dataset",  # noqa: F822  -- lazy module attribute via ``__getattr__``
    "is_dynamic_schema",
]

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


def spark_typed_cast(
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
# Dataset re-export
# ---------------------------------------------------------------------------
#
# Historically :class:`Dataset` lived here as a standalone Spark-DataFrame
# wrapper while :class:`yggdrasil.io.tabular.SparkTabular` carried the
# :class:`Tabular` contract. They've been merged into one class,
# :class:`yggdrasil.spark.tabular.Dataset`, so:
#
# * the Tabular surface (read_arrow_batches / write_arrow_batches /
#   read_spark_frame / write_spark_frame) and
# * the rich Dataset surface (map / apply / filter / explode / cast,
#   executor module shipping, schema inference, the
#   :meth:`__getattr__` DataFrame proxy)
#
# live on one type. Existing call sites that import :class:`Dataset` from
# ``yggdrasil.spark.frame`` or :class:`SparkTabular` from
# ``yggdrasil.io.tabular`` keep working unchanged — both names resolve
# to the same class. The module-level helpers above
# (``is_dynamic_schema``, ``_emit_pickled``, ``_typed_cast``,
# ``_dynamic_rows``, ``_typed_rows``, ``_install_modules_on_executors``,
# ``DYNAMIC_SCHEMA``, ``PICKLE_COLUMN_NAME``) stay here —
# :class:`Dataset` imports them where needed.

def __getattr__(name: str) -> Any:
    """Lazy ``Dataset`` accessor — defers the import to break the cycle.

    Importing :class:`Dataset` at module top-level would form a cycle:
    ``yggdrasil.spark.tabular`` (where :class:`Dataset` lives) imports
    :class:`yggdrasil.io.tabular.Tabular`, whose package ``__init__``
    re-imports :class:`Dataset` — and our module is only half-loaded at
    that point. Resolving the attribute on first access (PEP 562) breaks
    the cycle without forcing every caller to spell out the new import
    path. ``SparkTabular`` lands on the same class for back-compat.
    """
    if name in ("Dataset", "SparkTabular"):
        from yggdrasil.spark.tabular import SparkDataset as _Dataset

        # Cache on the module so subsequent lookups skip the resolver.
        globals()[name] = _Dataset
        return _Dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
