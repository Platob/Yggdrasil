"""Single guard module for optional dependencies.

Every other yggdrasil module reaches optional packages through here
instead of through a per-package ``lib.py`` shim. Two access shapes:

* ``from yggdrasil.lazy_imports import polars`` — direct attribute
  access via :func:`__getattr__`, lazily resolving on first touch.
  The result is the live module, identical to ``import polars``.
* ``from yggdrasil.lazy_imports import polars_module`` and call —
  identical effect, kept for callers that want the import to be a
  function call (clearer in long-import-graph code).

Probe helpers (``has_X``) are non-raising — they return ``False``
when the package isn't installed.
"""
from functools import lru_cache
from typing import Any


# ===========================================================================
# Internal package classes / submodules — always present, just lazy
# ===========================================================================


@lru_cache(maxsize=1)
def bytes_io_class():
    from yggdrasil.io.bytes_io import BytesIO
    return BytesIO


@lru_cache(maxsize=1)
def io_class():
    from yggdrasil.io.base import IO
    return IO


@lru_cache(maxsize=1)
def media_type_class():
    from yggdrasil.enums import MediaType
    return MediaType


@lru_cache(maxsize=1)
def media_types_class():
    from yggdrasil.enums import MediaTypes
    return MediaTypes


@lru_cache(maxsize=1)
def mime_type_class():
    from yggdrasil.enums import MimeType
    return MimeType


@lru_cache(maxsize=1)
def mime_types_class():
    from yggdrasil.enums import MimeTypes
    return MimeTypes


@lru_cache(maxsize=1)
def path_class():
    from yggdrasil.io.path.path import Path
    return Path


@lru_cache(maxsize=1)
def local_path_class():
    from yggdrasil.io.path.local_path import LocalPath
    return LocalPath


@lru_cache(maxsize=1)
def databricks_path_class():
    from yggdrasil.databricks.fs import DatabricksPath
    return DatabricksPath


@lru_cache(maxsize=1)
def databricks_client_class():
    from yggdrasil.databricks import DatabricksClient
    return DatabricksClient


@lru_cache(maxsize=1)
def field_class():
    from yggdrasil.data.data_field import Field
    return Field


@lru_cache(maxsize=1)
def struct_type_class():
    from yggdrasil.data.types.nested import StructType
    return StructType


@lru_cache(maxsize=1)
def schema_class():
    from yggdrasil.data.schema import Schema
    return Schema


@lru_cache(maxsize=1)
def pyarrow_dataset_module():
    import pyarrow.dataset as ds
    return ds


@lru_cache(maxsize=1)
def pyarrow_compute_module():
    import pyarrow.compute as pc
    return pc


@lru_cache(maxsize=1)
def tabular_io_class():
    from yggdrasil.io.tabular import Tabular
    return Tabular


@lru_cache(maxsize=1)
def aws_client_class():
    from yggdrasil.aws.client import AWSClient
    return AWSClient


@lru_cache(maxsize=1)
def aws_s3_path_class():
    from yggdrasil.aws.fs.path import S3Path
    return S3Path


# ===========================================================================
# Optional third-party packages — try-import, opt-in install on miss
# ===========================================================================


#: Process-wide module cache shared by every loader below. Replaces the
#: per-function :func:`functools.lru_cache` previous design used — that
#: keyed the cache by call arguments, so ``polars_module()`` and
#: ``polars_module(install=True)`` ended up as separate entries even
#: though both resolve the same module. A plain ``dict[name -> module]``
#: collapses both to one slot and lets the ``install`` kwarg control
#: only the on-miss path.
_LAZY_CACHE: "dict[str, Any]" = {}


def _lazy_import(
    module_name: str,
    pip_name: str | None = None,
    *,
    install: bool = False,
    hint: str | None = None,
) -> Any:
    """Cached import with optional pip-install fallback.

    Look *module_name* up in :data:`_LAZY_CACHE`; on miss, run
    ``importlib.import_module``. When the import raises and *install*
    is ``True``, hand off to :meth:`PyEnv.runtime_import_module` —
    which anchors on the active interpreter (``sys.executable``) and
    routes the install through ``uv pip --python <python>`` or
    ``<python> -m pip``, so the package lands in the same
    site-packages the running interpreter already reads from. The
    historical default was ``install=True`` (every loader auto-pulled
    its package on first touch); that was too eager — debugging an
    accidental network install three frames deep into a cast was the
    main pain point. Default flipped to ``install=False``: callers
    that genuinely want a side-effect install opt in explicitly.

    *hint* is substituted into the :class:`ImportError` message when
    the import fails and we are not installing — used by packages
    whose install string is non-obvious (``pymongoarrow``,
    ``adbc-driver-postgresql``, …) so the user sees the right
    ``pip install`` command from the traceback.
    """
    cached = _LAZY_CACHE.get(module_name)
    if cached is not None:
        return cached
    try:
        import importlib
        mod = importlib.import_module(module_name)
    except ImportError as exc:
        if not install:
            if hint is not None:
                raise ImportError(hint) from exc
            raise
        from yggdrasil.environ import PyEnv
        mod = PyEnv.runtime_import_module(
            module_name=module_name,
            pip_name=pip_name or module_name,
            install=True,
        )
    _LAZY_CACHE[module_name] = mod
    return mod


def _import_or_install(
    module_name: str,
    pip_name: str | None = None,
    *,
    install: bool = False,
) -> Any:
    """Back-compat alias for :func:`_lazy_import` — *install* defaults to ``False``.

    Older internal callers spelled this name; kept as a thin alias so a
    grep-driven cleanup can defer renaming.
    """
    return _lazy_import(module_name, pip_name, install=install)


def polars_module(*, install: bool = False):
    return _lazy_import("polars", install=install)


def pandas_module(*, install: bool = False):
    return _lazy_import("pandas", install=install)


def pyarrow_module(*, install: bool = False):
    return _lazy_import("pyarrow", install=install)


def fastapi_module(*, install: bool = False):
    return _lazy_import("fastapi", install=install)


def requests_module(*, install: bool = False):
    return _lazy_import("requests", install=install)


def confluent_kafka_module(*, install: bool = False):
    return _lazy_import("confluent_kafka", "confluent-kafka", install=install)


@lru_cache(maxsize=1)
def spark_sql_module():
    import pyspark.sql as sql
    return sql


@lru_cache(maxsize=1)
def spark_dataframe_classes() -> tuple[type, ...]:
    """Tuple of Spark DataFrame classes for ``isinstance`` checks.

    On PySpark 3.5 and earlier, ``pyspark.sql.connect.dataframe.DataFrame``
    (the class Databricks Connect / Spark Connect sessions return from
    ``spark.sql(...)``) is **not** a subclass of
    ``pyspark.sql.DataFrame`` — they are parallel implementations with
    duck-type-compatible APIs.  A bare ``isinstance(obj,
    pyspark.sql.DataFrame)`` check rejects a perfectly valid Connect
    DataFrame.  PySpark 4.0 collapsed both onto a common ancestor, so
    the tuple is redundant there but still correct.

    Use:

        from yggdrasil.lazy_imports import spark_dataframe_classes
        if isinstance(obj, spark_dataframe_classes()):
            ...
    """
    classes: list[type] = []
    try:
        from pyspark.sql import DataFrame as _ClassicDataFrame
        classes.append(_ClassicDataFrame)
    except ImportError:
        pass
    try:
        from pyspark.sql.connect.dataframe import DataFrame as _ConnectDataFrame
        if _ConnectDataFrame not in classes:
            classes.append(_ConnectDataFrame)
    except ImportError:
        pass
    return tuple(classes)


@lru_cache(maxsize=1)
def spark_column_classes() -> tuple[type, ...]:
    """Tuple of Spark Column classes for ``isinstance`` checks.

    Same Connect / classic split as :func:`spark_dataframe_classes` —
    ``pyspark.sql.connect.column.Column`` is not a subclass of
    ``pyspark.sql.Column`` on PySpark 3.5 and earlier.
    """
    classes: list[type] = []
    try:
        from pyspark.sql import Column as _ClassicColumn
        classes.append(_ClassicColumn)
    except ImportError:
        pass
    try:
        from pyspark.sql.connect.column import Column as _ConnectColumn
        if _ConnectColumn not in classes:
            classes.append(_ConnectColumn)
    except ImportError:
        pass
    return tuple(classes)


def boto3_module(*, install: bool = False):
    return _lazy_import("boto3", install=install)


_BOTOCORE_HINT = (
    "yggdrasil.aws requires 'botocore' (a transitive dep of boto3). "
    "Install boto3 with `pip install boto3` to get it."
)


def botocore_module(*, install: bool = False):
    """Lazy-import botocore.

    We want the top-level ``botocore`` module so callers can reach
    ``botocore.exceptions``, ``botocore.config``, ``botocore.credentials``,
    and ``botocore.session`` without triggering separate imports.
    Touches the four submodules we use after the parent loads so each
    is registered in ``sys.modules`` for cheap attribute access on
    later calls.
    """
    # boto3 is the carrier — botocore ships alongside it. When the
    # caller asks to install, we route through the ``boto3`` pip name.
    botocore = _lazy_import("botocore", "boto3", install=install, hint=_BOTOCORE_HINT)
    for sub in ("botocore.exceptions", "botocore.config", "botocore.credentials", "botocore.session"):
        _lazy_import(sub, "boto3", install=False, hint=_BOTOCORE_HINT)
    return botocore


# ---------------------------------------------------------------------------
# Databricks SDK
# ---------------------------------------------------------------------------


def databricks_sdk_module(*, install: bool = False):
    return _lazy_import("databricks.sdk", "databricks-sdk", install=install)


@lru_cache(maxsize=1)
def databricks_account_client_class():
    return databricks_sdk_module().AccountClient


@lru_cache(maxsize=1)
def databricks_workspace_client_class():
    return databricks_sdk_module().WorkspaceClient


@lru_cache(maxsize=1)
def databricks_config_class():
    from databricks.sdk.config import Config
    return Config


@lru_cache(maxsize=1)
def databricks_error_class():
    from databricks.sdk.errors import DatabricksError
    return DatabricksError


# ---------------------------------------------------------------------------
# Postgres — psycopg + ADBC; both optional, ADBC is the Arrow fast path
# ---------------------------------------------------------------------------


_PSYCOPG_HINT = (
    "psycopg (psycopg 3) is required for yggdrasil.postgres; "
    "install it with `pip install ygg[postgres]` or "
    "`pip install psycopg[binary]`."
)
_ADBC_HINT = (
    "adbc_driver_postgresql is required for the Arrow-native "
    "Postgres path; install it with `pip install ygg[postgres]` "
    "or `pip install adbc-driver-postgresql`."
)


def psycopg_module(*, install: bool = False):
    return _lazy_import("psycopg", "psycopg[binary]", install=install, hint=_PSYCOPG_HINT)


def adbc_dbapi_module(*, install: bool = False):
    # The dbapi submodule lives under ``adbc_driver_postgresql``; we
    # install the driver package and return its ``dbapi`` attribute.
    mod = _lazy_import(
        "adbc_driver_postgresql.dbapi",
        "adbc-driver-postgresql",
        install=install,
        hint=_ADBC_HINT,
    )
    return mod


def has_psycopg() -> bool:
    """Probe-only — never raises."""
    try:
        psycopg_module()
    except ImportError:
        return False
    return True


def has_adbc() -> bool:
    """Probe-only — never raises."""
    try:
        adbc_dbapi_module()
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# MongoDB — pymongo (+ bson) and the optional pymongoarrow fast path
# ---------------------------------------------------------------------------


_PYMONGO_HINT = (
    "pymongo is required for yggdrasil.mongo; install it with "
    "`pip install ygg[mongo]` or `pip install pymongo>=4.5`."
)
_PMA_HINT = (
    "pymongoarrow is required for the Arrow-native MongoDB path; "
    "install it with `pip install ygg[mongo]` or "
    "`pip install pymongoarrow>=1.3`."
)


def pymongo_module(*, install: bool = False):
    return _lazy_import("pymongo", install=install, hint=_PYMONGO_HINT)


def bson_module(*, install: bool = False):
    return _lazy_import("bson", "pymongo", install=install, hint=_PYMONGO_HINT)


def pymongoarrow_module(*, install: bool = False):
    return _lazy_import("pymongoarrow", install=install, hint=_PMA_HINT)


def pymongoarrow_api_module(*, install: bool = False):
    return _lazy_import("pymongoarrow.api", "pymongoarrow", install=install, hint=_PMA_HINT)


def pymongoarrow_schema_module(*, install: bool = False):
    return _lazy_import("pymongoarrow.schema", "pymongoarrow", install=install, hint=_PMA_HINT)


def pymongoarrow_writer_module(*, install: bool = False):
    """``pymongoarrow.writer`` — only present in newer pymongoarrow.

    Returns ``None`` when the installed pymongoarrow is too old for
    the writer surface; callers should fall back to
    ``pymongoarrow.api.write`` or the pymongo bulk path.
    """
    pymongoarrow_module(install=install)
    try:
        return _lazy_import("pymongoarrow.writer", "pymongoarrow", install=False)
    except ImportError:
        return None


def has_pymongo() -> bool:
    """Probe-only — never raises."""
    try:
        pymongo_module()
    except ImportError:
        return False
    return True


def has_pymongoarrow() -> bool:
    """Probe-only — never raises."""
    try:
        pymongoarrow_module()
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# SQL — sqlglot is the parser; polars is the preferred execution backend
# ---------------------------------------------------------------------------


_SQLGLOT_HINT = (
    "sqlglot is required for yggdrasil.sql parsing; "
    "install it with `pip install sqlglot` or "
    "`pip install ygg[sql]`."
)


def sqlglot_module(*, install: bool = False):
    return _lazy_import("sqlglot", install=install, hint=_SQLGLOT_HINT)


def sqlglot_expressions(*, install: bool = False):
    return _lazy_import(
        "sqlglot.expressions", "sqlglot", install=install, hint=_SQLGLOT_HINT,
    )


def has_sqlglot() -> bool:
    """Probe-only — never raises."""
    try:
        sqlglot_module()
    except ImportError:
        return False
    return True


def has_polars() -> bool:
    """Probe-only — never raises."""
    try:
        import polars  # noqa: F401
    except ImportError:
        return False
    return True


# ===========================================================================
# Module-level attribute access — `from yggdrasil.lazy_imports import polars`
# ===========================================================================


#: Map attribute name → loader function. Touching one of these on
#: this module triggers the loader and caches the result via
#: ``lru_cache`` on the loader itself.
_LAZY_ATTRS: "dict[str, Any]" = {
    "polars": polars_module,
    "pandas": pandas_module,
    "pyarrow": pyarrow_module,
    "fastapi": fastapi_module,
    "requests": requests_module,
    "confluent_kafka": confluent_kafka_module,
    "databricks_sdk": databricks_sdk_module,
    "AccountClient": databricks_account_client_class,
    "WorkspaceClient": databricks_workspace_client_class,
    "Config": databricks_config_class,
    "DatabricksError": databricks_error_class,
    "psycopg": psycopg_module,
    "adbc_dbapi": adbc_dbapi_module,
    "pymongo": pymongo_module,
    "bson": bson_module,
    "pymongoarrow": pymongoarrow_module,
    "sqlglot": sqlglot_module,
}


def __getattr__(name: str) -> Any:
    loader = _LAZY_ATTRS.get(name)
    if loader is not None:
        return loader()
    raise AttributeError(
        f"module 'yggdrasil.lazy_imports' has no attribute {name!r}"
    )


PATH_SCHEME_FACTORY = {
    "file": local_path_class,
    "s3": aws_s3_path_class,
    "s3a": aws_s3_path_class,
    "s3n": aws_s3_path_class,
    "dbfs": databricks_path_class,
}
