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
def media_type_class():
    from yggdrasil.data.enums import MediaType
    return MediaType


@lru_cache(maxsize=1)
def media_types_class():
    from yggdrasil.data.enums import MediaTypes
    return MediaTypes


@lru_cache(maxsize=1)
def mime_type_class():
    from yggdrasil.data.enums import MimeType
    return MimeType


@lru_cache(maxsize=1)
def mime_types_class():
    from yggdrasil.data.enums import MimeTypes
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
def aws_config_class():
    from yggdrasil.aws.config import AWSConfig
    return AWSConfig


@lru_cache(maxsize=1)
def aws_s3_path_class():
    from yggdrasil.aws.fs.path import S3Path
    return S3Path


# ===========================================================================
# Optional third-party packages — try-import, fall back to runtime install
# ===========================================================================


def _import_or_install(module_name: str, pip_name: str | None = None) -> Any:
    """Try a plain import; on failure, fall back to the runtime installer.

    The fallback path mirrors what the old per-package ``lib.py``
    files did — ``PyEnv.runtime_import_module`` will pip-install the
    package if the runtime allows it, then re-import.
    """
    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError:
        from yggdrasil.environ import PyEnv
        return PyEnv.runtime_import_module(
            module_name=module_name,
            pip_name=pip_name or module_name,
            install=True,
        )


@lru_cache(maxsize=1)
def polars_module():
    return _import_or_install("polars", "polars")


@lru_cache(maxsize=1)
def pandas_module():
    return _import_or_install("pandas", "pandas")


@lru_cache(maxsize=1)
def pyarrow_module():
    return _import_or_install("pyarrow", "pyarrow")


@lru_cache(maxsize=1)
def fastapi_module():
    return _import_or_install("fastapi", "fastapi")


@lru_cache(maxsize=1)
def requests_module():
    return _import_or_install("requests", "requests")


@lru_cache(maxsize=1)
def xxhash_module():
    return _import_or_install("xxhash", "xxhash")


@lru_cache(maxsize=1)
def confluent_kafka_module():
    return _import_or_install("confluent_kafka", "confluent-kafka")


@lru_cache(maxsize=1)
def spark_sql_module():
    import pyspark.sql as sql
    return sql


@lru_cache(maxsize=1)
def boto3_module():
    try:
        import boto3
        return boto3
    except ImportError:
        from yggdrasil.environ import runtime_import_module
        return runtime_import_module("boto3", install=True)


@lru_cache(maxsize=1)
def botocore_module():
    """Lazy-import botocore.

    We want the top-level ``botocore`` module so callers can reach
    ``botocore.exceptions``, ``botocore.config``, ``botocore.credentials``,
    and ``botocore.session`` without triggering separate imports.
    """
    try:
        import botocore
        # Touch submodules we use so they're in sys.modules — this
        # makes ``botocore.exceptions.ClientError`` etc. resolve
        # without each call paying its own import.
        import botocore.exceptions  # noqa: F401
        import botocore.config  # noqa: F401
        import botocore.credentials  # noqa: F401
        import botocore.session  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "yggdrasil.aws requires 'botocore' (a transitive dep of boto3). "
            "Install boto3 with `pip install boto3` to get it."
        ) from exc
    return botocore


# ---------------------------------------------------------------------------
# Databricks SDK
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def databricks_sdk_module():
    return _import_or_install("databricks.sdk", "databricks-sdk")


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


@lru_cache(maxsize=1)
def psycopg_module():
    try:
        import psycopg
    except ImportError as exc:
        raise ImportError(_PSYCOPG_HINT) from exc
    return psycopg


@lru_cache(maxsize=1)
def adbc_dbapi_module():
    try:
        from adbc_driver_postgresql import dbapi
    except ImportError as exc:
        raise ImportError(_ADBC_HINT) from exc
    return dbapi


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


@lru_cache(maxsize=1)
def pymongo_module():
    try:
        import pymongo
    except ImportError as exc:
        raise ImportError(_PYMONGO_HINT) from exc
    return pymongo


@lru_cache(maxsize=1)
def bson_module():
    try:
        import bson
    except ImportError as exc:
        raise ImportError(_PYMONGO_HINT) from exc
    return bson


@lru_cache(maxsize=1)
def pymongoarrow_module():
    try:
        import pymongoarrow
    except ImportError as exc:
        raise ImportError(_PMA_HINT) from exc
    return pymongoarrow


@lru_cache(maxsize=1)
def pymongoarrow_api_module():
    try:
        from pymongoarrow import api
    except ImportError as exc:
        raise ImportError(_PMA_HINT) from exc
    return api


@lru_cache(maxsize=1)
def pymongoarrow_schema_module():
    try:
        from pymongoarrow import schema
    except ImportError as exc:
        raise ImportError(_PMA_HINT) from exc
    return schema


@lru_cache(maxsize=1)
def pymongoarrow_writer_module():
    """``pymongoarrow.writer`` — only present in newer pymongoarrow.

    Returns ``None`` when the installed pymongoarrow is too old for
    the writer surface; callers should fall back to
    ``pymongoarrow.api.write`` or the pymongo bulk path.
    """
    pymongoarrow_module()
    try:
        from pymongoarrow import writer
    except ImportError:
        return None
    return writer


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


@lru_cache(maxsize=1)
def sqlglot_module():
    try:
        import sqlglot
    except ImportError as exc:
        raise ImportError(_SQLGLOT_HINT) from exc
    return sqlglot


@lru_cache(maxsize=1)
def sqlglot_expressions():
    sqlglot_module()
    from sqlglot import expressions
    return expressions


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
    "xxhash": xxhash_module,
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
