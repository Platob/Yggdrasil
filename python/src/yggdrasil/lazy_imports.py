from functools import lru_cache


@lru_cache(maxsize=1)
def bytes_io_class():
    from yggdrasil.io.buffer.bytes_io import BytesIO
    return BytesIO


@lru_cache(maxsize=1)
def media_type_class():
    from yggdrasil.io.enums import MediaType
    return MediaType


@lru_cache(maxsize=1)
def media_types_class():
    from yggdrasil.io.enums import MediaTypes
    return MediaTypes


@lru_cache(maxsize=1)
def mime_type_class():
    from yggdrasil.io.enums import MimeType
    return MimeType


@lru_cache(maxsize=1)
def mime_types_class():
    from yggdrasil.io.enums import MimeTypes
    return MimeTypes


@lru_cache(maxsize=1)
def path_class():
    from yggdrasil.io.fs.path import Path
    return Path


@lru_cache(maxsize=1)
def local_path_class():
    from yggdrasil.io.fs.local_path import LocalPath
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
def polars_module():
    import polars as pl
    return pl


@lru_cache(maxsize=1)
def pandas_module():
    import pandas as pd
    return pd


@lru_cache(maxsize=1)
def spark_sql_module():
    import pyspark.sql as sql
    return sql


@lru_cache(maxsize=1)
def tabular_io_class():
    from yggdrasil.io.tabular import TabularIO
    return TabularIO


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
    """Lazy-import AWS S3 path class."""
    from yggdrasil.aws.fs.path import S3Path
    return S3Path


PATH_SCHEME_FACTORY = {
    "file": local_path_class,
    "s3": aws_s3_path_class,
    "s3a": aws_s3_path_class,
    "s3n": aws_s3_path_class,
    "dbfs": databricks_path_class,
}