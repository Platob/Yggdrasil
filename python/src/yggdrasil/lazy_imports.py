from functools import lru_cache

__all__ = [
    "fragment_class",
    "fragment_infos_class",
    "path_class",
    "local_path_class",
    "databricks_path_class",
    "databricks_client_class",
    "pyarrow_dataset_module",
    "bytes_io_class",
    "media_type_class",
    "media_types_class",
    "mime_type_class",
    "mime_types_class",
    "polars_module",
    "pandas_module",
    "spark_sql_module",
    "pyarrow_compute_module",
    "field_class",
    "struct_type_class",
    "schema_class",
    "tabular_io_class",
    "primitive_io_class"
]


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
def primitive_io_class():
    from yggdrasil.io.buffer.primitive import PrimitiveIO
    return PrimitiveIO


@lru_cache(maxsize=1)
def fragment_class():
    from yggdrasil.io.fragment import Fragment
    return Fragment


@lru_cache(maxsize=1)
def fragment_infos_class():
    from yggdrasil.io.fragment import FragmentInfos
    return FragmentInfos