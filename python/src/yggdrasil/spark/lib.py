try:
    import pyspark
except ImportError:
    from ..pyutils.pyenv import PyEnv

    pyspark = PyEnv.runtime_import_module(module_name="pyspark", pip_name="pyspark")

pyspark_sql = pyspark.sql


__all__ = [
    "pyspark",
    "pyspark_sql"
]
