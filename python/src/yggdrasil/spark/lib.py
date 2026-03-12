try:
    import pyspark
except ImportError:
    from yggdrasil.environ import runtime_import_module

    pyspark = runtime_import_module(module_name="pyspark", pip_name="pyspark", install=True)

pyspark_sql = pyspark.sql


__all__ = [
    "pyspark",
    "pyspark_sql"
]
