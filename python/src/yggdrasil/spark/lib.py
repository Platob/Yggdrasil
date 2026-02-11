try:
    import pyspark

    pyspark_sql = pyspark.sql
except ImportError:
    from ..pyutils.dummy import Dummy

    pyspark = Dummy.from_name(
        "pyspark",
        to_class=False
    )

    pyspark_sql = Dummy.from_name(
        "pyspark", "sql",
        to_class=False
    )

__all__ = [
    "pyspark",
    "pyspark_sql"
]
