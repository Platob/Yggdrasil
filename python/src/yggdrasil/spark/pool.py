from dataclasses import field
from typing import TYPE_CHECKING

from yggdrasil.concurrent.threading import JobPoolExecutor
from yggdrasil.environ import PyEnv

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


__all__ = [
    "SparkPoolExecutor",
]


class SparkPoolExecutor(JobPoolExecutor):
    spark_session: "SparkSession" = field(default_factory=PyEnv.spark_session)

