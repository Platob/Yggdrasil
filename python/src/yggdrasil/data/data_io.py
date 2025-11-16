"""
Abstract DataWriter class for writing data in different formats.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from .table_location import TableLocation
from ..types.field import DataField
from ..utils.spark_utils import spark_sql

try:
    from delta.tables import DeltaTable
except ImportError:
    DeltaTable = None

__all__ = [
    "SaveMode",
    "DataTableIO"
]


class SaveMode(StrEnum):
    Overwrite = "overwrite"
    Append = "append"


@dataclass
class DataTableIO(ABC):
    """
    Abstract base class for data writers.

    Implementations should provide methods to write data from polars and spark
    dataframes to various destinations.
    """
    location: TableLocation
    schema: DataField | None

    @classmethod
    def get_spark(cls, raise_error: bool = True):
        session = spark_sql.SparkSession.getActiveSession()

        if raise_error and not session:
            raise ValueError(f"No spark session available for {cls.__name__}")

        return session

    @classmethod
    @abstractmethod
    def load_schema(cls, location: TableLocation) -> DataField:
        """
        Get the DataField representing the schema of the data.

        Args:
            location: The TableLocation object representing the data location.
        Returns:
            DataField: The field representing the schema of the data.
        """
        pass


@dataclass
class DataTransform:
    pass
