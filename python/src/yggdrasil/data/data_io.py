"""
Abstract DataWriter class for writing data in different formats.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

from yggdrasil.data.table_location import TableLocation
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


@dataclass(frozen=True)
class DataTableIO(ABC):
    """
    Abstract base class for data writers.

    Implementations should provide methods to write data from polars and spark
    dataframes to various destinations.
    """
    location: TableLocation
    schema: DataField | None

    def __post_init__(self):
        setattr(self, "location", TableLocation.parse_any(self.location))

        if self.schema:
            setattr(self, "schema", DataField.parse_any(self.schema))

    @classmethod
    def get_spark(cls):
        session = spark_sql.SparkSession.getActiveSession()

        if not session:
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
