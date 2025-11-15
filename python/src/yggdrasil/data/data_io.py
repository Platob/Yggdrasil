"""
Abstract DataWriter class for writing data in different formats.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

from yggdrasil.data.table_location import TableLocation

from ..types.field import DataField
from ..utils.spark_utils import spark_sql

try:
    from delta.tables import DeltaTable
except ImportError:
    DeltaTable = None

__all__ = [
    "SaveMode",
    "DataIO"
]

class SaveMode(Enum):
    Overwrite = "overwrite"
    Append = "append"


class DataIO(ABC):
    """
    Abstract base class for data writers.

    Implementations should provide methods to write data from polars and spark
    dataframes to various destinations.
    """
    __spark: spark_sql.SparkSession = None

    @classmethod
    def get_spark(cls):
        if cls.__spark is None:
            cls.__spark = spark_sql.SparkSession.getActiveSession()

        if not cls.__spark:
            raise ValueError(f"No spark session available for {cls.__name__}")

        return cls.__spark

    @abstractmethod
    def get_schema(self, location: TableLocation) -> DataField:
        """
        Get the DataField representing the schema of the data.

        Args:
            location: The TableLocation object representing the data location.
        Returns:
            DataField: The field representing the schema of the data.
        """
        pass

    @abstractmethod
    def read_spark(
        self,
        location: TableLocation,
        schema: Optional[DataField] = None,
        **kwargs
    ) -> spark_sql.DataFrame:
        """
        Write data from a spark DataFrame.

        Args:
            location: TableLocation
            schema: DataField to select as or select all
            **kwargs: Additional arguments for the writer.
        """
        pass

    def write_spark(
        self,
        location: TableLocation,
        df: spark_sql.DataFrame,
        mode: SaveMode = SaveMode.Overwrite,
        **kwargs
    ) -> None:
        """
        Write data from a spark DataFrame.

        Args:
            location: TableLocation
            df: The spark DataFrame to write.
            mode: The mode to write the data to.
            **kwargs: Additional arguments for the writer.
        """
        target_schema = self.get_schema(location)
        casted = target_schema.cast_spark_dataframe(df)

        return self._write_spark(
            location=location,
            df=casted,
            mode=mode,
            **kwargs
        )


    @abstractmethod
    def _write_spark(
        self,
        location: TableLocation,
        df: spark_sql.DataFrame,
        mode: SaveMode = SaveMode.Overwrite,
        **kwargs
    ) -> None:
        """
        Write data from a spark DataFrame.

        Args:
            location: TableLocation
            df: The spark DataFrame to write.
            mode: The mode to write the data to.
            **kwargs: Additional arguments for the writer.
        """
        pass
