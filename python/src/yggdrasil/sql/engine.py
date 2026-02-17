from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, TYPE_CHECKING

import pyarrow as pa

from .statement_result import StatementResult
from ..pyutils.waiting_config import WaitingConfigArg
from ..types.cast.cast_options import CastOptions

if TYPE_CHECKING:
    import polars
    import pandas


__all__ = [
    "SQLEngine"
]


@dataclass
class SQLEngine(ABC):
    catalog_name: Optional[str] = None
    schema_name: Optional[str] = None

    @abstractmethod
    def execute(
        self,
        statement: str,
        *,
        row_limit: Optional[int] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        wait: Optional[WaitingConfigArg] = True,
        **kwargs
    ) -> StatementResult:
        raise NotImplementedError

    @abstractmethod
    def insert_into(
        self,
        data: Union[
            pa.Table, pa.RecordBatch, pa.RecordBatchReader,
            dict, list, str,
            "pandas.DataFrame", "polars.DataFrame",
            "pyspark.sql.DataFrame"
        ],
        *,
        mode: str = "auto",
        location: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        cast_options: Optional[CastOptions] = None,
        overwrite_schema: bool | None = None,
        match_by: Optional[list[str]] = None,
        wait: Optional[WaitingConfigArg] = True
    ):
        raise NotImplementedError
