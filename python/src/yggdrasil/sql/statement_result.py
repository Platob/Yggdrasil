from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow
    import polars


__all__ = [
    "StatementResult"
]


@dataclass
class StatementResult:

    @abstractmethod
    def to_arrow_table(
        self,
        *args,
        **kwargs
    ) -> "pyarrow.Table":
        raise NotImplementedError

    @abstractmethod
    def to_polars(
        self,
        *args,
        **kwargs
    ) -> "polars.DataFrame":
        raise NotImplementedError
