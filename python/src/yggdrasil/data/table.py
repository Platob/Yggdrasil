from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from yggdrasil.dataclasses.waiting import WaitingConfigArg
from ..types.cast.cast_options import CastOptionsArg

if TYPE_CHECKING:
    from .statement_result import StatementResult

__all__ = [
    "DataTable"
]


class DataTable(ABC):

    @abstractmethod
    def execute(
        self,
        statement: str,
        *,
        wait: WaitingConfigArg | None = None
    ) -> StatementResult:
        pass

    @abstractmethod
    def insert(
        self,
        data: Any,
        *,
        cast: CastOptionsArg = None
    ):
        pass