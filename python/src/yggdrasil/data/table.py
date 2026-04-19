from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from yggdrasil.dataclasses.waiting import WaitingConfigArg
from . import CastOptionsArg

if TYPE_CHECKING:
    from .statement_result import Statement

__all__ = [
    "DataTable"
]


class DataTable(ABC):

    @abstractmethod
    def execute(
        self,
        statement: str,
        *,
        wait: WaitingConfigArg = None
    ) -> Statement:
        pass

    @abstractmethod
    def insert(
        self,
        data: Any,
        *,
        cast: CastOptionsArg = None
    ):
        pass