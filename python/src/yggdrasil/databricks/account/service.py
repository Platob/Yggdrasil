from dataclasses import dataclass

from ..client import DatabricksService

__all__ = [
    "Accounts"
]

@dataclass(frozen=True)
class Accounts(DatabricksService):
    pass
