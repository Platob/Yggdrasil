from dataclasses import dataclass

from ..client import DatabricksService

__all__ = [
    "Accounts"
]

@dataclass
class Accounts(DatabricksService):
    pass
