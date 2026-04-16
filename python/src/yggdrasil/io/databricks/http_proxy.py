from dataclasses import dataclass

from ..http_ import HTTPSession

__all__ = [
    "DBXHTTPSession"
]


@dataclass
class DBXHTTPSession(HTTPSession):
    pass