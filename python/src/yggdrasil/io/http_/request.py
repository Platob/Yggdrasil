from dataclasses import dataclass

from ..request import PreparedRequest


__all__ = [
    "HTTPRequest"
]

@dataclass
class HTTPRequest(PreparedRequest):
    pass