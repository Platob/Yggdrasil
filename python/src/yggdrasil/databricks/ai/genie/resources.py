from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

__all__ = [
    "GenieAnswer",
    "GenieSpace",
]


@dataclass(frozen=True, slots=True)
class GenieSpace:
    space_id: str
    conversation_id: str


@dataclass(frozen=True, slots=True)
class GenieAnswer:
    space_id: str
    conversation_id: str
    message_id: str
    status: Optional[str] = None
    content: Optional[str] = None
    text: Optional[str] = None
    attachment_id: Optional[str] = None
    query: Optional[str] = None
    query_result: Optional[Any] = None
    raw_message: Optional[Any] = None
