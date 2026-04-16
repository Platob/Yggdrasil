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
    status: str | None = None
    content: str | None = None
    text: str | None = None
    attachment_id: str | None = None
    query: str | None = None
    query_result: Optional[Any] = None
    raw_message: Optional[Any] = None
