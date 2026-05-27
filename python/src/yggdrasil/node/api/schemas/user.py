from __future__ import annotations

from pydantic import Field

from .common import StrictModel


class UserCard(StrictModel):
    user_id: int
    key: str
    hostname: str
    email: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    node_id: str = ""
    role: str = "user"
    online: bool = True
    last_seen_at: str = ""


class UserListResponse(StrictModel):
    node_id: str
    users: list[UserCard]
