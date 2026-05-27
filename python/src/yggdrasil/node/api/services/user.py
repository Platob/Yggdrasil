from __future__ import annotations

import datetime as dt
import logging
import socket
from threading import Lock

from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.environ.userinfo import UserInfo

from ...config import Settings
from ...exceptions import NotFoundError
from ...ids import make_id
from ..schemas.user import UserCard, UserListResponse

LOGGER = logging.getLogger(__name__)


class UserService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._users: ExpiringDict[int, UserCard] = ExpiringDict(default_ttl=600.0, max_size=256)
        self._lock = Lock()
        self._self_card: UserCard | None = None

        # Auto-register the local user on startup
        info = UserInfo.current()
        now = self._now()
        user_id = make_id(info.key)
        card = UserCard(
            user_id=user_id,
            key=info.key,
            hostname=info.hostname,
            email=info.email,
            first_name=info.first_name,
            last_name=info.last_name,
            node_id=settings.node_id,
            role="admin",
            online=True,
            last_seen_at=now,
        )
        with self._lock:
            self._users.set(user_id, card)
        self._self_card = card
        LOGGER.info("Registered local user %r on node %r", info.key, settings.node_id)

    def get_self(self) -> UserCard:
        return self._self_card  # type: ignore[return-value]

    def register(self, card: UserCard) -> UserCard:
        now = self._now()
        card = card.model_copy(update={"last_seen_at": now})
        with self._lock:
            self._users.set(card.user_id, card)
        LOGGER.info("Registered user %r from node %r", card.key, card.node_id)
        return card

    def list(self) -> UserListResponse:
        with self._lock:
            items = list(self._users.values())
        return UserListResponse(node_id=self.settings.node_id, users=items)

    def get(self, user_id: int) -> UserCard:
        with self._lock:
            card = self._users.get(user_id)
        if card is None:
            raise NotFoundError(f"User {user_id!r} not found")
        return card

    @staticmethod
    def _now() -> str:
        return dt.datetime.now(dt.timezone.utc).isoformat()
