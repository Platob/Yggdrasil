from __future__ import annotations

import getpass
import socket
import os
from dataclasses import dataclass

__all__ = ["UserInfo", "get_user_info"]


@dataclass(frozen=True, slots=True)
class UserInfo:
    email: str | None
    username: str
    computer: str


def get_user_info() -> UserInfo:
    return UserInfo(
        email=os.environ.get("EMAIL"),
        username=getpass.getuser(),
        computer=socket.gethostname(),
    )
