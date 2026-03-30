from __future__ import annotations

from typing import Any, Dict

from .database import HttpDatabase
from .transport import HttpTransport


class HttpMongoClient:
    """PyMongo-ish client that MongoEngine can use via mongo_client_class."""

    def __init__(self, host: str | None = None, *args: Any, **kwargs: Any) -> None:
        self._transport = HttpTransport(host=host, **kwargs)

    def __getitem__(self, db_name: str) -> HttpDatabase:
        return HttpDatabase(self._transport, db_name)

    def get_database(self, name: str, **_: Any) -> HttpDatabase:
        return self[name]

    @property
    def admin(self) -> HttpDatabase:
        return self["admin"]

    def server_info(self) -> Dict[str, Any]:
        return self._transport.rpc(database="admin", operation="server_info", arguments={})

    def close(self) -> None:
        self._transport.close()
