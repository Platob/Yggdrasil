from __future__ import annotations

from typing import Any, Dict, List

from .collection import HttpCollection
from .transport import HttpTransport


class HttpDatabase:
    def __init__(self, transport: HttpTransport, name: str) -> None:
        self._transport = transport
        self.name = name

    def __getitem__(self, collection_name: str) -> HttpCollection:
        return HttpCollection(self._transport, self.name, collection_name)

    def get_collection(self, name: str, **_: Any) -> HttpCollection:
        return self[name]

    def list_collection_names(self) -> List[str]:
        return self._transport.rpc(database=self.name, operation="list_collection_names", arguments={})

    def command(self, command: Any, **kwargs: Any) -> Dict[str, Any]:
        if isinstance(command, str):
            payload = {command: 1, **kwargs} if kwargs else {command: 1}
        else:
            payload = command
        return self._transport.rpc(database=self.name, operation="command", arguments={"command": payload})
