from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests

from .exceptions import HttpMongoError, HttpMongoTransportError
from .serde import BsonJsonSerde


class HttpTransport:
    """Tiny RPC transport over HTTP for the Mongo gateway."""

    def __init__(
        self,
        host: Optional[str] = None,
        token: Optional[str] = None,
        timeout: float = 30.0,
        session: Optional[requests.Session] = None,
        **_: Any,
    ) -> None:
        self.base_url = (host or os.environ.get("MONGODB_URI") or "http://127.0.0.1:8000").rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.serde = BsonJsonSerde()
        self.headers = {
            "Content-Type": "application/json",
            "X-Auth-Token": token or os.environ.get("HTTPMONGO_TOKEN", "dev-secret"),
        }

    def close(self) -> None:
        self.session.close()

    def rpc(
        self,
        *,
        database: str,
        operation: str,
        arguments: Dict[str, Any],
        collection: Optional[str] = None,
    ) -> Any:
        payload = {
            "database": database,
            "collection": collection,
            "operation": operation,
            "arguments": arguments,
        }

        try:
            response = self.session.post(
                f"{self.base_url}/rpc",
                headers=self.headers,
                data=self.serde.dumps(payload),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise HttpMongoTransportError(f"request failed: {exc}") from exc

        try:
            data = self.serde.loads(response.text)
        except Exception as exc:  # noqa: BLE001
            raise HttpMongoTransportError(
                f"invalid response: status={response.status_code}, body={response.text[:500]}"
            ) from exc

        if not response.ok:
            raise HttpMongoTransportError(f"http error {response.status_code}: {data}")

        if data.get("ok") != 1:
            error = data.get("error", {})
            raise HttpMongoError(f"{error.get('type', 'RemoteError')}: {error.get('message', data)}")

        return data.get("result")
