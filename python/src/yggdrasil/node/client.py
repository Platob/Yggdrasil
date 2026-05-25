from __future__ import annotations

import logging
from typing import Any, Callable, Iterator

import pyarrow as pa

from yggdrasil.node.transport import (
    CONTENT_TYPE_ARROW_STREAM,
    CONTENT_TYPE_PICKLE,
    deserialize_result,
    read_arrow_stream,
    serialize_pickle,
)

LOGGER = logging.getLogger(__name__)


class NodeClient:
    """Client for calling @remote functions on a bot server.

    Usage::

        client = NodeClient("http://localhost:8100")

        # Call a registered remote function
        result = client.call(my_func, 1, 2, key="val")

        # Or by name
        result = client.call("mymodule:my_func", 1, 2)

        # Execute raw Python code
        result = client.execute("print('hello')")

        # Stream Arrow IPC results
        for batch in client.call_stream(big_query_func, params):
            process(batch)
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8100",
        *,
        api_prefix: str = "/api",
        timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_prefix = api_prefix
        self.timeout = timeout
        self._session = None

    @property
    def session(self):
        if self._session is None:
            try:
                import urllib3
                self._session = urllib3.PoolManager(
                    timeout=urllib3.Timeout(connect=10.0, read=self.timeout),
                )
            except ImportError:
                import urllib.request
                self._session = urllib.request
        return self._session

    def _url(self, path: str) -> str:
        return f"{self.base_url}{self.api_prefix}{path}"

    def _post(
        self,
        path: str,
        body: bytes,
        content_type: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> tuple[bytes, str, dict[str, str]]:
        url = self._url(path)
        hdrs = {"Content-Type": content_type}
        if headers:
            hdrs.update(headers)

        import urllib3
        resp = self.session.request(
            "POST",
            url,
            body=body,
            headers=hdrs,
            timeout=urllib3.Timeout(connect=10.0, read=timeout or self.timeout),
        )

        if resp.status >= 400:
            detail = resp.data.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Bot server returned {resp.status}: {detail}"
            )

        resp_ct = resp.headers.get("Content-Type", "application/octet-stream")
        resp_headers = dict(resp.headers)
        return resp.data, resp_ct, resp_headers

    def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        from yggdrasil.pickle.json import dumps, loads
        body = dumps(payload, to_bytes=True)
        data, ct, _ = self._post(
            path, body, "application/json", timeout=timeout
        )
        return loads(data)

    def _resolve_func(self, func: Callable | str) -> tuple[str, float, list[str] | None]:
        if callable(func) and hasattr(func, "_remote_key"):
            return (
                func._remote_key,
                getattr(func, "_remote_timeout", None) or self.timeout,
                getattr(func, "_remote_modules", None),
            )
        if isinstance(func, str):
            return func, self.timeout, None
        raise TypeError(
            f"Expected a @remote-decorated function or a string key, "
            f"got {type(func).__name__}"
        )

    def call(
        self,
        func: Callable | str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call a @remote function on the bot server."""
        func_key, timeout, modules = self._resolve_func(func)

        call_payload: dict[str, Any] = {
            "func": func_key,
            "args": args,
            "kwargs": kwargs,
        }
        if modules:
            call_payload["modules"] = modules

        payload = serialize_pickle(call_payload)

        data, content_type, resp_headers = self._post(
            "/call",
            payload,
            CONTENT_TYPE_PICKLE,
            timeout=timeout,
        )

        return deserialize_result(data, content_type)

    def call_stream(
        self,
        func: Callable | str,
        *args: Any,
        **kwargs: Any,
    ) -> Iterator[pa.RecordBatch]:
        """Call a @remote function and stream Arrow batches back."""
        func_key, timeout, modules = self._resolve_func(func)

        call_payload: dict[str, Any] = {
            "func": func_key,
            "args": args,
            "kwargs": kwargs,
            "stream": True,
        }
        if modules:
            call_payload["modules"] = modules

        payload = serialize_pickle(call_payload)

        data, content_type, resp_headers = self._post(
            "/call",
            payload,
            CONTENT_TYPE_PICKLE,
            headers={"Accept": CONTENT_TYPE_ARROW_STREAM},
            timeout=timeout,
        )

        if content_type.startswith(CONTENT_TYPE_ARROW_STREAM):
            import pyarrow.ipc as ipc
            reader = ipc.open_stream(data)
            while True:
                try:
                    yield reader.read_next_batch()
                except StopIteration:
                    break
        else:
            result = deserialize_result(data, content_type)
            from yggdrasil.node.transport import to_arrow_table
            table = to_arrow_table(result)
            yield from table.to_batches()

    def execute(self, code: str, **kwargs: Any) -> dict[str, Any]:
        """Execute raw Python code on the bot server."""
        return self._post_json("/python", {"code": code, **kwargs})

    def cmd(self, command: list[str], **kwargs: Any) -> dict[str, Any]:
        """Execute a shell command on the bot server."""
        return self._post_json("/cmd", {"command": command, **kwargs})

    def list_functions(self) -> dict[str, str]:
        """List registered @remote functions on the bot server."""
        from yggdrasil.pickle.json import loads
        data, _, _ = self._post(
            "/call/registry",
            b"",
            "application/json",
        )
        return loads(data)

    def __repr__(self) -> str:
        return f"NodeClient({self.base_url!r})"
