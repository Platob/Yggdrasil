import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Mapping, Any, Union, TYPE_CHECKING

from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg, DEFAULT_WAITING_CONFIG
from .dynamic_buffer import DynamicBuffer
from .request import PreparedRequest
from .response import Response
from .url import URL

if TYPE_CHECKING:
    from ..databricks.sql.table import Table


__all__ = ["Session"]


@dataclass
class Session(ABC):
    base_url: Optional[URL] = None
    verify: bool = True
    waiting: WaitingConfig = field(default_factory=lambda: DEFAULT_WAITING_CONFIG)

    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.base_url:
            self.base_url = URL.parse_any(self.base_url)

        if self._lock is None:
            self._lock = threading.RLock()

    def __getstate__(self) -> dict:
        return {
            "base_url": None if self.base_url is None else self.base_url.to_string(),
            "verify": bool(self.verify),
            "waiting": self.waiting,
        }

    def __setstate__(self, state: dict) -> None:
        base_url_s = state.get("base_url")
        self.base_url = URL.parse_any(base_url_s) if base_url_s else None
        self.verify = bool(state.get("verify", True))
        self.waiting = state.get("waiting") or DEFAULT_WAITING_CONFIG
        self._lock = threading.RLock()

    @classmethod
    def from_url(
        cls,
        url: Union[URL, str],
        *,
        verify: bool = True,
        normalize: bool = True,
        waiting: Optional[WaitingConfigArg] = True,
    ):
        url = URL.parse_any(url, normalize=normalize)

        if url.scheme.startswith("http"):
            from .http_ import HTTPSession

            return HTTPSession(
                base_url=url,
                verify=verify,
                waiting=WaitingConfig.check_arg(waiting) if waiting is not None else None,
            )

        raise ValueError(f"Cannot build session from scheme: {url.scheme}")

    @abstractmethod
    def send(
        self,
        request: PreparedRequest,
        *,
        add_statistics: Optional[bool] = None,
        stream: bool = True,
        wait: Optional[WaitingConfigArg] = None,
        cache: Optional["Table"] = None
    ) -> Response:
        raise NotImplementedError

    # --- Convenience HTTP Methods ---

    def get(
        self,
        url: Optional[Union[URL, str]] = None,
        *,
        params: Optional[Mapping[str, str]] = None,
        add_statistics: Optional[bool] = None,
        headers: Optional[Mapping[str, str]] = None,
        stream: bool = True,
        wait: Optional[WaitingConfigArg] = None,
        normalize: bool = True,
        cache: Optional["Table"] = None
    ) -> Response:
        return self.request(
            "GET",
            url,
            params=params,
            add_statistics=add_statistics,
            headers=headers,
            stream=stream,
            wait=wait,
            normalize=normalize,
            cache=cache,
        )

    def post(
        self,
        url: Optional[Union[URL, str]] = None,
        *,
        params: Optional[Mapping[str, str]] = None,
        add_statistics: Optional[bool] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Union[DynamicBuffer, bytes]] = None,
        json: Optional[Any] = None,
        stream: bool = True,
        wait: Optional[WaitingConfigArg] = None,
        normalize: bool = True,
        cache: Optional["Table"] = None
    ) -> Response:
        return self.request(
            "POST",
            url,
            params=params,
            add_statistics=add_statistics,
            headers=headers,
            body=body,
            json=json,
            stream=stream,
            wait=wait,
            normalize=normalize,
            cache=cache,
        )

    def put(
        self,
        url: Optional[Union[URL, str]] = None,
        *,
        params: Optional[Mapping[str, str]] = None,
        add_statistics: Optional[bool] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Union[DynamicBuffer, bytes]] = None,
        json: Optional[Any] = None,
        stream: bool = True,
        wait: Optional[WaitingConfigArg] = None,
        normalize: bool = True,
        cache: Optional["Table"] = None
    ) -> Response:
        return self.request(
            "PUT",
            url,
            params=params,
            add_statistics=add_statistics,
            headers=headers,
            body=body,
            json=json,
            stream=stream,
            wait=wait,
            normalize=normalize,
            cache=cache,
        )

    def patch(
        self,
        url: Optional[Union[URL, str]] = None,
        *,
        params: Optional[Mapping[str, str]] = None,
        add_statistics: Optional[bool] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Union[DynamicBuffer, bytes]] = None,
        json: Optional[Any] = None,
        stream: bool = True,
        wait: Optional[WaitingConfigArg] = None,
        normalize: bool = True,
        cache: Optional["Table"] = None
    ) -> Response:
        return self.request(
            "PATCH",
            url,
            params=params,
            add_statistics=add_statistics,
            headers=headers,
            body=body,
            json=json,
            stream=stream,
            wait=wait,
            normalize=normalize,
            cache=cache,
        )

    def delete(
        self,
        url: Optional[Union[URL, str]] = None,
        *,
        params: Optional[Mapping[str, str]] = None,
        add_statistics: Optional[bool] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Union[DynamicBuffer, bytes]] = None,
        json: Optional[Any] = None,
        stream: bool = True,
        wait: Optional[WaitingConfigArg] = None,
        normalize: bool = True,
        cache: Optional["Table"] = None
    ) -> Response:
        return self.request(
            "DELETE",
            url,
            params=params,
            add_statistics=add_statistics,
            headers=headers,
            body=body,
            json=json,
            stream=stream,
            wait=wait,
            normalize=normalize,
            cache=cache,
        )

    def head(
        self,
        url: Optional[Union[URL, str]] = None,
        *,
        params: Optional[Mapping[str, str]] = None,
        add_statistics: Optional[bool] = None,
        headers: Optional[Mapping[str, str]] = None,
        stream: bool = False,
        wait: Optional[WaitingConfigArg] = None,
        normalize: bool = True,
        cache: Optional["Table"] = None
    ) -> Response:
        return self.request(
            "HEAD",
            url,
            params=params,
            add_statistics=add_statistics,
            headers=headers,
            stream=stream,
            wait=wait,
            normalize=normalize,
            cache=cache,
        )

    def options(
        self,
        url: Optional[Union[URL, str]] = None,
        *,
        params: Optional[Mapping[str, str]] = None,
        add_statistics: Optional[bool] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Union[DynamicBuffer, bytes]] = None,
        json: Optional[Any] = None,
        stream: bool = True,
        wait: Optional[WaitingConfigArg] = None,
        normalize: bool = True,
        cache: Optional["Table"] = None
    ) -> Response:
        return self.request(
            "OPTIONS",
            url,
            params=params,
            add_statistics=add_statistics,
            headers=headers,
            body=body,
            json=json,
            stream=stream,
            wait=wait,
            normalize=normalize,
            cache=cache,
        )

    # --- Request Orchestration ---

    def request(
        self,
        method: str,
        url: Optional[Union[URL, str]] = None,
        *,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Union[DynamicBuffer, bytes]] = None,
        json: Optional[Any] = None,
        stream: bool = True,
        add_statistics: Optional[bool] = None,
        wait: Optional[WaitingConfigArg] = None,
        normalize: bool = True,
        cache: Optional["Table"] = None
    ) -> Response:
        if add_statistics is None:
            add_statistics = cache is not None

        request = self.prepare_request(
            method=method,
            url=url,
            params=params,
            headers=headers,
            body=body,
            json=json,
            normalize=normalize,
        )

        return self.send(
            request=request,
            add_statistics=add_statistics,
            stream=stream,
            wait=wait,
            cache=cache
        )

    def prepare_request(
        self,
        method: str,
        url: Optional[Union[URL, str]] = None,
        *,
        params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[Union[DynamicBuffer, bytes]] = None,
        json: Optional[Any] = None,
        normalize: bool = True,
        cache: Optional["Table"] = None
    ) -> PreparedRequest:
        full_url = url
        if self.base_url:
            full_url = self.base_url.join(url) if url else self.base_url
        elif url is None:
            raise ValueError("URL is required if base_url is not set.")

        # Apply params (merge with existing query)
        if params:
            u = URL.parse_any(full_url, normalize=normalize)
            items = list(u.query_items(keep_blank_values=True))
            items.extend((k, v) for k, v in params.items())
            full_url = u.with_query_items(tuple(items))

        return PreparedRequest.prepare(
            method=method,
            url=full_url,
            headers=headers,
            body=body,
            json=json,
            normalize=normalize,
        )
