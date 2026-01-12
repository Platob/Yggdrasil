"""HTTP session helpers with retry-enabled defaults."""

from typing import Optional, Dict

from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

__all__ = [
    "YGGSession"
]


class YGGSession(Session):
    """Requests session with preconfigured retry adapter support.

    Args:
        Session: Base requests session type.
    """
    def __init__(
        self,
        num_retry: int = 4,
        headers: Optional[Dict[str, str]] = None,
        *args,
        **kwargs
    ):
        """Initialize the session with retries and optional default headers.

        Args:
            num_retry: Number of retries for connection and read errors.
            headers: Optional default headers to merge into the session.
            *args: Additional positional arguments passed to Session.
            **kwargs: Additional keyword arguments passed to Session.

        Returns:
            None.
        """
        super(YGGSession, self).__init__()

        retry = Retry(
            total=num_retry,
            read=num_retry,
            connect=num_retry,
            backoff_factor=0.1
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.mount('https://', adapter)
        self.mount('http://', adapter)

        if headers:
            for k, v in headers.items():
                self.headers[k] = self.headers.get(k, v)
