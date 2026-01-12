"""requests.session module documentation."""

from typing import Optional, Dict

from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

__all__ = [
    "YGGSession"
]


class YGGSession(Session):
    def __init__(
        self,
        num_retry: int = 4,
        headers: Optional[Dict[str, str]] = None,
        *args,
        **kwargs
    ):
        """
        __init__ documentation.
        
        Args:
            num_retry: Parameter.
            headers: Parameter.
            *args: Parameter.
            **kwargs: Parameter.
        
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
