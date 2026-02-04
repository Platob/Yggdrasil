"""MSAL-backed authentication helpers for requests sessions."""

# auth_session.py
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .session import YGGSession

try:
    import msal

    ConfidentialClientApplication = msal.ConfidentialClientApplication
except ImportError:
    msal = None
    ConfidentialClientApplication = Any


__all__ = [
    "MSALSession",
    "MSALAuth"
]


@dataclass
class MSALAuth:
    """Configuration and token cache for MSAL client credential flows.

    Args:
        tenant_id: Azure tenant ID.
        client_id: Azure application client ID.
        client_secret: Azure application client secret.
        authority: Optional authority URL override.
        scopes: List of scopes to request.
    """
    tenant_id: Optional[str] = field(default_factory=lambda: os.environ.get("AZURE_TENANT_ID"))
    client_id: Optional[str] = field(default_factory=lambda: os.environ.get("AZURE_CLIENT_ID"))
    client_secret: Optional[str] = field(default_factory=lambda: os.environ.get("AZURE_CLIENT_SECRET"))
    authority: Optional[str] = field(default_factory=lambda: os.environ.get("AZURE_AUTHORITY"))
    scopes: list[str] | None = field(default_factory=lambda: os.environ.get("AZURE_SCOPES"))

    _auth_app: ConfidentialClientApplication | None = None
    _expires_at: float | None = None
    _access_token: Optional[str] = None

    def __setitem__(self, key, value):
        """Set an attribute via mapping-style assignment.

        Args:
            key: Attribute name to set.
            value: Value to assign.

        Returns:
            None.
        """
        self.__setattr__(key, value)

    def __getitem__(self, item):
        """Return attribute values via mapping-style access.

        Args:
            item: Attribute name to fetch.

        Returns:
            The attribute value.
        """
        return getattr(self, item)

    def __post_init__(self):
        """Populate defaults from environment variables and validate.

        Returns:
            None.
        """
        if not self.authority:
            assert self.tenant_id, "tenant_id is required to build authority URL"

            self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"

        if self.scopes:
            if isinstance(self.scopes, str):
                self.scopes = self.scopes.split(",")

    @property
    def auth_app(self) -> ConfidentialClientApplication:
        """Return or initialize the MSAL confidential client.

        Returns:
            MSAL confidential client instance.
        """
        if not self._auth_app:
            self._auth_app = ConfidentialClientApplication(
                client_id=self.client_id,
                client_credential=self.client_secret,
                authority=self.authority,

            )
        return self._auth_app

    @property
    def expires_in(self) -> float:
        """Return the number of seconds since the token expiry timestamp.

        Returns:
            Seconds elapsed since expiry (negative if not expired).
        """
        return time.time() - self.expires_at

    @property
    def expires_at(self) -> float:
        """Ensure the token is fresh and return the expiry timestamp.

        Returns:
            Token expiration time as a Unix timestamp.
        """
        self.refresh()

        return self._expires_at

    @property
    def expired(self) -> bool:
        """Return True when the token is missing or past its expiry time.

        Returns:
            True if expired or missing; False otherwise.
        """
        return not self._expires_at or time.time() >= self._expires_at

    def refresh(self, force: bool | None = None):
        """Acquire or refresh the token if needed.

        Args:
            force: Force refresh even if not expired.

        Returns:
            The updated MSALAuth instance.
        """
        if self.expired or force:
            app = self.auth_app
            result = app.acquire_token_for_client(scopes=self.scopes)
            current_time = time.time()

            self._access_token = result.get("access_token")

            if not self._access_token:
                raise RuntimeError(f"Failed to acquire token: {result.get('error_description') or result}")

            # msal returns expires_in in seconds
            self._expires_at = current_time + int(result.get("expires_in", 3600))

        return self

    @property
    def access_token(self) -> str:
        """Return access token.

        Returns:
            Access token string.
        """
        self.refresh()
        return self._access_token

    @property
    def authorization(self) -> str:
        """Return authorization token.

        Returns:
            Authorization header value.
        """
        return f"Bearer {self.access_token}"

    def requests_session(self, **kwargs):
        """Build a requests session that injects the MSAL authorization header.

        Args:
            **kwargs: Passed through to MSALSession.

        Returns:
            Configured MSALSession.
        """
        return MSALSession(
            msal_auth=self,
            **kwargs
        )


class MSALSession(YGGSession):
    """YGGSession subclass that injects MSAL authorization headers.

    Args:
        YGGSession: Base retry-capable session.
    """
    msal_auth: MSALAuth | None = None

    def __init__(
        self,
        msal_auth: Optional[MSALAuth] = None,
        *args,
        **kwargs: dict
    ):
        """Initialize the session with optional MSAL auth configuration.

        Args:
            msal_auth: MSALAuth configuration for token injection.
            *args: Positional args for YGGSession.
            **kwargs: Keyword args for YGGSession.

        Returns:
            None.
        """
        super().__init__(*args, **kwargs)
        self.msal_auth = msal_auth

    def prepare_request(self, request):
        """Prepare the request with an Authorization header when needed.

        Args:
            request: requests.PreparedRequest to mutate.

        Returns:
            Prepared request.
        """
        # called before sending; ensure header exists
        if self.msal_auth is not None:
            request.headers["Authorization"] = request.headers.get("Authorization", self.msal_auth.authorization)

        return super().prepare_request(request)
