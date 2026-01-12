"""requests.msal module documentation."""

# auth_session.py
import os
import time
from typing import Any, Mapping, Optional

import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from dataclasses import dataclass

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
    tenant_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    authority: Optional[str] = None
    scopes: list[str] | None = None

    _auth_app: ConfidentialClientApplication | None = None
    _expires_at: float | None = None
    _access_token: Optional[str] = None

    def __setitem__(self, key, value):
        """
        __setitem__ documentation.
        
        Args:
            key: Parameter.
            value: Parameter.
        
        Returns:
            The result.
        """

        self.__setattr__(key, value)

    def __getitem__(self, item):
        """
        __getitem__ documentation.
        
        Args:
            item: Parameter.
        
        Returns:
            The result.
        """

        return getattr(self, item)

    def __post_init__(self):
        """
        __post_init__ documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        self.tenant_id = self.tenant_id or os.environ.get("AZURE_TENANT_ID")
        self.client_id = self.client_id or os.environ.get("AZURE_CLIENT_ID")
        self.client_secret = self.client_secret or os.environ.get("AZURE_CLIENT_SECRET")

        self.authority = self.authority or os.environ.get("AZURE_AUTHORITY")
        if not self.authority:
            self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"

        self.scopes = self.scopes or os.environ.get("AZURE_SCOPES")
        if self.scopes:
            if isinstance(self.scopes, str):
                self.scopes = self.scopes.split(",")

        self._validate_config()

    def _validate_config(self):
        """Validate that all required configuration is present."""
        missing = []

        if not self.client_id:
            missing.append("azure_client_id (AZURE_CLIENT_ID)")
        if not self.client_secret:
            missing.append("azure_client_secret (AZURE_CLIENT_SECRET)")
        if not self.tenant_id:
            missing.append("azure_client_secret (AZURE_TENANT_ID)")
        if not self.scopes:
            missing.append("scopes (AZURE_SCOPES)")

        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")

    @classmethod
    def find_in_env(
        cls,
        env: Mapping = None,
        prefix: Optional[str] = None
    ) -> "MSALAuth":
        """
        find_in_env documentation.
        
        Args:
            env: Parameter.
            prefix: Parameter.
        
        Returns:
            The result.
        """

        if not env:
            env = os.environ
        prefix = prefix or "AZURE_"

        required = {
            key: env.get(prefix + key.upper())
            for key in (
                "client_id", "client_secret", "tenant_id", "scopes"
            )
        }

        if all(required.values()):
            scopes = required["scopes"].split(",") if required["scopes"] else None
            return MSALAuth(
                tenant_id=required["tenant_id"],
                client_id=required["client_id"],
                client_secret=required["client_secret"],
                scopes=scopes,
                authority=env.get(prefix + "AUTHORITY"),
            )

        return None

    def export_to(self, to: dict = os.environ):
        """
        export_to documentation.
        
        Args:
            to: Parameter.
        
        Returns:
            The result.
        """

        for key, value in (
            ("AZURE_CLIENT_ID", self.client_id),
            ("AZURE_CLIENT_SECRET", self.client_secret),
            ("AZURE_AUTHORITY", self.authority),
            ("AZURE_SCOPES", ",".join(self.scopes)),
        ):
            if value:
                to[key] = value

    @property
    def auth_app(self) -> ConfidentialClientApplication:
        """
        auth_app documentation.
        
        Args:
            None.
        
        Returns:
            The result.
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
        """
        expires_in documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return time.time() - self.expires_at

    @property
    def expires_at(self) -> float:
        """
        expires_at documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        self.refresh()

        return self._expires_at

    @property
    def expired(self) -> bool:
        """
        expired documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return not self._expires_at or time.time() >= self._expires_at

    def refresh(self, force: bool | None = None):
        """
        refresh documentation.
        
        Args:
            force: Parameter.
        
        Returns:
            The result.
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
        """Return access token."""
        self.refresh()
        return self._access_token

    @property
    def authorization(self) -> str:
        """Return authorization token."""
        return f"Bearer {self.access_token}"

    def requests_session(self, **kwargs):
        """
        requests_session documentation.
        
        Args:
            **kwargs: Parameter.
        
        Returns:
            The result.
        """

        return MSALSession(
            msal_auth=self,
            **kwargs
        )


class MSALSession(YGGSession):
    msal_auth: MSALAuth | None = None

    def __init__(
        self,
        msal_auth: Optional[MSALAuth] = None,
        *args,
        **kwargs: dict
    ):
        """
        __init__ documentation.
        
        Args:
            msal_auth: Parameter.
            *args: Parameter.
            **kwargs: Parameter.
        
        Returns:
            None.
        """

        super().__init__(*args, **kwargs)
        self.msal_auth = msal_auth


    def prepare_request(self, request):
        # called before sending; ensure header exists
        """
        prepare_request documentation.
        
        Args:
            request: Parameter.
        
        Returns:
            The result.
        """

        if self.msal_auth is not None:
            request.headers["Authorization"] = request.headers.get("Authorization", self.msal_auth.authorization)

        return super().prepare_request(request)
