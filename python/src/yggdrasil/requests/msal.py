"""MSAL-backed authentication helpers for requests sessions."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Iterable

from .session import YGGSession

try:
    from msal import ConfidentialClientApplication, PublicClientApplication
except ImportError:
    # Local helper that can pip-install at runtime (if that's your org's vibe)
    from ..pyutils.pyenv import PyEnv

    msal_mod = PyEnv.runtime_import_module(module_name="msal", pip_name="msal", install=True)
    ConfidentialClientApplication = msal_mod.ConfidentialClientApplication
    PublicClientApplication = msal_mod.PublicClientApplication


__all__ = ["MSALSession", "MSALAuth"]


def _parse_scopes(value: object) -> list[str] | None:
    """
    Accept:
      - None
      - list[str]
      - "scope1 scope2"
      - "scope1,scope2"
    """
    if value is None:
        return None
    if isinstance(value, list):
        scopes = [str(x).strip() for x in value if str(x).strip()]
        return scopes or None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        sep = "," if "," in s else " "
        scopes = [p.strip() for p in s.split(sep) if p.strip()]
        return scopes or None
    raise TypeError(f"scopes must be None, list[str], or str; got {type(value)!r}")


@dataclass
class MSALAuth:
    """Configuration and token cache for MSAL client credential flows."""

    tenant_id: Optional[str] = field(default_factory=lambda: os.environ.get("AZURE_TENANT_ID"))
    client_id: Optional[str] = field(default_factory=lambda: os.environ.get("AZURE_CLIENT_ID"))
    client_secret: Optional[str] = field(default_factory=lambda: os.environ.get("AZURE_CLIENT_SECRET"))
    authority: Optional[str] = field(default_factory=lambda: os.environ.get("AZURE_AUTHORITY"))
    scopes: list[str] | None = field(default_factory=lambda: _parse_scopes(os.environ.get("AZURE_SCOPES")))

    # Refresh a bit early to avoid edge-of-expiry races.
    expiry_skew_seconds: int = field(default=30, repr=False, compare=False)

    _auth_app: ConfidentialClientApplication | PublicClientApplication | None = field(
        default=None, repr=False, compare=False, hash=False
    )
    _expires_at: float | None = field(default=None, repr=False, compare=False, hash=False)
    _access_token: Optional[str] = field(default=None, repr=False, compare=False, hash=False)

    # --- pickle safety -------------------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_auth_app"] = None  # MSAL client isn't picklable
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._auth_app = None

    # --- minimal mapping sugar ----------------------------------------

    def __setitem__(self, key: str, value) -> None:
        setattr(self, key, value)

    def __getitem__(self, key: str):
        return getattr(self, key)

    # --- init / validation --------------------------------------------

    def __post_init__(self) -> None:
        # Normalize string fields
        self.tenant_id = self.tenant_id.strip() if isinstance(self.tenant_id, str) else self.tenant_id
        self.client_id = self.client_id.strip() if isinstance(self.client_id, str) else self.client_id
        self.client_secret = self.client_secret.strip() if isinstance(self.client_secret, str) else self.client_secret
        self.authority = self.authority.strip() if isinstance(self.authority, str) else self.authority

        # Accept legacy: scopes accidentally passed as string
        self.scopes = _parse_scopes(self.scopes)

        if not self.authority:
            if not self.tenant_id:
                raise ValueError("tenant_id is required when authority is not set.")
            self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"

    def _ensure_confidential_flow_ready(self) -> None:
        if not self.client_id:
            raise ValueError("client_id is required. Set AZURE_CLIENT_ID.")
        if not self.authority:
            raise ValueError("authority is required. Set AZURE_AUTHORITY or AZURE_TENANT_ID.")
        if not self.scopes:
            raise ValueError(
                "scopes are required for acquire_token_for_client. "
                "Set AZURE_SCOPES (e.g. 'api://<app-id>/.default' or '<resource>/.default')."
            )

    # --- msal app ------------------------------------------------------

    @property
    def auth_app(self) -> ConfidentialClientApplication | PublicClientApplication:
        """
        Lazily create MSAL client.

        - If client_secret is set: ConfidentialClientApplication (app-to-app).
        - Else: PublicClientApplication (interactive/device code/etc).
        """
        if self._auth_app is not None:
            return self._auth_app

        if not self.client_id:
            raise ValueError("client_id is required. Set AZURE_CLIENT_ID.")
        if not self.authority:
            raise ValueError("authority is required. Set AZURE_AUTHORITY or AZURE_TENANT_ID.")

        if self.client_secret:
            self._auth_app = ConfidentialClientApplication(
                client_id=self.client_id,
                client_credential=self.client_secret,
                authority=self.authority,
            )
        else:
            # Note: PublicClientApplication does NOT accept client_credential/scopes in ctor.
            self._auth_app = PublicClientApplication(
                client_id=self.client_id,
                authority=self.authority,
            )

        return self._auth_app

    # --- token lifecycle ----------------------------------------------

    @property
    def is_expired(self) -> bool:
        if not self._expires_at:
            return True
        return (time.time() + self.expiry_skew_seconds) >= self._expires_at

    @property
    def seconds_to_expiry(self) -> float:
        """Positive means still valid, negative means expired."""
        if not self._expires_at:
            return float("-inf")
        return self._expires_at - time.time()

    @property
    def scope(self) -> str | None:
        """Scopes as a normalized list[str] (or None)."""
        if self.scopes:
            return " ".join(self.scopes)
        return None

    @scope.setter
    def scope(self, value: object) -> None:
        """
        Accept None | list[str] | tuple[str,...] | set[str] | "a b" | "a,b".
        Normalizes + resets cached token (since token depends on scopes).
        """
        if isinstance(value, (tuple, set)):
            value = list(value)
        # also accept any iterable of strings, but avoid treating str as iterable here
        if value is not None and not isinstance(value, (str, list)):
            if isinstance(value, Iterable):
                value = list(value)  # type: ignore[arg-type]

        self.scopes = _parse_scopes(value)

        # scopes changed => cached token may be wrong
        self._access_token = None
        self._expires_at = None

    def refresh(self, force: bool = False) -> "MSALAuth":
        """Acquire/refresh token for confidential client flow."""
        if force or self.is_expired or not self._access_token:
            self._ensure_confidential_flow_ready()
            auth_app = self.auth_app

            if self.client_secret:
                result = auth_app.acquire_token_for_client(scopes=self.scopes)
            else:
                result = auth_app.acquire_token_interactive(scopes=self.scopes)

            token = result.get("access_token")
            if not token:
                raise RuntimeError(
                    f"Failed to acquire token: {result.get('error_description') or result}"
                )

            now = time.time()
            expires_in = int(result.get("expires_in", 3600))
            self._access_token = token
            self._expires_at = now + expires_in

        return self

    @property
    def access_token(self) -> str:
        self.refresh()
        # refresh() guarantees it
        return self._access_token  # type: ignore[return-value]

    @property
    def authorization(self) -> str:
        return f"Bearer {self.access_token}"

    def requests_session(self, **kwargs) -> "MSALSession":
        return MSALSession(msal_auth=self, **kwargs)


class MSALSession(YGGSession):
    """YGGSession subclass that injects MSAL authorization headers."""

    def __init__(self, msal_auth: Optional[MSALAuth] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.msal_auth = msal_auth

    def prepare_request(self, request):
        if self.msal_auth is not None and "Authorization" not in request.headers:
            request.headers["Authorization"] = self.msal_auth.authorization
        return super().prepare_request(request)
