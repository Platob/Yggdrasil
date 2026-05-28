"""MSAL-backed authentication helpers for requests sessions."""

from __future__ import annotations

import os
import threading
import time
from typing import ClassVar, Iterable, Optional, Tuple

try:
    from msal import ConfidentialClientApplication, PublicClientApplication
except ImportError:
    from yggdrasil.environ import PyEnv

    msal_mod = PyEnv.runtime_import_module(module_name="msal", pip_name="msal", install=True)
    ConfidentialClientApplication = msal_mod.ConfidentialClientApplication
    PublicClientApplication = msal_mod.PublicClientApplication

from yggdrasil.dataclasses.expiring import ExpiringDict

from .base import Authorization


__all__ = ["MSALAuth"]


# Cache key tuple: (tenant_id, client_id, client_secret, authority,
# scopes_tuple). Scopes are kept as a tuple so the key is hashable and
# scope order matters for cache identity (Azure normalizes order
# internally, but we'd rather over-share than under-share — two callers
# with the same scope set get one token).
_CacheKey = Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Tuple[str, ...]]


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


def _normalize_str(value: object) -> Optional[str]:
    return value.strip() if isinstance(value, str) else value  # type: ignore[return-value]


def _resolve_config(
    tenant_id: object,
    client_id: object,
    client_secret: object,
    authority: object,
    scopes: object,
) -> _CacheKey:
    """Normalize constructor args into the canonical cache key tuple.

    Mirrors :meth:`MSALAuth.__init__` so :meth:`__new__` and ``__init__``
    end up with the exact same (tenant_id, client_id, client_secret,
    authority, scopes_tuple) values. Args left as ``...`` fall back to
    the matching ``AZURE_*`` env var; explicit ``None`` stays ``None``.
    Authority is derived from ``tenant_id`` when not set so two callers
    passing only the tenant collapse to the same singleton.
    """
    if tenant_id is ...:
        tenant_id = os.environ.get("AZURE_TENANT_ID")
    if client_id is ...:
        client_id = os.environ.get("AZURE_CLIENT_ID")
    if client_secret is ...:
        client_secret = os.environ.get("AZURE_CLIENT_SECRET")
    if authority is ...:
        authority = os.environ.get("AZURE_AUTHORITY")
    if scopes is ...:
        scopes = os.environ.get("AZURE_SCOPES")

    tenant_id = _normalize_str(tenant_id)
    client_id = _normalize_str(client_id)
    client_secret = _normalize_str(client_secret)
    authority = _normalize_str(authority)
    scope_list = _parse_scopes(scopes)

    if not authority:
        if not tenant_id:
            raise ValueError("tenant_id is required when authority is not set.")
        authority = f"https://login.microsoftonline.com/{tenant_id}"

    return (
        tenant_id,
        client_id,
        client_secret,
        authority,
        tuple(scope_list) if scope_list else (),
    )


class MSALAuth(Authorization):
    """Configuration and token cache for MSAL client credential flows.

    Singleton-by-config: two callers building an :class:`MSALAuth` with
    the same ``(tenant_id, client_id, client_secret, authority, scopes)``
    receive the **same** instance — and therefore share one MSAL
    application, one access token, one in-flight refresh. The first
    ``refresh()`` pays the token round-trip; every subsequent caller on
    that config reuses the cached token until it expires. Pickle
    round-trips inside the same process collapse to the live singleton
    via :meth:`__getnewargs__`; cross-process unpickle rebuilds the MSAL
    handle (``_auth_app``) lazily on first use.
    """

    # Per-(cls, cache_key) singleton cache. Subclasses inherit the slot.
    # The ``cls`` component lets a subclass with different defaults stay
    # distinct from the base even for an identical config tuple.
    # ``ExpiringDict.get_or_set`` is lock-free and atomic under the
    # CPython GIL — no external mutex needed, and a TTL can be wired
    # in later (e.g. drop unused configs after N hours) without touching
    # this site. ``default_ttl=None`` keeps entries for the process
    # lifetime; per-token expiry lives inside the singleton itself.
    _INSTANCES: ClassVar[ExpiringDict[Tuple[type, _CacheKey], "MSALAuth"]] = ExpiringDict(
        default_ttl=None
    )

    # Live MSAL application handle and the per-instance refresh lock
    # aren't picklable — drop them on ``__getstate__`` and rebuild on
    # the receiver side. The access token *does* travel with the pickle
    # so cross-process consumers don't pay another token round-trip.
    _TRANSIENT_STATE_ATTRS: ClassVar[frozenset[str]] = frozenset({"_auth_app", "_refresh_lock"})

    def __new__(
        cls,
        tenant_id: Optional[str] = ...,  # type: ignore[assignment]
        client_id: Optional[str] = ...,  # type: ignore[assignment]
        client_secret: Optional[str] = ...,  # type: ignore[assignment]
        authority: Optional[str] = ...,  # type: ignore[assignment]
        scopes: object = ...,
        expiry_skew_seconds: int = 30,
    ) -> "MSALAuth":
        cache_key = _resolve_config(tenant_id, client_id, client_secret, authority, scopes)
        key = (cls, cache_key)
        # ExpiringDict.get_or_set is lock-free + GIL-atomic — two
        # callers racing the same config either both observe the
        # winner's instance, or both run the factory once and the
        # loser's fresh instance is silently discarded by the
        # check-after-build re-probe. No external mutex needed.
        return cls._INSTANCES.get_or_set(key, lambda: object.__new__(cls))

    def __init__(
        self,
        tenant_id: Optional[str] = ...,  # type: ignore[assignment]
        client_id: Optional[str] = ...,  # type: ignore[assignment]
        client_secret: Optional[str] = ...,  # type: ignore[assignment]
        authority: Optional[str] = ...,  # type: ignore[assignment]
        scopes: object = ...,
        expiry_skew_seconds: int = 30,
    ) -> None:
        # Singleton-cached instances are re-entered on every constructor
        # call (Python always invokes __init__ after __new__); skip the
        # second pass so the live MSAL app and any in-flight token stay
        # untouched.
        if getattr(self, "_initialized", False):
            return

        (
            self.tenant_id,
            self.client_id,
            self.client_secret,
            self.authority,
            scope_tuple,
        ) = _resolve_config(tenant_id, client_id, client_secret, authority, scopes)
        self.scopes: list[str] | None = list(scope_tuple) if scope_tuple else None
        self.expiry_skew_seconds = expiry_skew_seconds

        # Cache key is what we hash and compare on; keep it once so the
        # singleton table lookup, ``__hash__``, and ``__eq__`` all agree
        # even if a caller mutates ``self.scopes`` post-construction.
        self._cache_key: _CacheKey = (
            self.tenant_id,
            self.client_id,
            self.client_secret,
            self.authority,
            scope_tuple,
        )

        self._auth_app: ConfidentialClientApplication | PublicClientApplication | None = None
        self._access_token: Optional[str] = None
        self._expires_at: float | None = None
        # One refresh per singleton at a time — two callers racing into
        # the same expired token shouldn't both hit AAD.
        self._refresh_lock: threading.Lock = threading.Lock()
        self._initialized = True

    # --- identity ------------------------------------------------------

    def __hash__(self) -> int:
        return hash((type(self), self._cache_key))

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if type(self) is not type(other):
            return NotImplemented
        return self._cache_key == other._cache_key  # type: ignore[attr-defined]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(tenant_id={self.tenant_id!r}, "
            f"client_id={self.client_id!r}, scopes={self.scopes!r})"
        )

    # --- pickle safety -------------------------------------------------

    def __getnewargs__(self) -> tuple:
        # Route in-process unpickle through ``__new__`` so a pickle
        # round-trip on the same machine collapses to the live
        # singleton instead of cloning the MSAL app and token cache.
        return (
            self.tenant_id,
            self.client_id,
            self.client_secret,
            self.authority,
            list(self.scopes) if self.scopes else None,
            self.expiry_skew_seconds,
        )

    def __getstate__(self) -> dict:
        return {
            k: v for k, v in self.__dict__.items()
            if k not in self._TRANSIENT_STATE_ATTRS
        }

    def __setstate__(self, state: dict) -> None:
        # __new__ may have returned a live singleton — keep its MSAL app
        # and token cache intact rather than clobbering with the pickle.
        if getattr(self, "_initialized", False):
            return
        self.__dict__.update(state)
        self._auth_app = None
        self._refresh_lock = threading.Lock()
        self._initialized = True

    # --- minimal mapping sugar ----------------------------------------

    def __setitem__(self, key: str, value) -> None:
        setattr(self, key, value)

    def __getitem__(self, key: str):
        return getattr(self, key)

    # --- validation ----------------------------------------------------

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
            # PublicClientApplication does NOT accept client_credential/scopes in ctor.
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
        """Scopes as a space-joined string (or None)."""
        if self.scopes:
            return " ".join(self.scopes)
        return None

    @scope.setter
    def scope(self, value: object) -> None:
        """
        Accept None | list[str] | tuple[str,...] | set[str] | "a b" | "a,b".
        Normalizes + resets cached token (since token depends on scopes).

        Note: this mutates the live singleton in place. Callers that
        want a *separate* token cache for a different scope set should
        construct a new ``MSALAuth(..., scopes=...)`` — the
        ``_INSTANCES`` cache keys by scopes, so a different scope tuple
        gives a different singleton.
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
        """Acquire/refresh token for confidential client flow.

        Serialized per-instance: two callers sharing the singleton
        won't both hit AAD when the token expires — the second waits
        on the lock and re-checks the cache before deciding to mint.
        """
        if not (force or self.is_expired or not self._access_token):
            return self

        with self._refresh_lock:
            # Re-check inside the lock: a racer ahead of us may have
            # already minted a fresh token while we waited.
            if not force and not self.is_expired and self._access_token:
                return self

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
