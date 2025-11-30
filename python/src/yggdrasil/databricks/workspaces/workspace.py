import dataclasses
import hashlib
import json
import os
from abc import ABC
from dataclasses import dataclass
from typing import Mapping, Optional

from ...libs import require_pyspark
from ...libs.databricks import require_databricks_sdk, databricks_sdk, databricks
from ...requests import MSALAuth

try:
    from pyspark.sql import SparkSession
except ImportError:
    SparkSession = None

try:
    from delta.tables import DeltaTable
except ImportError:
    DeltaTable = None



@dataclass
class DBXWorkspace:
    host: str = None
    token: str = None
    msal_auth: Optional[MSALAuth] = None
    auth_type: Optional[str] = None
    product: Optional[str] = None
    product_version: Optional[str] = None

    _sdk: "databricks.sdk.WorkspaceClient" = dataclasses.field(init=False, default=None)
    _was_connected: bool = dataclasses.field(init=False, default=False)

    @classmethod
    def find_in_env(
        cls,
        env: Mapping = None,
        prefix: Optional[str] = None
    ):
        env = env or os.environ
        prefix = prefix or "DATABRICKS_"
        msal_auth = MSALAuth.find_in_env(env=env, prefix="AZURE_")
        options = {
            k: env.get(prefix + k.upper())
            for k in (
                "token", "host", "auth_type",
                "product", "product_version"
            )
            if env.get(prefix + k.upper())
        }

        return cls(
            msal_auth=msal_auth,
            **options,
        )

    def __enter__(self):
        self._was_connected = self._sdk is not None
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._was_connected:
            self.close()

    def sdk(self):
        self.connect()
        return self._sdk

    @require_databricks_sdk
    def connect(
        self,
        reset: bool = False
    ):
        if reset or self._sdk is None:
            auth_type = self.auth_type
            if auth_type == "u2m-oauth":
                auth_type = "external-browser"
            if auth_type == "external-browser":
                self._configure_external_browser_cache()
            self._sdk = databricks_sdk.WorkspaceClient(
                host=self.host,
                token=self.token,
                auth_type=auth_type,
                product=self.product,
                product_version=self.product_version or "0.0.0"
            )
        return self._sdk

    def close(self):
        if self._sdk:
            self._sdk = None
            self._was_connected = False

    def _configure_external_browser_cache(self):
        oauth = getattr(databricks_sdk, "oauth", None)
        if oauth is None:
            return

        token_cache_cls = getattr(oauth, "TokenCache", None)
        if token_cache_cls is None:
            session_credentials_cls = getattr(oauth, "SessionCredentials", None)
            token_cls = getattr(oauth, "Token", None)
            if not session_credentials_cls or not token_cls:
                return

            class TokenCache:
                BASE_PATH = "~/.ygg/databricks/auth/cache/external-browser"

                def __init__(
                    self,
                    host: str,
                    oidc_endpoints,
                    client_id: str,
                    redirect_url: Optional[str] = None,
                    client_secret: Optional[str] = None,
                    scopes: Optional[list[str]] = None,
                ) -> None:
                    self._host = host
                    self._client_id = client_id
                    self._oidc_endpoints = oidc_endpoints
                    self._redirect_url = redirect_url
                    self._client_secret = client_secret
                    self._scopes = scopes or []

                @property
                def filename(self) -> str:
                    hash_value = hashlib.sha256()
                    for chunk in [self._host, self._client_id, ",".join(self._scopes)]:
                        hash_value.update(chunk.encode("utf-8"))
                    return os.path.expanduser(
                        os.path.join(
                            self.__class__.BASE_PATH,
                            hash_value.hexdigest() + ".json",
                        )
                    )

                def load(self):
                    if not os.path.exists(self.filename):
                        return None
                    try:
                        with open(self.filename, "r") as f:
                            raw = json.load(f)
                            return session_credentials_cls.from_dict(
                                raw,
                                token_endpoint=self._oidc_endpoints.token_endpoint,
                                client_id=self._client_id,
                                client_secret=self._client_secret,
                                redirect_url=self._redirect_url,
                            )
                    except Exception:
                        return None

                def save(self, credentials):
                    os.makedirs(os.path.dirname(self.filename), exist_ok=True)
                    with open(self.filename, "w") as f:
                        json.dump(credentials.as_dict(), f)
                    os.chmod(self.filename, 0o600)

            oauth.TokenCache = TokenCache
            token_cache_cls = TokenCache

        token_cache_cls.BASE_PATH = os.path.expanduser(
            "~/.ygg/databricks/auth/cache/external-browser"
        )

    @require_pyspark(active_session=True)
    def spark_session(self):
        return SparkSession.getActiveSession()

    def sql(
        self
    ):
        from ..sql import DBXSQL

        return DBXSQL(workspace=self)


@dataclass
class DBXWorkspaceObject(ABC):
    workspace: DBXWorkspace

    def __enter__(self):
        self.workspace.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.workspace.__exit__(exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb)

    @staticmethod
    def temp_volume_folder(suffix: Optional[str] = None):
        base = "/Shared/.ygg/tmp/"

        if not suffix:
            return base

        if suffix.startswith("/"):
            suffix = suffix[1:]

        return base + suffix

    def fs(self):
        return self.workspace.sdk().dbutils.fs
