import dataclasses
import os
import platform
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Union

from ...libs import require_pyspark
from ...libs.databrickslib import require_databricks_sdk, databricks_sdk, databricks
from ...requests import MSALAuth

try:
    from pyspark.sql import SparkSession
except ImportError:
    SparkSession = None

try:
    from delta.tables import DeltaTable
except ImportError:
    DeltaTable = None


class DBXAuthType(Enum):
    external_browser = "external-browser"


@dataclass
class DBXWorkspace:
    host: str = None
    token: str = None
    msal_auth: Optional[MSALAuth] = None
    auth_type: Optional[Union[str, DBXAuthType]] = None
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
            if isinstance(auth_type, DBXAuthType):
                auth_type = auth_type.value
            elif self.token is None and auth_type is None:
                if platform.system() == "Windows":
                    # default to external browser on Windows
                    auth_type = DBXAuthType.external_browser.value

            self._sdk = databricks_sdk.WorkspaceClient(
                host=self.host,
                token=self.token,
                auth_type=auth_type,
                product=self.product,
                product_version=self.product_version or "0.0.0"
            )

            if not self.host:
                self.host = self._sdk.config.host

            if not self.auth_type:
                self.auth_type = self._sdk.config.auth_type
                
        return self._sdk

    def close(self):
        if self._sdk:
            self._sdk = None
            self._was_connected = False

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
