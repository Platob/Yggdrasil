import logging
import os
import re
import time
from abc import ABC
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from threading import RLock
from typing import Optional, Any, Type, ClassVar, TypeVar, TYPE_CHECKING, Callable, Union

from databricks.sdk import AccountClient as DAC, WorkspaceClient as DWC
from databricks.sdk.client_types import ClientType
from databricks.sdk.config import Config

from yggdrasil.environ import UserInfo
from yggdrasil.io.url import URL, URLResource, url_resource_class
from ..concurrent.threading import Job
from ..dataclasses import WaitingConfigArg, WaitingConfig, ExpiringDict
from ..dataclasses.dataclass import serialize_dataclass_state, restore_dataclass_state
from ..io import BytesIO, MimeType

if TYPE_CHECKING:
    from .iam import IAM
    from .sql.engine import SQLEngine
    from .sql.warehouse import SQLWarehouse
    from .compute.service import Compute
    from .secrets.secret import Secrets
    from .workspaces import Workspaces, Workspace, DatabricksPath

__all__ = [
    "DatabricksClient",
    "DatabricksService",
    "DatabricksResource"
]

LOGGER = logging.getLogger(__name__)
CURRENT_BASE_CLIENT: Optional["DatabricksClient"] = None
CURRENT_BASE_CLIENT_LOCK: RLock = RLock()

T = TypeVar("T")
TC = TypeVar("TC", bound="DatabricksClient")
TS = TypeVar("TS", bound="DatabricksService")

_ALLOWED = re.compile(r"^[\d \w\+\-=\.:/@]*$") # noqa
_SANITIZE = re.compile(r"[^\d \w\+\-=\.:/@]+") # noqa



def getenv(name: str) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else None


def getenv_factory(name: str):
    def factory():
        return getenv(name)
    return factory


def env_field(
    name: str,
    *,
    repr_: bool = False,
    compare: bool = True,
    hash_: bool = True
):
    return field(
        default_factory=getenv_factory(name),
        repr=repr_,
        compare=compare,
        hash=hash_
    )


@url_resource_class
@dataclass
class DatabricksClient(URLResource):
    """
    Thin wrapper around databricks.sdk.config.Config.

    Public state is intentionally minimal:
      - config: Config

    Extra wrapper-only metadata is kept separately.
    """

    # Databricks / generic
    host: Optional[str] = env_field("DATABRICKS_HOST", repr_=True)
    account_id: Optional[str] = env_field("DATABRICKS_ACCOUNT_ID", repr_=False)
    workspace_id: Optional[str] = env_field("DATABRICKS_WORKSPACE_ID", repr_=False)
    token: Optional[str] = env_field("DATABRICKS_TOKEN", repr_=False)
    client_id: Optional[str] = env_field("DATABRICKS_CLIENT_ID", repr_=False)
    client_secret: Optional[str] = env_field("DATABRICKS_CLIENT_SECRET", repr_=False)
    token_audience: Optional[str] = env_field("DATABRICKS_TOKEN_AUDIENCE", repr_=False)

    # Azure
    azure_workspace_resource_id: Optional[str] = env_field("ARM_RESOURCE_ID", repr_=False)
    azure_use_msi: Optional[bool] = field(default=None, repr=False)
    azure_client_secret: Optional[str] = env_field("ARM_CLIENT_SECRET", repr_=False)
    azure_client_id: Optional[str] = env_field("ARM_CLIENT_ID", repr_=False)
    azure_tenant_id: Optional[str] = env_field("ARM_TENANT_ID", repr_=False)
    azure_environment: Optional[str] = env_field("ARM_ENVIRONMENT", repr_=False)

    # GCP
    google_credentials: Optional[str] = env_field("GOOGLE_CREDENTIALS", repr_=False)
    google_service_account: Optional[str] = env_field("DATABRICKS_GOOGLE_SERVICE_ACCOUNT", repr_=False)

    # Config profile
    profile: Optional[str] = env_field("DATABRICKS_CONFIG_PROFILE", repr_=False, compare=False, hash_=False)
    config_file: Optional[str] = env_field("DATABRICKS_CONFIG_FILE", repr_=False, compare=False, hash_=False)

    # HTTP / client behavior
    auth_type: Optional[str] = None
    http_timeout_seconds: Optional[int] = field(default=None, repr=False)
    retry_timeout_seconds: Optional[int] = field(default=None, repr=False)
    debug_truncate_bytes: Optional[int] = field(default=None, repr=False)
    debug_headers: Optional[bool] = field(default=None, repr=False)
    rate_limit: Optional[int] = field(default=None, repr=False)

    # Extras
    product: Optional[str] = field(default=None, repr=False)
    product_version: Optional[str] = field(default=None, repr=False)
    custom_tags: Optional[dict] = field(default=None, repr=False)

    # Internal cached SDK clients
    _was_connected: bool = field(default=False, init=False, repr=False, compare=False, hash=False)
    _workspace_config: Optional[Config] = field(default=None, init=False, repr=False, compare=False, hash=False)
    _workspace_client: Optional[DWC] = field(default=None, init=False, repr=False, compare=False, hash=False)
    _account_config: Optional[Config] = field(default=None, init=False, repr=False, compare=False, hash=False)
    _account_client: Optional[DAC] = field(default=None, init=False, repr=False, compare=False, hash=False)

    _cache: ClassVar[dict[str, "DatabricksClient"]] = {}

    def __post_init__(self):
        if self.account_id:
            if not self.host:
                object.__setattr__(self, "host", f"https://accounts.cloud.databricks.com")

        if self.host:
            normalized = self.host.split("://")[-1].split("/")[0]
            object.__setattr__(self, "host", f"https://{normalized}")

    def __getstate__(self):
        state = serialize_dataclass_state(self)

        auth_type = state.get("auth_type")
        if auth_type in {"runtime", "databricks-cli", "external-browser"}:
            del state["auth_type"]

        return state

    def __setstate__(self, state):
        restore_dataclass_state(self, state)
        self.__post_init__()

    # -------------------------------------------------------------------------
    # URLResource
    # -------------------------------------------------------------------------

    @classmethod
    def url_scheme(cls) -> str:
        return "dbks"

    @property
    def base_url(self):
        return URL.parse_str(self.host)

    def to_url(self, scheme: str | None = None) -> URL:
        query: dict[str, Any] = {}
        keys = [
            f.name
            for f in fields(self)
            if f.init and f.name not in (
                "host", "token", "client_id", "client_secret"
            )
        ]

        for key in keys:
            value = getattr(self, key)
            if value is not None:
                query[key] = value

        if self.token:
            user, password = None, self.token
        elif self.client_id and self.client_secret:
            user, password = self.client_id, self.client_secret
        else:
            user, password = None, None

        return (
            self.base_url
            .with_scheme(scheme or self.url_scheme())
            .with_query_items(query)
            .with_user_password(user=user, password=password)
        )

    @classmethod
    def parse(
        cls,
        obj: Any,
    ):
        if isinstance(obj, cls):
            return obj
        elif isinstance(obj, URL):
            return cls.from_parsed_url(obj)
        elif isinstance(obj, str):
            url = URL.parse_str(obj, default_scheme="https")
            return cls.from_parsed_url(url)
        elif isinstance(obj, dict):
            return cls(**obj)
        else:
            raise ValueError(f"Cannot parse {cls.__name__} from object of type {type(obj)}: {obj!r}")

    @classmethod
    def from_parsed_url(
        cls: Type[TC],
        url: URL,
        *,
        safe: bool = False
    ) -> TC:
        if not url.path:
            raise ValueError(f"Invalid path for {cls.__name__}: {url!r}")

        kwargs = {}
        for key, value in url.query_items():
            kwargs[key] = value

        host = kwargs.pop("host", None) or (f"https://{url.host}/" if url.host else None)

        if not host:
            raise ValueError(f"Host is required for {cls.__name__} URL: {url!r}")

        user, password = url.user, url.password

        if user and password:
            kwargs["client_id"] = user
            kwargs["client_secret"] = password
        elif password:
            kwargs["token"] = password

        return cls(host=host, **kwargs)

    # -------------------------------------------------------------------------
    # Singleton helpers
    # -------------------------------------------------------------------------

    @classmethod
    def current(cls, reset: bool = False, **overrides: Any) -> "DatabricksClient":
        global CURRENT_BASE_CLIENT

        if reset or CURRENT_BASE_CLIENT is None:
            with CURRENT_BASE_CLIENT_LOCK:
                if reset or CURRENT_BASE_CLIENT is None:
                    client = cls(**overrides)
                    CURRENT_BASE_CLIENT = client

        return CURRENT_BASE_CLIENT

    @classmethod
    def set_current(cls, workspace: Optional["DatabricksClient"]) -> None:
        global CURRENT_BASE_CLIENT
        with CURRENT_BASE_CLIENT_LOCK:
            CURRENT_BASE_CLIENT = workspace

    # -------------------------------------------------------------------------
    # Repr / context manager
    # -------------------------------------------------------------------------

    def __repr__(self):
        return f"{self.__class__.__name__}(host={self.host!r}, auth_type={self.auth_type!r})"

    def __str__(self):
        return self.__repr__()

    def __enter__(self) -> TC:
        object.__setattr__(self, "_was_connected", self.connected)
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self._was_connected:
            self.close()

    def __del__(self):
        self.close()

    # -------------------------------------------------------------------------
    # Connection lifecycle
    # -------------------------------------------------------------------------
    @property
    def config(self):
        if self._workspace_config is None or self._account_config is None:
            config = self.make_config()

            if config.client_type == ClientType.WORKSPACE:
                object.__setattr__(self, "_workspace_config", config)
            elif config.client_type == ClientType.ACCOUNT:
                object.__setattr__(self, "_account_config", config)
            else:
                raise ValueError(f"Unexpected client_type in config: {config.client_type}")
        return self._workspace_config or self._account_config

    @property
    def workspace_config(self) -> Config:
        if self._workspace_config is None:
            config = self.make_config(client_type=ClientType.WORKSPACE)
            object.__setattr__(self, "_workspace_config", config)
        return self._workspace_config

    @property
    def account_config(self) -> Config:
        if self._account_config is None:
            config = self.make_config(client_type=ClientType.ACCOUNT)

            object.__setattr__(self, "_account_config", config)
        return self._account_config

    @property
    def default_client_type(self) -> ClientType:
        return self.config.client_type

    def _make_base_config(
        self,
        client_type: Optional[ClientType] = None
    ):
        if client_type == ClientType.ACCOUNT:
            host = "https://accounts.cloud.databricks.com"

            if not self.account_id:
                self.account_id = self.get_account_id()
        else:
            host = self.host

        config = Config(
            host=host,
            token=self.token,
            client_id=self.client_id,
            client_secret=self.client_secret,
            account_id=self.account_id,
            token_audience=self.token_audience,
            azure_workspace_resource_id=self.azure_workspace_resource_id,
            azure_use_msi=self.azure_use_msi,
            azure_client_secret=self.azure_client_secret,
            azure_client_id=self.azure_client_id,
            azure_tenant_id=self.azure_tenant_id,
            azure_environment=self.azure_environment,
            google_credentials=self.google_credentials,
            google_service_account=self.google_service_account,
            profile=self.profile,
            config_file=self.config_file,
            auth_type=self.auth_type,
            http_timeout_seconds=self.http_timeout_seconds,
            retry_timeout_seconds=self.retry_timeout_seconds,
            debug_truncate_bytes=self.debug_truncate_bytes,
            debug_headers=self.debug_headers,
            rate_limit=self.rate_limit,
            product=self.product,
            product_version=self.product_version,
        )

        if client_type is not None:
            if config.client_type != client_type:
                raise ValueError(f"Config client_type {config.client_type} does not match expected {client_type}")

        return config

    def make_config(
        self,
        client_type: Optional[ClientType] = None
    ):
        try:
            config = self._make_base_config(client_type=client_type)
        except ValueError:
            if self.auth_type is None:
                object.__setattr__(
                    self, "auth_type",
                    "runtime" if self.is_in_databricks_environment() else "external-browser"
                )
                config = self._make_base_config(client_type=client_type)
            else:
                raise

        for key in (
            "auth_type",
            "token", "client_id", "client_secret", "account_id", "token_audience",
            "azure_workspace_resource_id", "azure_use_msi", "azure_client_secret",
            "azure_client_id", "azure_tenant_id", "azure_environment",
            "google_credentials", "google_service_account",
            "http_timeout_seconds", "retry_timeout_seconds", "debug_truncate_bytes",
            "debug_headers", "rate_limit"
        ):
            value = getattr(config, key, None)
            if value is not None:
                object.__setattr__(self, key, value)
                
        return config

    @property
    def local_config_folder(self):
        return Path.home() / ".config" / "databricks-sdk-py"

    @property
    def connected(self) -> bool:
        return self._workspace_client is not None or self._account_client is not None

    def connect(self, *, reset: bool = False) -> "DatabricksClient":
        if reset:
            self.close()

        config = self.config

        if config.client_type == ClientType.ACCOUNT:
            self.account_client()
        else:
            self.workspace_client()

        return self

    def close(self) -> None:
        for key in ("_workspace_client", "_account_client"):
            object.__setattr__(self, key, None)

        object.__setattr__(self, "_was_connected", False)

    def workspace_client(self) -> DWC:
        if self._workspace_client is None:
            object.__setattr__(self, "_workspace_client", DWC(config=self.workspace_config))
        return self._workspace_client

    def account_client(self) -> DAC:
        if self._account_client is None:
            object.__setattr__(self, "_account_client", DAC(config=self.account_config))
        return self._account_client

    def get_workspace_id(self) -> int:
        if self.workspace_id:
            return self.workspace_id

        self.workspace_id = int(self.workspace_client().get_workspace_id())
        return self.workspace_id

    def get_account_id(self) -> str:
        from yggdrasil.polars.lib import polars as pl

        if self.account_id:
            return self.account_id

        local_cache = self.local_config_folder / "workspaces_latest.parquet"
        buff = BytesIO(local_cache, copy=False)
        workspace_id = self.get_workspace_id()

        if local_cache.exists() and local_cache.stat().st_size > 0:
            existing_data = buff.media_io(MimeType.PARQUET).read_polars_frame()
            filtered = existing_data.filter(pl.col("workspace_id") == workspace_id)

            for record in filtered.to_dicts():
                self.set_account_id(record.get("account_id"))
                return self.account_id
        else:
            existing_data = None

        new_data = self.sql.execute(
            "select distinct account_id, cast(workspace_id as bigint) as workspace_id, workspace_url"
            " from system.access.workspaces_latest"
            " where status = 'RUNNING'"
        ).to_polars(stream=False)

        if existing_data is not None:
            new_data = pl.concat([existing_data, new_data]).unique()

        buff.truncate()
        buff.seek(0)
        buff.media_io(MimeType.PARQUET).write_arrow_table(new_data.to_arrow())
        buff.close()

        filtered = new_data.filter(pl.col("workspace_id") == workspace_id)

        for record in filtered.to_dicts():
            self.set_account_id(record.get("account_id"))
            return self.account_id

        raise ValueError(f"Could not find account_id for workspace_id {workspace_id!r}")

    def set_account_id(self, account_id: str) -> "DatabricksClient":
        if account_id:
            object.__setattr__(self, "account_id", account_id)

            if self._workspace_config is not None:
                object.__setattr__(self._workspace_config, "account_id", account_id)

            if self._account_client is not None:
                self._account_client.config.account_id = account_id

        return self

    @staticmethod
    def is_in_databricks_environment() -> bool:
        return getenv("DATABRICKS_RUNTIME_VERSION") is not None

    def default_tags(self, update: bool = True):
        """Return default resource tags for Databricks assets.

        Returns:
            A dict of default tags.
        """
        if update:
            base = dict()
        else:
            userinfo = UserInfo.current()

            base = {
                k: self.safe_tag_value(v)
                for k, v in (
                    ("Product", self.product),
                    ("ProductVersion", self.product_version),
                    ("UserMail", userinfo.email),
                    ("UserHost", userinfo.hostname),
                    ("UserURL", userinfo.url.to_string()),
                    ("GitURL", userinfo.git_url.to_string()),
                )
                if v
            }

        if self.custom_tags:
            base.update(self.custom_tags)

        return base

    @staticmethod
    def safe_tag_value(value: str, *, repl: str = "-") -> str:
        """
        Sanitize a tag string to match: ^[\\d \\w\\+\\-=:.:/@]*$
        Replaces any illegal characters with `repl` and collapses repeats.
        """
        if value is None:
            return ""

        s = str(value).strip()

        # fast path
        if _ALLOWED.fullmatch(s):
            return s

        # replace illegal chars (e.g. '#', '?', '&', '%') with repl
        s = _SANITIZE.sub(repl, s)

        # collapse repeated repl and trim edges
        if repl:
            s = re.sub(re.escape(repl) + r"{2,}", repl, s).strip(repl + " ")

        return s

    # Path

    def dbfs_path(
        self,
        parts: Union[list[str], str],
        temporary: bool = False
    ):
        """Create a DatabricksPath in this workspace.

        Args:
            parts: Path parts or string to parse.
            temporary: Temporary path

        Returns:
            A DatabricksPath instance.
        """
        from .workspaces.path import DatabricksPath

        return DatabricksPath.parse(
            obj=parts,
            client=self,
            temporary=temporary
        )

    @staticmethod
    def _base_tmp_path(
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        volume_name: Optional[str] = None,
    ) -> str:
        if catalog_name and schema_name:
            base_path = "/Volumes/%s/%s/%s" % (
                catalog_name, schema_name, volume_name or "tmp"
            )
        else:
            base_path = "/Workspace/Shared/.ygg/tmp"

        return base_path

    def tmp_path(
        self,
        suffix: Optional[str] = None,
        extension: Optional[str] = None,
        max_lifetime: Optional[float] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        volume_name: Optional[str] = None,
        base_path: Optional[str] = None,
    ) -> "DatabricksPath":
        """
        Shared cache base under Volumes for the current user.

        Args:
            suffix: Optional suffix
            extension: Optional extension suffix to append.
            max_lifetime: Max lifetime of temporary path
            catalog_name: Unity catalog name for volume path
            schema_name: Unity schema name for volume path
            volume_name: Unity volume name for volume path
            base_path: Base temporary path

        Returns:
            A DatabricksPath pointing at the shared cache location.
        """
        start = int(time.time())
        max_lifetime = int(max_lifetime or 48 * 3600)
        end = max(0, int(start + max_lifetime))

        base_path = base_path or self._base_tmp_path(
            catalog_name=catalog_name,
            schema_name=schema_name,
            volume_name=volume_name
        )

        rnd = os.urandom(4).hex()
        temp_path = f"tmp-{start}-{end}-{rnd}"

        if suffix:
            temp_path += suffix

        if extension:
            temp_path += ".%s" % extension

        self.clean_tmp_folder(
            raise_error=False,
            wait=False,
            base_path=base_path
        )

        return self.dbfs_path(f"{base_path}/{temp_path}")

    def clean_tmp_folder(
        self,
        raise_error: bool = True,
        wait: WaitingConfigArg = True,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        volume_name: Optional[str] = None,
        base_path: Optional[str] = None,
    ):
        wait = WaitingConfig.check_arg(wait)

        base_path = base_path or self._base_tmp_path(
            catalog_name=catalog_name,
            schema_name=schema_name,
            volume_name=volume_name
        )

        if is_checked_tmp_path(
            host=self.base_url.to_string(),
            base_path=base_path
        ):
            return self

        if wait.timeout:
            base_path = self.dbfs_path(base_path)

            LOGGER.debug(
                "Cleaning temp path %s",
                base_path
            )

            for path in base_path.ls(recursive=False, allow_not_found=True):
                if path.name.startswith("tmp"):
                    parts = path.name.split("-")

                    if len(parts) > 2 and parts[0] == "tmp" and parts[1].isdigit() and parts[2].isdigit():
                        end = int(parts[2])

                        if end and time.time() > end:
                            path.remove(recursive=True)

            LOGGER.info(
                "Cleaned temp path %s",
                base_path
            )
        else:
            (
                Job
                .make(self.clean_tmp_folder, raise_error=raise_error, base_path=base_path)
                .fire_and_forget()
            )

        return self

    # Sub services with optional caching for default workspace

    @staticmethod
    def lazy_property(
        self,
        *,
        cache_attr: str,
        factory: Callable[[], T],
        use_cache: bool,
    ) -> T:
        if use_cache:
            cached = getattr(self, cache_attr, None)
            if cached is not None:
                return cached

            created = factory()
            object.__setattr__(self, cache_attr, created)
            return created

        return factory()

    @property
    def workspaces(self) -> "Workspaces":
        from .workspaces.service import Workspaces

        return Workspaces(client=self)

    @property
    def workspace(self) -> "Workspace":
        from .workspaces import Workspace

        return self.lazy_property(
            self,
            cache_attr="_workspace",
            factory=lambda: Workspace(
                **{
                    f.name: getattr(self, f.name)
                    for f in fields(self)
                    if f.init
                }
            ),
            use_cache=True,
        )

    @property
    def sql(self) -> "SQLEngine":
        from .sql.engine import SQLEngine

        return self.lazy_property(
            self,
            cache_attr="_sql",
            factory=lambda: SQLEngine(client=self),
            use_cache=True,
        )

    @property
    def warehouses(self) -> "SQLWarehouse":
        from .sql.warehouse import SQLWarehouse

        return self.lazy_property(
            self,
            cache_attr="_warehouses",
            factory=lambda: SQLWarehouse(client=self),
            use_cache=True,
        )

    @property
    def compute(self) -> "Compute":
        """Default cluster helper for this client."""
        from .compute.service import Compute

        return self.lazy_property(
            self,
            cache_attr="_compute",
            factory=lambda: Compute(client=self),
            use_cache=True,
        )

    @property
    def secrets(self) -> "Secrets":
        """Default secrets helper for this client."""
        from .secrets.secret import Secrets

        return self.lazy_property(
            self,
            cache_attr="_secrets",
            factory=lambda: Secrets(client=self),
            use_cache=True,
        )

    @property
    def iam(self) -> "IAM":
        from .iam import IAM

        return self.lazy_property(
            self,
            cache_attr="_iam",
            factory=lambda: IAM(client=self),
            use_cache=True,
        )


DATABRICKS_CLIENT_INIT_NAMES = frozenset(f.name for f in fields(DatabricksClient) if f.init)
CHECKED_TMP_WORKSPACES: ExpiringDict[str, set[str]] = ExpiringDict()


def is_checked_tmp_path(
    host: str,
    base_path: str
):
    existing = CHECKED_TMP_WORKSPACES.get(host)

    if existing is None:
        CHECKED_TMP_WORKSPACES[host] = set(base_path)

        return False

    if base_path in existing:
        return True

    existing.add(base_path)

    return False

@dataclass(frozen=True)
class DatabricksService(ABC):
    client: DatabricksClient = field(
        default_factory=DatabricksClient.current,
        repr=False, compare=False, hash=False
    )

    _current: ClassVar[Optional["DatabricksService"]] = None

    def __post_init__(self):
        pass

    def __getstate__(self):
        return serialize_dataclass_state(self)

    def __setstate__(self, state):
        restore_dataclass_state(self, state)
        self.__post_init__()

    @staticmethod
    def check_client(
        client: Optional[DatabricksClient] = None,
        **client_kwargs: Any
    ):
        if client is None and not client_kwargs:
            return DatabricksClient.current()

        client = client or DatabricksClient.current()
        return replace(client, **client_kwargs)

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self) -> TS:
        self.client.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.client.__exit__(exc_type, exc_val, exc_tb)

    def connect(self):
        return self

    def close(self):
        pass

    @classmethod
    def service_name(cls) -> str:
        return cls.__name__.lower()

    @classmethod
    def url_scheme(cls) -> str:
        return f"dbks+{cls.service_name()}"

    @classmethod
    def current(cls, reset: bool = False) -> TS:
        if reset or cls._current is None:
            cls._current = cls(client=DatabricksClient.current())
        return cls._current

    @classmethod
    def from_parsed_url(cls, url: URL):
        mod = __import__(f"yggdrasil.databricks.{cls.service_name()}", fromlist=[cls.__name__])
        service_cls = getattr(mod, cls.__name__)

        return service_cls(
            client=DatabricksClient.from_parsed_url(url)
        )

    def to_url(self, scheme: str | None = None) -> URL:
        return (
            self.client
            .to_url(scheme=scheme or self.url_scheme())
            .with_path(f"/{self.service_name()}")
        )

    def is_in_databricks_environment(self):
        return self.client.is_in_databricks_environment()

    def default_tags(self, update: bool = True):
        """Return default resource tags for Databricks assets.

        Returns:
            A dict of default tags.
        """
        base = self.client.default_tags(update=update)
        base["ServiceName"] = self.service_name()

        return base

    @property
    def sql(self) -> "SQLEngine":
        return self.client.sql

    @property
    def warehouses(self) -> "SQLWarehouse":
        return self.client.warehouses

    @property
    def compute(self) -> "Compute":
        return self.client.compute

    @property
    def secrets(self) -> "Secrets":
        return self.client.secrets


@dataclass
class DatabricksResource(ABC):
    service: DatabricksService

    def __post_init__(self):
        pass

    def __getstate__(self):
        return serialize_dataclass_state(self)

    def __setstate__(self, state):
        restore_dataclass_state(self, state)
        self.__post_init__()

    @property
    def client(self) -> DatabricksClient:
        return self.service.client
