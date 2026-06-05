"""Base class for every Databricks service wrapper.

Split out of :mod:`yggdrasil.databricks.client` so service modules
can import :class:`DatabricksService` without pulling the whole
client module's transitive surface.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar, Optional, TYPE_CHECKING, TypeVar

from yggdrasil.url import URL

from .client import DatabricksClient

if TYPE_CHECKING:
    from .ai import DatabricksAI
    from .catalog.catalogs import Catalogs
    from .compute.service import Compute
    from .genie import Genie
    from .job.service import Jobs, JobRuns
    from .schema.schemas import Schemas
    from .secrets.service import Secrets
    from .sql.engine import SQLEngine
    from .table.tables import Tables
    from .volume.volumes import Volumes
    from .warehouse.service import Warehouses

__all__ = ["DatabricksService"]

TS = TypeVar("TS", bound="DatabricksService")


class DatabricksService(ABC):
    """Base class for every Databricks service wrapper.

    Subclasses are plain classes (no ``@dataclass``); they inherit the
    ``client``-only constructor by default and override :meth:`__init__`
    explicitly when they need to accept additional configuration.
    """

    _current: ClassVar[Optional["DatabricksService"]] = None

    def __init__(self, client: Optional[DatabricksClient] = None):
        self.client = client if client is not None else DatabricksClient.current()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(client={self.client!r})"

    def __getstate__(self):
        return {"client": self.client}

    def __setstate__(self, state):
        self.client = state["client"]

    @staticmethod
    def check_client(
        client: Optional[DatabricksClient] = None,
        **client_kwargs: Any,
    ):
        if client is None and not client_kwargs:
            return DatabricksClient.current()

        client = client or DatabricksClient.current()
        if not client_kwargs:
            return client
        merged = {name: getattr(client, name, None) for name in client._INIT_NAMES}
        merged.update(client_kwargs)
        return type(client)(**merged)

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

        return service_cls(client=DatabricksClient.from_parsed_url(url))

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
    def warehouses(self) -> "Warehouses":
        return self.client.warehouses

    @property
    def compute(self) -> "Compute":
        return self.client.compute

    @property
    def secrets(self) -> "Secrets":
        return self.client.secrets

    @property
    def tables(self) -> "Tables":
        """Collection-level Unity Catalog table service (shorthand for ``client.tables``)."""
        return self.client.tables

    @property
    def views(self) -> "Tables":
        """Alias for :attr:`tables` — :class:`Table` covers both managed/external
        tables and view-shaped securables."""
        return self.client.tables

    @property
    def catalogs(self) -> "Catalogs":
        """Collection-level Unity Catalog hierarchy service (shorthand for ``client.catalogs``)."""
        return self.client.catalogs

    @property
    def schemas(self) -> "Schemas":
        """Collection-level Unity Catalog schema service (shorthand for ``client.schemas``)."""
        return self.client.schemas

    @property
    def volumes(self) -> "Volumes":
        """Collection-level Unity Catalog volume service (shorthand for ``client.volumes``)."""
        return self.client.volumes

    @property
    def genie(self) -> "Genie":
        """Genie service (shorthand for ``client.genie``)."""
        return self.client.genie

    @property
    def ai(self) -> "DatabricksAI":
        """Databricks AI umbrella service (shorthand for ``client.ai``)."""
        return self.client.ai

    @property
    def jobs(self) -> "Jobs":
        return self.client.jobs

    @property
    def job_runs(self) -> "JobRuns":
        return self.client.job_runs

