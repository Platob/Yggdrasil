"""DatabricksLoki — the specialized Databricks agent.

A :class:`~yggdrasil.loki.Loki` that detects its workspace only from the
``ygg databricks configure`` session, reasons through a Databricks serving
endpoint, and can deploy itself to run on Databricks compute.

The specialized **service skills** live next to the code they drive — one
``loki.py`` per service module (``databricks/sql/loki.py``,
``databricks/table/loki.py``, …) for isolation. Importing this package imports
them all, so they register into the global Loki catalog (``ygg loki skills``).
"""
from .agent import DatabricksLoki

# Import each service's Loki skill module so it registers on package import.
# One per dbc.<service> accessor — keep this list in step with the services.
from ..sql import loki as _sql          # noqa: F401  databricks-sql
from ..catalog import loki as _catalog  # noqa: F401  databricks-catalogs
from ..schema import loki as _schema    # noqa: F401  databricks-schemas
from ..table import loki as _table      # noqa: F401  databricks-tables
from ..warehouse import loki as _wh     # noqa: F401  databricks-warehouses
from ..compute import loki as _compute  # noqa: F401  databricks-clusters
from ..job import loki as _job          # noqa: F401  databricks-jobs / -job-runs
from ..volume import loki as _volume    # noqa: F401  databricks-volumes
from ..secrets import loki as _secrets  # noqa: F401  databricks-secrets
from ..iam import loki as _iam          # noqa: F401  databricks-iam
from ..ai import loki as _ai            # noqa: F401  databricks-serving
from ..genie import loki as _genie      # noqa: F401  genie
from . import skills as _skills         # noqa: F401  databricks-mcp (cross-service)

__all__ = ["DatabricksLoki"]
