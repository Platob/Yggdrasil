"""Per-service CLI sub-commands for ``ygg-databricks``.

Each command class registers itself as a sub-parser under the main CLI
and dispatches to the yggdrasil service layer.
"""
from .clusters import ClustersCommand
from .fs import FSCommand
from .jobs import JobsCommand
from .tables import TablesCommand
from .warehouses import WarehousesCommand

__all__ = [
    "ClustersCommand",
    "FSCommand",
    "JobsCommand",
    "TablesCommand",
    "WarehousesCommand",
]
