"""Per-service CLI sub-commands for ``ygg-databricks``.

Each command class registers itself as a sub-parser under the main CLI
and dispatches to the yggdrasil service layer.
"""
from .clusters import ClustersCommand
from .configure import ConfigureCommand
from .deploy import DeployCommand
from .environment import EnvironmentCommand
from .fs import FSCommand
from .jobs import JobsCommand
from .seed import SeedCommand
from .sql import SQLCommand
from .tables import TablesCommand
from .warehouses import WarehousesCommand
from .wheel import WheelCommand

__all__ = [
    "ClustersCommand",
    "ConfigureCommand",
    "DeployCommand",
    "EnvironmentCommand",
    "FSCommand",
    "JobsCommand",
    "SeedCommand",
    "SQLCommand",
    "TablesCommand",
    "WarehousesCommand",
    "WheelCommand",
]
