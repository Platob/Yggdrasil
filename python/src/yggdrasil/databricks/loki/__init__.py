"""DatabricksLoki — the specialized Databricks agent.

A :class:`~yggdrasil.loki.Loki` that detects its workspace only from the
``ygg databricks configure`` session, reasons through a Databricks serving
endpoint, and can deploy itself to run on Databricks compute.
"""
from .agent import DatabricksLoki

# Import the specialized service behaviors so they register on package import.
from . import behaviors as _behaviors  # noqa: F401

__all__ = ["DatabricksLoki"]
