"""DatabricksLoki — the specialized Databricks agent.

A :class:`~yggdrasil.loki.Loki` that detects its workspace only from the
``ygg databricks configure`` session, reasons through a Databricks serving
endpoint, and can deploy itself to run on Databricks compute.
"""
from .agent import DatabricksLoki

__all__ = ["DatabricksLoki"]
