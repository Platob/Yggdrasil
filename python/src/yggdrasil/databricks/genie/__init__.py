"""Databricks AI/BI Genie — code-oriented manipulation of Genie spaces.

Reach it through ``client.genie`` / ``dbc.genie`` (the :class:`Genie`
service). See :mod:`yggdrasil.databricks.genie.service`.
"""
from .answer import GenieAnswer
from .service import Genie
from .space import GenieSpace

__all__ = ["Genie", "GenieSpace", "GenieAnswer"]
