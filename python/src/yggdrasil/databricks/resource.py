"""Base class for every Databricks resource wrapper.

Split out of :mod:`yggdrasil.databricks.client` so resource modules
can import :class:`DatabricksResource` without pulling the whole
client module's transitive surface.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from .client import DatabricksClient
from .service import DatabricksService

if TYPE_CHECKING:
    from .sql.engine import SQLEngine

__all__ = ["DatabricksResource"]


class DatabricksResource(ABC):

    def __getstate__(self):
        # ``object.__getstate__`` only exists in Python 3.11+, but the
        # project floor is 3.10 (per ``CLAUDE.md``). Build the state dict
        # explicitly from ``__dict__`` so calling ``super().__getstate__()``
        # in subclasses (Table, Cluster, ...) works regardless of runtime.
        # ``service`` is already in ``__dict__`` after ``__init__`` but
        # we restate it so slot-based subclasses can still rely on it.
        state = dict(getattr(self, "__dict__", {}) or {})
        state["service"] = self.service
        return state

    def __setstate__(self, state):
        # Same reason: ``object.__setstate__`` isn't reliable on 3.10.
        # Restore the full instance dict; subclasses that override are
        # free to ``super().__setstate__(state)`` to inherit this behavior.
        self.__dict__.update(state)

    def __init__(self, service=None, *args, **kwargs):
        self.service = DatabricksService.current() if service is None else service
        super().__init__(*args, **kwargs)

    @property
    def client(self) -> DatabricksClient:
        return self.service.client

    @property
    def sql(self) -> "SQLEngine":
        """Shorthand for ``self.service.client.sql`` — the active :class:`SQLEngine`."""
        return self.client.sql
