"""Base class for every Databricks resource wrapper.

Split out of :mod:`yggdrasil.databricks.client` so resource modules
can import :class:`DatabricksResource` without pulling the whole
client module's transitive surface.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Optional

from yggdrasil.io.url import URL

from .client import DatabricksClient
from .service import DatabricksService

if TYPE_CHECKING:
    from .sql.engine import SQLEngine

__all__ = ["DatabricksResource"]


class DatabricksResource(ABC):

    @property
    def explore_url(self) -> Optional[URL]:
        """Workspace UI deep-link for this resource, or ``None``.

        Concrete resources (:class:`Catalog`, :class:`Schema`,
        :class:`Volume`, :class:`Table`, :class:`SQLWarehouse`,
        :class:`Job`, :class:`VolumePath`, …) override this to return
        the ``/explore/data/...`` / ``/sql/warehouses/...`` / ``/jobs/...``
        URL that opens the resource in the workspace UI. The default
        :meth:`__repr__` keys off the override — anything that returns
        a non-``None`` URL gets a ``ClassName(<url>)``-style repr for
        free without restating the format string on every subclass.
        """
        return None

    def __repr__(self) -> str:
        # ``explore_url`` reaches through ``self.service.client.base_url``,
        # which can be ``None`` on a half-constructed / mid-tear-down
        # instance (``__del__`` during interpreter shutdown,
        # ``Session._INSTANCES.clear()`` racing a GC pass, a pickle-
        # restored object whose service hasn't rebound yet). Bake a
        # try/except so the repr survives — having a degraded
        # ``ClassName(...)`` log line is strictly better than swallowing
        # a downstream error because its message couldn't be formatted.
        try:
            url = self.explore_url
        except Exception:
            url = None
        if url is None:
            return super().__repr__()
        return f"{type(self).__name__}({url!r})"

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
