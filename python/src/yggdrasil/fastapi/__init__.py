"""Yggdrasil FastAPI service — Arrow-first catalog and data endpoints.

The service wraps :class:`yggdrasil.io.tabular.TabularEngine` (the
process-wide ``catalog.schema.name`` registry) and exposes:

- catalog navigation (catalogs / schemas / tables / per-table schema)
- source registration (path / inline rows / binary upload / delete)
- data exposure (Arrow IPC stream by default, plus any media type
  the :class:`Tabular` registry knows about)

See :func:`yggdrasil.fastapi.app.create_app` for the factory and
:func:`yggdrasil.fastapi.main.main` for the uvicorn entry point.
"""

from .app import app, create_app
from .config import Settings, get_settings


__all__ = ["app", "create_app", "Settings", "get_settings"]
