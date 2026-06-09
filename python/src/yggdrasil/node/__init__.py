"""yggdrasil.node — local FastAPI node server.

The node is a lightweight HTTP service that exposes:

- ``/api/v2/fs/``        — filesystem browser (ls, read, write, delete)
- ``/api/v2/tabular/``   — tabular file inspection and preview
- ``/api/v2/analysis/``  — aggregation, series, OHLC, pivot, forecast, finance
- ``/api/v2/saga/``      — SQL catalog + execution engine (DuckDB-backed)
- ``/api/v2/network/``   — node peering / cluster
- ``/api/v2/pyfunc``     — Python function registry
- ``/api/v2/pyenv``      — Python env stubs
- ``/api/v2/audit``      — operation log
- ``/api/call``          — remote function dispatch (Arrow/pickle)
- ``/api/messenger``     — in-memory message bus
- ``/api/ping``, ``/api/v2/health``, ``/api/v2/stats``, ``/api/v2/backend``

Start with ``ygg node serve`` (CLI, not yet wired) or programmatically::

    from yggdrasil.node.app import create_app
    from yggdrasil.node.config import Settings
    import uvicorn
    uvicorn.run(create_app(Settings()), host="0.0.0.0", port=8100)
"""
from __future__ import annotations

from yggdrasil.node import transport

__all__ = ["transport"]
