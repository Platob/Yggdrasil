"""``yggdrasil.node`` — FastAPI trading backend.

Entry points::

    from yggdrasil.node import create_app, serve
    from yggdrasil.node import remote, Settings
    from yggdrasil.node.transport import serialize_result, CONTENT_TYPE_ARROW_STREAM

Start the server::

    serve()                              # reads YGG_NODE_* env vars
    serve(Settings(port=9000))           # explicit settings

Register remote functions::

    @remote()
    def add(x: int, y: int) -> int:
        return x + y
"""
from __future__ import annotations

from yggdrasil.node.app import create_app, serve
from yggdrasil.node.config import Settings
from yggdrasil.node.remote import call_registered, list_registered, remote
from yggdrasil.node.transport import (
    CONTENT_TYPE_ARROW_STREAM,
    CONTENT_TYPE_PICKLE,
    deserialize_pickle,
    deserialize_result,
    is_tabular,
    read_arrow_stream,
    serialize_pickle,
    serialize_result,
    to_arrow_table,
    write_arrow_stream,
    write_arrow_stream_chunked,
)

__all__ = [
    "create_app",
    "serve",
    "Settings",
    "remote",
    "list_registered",
    "call_registered",
    "CONTENT_TYPE_ARROW_STREAM",
    "CONTENT_TYPE_PICKLE",
    "serialize_pickle",
    "deserialize_pickle",
    "write_arrow_stream",
    "write_arrow_stream_chunked",
    "read_arrow_stream",
    "is_tabular",
    "to_arrow_table",
    "serialize_result",
    "deserialize_result",
]
