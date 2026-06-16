"""Node service exceptions.

Raised by :mod:`yggdrasil.node` services when a lookup fails or a request
can't be satisfied. All derive from :class:`YGGException` so callers catch
them alongside the rest of the library.
"""
from __future__ import annotations

from .base import YGGException

__all__ = ["NodeError", "NodeNotFoundError", "NodeBadRequestError"]


class NodeError(YGGException):
    """Base for every node-service error."""


class NodeNotFoundError(NodeError):
    """A requested node resource (channel, message, function, file) is absent."""


class NodeBadRequestError(NodeError):
    """The request is malformed or references a nonexistent column/path."""
