"""Genie API service wrappers."""

from .resources import (
    DEFAULT_POLL_INTERVAL_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    GENIE_TERMINAL_STATUSES,
    GenieAnswer,
    GenieConversation,
    GenieDefaults,
    GenieSpace,
)
from .service import Genie

__all__ = [
    "Genie",
    "GenieAnswer",
    "GenieConversation",
    "GenieDefaults",
    "GenieSpace",
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "DEFAULT_TIMEOUT_SECONDS",
    "GENIE_TERMINAL_STATUSES",
]
