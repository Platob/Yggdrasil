"""Loki agent exceptions.

The errors the Loki agent raises on its own — chiefly token-budget
enforcement, so an interactive session can stop and ask the user to raise
the cap rather than spending without bound. All derive from
:class:`~yggdrasil.exceptions.base.YGGException`.
"""
from __future__ import annotations

from .base import YGGException

__all__ = ["LokiError", "TokenBudgetExceeded"]


class LokiError(YGGException):
    """Base for every error the Loki agent raises on its own."""


class TokenBudgetExceeded(LokiError):
    """Raised when token consumption would exceed the configured budget.

    Carries the running :attr:`used` total and the :attr:`limit` it crossed
    so a caller (the interactive CLI) can show the gap and offer to raise the
    cap step by step.
    """

    def __init__(self, used: int, limit: int, message: str | None = None) -> None:
        self.used = used
        self.limit = limit
        super().__init__(
            message
            or f"token budget reached: {used:,} used ≥ {limit:,} limit"
        )
