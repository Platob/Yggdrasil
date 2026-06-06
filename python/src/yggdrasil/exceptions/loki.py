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
    """Raised when USD spend would exceed the configured cost budget.

    Carries the running :attr:`used` spend and the :attr:`limit` it crossed
    (both USD) so a caller (the interactive CLI) can show the gap and offer to
    raise the cap step by step.
    """

    def __init__(self, used: float, limit: float, message: str | None = None) -> None:
        self.used = used
        self.limit = limit
        super().__init__(
            message or f"cost budget reached: ${used:.4f} spent ≥ ${limit:.2f} cap"
        )
