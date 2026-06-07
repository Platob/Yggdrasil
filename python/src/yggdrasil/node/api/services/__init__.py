"""Business-logic services for the node API.

One service per domain — :class:`MarketDataService`, :class:`PortfolioService`,
:class:`AnalysisService`, :class:`FsService`. Route handlers hold no logic;
they resolve a service off ``app.state`` and translate service exceptions
(``KeyError``/``ValueError``/``FileNotFoundError``) into HTTP status codes.
"""
from __future__ import annotations

from .analysis import AnalysisService
from .fs import FsService
from .market import MarketDataService
from .portfolio import PortfolioService

__all__ = [
    "MarketDataService",
    "PortfolioService",
    "AnalysisService",
    "FsService",
]
