"""FX rate backend contract.

A :class:`Backend` knows how to talk to one upstream FX provider. The
:class:`~yggdrasil.fxrate.session.FxRate` session walks its configured
backends in order, falling through to the next on
:class:`~yggdrasil.fxrate.session.BackendError`.
"""
from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yggdrasil.enums.currency import Currency
    from yggdrasil.fxrate.session import FxQuote
    from yggdrasil.http_.session import HTTPSession


class Backend(ABC):
    """Contract for a single upstream FX provider."""

    name: str
    base_url: str
    default_sampling: str = "1d"

    @abstractmethod
    def fetch_timeseries(
        self,
        session: HTTPSession,
        *,
        source: Currency,
        targets: list[Currency],
        start: dt.datetime,
        end: dt.datetime,
        sampling: str,
    ) -> list[FxQuote]:
        """Fetch a sampled time series for ``source -> each target``."""

    @abstractmethod
    def fetch_latest(
        self,
        session: HTTPSession,
        *,
        source: Currency,
        targets: list[Currency],
        at: dt.datetime | None,
    ) -> list[FxQuote]:
        """Fetch the most recent quote for ``source -> each target``."""
