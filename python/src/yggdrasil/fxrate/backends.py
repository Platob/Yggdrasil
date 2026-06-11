"""FX rate backend contract.

A :class:`Backend` is one upstream FX provider (Frankfurter, the
exchangerate.host family, ER-API, …). :class:`FxRate` owns a list of
them and walks the list on failure, so a backend only has to do two
things: fetch a timeseries and fetch the latest snapshot. Both return
a flat ``list[FxQuote]`` — one row per (target, sampling bucket) — which
:class:`FxRate` assembles into a long polars frame.

A backend that can't answer raises :class:`BackendError`; that's the
signal for :class:`FxRate` to roll over to the next provider. Any other
exception propagates (it's a bug, not an outage).

The ``session`` argument handed to ``fetch_*`` is the orchestrating
:class:`FxRate` instance, so a concrete backend can reach shared HTTP
machinery / config without threading it through every call site.
"""
from __future__ import annotations

import abc
import datetime as dt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .session import FxQuote, FxRate

__all__ = ["Backend"]


class Backend(abc.ABC):
    """Abstract FX provider.

    Concrete backends set the three class attributes and implement the
    two fetch methods. Identity is the ``base_url`` — two backends with
    the same ``base_url`` address the same upstream.
    """

    name: str
    base_url: str
    default_sampling: str

    @abc.abstractmethod
    def fetch_timeseries(
        self,
        session: FxRate,
        *,
        source: str,
        targets: list[str],
        start: dt.datetime,
        end: dt.datetime,
        sampling: str,
    ) -> list[FxQuote]:
        """Return one quote per (target, bucket) over ``[start, end]``.

        Raise :class:`BackendError` if this provider can't serve the
        request (down, rate-limited, pair unsupported) so the caller
        falls back to the next backend.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fetch_latest(
        self,
        session: FxRate,
        *,
        source: str,
        targets: list[str],
        at: dt.datetime,
    ) -> list[FxQuote]:
        """Return the most recent quote per target as of ``at``."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, base_url={self.base_url!r})"
