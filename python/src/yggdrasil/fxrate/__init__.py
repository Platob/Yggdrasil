"""FX rate data — multi-source exchange rate fetching, coercion, and framing.

Fetch exchange rates for currency pairs from public backends (Frankfurter/ECB,
Fawaz, others) into tidy polars frames. Rides the yggdrasil HTTPSession for
connection pooling and retry. Multi-backend orchestration with fallback:
BackendError triggers the next backend in the chain.

    from yggdrasil.fxrate import FxRate

    fx = FxRate()                                       # default backends
    df = fx.fetch([("EUR", "USD")], "2024-01-01", "2024-01-31")
    df = fx.latest([("EUR", "USD"), ("GBP", "USD")])

    # Custom backends for testing / alternate upstreams:
    fx = FxRate(backends=[FrankfurterBackend(), FawazBackend()])

PARITY: Python-only. The JS/TS port (``packages/yggdrasil/``) mirrors the
core data-interchange primitives (enums / data / http_ / io / url) and has no
markets/fxrate module; nothing to mirror until a JS counterpart is wanted.
"""
from .session import BackendError, FxQuote, FxRate

__all__ = ["FxRate", "FxQuote", "BackendError"]
