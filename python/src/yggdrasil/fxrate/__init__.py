"""FX rate fetching with multi-source fallback.

Public surface:

* :class:`FxRate` — :class:`HTTPSession` subclass; one session, many
  free public FX sources behind the same call.
* :class:`FxQuote` — single FX observation row (used for typing /
  fixtures).
* :class:`Backend` and concrete drivers (:class:`FrankfurterBackend`,
  :class:`FawazBackend`, :class:`ErApiBackend`) for callers that want
  to pin or reorder the fallback chain.
* :func:`deploy_scheduled_fxrate_job` — one-call factory to upsert a
  scheduled Databricks Job that runs the FX fetch on a cron and
  appends to a Delta table. Requires ``ygg[databricks]``.

See :class:`FxRate` for the call shape and example.
"""
from .backends import (
    Backend,
    BackendError,
    DEFAULT_BACKENDS,
    ErApiBackend,
    FawazBackend,
    FrankfurterBackend,
)
from .job import (
    FXRATE_INGESTION_PROVENANCE_COLUMNS,
    deploy_scheduled_fxrate_job,
    fxrate_ingestion_entrypoint,
)
from .session import (
    FX_FRAME_COLUMNS,
    FX_FRAME_GEO_COLUMNS,
    DateLike,
    FxQuote,
    FxRate,
    PairLike,
)


__all__ = [
    "FxRate",
    "FxQuote",
    "PairLike",
    "DateLike",
    "FX_FRAME_COLUMNS",
    "FX_FRAME_GEO_COLUMNS",
    "Backend",
    "BackendError",
    "DEFAULT_BACKENDS",
    "FrankfurterBackend",
    "FawazBackend",
    "ErApiBackend",
    "deploy_scheduled_fxrate_job",
    "fxrate_ingestion_entrypoint",
    "FXRATE_INGESTION_PROVENANCE_COLUMNS",
]
