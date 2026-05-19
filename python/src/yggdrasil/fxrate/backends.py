"""Backend drivers for the free public FX-rate APIs.

Each driver is a small, stateless object that knows how to:

* build a per-source URL + query params for a ``(source, targets, window,
  sampling)`` tuple, and
* parse the wire payload into a list of :class:`FxQuote` rows.

Drivers do not own an :class:`HTTPSession`; :class:`yggdrasil.fxrate
.FxRate` passes its own session into :meth:`Backend.fetch_timeseries` /
:meth:`Backend.fetch_latest`. That keeps the connection pool, retry
policy, caching, and notifier wiring in one place — the session — and
makes a backend swap a one-line change (different driver, same
session).

The drivers ship in priority order: :class:`FrankfurterBackend` first
(ECB reference rates, the cleanest historical timeseries shape),
:class:`FawazBackend` second (200+ currencies including crypto,
jsDelivr CDN, no rate limit), :class:`ErApiBackend` last (latest
snapshot only, but a useful liveness probe).
"""
from __future__ import annotations

import datetime as dt
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

from yggdrasil.data.enums.currency import Currency

if TYPE_CHECKING:
    from .session import FxQuote, FxRate


__all__ = [
    "Backend",
    "BackendError",
    "FrankfurterBackend",
    "ErApiBackend",
    "FawazBackend",
    "DEFAULT_BACKENDS",
]


LOGGER = logging.getLogger(__name__)


class BackendError(RuntimeError):
    """Raised when a backend cannot fulfil a request.

    Carries the originating exception (HTTP error, parse failure,
    unsupported currency) on :attr:`__cause__`. :class:`FxRate`
    catches this and rolls over to the next backend in the chain.
    """


class Backend(ABC):
    """Driver protocol for an FX source.

    Subclasses declare:

    * :attr:`name` — short identifier (logging, ``source`` provenance).
    * :attr:`base_url` — used for URL construction; the session reaches
      the backend via this absolute prefix.
    * :meth:`fetch_timeseries` — build URL, send via *session*, parse,
      return a flat list of :class:`FxQuote` rows.
    * :meth:`fetch_latest` — default is one-day window ending at *at*;
      override when the backend has a cheaper "latest" endpoint.
    """

    name: str
    base_url: str
    default_sampling: str = "1d"

    @abstractmethod
    def fetch_timeseries(
        self,
        session: "FxRate",
        *,
        source: Currency,
        targets: Sequence[Currency],
        start: dt.datetime,
        end: dt.datetime,
        sampling: str,
    ) -> Sequence["FxQuote"]:
        raise NotImplementedError

    def fetch_latest(
        self,
        session: "FxRate",
        *,
        source: Currency,
        targets: Sequence[Currency],
        at: dt.datetime,
    ) -> Sequence["FxQuote"]:
        return self.fetch_timeseries(
            session,
            source=source,
            targets=targets,
            start=at - dt.timedelta(days=2),
            end=at,
            sampling=self.default_sampling,
        )


def _format_iso_date(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).date().isoformat()


# ---------------------------------------------------------------------------
# Frankfurter — ECB reference rates, ~32 currencies, history since 1999
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrankfurterBackend(Backend):
    """ECB reference rates via `api.frankfurter.dev`_.

    .. _api.frankfurter.dev: https://frankfurter.dev/
    """

    name: str = "frankfurter"
    base_url: str = "https://api.frankfurter.dev"
    default_sampling: str = "1d"

    def fetch_timeseries(
        self,
        session: "FxRate",
        *,
        source: Currency,
        targets: Sequence[Currency],
        start: dt.datetime,
        end: dt.datetime,
        sampling: str,
    ) -> Sequence["FxQuote"]:

        params: dict[str, str] = {
            "base": source.code,
            "from": _format_iso_date(start),
            "to": _format_iso_date(end),
        }
        if targets:
            params["quotes"] = ",".join(t.code for t in targets)
        try:
            response = session.get(
                f"{self.base_url}/v2/rates",
                params=params,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise BackendError(
                f"Frankfurter timeseries fetch failed for {source.code}->"
                f"{[t.code for t in targets]}: {exc}"
            ) from exc

        return list(_parse_frankfurter_records(payload, sampling=sampling))

    def fetch_latest(
        self,
        session: "FxRate",
        *,
        source: Currency,
        targets: Sequence[Currency],
        at: dt.datetime,
    ) -> Sequence["FxQuote"]:

        params: dict[str, str] = {"base": source.code}
        if targets:
            params["quotes"] = ",".join(t.code for t in targets)
        try:
            response = session.get(
                f"{self.base_url}/v2/rates",
                params=params,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise BackendError(
                f"Frankfurter latest fetch failed for {source.code}->"
                f"{[t.code for t in targets]}: {exc}"
            ) from exc

        return list(_parse_frankfurter_records(payload, sampling=self.default_sampling))


def _parse_frankfurter_records(
    payload: Any, *, sampling: str,
) -> "Iterable[FxQuote]":
    from .session import FxQuote

    if not isinstance(payload, list):
        raise BackendError(
            f"Frankfurter response must be a list of records; got "
            f"{type(payload).__name__}. Payload preview: {str(payload)[:200]!r}."
        )

    for record in payload:
        if not isinstance(record, Mapping):
            continue
        date_str = record.get("date")
        base = record.get("base")
        quote = record.get("quote")
        rate = record.get("rate")
        if not (isinstance(date_str, str) and isinstance(base, str)
                and isinstance(quote, str) and isinstance(rate, (int, float))):
            continue
        from_ts = dt.datetime.fromisoformat(date_str).replace(tzinfo=dt.timezone.utc)
        to_ts = from_ts + dt.timedelta(days=1)
        yield FxQuote(
            source=base.upper(),
            target=quote.upper(),
            from_timestamp=from_ts,
            to_timestamp=to_ts,
            sampling=sampling,
            value=float(rate),
        )


# ---------------------------------------------------------------------------
# fawazahmed/currency-api — 200+ currencies inc. crypto, jsDelivr CDN
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FawazBackend(Backend):
    """200+ currencies via `currency-api`_ on the jsDelivr CDN.

    Historical lookups switch the path's version tag (``@latest`` →
    ``@YYYY-MM-DD``); one JSON file per ``(date, source)``, so a
    multi-day window fans out to one request per day. No rate
    limits, but the per-day fan-out makes long windows expensive
    relative to Frankfurter's single timeseries call — :class:`FxRate`
    therefore lists this driver after Frankfurter and only fans out
    here when Frankfurter rejects the pair (or is down).

    .. _currency-api: https://github.com/fawazahmed0/exchange-api
    """

    name: str = "fawaz"
    # jsDelivr glues the version tag to the package with ``@`` *and no
    # slash* — the URL form is ``…/currency-api@<date>/v1/…`` rather
    # than ``…/currency-api/@<date>/v1/…``. Store the base without the
    # trailing slash so :meth:`_fetch_payload` can splice ``@<version>``
    # straight on.
    base_url: str = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api"
    # Cloudflare Pages mirror — same content but keyed by *subdomain*
    # rather than path. ``<version>.currency-api.pages.dev/v1/…``.
    fallback_host: str = "currency-api.pages.dev"
    default_sampling: str = "1d"

    def fetch_timeseries(
        self,
        session: "FxRate",
        *,
        source: Currency,
        targets: Sequence[Currency],
        start: dt.datetime,
        end: dt.datetime,
        sampling: str,
    ) -> Sequence["FxQuote"]:
        # One JSON document per (date, base) — fan out across the
        # window day-by-day. Stride is one day even when the caller
        # asked for finer sampling because the upstream cadence is
        # daily; the ``sampling`` column reflects the bucket size
        # the caller asked for so downstream queries can still
        # filter on it consistently.
        start_date = start.astimezone(dt.timezone.utc).date()
        end_date = end.astimezone(dt.timezone.utc).date()
        if start_date > end_date:
            raise BackendError(
                f"Fawaz: window start {start_date.isoformat()} is after end "
                f"{end_date.isoformat()}."
            )

        quotes: list["FxQuote"] = []
        date_cursor = start_date
        first_error: Exception | None = None
        while date_cursor <= end_date:
            try:
                quotes.extend(
                    self._fetch_one_day(
                        session,
                        source=source,
                        targets=targets,
                        date=date_cursor,
                        sampling=sampling,
                    )
                )
            except Exception as exc:
                # One bad day shouldn't sink the whole window — the
                # fawaz CDN occasionally 404s historical dates that
                # the rest of the window covers fine. We remember
                # the first failure so we can surface something if
                # *every* day in the window 404s.
                if first_error is None:
                    first_error = exc
                LOGGER.debug(
                    "Fawaz: skipping %s for %s -> %s (%s)",
                    date_cursor.isoformat(), source.code,
                    [t.code for t in targets], exc,
                )
            date_cursor += dt.timedelta(days=1)

        if not quotes and first_error is not None:
            raise BackendError(
                f"Fawaz: every day in [{start_date.isoformat()}, "
                f"{end_date.isoformat()}] failed for {source.code}: "
                f"{first_error}"
            ) from first_error

        return quotes

    def fetch_latest(
        self,
        session: "FxRate",
        *,
        source: Currency,
        targets: Sequence[Currency],
        at: dt.datetime,
    ) -> Sequence["FxQuote"]:
        # ``@latest`` is the cheap path: one request, one row per target.
        try:
            payload = self._fetch_payload(
                session,
                source=source,
                version="latest",
            )
        except Exception as exc:
            raise BackendError(
                f"Fawaz latest fetch failed for {source.code}: {exc}"
            ) from exc
        return list(
            _parse_fawaz_payload(
                payload,
                source=source.code,
                targets=tuple(t.code for t in targets),
                sampling=self.default_sampling,
            )
        )

    def _fetch_one_day(
        self,
        session: "FxRate",
        *,
        source: Currency,
        targets: Sequence[Currency],
        date: dt.date,
        sampling: str,
    ) -> "Iterable[FxQuote]":
        payload = self._fetch_payload(
            session,
            source=source,
            version=date.isoformat(),
        )
        return _parse_fawaz_payload(
            payload,
            source=source.code,
            targets=tuple(t.code for t in targets),
            sampling=sampling,
        )

    def _fetch_payload(
        self,
        session: "FxRate",
        *,
        source: Currency,
        version: str,
    ) -> Any:
        # jsDelivr lower-cases the currency code in the path; fawaz's
        # docs are explicit about that. ``@{version}`` is appended
        # directly to the package name with no intervening slash.
        code = source.code.lower()
        primary_url = (
            f"{self.base_url}@{version}/v1/currencies/{code}.json"
        )
        try:
            response = session.get(
                primary_url,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            return response.json()
        except Exception as primary_exc:
            # Cloudflare Pages mirror — same content, different CDN.
            # Version is the subdomain, not a path segment, on this
            # host. Worth one retry before giving up because jsDelivr
            # occasionally rate-limits aggressive fan-outs.
            fallback_url = (
                f"https://{version}.{self.fallback_host}/v1/currencies/{code}.json"
            )
            try:
                response = session.get(
                    fallback_url,
                    headers={"Accept": "application/json"},
                )
                response.raise_for_status()
                return response.json()
            except Exception:
                raise primary_exc


def _parse_fawaz_payload(
    payload: Any,
    *,
    source: str,
    targets: Sequence[str],
    sampling: str,
) -> "Iterable[FxQuote]":
    from .session import FxQuote

    if not isinstance(payload, Mapping):
        raise BackendError(
            f"Fawaz response must be a JSON object; got "
            f"{type(payload).__name__}."
        )

    date_str = payload.get("date")
    if not isinstance(date_str, str):
        raise BackendError(
            f"Fawaz response missing 'date' field; got keys "
            f"{sorted(payload.keys())[:10]}."
        )
    rates_key = source.lower()
    rates = payload.get(rates_key)
    if not isinstance(rates, Mapping):
        raise BackendError(
            f"Fawaz response missing {rates_key!r} rates object; got keys "
            f"{sorted(payload.keys())[:10]}."
        )

    from_ts = dt.datetime.fromisoformat(date_str).replace(tzinfo=dt.timezone.utc)
    to_ts = from_ts + dt.timedelta(days=1)
    src_upper = source.upper()
    target_set = {t.upper() for t in targets}

    for code, value in rates.items():
        if not isinstance(code, str) or not isinstance(value, (int, float)):
            continue
        code_upper = code.upper()
        if target_set and code_upper not in target_set:
            continue
        if code_upper == src_upper:
            continue
        yield FxQuote(
            source=src_upper,
            target=code_upper,
            from_timestamp=from_ts,
            to_timestamp=to_ts,
            sampling=sampling,
            value=float(value),
        )


# ---------------------------------------------------------------------------
# open.er-api.com — latest snapshot, no history; useful as last-resort
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ErApiBackend(Backend):
    """Latest-only snapshot via `open.er-api.com`_.

    The free tier has no historical window endpoint, so for a
    timeseries call this backend returns one row per target using
    the response's ``time_last_update_unix`` boundary. :class:`FxRate`
    will only consult this driver when both Frankfurter and Fawaz
    declined the pair — it's a useful safety net (different upstream
    data provider) but a poor historical source.

    .. _open.er-api.com: https://www.exchangerate-api.com/docs/free
    """

    name: str = "er-api"
    base_url: str = "https://open.er-api.com"
    default_sampling: str = "1d"

    def fetch_timeseries(
        self,
        session: "FxRate",
        *,
        source: Currency,
        targets: Sequence[Currency],
        start: dt.datetime,
        end: dt.datetime,
        sampling: str,
    ) -> Sequence["FxQuote"]:
        return list(self._fetch_snapshot(session, source=source, targets=targets, sampling=sampling))

    def fetch_latest(
        self,
        session: "FxRate",
        *,
        source: Currency,
        targets: Sequence[Currency],
        at: dt.datetime,
    ) -> Sequence["FxQuote"]:
        return list(self._fetch_snapshot(session, source=source, targets=targets, sampling=self.default_sampling))

    def _fetch_snapshot(
        self,
        session: "FxRate",
        *,
        source: Currency,
        targets: Sequence[Currency],
        sampling: str,
    ) -> "Iterable[FxQuote]":
        try:
            response = session.get(
                f"{self.base_url}/v6/latest/{source.code}",
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise BackendError(
                f"ER-API snapshot fetch failed for {source.code}: {exc}"
            ) from exc

        return _parse_er_api_payload(
            payload, source=source.code,
            targets=tuple(t.code for t in targets), sampling=sampling,
        )


def _parse_er_api_payload(
    payload: Any,
    *,
    source: str,
    targets: Sequence[str],
    sampling: str,
) -> "Iterable[FxQuote]":
    from .session import FxQuote

    if not isinstance(payload, Mapping):
        raise BackendError(
            f"ER-API response must be a JSON object; got "
            f"{type(payload).__name__}."
        )
    if payload.get("result") != "success":
        raise BackendError(
            f"ER-API non-success response: result={payload.get('result')!r}, "
            f"error-type={payload.get('error-type')!r}."
        )
    rates = payload.get("rates")
    if not isinstance(rates, Mapping):
        raise BackendError(
            f"ER-API 'rates' field must be an object; got "
            f"{type(rates).__name__}."
        )

    from_unix = payload.get("time_last_update_unix")
    to_unix = payload.get("time_next_update_unix")
    if not isinstance(from_unix, (int, float)):
        from_unix = dt.datetime.now(dt.timezone.utc).timestamp()
    if not isinstance(to_unix, (int, float)):
        to_unix = from_unix + 86400.0

    from_ts = dt.datetime.fromtimestamp(float(from_unix), tz=dt.timezone.utc)
    to_ts = dt.datetime.fromtimestamp(float(to_unix), tz=dt.timezone.utc)

    src_upper = source.upper()
    target_set = {t.upper() for t in targets}

    for code, value in rates.items():
        if not isinstance(code, str) or not isinstance(value, (int, float)):
            continue
        code_upper = code.upper()
        if target_set and code_upper not in target_set:
            continue
        if code_upper == src_upper:
            continue
        yield FxQuote(
            source=src_upper,
            target=code_upper,
            from_timestamp=from_ts,
            to_timestamp=to_ts,
            sampling=sampling,
            value=float(value),
        )


#: Default backend chain, in priority order. ``FxRate()`` consults
#: Frankfurter first (cleanest historical timeseries), then Fawaz
#: (broadest currency coverage including crypto), then ER-API
#: (latest snapshot fallback). Override via ``FxRate(backends=…)``.
DEFAULT_BACKENDS: tuple[Backend, ...] = (
    FrankfurterBackend(),
    FawazBackend(),
    ErApiBackend(),
)
