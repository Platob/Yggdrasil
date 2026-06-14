"""Frankfurter FX backend — free ECB reference rates, no auth.

https://www.frankfurter.app — daily ECB rates, a public mirror of the
European Central Bank reference series. Used as the default backend when an
``FxRate`` session is constructed without explicit backends.
"""
from __future__ import annotations

import datetime as dt

from yggdrasil.fxrate.backends import Backend
from yggdrasil.fxrate.session import BackendError, FxQuote


class FrankfurterBackend(Backend):
    name = "frankfurter"
    base_url = "https://api.frankfurter.app"
    default_sampling = "1d"

    def fetch_timeseries(
        self, session, *, source, targets, start, end, sampling
    ) -> list[FxQuote]:
        start_s = start.date().isoformat()
        end_s = end.date().isoformat()
        symbols = ",".join(t.code for t in targets)
        url = f"{self.base_url}/{start_s}..{end_s}?from={source.code}&to={symbols}"
        data = self._get_json(url)
        rates = data.get("rates", {})
        quotes: list[FxQuote] = []
        for day_str, by_target in sorted(rates.items()):
            day = dt.datetime.fromisoformat(day_str).replace(tzinfo=dt.timezone.utc)
            for target in targets:
                value = by_target.get(target.code)
                if value is None:
                    continue
                quotes.append(
                    FxQuote(
                        source=source.code,
                        target=target.code,
                        from_timestamp=day,
                        to_timestamp=day + dt.timedelta(days=1),
                        sampling=sampling,
                        value=float(value),
                    )
                )
        return quotes

    def fetch_latest(self, session, *, source, targets, at) -> list[FxQuote]:
        when = "latest" if at is None else at.date().isoformat()
        symbols = ",".join(t.code for t in targets)
        url = f"{self.base_url}/{when}?from={source.code}&to={symbols}"
        data = self._get_json(url)
        day_str = data.get("date")
        day = (
            dt.datetime.fromisoformat(day_str).replace(tzinfo=dt.timezone.utc)
            if day_str
            else dt.datetime.now(tz=dt.timezone.utc)
        )
        rates = data.get("rates", {})
        quotes: list[FxQuote] = []
        for target in targets:
            value = rates.get(target.code)
            if value is None:
                continue
            quotes.append(
                FxQuote(
                    source=source.code,
                    target=target.code,
                    from_timestamp=day,
                    to_timestamp=day,
                    sampling="latest",
                    value=float(value),
                )
            )
        return quotes

    def _get_json(self, url: str) -> dict:
        import httpx

        try:
            resp = httpx.get(url, timeout=10.0)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            raise BackendError(f"{self.name} request failed for {url!r}: {exc}") from exc
