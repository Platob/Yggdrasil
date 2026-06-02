"""Technical analysis indicators over price series.

All indicators run on numpy arrays after a single lazy polars scan + collect.
Pure Python/numpy/polars — no TA-Lib dependency. The lazy scan projects only
the columns each requested indicator needs, sorts on the index column when one
is given, and streams the collect with bounded memory.
"""
from __future__ import annotations

import math
from functools import partial

import numpy as np
import polars as pl
from fastapi.concurrency import run_in_threadpool

from yggdrasil.data.options import CastOptions
from yggdrasil.exceptions.api import BadRequestError
from yggdrasil.path import Path as YggPath

from ...config import Settings
from ...exceptions import ForbiddenError, NotFoundError
from ..schemas.technical import IndicatorSeries, TechnicalRequest, TechnicalResult
from .fs import FsService


def _safe(v):
    """JSON-safe scalar: drop NaN/inf, coerce numpy scalars to plain Python."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, str)):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _to_list(arr: np.ndarray) -> list:
    """Bulk numpy → Python list, replacing non-finite values with None.

    ``arr.tolist()`` is a single C loop; patching only the non-finite indices
    afterwards is orders of magnitude faster than a per-element Python loop on
    large arrays (500k rows: ~5ms vs ~2000ms)."""
    result: list = arr.tolist()
    for i in np.where(~np.isfinite(arr))[0]:
        result[i] = None
    return result


def _rsi(price: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder RSI via polars EWM (alpha = 1/period, Rust kernel — ~100× faster
    than an equivalent Python loop on large arrays)."""
    delta = np.diff(price, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    alpha = 1.0 / period
    avg_gain = pl.Series(gain).ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True).to_numpy().astype(float)
    avg_loss = pl.Series(loss).ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True).to_numpy().astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, np.inf)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi[:period] = np.nan
    return rsi


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    """Standard EMA (alpha = 2/(span+1)) via polars' EWM Rust kernel.

    NaN (e.g. from a Float64 cast of a null price) is mapped to a polars null so
    ``ignore_nulls=True`` carries the prior EMA forward instead of poisoning the
    whole recurrence — matching a hand-rolled NaN-skipping EMA loop."""
    s = pl.Series(arr)
    s = pl.select(pl.when(s.is_nan()).then(None).otherwise(s).alias("x")).to_series()
    return s.ewm_mean(span=span, adjust=False, ignore_nulls=True).to_numpy().astype(float)


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    """Rolling mean via polars' vectorized kernel. A window containing any NaN
    yields NaN (matches a no-partial-window simple moving average)."""
    # Default window == period, so leading rows stay null until the window fills.
    return pl.Series(arr).rolling_mean(period).to_numpy().astype(float)


def _bb(price: np.ndarray, period: int = 20, std_dev: float = 2.0):
    s = pl.Series(price)
    sma = s.rolling_mean(period).to_numpy().astype(float)
    # ddof=1 sample std, matching np.std(..., ddof=1) on each window.
    std = s.rolling_std(period, ddof=1).to_numpy().astype(float)
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    return upper, sma, lower


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder ATR. The true range is vectorized; the smoothing is a documented
    sequential recurrence (Wilder seeds the first ATR as the simple mean of the
    first ``period`` true ranges, which a plain EWM does not reproduce), so the
    smoothing stays a Python loop over the n bars."""
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    tr[0] = np.nan
    alpha = 1.0 / period
    atr = np.full_like(tr, np.nan, dtype=float)
    if len(tr) >= period:
        valid = tr[1:period + 1]
        if not np.any(np.isnan(valid)):
            atr[period] = np.mean(valid)
        for i in range(period + 1, len(tr)):
            if not np.isnan(tr[i]) and not np.isnan(atr[i - 1]):
                atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
    return atr


def _vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    typical = (high + low + close) / 3.0
    cumvol = np.cumsum(np.where(np.isnan(volume), 0, volume))
    cumtpv = np.cumsum(np.where(np.isnan(typical * volume), 0, typical * volume))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cumvol > 0, cumtpv / cumvol, np.nan)


def _obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    direction = np.where(delta > 0, 1.0, np.where(delta < 0, -1.0, 0.0))
    return np.nancumsum(direction * volume)


def _stoch(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3):
    # Rolling highest-high / lowest-low via polars kernels; %K = position of the
    # close inside that band. Flat bands (hh == ll) stay NaN.
    hh = pl.Series(high).rolling_max(k_period).to_numpy().astype(float)
    ll = pl.Series(low).rolling_min(k_period).to_numpy().astype(float)
    span = hh - ll
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(span > 0, 100.0 * (close - ll) / span, np.nan)
    d = _sma(k, d_period)
    return k, d


class TechnicalService:
    def __init__(self, settings: Settings, *, fs: FsService) -> None:
        self.settings = settings
        self.fs = fs

    async def compute(self, req: TechnicalRequest) -> TechnicalResult:
        return await run_in_threadpool(partial(self._compute, req))

    def _compute(self, req: TechnicalRequest) -> TechnicalResult:
        resolved = self.fs._resolve(req.path)
        if not resolved.exists():
            raise NotFoundError(f"File not found: {req.path!r}")
        if resolved.is_dir():
            raise ForbiddenError(f"Not a file: {req.path!r}")

        ext = resolved.suffix.lstrip(".").lower()
        try:
            if ext in ("parquet", "pq"):
                lf = pl.scan_parquet(str(resolved))
            elif ext == "csv":
                lf = pl.scan_csv(str(resolved))
            elif ext == "ndjson":
                lf = pl.scan_ndjson(str(resolved))
            else:
                with YggPath.from_(str(resolved)).open("rb") as bio:
                    table = bio.read_arrow_table(options=CastOptions(row_limit=self.settings.analysis_max_rows + 1))
                lf = pl.from_arrow(table).lazy()
        except Exception as exc:
            raise BadRequestError(f"Cannot read {req.path!r}: {exc}")

        schema = lf.collect_schema()
        cols = set(schema.names())
        if req.close not in cols:
            raise BadRequestError(f"close column {req.close!r} not found; available: {sorted(cols)}")

        # Project only the columns the requested indicators actually need.
        needed = {req.close}
        if req.x and req.x in cols:
            needed.add(req.x)
        for ind in req.indicators:
            t = ind.type.lower()
            if t in ("atr", "vwap", "stoch"):
                for col in (req.high, req.low):
                    if col and col in cols:
                        needed.add(col)
                    elif col:
                        raise BadRequestError(f"column {col!r} not found for {t.upper()}")
            if t in ("vwap", "obv") and req.volume:
                if req.volume in cols:
                    needed.add(req.volume)
                else:
                    raise BadRequestError(f"volume column {req.volume!r} not found for {t.upper()}")

        plan = lf.select(list(needed))
        if req.x and req.x in cols:
            plan = plan.sort(req.x)
        df = plan.collect(engine="streaming")
        source_rows = df.height

        close_arr = df[req.close].cast(pl.Float64, strict=False).to_numpy().astype(float)
        x_vals = df[req.x].to_list() if (req.x and req.x in cols) else list(range(source_rows))
        close_list = _to_list(close_arr)

        # Only the columns an indicator actually needs were projected into `df`;
        # pull high/low/volume from `needed`, not the full file schema.
        high_arr = df[req.high].cast(pl.Float64, strict=False).to_numpy().astype(float) if (req.high and req.high in needed) else None
        low_arr = df[req.low].cast(pl.Float64, strict=False).to_numpy().astype(float) if (req.low and req.low in needed) else None
        vol_arr = df[req.volume].cast(pl.Float64, strict=False).to_numpy().astype(float) if (req.volume and req.volume in needed) else None

        indicators: list[IndicatorSeries] = []

        for spec in req.indicators:
            t = spec.type.lower()

            if t == "rsi":
                period = spec.period or 14
                arr = _rsi(close_arr, period)
                indicators.append(IndicatorSeries(name=f"RSI({period})", series=_to_list(arr)))

            elif t == "sma":
                period = spec.period or 20
                arr = _sma(close_arr, period)
                indicators.append(IndicatorSeries(name=f"SMA({period})", series=_to_list(arr)))

            elif t == "ema":
                period = spec.period or 20
                arr = _ema(close_arr, period)
                indicators.append(IndicatorSeries(name=f"EMA({period})", series=_to_list(arr)))

            elif t == "macd":
                fast = spec.fast or 12
                slow = spec.slow or 26
                sig = spec.signal or 9
                ema_fast = _ema(close_arr, fast)
                ema_slow = _ema(close_arr, slow)
                macd_line = ema_fast - ema_slow
                signal_line = _ema(macd_line, sig)
                histogram = macd_line - signal_line
                indicators.append(IndicatorSeries(name=f"MACD({fast},{slow})", series=_to_list(macd_line)))
                indicators.append(IndicatorSeries(name=f"MACD_signal({sig})", series=_to_list(signal_line)))
                indicators.append(IndicatorSeries(name="MACD_hist", series=_to_list(histogram)))

            elif t == "bb":
                period = spec.period or 20
                std_dev = spec.std_dev or 2.0
                upper, mid, lower = _bb(close_arr, period, std_dev)
                indicators.append(IndicatorSeries(name=f"BB_upper({period})", series=_to_list(upper)))
                indicators.append(IndicatorSeries(name=f"BB_mid({period})", series=_to_list(mid)))
                indicators.append(IndicatorSeries(name=f"BB_lower({period})", series=_to_list(lower)))

            elif t == "atr":
                if high_arr is None or low_arr is None:
                    raise BadRequestError("ATR requires high and low columns")
                period = spec.period or 14
                arr = _atr(high_arr, low_arr, close_arr, period)
                indicators.append(IndicatorSeries(name=f"ATR({period})", series=_to_list(arr)))

            elif t == "vwap":
                if high_arr is None or low_arr is None or vol_arr is None:
                    raise BadRequestError("VWAP requires high, low, and volume columns")
                arr = _vwap(high_arr, low_arr, close_arr, vol_arr)
                indicators.append(IndicatorSeries(name="VWAP", series=_to_list(arr)))

            elif t == "obv":
                if vol_arr is None:
                    raise BadRequestError("OBV requires a volume column")
                arr = _obv(close_arr, vol_arr)
                indicators.append(IndicatorSeries(name="OBV", series=_to_list(arr)))

            elif t == "stoch":
                if high_arr is None or low_arr is None:
                    raise BadRequestError("Stochastic requires high and low columns")
                k_period = spec.period or 14
                d_period = spec.d_period or 3
                k, d = _stoch(high_arr, low_arr, close_arr, k_period, d_period)
                indicators.append(IndicatorSeries(name=f"Stoch_K({k_period})", series=_to_list(k)))
                indicators.append(IndicatorSeries(name=f"Stoch_D({d_period})", series=_to_list(d)))

            else:
                raise BadRequestError(
                    f"unknown indicator type {spec.type!r}; supported: rsi|macd|bb|sma|ema|atr|vwap|obv|stoch")

        return TechnicalResult(
            node_id=self.settings.node_id,
            path=req.path,
            x=x_vals,
            close=close_list,
            indicators=indicators,
            source_rows=source_rows,
        )
