"""Lightweight, dependency-graceful time-series forecaster.

Engineers trend + lag + rolling-mean + Fourier-seasonal features from a single
series and fits a regressor, forecasting ``horizon`` steps recursively with a
widening confidence band from the in-sample residuals. The model is chosen by a
fallback chain — gradient boosting (xgboost, then scikit-learn) when installed,
else a closed-form numpy ridge — so it works on a bare node and gets sharper
where the ML libs exist.
"""
from __future__ import annotations

import numpy as np


class _RidgeNP:
    """Closed-form ridge regression (no scikit-learn needed)."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._w: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_RidgeNP":
        xb = np.hstack([np.ones((x.shape[0], 1)), x])           # bias column
        n_feat = xb.shape[1]
        reg = self.alpha * np.eye(n_feat)
        reg[0, 0] = 0.0                                         # don't penalise bias
        self._w = np.linalg.solve(xb.T @ xb + reg, xb.T @ y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        xb = np.hstack([np.ones((x.shape[0], 1)), x])
        return xb @ self._w


def make_model(model: str):
    """Return (estimator, name) honouring the request, with graceful fallback."""
    m = (model or "auto").lower()
    if m in ("auto", "xgboost", "xgb"):
        try:
            import xgboost as xgb
            return xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1,
                                    subsample=0.9, verbosity=0), "xgboost"
        except Exception:
            if m in ("xgboost", "xgb"):
                pass
    if m in ("auto", "xgboost", "xgb", "gbr", "sklearn"):
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.1), "gbr"
        except Exception:
            pass
    return _RidgeNP(alpha=1.0), "ridge"


def _features(idx: int, window: np.ndarray, period: int | None) -> list[float]:
    f: list[float] = [float(idx)]                              # trend
    f.extend(float(v) for v in window)                         # lags (oldest→newest)
    f.append(float(np.mean(window)))                           # rolling mean
    f.append(float(window[-1] - window[0]))                    # window slope
    if period and period > 1:
        ang = 2.0 * np.pi * idx / period
        f.extend([float(np.sin(ang)), float(np.cos(ang))])
    return f


def forecast_series(y: list[float], horizon: int, period: int | None, model: str):
    """Forecast ``horizon`` steps. Returns (preds, lower, upper, rmse, model_used)."""
    yv = np.asarray([v for v in y if v is not None], dtype=float)
    n = len(yv)
    horizon = max(1, int(horizon))
    if n < 5:                                                  # too short to learn
        last = float(yv[-1]) if n else 0.0
        return [last] * horizon, [last] * horizon, [last] * horizon, None, "naive"

    lag = max(2, min(16, n // 3))
    period = period if (period and 1 < period <= n // 2) else None

    rows_x, rows_y = [], []
    for i in range(lag, n):
        rows_x.append(_features(i, yv[i - lag:i], period))
        rows_y.append(yv[i])
    x = np.asarray(rows_x, dtype=float)
    ytr = np.asarray(rows_y, dtype=float)

    reg, used = make_model(model)
    try:
        reg.fit(x, ytr)
    except Exception:                                          # any backend failure → ridge
        reg, used = _RidgeNP(alpha=1.0), "ridge"
        reg.fit(x, ytr)

    resid = ytr - reg.predict(x)
    sd = float(np.std(resid)) if resid.size else 0.0
    rmse = float(np.sqrt(np.mean(resid ** 2))) if resid.size else None

    hist = list(yv)
    preds, lower, upper = [], [], []
    for h in range(horizon):
        idx = n + h
        window = np.asarray(hist[-lag:], dtype=float)
        p = float(reg.predict(np.asarray([_features(idx, window, period)], dtype=float))[0])
        preds.append(p)
        hist.append(p)
        band = 1.96 * sd * float(np.sqrt(1.0 + h))             # widen with horizon
        lower.append(p - band)
        upper.append(p + band)
    return preds, lower, upper, rmse, used
