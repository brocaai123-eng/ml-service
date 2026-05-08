"""
Market Volatility Prediction — GARCH(1,1) via the `arch` library.

Calculates daily price returns from market_snapshots and fits a GARCH(1,1)
model to forecast future volatility. This is the industry-standard approach
used in financial markets.

Retraining: GARCH re-estimates alpha/beta parameters from new price data.
"""
import logging

import numpy as np

log = logging.getLogger(__name__)

_cached_params: dict[str, dict] = {}


def _build_returns(zip_code: str) -> np.ndarray | None:
    from db import fetch_market_snapshots

    snapshots = fetch_market_snapshots(zip_code, limit=365)
    prices = [float(s.get("median_price") or 0) for s in snapshots if float(s.get("median_price") or 0) > 0]

    if len(prices) < 10:
        return None

    arr = np.array(prices)
    returns = np.diff(arr) / arr[:-1] * 100
    return returns


def _fit_garch(returns: np.ndarray, zip_code: str | None = None) -> dict:
    """Fit GARCH(1,1) and return volatility metrics."""
    from arch import arch_model

    # Scale returns for numerical stability
    am = arch_model(returns, vol="Garch", p=1, q=1, mean="Zero", rescale=True)

    try:
        res = am.fit(disp="off", show_warning=False)
    except Exception:
        # Fallback to simpler model if GARCH fails
        am = arch_model(returns, vol="EWMA", mean="Zero", rescale=True)
        res = am.fit(disp="off", show_warning=False)

    # Extract conditional volatility
    cond_vol = res.conditional_volatility
    current_vol = float(cond_vol.iloc[-1]) if len(cond_vol) > 0 else float(np.std(returns))

    # Forecast next 30 periods
    try:
        forecasts = res.forecast(horizon=30)
        forecast_variance = forecasts.variance.iloc[-1].values
        forecast_vol = float(np.sqrt(np.mean(forecast_variance)))
    except Exception:
        forecast_vol = current_vol

    # Cache parameters for this ZIP
    params = {
        "current_volatility": current_vol,
        "forecast_volatility": forecast_vol,
        "rolling_variance": cond_vol.tolist()[-20:] if len(cond_vol) > 0 else [],
    }

    if zip_code:
        _cached_params[zip_code] = params

    return params


def run(zip_code: str) -> dict | None:
    returns = _build_returns(zip_code)
    if returns is None:
        return None

    try:
        params = _fit_garch(returns, zip_code)
    except Exception as e:
        log.warning(f"[volatility] GARCH fit failed for {zip_code}: {e}")
        # Fallback to EWMA
        lambda_ = 0.94
        ewma_var = float(np.mean(returns ** 2))
        for r in returns:
            ewma_var = lambda_ * ewma_var + (1 - lambda_) * r ** 2
        current_vol = float(np.sqrt(ewma_var))
        params = {
            "current_volatility": current_vol,
            "forecast_volatility": current_vol,
            "rolling_variance": [],
        }

    current_vol = params["current_volatility"]
    avg_swing = float(np.mean(np.abs(returns)))

    # Normalize to 0-100 index
    volatility_index = max(0, min(100, current_vol * 20))

    level = "High" if volatility_index > 60 else "Medium" if volatility_index > 30 else "Low"

    recommendation = {
        "High": "High volatility — consider waiting for stabilization before transacting",
        "Medium": "Moderate volatility — proceed with caution, use contingencies",
        "Low": "Low volatility — stable window for transactions",
    }[level]

    score = max(0, min(100, round(100 - volatility_index)))
    confidence = max(40, min(85, 40 + min(len(returns), 60) * 0.75))

    # Volatility trend: increasing or decreasing?
    rolling = params.get("rolling_variance", [])
    if len(rolling) >= 6:
        recent_avg = np.mean(rolling[-len(rolling) // 3:])
        older_avg = np.mean(rolling[:len(rolling) // 3])
        vol_trend = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
    else:
        vol_trend = 0

    direction = "up" if vol_trend > 10 else "down" if vol_trend < -10 else "stable"

    headline = (
        f"{level} volatility | {avg_swing:.1f}% avg swing | "
        f"Window: {'Stable for transactions' if level == 'Low' else 'Proceed with caution' if level == 'Medium' else 'Wait for stabilization'}"
    )

    return {
        "model_key": "market_volatility",
        "headline": headline,
        "score": round(score, 2),
        "confidence_pct": round(confidence, 2),
        "direction": direction,
        "payload": {
            "volatility_index": round(volatility_index, 2),
            "level": level,
            "avg_daily_swing_pct": round(avg_swing, 3),
            "recommendation": recommendation,
            "data_points": len(returns) + 1,
            "rolling_variance": [round(v, 4) for v in (rolling[-20:] if rolling else [])],
        },
        "model_version": "garch-v1",
    }


def retrain() -> None:
    """Re-fit GARCH models for all monitored ZIPs."""
    MONITORED_ZIPS = [
        "33470", "33411", "33401", "33413", "33418",
        "33458", "33467", "33328", "33309", "33063",
    ]

    for zip_code in MONITORED_ZIPS:
        try:
            returns = _build_returns(zip_code)
            if returns is not None and len(returns) >= 10:
                _fit_garch(returns, zip_code)
                log.info(f"[volatility] Retrained GARCH for {zip_code} ({len(returns)} returns)")
        except Exception as e:
            log.warning(f"[volatility] Retrain failed for {zip_code}: {e}")
