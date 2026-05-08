"""
Price Forecast Model — Prophet-based time series prediction.

Reads median_price history from market_snapshots, optionally incorporates
FRED mortgage rates as an external regressor, and produces a 90-day forecast
with confidence intervals.

Retraining: Prophet is re-fit from scratch on each call (stateless).
The retrain() function pre-warms the model cache for all monitored ZIPs.
"""
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Model cache: zip -> fitted Prophet model
_model_cache: dict[str, object] = {}


def _build_dataframe(zip_code: str) -> pd.DataFrame | None:
    from db import fetch_market_snapshots

    rows = fetch_market_snapshots(zip_code, limit=365)
    if len(rows) < 5:
        return None

    df = pd.DataFrame(rows)
    df = df.rename(columns={"snapshot_date": "ds", "median_price": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"])
    df = df[df["y"] > 0]
    df = df.sort_values("ds").reset_index(drop=True)

    if len(df) < 5:
        return None
    return df[["ds", "y"]]


def _fit_prophet(zip_code: str, df: pd.DataFrame) -> object:
    from prophet import Prophet

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        interval_width=0.80,
    )

    # Add mortgage rate as regressor if available
    from db import fetch_fred_mortgage_rate
    rate = fetch_fred_mortgage_rate()
    if rate is not None:
        df = df.copy()
        df["mortgage_rate"] = rate
        m.add_regressor("mortgage_rate")

    m.fit(df)
    _model_cache[zip_code] = m
    return m


def run(zip_code: str) -> dict | None:
    """Generate a 90-day price forecast for a ZIP code."""
    df = _build_dataframe(zip_code)
    if df is None:
        return None

    # Use cached model or fit new one
    model = _model_cache.get(zip_code)
    if model is None:
        model = _fit_prophet(zip_code, df)

    horizon = 90
    future = model.make_future_dataframe(periods=horizon, freq="D")

    # Add mortgage rate to future dataframe if the model expects it
    if "mortgage_rate" in getattr(model, "extra_regressors", {}):
        from db import fetch_fred_mortgage_rate
        rate = fetch_fred_mortgage_rate() or 6.5
        future["mortgage_rate"] = rate

    forecast = model.predict(future)

    # Extract results
    historical_end = df["ds"].max()
    future_rows = forecast[forecast["ds"] > historical_end].tail(horizon)

    if future_rows.empty:
        return None

    start_price = float(df["y"].iloc[-1])
    end_price = float(future_rows["yhat"].iloc[-1])
    change_pct = ((end_price - start_price) / start_price) * 100 if start_price > 0 else 0

    # Build weekly series for the frontend chart
    weekly = future_rows.iloc[::7]  # sample every 7 days
    series = []
    for _, row in weekly.iterrows():
        series.append({
            "date": row["ds"].strftime("%Y-%m-%d"),
            "y": round(float(row["yhat"])),
            "lower": round(float(row["yhat_lower"])),
            "upper": round(float(row["yhat_upper"])),
        })

    confidence = min(85, max(40, 40 + len(df) * 0.5))
    # Adjust confidence based on change magnitude
    if abs(change_pct) > 15:
        confidence *= 0.8

    score = max(0, min(100, round(50 + change_pct * 3)))
    direction = "up" if change_pct > 1 else "down" if change_pct < -1 else "stable"

    end_date = future_rows["ds"].iloc[-1].strftime("%Y-%m-%d")
    sign = "+" if change_pct >= 0 else ""
    headline = (
        f"{sign}{change_pct:.1f}% → ${end_price:,.0f} by {end_date} | "
        f"Confidence: {confidence:.0f}%"
    )

    return {
        "model_key": "price_forecast",
        "headline": headline,
        "score": round(score, 2),
        "confidence_pct": round(confidence, 2),
        "direction": direction,
        "payload": {
            "start_price": round(start_price),
            "end_price": round(end_price),
            "change_pct": round(change_pct, 2),
            "series": series,
            "mortgage_rate": None,
            "data_points": len(df),
            "horizon_days": horizon,
        },
        "model_version": "prophet-v1",
    }


def retrain() -> None:
    """Pre-train Prophet models for all monitored ZIPs."""
    from db import fetch_market_snapshots

    MONITORED_ZIPS = [
        "33470", "33411", "33401", "33413", "33418",
        "33458", "33467", "33328", "33309", "33063",
    ]

    for zip_code in MONITORED_ZIPS:
        try:
            df = _build_dataframe(zip_code)
            if df is not None:
                _fit_prophet(zip_code, df)
                log.info(f"[price_forecast] Retrained for {zip_code} ({len(df)} pts)")
        except Exception as e:
            log.warning(f"[price_forecast] Retrain failed for {zip_code}: {e}")
