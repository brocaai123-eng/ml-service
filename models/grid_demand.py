"""
Grid & Infrastructure Demand Prediction — Scikit-learn LinearRegression.

Projects energy demand from consumption trends, price signals,
new development activity, and population growth pressure.
Retrains from resolved prediction_feedback when available.
"""
import logging

import numpy as np
from sklearn.linear_model import LinearRegression

log = logging.getLogger(__name__)

_model: LinearRegression | None = None
_feature_names = [
    "energy_growth_pct",
    "listing_growth_pct",
    "price_per_kwh",
    "pop_pressure",
]


def _build_features(zip_code: str, population_score: float | None = None) -> dict:
    from db import fetch_energy_data, fetch_market_snapshots

    energy = fetch_energy_data(zip_code, limit=24)
    snapshots = fetch_market_snapshots(zip_code, limit=60)

    consumption = [float(e.get("consumption_mwh") or 0) for e in energy if float(e.get("consumption_mwh") or 0) > 0]
    listings = [float(s.get("new_listings") or 0) for s in snapshots]

    def trend(vals):
        if len(vals) < 2:
            return 0.0
        half = max(1, len(vals) // 2)
        r = np.mean(vals[-half:])
        o = np.mean(vals[:half])
        return ((r - o) / o * 100) if o > 0 else 0.0

    latest_energy = energy[0] if energy else {}
    price_kwh = float(latest_energy.get("price_cents_kwh") or 0)
    pop_pressure = ((population_score or 50) - 50) * 0.05

    return {
        "energy_growth_pct": trend(consumption),
        "listing_growth_pct": trend(listings),
        "price_per_kwh": price_kwh,
        "pop_pressure": pop_pressure,
        "current_consumption": float(latest_energy.get("consumption_mwh") or 0) if latest_energy else None,
        "_energy_count": len(energy),
        "_listing_count": len(listings),
    }


def run(zip_code: str, population_score: float | None = None) -> dict | None:
    features = _build_features(zip_code, population_score)

    if features["_energy_count"] == 0:
        return None

    global _model
    if _model is not None:
        try:
            X = np.array([[features[f] for f in _feature_names]])
            capacity_pct = float(np.clip(_model.predict(X)[0], 40, 98))
        except Exception:
            capacity_pct = None

    if _model is None or capacity_pct is None:
        # Rule-based fallback
        eg = features["energy_growth_pct"]
        lg = features["listing_growth_pct"]
        pp = features["pop_pressure"]
        demand_growth = eg * 0.5 + lg * 0.2 + pp * 30
        price_stress = max(0, (features["price_per_kwh"] - 10) * 2)
        capacity_pct = max(40, min(98, 70 + price_stress + demand_growth * 0.5))
        demand_growth_pct = demand_growth
    else:
        eg = features["energy_growth_pct"]
        lg = features["listing_growth_pct"]
        pp = features["pop_pressure"]
        demand_growth_pct = eg * 0.5 + lg * 0.2 + pp * 30

    risk = "High" if capacity_pct > 85 else "Medium" if capacity_pct > 70 else "Low"
    score = max(0, min(100, round(50 + demand_growth_pct * 2 + (capacity_pct - 70) * 0.5)))
    confidence = max(35, min(80, 40 + features["_energy_count"] * 3 + (10 if features["_listing_count"] > 10 else 0)))
    if _model is not None:
        confidence = min(85, confidence + 10)

    direction = "up" if demand_growth_pct > 2 else "down" if demand_growth_pct < -2 else "stable"

    headline = (
        f"Grid at {capacity_pct:.0f}% capacity | "
        f"{'+' if demand_growth_pct >= 0 else ''}{demand_growth_pct:.1f}% demand projected | "
        f"Risk: {risk}"
    )

    factors = [
        {"name": "Energy Consumption Growth", "value": f"{features['energy_growth_pct']:.1f}%", "signal": "up" if features["energy_growth_pct"] > 2 else "stable"},
        {"name": "New Development Activity", "value": f"{features['listing_growth_pct']:.1f}%", "signal": "up" if features["listing_growth_pct"] > 5 else "stable"},
        {"name": "Price per kWh", "value": f"{features['price_per_kwh']:.1f}¢", "signal": "up" if features["price_per_kwh"] > 12 else "stable"},
        {"name": "Population Pressure", "value": str(population_score or "N/A"), "signal": "up" if (population_score or 50) > 55 else "stable"},
    ]

    return {
        "model_key": "grid_demand",
        "headline": headline,
        "score": round(score, 2),
        "confidence_pct": round(confidence, 2),
        "direction": direction,
        "payload": {
            "current_consumption": features["current_consumption"],
            "price_per_kwh": features["price_per_kwh"] or None,
            "demand_growth_pct": round(demand_growth_pct, 2),
            "capacity_util_pct": round(capacity_pct, 2),
            "risk_level": risk,
            "new_listing_growth": round(features["listing_growth_pct"], 2),
            "factors": factors,
        },
        "model_version": "grid-lr-v1" if _model else "grid-composite-v1",
    }


def retrain() -> None:
    """Retrain LinearRegression from resolved feedback data."""
    from db import fetch_prediction_feedback

    feedback = fetch_prediction_feedback(metric="grid_capacity_pct", days_back=365)
    if len(feedback) < 10:
        log.info("[grid] Not enough feedback (%d rows)", len(feedback))
        return

    X_rows, y_rows = [], []
    for row in feedback:
        try:
            feats = _build_features(row["zip"])
            X_rows.append([feats[f] for f in _feature_names])
            y_rows.append(float(row["actual_value"]))
        except Exception:
            continue

    if len(X_rows) < 10:
        return

    global _model
    X = np.array(X_rows)
    y = np.array(y_rows)
    lr = LinearRegression()
    lr.fit(X, y)
    _model = lr

    coefs = dict(zip(_feature_names, lr.coef_))
    log.info(f"[grid] Retrained LR on {len(X)} samples. Coefs: {coefs}")
