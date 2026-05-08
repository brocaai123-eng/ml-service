"""
Population & Migration Prediction — Scikit-learn RandomForest.

Combines Census ACS demographics, ZIP Business Patterns, market listing trends,
and energy consumption trends into a feature vector. A RandomForestRegressor
learns which combination of signals best predicts population growth pressure.

When insufficient resolved feedback data exists, falls back to a rule-based
composite score (same logic as the TypeScript version but with ML-ready structure).
"""
import logging

import numpy as np
from sklearn.ensemble import RandomForestRegressor

log = logging.getLogger(__name__)

_model: RandomForestRegressor | None = None
_feature_names = [
    "new_listing_trend",
    "energy_trend",
    "population",
    "median_income",
    "business_establishments",
]


def _trend_from_series(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    half = max(1, len(values) // 2)
    recent = values[-half:]
    older = values[:half]
    r_avg = np.mean(recent)
    o_avg = np.mean(older)
    if o_avg == 0:
        return 0.0
    return ((r_avg - o_avg) / o_avg) * 100


def _build_features(zip_code: str) -> dict:
    from db import (
        fetch_market_snapshots,
        fetch_energy_data,
        fetch_census_acs,
        fetch_census_zbp,
    )

    snapshots = fetch_market_snapshots(zip_code, limit=90)
    energy = fetch_energy_data(zip_code, limit=52)
    census = fetch_census_acs(zip_code)
    zbp = fetch_census_zbp(zip_code)

    new_listings = [float(s.get("new_listings") or 0) for s in snapshots]
    consumption = [float(e.get("consumption_mwh") or 0) for e in energy if float(e.get("consumption_mwh") or 0) > 0]

    return {
        "new_listing_trend": _trend_from_series(new_listings),
        "energy_trend": _trend_from_series(consumption),
        "population": float(census.get("population") or 0),
        "median_income": float(census.get("median_income") or 0),
        "business_establishments": float(zbp or 0),
        "_census": census,
        "_new_listing_trend_raw": _trend_from_series(new_listings),
        "_energy_trend_raw": _trend_from_series(consumption),
        "_data_points": len(snapshots),
    }


def _rule_based_score(features: dict) -> float:
    """Fallback composite score when no ML model is available."""
    score = 50.0
    weights_sum = 0.0

    nlt = features["new_listing_trend"]
    listing_score = max(0, min(100, 50 + nlt * 2))
    score += listing_score * 0.25
    weights_sum += 0.25

    et = features["energy_trend"]
    energy_score = max(0, min(100, 50 + et * 3))
    score += energy_score * 0.20
    weights_sum += 0.20

    pop = features["population"]
    if pop > 0:
        pop_score = 70 if pop > 50000 else 60 if pop > 20000 else 50
        score += pop_score * 0.25
        weights_sum += 0.25

    biz = features["business_establishments"]
    if biz > 0:
        biz_score = 75 if biz > 1000 else 65 if biz > 500 else 55 if biz > 100 else 45
        score += biz_score * 0.15
        weights_sum += 0.15

    income = features["median_income"]
    if income > 0:
        inc_score = 70 if income > 80000 else 60 if income > 50000 else 50
        score += inc_score * 0.15
        weights_sum += 0.15

    if weights_sum > 0:
        score /= weights_sum

    return max(0, min(100, score))


def run(zip_code: str) -> dict | None:
    features = _build_features(zip_code)

    # Try ML model first, fall back to rule-based
    global _model
    if _model is not None:
        try:
            X = np.array([[features[f] for f in _feature_names]])
            score = float(np.clip(_model.predict(X)[0], 0, 100))
        except Exception:
            score = _rule_based_score(features)
    else:
        score = _rule_based_score(features)

    score = round(score, 2)
    projected_growth = (score - 50) * 0.1
    direction = "up" if projected_growth > 0.5 else "down" if projected_growth < -0.5 else "stable"

    confidence = 45.0
    if features["_data_points"] > 5:
        confidence += 10
    if features["population"] > 0:
        confidence += 15
    if features["business_establishments"] > 0:
        confidence += 10
    confidence = min(85, max(30, confidence))
    if _model is not None:
        confidence = min(90, confidence + 10)

    dir_label = (
        "Moderate inflow expected" if direction == "up"
        else "Outflow pressure detected" if direction == "down"
        else "Stable population"
    )
    sign = "+" if projected_growth >= 0 else ""
    headline = f"{dir_label} | {sign}{projected_growth:.1f}% projected growth | Score: {score:.0f}"

    factors = [
        {"name": "New Listing Trend", "value": round(features["new_listing_trend"], 1), "weight": 0.25, "signal": "up" if features["new_listing_trend"] > 5 else "down" if features["new_listing_trend"] < -5 else "stable"},
        {"name": "Energy Consumption Trend", "value": round(features["energy_trend"], 1), "weight": 0.20, "signal": "up" if features["energy_trend"] > 2 else "stable"},
        {"name": "Census Population", "value": features["population"], "weight": 0.25, "signal": "stable"},
        {"name": "Business Establishments", "value": features["business_establishments"], "weight": 0.15, "signal": "up" if features["business_establishments"] > 500 else "stable"},
    ]

    return {
        "model_key": "population_migration",
        "headline": headline,
        "score": score,
        "confidence_pct": round(confidence, 2),
        "direction": direction,
        "payload": {
            "population": features.get("_census", {}).get("population"),
            "median_income": features.get("_census", {}).get("median_income"),
            "new_listing_trend": round(features["new_listing_trend"], 1),
            "energy_trend": round(features["energy_trend"], 1),
            "building_permits": None,
            "business_establishments": features["business_establishments"] or None,
            "factors": factors,
        },
        "model_version": "pop-rf-v1" if _model else "pop-composite-v1",
    }


def retrain() -> None:
    """Retrain RandomForest from resolved prediction_feedback data."""
    from db import fetch_prediction_feedback, fetch_market_snapshots, fetch_energy_data, fetch_census_acs, fetch_census_zbp

    feedback = fetch_prediction_feedback(metric="population_score", days_back=365)
    if len(feedback) < 10:
        log.info("[population] Not enough feedback data for ML training (%d rows), keeping rule-based", len(feedback))
        return

    X_rows, y_rows = [], []
    for row in feedback:
        z = row["zip"]
        try:
            feats = _build_features(z)
            X_rows.append([feats[f] for f in _feature_names])
            y_rows.append(float(row["actual_value"]))
        except Exception:
            continue

    if len(X_rows) < 10:
        log.info("[population] Not enough valid training samples (%d)", len(X_rows))
        return

    global _model
    X = np.array(X_rows)
    y = np.array(y_rows)
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X, y)
    _model = rf

    importances = dict(zip(_feature_names, rf.feature_importances_))
    log.info(f"[population] Retrained RF on {len(X)} samples. Importances: {importances}")
