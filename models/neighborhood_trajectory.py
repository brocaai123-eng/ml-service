"""
Neighborhood Trajectory Prediction — Scikit-learn GradientBoosting.

Classifies ZIP codes into stages: Early Gentrification, Active Gentrification,
Stable, or Decline. Uses crime trends, price trends, news sentiment, and
Census demographics as features.

The GradientBoostingRegressor learns from resolved prediction_feedback to
improve the composite score over time.
"""
import logging

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

log = logging.getLogger(__name__)

_model: GradientBoostingRegressor | None = None
_feature_names = [
    "crime_trend",
    "price_trend",
    "avg_sentiment",
    "median_income",
    "owner_ratio",
]


def _trend(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    half = max(1, len(values) // 2)
    r = np.mean(values[-half:])
    o = np.mean(values[:half])
    return ((r - o) / o * 100) if o > 0 else 0.0


def _build_features(zip_code: str) -> dict:
    from db import fetch_crime_records, fetch_news_signals, fetch_market_snapshots, fetch_census_acs

    crime = fetch_crime_records(zip_code, limit=90)
    news = fetch_news_signals(zip_code, limit=30)
    snapshots = fetch_market_snapshots(zip_code, limit=90)
    census = fetch_census_acs(zip_code)

    crime_vals = [float(c.get("incident_count") or 0) for c in crime]
    price_vals = [float(s.get("median_price") or 0) for s in snapshots if float(s.get("median_price") or 0) > 0]

    sentiment_map = {"positive": 1, "buy": 1, "negative": -1, "sell": -1}
    sentiments = [sentiment_map.get(str(n.get("sentiment", "")).lower(), 0) for n in news]
    avg_sent = float(np.mean(sentiments)) if sentiments else 0.0

    return {
        "crime_trend": _trend(crime_vals),
        "price_trend": _trend(price_vals),
        "avg_sentiment": avg_sent,
        "median_income": float(census.get("median_income") or 0),
        "owner_ratio": float(census.get("owner_ratio") or 0),
        "_crime_count": len(crime),
        "_price_count": len(price_vals),
        "_news_count": len(news),
        "_census": census,
    }


def _rule_based_score(features: dict) -> float:
    composite = 0.0
    total_w = 0.0

    ct = features["crime_trend"]
    crime_signal = 1 if ct < -5 else -1 if ct > 10 else 0
    composite += crime_signal * 0.25
    total_w += 0.25

    pt = features["price_trend"]
    price_signal = 1 if pt > 3 else -1 if pt < -3 else 0
    composite += price_signal * 0.25
    total_w += 0.25

    ns = features["avg_sentiment"]
    news_signal = 1 if ns > 0.2 else -1 if ns < -0.2 else 0
    composite += news_signal * 0.20
    total_w += 0.20

    income = features["median_income"]
    ow = features["owner_ratio"]
    if income > 0 or ow > 0:
        inc_sig = 1 if income > 80000 else 0.5 if income > 50000 else 0
        own_sig = 0.5 if ow > 0.6 else -0.5 if ow < 0.3 else 0
        demo_val = (inc_sig + own_sig) / 2
        composite += demo_val * 0.30
        total_w += 0.30

    if total_w > 0:
        composite /= total_w

    return max(0, min(100, 50 + composite * 50))


def run(zip_code: str) -> dict | None:
    features = _build_features(zip_code)

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
    composite = (score - 50) / 50

    if composite > 0.5:
        stage, rec = "Active Gentrification", "BUY"
    elif composite > 0.15:
        stage, rec = "Early Gentrification", "BUY"
    elif composite > -0.15:
        stage, rec = "Stable", "HOLD"
    else:
        stage, rec = "Decline", "SELL"

    direction = "up" if composite > 0.15 else "down" if composite < -0.15 else "stable"

    confidence = 40.0
    if features["_crime_count"] > 5:
        confidence += 10
    if features["_price_count"] > 5:
        confidence += 10
    if features["_news_count"] > 3:
        confidence += 5
    if features["median_income"] > 0:
        confidence += 10
    if _model is not None:
        confidence += 10
    confidence = min(85, max(30, confidence))

    crime_dir = "Declining (positive)" if features["crime_trend"] < -5 else "Rising (negative)" if features["crime_trend"] > 10 else "Stable"
    price_dir = "Rising" if features["price_trend"] > 3 else "Declining" if features["price_trend"] < -3 else "Stable"
    demo_shift = "Upgrading" if (features["median_income"] > 80000 and features["owner_ratio"] > 0.6) else "Declining" if features["owner_ratio"] < 0.3 else "Stable"

    headline = f"{stage} | Score: {score:.0f} | Recommendation: {rec}"

    factors = [
        {"name": "Crime Trend", "signal": 1 if features["crime_trend"] < -5 else -1 if features["crime_trend"] > 10 else 0, "weight": 0.25, "detail": f"{features['crime_trend']:.1f}% change"},
        {"name": "Price Trend", "signal": 1 if features["price_trend"] > 3 else -1 if features["price_trend"] < -3 else 0, "weight": 0.25, "detail": f"{features['price_trend']:.1f}% change"},
        {"name": "News Sentiment", "signal": 1 if features["avg_sentiment"] > 0.2 else -1 if features["avg_sentiment"] < -0.2 else 0, "weight": 0.20, "detail": f"avg {features['avg_sentiment']:.2f}"},
        {"name": "Demographics", "signal": 0.5 if features["median_income"] > 80000 else 0, "weight": 0.30, "detail": f"Income: ${features['median_income']:,.0f}"},
    ]

    return {
        "model_key": "neighborhood_trajectory",
        "headline": headline,
        "score": score,
        "confidence_pct": round(confidence, 2),
        "direction": direction,
        "payload": {
            "stage": stage,
            "recommendation": rec,
            "factors": factors,
            "crime_direction": crime_dir,
            "price_direction": price_dir,
            "demographic_shift": demo_shift,
        },
        "model_version": "neighborhood-gb-v1" if _model else "neighborhood-classify-v1",
    }


def retrain() -> None:
    """Retrain GradientBoosting from resolved feedback data."""
    from db import fetch_prediction_feedback

    feedback = fetch_prediction_feedback(metric="neighborhood_score", days_back=365)
    if len(feedback) < 10:
        log.info("[neighborhood] Not enough feedback (%d rows)", len(feedback))
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
    gb = GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    gb.fit(X, y)
    _model = gb

    importances = dict(zip(_feature_names, gb.feature_importances_))
    log.info(f"[neighborhood] Retrained GB on {len(X)} samples. Importances: {importances}")
