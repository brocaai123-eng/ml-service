"""
Cross-Industry Prediction — Weighted ensemble of all 6 individual models.

Combines outputs into a single Market Momentum Score that reflects the
combined intelligence of every data source. No ML needed — this is a
composition layer.
"""


WEIGHTS = {
    "price_forecast": 0.25,
    "neighborhood_trajectory": 0.20,
    "population_migration": 0.15,
    "market_volatility": 0.15,
    "grid_demand": 0.10,
    "motivated_seller_agg": 0.15,
}


def run(
    price: dict | None = None,
    population: dict | None = None,
    grid: dict | None = None,
    neighborhood: dict | None = None,
    seller: dict | None = None,
    volatility: dict | None = None,
) -> dict | None:
    models = {
        "price_forecast": price,
        "population_migration": population,
        "grid_demand": grid,
        "neighborhood_trajectory": neighborhood,
        "motivated_seller_agg": seller,
        "market_volatility": volatility,
    }

    available = {k: v for k, v in models.items() if v is not None}
    if not available:
        return None

    # Re-normalize weights for available models
    total_weight = sum(WEIGHTS[k] for k in available)
    if total_weight == 0:
        return None

    weighted_score = 0.0
    drivers = []

    for key, result in available.items():
        w = WEIGHTS[key] / total_weight
        s = float(result.get("score", 50))
        weighted_score += s * w
        drivers.append({
            "model": key,
            "contribution": round(w * 100, 1),
            "score": round(s, 1),
            "direction": result.get("direction", "stable"),
        })

    momentum = max(0, min(100, round(weighted_score)))

    # Sort by contribution
    drivers.sort(key=lambda d: d["contribution"], reverse=True)
    top_driver = drivers[0]["model"].replace("_", " ").title() if drivers else "N/A"

    if momentum >= 70:
        rec = "BUY"
        label = "Strong"
    elif momentum >= 55:
        rec = "BUY"
        label = "Positive"
    elif momentum >= 45:
        rec = "HOLD"
        label = "Neutral"
    elif momentum >= 30:
        rec = "HOLD"
        label = "Cautious"
    else:
        rec = "SELL"
        label = "Weak"

    direction_counts = {"up": 0, "down": 0, "stable": 0}
    for d in drivers:
        direction_counts[d["direction"]] = direction_counts.get(d["direction"], 0) + 1
    direction = max(direction_counts, key=direction_counts.get)

    confidence = min(85, max(40, 40 + len(available) * 8))

    headline = (
        f"Momentum: {momentum} ({label}) | "
        f"Top driver: {top_driver} | "
        f"Recommendation: {rec}"
    )

    return {
        "model_key": "cross_industry",
        "headline": headline,
        "score": float(momentum),
        "confidence_pct": round(confidence, 2),
        "direction": direction,
        "payload": {
            "momentum_score": momentum,
            "recommendation": rec,
            "top_drivers": drivers,
            "model_count": len(available),
        },
        "model_version": "cross-composite-v1",
    }
