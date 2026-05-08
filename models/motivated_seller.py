"""
Motivated Seller Aggregate — Statistical aggregation from properties table.

Aggregates individual property distress scores to a ZIP-level summary.
Pure statistics, no ML model needed — but benefits from more properties
being analyzed over time.
"""


def run(zip_code: str) -> dict | None:
    from db import fetch_properties

    properties = fetch_properties(zip_code)
    if not properties:
        return None

    total = len(properties)
    scores = [float(p.get("motivated_seller_score") or 0) for p in properties]
    avg_score = round(sum(scores) / total)

    high = sum(1 for p in properties if p.get("motivated_seller_label") == "HIGH")
    moderate = sum(1 for p in properties if p.get("motivated_seller_label") == "MODERATE")
    low = sum(1 for p in properties if p.get("motivated_seller_label") == "LOW")
    pct_distressed = round((high / total) * 100)

    # Aggregate top signals
    signal_counts: dict[str, int] = {}
    for p in properties:
        breakdown = p.get("motivated_seller_breakdown") or []
        if isinstance(breakdown, list):
            for b in breakdown:
                if isinstance(b, dict) and b.get("active"):
                    label = b.get("label", "Unknown")
                    signal_counts[label] = signal_counts.get(label, 0) + 1

    top_signals = sorted(signal_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_signal_name = top_signals[0][0] if top_signals else "None"

    score = max(0, min(100, avg_score))
    confidence = max(40, min(90, 40 + min(total, 100) * 0.5))
    direction = "up" if pct_distressed > 15 else "down" if pct_distressed < 5 else "stable"

    headline = (
        f"{high} motivated sellers in {zip_code} | "
        f"Avg distress: {avg_score} | Top signal: {top_signal_name}"
    )

    return {
        "model_key": "motivated_seller_agg",
        "headline": headline,
        "score": round(score, 2),
        "confidence_pct": round(confidence, 2),
        "direction": direction,
        "payload": {
            "total_properties": total,
            "high_count": high,
            "moderate_count": moderate,
            "low_count": low,
            "avg_score": avg_score,
            "pct_distressed": pct_distressed,
            "top_signals": [{"signal": s, "count": c} for s, c in top_signals],
        },
        "model_version": "seller-agg-v1",
    }
