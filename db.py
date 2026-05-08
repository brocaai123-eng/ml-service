"""
Supabase database helper for the ML service.
Reads training data and writes predictions.
"""
import os
from datetime import date, timedelta
from typing import Any

from supabase import create_client, Client

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        url = os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL") or ""
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY") or ""
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        _client = create_client(url, key)
    return _client


# ── Read helpers ────────────────────────────────────────────────────────────

def fetch_market_snapshots(zip_code: str, limit: int = 365) -> list[dict]:
    res = (
        get_client()
        .table("market_snapshots")
        .select("snapshot_date, median_price, active_listings, new_listings, avg_days_on_market")
        .eq("zip", zip_code)
        .order("snapshot_date", desc=False)
        .limit(limit)
        .execute()
    )
    return res.data or []


def fetch_crime_records(zip_code: str, limit: int = 180) -> list[dict]:
    res = (
        get_client()
        .table("crime_records")
        .select("record_date, incident_count")
        .eq("zip", zip_code)
        .order("record_date", desc=False)
        .limit(limit)
        .execute()
    )
    return res.data or []


def fetch_energy_data(state: str = "FL", limit: int = 52) -> list[dict]:
    res = (
        get_client()
        .table("energy_data")
        .select("collected_at, consumption_mwh, price_cents_kwh, generation_revenue")
        .eq("state", state)
        .order("collected_at", desc=False)
        .limit(limit)
        .execute()
    )
    return res.data or []


def fetch_news_signals(zip_code: str, limit: int = 60) -> list[dict]:
    res = (
        get_client()
        .table("news_signals")
        .select("published_at, sentiment")
        .eq("zip", zip_code)
        .order("published_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []


def fetch_properties(zip_code: str) -> list[dict]:
    res = (
        get_client()
        .table("properties")
        .select("motivated_seller_score, motivated_seller_label, motivated_seller_breakdown")
        .eq("zip", zip_code)
        .execute()
    )
    return res.data or []


def fetch_prediction_feedback(
    metric: str | None = None,
    days_back: int = 180,
) -> list[dict]:
    """Fetch resolved predictions (with actual_value) for retraining."""
    since = (date.today() - timedelta(days=days_back)).isoformat()
    q = (
        get_client()
        .table("prediction_feedback")
        .select("zip, metric, predicted_value, actual_value, confidence_score, prediction_date, model_version")
        .gte("prediction_date", since)
        .not_.is_("actual_value", "null")
        .order("prediction_date", desc=False)
        .limit(2000)
    )
    if metric:
        q = q.eq("metric", metric)
    return q.execute().data or []


# ── Write helpers ───────────────────────────────────────────────────────────

def upsert_prediction(row: dict[str, Any]) -> None:
    """Upsert into model_predictions (zip, model_key, predicted_at)."""
    client = get_client()
    existing = (
        client.table("model_predictions")
        .select("id")
        .eq("zip", row["zip"])
        .eq("model_key", row["model_key"])
        .eq("predicted_at", row["predicted_at"])
        .maybe_single()
        .execute()
    )
    if existing.data and existing.data.get("id"):
        client.table("model_predictions").update(row).eq("id", existing.data["id"]).execute()
    else:
        client.table("model_predictions").insert(row).execute()


def write_feedback(row: dict[str, Any]) -> None:
    """Insert into prediction_feedback for accuracy tracking."""
    get_client().table("prediction_feedback").insert(row).execute()


# ── External API helpers ────────────────────────────────────────────────────

def fetch_fred_mortgage_rate() -> float | None:
    """Fetch current 30-year mortgage rate from FRED."""
    import httpx

    key = os.environ.get("FRED_API_KEY", "")
    if not key:
        return None
    try:
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id=MORTGAGE30US&api_key={key}&file_type=json"
            f"&sort_order=desc&limit=1"
        )
        resp = httpx.get(url, timeout=15)
        obs = resp.json().get("observations", [])
        if obs and obs[0]["value"] != ".":
            return float(obs[0]["value"])
    except Exception:
        pass
    return None


def fetch_census_acs(zip_code: str) -> dict:
    """Fetch Census ACS demographics for a ZIP."""
    import httpx

    key = os.environ.get("CENSUS_API_KEY", "")
    if not key:
        return {}
    try:
        variables = "B01003_001E,B19013_001E,B25003_002E,B25003_003E"
        url = (
            f"https://api.census.gov/data/2022/acs/acs5"
            f"?get={variables}&for=zip%20code%20tabulation%20area:{zip_code}&key={key}"
        )
        resp = httpx.get(url, timeout=15)
        rows = resp.json()
        if not rows or len(rows) < 2:
            return {}
        d = rows[1]
        owners = int(d[2] or 0)
        renters = int(d[3] or 0)
        return {
            "population": int(d[0]) if d[0] else None,
            "median_income": int(d[1]) if d[1] else None,
            "owner_ratio": owners / (owners + renters) if (owners + renters) > 0 else None,
        }
    except Exception:
        return {}


def fetch_census_zbp(zip_code: str) -> int | None:
    """Fetch business establishment count from ZIP Business Patterns."""
    import httpx

    key = os.environ.get("CENSUS_API_KEY", "")
    if not key:
        return None
    try:
        url = f"https://api.census.gov/data/2021/zbp?get=ESTAB&for=zipcode:{zip_code}&key={key}"
        resp = httpx.get(url, timeout=15)
        rows = resp.json()
        if rows and len(rows) >= 2:
            return int(rows[1][0]) if rows[1][0] else None
    except Exception:
        pass
    return None
