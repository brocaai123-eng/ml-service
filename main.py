"""
BrocaAI ML Prediction Service — FastAPI application.
Runs 7 prediction models using Prophet, Scikit-learn, and GARCH.
"""
import os
import traceback
from datetime import date

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

app = FastAPI(title="BrocaAI ML Service", version="1.0.0")

AUTH_SECRET = os.environ.get("ML_AUTH_SECRET", "")

@app.on_event("startup")
async def startup_event():
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info("BrocaAI ML Service starting on port %s", os.environ.get("PORT", "8000"))


def _check_auth(authorization: str | None):
    if AUTH_SECRET and authorization != f"Bearer {AUTH_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── Response schema ─────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    model_key: str
    headline: str
    score: float
    confidence_pct: float
    direction: str
    payload: dict
    model_version: str


class PredictResponse(BaseModel):
    zip: str
    source: str
    predictions: list[PredictionResult]
    errors: list[str] = []


class RetrainResponse(BaseModel):
    success: bool
    models_retrained: list[str]
    errors: list[str] = []


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/predict/{zip_code}", response_model=PredictResponse)
async def predict(zip_code: str, authorization: str | None = Header(None)):
    _check_auth(authorization)

    if not zip_code or len(zip_code) != 5 or not zip_code.isdigit():
        raise HTTPException(status_code=400, detail="Invalid ZIP code")

    from models.price_forecast import run as run_price
    from models.population_migration import run as run_population
    from models.grid_demand import run as run_grid
    from models.neighborhood_trajectory import run as run_neighborhood
    from models.motivated_seller import run as run_seller
    from models.market_volatility import run as run_volatility
    from models.cross_industry import run as run_cross

    results: list[PredictionResult] = []
    errors: list[str] = []

    # Run independent models
    model_runners = {
        "price_forecast": run_price,
        "population_migration": run_population,
        "neighborhood_trajectory": run_neighborhood,
        "motivated_seller_agg": run_seller,
        "market_volatility": run_volatility,
    }

    model_outputs: dict[str, dict | None] = {}

    for key, runner in model_runners.items():
        try:
            result = runner(zip_code)
            if result:
                model_outputs[key] = result
                results.append(PredictionResult(**result))
        except Exception as e:
            errors.append(f"{key}: {str(e)}")
            traceback.print_exc()

    # Grid depends on population score
    pop_score = model_outputs.get("population_migration", {}).get("score")
    try:
        grid_result = run_grid(zip_code, population_score=pop_score)
        if grid_result:
            model_outputs["grid_demand"] = grid_result
            results.append(PredictionResult(**grid_result))
    except Exception as e:
        errors.append(f"grid_demand: {str(e)}")
        traceback.print_exc()

    # Cross-industry depends on all others
    try:
        cross_result = run_cross(
            price=model_outputs.get("price_forecast"),
            population=model_outputs.get("population_migration"),
            grid=model_outputs.get("grid_demand"),
            neighborhood=model_outputs.get("neighborhood_trajectory"),
            seller=model_outputs.get("motivated_seller_agg"),
            volatility=model_outputs.get("market_volatility"),
        )
        if cross_result:
            results.append(PredictionResult(**cross_result))
    except Exception as e:
        errors.append(f"cross_industry: {str(e)}")
        traceback.print_exc()

    # Persist predictions to Supabase
    today = date.today().isoformat()
    from db import upsert_prediction, write_feedback

    for r in results:
        try:
            upsert_prediction({
                "zip": zip_code,
                "model_key": r.model_key,
                "predicted_at": today,
                "horizon_days": 90,
                "headline": r.headline,
                "score": r.score,
                "confidence_pct": r.confidence_pct,
                "direction": r.direction,
                "payload": r.payload,
                "model_version": r.model_version,
            })
        except Exception as e:
            errors.append(f"upsert {r.model_key}: {str(e)}")

    # Write prediction_feedback for accuracy tracking
    feedback_map = {
        "price_forecast": ("price", lambda p: p.get("end_price")),
        "population_migration": ("population_score", lambda p: None),
        "grid_demand": ("grid_capacity_pct", lambda p: p.get("capacity_util_pct")),
        "neighborhood_trajectory": ("neighborhood_score", lambda p: None),
        "motivated_seller_agg": ("seller_avg_score", lambda p: p.get("avg_score")),
        "market_volatility": ("volatility_index", lambda p: p.get("volatility_index")),
    }

    for r in results:
        if r.model_key not in feedback_map:
            continue
        metric, value_fn = feedback_map[r.model_key]
        predicted_value = value_fn(r.payload)
        if predicted_value is None:
            predicted_value = r.score
        try:
            write_feedback({
                "zip": zip_code,
                "metric": metric,
                "model_version": r.model_version,
                "predicted_value": predicted_value,
                "confidence_score": r.confidence_pct,
                "prediction_date": today,
            })
        except Exception:
            pass

    return PredictResponse(
        zip=zip_code,
        source="ml-service",
        predictions=results,
        errors=errors,
    )


@app.post("/retrain", response_model=RetrainResponse)
async def retrain(authorization: str | None = Header(None)):
    _check_auth(authorization)

    from models.price_forecast import retrain as retrain_price
    from models.population_migration import retrain as retrain_population
    from models.grid_demand import retrain as retrain_grid
    from models.neighborhood_trajectory import retrain as retrain_neighborhood
    from models.market_volatility import retrain as retrain_volatility

    retrained: list[str] = []
    errors: list[str] = []

    trainers = {
        "price_forecast": retrain_price,
        "population_migration": retrain_population,
        "grid_demand": retrain_grid,
        "neighborhood_trajectory": retrain_neighborhood,
        "market_volatility": retrain_volatility,
    }

    for name, trainer in trainers.items():
        try:
            trainer()
            retrained.append(name)
        except Exception as e:
            errors.append(f"{name}: {str(e)}")
            traceback.print_exc()

    return RetrainResponse(
        success=len(errors) == 0,
        models_retrained=retrained,
        errors=errors,
    )
