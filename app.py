from pathlib import Path

import json

from datetime import date, datetime, timezone

import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from model_service import forecast_service

BASE = Path(__file__).parent

# Frontend public directory
PUBLIC_DIR = BASE / "ps26079-node/public"

# Load and bootstrap model service at FastAPI startup
forecast_service.load_model()

app = FastAPI(
    title="PS26079 Forecast Bust Detection & Confidence API",
    description=(
        "Operational AI/ML Decision-Support API for NWP Forecast Bust Prediction, "
        "Confidence Mapping, and Meteorological Uncertainty Attribution."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# FRONTEND
# ---------------------------------------------------------

@app.get("/app.jsx")
def serve_app_js():
    return FileResponse(
        PUBLIC_DIR / "app.jsx",
        media_type="application/javascript"
    )


@app.get("/dashboard")
def dashboard():
    return FileResponse(PUBLIC_DIR / "index.html")


@app.get("/")
def home():
    return FileResponse(PUBLIC_DIR / "index.html")


# ---------------------------------------------------------
# API ROOT
# ---------------------------------------------------------

@app.get("/api")
def api_root():
    status = forecast_service.get_status()

    return {
        "message": "PS26079 Operational API",
        "model_status": status["model_status"],
        "model_type": status["model_type"],
        "current_operational_date": date.today().isoformat(),
        "endpoints": [
            "/api/forecast/current",
            "/api/confidence-map",
            "/api/bust-probability",
            "/api/explain",
            "/api/error-prone-areas",
            "/api/model-info",
            "/api/review-queue",
            "/api/historical/dates",
        ],
    }


# ---------------------------------------------------------
# MODEL STATUS
# ---------------------------------------------------------

@app.get("/api/model-info")
def model_info():
    return forecast_service.get_status()


# ---------------------------------------------------------
# CURRENT FORECAST
# ---------------------------------------------------------

@app.get("/api/forecast/current")
def current_forecast():
    try:
        return forecast_service.predict_current_forecast()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ---------------------------------------------------------
# CONFIDENCE MAP
# ---------------------------------------------------------

@app.get("/api/confidence-map")
def confidence_map():
    try:
        return forecast_service.get_confidence_map()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ---------------------------------------------------------
# BUST PROBABILITY
# ---------------------------------------------------------

@app.get("/api/bust-probability")
def bust_probability():
    try:
        return forecast_service.get_bust_probability()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ---------------------------------------------------------
# EXPLANATION
# ---------------------------------------------------------

@app.get("/api/explain")
def explain():
    try:
        return forecast_service.get_explanation()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ---------------------------------------------------------
# ERROR-PRONE AREAS
# ---------------------------------------------------------

@app.get("/api/error-prone-areas")
def error_prone_areas():
    try:
        return forecast_service.get_error_prone_areas()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ---------------------------------------------------------
# REVIEW QUEUE
# ---------------------------------------------------------

@app.get("/api/review-queue")
def review_queue():
    try:
        return forecast_service.get_review_queue()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ---------------------------------------------------------
# HISTORICAL DATES
# ---------------------------------------------------------

@app.get("/api/historical/dates")
def historical_dates():
    try:
        return forecast_service.get_historical_dates()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
