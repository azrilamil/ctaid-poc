"""CTAID POC web app and API."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.features import FEATURE_SPECS, SYNTHETIC_SAMPLES
from app import model as demo_model

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="CTAID POC",
    description="Portfolio demo of a CT atherosclerosis detection research app. Demo with synthetic data, not a medical tool.",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.mount("/static", StaticFiles(directory=STATIC), name="static")


class PredictRequest(BaseModel):
    age_years: float = Field(..., ge=30, le=90)
    systolic_bp: float = Field(..., ge=90, le=200)
    total_cholesterol: float = Field(..., ge=120, le=350)
    calcium_score: float = Field(..., ge=0, le=800)
    stenosis_pct: float = Field(..., ge=0, le=90)
    smoking: int = Field(..., ge=0, le=1)
    diabetes: int = Field(..., ge=0, le=1)


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/version.json")
def version() -> dict:
    return {"version": "20261113-mvp1"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/schema")
def schema() -> dict:
    return {
        "features": FEATURE_SPECS,
        "samples": SYNTHETIC_SAMPLES,
        "metrics": demo_model.metrics(),
        "banner": "Demo with synthetic data, not a medical tool.",
    }


@app.post("/api/predict")
def predict(body: PredictRequest) -> dict:
    try:
        return demo_model.predict(body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/manifest.webmanifest")
def manifest() -> FileResponse:
    return FileResponse(STATIC / "manifest.webmanifest", media_type="application/manifest+json")


@app.get("/favicon.svg")
def favicon() -> FileResponse:
    return FileResponse(STATIC / "favicon.svg", media_type="image/svg+xml")
