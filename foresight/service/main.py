"""
Project FORESIGHT — Scoring service (D6)

A small FastAPI service that returns forecast + risk for a SKU (or a batch
of SKUs), backed by the outputs of src/forecast.py and src/risk.py.

Run locally:   uvicorn service.main:app --reload --port 8000
Interactive docs (auto-generated): http://localhost:8000/docs

Endpoints
---------
GET  /health                    liveness + whether scoring data is loaded
GET  /skus                      list every SKU the service can score
GET  /score/{sku_id}            forecast + risk for one SKU
POST /score/batch                forecast + risk for a list of SKUs

Bad input is handled explicitly rather than left to crash the process:
unknown SKUs return a 404 with a clear message (never a stack trace); an
empty or oversized batch request returns a 422 with a clear message; if the
underlying data hasn't been generated yet, every scoring endpoint returns a
503 explaining exactly which pipeline step to run, rather than an opaque
500.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MAX_BATCH_SIZE = 50

app = FastAPI(
    title="Project FORESIGHT — Scoring Service",
    description=(
        "Returns weekly demand forecast and stockout/overstock risk for NorthBay Living "
        "SKUs. Backed by a LightGBM forecast (or the seasonal-naive baseline, whichever "
        "won the honest backtest — see /health) and a transparent risk-scoring rule. "
        "See `reports/model_performance.md` and `reports/risk_summary.md` for methodology."
    ),
    version="1.0.0",
)

_forecast: Optional[pd.DataFrame] = None
_risk: Optional[pd.DataFrame] = None
_load_error: Optional[str] = None


def _load_data() -> None:
    global _forecast, _risk, _load_error
    try:
        forecast_path = PROCESSED_DIR / "forecast_future.parquet"
        risk_path = PROCESSED_DIR / "risk_scores.parquet"
        if not forecast_path.exists() or not risk_path.exists():
            _load_error = (
                "Scoring data not found. Run the pipeline first: "
                "`python3 src/pipeline.py && python3 src/forecast.py && python3 src/risk.py`"
            )
            return
        _forecast = pd.read_parquet(forecast_path)
        _forecast["week_start"] = pd.to_datetime(_forecast["week_start"]).dt.strftime("%Y-%m-%d")
        _risk = pd.read_parquet(risk_path)
        _load_error = None
    except Exception as e:  # noqa: BLE001 — deliberately broad: never let startup crash the service
        _load_error = f"Failed to load scoring data: {e}"


_load_data()


# ----------------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------------
class ForecastWeek(BaseModel):
    week_start: str = Field(..., description="Monday of the forecast week, YYYY-MM-DD")
    horizon: int = Field(..., description="Weeks ahead from the last known data")
    forecast: float
    forecast_low80: float = Field(..., description="Lower bound of the 80% interval")
    forecast_high80: float = Field(..., description="Upper bound of the 80% interval")


class RiskInfo(BaseModel):
    quadrant: str = Field(..., description="Reorder now | Watch / volatile | Markdown / clear | Healthy")
    recommended_action: str
    stockout_risk: float = Field(..., description="0-1 probability of stocking out over the SKU's lead time")
    overstock_risk: float = Field(..., description="0-1 severity of holding excess stock")
    weeks_of_cover: float
    on_hand_units: float
    on_order_units: float
    lead_time_days: float
    reorder_point: float
    sales_at_risk: float = Field(..., description="Expected lost revenue (Rs) if stockout risk materialises")
    excess_capital_at_risk: float = Field(..., description="Capital (Rs) tied up beyond a healthy stock level")
    revenue_at_stake: float = Field(..., description="sales_at_risk + excess_capital_at_risk")


class SKUScore(BaseModel):
    sku_id: str
    category: str
    subcategory: str
    low_confidence_forecast: bool = Field(..., description="True if under 12 weeks of sales history")
    forecast: list[ForecastWeek]
    risk: RiskInfo


class BatchRequest(BaseModel):
    sku_ids: list[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE,
                                description=f"1-{MAX_BATCH_SIZE} SKU ids")


class BatchError(BaseModel):
    sku_id: str
    error: str


class BatchResponse(BaseModel):
    results: list[SKUScore]
    errors: list[BatchError]


class HealthResponse(BaseModel):
    status: str
    data_loaded: bool
    n_skus: int
    detail: Optional[str] = None


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _ensure_data_loaded() -> None:
    if _forecast is None or _risk is None:
        raise HTTPException(status_code=503, detail=_load_error or "Scoring data not available.")


def _score_one(sku_id: str) -> SKUScore:
    sku_id = sku_id.strip()
    risk_row = _risk[_risk["sku_id"] == sku_id]
    if risk_row.empty:
        raise KeyError(sku_id)
    r = risk_row.iloc[0]

    fc_rows = _forecast[_forecast["sku_id"] == sku_id].sort_values("horizon")
    weeks = [
        ForecastWeek(
            week_start=row["week_start"], horizon=int(row["horizon"]),
            forecast=round(float(row["forecast"]), 2),
            forecast_low80=round(float(row["forecast_low80"]), 2),
            forecast_high80=round(float(row["forecast_high80"]), 2),
        )
        for _, row in fc_rows.iterrows()
    ]

    risk_info = RiskInfo(
        quadrant=r["quadrant"], recommended_action=r["recommended_action"],
        stockout_risk=round(float(r["stockout_risk"]), 4),
        overstock_risk=round(float(r["overstock_risk"]), 4),
        weeks_of_cover=round(float(r["weeks_of_cover"]), 2),
        on_hand_units=float(r["on_hand_units"]), on_order_units=float(r["on_order_units"]),
        lead_time_days=float(r["lead_time_days"]), reorder_point=float(r["reorder_point"]),
        sales_at_risk=round(float(r["sales_at_risk"]), 2),
        excess_capital_at_risk=round(float(r["excess_capital_at_risk"]), 2),
        revenue_at_stake=round(float(r["revenue_at_stake"]), 2),
    )

    return SKUScore(
        sku_id=sku_id, category=r["category"], subcategory=r["subcategory"],
        low_confidence_forecast=bool(fc_rows["low_confidence"].iloc[0]) if not fc_rows.empty else False,
        forecast=weeks, risk=risk_info,
    )


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    loaded = _forecast is not None and _risk is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        data_loaded=loaded,
        n_skus=int(_risk["sku_id"].nunique()) if loaded else 0,
        detail=_load_error,
    )


@app.get("/skus", response_model=list[str], tags=["meta"])
def list_skus():
    _ensure_data_loaded()
    return sorted(_risk["sku_id"].unique().tolist())


@app.get("/score/{sku_id}", response_model=SKUScore, tags=["scoring"])
def score_sku(sku_id: str):
    """Forecast + risk for a single SKU. 404 with a clear message if the SKU is unknown."""
    _ensure_data_loaded()
    try:
        return _score_one(sku_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"SKU '{sku_id}' not found. GET /skus for the list of valid SKU ids.",
        )


@app.post("/score/batch", response_model=BatchResponse, tags=["scoring"])
def score_batch(request: BatchRequest):
    """
    Forecast + risk for up to {MAX_BATCH_SIZE} SKUs at once. Unknown SKUs are
    reported in `errors` rather than failing the whole batch — a request with
    one typo'd SKU still returns results for every valid one.
    """
    _ensure_data_loaded()
    results, errors = [], []
    for sku_id in request.sku_ids:
        try:
            results.append(_score_one(sku_id))
        except KeyError:
            errors.append(BatchError(sku_id=sku_id, error="SKU not found"))
    return BatchResponse(results=results, errors=errors)
