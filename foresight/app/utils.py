"""
Shared data-loading and formatting helpers for the FORESIGHT Streamlit
dashboard (D5). Every page imports from here so caching, empty/loading
states, and currency formatting stay consistent across the app.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

QUADRANT_COLORS = {
    "Reorder now": "#E03131",
    "Watch / volatile": "#F08C00",
    "Markdown / clear": "#5C7CFA",
    "Healthy": "#2F9E44",
}


def inr(x: float, decimals: int = 0) -> str:
    """Format a number as Indian Rupees with lakh/crore-friendly grouping (approx, comma-based)."""
    if pd.isna(x):
        return "—"
    sign = "-" if x < 0 else ""
    x = abs(x)
    s = f"{x:,.{decimals}f}"
    return f"{sign}Rs {s}"


def missing_data_notice(what: str) -> None:
    st.warning(
        f"**{what} not found yet.** Run the pipeline first:\n\n"
        "```bash\n"
        "python3 src/generate_data.py   # only needed once, if data/raw is empty\n"
        "python3 src/pipeline.py\n"
        "python3 src/forecast.py\n"
        "python3 src/risk.py\n"
        "```\n"
        "Then reload this page."
    )


@st.cache_data(show_spinner="Loading master dataset...")
def load_master() -> pd.DataFrame | None:
    path = PROCESSED_DIR / "master_dataset.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner="Loading forecast...")
def load_forecast() -> pd.DataFrame | None:
    path = PROCESSED_DIR / "forecast_future.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df


@st.cache_data(show_spinner="Loading risk scores...")
def load_risk() -> pd.DataFrame | None:
    path = PROCESSED_DIR / "risk_scores.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading model performance...")
def load_model_performance() -> dict | None:
    import json
    path = REPORTS_DIR / "model_performance.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading backtest predictions...")
def load_backtest_predictions() -> pd.DataFrame | None:
    path = PROCESSED_DIR / "backtest_predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["target_week"] = pd.to_datetime(df["target_week"])
    return df


def weekly_actuals(master: pd.DataFrame, sku_id: str | None = None, category: str | None = None) -> pd.DataFrame:
    df = master
    if sku_id:
        df = df[df["sku_id"] == sku_id]
    elif category and category != "All":
        df = df[df["category"] == category]
    df = df.copy()
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="D")
    return df.groupby("week_start", as_index=False)["units_sold"].sum()


def sidebar_filters(master: pd.DataFrame, key_prefix: str = "") -> tuple[str, str]:
    categories = ["All"] + sorted(master["category"].dropna().unique().tolist())
    category = st.sidebar.selectbox("Category", categories, key=f"{key_prefix}_category")

    skus = master.loc[master["category"] == category, "sku_id"] if category != "All" else master["sku_id"]
    sku_options = ["All"] + sorted(skus.dropna().unique().tolist())
    sku = st.sidebar.selectbox("SKU", sku_options, key=f"{key_prefix}_sku")
    return category, sku
