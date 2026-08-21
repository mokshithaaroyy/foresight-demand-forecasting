"""
Project FORESIGHT — Planning Dashboard (D5)
Home page. Run with:  streamlit run app/Home.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import inr, load_forecast, load_master, load_model_performance, load_risk, missing_data_notice  # noqa: E402

st.set_page_config(page_title="FORESIGHT — NorthBay Living", page_icon="📦", layout="wide")

st.title("📦 Project FORESIGHT")
st.caption("AI-powered demand & inventory intelligence — NorthBay Living")

master = load_master()
forecast = load_forecast()
risk = load_risk()
perf = load_model_performance()

if master is None:
    missing_data_notice("Master dataset")
    st.stop()

st.markdown(
    "NorthBay Living plans ~200 SKUs on gut feel and spreadsheets today. This dashboard "
    "turns their sales, inventory, and calendar data into a weekly demand forecast and a "
    "stockout/overstock early-warning system the operations team can act on **without a "
    "data scientist in the room.**"
)

st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("SKUs tracked", f"{master['sku_id'].nunique():,}")
col2.metric("History", f"{master['date'].min().date()} → {master['date'].max().date()}")

if perf is not None:
    wape_improve = perf.get("wape_improvement_pct")
    delta = f"{wape_improve:+.1f}% vs seasonal-naive" if wape_improve is not None else None
    col3.metric("Forecast WAPE (backtest)", f"{perf['pooled_model_WAPE']:.1%}", delta=delta,
                delta_color="normal" if (wape_improve or 0) > 0 else "inverse")
else:
    col3.metric("Forecast WAPE (backtest)", "—")

if risk is not None:
    col4.metric("Sales at risk (portfolio)", inr(risk["sales_at_risk"].sum()))
else:
    col4.metric("Sales at risk (portfolio)", "—")

st.divider()

if risk is not None:
    st.subheader("Where the catalog stands right now")
    counts = risk["quadrant"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 Reorder now", int(counts.get("Reorder now", 0)))
    c2.metric("🟠 Watch / volatile", int(counts.get("Watch / volatile", 0)))
    c3.metric("🔵 Markdown / clear", int(counts.get("Markdown / clear", 0)))
    c4.metric("🟢 Healthy", int(counts.get("Healthy", 0)))
else:
    missing_data_notice("Risk scores")

st.divider()
st.subheader("Where to go")

nav_cols = st.columns(3)
nav_items = [
    ("📈 Sales Analytics", "Demand trends, seasonality, top movers and dead stock."),
    ("🔮 Forecast", "SKU-level weekly forecast with an 80% confidence interval, vs the seasonal-naive baseline."),
    ("📦 Inventory Dashboard", "Current stock position, weeks of cover, reorder points."),
    ("⚠️ Risk Dashboard", "The stockout-vs-overstock grid — where to act first."),
    ("🔍 Product Details", "Single-SKU deep dive: history, forecast, risk, and the recommended action."),
    ("🧾 Executive Summary", "The rupee-impact story for leadership."),
]
for i, (name, desc) in enumerate(nav_items):
    with nav_cols[i % 3]:
        st.markdown(f"**{name}**")
        st.caption(desc)

st.divider()
st.caption(
    "Data pipeline, forecast model, and risk scoring all re-run reproducibly from raw data "
    "with `python3 src/pipeline.py && python3 src/forecast.py && python3 src/risk.py`. "
    "See the README for full setup."
)
