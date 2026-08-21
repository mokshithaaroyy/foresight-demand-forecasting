import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import QUADRANT_COLORS, inr, load_master, load_model_performance, load_risk, missing_data_notice  # noqa: E402

st.set_page_config(page_title="Executive Summary — FORESIGHT", page_icon="🧾", layout="wide")
st.title("🧾 Executive Summary")
st.caption("The rupee-impact story — for the Head of Operations and Finance.")

master = load_master()
risk = load_risk()
perf = load_model_performance()

if master is None or risk is None:
    missing_data_notice("Executive summary data")
    st.stop()

counts = risk["quadrant"].value_counts()

st.subheader("The business case in one screen")
c1, c2, c3 = st.columns(3)
c1.metric("Portfolio-wide expected sales at risk", inr(risk["sales_at_risk"].sum()),
          help="Probability-weighted, summed across all SKUs — see Risk Dashboard for methodology.")
c2.metric("Capital locked in excess stock", inr(risk["excess_capital_at_risk"].sum()))
c3.metric("Total on-hand inventory value", inr(risk["on_hand_value"].sum()))

if perf:
    c4, c5 = st.columns(2)
    c4.metric("Forecast accuracy (WAPE) vs baseline",
              f"{perf['pooled_model_WAPE']:.1%}",
              delta=f"{perf['wape_improvement_pct']:+.1f}% better than seasonal-naive" if perf["model_beats_baseline"] else "baseline shipped instead")
    c5.metric("SKUs flagged for action today",
              int(counts.get("Reorder now", 0) + counts.get("Markdown / clear", 0) + counts.get("Watch / volatile", 0)),
              help="Out of the full catalog — see Risk Dashboard for the prioritised list.")

st.divider()
col_a, col_b = st.columns([1, 1])
with col_a:
    st.subheader("Portfolio risk mix")
    pie_df = counts.rename_axis("Quadrant").reset_index(name="SKUs")
    fig = px.pie(pie_df, names="Quadrant", values="SKUs", color="Quadrant",
                 color_discrete_map=QUADRANT_COLORS, hole=0.45)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("Recommended actions")
    st.markdown(
        f"- **{int(counts.get('Reorder now', 0))} SKUs — Reorder now.** Raise replenishment "
        "orders before these run out; they carry real, quantified sales-at-risk.\n"
        f"- **{int(counts.get('Markdown / clear', 0))} SKUs — Markdown / clear.** These are "
        "holding far more stock than forecast demand justifies; promote or discount to free "
        "up working capital.\n"
        f"- **{int(counts.get('Watch / volatile', 0))} SKUs — Watch / volatile.** High risk on "
        "both fronts — demand is erratic here; review manually rather than trusting either "
        "signal alone.\n"
        f"- **{int(counts.get('Healthy', 0))} SKUs — Healthy.** No action needed."
    )

st.divider()
st.subheader("Honest limitations")
st.markdown(
    "- The forecast predicts **recorded sell-through demand**, not true unconstrained "
    "demand. SKUs that have been chronically stocked out may want *more* than the forecast "
    "shows — inventory position was deliberately excluded from the model to avoid teaching "
    "it that low stock means low demand (see `src/forecast.py` for the full rationale).\n"
    "- Rupee figures use current list price / unit cost and a lead-time-based stockout "
    "probability; they are planning estimates, not guaranteed outcomes.\n"
    "- Forecasts for recently launched SKUs (under 12 weeks of history) are marked "
    "low-confidence and should be read directionally."
)

st.caption("Full write-up: `reports/executive_readout.md`. Full model performance: "
           "`reports/model_performance.md`. Full risk detail: `reports/risk_summary.md`.")
