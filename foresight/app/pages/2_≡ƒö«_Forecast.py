import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_forecast, load_master, load_model_performance, missing_data_notice, weekly_actuals  # noqa: E402

st.set_page_config(page_title="Forecast — FORESIGHT", page_icon="🔮", layout="wide")
st.title("🔮 Demand Forecast")

master = load_master()
forecast = load_forecast()
perf = load_model_performance()

if master is None or forecast is None:
    missing_data_notice("Forecast data")
    st.stop()

st.sidebar.subheader("Backtested accuracy")
if perf:
    st.sidebar.metric("Model WAPE", f"{perf['pooled_model_WAPE']:.1%}")
    st.sidebar.metric("Baseline WAPE", f"{perf['pooled_baseline_WAPE']:.1%}")
    verdict = "✅ beats baseline" if perf["model_beats_baseline"] else "⚠️ baseline wins — shipped as-is, honestly"
    st.sidebar.caption(verdict)
    st.sidebar.caption(f"Shipped model: `{perf['shipped_model']}`")

sku_list = sorted(master["sku_id"].unique().tolist())
default_idx = 0
sku = st.selectbox("Choose a SKU", sku_list, index=default_idx)

hist = weekly_actuals(master, sku_id=sku).sort_values("week_start")
fut = forecast[forecast["sku_id"] == sku].sort_values("week_start")

if fut.empty:
    st.warning("No forecast available for this SKU.")
    st.stop()

low_conf = bool(fut["low_confidence"].iloc[0])
if low_conf:
    st.info("⚠️ **Low-confidence forecast** — this SKU has under 12 weeks of sales history. "
            "Treat the point forecast as directional; lean on the category pattern until more history accrues.")

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist["week_start"], y=hist["units_sold"], mode="lines",
                          name="Actual demand (history)", line=dict(color="#343A40", width=1.5)))
fig.add_trace(go.Scatter(
    x=pd.concat([fut["week_start"], fut["week_start"][::-1]]),
    y=pd.concat([fut["forecast_high80"], fut["forecast_low80"][::-1]]),
    fill="toself", fillcolor="rgba(76,110,245,0.18)", line=dict(color="rgba(0,0,0,0)"),
    name="80% interval", hoverinfo="skip",
))
fig.add_trace(go.Scatter(x=fut["week_start"], y=fut["forecast"], mode="lines+markers",
                          name="Forecast", line=dict(color="#4C6EF5", width=2.5)))
fig.add_vline(x=hist["week_start"].max(), line_dash="dot", line_color="gray")
fig.update_layout(title=f"{sku} — actual demand and {len(fut)}-week forecast",
                   xaxis_title="Week", yaxis_title="Units/week", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Forecast detail")
show = fut[["week_start", "horizon", "forecast", "forecast_low80", "forecast_high80"]].copy()
show.columns = ["Week", "Weeks ahead", "Forecast", "Low (80%)", "High (80%)"]
for c in ["Forecast", "Low (80%)", "High (80%)"]:
    show[c] = show[c].round(1)
st.dataframe(show, use_container_width=True, hide_index=True)

with st.expander("How to read this chart"):
    st.markdown(
        "- The **black line** is actual recorded sales history.\n"
        "- The **blue line** is the model's point forecast for the next few weeks.\n"
        "- The **shaded band** is an 80% interval — actual demand is expected to land inside "
        "it 8 times out of 10. A wide band means less certainty; use it to decide how much "
        "buffer stock to hold, not just the single forecast number.\n"
        "- The forecast targets *recorded* sell-through demand. For SKUs that have been "
        "chronically stocked out, true underlying demand may be higher than shown here — "
        "see the Risk Dashboard and the EDA memo for that limitation."
    )
