import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import QUADRANT_COLORS, inr, load_forecast, load_master, load_risk, missing_data_notice, weekly_actuals  # noqa: E402

st.set_page_config(page_title="Product Details — FORESIGHT", page_icon="🔍", layout="wide")
st.title("🔍 Product Details")
st.caption("Everything about one SKU in one place — the \"what do I do about this product?\" view.")

master = load_master()
forecast = load_forecast()
risk = load_risk()
if master is None or forecast is None or risk is None:
    missing_data_notice("Product data")
    st.stop()

sku = st.selectbox("Search for a SKU", sorted(master["sku_id"].unique().tolist()))

meta = master[master["sku_id"] == sku].iloc[-1]
r = risk[risk["sku_id"] == sku]
if r.empty:
    st.warning("No risk score available for this SKU.")
    st.stop()
r = r.iloc[0]

st.subheader(f"{sku} — {meta['subcategory']} ({meta['category']})")

badge_color = QUADRANT_COLORS.get(r["quadrant"], "#868E96")
st.markdown(
    f"<span style='background-color:{badge_color};color:white;padding:4px 12px;"
    f"border-radius:12px;font-weight:600'>{r['quadrant']}</span> &nbsp; {r['recommended_action']}",
    unsafe_allow_html=True,
)

st.write("")
c1, c2, c3, c4 = st.columns(4)
c1.metric("List price", inr(meta["list_price"]))
c2.metric("On-hand stock", f"{r['on_hand_units']:,.0f} units")
c3.metric("Weeks of cover", f"{r['weeks_of_cover']:.1f}")
c4.metric("Lead time", f"{r['lead_time_days']:.0f} days")

c5, c6, c7 = st.columns(3)
c5.metric("Stockout risk", f"{r['stockout_risk']:.0%}")
c6.metric("Overstock risk", f"{r['overstock_risk']:.0%}")
c7.metric("Revenue at stake", inr(r["revenue_at_stake"]))

hist = weekly_actuals(master, sku_id=sku).sort_values("week_start")
fut = forecast[forecast["sku_id"] == sku].sort_values("week_start")

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist["week_start"], y=hist["units_sold"], mode="lines",
                          name="Actual demand", line=dict(color="#343A40")))
if not fut.empty:
    fig.add_trace(go.Scatter(
        x=pd.concat([fut["week_start"], fut["week_start"][::-1]]),
        y=pd.concat([fut["forecast_high80"], fut["forecast_low80"][::-1]]),
        fill="toself", fillcolor="rgba(76,110,245,0.18)", line=dict(color="rgba(0,0,0,0)"),
        name="80% interval", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=fut["week_start"], y=fut["forecast"], mode="lines+markers",
                              name="Forecast", line=dict(color="#4C6EF5", width=2.5)))
fig.update_layout(title="Demand history & forecast", xaxis_title="Week", yaxis_title="Units/week", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

with st.expander("Product master data"):
    st.write({
        "SKU": sku, "Category": meta["category"], "Subcategory": meta["subcategory"],
        "Launch date": str(meta["launch_date"].date()),
        "Unit cost": inr(meta["unit_cost"]), "List price": inr(meta["list_price"]),
        "On-order units": f"{r['on_order_units']:,.0f}", "Reorder point": f"{r['reorder_point']:,.0f}",
    })
