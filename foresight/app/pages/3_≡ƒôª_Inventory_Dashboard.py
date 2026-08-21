import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import inr, load_master, load_risk, missing_data_notice, sidebar_filters  # noqa: E402

st.set_page_config(page_title="Inventory Dashboard — FORESIGHT", page_icon="📦", layout="wide")
st.title("📦 Inventory Dashboard")

master = load_master()
risk = load_risk()
if master is None or risk is None:
    missing_data_notice("Inventory / risk data")
    st.stop()

category, sku = sidebar_filters(master, key_prefix="inv")
df = risk.copy()
if category != "All":
    df = df[df["category"] == category]
if sku != "All":
    df = df[df["sku_id"] == sku]

if df.empty:
    st.info("No SKUs match the current filter.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("SKUs in view", f"{len(df):,}")
c2.metric("Total on-hand value (at cost)", inr(df["on_hand_value"].sum()))
c3.metric("Median weeks of cover", f"{df['weeks_of_cover'].median():.1f}")
c4.metric("SKUs below reorder point", int((df["on_hand_units"] < df["reorder_point"]).sum()))

st.subheader("Weeks of cover distribution")
fig = px.histogram(df, x="weeks_of_cover", nbins=40, color_discrete_sequence=["#4C6EF5"])
fig.add_vline(x=df["target_cover_weeks"].median(), line_dash="dot", line_color="green",
              annotation_text="typical healthy target")
fig.update_layout(xaxis_title="Weeks of cover (on-hand ÷ avg forecast weekly demand)", yaxis_title="SKU count")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Current stock position")
show_cols = ["sku_id", "category", "subcategory", "on_hand_units", "on_order_units",
             "reorder_point", "lead_time_days", "weeks_of_cover", "quadrant"]
show = df[show_cols].sort_values("weeks_of_cover")
st.dataframe(show, use_container_width=True, hide_index=True)

st.caption(
    "`on_hand_units` / `on_order_units` / `lead_time_days` / `reorder_point` reflect each "
    "SKU's most recent inventory snapshot in the provided data. `weeks_of_cover` = on-hand "
    "stock ÷ average forecast weekly demand."
)
