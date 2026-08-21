import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import QUADRANT_COLORS, inr, load_risk, missing_data_notice  # noqa: E402

st.set_page_config(page_title="Risk Dashboard — FORESIGHT", page_icon="⚠️", layout="wide")
st.title("⚠️ Stockout / Overstock Risk Dashboard")

risk = load_risk()
if risk is None:
    missing_data_notice("Risk scores")
    st.stop()

categories = ["All"] + sorted(risk["category"].dropna().unique().tolist())
category = st.sidebar.selectbox("Category", categories)
df = risk if category == "All" else risk[risk["category"] == category]

counts = df["quadrant"].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("🔴 Reorder now", int(counts.get("Reorder now", 0)))
c2.metric("🟠 Watch / volatile", int(counts.get("Watch / volatile", 0)))
c3.metric("🔵 Markdown / clear", int(counts.get("Markdown / clear", 0)))
c4.metric("🟢 Healthy", int(counts.get("Healthy", 0)))

st.subheader("Decisioning grid — every SKU placed on stockout vs overstock risk")
fig = px.scatter(
    df, x="overstock_risk", y="stockout_risk", size="revenue_at_stake", color="quadrant",
    color_discrete_map=QUADRANT_COLORS, hover_name="sku_id",
    hover_data={"category": True, "subcategory": True, "weeks_of_cover": ":.1f",
                "sales_at_risk": ":,.0f", "excess_capital_at_risk": ":,.0f",
                "overstock_risk": False, "stockout_risk": False},
    size_max=45,
)
fig.add_vline(x=0.5, line_dash="dot", line_color="gray")
fig.add_hline(y=0.5, line_dash="dot", line_color="gray")
fig.update_layout(
    xaxis=dict(title="Overstock risk →", range=[-0.02, 1.02]),
    yaxis=dict(title="Stockout risk →", range=[-0.02, 1.02]),
    legend_title="Quadrant",
)
fig.add_annotation(x=0.02, y=0.98, text="REORDER NOW", showarrow=False, font=dict(color="#E03131", size=12), xanchor="left")
fig.add_annotation(x=0.98, y=0.98, text="WATCH / VOLATILE", showarrow=False, font=dict(color="#F08C00", size=12), xanchor="right")
fig.add_annotation(x=0.02, y=0.02, text="HEALTHY", showarrow=False, font=dict(color="#2F9E44", size=12), xanchor="left")
fig.add_annotation(x=0.98, y=0.02, text="MARKDOWN / CLEAR", showarrow=False, font=dict(color="#5C7CFA", size=12), xanchor="right")
st.plotly_chart(fig, use_container_width=True)
st.caption("Bubble size = revenue at stake (sales at risk + excess capital at risk). "
           "Dotted lines mark the 0.5 risk threshold on each axis.")

st.subheader("Prioritised action list")
tab1, tab2, tab3 = st.tabs(["🔴 Reorder now", "🔵 Markdown / clear", "🟠 Watch / volatile"])

def _table(sub, cols, sort_col):
    if sub.empty:
        st.info("No SKUs currently in this quadrant.")
        return
    show = sub[cols].sort_values(sort_col, ascending=False).reset_index(drop=True)
    st.dataframe(show, use_container_width=True, hide_index=True)

with tab1:
    sub = df[df["quadrant"] == "Reorder now"]
    _table(sub, ["sku_id", "category", "stockout_risk", "weeks_of_cover", "sales_at_risk", "recommended_action"], "sales_at_risk")
    if not sub.empty:
        st.metric("Total sales at risk (this list)", inr(sub["sales_at_risk"].sum()))

with tab2:
    sub = df[df["quadrant"] == "Markdown / clear"]
    _table(sub, ["sku_id", "category", "overstock_risk", "weeks_of_cover", "excess_capital_at_risk", "recommended_action"], "excess_capital_at_risk")
    if not sub.empty:
        st.metric("Total capital to free up (this list)", inr(sub["excess_capital_at_risk"].sum()))

with tab3:
    sub = df[df["quadrant"] == "Watch / volatile"]
    _table(sub, ["sku_id", "category", "stockout_risk", "overstock_risk", "revenue_at_stake", "recommended_action"], "revenue_at_stake")

with st.expander("How risk is scored (transparent, not a black box)"):
    st.markdown(
        "**Stockout risk** — probability that forecast demand over the SKU's own lead time "
        "exceeds on-hand + on-order stock, using the forecast's 80% interval to estimate "
        "uncertainty.\n\n"
        "**Overstock risk** — current weeks-of-cover (on-hand ÷ average forecast weekly "
        "demand) compared to a healthy target (lead time + 2 weeks' safety buffer). Risk "
        "ramps from 0 at the target up to 1 once holding 8+ weeks more than that target.\n\n"
        "**Quadrant** — both risks compared against a 0.5 threshold, exactly as in the "
        "engagement brief's decisioning grid."
    )
