import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_master, missing_data_notice, sidebar_filters  # noqa: E402

st.set_page_config(page_title="Sales Analytics — FORESIGHT", page_icon="📈", layout="wide")
st.title("📈 Sales Analytics")

master = load_master()
if master is None:
    missing_data_notice("Master dataset")
    st.stop()

category, sku = sidebar_filters(master, key_prefix="sales")
df = master.copy()
if category != "All":
    df = df[df["category"] == category]
if sku != "All":
    df = df[df["sku_id"] == sku]

if df.empty:
    st.info("No rows match the current filter.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Total units sold", f"{df['units_sold'].sum():,.0f}")
c2.metric("Total revenue", f"Rs {df['revenue'].sum():,.0f}")
c3.metric("SKUs in view", f"{df['sku_id'].nunique():,}")

st.subheader("Weekly demand trend")
weekly = df.copy()
weekly["week_start"] = weekly["date"] - pd.to_timedelta(weekly["date"].dt.dayofweek, unit="D")
weekly_agg = weekly.groupby("week_start", as_index=False)["units_sold"].sum()
fig = px.line(weekly_agg, x="week_start", y="units_sold", markers=False)
fig.update_layout(xaxis_title="Week", yaxis_title="Units sold/week", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Demand by day of week")
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_avg = df.groupby("day_of_week")["units_sold"].mean().reindex(range(7))
    fig_dow = go.Figure(go.Bar(x=dow_names, y=dow_avg.values, marker_color="#4C6EF5"))
    fig_dow.update_layout(yaxis_title="Avg units sold/SKU-day")
    st.plotly_chart(fig_dow, use_container_width=True)

with col_b:
    st.subheader("Demand by season")
    season_avg = df.groupby("season")["units_sold"].mean().reindex(["Spring", "Summer", "Autumn", "Winter"])
    fig_season = go.Figure(go.Bar(x=season_avg.index, y=season_avg.values, marker_color="#F59F00"))
    fig_season.update_layout(yaxis_title="Avg units sold/SKU-day")
    st.plotly_chart(fig_season, use_container_width=True)

st.subheader("Promotion effect")
promo_avg = df.groupby("is_promo")["units_sold"].mean()
if len(promo_avg) == 2:
    lift = promo_avg.get(1, 0) / promo_avg.get(0, 1) - 1 if promo_avg.get(0, 0) else float("nan")
    st.metric("Promo-day lift in avg units sold", f"{lift:+.1%}" if pd.notna(lift) else "—")

st.subheader("Top movers vs dead stock")
sku_totals = df.groupby("sku_id").agg(
    total_units=("units_sold", "sum"), total_revenue=("revenue", "sum"),
    category=("category", "first"), subcategory=("subcategory", "first"),
).sort_values("total_units", ascending=False)

tab1, tab2 = st.tabs(["Top 15 movers", "Bottom 15 (dead-stock candidates)"])
with tab1:
    st.dataframe(sku_totals.head(15), use_container_width=True)
with tab2:
    st.dataframe(sku_totals.tail(15), use_container_width=True)
