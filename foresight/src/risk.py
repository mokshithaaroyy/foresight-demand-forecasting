"""
Project FORESIGHT — Stockout / overstock risk scoring (D4)

Converts the demand forecast (D3) plus each SKU's current stock position
into a decision the ops team can act on today, following brief §08 exactly:

  Stockout risk  — forecast demand over the SKU's own replenishment lead
                    time, compared against on-hand + on-order stock. We
                    treat the forecast's 80% interval as an estimate of
                    uncertainty and compute an actual probability of
                    running out (not just a threshold flag): assuming
                    demand over the lead time is approximately normal with
                    mean = summed point forecast and std backed out from the
                    q10/q90 interval width, stockout risk =
                    P(demand over lead time > available stock).

  Overstock risk — current weeks-of-cover (on-hand stock / average forecast
                    weekly demand) compared against a "healthy" target cover
                    (lead time + a safety buffer). Risk ramps linearly from
                    0 at the healthy target to 1 once holding 8+ weeks more
                    than that target — a transparent, explainable rule, not
                    a black box.

Every SKU is placed on the same stockout-vs-overstock grid as brief Figure 6
and given one of four recommended actions. Rupee value at stake is computed
two ways and reconciles with the grid:
  * sales_at_risk        — expected lost revenue if the stockout risk
                            materialises (stockout_risk x demand-over-lead-time x price)
  * excess_capital_at_risk — capital tied up in stock beyond the healthy
                            cover target (at cost)

Run:  python3 src/risk.py   (after src/pipeline.py and src/forecast.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

SAFETY_BUFFER_WEEKS = 2.0     # added to lead time to define "healthy" cover
OVERSTOCK_SATURATION_WEEKS = 8.0  # weeks of *excess* cover at which overstock_risk saturates to 1
RISK_QUADRANT_THRESHOLD = 0.5

QUADRANT_ACTIONS = {
    "Reorder now": "Raise a replenishment order before stock runs out.",
    "Markdown / clear": "Promote or discount to free up capital.",
    "Watch / volatile": "Investigate — demand is erratic; review manually.",
    "Healthy": "No action needed; leave as is.",
}


def _classify_quadrant(stockout_risk: float, overstock_risk: float) -> str:
    hi_stock = stockout_risk >= RISK_QUADRANT_THRESHOLD
    hi_over = overstock_risk >= RISK_QUADRANT_THRESHOLD
    if hi_stock and not hi_over:
        return "Reorder now"
    if hi_over and not hi_stock:
        return "Markdown / clear"
    if hi_stock and hi_over:
        return "Watch / volatile"
    return "Healthy"


def load_latest_position(master: pd.DataFrame) -> pd.DataFrame:
    cols = ["sku_id", "category", "subcategory", "unit_cost", "list_price",
            "on_hand_units", "on_order_units", "lead_time_days", "reorder_point", "date"]
    latest = (
        master.sort_values("date")
        .groupby("sku_id")[cols[1:]]
        .last()
        .reset_index()
    )
    return latest


def score_risk(forecast: pd.DataFrame, latest_position: pd.DataFrame) -> pd.DataFrame:
    horizon = forecast["horizon"].max()

    # Per-SKU forecast summary across the full horizon
    agg = forecast.groupby("sku_id").agg(
        total_forecast=("forecast", "sum"),
        avg_weekly_forecast=("forecast", "mean"),
        total_low80=("forecast_low80", "sum"),
        total_high80=("forecast_high80", "sum"),
        low_confidence=("low_confidence", "max"),
    ).reset_index()

    df = latest_position.merge(agg, on="sku_id", how="left")
    df[["total_forecast", "avg_weekly_forecast", "total_low80", "total_high80"]] = \
        df[["total_forecast", "avg_weekly_forecast", "total_low80", "total_high80"]].fillna(0.0)

    df["lead_time_weeks"] = (df["lead_time_days"].fillna(df["lead_time_days"].median()) / 7.0).clip(lower=0.5)
    lt_weeks = df["lead_time_weeks"].clip(upper=horizon)  # can't estimate demand beyond the forecast horizon

    # --- Stockout risk: probability demand-over-lead-time exceeds available stock ---
    demand_over_lt = df["avg_weekly_forecast"] * lt_weeks
    # scale the horizon-level 80% interval down to a single week's sigma, then to the
    # lead-time window, assuming per-week forecast errors are roughly independent
    # (sigma scales with sqrt(n_weeks)).
    horizon_n = max(int(horizon), 1)
    weekly_sigma = (df["total_high80"] - df["total_low80"]) / (2 * 1.2816 * np.sqrt(horizon_n))
    sigma_over_lt = weekly_sigma * np.sqrt(lt_weeks.clip(lower=0.1))
    sigma_over_lt = sigma_over_lt.replace(0, np.nan)

    available = df["on_hand_units"].fillna(0) + df["on_order_units"].fillna(0)
    z = (demand_over_lt - available) / sigma_over_lt
    stockout_risk = norm.cdf(z.fillna(0))
    # SKUs with essentially zero forecast demand and zero uncertainty -> no stockout risk
    stockout_risk = np.where(sigma_over_lt.isna() & (available >= demand_over_lt), 0.0, stockout_risk)
    stockout_risk = np.where(sigma_over_lt.isna() & (available < demand_over_lt), 1.0, stockout_risk)
    df["stockout_risk"] = np.clip(stockout_risk, 0, 1)

    # --- Overstock risk: weeks-of-cover vs healthy target ---
    weeks_of_cover = np.where(df["avg_weekly_forecast"] > 0.01,
                               df["on_hand_units"].fillna(0) / df["avg_weekly_forecast"].replace(0, np.nan),
                               999.0)
    df["weeks_of_cover"] = weeks_of_cover
    target_cover_weeks = df["lead_time_weeks"] + SAFETY_BUFFER_WEEKS
    df["target_cover_weeks"] = target_cover_weeks
    excess_weeks = np.clip(df["weeks_of_cover"] - target_cover_weeks, 0, None)
    df["overstock_risk"] = np.clip(excess_weeks / OVERSTOCK_SATURATION_WEEKS, 0, 1)

    # --- Rupee impact ---
    df["sales_at_risk"] = df["stockout_risk"] * demand_over_lt * df["list_price"]
    target_stock_units = target_cover_weeks * df["avg_weekly_forecast"]
    excess_units = np.clip(df["on_hand_units"].fillna(0) - target_stock_units, 0, None)
    df["excess_capital_at_risk"] = excess_units * df["unit_cost"]
    df["on_hand_value"] = df["on_hand_units"].fillna(0) * df["unit_cost"]
    df["revenue_at_stake"] = df["sales_at_risk"] + df["excess_capital_at_risk"]

    df["quadrant"] = [
        _classify_quadrant(s, o) for s, o in zip(df["stockout_risk"], df["overstock_risk"])
    ]
    df["recommended_action"] = df["quadrant"].map(QUADRANT_ACTIONS)

    return df.sort_values("revenue_at_stake", ascending=False).reset_index(drop=True)


def write_summary(scored: pd.DataFrame) -> dict:
    counts = scored["quadrant"].value_counts().to_dict()
    summary = {
        "n_skus": int(len(scored)),
        "quadrant_counts": counts,
        "total_sales_at_risk": round(float(scored["sales_at_risk"].sum()), 2),
        "total_excess_capital_at_risk": round(float(scored["excess_capital_at_risk"].sum()), 2),
        "total_on_hand_value": round(float(scored["on_hand_value"].sum()), 2),
        "flagged_reorder_now_sales_at_risk": round(
            float(scored.loc[scored["quadrant"] == "Reorder now", "sales_at_risk"].sum()), 2),
        "n_low_confidence_forecast": int(scored["low_confidence"].sum()),
    }
    with open(REPORTS_DIR / "risk_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    top_reorder = scored[scored["quadrant"] == "Reorder now"].head(10)
    top_markdown = scored[scored["quadrant"] == "Markdown / clear"].head(10)
    flagged_sales_at_risk = scored.loc[scored["quadrant"] == "Reorder now", "sales_at_risk"].sum()

    lines = [
        "# Risk Scoring Summary — Project FORESIGHT (D4)\n\n",
        f"Scored {summary['n_skus']} SKUs against the stockout-vs-overstock grid "
        "(brief §08, Figure 6).\n\n",
        "| Quadrant | SKUs | Recommended action |\n|---|---|---|\n",
    ]
    for q, action in QUADRANT_ACTIONS.items():
        lines.append(f"| {q} | {counts.get(q, 0)} | {action} |\n")
    lines += [
        f"\n**Total sales at risk from stockouts, portfolio-wide expected value:** "
        f"Rs {summary['total_sales_at_risk']:,.0f}\n\n",
        "_(This is a probability-weighted figure summed across all 200 SKUs — "
        "stockout_risk x demand-over-lead-time x price for every SKU, not just the ones "
        "flagged \"Reorder now\" below. A handful of very high-volume SKUs with even a "
        "moderate stockout probability can dominate this total; that is intentional — it is "
        "the same expected-value logic an insurer or a finance team would use, not a bug. "
        f"The narrower figure — summed only over the {counts.get('Reorder now', 0)} SKUs "
        f"actually flagged \"Reorder now\" — is Rs {flagged_sales_at_risk:,.0f}.)_\n\n",
        f"**Total capital locked in excess stock (beyond healthy cover):** "
        f"Rs {summary['total_excess_capital_at_risk']:,.0f}\n\n",
        f"**Total value of all on-hand stock (at cost):** Rs {summary['total_on_hand_value']:,.0f}\n\n",
        f"**SKUs with low-confidence forecasts (under 12 weeks of history):** {summary['n_low_confidence_forecast']}\n\n",
        "## Top 10 — Reorder now\n\n",
        top_reorder[["sku_id", "category", "stockout_risk", "sales_at_risk", "weeks_of_cover"]].to_markdown(index=False),
        "\n\n## Top 10 — Markdown / clear\n\n",
        top_markdown[["sku_id", "category", "overstock_risk", "excess_capital_at_risk", "weeks_of_cover"]].to_markdown(index=False),
        "\n",
    ]
    with open(REPORTS_DIR / "risk_summary.md", "w") as f:
        f.writelines(lines)
    return summary


def run_risk_scoring():
    master = pd.read_parquet(PROCESSED_DIR / "master_dataset.parquet")
    forecast = pd.read_parquet(PROCESSED_DIR / "forecast_future.parquet")

    latest_position = load_latest_position(master)
    scored = score_risk(forecast, latest_position)

    scored.to_parquet(PROCESSED_DIR / "risk_scores.parquet", index=False)
    scored.to_csv(PROCESSED_DIR / "risk_scores.csv", index=False)

    summary = write_summary(scored)
    print(json.dumps(summary, indent=2))
    print(f"\nWritten: {PROCESSED_DIR / 'risk_scores.parquet'}")
    print(f"Written: {REPORTS_DIR / 'risk_summary.md'}")
    return scored


if __name__ == "__main__":
    run_risk_scoring()
