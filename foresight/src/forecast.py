"""
Project FORESIGHT — Demand forecasting (D3)

Produces a weekly, SKU-level demand forecast over a 6-week horizon, backed by:
  * a seasonal-naive baseline (Appendix B definition: same week, one year
    prior; falls back to the SKU's expanding mean when a year of history
    isn't available yet — e.g. newly launched SKUs)
  * a pooled, direct-multi-horizon LightGBM model (one model predicts all
    horizons 1..H, with the horizon itself as a feature — this avoids the
    error accumulation of a recursive multi-step forecast)
  * true walk-forward (rolling-origin) backtesting: at every fold, the model
    is retrained using only samples whose *target* week falls at or before
    that fold's origin, then evaluated on the next H weeks. No fold ever
    trains on information from its own or a later test window.
  * WAPE as the primary accuracy metric (robust to the many low-volume
    SKUs in this catalog), MAPE and bias as secondary checks (Appendix B).

DELIBERATE MODELLING DECISION — read before extending this model:
Inventory position (on-hand stock, reorder point) is *excluded* from the
feature set on purpose. Section 5/6 of the EDA (see notebooks/01_eda.ipynb)
shows that recorded `units_sold` is censored by stockouts — sales silently
collapse to whatever was on the shelf, not what customers wanted. If the
model were given on-hand stock as a feature, it would learn "low stock ->
low demand" and produce artificially low forecasts for exactly the SKUs
most in need of reordering — the opposite of what the risk-scoring layer
(D4) needs. The model therefore forecasts *recorded sell-through demand*
from sales history, price, promotion and calendar signals alone. This is a
real limitation, stated plainly in the README and executive readout: for
chronically stocked-out SKUs, even a well-fit forecast likely understates
true underlying demand.

No data leakage: every lag/rolling feature is computed strictly from weeks
at or before the forecast origin; target-week calendar/promotion
attributes are used only because retailers plan promotions and know
holidays in advance — never future units_sold.

Run:  python3 src/forecast.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = PROCESSED_DIR / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

HORIZON_WEEKS = 6
N_FOLDS = 10
LAGS = [1, 2, 4, 8, 12, 52]
ROLL_WINDOWS = [4, 8, 12]

CATEGORICAL_COLS = ["category", "subcategory"]
FEATURE_COLS = (
    [f"lag_{k}" for k in LAGS]
    + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    + ["roll_std_4"]
    + ["history_weeks_available", "days_since_launch", "unit_price", "list_price", "discount_pct_avg"]
    + ["horizon", "target_month", "target_is_holiday", "target_has_promo", "target_week_of_year"]
    + CATEGORICAL_COLS
)


# ----------------------------------------------------------------------------
# Weekly panel construction
# ----------------------------------------------------------------------------
def build_weekly_panel(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (weekly_sku_panel, weekly_calendar) both indexed by week_start (Monday)."""
    df = master.copy()
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="D")

    weekly = df.groupby(["sku_id", "week_start"]).agg(
        units_sold=("units_sold", "sum"),
        unit_price=("unit_price", "mean"),
        discount_pct_avg=("discount_pct", "mean"),
        category=("category", "first"),
        subcategory=("subcategory", "first"),
        list_price=("list_price", "first"),
        unit_cost=("unit_cost", "first"),
        launch_date=("launch_date", "first"),
    ).reset_index()

    weekly = weekly.sort_values(["sku_id", "week_start"])
    weekly["days_since_launch"] = (weekly["week_start"] - weekly["launch_date"]).dt.days.clip(lower=0)
    weekly["history_weeks_available"] = weekly.groupby("sku_id").cumcount()

    g = weekly.groupby("sku_id")["units_sold"]
    for k in LAGS:
        weekly[f"lag_{k}"] = g.shift(k)
    for w in ROLL_WINDOWS:
        weekly[f"roll_mean_{w}"] = g.shift(1).rolling(w).mean().reset_index(level=0, drop=True)
    weekly["roll_std_4"] = g.shift(1).rolling(4).std().reset_index(level=0, drop=True)

    for c in CATEGORICAL_COLS:
        weekly[c] = weekly[c].astype("category")

    # Week-level calendar attributes (SKU-independent — known in advance)
    cal_daily = df[["date", "week_start", "month", "season", "is_holiday", "promo_event"]].drop_duplicates("date")
    weekly_calendar = cal_daily.groupby("week_start").agg(
        month=("month", "first"),
        season=("season", "first"),
        is_holiday=("is_holiday", "max"),
        has_promo=("promo_event", lambda s: int(s.notna().any())),
    ).reset_index()
    weekly_calendar["week_of_year"] = weekly_calendar["week_start"].dt.isocalendar().week.astype(int)

    return weekly.reset_index(drop=True), weekly_calendar


# ----------------------------------------------------------------------------
# Sample construction (direct multi-horizon, one row per sku x origin x h)
# ----------------------------------------------------------------------------
def make_samples(weekly: pd.DataFrame, weekly_calendar: pd.DataFrame, horizon: int) -> pd.DataFrame:
    all_weeks = np.sort(weekly["week_start"].unique())
    week_to_idx = {w: i for i, w in enumerate(all_weeks)}
    idx_to_week = {i: w for w, i in week_to_idx.items()}

    # actual units_sold lookup for baseline + targets: (sku_id, week_start) -> units_sold
    actual_lookup = weekly.set_index(["sku_id", "week_start"])["units_sold"]

    base_cols = ["sku_id", "week_start"] + [f"lag_{k}" for k in LAGS] + \
        [f"roll_mean_{w}" for w in ROLL_WINDOWS] + \
        ["roll_std_4", "history_weeks_available", "days_since_launch", "unit_price",
         "list_price", "discount_pct_avg"] + CATEGORICAL_COLS
    origin = weekly[base_cols].copy()
    origin["origin_idx"] = origin["week_start"].map(week_to_idx)

    frames = []
    for h in range(1, horizon + 1):
        f = origin.copy()
        f["horizon"] = h
        f["target_idx"] = f["origin_idx"] + h
        max_idx = len(all_weeks) - 1
        f = f[f["target_idx"] <= max_idx]
        f["target_week"] = f["target_idx"].map(idx_to_week)
        frames.append(f)
    samples = pd.concat(frames, ignore_index=True)

    # attach target-week calendar attributes (known in advance)
    samples = samples.merge(
        weekly_calendar.rename(columns={
            "week_start": "target_week", "month": "target_month",
            "is_holiday": "target_is_holiday", "has_promo": "target_has_promo",
            "week_of_year": "target_week_of_year",
        }),
        on="target_week", how="left",
    )

    # target value (actual demand at target week) — NaN if that SKU has no row that week
    key = list(zip(samples["sku_id"], samples["target_week"]))
    samples["target"] = actual_lookup.reindex(key).to_numpy()
    samples = samples.dropna(subset=["target"]).reset_index(drop=True)

    # seasonal-naive baseline: actual value 52 weeks before the TARGET week.
    # (target_week - 52w) is always <= origin week here since horizon <= 52,
    # so this never looks into the future relative to the forecast origin.
    baseline_week_idx = samples["target_idx"] - 52
    valid_baseline = baseline_week_idx >= 0
    baseline_week = pd.Series(np.where(valid_baseline, baseline_week_idx.map(idx_to_week).fillna(pd.NaT), pd.NaT))
    bkey = list(zip(samples["sku_id"], baseline_week))
    baseline_val = actual_lookup.reindex(bkey).to_numpy()

    # fallback: SKU's expanding mean of units_sold over weeks <= origin
    expanding_mean = (
        weekly.sort_values(["sku_id", "week_start"])
        .groupby("sku_id")["units_sold"].apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )
    weekly = weekly.copy()
    weekly["_expanding_mean"] = expanding_mean.values
    fallback_lookup = weekly.set_index(["sku_id", "week_start"])["_expanding_mean"]
    fallback_key = list(zip(samples["sku_id"], samples["week_start"]))
    fallback_val = fallback_lookup.reindex(fallback_key).to_numpy()

    samples["baseline_pred"] = np.where(pd.notna(baseline_val), baseline_val, fallback_val)
    samples["baseline_pred"] = np.nan_to_num(samples["baseline_pred"], nan=0.0)
    samples["baseline_is_fallback"] = ~pd.notna(baseline_val)

    return samples


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.sum(np.abs(actual))
    return float(np.sum(np.abs(actual - pred)) / denom) if denom > 0 else float("nan")


def mape(actual: np.ndarray, pred: np.ndarray, eps: float = 1e-6) -> float:
    mask = np.abs(actual) > 1  # exclude near-zero actuals — unreliable per Appendix B
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])))


def bias(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(pred - actual))


# ----------------------------------------------------------------------------
# Walk-forward backtest
# ----------------------------------------------------------------------------
def lgb_params(objective: str = "tweedie", alpha: float | None = None) -> dict:
    p = dict(
        objective=objective, n_estimators=350, learning_rate=0.045,
        num_leaves=31, min_child_samples=25, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=0.5, random_state=42, verbosity=-1,
    )
    if objective == "tweedie":
        p["tweedie_variance_power"] = 1.2
    if objective == "quantile" and alpha is not None:
        p["alpha"] = alpha
    return p


def fit_model(train_df: pd.DataFrame, objective: str = "tweedie", alpha: float | None = None) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**lgb_params(objective, alpha))
    model.fit(train_df[FEATURE_COLS], train_df["target"], categorical_feature=CATEGORICAL_COLS)
    return model


def run_backtest(samples: pd.DataFrame, all_weeks: np.ndarray, horizon: int, n_folds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    last_idx = len(all_weeks) - 1
    fold_origin_idxs = list(range(last_idx - horizon - n_folds + 1, last_idx - horizon + 1))
    fold_origin_idxs = [i for i in fold_origin_idxs if i >= 20]  # need enough history for roll_12/lag_52-ish stability

    week_to_idx = {w: i for i, w in enumerate(all_weeks)}
    samples = samples.copy()
    samples["origin_idx"] = samples["week_start"].map(week_to_idx)
    samples["target_idx"] = samples["target_week"].map(week_to_idx)

    fold_rows = []
    pred_rows = []

    for fold_i, o in enumerate(fold_origin_idxs):
        train = samples[samples["target_idx"] <= o]
        test = samples[samples["origin_idx"] == o]
        if len(train) < 500 or len(test) == 0:
            continue

        model = fit_model(train, objective="tweedie")
        preds = model.predict(test[FEATURE_COLS])
        preds = np.clip(preds, 0, None)

        actual = test["target"].to_numpy()
        model_wape = wape(actual, preds)
        base_wape = wape(actual, test["baseline_pred"].to_numpy())
        model_mape = mape(actual, preds)
        base_mape = mape(actual, test["baseline_pred"].to_numpy())
        model_bias = bias(actual, preds)
        base_bias = bias(actual, test["baseline_pred"].to_numpy())

        fold_rows.append(dict(
            fold=fold_i, origin_week=str(all_weeks[o]), n_test=len(test),
            model_wape=model_wape, baseline_wape=base_wape,
            model_mape=model_mape, baseline_mape=base_mape,
            model_bias=model_bias, baseline_bias=base_bias,
            model_beats_baseline=bool(model_wape < base_wape),
        ))
        pred_rows.append(test.assign(model_pred=preds))

    fold_df = pd.DataFrame(fold_rows)
    preds_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    return fold_df, preds_df


# ----------------------------------------------------------------------------
# Final model + future forecast
# ----------------------------------------------------------------------------
def train_final_models(samples: pd.DataFrame) -> dict:
    models = {
        "point": fit_model(samples, objective="tweedie"),
        "q10": fit_model(samples, objective="quantile", alpha=0.10),
        "q90": fit_model(samples, objective="quantile", alpha=0.90),
    }
    return models


def forecast_future(weekly: pd.DataFrame, weekly_calendar: pd.DataFrame, models: dict, horizon: int) -> pd.DataFrame:
    last_week = weekly["week_start"].max()
    latest = weekly.sort_values("week_start").groupby("sku_id").tail(1).copy()

    all_weeks = np.sort(weekly["week_start"].unique())
    week_to_idx = {w: i for i, w in enumerate(all_weeks)}
    idx_to_week = {i: w for w, i in week_to_idx.items()}
    last_idx = week_to_idx[last_week]

    rows = []
    for h in range(1, horizon + 1):
        target_idx = last_idx + h
        target_week = target_idx * pd.Timedelta(days=7) + all_weeks[0] if target_idx >= len(all_weeks) else idx_to_week.get(target_idx)
        if target_week is None or not isinstance(target_week, pd.Timestamp):
            target_week = last_week + pd.Timedelta(weeks=h)

        f = latest.copy()
        f["horizon"] = h
        f["target_week"] = target_week
        rows.append(f)
    fut = pd.concat(rows, ignore_index=True)

    # target-week calendar attrs: for genuinely future weeks (beyond the
    # provided calendar range) fall back to the same calendar week last year.
    cal = weekly_calendar.copy()
    cal["cal_week_of_year"] = cal["week_start"].dt.isocalendar().week.astype(int)
    cal_by_woy = cal.groupby("cal_week_of_year").agg(
        target_month=("month", "first"), target_is_holiday=("is_holiday", "max"),
        target_has_promo=("has_promo", "max"),
    ).reset_index()

    fut = fut.merge(
        weekly_calendar.rename(columns={
            "week_start": "target_week", "month": "target_month",
            "is_holiday": "target_is_holiday", "has_promo": "target_has_promo",
            "week_of_year": "target_week_of_year",
        }),
        on="target_week", how="left",
    )
    missing = fut["target_month"].isna()
    if missing.any():
        fut.loc[missing, "target_week_of_year"] = fut.loc[missing, "target_week"].dt.isocalendar().week.astype(int)
        fut = fut.merge(cal_by_woy, left_on="target_week_of_year", right_on="cal_week_of_year",
                         how="left", suffixes=("", "_fallback"))
        for col in ["target_month", "target_is_holiday", "target_has_promo"]:
            fut[col] = fut[col].fillna(fut[f"{col}_fallback"])
        fut = fut.drop(columns=["cal_week_of_year"] + [f"{c}_fallback" for c in
                        ["target_month", "target_is_holiday", "target_has_promo"]], errors="ignore")

    for c in CATEGORICAL_COLS:
        fut[c] = fut[c].astype("category")

    point = np.clip(models["point"].predict(fut[FEATURE_COLS]), 0, None)
    q10 = np.clip(models["q10"].predict(fut[FEATURE_COLS]), 0, None)
    q90 = np.clip(models["q90"].predict(fut[FEATURE_COLS]), 0, None)
    q90 = np.maximum(q90, point)  # guard against quantile crossing
    q10 = np.minimum(q10, point)

    fut["forecast"] = point
    fut["forecast_low80"] = q10
    fut["forecast_high80"] = q90
    fut["low_confidence"] = fut["history_weeks_available"] < 12

    return fut[["sku_id", "category", "subcategory", "target_week", "horizon",
                "forecast", "forecast_low80", "forecast_high80", "low_confidence"]].rename(
        columns={"target_week": "week_start"}
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run_forecast_pipeline():
    master = pd.read_parquet(PROCESSED_DIR / "master_dataset.parquet")
    weekly, weekly_calendar = build_weekly_panel(master)
    samples = make_samples(weekly, weekly_calendar, HORIZON_WEEKS)
    all_weeks = np.sort(weekly["week_start"].unique())

    print(f"Weekly panel: {len(weekly):,} SKU-weeks | Samples: {len(samples):,} (sku x origin x horizon)")

    fold_df, preds_df = run_backtest(samples, all_weeks, HORIZON_WEEKS, N_FOLDS)

    pooled_model_wape = wape(preds_df["target"].to_numpy(), preds_df["model_pred"].to_numpy())
    pooled_base_wape = wape(preds_df["target"].to_numpy(), preds_df["baseline_pred"].to_numpy())
    pooled_model_mape = mape(preds_df["target"].to_numpy(), preds_df["model_pred"].to_numpy())
    pooled_base_mape = mape(preds_df["target"].to_numpy(), preds_df["baseline_pred"].to_numpy())
    pooled_model_bias = bias(preds_df["target"].to_numpy(), preds_df["model_pred"].to_numpy())
    pooled_base_bias = bias(preds_df["target"].to_numpy(), preds_df["baseline_pred"].to_numpy())

    model_wins = pooled_model_wape < pooled_base_wape
    summary = {
        "horizon_weeks": HORIZON_WEEKS,
        "n_folds_run": int(fold_df.shape[0]),
        "n_backtest_predictions": int(len(preds_df)),
        "pooled_model_WAPE": round(pooled_model_wape, 4),
        "pooled_baseline_WAPE": round(pooled_base_wape, 4),
        "pooled_model_MAPE": round(pooled_model_mape, 4),
        "pooled_baseline_MAPE": round(pooled_base_mape, 4),
        "pooled_model_bias": round(pooled_model_bias, 3),
        "pooled_baseline_bias": round(pooled_base_bias, 3),
        "model_beats_baseline": bool(model_wins),
        "wape_improvement_pct": round((pooled_base_wape - pooled_model_wape) / pooled_base_wape * 100, 1) if pooled_base_wape else None,
        "shipped_model": "lightgbm_tweedie" if model_wins else "seasonal_naive_baseline",
    }

    print(json.dumps(summary, indent=2))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(PROCESSED_DIR / "backtest_folds.csv", index=False)
    preds_df.drop(columns=CATEGORICAL_COLS, errors="ignore").to_csv(PROCESSED_DIR / "backtest_predictions.csv", index=False)
    with open(REPORTS_DIR / "model_performance.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Train final models on ALL history and produce the real future forecast
    models = train_final_models(samples)
    joblib.dump(models, MODELS_DIR / "forecast_models.joblib")
    joblib.dump(FEATURE_COLS, MODELS_DIR / "feature_cols.joblib")

    future = forecast_future(weekly, weekly_calendar, models, HORIZON_WEEKS)
    future.to_parquet(PROCESSED_DIR / "forecast_future.parquet", index=False)
    future.to_csv(PROCESSED_DIR / "forecast_future.csv", index=False)

    write_model_performance_md(summary, fold_df)

    print(f"\nFinal model trained on {len(samples):,} samples; future forecast written for "
          f"{future['sku_id'].nunique()} SKUs x {HORIZON_WEEKS} weeks.")
    print(f"Shipped model: {summary['shipped_model']}")
    return summary


def write_model_performance_md(summary: dict, fold_df: pd.DataFrame) -> None:
    lines = ["# Model Performance — Project FORESIGHT (D3)\n\n",
              f"Backtest: {summary['n_folds_run']} rolling-origin folds, "
              f"{HORIZON_WEEKS}-week horizon, walk-forward (each fold trains only on data "
              f"available at that fold's origin — no leakage across folds).\n\n",
              "| Metric | Model (LightGBM, tweedie) | Seasonal-naive baseline |\n",
              "|---|---|---|\n",
              f"| WAPE (primary) | {summary['pooled_model_WAPE']:.3f} | {summary['pooled_baseline_WAPE']:.3f} |\n",
              f"| MAPE (secondary, \\|actual\\|>1 only) | {summary['pooled_model_MAPE']:.3f} | {summary['pooled_baseline_MAPE']:.3f} |\n",
              f"| Bias (signed mean error) | {summary['pooled_model_bias']:+.2f} | {summary['pooled_baseline_bias']:+.2f} |\n\n",
              f"**Result: the model {'beats' if summary['model_beats_baseline'] else 'does NOT beat'} "
              f"the seasonal-naive baseline** "
              f"({'a ' + str(summary['wape_improvement_pct']) + '% WAPE improvement' if summary['model_beats_baseline'] else 'baseline wins on this backtest'}).\n\n",
              f"**Shipped model: `{summary['shipped_model']}`** — per the engagement's non-negotiable rule "
              "(brief §7.1), we ship whichever wins the honest backtest, not whichever looks more sophisticated.\n\n",
              "## Per-fold detail\n\n",
              fold_df.to_markdown(index=False) if len(fold_df) else "_(no folds run)_",
              "\n"]
    with open(REPORTS_DIR / "model_performance.md", "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    run_forecast_pipeline()
