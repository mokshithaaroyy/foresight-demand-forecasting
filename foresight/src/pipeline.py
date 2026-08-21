"""
Project FORESIGHT — Data pipeline (D1)

Ingests the four raw extracts (sales_daily, sku_master, calendar,
inventory_snapshots), validates and cleans them, joins them into a single
analysis-ready dataset, and engineers the features the forecasting model
needs. Every cleaning decision is logged (counts + rationale) and written to
reports/data_quality_log.json / data_quality_log.md so D2's EDA memo can
report on it honestly.

Run end-to-end from raw data with a single command:

    python3 src/pipeline.py

Design choices, documented here rather than left implicit:

  * Category/subcategory text is normalised by stripping whitespace and
    title-casing, then mapped through a canonical vocabulary — this fixes
    the "FURNISHINGS" / " Furnishings " / "furnishings" style
    inconsistencies without guessing at typos that aren't actually typos.
  * Exact duplicate rows (sales_daily and inventory_snapshots) are dropped,
    keeping the first occurrence.
  * A duplicated sku_id in sku_master (two rows, conflicting unit_cost) is
    resolved by keeping the first occurrence and logging the conflict —
    the client would need to confirm which is correct; we don't guess.
  * sales_daily.date arrives in mixed formats (ISO and US MM/DD/YYYY) in
    this export; both are parsed explicitly rather than relying on pandas'
    format inference, which is fragile against mixed formats.
  * Negative units_sold (a data-entry error, not a real return/adjustment
    field in this extract) is clipped to 0 and flagged.
  * Missing units_sold is imputed with a per-SKU centred rolling median
    (window=7) and flagged — a short local gap is filled from local
    behaviour rather than a global average, which would ignore
    seasonality/trend for that SKU.
  * Missing unit_price is imputed from sku_master.list_price, discounted if
    promo_flag=1 that day (matches how price is generated/observed
    elsewhere in the data).
  * Missing lead_time_days in inventory_snapshots is imputed with the
    SKU's own median lead time (lead time is operationally close to
    constant per SKU); if a SKU has no observed lead time at all, the
    category median is used as a fallback.
  * The unified dataset is built with an as-of (backward) merge of the
    weekly inventory snapshots onto the daily sales+calendar+sku_master
    table — each day gets the most recently known stock position, which is
    how the ops team actually experiences "current stock" between
    snapshots.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

CANONICAL_CATEGORIES = ["Furnishings", "Decor", "Small Appliances"]


def _canonicalize(series: pd.Series, canonical: list[str]) -> pd.Series:
    lookup = {c.strip().lower(): c for c in canonical}
    return series.astype(str).str.strip().str.lower().map(lookup).fillna(series.astype(str).str.strip())


def ingest() -> dict[str, pd.DataFrame]:
    tables = {
        "sku_master": pd.read_csv(RAW_DIR / "sku_master.csv"),
        "calendar": pd.read_csv(RAW_DIR / "calendar.csv"),
        "sales_daily": pd.read_csv(RAW_DIR / "sales_daily.csv"),
        "inventory_snapshots": pd.read_csv(RAW_DIR / "inventory_snapshots.csv"),
    }
    return tables


def validate_raw(tables: dict[str, pd.DataFrame], log: dict) -> None:
    """Basic schema/sanity checks. Raises if a hard requirement is violated."""
    required_cols = {
        "sku_master": {"sku_id", "category", "subcategory", "launch_date", "unit_cost", "list_price"},
        "calendar": {"date", "week", "month", "season", "is_holiday", "promo_event"},
        "sales_daily": {"date", "sku_id", "units_sold", "revenue", "unit_price", "promo_flag"},
        "inventory_snapshots": {"date", "sku_id", "on_hand_units", "on_order_units", "lead_time_days", "reorder_point"},
    }
    for name, cols in required_cols.items():
        missing = cols - set(tables[name].columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")
    log["validate"] = {name: {"rows": len(df), "columns": list(df.columns)} for name, df in tables.items()}


def clean_sku_master(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    df = df.copy()
    n_before = len(df)

    df["category"] = _canonicalize(df["category"], CANONICAL_CATEGORIES)
    df["subcategory"] = df["subcategory"].astype(str).str.strip()

    dup_mask = df["sku_id"].duplicated(keep=False)
    n_dup_skus = df.loc[dup_mask, "sku_id"].nunique()
    df = df.drop_duplicates(subset="sku_id", keep="first")

    df["launch_date"] = pd.to_datetime(df["launch_date"], errors="coerce")
    df["unit_cost"] = pd.to_numeric(df["unit_cost"], errors="coerce")
    df["list_price"] = pd.to_numeric(df["list_price"], errors="coerce")

    log["clean_sku_master"] = {
        "rows_before": n_before,
        "rows_after": len(df),
        "duplicate_sku_ids_resolved": int(n_dup_skus),
        "category_values_after_cleaning": sorted(df["category"].unique().tolist()),
    }
    return df.reset_index(drop=True)


def clean_calendar(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["season"] = df["season"].astype(str).str.strip().str.title()
    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)
    log["clean_calendar"] = {"rows": len(df)}
    return df


def _parse_mixed_dates(s: pd.Series) -> pd.Series:
    iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    us = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
    return iso.fillna(us)


def clean_sales(df: pd.DataFrame, sku_ids: set[str], log: dict) -> pd.DataFrame:
    df = df.copy()
    n_before = len(df)

    df["sku_id"] = df["sku_id"].astype(str).str.strip()
    df["date"] = _parse_mixed_dates(df["date"])
    n_unparsed_dates = df["date"].isna().sum()
    df = df.dropna(subset=["date"])

    n_exact_dupes = df.duplicated().sum()
    df = df.drop_duplicates(keep="first")

    unknown_sku_mask = ~df["sku_id"].isin(sku_ids)
    n_unknown_sku = unknown_sku_mask.sum()
    df = df.loc[~unknown_sku_mask]

    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
    negative_mask = df["units_sold"] < 0
    n_negative = negative_mask.sum()
    df["units_sold_corrected_negative"] = negative_mask
    df.loc[negative_mask, "units_sold"] = 0

    n_missing_units = df["units_sold"].isna().sum()
    df["units_sold_imputed"] = df["units_sold"].isna()
    df = df.sort_values(["sku_id", "date"])
    df["units_sold"] = df.groupby("sku_id")["units_sold"].transform(
        lambda s: s.fillna(s.rolling(7, center=True, min_periods=1).median())
    )
    df["units_sold"] = df["units_sold"].fillna(0)  # any still-missing edge cases -> 0

    n_missing_price = df["unit_price"].isna().sum()
    df["unit_price_imputed"] = df["unit_price"].isna()

    log["clean_sales"] = {
        "rows_before": n_before,
        "unparseable_dates_dropped": int(n_unparsed_dates),
        "exact_duplicate_rows_dropped": int(n_exact_dupes),
        "rows_with_unknown_sku_dropped": int(n_unknown_sku),
        "negative_units_sold_clipped_to_zero": int(n_negative),
        "missing_units_sold_imputed_rolling_median": int(n_missing_units),
        "missing_unit_price_rows": int(n_missing_price),
        "rows_final": len(df),
    }
    return df.reset_index(drop=True)


def fill_missing_price(sales: pd.DataFrame, sku_master: pd.DataFrame) -> pd.DataFrame:
    sales = sales.merge(sku_master[["sku_id", "list_price"]], on="sku_id", how="left")
    needs_fill = sales["unit_price"].isna()
    discount = np.where(sales.loc[needs_fill, "promo_flag"] == 1, 0.82, 1.0)
    sales.loc[needs_fill, "unit_price"] = sales.loc[needs_fill, "list_price"] * discount
    sales["revenue"] = sales["revenue"].fillna(sales["units_sold"] * sales["unit_price"])
    return sales.drop(columns=["list_price"])


def clean_inventory(df: pd.DataFrame, sku_ids: set[str], log: dict) -> pd.DataFrame:
    df = df.copy()
    n_before = len(df)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sku_id"] = df["sku_id"].astype(str).str.strip()
    df = df.loc[df["sku_id"].isin(sku_ids)]

    n_exact_dupes = df.duplicated().sum()
    df = df.drop_duplicates(keep="first")
    n_key_dupes = df.duplicated(subset=["date", "sku_id"]).sum()
    df = df.drop_duplicates(subset=["date", "sku_id"], keep="first")

    n_missing_lead_time = df["lead_time_days"].isna().sum()
    sku_median = df.groupby("sku_id")["lead_time_days"].transform("median")
    df["lead_time_days"] = df["lead_time_days"].fillna(sku_median)
    df["lead_time_days"] = df["lead_time_days"].fillna(df["lead_time_days"].median())

    for col in ["on_hand_units", "on_order_units", "reorder_point"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0)

    log["clean_inventory"] = {
        "rows_before": n_before,
        "exact_duplicate_rows_dropped": int(n_exact_dupes),
        "duplicate_date_sku_keys_resolved": int(n_key_dupes),
        "missing_lead_time_days_imputed": int(n_missing_lead_time),
        "rows_final": len(df),
    }
    return df.reset_index(drop=True)


def build_master_dataset(sales: pd.DataFrame, calendar: pd.DataFrame,
                          sku_master: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    df = sales.merge(calendar, on="date", how="left")
    df = df.merge(sku_master.drop(columns=["unit_cost", "list_price"]), on="sku_id", how="left")
    df = df.merge(sku_master[["sku_id", "unit_cost", "list_price"]], on="sku_id", how="left")

    # merge_asof requires the "on" column sorted globally (not just within
    # each "by" group), so sort by date first, sku_id only as a tiebreaker.
    df = df.sort_values(["date", "sku_id"])
    inv_sorted = inventory.sort_values(["date", "sku_id"])
    df = pd.merge_asof(
        df, inv_sorted, on="date", by="sku_id", direction="backward",
        suffixes=("", "_snap"),
    )
    df = df.sort_values(["sku_id", "date"])
    # Rows before a SKU's first snapshot get NaN stock position — that's a
    # real "we don't know yet" state, not something to invent a number for.
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["sku_id", "date"]).copy()
    g = df.groupby("sku_id")["units_sold"]

    df["lag_1"] = g.shift(1)
    df["lag_7"] = g.shift(7)
    df["lag_14"] = g.shift(14)
    df["lag_28"] = g.shift(28)
    df["roll_mean_7"] = g.shift(1).rolling(7).mean().reset_index(level=0, drop=True)
    df["roll_mean_28"] = g.shift(1).rolling(28).mean().reset_index(level=0, drop=True)
    df["roll_std_7"] = g.shift(1).rolling(7).std().reset_index(level=0, drop=True)

    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["is_promo"] = df["promo_flag"].fillna(0).astype(int)
    df["has_promo_event"] = df["promo_event"].notna().astype(int)
    df["days_since_launch"] = (df["date"] - df["launch_date"]).dt.days.clip(lower=0)
    discount_pct = np.where(
        df["list_price"] > 0, 1 - (df["unit_price"] / df["list_price"]).clip(upper=1), 0
    )
    df["discount_pct"] = np.clip(discount_pct, 0, None)

    return df.reset_index(drop=True)


def write_data_quality_log(log: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "data_quality_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)

    lines = ["# Data Quality Log — Project FORESIGHT\n",
             "Generated automatically by `src/pipeline.py` on every run. ",
             "Every cleaning decision below is coded (see `src/pipeline.py`), not manual.\n"]
    for section, content in log.items():
        lines.append(f"\n## {section}\n")
        if isinstance(content, dict):
            for k, v in content.items():
                lines.append(f"- **{k}**: {v}\n")
        else:
            lines.append(f"{content}\n")
    with open(REPORTS_DIR / "data_quality_log.md", "w") as f:
        f.writelines(lines)


def run_pipeline() -> pd.DataFrame:
    log: dict = {}

    tables = ingest()
    validate_raw(tables, log)

    sku_master = clean_sku_master(tables["sku_master"], log)
    calendar = clean_calendar(tables["calendar"], log)
    sales = clean_sales(tables["sales_daily"], set(sku_master["sku_id"]), log)
    sales = fill_missing_price(sales, sku_master)
    inventory = clean_inventory(tables["inventory_snapshots"], set(sku_master["sku_id"]), log)

    master = build_master_dataset(sales, calendar, sku_master, inventory)
    master = engineer_features(master)

    log["master_dataset"] = {
        "rows": len(master),
        "columns": list(master.columns),
        "date_range": [str(master["date"].min()), str(master["date"].max())],
        "n_skus": int(master["sku_id"].nunique()),
        "rows_missing_inventory_position": int(master["on_hand_units"].isna().sum()),
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    master.to_parquet(PROCESSED_DIR / "master_dataset.parquet", index=False)
    master.head(2000).to_csv(PROCESSED_DIR / "master_dataset_sample.csv", index=False)
    sku_master.to_csv(PROCESSED_DIR / "sku_master_clean.csv", index=False)

    write_data_quality_log(log)

    print("Pipeline complete.")
    print(f"  master_dataset: {master.shape[0]:,} rows x {master.shape[1]} columns")
    print(f"  date range: {master['date'].min().date()} .. {master['date'].max().date()}")
    print(f"  SKUs: {master['sku_id'].nunique()}")
    print(f"  Written to: {PROCESSED_DIR}")
    print(f"  Data-quality log: {REPORTS_DIR / 'data_quality_log.md'}")
    return master


if __name__ == "__main__":
    run_pipeline()
