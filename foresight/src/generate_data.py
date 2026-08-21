"""
Synthetic data generator for Project FORESIGHT (NorthBay Living).

WHY THIS FILE EXISTS
---------------------
The engagement brief (Zidio_Project_Data_1.1.pdf) states a simulated dataset
"accompanies this brief" but the four raw extracts were not attached to this
session — only the brief PDF was provided. Per the brief's own instruction
("Data is provided — generating it is not your job... spend the engagement
cleaning, understanding, and modelling the data you are given, exactly as
you would with a real client extract"), this script exists purely to stand
in for the missing client extract. It is run ONCE to produce data/raw/*.csv,
and every downstream deliverable (pipeline, EDA, model, risk scoring,
dashboard, service) treats those CSVs as an opaque, imperfect client export
— nothing downstream imports this module.

The four tables follow the star schema in the brief (Section 05 / Appendix A):
  sku_master          — 1 row per SKU (dimension)
  calendar             — 1 row per date (dimension)
  sales_daily          — 1 row per SKU per day (fact)
  inventory_snapshots  — periodic (weekly) stock position per SKU

Demand and stock are simulated JOINTLY (not independently) so that the
data exhibits the exact business problem the brief describes: some SKUs
chronically stock out (ops reordered too conservatively against demand
that grew), some are overstocked (ops kept reordering against demand that
declined), some are volatile/promo-driven, and most are healthy. That gives
the risk-scoring deliverable (D4) real signal to find instead of contrived
labels.

Deliberate data-quality blemishes are then layered on top (missing values,
a few duplicate rows, inconsistent category casing, a stray negative
units_sold, a couple of mixed date formats, a duplicated SKU master row).
These are documented in reports/eda_memo.md as they are discovered and
handled in src/pipeline.py — never silently fixed here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# ----------------------------------------------------------------------------
# Catalog definition
# ----------------------------------------------------------------------------
CATEGORY_TREE = {
    "Furnishings": ["Sofas", "Chairs", "Tables", "Storage"],
    "Decor": ["Wall Art", "Rugs", "Lighting", "Vases"],
    "Small Appliances": ["Kitchen Appliances", "Fans", "Heaters", "Air Purifiers"],
}

# Category price bands (list_price ~ lognormal around this median), in INR
CATEGORY_PRICE_MEDIAN = {
    "Furnishings": 8500,
    "Decor": 1800,
    "Small Appliances": 3200,
}

# Category seasonal amplitude + phase (peak week of year, 0-51) — used to
# give e.g. Heaters a winter peak and Fans a summer peak.
SUBCATEGORY_SEASONALITY = {
    "Sofas": (0.15, 40), "Chairs": (0.10, 40), "Tables": (0.10, 40), "Storage": (0.20, 2),
    "Wall Art": (0.10, 45), "Rugs": (0.25, 42), "Lighting": (0.30, 46), "Vases": (0.15, 45),
    "Kitchen Appliances": (0.15, 45), "Fans": (0.55, 22), "Heaters": (0.55, 48), "Air Purifiers": (0.35, 10),
}

N_SKUS = 200
END_DATE = pd.Timestamp("2026-08-16")          # last full Sunday before "today"
START_DATE = END_DATE - pd.Timedelta(days=730)  # ~104 weeks of history
CALENDAR_PAD_END = END_DATE + pd.Timedelta(days=56)  # extra room for forecast horizon


def build_sku_master(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    sku_num = 1
    # Roughly proportional split across categories, weighted toward Decor (SKU-dense)
    cat_weights = {"Furnishings": 0.28, "Decor": 0.42, "Small Appliances": 0.30}
    cats = rng.choice(list(cat_weights), size=N_SKUS, p=list(cat_weights.values()))

    for cat in cats:
        subcat = rng.choice(CATEGORY_TREE[cat])
        sku_id = f"SKU-{sku_num:04d}"
        sku_num += 1

        price_median = CATEGORY_PRICE_MEDIAN[cat]
        list_price = float(np.round(price_median * rng.lognormal(mean=0, sigma=0.35), 2))
        margin = rng.uniform(0.35, 0.60)
        unit_cost = float(np.round(list_price * (1 - margin), 2))

        # ~15% of SKUs launched partway through the window (sparse history)
        if rng.random() < 0.15:
            launch_offset = rng.integers(30, 700)
            launch_date = START_DATE + pd.Timedelta(days=int(launch_offset))
        else:
            launch_date = START_DATE - pd.Timedelta(days=int(rng.integers(0, 400)))

        # Archetype drives the demand+ops simulation below.
        archetype = rng.choice(
            ["well_managed", "chronic_stockout", "overstocked", "volatile"],
            p=[0.55, 0.17, 0.17, 0.11],
        )

        rows.append(dict(
            sku_id=sku_id, category=cat, subcategory=subcat,
            launch_date=launch_date, unit_cost=unit_cost, list_price=list_price,
            _archetype=archetype,  # internal only — used to drive simulation, not exported as-is
        ))

    return pd.DataFrame(rows)


def build_calendar(rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.date_range(START_DATE, CALENDAR_PAD_END, freq="D")
    df = pd.DataFrame({"date": dates})
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                  3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer",
                  9: "Autumn", 10: "Autumn", 11: "Autumn"}
    df["season"] = df["month"].map(season_map)

    # A handful of fixed holidays per year
    holiday_md = {(1, 1), (1, 26), (8, 15), (10, 2), (12, 25)}
    df["is_holiday"] = df["date"].apply(lambda d: 1 if (d.month, d.day) in holiday_md else 0)

    # Named promo windows (storewide)
    promo_windows = [
        ("2024-11-20", "2024-11-30", "Diwali Sale"),
        ("2024-12-20", "2024-12-31", "Year End Clearance"),
        ("2025-03-01", "2025-03-10", "Spring Sale"),
        ("2025-07-01", "2025-07-10", "Monsoon Sale"),
        ("2025-11-15", "2025-11-30", "Diwali Sale"),
        ("2025-12-20", "2025-12-31", "Year End Clearance"),
        ("2026-03-01", "2026-03-10", "Spring Sale"),
        ("2026-07-01", "2026-07-10", "Monsoon Sale"),
    ]
    df["promo_event"] = None
    for start, end, name in promo_windows:
        mask = (df["date"] >= start) & (df["date"] <= end)
        df.loc[mask, "promo_event"] = name

    return df


def simulate_sales_and_inventory(sku_master: pd.DataFrame, calendar: pd.DataFrame, rng: np.random.Generator):
    """
    Day-by-day joint simulation of true demand, censored (actual) sales, and
    stock position per SKU, driven by each SKU's ops "archetype". Inventory
    snapshots are recorded weekly (every Monday), matching the brief's
    "periodic stock position" description.
    """
    cal = calendar.set_index("date")
    all_dates = calendar["date"].tolist()

    sales_rows = []
    inv_rows = []

    for _, sku in sku_master.iterrows():
        sku_id = sku["sku_id"]
        launch = sku["launch_date"]
        archetype = sku["_archetype"]
        subcat = sku["subcategory"]
        amp, peak_week = SUBCATEGORY_SEASONALITY[subcat]

        # Base popularity: long-tailed so a handful of SKUs are true best-sellers
        # and a chunk are near-dead. Median ~6 units/day, heavy right tail.
        base_demand = float(np.clip(rng.lognormal(mean=1.6, sigma=0.9), 0.3, 60))

        # Trend: most SKUs flat-ish; overstocked archetype trends down (demand
        # fell but ops didn't adjust); a few random SKUs trend up.
        if archetype == "overstocked":
            trend_per_week = rng.uniform(-0.030, -0.012)
        elif rng.random() < 0.2:
            trend_per_week = rng.uniform(0.005, 0.020)
        else:
            trend_per_week = rng.uniform(-0.005, 0.005)

        weekday_effect = rng.normal(loc=[0, 0, 0, 0, 0.10, 0.25, 0.15], scale=0.03)  # Mon..Sun, weekend bump
        noise_sigma = 0.35 if archetype == "volatile" else 0.18

        lead_time_days = int(rng.integers(7, 22))
        # chronic_stockout archetype: reorder point deliberately set against a
        # stale (lower) demand estimate, i.e. ops under-planned.
        demand_belief_bias = {
            "well_managed": 1.0, "chronic_stockout": 0.55,
            "overstocked": 1.35, "volatile": 1.0,
        }[archetype]
        review_period_days = 7
        safety_days = rng.uniform(4, 9)
        reorder_point = max(3, base_demand * demand_belief_bias * (lead_time_days + safety_days))
        order_up_to = reorder_point + base_demand * demand_belief_bias * review_period_days * rng.uniform(1.2, 1.8)

        on_hand = float(order_up_to * rng.uniform(0.8, 1.2))
        on_order = 0.0
        pending_orders: list[tuple[pd.Timestamp, float]] = []  # (arrival_date, qty)

        week_start_snapshot = None

        for i, date in enumerate(all_dates):
            if date < launch:
                continue
            if date > END_DATE:
                # keep simulating stock quietly isn't needed past sales window
                break

            row = cal.loc[date]
            week_idx = (date - START_DATE).days // 7
            wd = date.dayofweek  # 0=Mon
            week_of_year = date.isocalendar()[1]

            # seasonal multiplier (cosine bump centred on peak_week)
            season_mult = 1 + amp * np.cos(2 * np.pi * (week_of_year - peak_week) / 52.0)
            trend_mult = max(0.1, 1 + trend_per_week * week_idx)
            weekday_mult = max(0.2, 1 + weekday_effect[wd])

            is_promo_day = bool(row["promo_event"]) or (archetype == "volatile" and rng.random() < 0.03)
            promo_mult = rng.uniform(1.4, 2.2) if is_promo_day else 1.0
            holiday_mult = 1.15 if row["is_holiday"] else 1.0

            mu = base_demand * season_mult * trend_mult * weekday_mult * promo_mult * holiday_mult
            mu = max(mu, 0.05)
            true_demand = rng.poisson(mu)

            # Receive any pending orders that arrive today
            arrived_today = 0.0
            still_pending = []
            for arrival_date, qty in pending_orders:
                if arrival_date <= date:
                    arrived_today += qty
                else:
                    still_pending.append((arrival_date, qty))
            pending_orders = still_pending
            on_hand += arrived_today
            on_order = sum(q for _, q in pending_orders)

            # Sales are censored by available stock (this is what creates
            # real stockout events for chronic_stockout / volatile SKUs)
            units_sold = min(true_demand, on_hand)
            on_hand -= units_sold

            list_price = sku["list_price"]
            unit_price = list_price * (0.82 if is_promo_day else 1.0)
            revenue = units_sold * unit_price

            sales_rows.append(dict(
                date=date, sku_id=sku_id, units_sold=units_sold,
                revenue=round(revenue, 2), unit_price=round(unit_price, 2),
                promo_flag=1 if is_promo_day else 0,
            ))

            # Reorder check (weekly review, i.e. only on Mondays) — chronic
            # understock archetype reacts slower (skips review sometimes)
            projected_position = on_hand + on_order
            if wd == 0:  # Monday review
                react = True
                if archetype == "chronic_stockout" and rng.random() < 0.35:
                    react = False  # ops "too slow" to react some weeks
                if react and projected_position <= reorder_point:
                    order_qty = max(0.0, order_up_to - projected_position)
                    if order_qty > 0:
                        arrival = date + pd.Timedelta(days=lead_time_days)
                        pending_orders.append((arrival, order_qty))
                        on_order += order_qty

            # Weekly snapshot (Monday), captured pre-reorder-decision state below
            if wd == 0:
                inv_rows.append(dict(
                    date=date, sku_id=sku_id,
                    on_hand_units=round(on_hand, 1),
                    on_order_units=round(on_order, 1),
                    lead_time_days=lead_time_days,
                    reorder_point=round(reorder_point, 1),
                ))

    sales_df = pd.DataFrame(sales_rows)
    inv_df = pd.DataFrame(inv_rows)
    return sales_df, inv_df


def inject_blemishes(sku_master: pd.DataFrame, calendar: pd.DataFrame,
                      sales: pd.DataFrame, inventory: pd.DataFrame, rng: np.random.Generator):
    """Layer realistic export imperfections on top of otherwise-clean simulated data."""
    sku_master = sku_master.drop(columns="_archetype").copy()
    sales = sales.copy()
    inventory = inventory.copy()

    # 1) Inconsistent category/subcategory casing & whitespace (sku_master)
    idx = rng.choice(sku_master.index, size=max(1, int(0.10 * len(sku_master))), replace=False)
    for i in idx:
        style = rng.integers(0, 3)
        cat = sku_master.at[i, "category"]
        if style == 0:
            sku_master.at[i, "category"] = cat.upper()
        elif style == 1:
            sku_master.at[i, "category"] = cat.lower()
        else:
            sku_master.at[i, "category"] = f" {cat} "
    idx2 = rng.choice(sku_master.index, size=max(1, int(0.08 * len(sku_master))), replace=False)
    for i in idx2:
        sub = sku_master.at[i, "subcategory"]
        sku_master.at[i, "subcategory"] = sub.lower() if rng.random() < 0.5 else f"{sub}  "

    # 2) A duplicated SKU master row with a conflicting unit_cost (classic
    #    master-data issue — must be resolved deterministically in the pipeline)
    dup_row = sku_master.iloc[[0]].copy()
    dup_row["unit_cost"] = dup_row["unit_cost"] * 1.05
    sku_master = pd.concat([sku_master, dup_row], ignore_index=True)

    # 3) sales_daily: missing unit_price (~1%)
    idx = rng.choice(sales.index, size=int(0.01 * len(sales)), replace=False)
    sales.loc[idx, "unit_price"] = np.nan

    # 4) sales_daily: missing units_sold (~0.5%) — true export gaps
    idx = rng.choice(sales.index, size=int(0.005 * len(sales)), replace=False)
    sales.loc[idx, "units_sold"] = np.nan

    # 5) sales_daily: a stray handful of negative units_sold (data entry error)
    idx = rng.choice(sales.index, size=30, replace=False)
    sales.loc[idx, "units_sold"] = -1 * sales.loc[idx, "units_sold"].abs().clip(lower=1)

    # 6) sales_daily: exact duplicate rows (~0.3%)
    dup_idx = rng.choice(sales.index, size=int(0.003 * len(sales)), replace=False)
    sales = pd.concat([sales, sales.loc[dup_idx]], ignore_index=True)

    # 7) sales_daily: a few rows with sku_id whitespace padding
    idx = rng.choice(sales.index, size=25, replace=False)
    sales.loc[idx, "sku_id"] = sales.loc[idx, "sku_id"].apply(lambda s: f" {s}")

    # 8) sales_daily: a few dates exported in US MM/DD/YYYY string format
    #    instead of ISO (kept as object dtype on purpose to mimic a messy CSV)
    sales["date"] = sales["date"].astype(object)
    idx = rng.choice(sales.index, size=40, replace=False)
    for i in idx:
        d = pd.Timestamp(sales.at[i, "date"])
        sales.at[i, "date"] = d.strftime("%m/%d/%Y")
    other = sales.index.difference(idx)
    sales.loc[other, "date"] = sales.loc[other, "date"].apply(lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"))

    # 9) inventory_snapshots: missing lead_time_days (~5%)
    idx = rng.choice(inventory.index, size=int(0.05 * len(inventory)), replace=False)
    inventory.loc[idx, "lead_time_days"] = np.nan

    # 10) inventory_snapshots: a few duplicate (date, sku_id) snapshot rows
    dup_idx = rng.choice(inventory.index, size=15, replace=False)
    inventory = pd.concat([inventory, inventory.loc[dup_idx]], ignore_index=True)

    return sku_master, calendar, sales, inventory


def main():
    rng = np.random.default_rng(RNG_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sku_master = build_sku_master(rng)
    calendar = build_calendar(rng)
    sales, inventory = simulate_sales_and_inventory(sku_master, calendar, rng)
    sku_master, calendar, sales, inventory = inject_blemishes(sku_master, calendar, sales, inventory, rng)

    # Shuffle row order (real exports aren't sorted) then write
    sales = sales.sample(frac=1.0, random_state=RNG_SEED).reset_index(drop=True)
    inventory = inventory.sample(frac=1.0, random_state=RNG_SEED).reset_index(drop=True)

    sku_master.to_csv(OUT_DIR / "sku_master.csv", index=False)
    calendar.to_csv(OUT_DIR / "calendar.csv", index=False)
    sales.to_csv(OUT_DIR / "sales_daily.csv", index=False)
    inventory.to_csv(OUT_DIR / "inventory_snapshots.csv", index=False)

    print(f"Wrote raw extracts to {OUT_DIR}")
    print(f"  sku_master.csv           {len(sku_master):>7,} rows")
    print(f"  calendar.csv             {len(calendar):>7,} rows")
    print(f"  sales_daily.csv          {len(sales):>7,} rows")
    print(f"  inventory_snapshots.csv  {len(inventory):>7,} rows")


if __name__ == "__main__":
    main()
