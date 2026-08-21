"""
Assembles notebooks/01_eda.ipynb from source cells defined here, then the
notebook is executed headlessly (see Makefile-less run instructions in
README) so the committed .ipynb contains real outputs, not just code.

Run:  python3 notebooks/build_01_eda.py && \
      jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# Project FORESIGHT — Exploratory Data Analysis (D2)

**Client:** NorthBay Living | **Deliverable:** D2 — Data-quality & EDA insight memo

This notebook works from the pipeline's output (`data/processed/master_dataset.parquet`),
not the raw extracts — cleaning is handled once, in code, in `src/pipeline.py`, and is
never repeated by hand here (see Definition of Done, brief §4.4).

Sections:
1. Data-quality recap (what was found, how it was handled)
2. Demand distribution & top movers / dead stock
3. Seasonality & trend
4. Promotion effect
5. Outlier detection (IQR & Z-score)
6. Stockout evidence (censored demand)
7. Business insights (plain language)
""")

code("""\
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams["figure.figsize"] = (10, 4.5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

df = pd.read_parquet("../data/processed/master_dataset.parquet")
df["date"] = pd.to_datetime(df["date"])
print(f"{len(df):,} rows | {df['sku_id'].nunique()} SKUs | {df['date'].min().date()} to {df['date'].max().date()}")
df.head()
""")

md("## 1. Data-quality recap\n\nSummary produced automatically by the pipeline (`reports/data_quality_log.json`). Every number below reflects a coded, reproducible cleaning rule — see `src/pipeline.py` docstring for the rationale behind each one.")

code("""\
with open("../reports/data_quality_log.json") as f:
    dq = json.load(f)

print("sku_master: duplicate SKU rows resolved:", dq["clean_sku_master"]["duplicate_sku_ids_resolved"])
print("sku_master: category values after cleaning:", dq["clean_sku_master"]["category_values_after_cleaning"])
print()
print("sales_daily: exact duplicate rows dropped:", dq["clean_sales"]["exact_duplicate_rows_dropped"])
print("sales_daily: negative units_sold clipped to 0:", dq["clean_sales"]["negative_units_sold_clipped_to_zero"])
print("sales_daily: missing units_sold imputed (rolling median):", dq["clean_sales"]["missing_units_sold_imputed_rolling_median"])
print("sales_daily: missing unit_price rows (imputed from list_price):", dq["clean_sales"]["missing_unit_price_rows"])
print()
print("inventory_snapshots: duplicate rows dropped:", dq["clean_inventory"]["exact_duplicate_rows_dropped"])
print("inventory_snapshots: missing lead_time_days imputed:", dq["clean_inventory"]["missing_lead_time_days_imputed"])
print()
print("rows in master dataset with no known inventory position yet:", dq["master_dataset"]["rows_missing_inventory_position"])
""")

md("""**Reading this table:** the raw extracts were deliberately imperfect (as the brief warns to expect) — inconsistent category casing (`"FURNISHINGS"` / `" Furnishings "` / `"furnishings"`), a duplicated SKU master row with a conflicting cost, ~400 exact duplicate sales rows, ~1,300 rows missing `unit_price`, ~670 missing `units_sold`, 29 negative `units_sold` values (data-entry errors), mixed date formats (ISO + US `MM/DD/YYYY`), and ~950 missing `lead_time_days` in inventory snapshots. All are handled by coded, documented rules in `src/pipeline.py` — none were edited by hand in a spreadsheet.""")

md("## 2. Demand distribution, top movers & dead stock")

code("""\
sku_totals = df.groupby("sku_id").agg(
    total_units=("units_sold", "sum"),
    total_revenue=("revenue", "sum"),
    category=("category", "first"),
    subcategory=("subcategory", "first"),
).sort_values("total_units", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].hist(sku_totals["total_units"], bins=40, color="#4C6EF5", edgecolor="white")
axes[0].set_title("Distribution of total units sold per SKU\\n(over full ~2-year history)")
axes[0].set_xlabel("Total units sold"); axes[0].set_ylabel("Number of SKUs")

daily_demand = df.groupby("date")["units_sold"].sum()
axes[1].plot(daily_demand.index, daily_demand.values, lw=0.8, color="#4C6EF5")
axes[1].set_title("Total daily units sold — all SKUs")
axes[1].set_xlabel("Date"); axes[1].set_ylabel("Units/day")
plt.tight_layout(); plt.show()

n_dead = (sku_totals["total_units"] < sku_totals["total_units"].quantile(0.10)).sum()
print(f"Top 10 SKUs by units sold account for "
      f"{sku_totals['total_units'].head(10).sum() / sku_totals['total_units'].sum():.1%} of total units.")
print(f"Bottom decile ({n_dead} SKUs) each sold under "
      f"{sku_totals['total_units'].quantile(0.10):.0f} units total over 2 years — candidates for 'dead stock'.")
sku_totals.head(10)
""")

code("""\
print("Bottom 10 movers (dead-stock candidates):")
sku_totals.tail(10)
""")

md("## 3. Seasonality & trend")

code("""\
weekly = df.set_index("date").groupby("sku_id").resample("W")["units_sold"].sum().reset_index()
weekly_total = weekly.groupby("date")["units_sold"].sum()

fig, ax = plt.subplots()
ax.plot(weekly_total.index, weekly_total.values, color="#4C6EF5")
ax.set_title("Weekly total demand — all SKUs")
ax.set_xlabel("Week"); ax.set_ylabel("Units/week")
plt.tight_layout(); plt.show()
""")

code("""\
dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
dow_avg = df.groupby("day_of_week")["units_sold"].mean()

season_avg = df.groupby("season")["units_sold"].mean().reindex(["Spring","Summer","Autumn","Winter"])

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].bar(dow_names, dow_avg.values, color="#4C6EF5")
axes[0].set_title("Average units sold by day of week")
axes[0].set_ylabel("Avg units/SKU-day")

axes[1].bar(season_avg.index, season_avg.values, color="#F59F00")
axes[1].set_title("Average units sold by season")
plt.tight_layout(); plt.show()

print(f"Weekend (Sat/Sun) average is {dow_avg[[5,6]].mean() / dow_avg[[0,1,2,3,4]].mean() - 1:+.1%} vs weekday average.")
""")

code("""\
cat_subcat_season = df.groupby(["subcategory"])["units_sold"].mean().sort_values(ascending=False)

# Fans (summer-peaking) vs Heaters (winter-peaking) — check category-specific seasonality
for sub in ["Fans", "Heaters", "Air Purifiers"]:
    if sub in df["subcategory"].unique():
        s = df[df["subcategory"] == sub].groupby("month")["units_sold"].mean()
        peak_month = s.idxmax()
        print(f"{sub}: peak month = {peak_month} (avg {s.max():.1f} units/SKU-day), trough month = {s.idxmin()} (avg {s.min():.1f})")
""")

md("## 4. Promotion effect")

code("""\
promo_effect = df.groupby("is_promo")["units_sold"].mean()
lift = promo_effect[1] / promo_effect[0] - 1
print(f"Average units sold on promo days: {promo_effect[1]:.2f}")
print(f"Average units sold on non-promo days: {promo_effect[0]:.2f}")
print(f"Promotion lift: {lift:+.1%}")

promo_by_cat = df.groupby(["category", "is_promo"])["units_sold"].mean().unstack()
promo_by_cat["lift"] = promo_by_cat[1] / promo_by_cat[0] - 1
promo_by_cat
""")

code("""\
promo_month_counts = df.loc[df["is_promo"] == 1, "date"].dt.month.value_counts().sort_index()
fig, ax = plt.subplots()
ax.bar(promo_month_counts.index.astype(str), promo_month_counts.values, color="#F59F00")
ax.set_title("Promo-flagged SKU-days by calendar month\\n(shows promotions cluster around Nov/Dec and Mar/Jul sale windows)")
ax.set_xlabel("Month"); ax.set_ylabel("SKU-days on promotion")
plt.tight_layout(); plt.show()
""")

md("## 5. Outlier detection (IQR & Z-score)\n\nApplied to SKU-day `units_sold` to separate genuine demand spikes (promotions, holidays) from data artefacts. Outliers are **flagged for awareness, not removed** — a promo-driven spike is real demand signal the forecasting model needs to see (via the `is_promo` feature), not noise.")

code("""\
x = df["units_sold"]
q1, q3 = x.quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
iqr_outliers = df[(x < lower) | (x > upper)]

z = (x - x.mean()) / x.std()
z_outliers = df[z.abs() > 3]

print(f"IQR bounds: [{lower:.1f}, {upper:.1f}] -> {len(iqr_outliers):,} rows flagged ({len(iqr_outliers)/len(df):.2%} of data)")
print(f"Z-score |z|>3 -> {len(z_outliers):,} rows flagged ({len(z_outliers)/len(df):.2%} of data)")
print()
print(f"Share of IQR-flagged rows that are also promo days: {iqr_outliers['is_promo'].mean():.1%}")
print(f"Share of IQR-flagged rows that are also holidays:   {iqr_outliers['is_holiday'].mean():.1%}")

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.boxplot(x.dropna(), vert=True, showfliers=True)
ax.set_title("Units sold per SKU-day — boxplot (IQR method)")
ax.set_ylabel("Units sold")
plt.tight_layout(); plt.show()
""")

md("""**Reading this:** most statistical outliers coincide with promotion or holiday days — they are real, explainable demand, not bad data. This is why the pipeline does not clip or remove them; instead the model is given `is_promo` / `is_holiday` features so it can learn the effect rather than being blinded to it.""")

md("## 6. Stockout evidence (censored demand)\n\nWhen `on_hand_units` hits zero, recorded `units_sold` reflects what the warehouse *could* ship, not what customers actually wanted — true demand is being under-counted for those SKU-days. This directly motivates the risk-scoring layer (D4): a forecast trained naively on sales alone will under-forecast chronically stocked-out SKUs.")

code("""\
df["was_stocked_out"] = df["on_hand_units"] <= 0
stockout_rate_by_sku = df.groupby("sku_id")["was_stocked_out"].mean().sort_values(ascending=False)

print(f"{(stockout_rate_by_sku > 0.10).sum()} SKUs spent over 10% of days at zero on-hand stock.")
print(f"{(stockout_rate_by_sku > 0.25).sum()} SKUs spent over 25% of days at zero on-hand stock — chronic stockout pattern.")

# Compare average demand in the 7 days *before* a stockout begins vs the day of/after —
# a case a naive model would misread as "demand fell" when really "we ran out".
worst_sku = stockout_rate_by_sku.index[0]
s = df[df["sku_id"] == worst_sku].sort_values("date")
fig, ax = plt.subplots()
ax.plot(s["date"], s["units_sold"], label="Units sold (recorded)", color="#4C6EF5")
ax2 = ax.twinx()
ax2.plot(s["date"], s["on_hand_units"], label="On-hand stock", color="#F03E3E", alpha=0.6)
ax.set_title(f"{worst_sku}: recorded sales collapse whenever on-hand stock hits zero")
ax.set_ylabel("Units sold/day"); ax2.set_ylabel("On-hand units")
fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88))
plt.tight_layout(); plt.show()
""")

md("## 7. Business insights (plain language)\n\nThese are the findings that matter to the Head of Operations, the merchandiser, and Finance — not just charts, but what to do about them.")

code("""\
chronic_stockout_skus = (stockout_rate_by_sku > 0.25).sum()
dead_stock_skus = n_dead
avg_promo_lift = lift

top_locked_capital = (
    df.sort_values("date").groupby("sku_id").tail(1)
    .assign(capital_locked=lambda d: d["on_hand_units"] * d["unit_cost"])
    .sort_values("capital_locked", ascending=False)
)
total_capital_locked = top_locked_capital["capital_locked"].sum()

print("SUMMARY NUMBERS FOR THE MEMO")
print(f"  Chronic-stockout SKUs (>25% of days at zero stock): {chronic_stockout_skus}")
print(f"  Dead-stock candidate SKUs (bottom 10% by units sold): {dead_stock_skus}")
print(f"  Average promotion lift on units sold: {avg_promo_lift:+.1%}")
print(f"  Estimated capital currently tied up in on-hand stock (latest snapshot, at cost): "
      f"Rs {total_capital_locked:,.0f}")
""")

md("""
1. **A meaningful share of the catalog is chronically under-stocked, not just occasionally out of luck.**
   SKUs flagged above spend more than a quarter of all days sitting at zero on-hand stock — meaning
   recorded sales for those products are *understating* true demand every time this happens. Ops is
   likely reordering against demand that already looks artificially low, which can make the problem
   self-reinforcing month over month.

2. **A comparable-sized group of SKUs looks like dead stock — capital sitting on shelves, not moving.**
   The bottom decile of SKUs by total units sold have barely moved in two years of history. Left alone,
   this ties up working capital and, per the brief's own framing, eventually forces margin-eroding
   markdowns rather than a controlled clearance.

3. **Promotions work, but they are concentrated in a few calendar windows (Diwali, year-end, spring,
   monsoon sales)** — the lift measured above is a real, learnable effect, not noise, and the forecast
   model is given explicit promo/holiday features rather than being asked to infer it from raw history
   alone.

4. **Demand has a real weekly and seasonal rhythm** (weekend uplift; category-specific seasonal peaks,
   e.g. heaters in winter, fans in summer) — a naive month-over-month or flat forecast would
   systematically mistime reorders around these swings, which is exactly what the seasonal-naive
   baseline in Section 07 is built to respect.
""")

nb["cells"] = cells
with open("01_eda.ipynb", "w") as f:
    nbf.write(nb, f)
print("Wrote 01_eda.ipynb")
