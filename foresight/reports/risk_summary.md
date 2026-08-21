# Risk Scoring Summary — Project FORESIGHT (D4)

Scored 200 SKUs against the stockout-vs-overstock grid (brief §08, Figure 6).

| Quadrant | SKUs | Recommended action |
|---|---|---|
| Reorder now | 5 | Raise a replenishment order before stock runs out. |
| Markdown / clear | 44 | Promote or discount to free up capital. |
| Watch / volatile | 0 | Investigate — demand is erratic; review manually. |
| Healthy | 151 | No action needed; leave as is. |

**Total sales at risk from stockouts, portfolio-wide expected value:** Rs 11,350,410

_(This is a probability-weighted figure summed across all 200 SKUs — stockout_risk x demand-over-lead-time x price for every SKU, not just the ones flagged "Reorder now" below. A handful of very high-volume SKUs with even a moderate stockout probability can dominate this total; that is intentional — it is the same expected-value logic an insurer or a finance team would use, not a bug. The narrower figure — summed only over the 5 SKUs actually flagged "Reorder now" — is Rs 311,920.)_

**Total capital locked in excess stock (beyond healthy cover):** Rs 25,280,431

**Total value of all on-hand stock (at cost):** Rs 54,962,882

**SKUs with low-confidence forecasts (under 12 weeks of history):** 2

## Top 10 — Reorder now

| sku_id   | category         |   stockout_risk |   sales_at_risk |   weeks_of_cover |
|:---------|:-----------------|----------------:|----------------:|-----------------:|
| SKU-0075 | Furnishings      |        0.612893 |        129804   |         0        |
| SKU-0012 | Small Appliances |        0.704076 |         66859.9 |         0        |
| SKU-0142 | Decor            |        0.809302 |         55018.9 |         0        |
| SKU-0030 | Decor            |        0.787055 |         39855.8 |         0        |
| SKU-0033 | Decor            |        0.77326  |         20380.7 |         0.732516 |

## Top 10 — Markdown / clear

| sku_id   | category         |   overstock_risk |   excess_capital_at_risk |   weeks_of_cover |
|:---------|:-----------------|-----------------:|-------------------------:|-----------------:|
| SKU-0059 | Furnishings      |         1        |              6.86637e+06 |          35.1843 |
| SKU-0129 | Furnishings      |         1        |              1.08439e+06 |          57.4762 |
| SKU-0099 | Furnishings      |         1        |              1.0626e+06  |          45.096  |
| SKU-0161 | Furnishings      |         1        |              1.01914e+06 |          50.6601 |
| SKU-0016 | Furnishings      |         1        |         915304           |          59.3841 |
| SKU-0168 | Small Appliances |         0.676725 |         879032           |          10.4138 |
| SKU-0196 | Furnishings      |         1        |         854479           |          39.9413 |
| SKU-0175 | Furnishings      |         1        |         840113           |          40.2058 |
| SKU-0181 | Small Appliances |         1        |         824646           |          89.9698 |
| SKU-0068 | Decor            |         1        |         791635           |          33.8421 |
