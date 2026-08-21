# Model Performance — Project FORESIGHT (D3)

Backtest: 10 rolling-origin folds, 6-week horizon, walk-forward (each fold trains only on data available at that fold's origin — no leakage across folds).

| Metric | Model (LightGBM, tweedie) | Seasonal-naive baseline |
|---|---|---|
| WAPE (primary) | 0.290 | 0.379 |
| MAPE (secondary, \|actual\|>1 only) | 0.388 | 0.588 |
| Bias (signed mean error) | -1.30 | +0.93 |

**Result: the model beats the seasonal-naive baseline** (a 23.5% WAPE improvement).

**Shipped model: `lightgbm_tweedie`** — per the engagement's non-negotiable rule (brief §7.1), we ship whichever wins the honest backtest, not whichever looks more sophisticated.

## Per-fold detail

|   fold | origin_week                |   n_test |   model_wape |   baseline_wape |   model_mape |   baseline_mape |   model_bias |   baseline_bias | model_beats_baseline   |
|-------:|:---------------------------|---------:|-------------:|----------------:|-------------:|----------------:|-------------:|----------------:|:-----------------------|
|      0 | 2026-04-27T00:00:00.000000 |     1188 |     0.257668 |        0.350692 |     0.327076 |        0.567386 |     0.119791 |       1.50019   | True                   |
|      1 | 2026-05-04T00:00:00.000000 |     1188 |     0.264623 |        0.344191 |     0.326483 |        0.55379  |    -1.63742  |       0.682138  | True                   |
|      2 | 2026-05-11T00:00:00.000000 |     1188 |     0.253465 |        0.349477 |     0.331913 |        0.554179 |     0.698418 |       1.29089   | True                   |
|      3 | 2026-05-18T00:00:00.000000 |     1188 |     0.250897 |        0.354251 |     0.332322 |        0.548928 |    -1.25723  |       1.31839   | True                   |
|      4 | 2026-05-25T00:00:00.000000 |     1188 |     0.2853   |        0.378902 |     0.367034 |        0.56907  |    -2.18589  |       0.181777  | True                   |
|      5 | 2026-06-01T00:00:00.000000 |     1194 |     0.310846 |        0.397172 |     0.461877 |        0.590661 |    -1.31351  |       0.991678  | True                   |
|      6 | 2026-06-08T00:00:00.000000 |     1194 |     0.310397 |        0.40358  |     0.40653  |        0.599652 |    -1.48832  |       0.897178  | True                   |
|      7 | 2026-06-15T00:00:00.000000 |     1194 |     0.317082 |        0.404101 |     0.432531 |        0.62883  |    -1.73278  |       1.34525   | True                   |
|      8 | 2026-06-22T00:00:00.000000 |     1194 |     0.316574 |        0.401028 |     0.425257 |        0.627584 |    -3.15847  |       1.07094   | True                   |
|      9 | 2026-06-29T00:00:00.000000 |     1194 |     0.326359 |        0.399137 |     0.466919 |        0.640009 |    -0.990369 |      -0.0215528 | True                   |
