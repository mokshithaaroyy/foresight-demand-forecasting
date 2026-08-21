# Data Quality Log — Project FORESIGHT
Generated automatically by `src/pipeline.py` on every run. Every cleaning decision below is coded (see `src/pipeline.py`), not manual.

## validate
- **sku_master**: {'rows': 201, 'columns': ['sku_id', 'category', 'subcategory', 'launch_date', 'unit_cost', 'list_price']}
- **calendar**: {'rows': 787, 'columns': ['date', 'week', 'month', 'season', 'is_holiday', 'promo_event']}
- **sales_daily**: {'rows': 133900, 'columns': ['date', 'sku_id', 'units_sold', 'revenue', 'unit_price', 'promo_flag']}
- **inventory_snapshots**: {'rows': 18998, 'columns': ['date', 'sku_id', 'on_hand_units', 'on_order_units', 'lead_time_days', 'reorder_point']}

## clean_sku_master
- **rows_before**: 201
- **rows_after**: 200
- **duplicate_sku_ids_resolved**: 1
- **category_values_after_cleaning**: ['Decor', 'Furnishings', 'Small Appliances']

## clean_calendar
- **rows**: 787

## clean_sales
- **rows_before**: 133900
- **unparseable_dates_dropped**: 0
- **exact_duplicate_rows_dropped**: 400
- **rows_with_unknown_sku_dropped**: 0
- **negative_units_sold_clipped_to_zero**: 29
- **missing_units_sold_imputed_rolling_median**: 667
- **missing_unit_price_rows**: 1335
- **rows_final**: 133500

## clean_inventory
- **rows_before**: 18998
- **exact_duplicate_rows_dropped**: 15
- **duplicate_date_sku_keys_resolved**: 0
- **missing_lead_time_days_imputed**: 949
- **rows_final**: 18983

## master_dataset
- **rows**: 133500
- **columns**: ['date', 'sku_id', 'units_sold', 'revenue', 'unit_price', 'promo_flag', 'units_sold_corrected_negative', 'units_sold_imputed', 'unit_price_imputed', 'week', 'month', 'season', 'is_holiday', 'promo_event', 'category', 'subcategory', 'launch_date', 'unit_cost', 'list_price', 'on_hand_units', 'on_order_units', 'lead_time_days', 'reorder_point', 'lag_1', 'lag_7', 'lag_14', 'lag_28', 'roll_mean_7', 'roll_mean_28', 'roll_std_7', 'day_of_week', 'is_weekend', 'week_of_year', 'is_promo', 'has_promo_event', 'days_since_launch', 'discount_pct']
- **date_range**: ['2024-08-16 00:00:00', '2026-08-16 00:00:00']
- **n_skus**: 200
- **rows_missing_inventory_position**: 619
