"""
Smoke tests for the Streamlit dashboard (D5) using Streamlit's AppTest harness.
Runs every page headlessly and asserts it produces no unhandled exception —
this is the automated check behind "loads on seeded data" in the D5
acceptance criteria.

Run:  python3 -m pytest tests/test_dashboard_pages.py -v
"""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parent.parent / "app"

PAGES = [
    APP_DIR / "Home.py",
    APP_DIR / "pages" / "1_📈_Sales_Analytics.py",
    APP_DIR / "pages" / "2_🔮_Forecast.py",
    APP_DIR / "pages" / "3_📦_Inventory_Dashboard.py",
    APP_DIR / "pages" / "4_⚠️_Risk_Dashboard.py",
    APP_DIR / "pages" / "5_🔍_Product_Details.py",
    APP_DIR / "pages" / "6_🧾_Executive_Summary.py",
]


@pytest.mark.parametrize("page_path", PAGES, ids=[p.name for p in PAGES])
def test_page_runs_without_exception(page_path):
    at = AppTest.from_file(str(page_path), default_timeout=60)
    at.run()
    if at.exception:
        for e in at.exception:
            print(f"\n--- Exception in {page_path.name} ---")
            print(e.stack_trace if hasattr(e, "stack_trace") else e.value)
    assert not at.exception, f"{page_path.name} raised: {[str(e) for e in at.exception]}"
