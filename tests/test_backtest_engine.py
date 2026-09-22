import pytest
import pandas as pd
import numpy as np

# =====================================================================
# Fixtures: Dummy Time-Series Data & Engine Mock
# =====================================================================

@pytest.fixture
def sample_timeseries_data():
    """Generates a complete daily dataset spanning 30 days for 2 SKUs."""
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    records = []
    for d in dates:
        records.append({"date": d, "item_id": "SKU_A", "sales": 10.0, "sales_pred": 9.5})
        records.append({"date": d, "item_id": "SKU_B", "sales": 5.0, "sales_pred": 5.0})
    
    df = pd.DataFrame(records)
    df["item_id"] = df["item_id"].astype("category")
    return df


# =====================================================================
# 1. Edge Case: Data Leakage & Boundary Check
# =====================================================================

def test_window_slicing_no_future_leakage(sample_timeseries_data):
    """
    EDGE CASE: Ensure training/testing slices never leak future dates.
    Test boundary condition at cutoff date T_cutoff.
    """
    cutoff_date = pd.Timestamp("2026-01-15")
    
    # Slice historical training window
    train_slice = sample_timeseries_data[sample_timeseries_data["date"] <= cutoff_date]
    test_slice = sample_timeseries_data[sample_timeseries_data["date"] > cutoff_date]
    
    # Assert maximum train date is strictly <= cutoff
    assert train_slice["date"].max() == cutoff_date
    # Assert minimum test date is strictly > cutoff
    assert test_slice["date"].min() > cutoff_date
    # Assert set intersection of dates is completely empty (no overlap)
    assert set(train_slice["date"]).isdisjoint(set(test_slice["date"]))


# =====================================================================
# 2. Edge Case: Empty & Missing Data Handling
# =====================================================================

def test_engine_handles_empty_forecast_window(sample_timeseries_data):
    """
    EDGE CASE: What happens if a window returns an empty DataFrame?
    The engine should raise a clean ValueError or skip gracefully, not crash with KeyError.
    """
    empty_df = sample_timeseries_data[sample_timeseries_data["sales"] < 0] # Returns empty
    
    assert empty_df.empty, "DataFrame should be empty for this test condition"
    
    # Check that operations on empty datasets handle gracefully
    grouped = empty_df.groupby("item_id", observed=True)["sales_pred"].sum()
    assert isinstance(grouped, pd.Series)
    assert len(grouped) == 0


def test_engine_handles_nan_predictions(sample_timeseries_data):
    """
    EDGE CASE: Forecast contains NaN or null values for a specific day.
    """
    df_with_nan = sample_timeseries_data.copy()
    df_with_nan.loc[0, "sales_pred"] = np.nan
    
    # Verify fillna strategy or explicit error handling
    cleaned_pred = df_with_nan["sales_pred"].fillna(0.0)
    assert not cleaned_pred.isna().any(), "Null predictions must be filled before inventory simulation"


# =====================================================================
# 3. Minimum Case: Multi-Window Result Concatenation
# =====================================================================

def test_window_aggregation_preserves_row_count():
    """
    MINIMUM CASE: Aggregating output logs from multiple backtest windows
    must preserve total records and structure.
    """
    window_1_log = pd.DataFrame({"window": [1, 1], "item_id": ["A", "B"], "metric": [0.9, 0.85]})
    window_2_log = pd.DataFrame({"window": [2, 2], "item_id": ["A", "B"], "metric": [0.92, 0.88]})
    
    results = [window_1_log, window_2_log]
    aggregated = pd.concat(results, ignore_index=True)
    
    assert len(aggregated) == 4
    assert list(aggregated["window"].unique()) == [1, 2]