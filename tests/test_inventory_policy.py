# File: tests/test_inventory_policy.py

import pytest
import pandas as pd
import numpy as np
from inventory_policy import InventoryPolicy, calculate_error_statistics 


@pytest.fixture
def simulation_setup():
    items = ["SKU_1", "SKU_2"]
    sim_dates = pd.date_range("2026-12-01", periods=14, freq="D")

    # 1. Warmup history (November 2026)
    hist_dates = pd.date_range("2026-11-01", periods=30, freq="D")
    hist_records = []
    for d in hist_dates:
        for item in items:
            hist_records.append(
                {
                    "date": d,
                    "item_id": item,
                    "sales": 10.0 if item == "SKU_1" else 20.0,
                    "sales_pred": 9.0 if item == "SKU_1" else 22.0,
                }
            )
    forecasts_df = pd.DataFrame(hist_records)

    # 2. Simulation inputs (Dec 1 - 14)
    actual_records, pred_records = [], []
    for d in sim_dates:
        for item in items:
            actual_records.append(
                {"date": d, "item_id": item, "sales": 10.0 if item == "SKU_1" else 20.0}
            )
            pred_records.append(
                {"date": d, "item_id": item, "sales_pred": 9.0 if item == "SKU_1" else 22.0}
            )

    actual_sales_df = pd.DataFrame(actual_records)
    forecasted_demand_df = pd.DataFrame(pred_records)

    full_forecasts_df = pd.concat(
        [
            forecasts_df,
            pd.merge(actual_sales_df, forecasted_demand_df, on=["date", "item_id"]),
        ],
        ignore_index=True,
    )

    # SKU_1 starts understocked (5 units) to force stockouts
    # SKU_2 starts well-stocked (100 units)
    initial_on_hand = pd.Series([5.0, 100.0], index=items)
    initial_rmse = pd.Series([1.0, 2.0], index=items)

    policy = InventoryPolicy(
        inventory=initial_on_hand,
        forecasts=full_forecasts_df,
        lead_time=2,
        review_period=4,  # Risk Period = 6 Days
        daily_unit_holding_cost=0.2,
        stockout_cost=1.0,
        oos_error_list=[initial_rmse],
    )

    return policy, actual_sales_df, forecasted_demand_df


def test_transit_pipeline_dates(simulation_setup):
    """Verify orders are placed on review days (0, 4, 8, 12) and arrive L=2 days later."""
    policy, actual_sales_df, forecasted_demand_df = simulation_setup
    policy.daily_simulation(actual_sales_df, forecasted_demand_df)

    expected_arrivals = [
        pd.Timestamp("2026-12-03"),  # Ordered Dec 01
        pd.Timestamp("2026-12-07"),  # Ordered Dec 05
        pd.Timestamp("2026-12-11"),  # Ordered Dec 09
        pd.Timestamp("2026-12-15"),  # Ordered Dec 13
    ]

    actual_arrival_dates = [arr_date for arr_date, _ in policy.transit_pipeline]
    assert actual_arrival_dates == expected_arrivals


def test_deterministic_stockouts_and_costs(simulation_setup):
    """Verify SKU_1 experiences stockouts on Days 1 and 2 before first order arrives on Day 3."""
    policy, actual_sales_df, forecasted_demand_df = simulation_setup
    results = policy.daily_simulation(actual_sales_df, forecasted_demand_df)

    sku1_results = results[results["item_id"] == "SKU_1"].sort_values("date")

    # Day 1: Starts with 5, Demand 10 -> Lost sales = 5, On hand = 0
    day1 = sku1_results.iloc[0]
    assert day1["lost_sales"] == 5.0
    assert day1["stockout_cost"] == 5.0 * policy.stockout_penalty
    assert day1["on_hand"] == 0.0

    # Day 2: Starts with 0, Demand 10 -> Lost sales = 10, On hand = 0
    day2 = sku1_results.iloc[1]
    assert day2["lost_sales"] == 10.0
    assert day2["stockout_cost"] == 10.0 * policy.stockout_penalty

    # Day 3: Order arrives (arriving_qty > 0), fulfilling sales and eliminating stockout
    day3 = sku1_results.iloc[2]
    assert day3["arriving_qty"] > 0
    assert day3["lost_sales"] == 0.0

    print(results)
    assert results['actual_sales'].mean()!=0


def test_rolling_oos_error_growth(simulation_setup):
    """Verify that OOS error evaluation list grows across review days."""
    policy, actual_sales_df, forecasted_demand_df = simulation_setup

    initial_error_count = len(policy.oos_errors)
    policy.daily_simulation(actual_sales_df, forecasted_demand_df)

    # 4 review days over 14 days -> 4 error calculations added
    assert len(policy.oos_errors) == initial_error_count + 4


def test_calculate_error_statistics_returns_series():
    """Verify error calculation produces a numeric Series rather than a DataFrame with Categoricals."""
    sample_data = pd.DataFrame({
        'item_id': pd.Categorical(['item_A', 'item_A', 'item_B', 'item_B']),
        'cum_error': [1.5, -2.0, 0.5, -1.0]
    })
    
    result = calculate_error_statistics(sample_data)
    
    # Assert result is a Series, not a DataFrame
    assert isinstance(result, pd.Series)
    # Assert values are numeric floats
    assert pd.api.types.is_numeric_dtype(result)

def test_safety_stock_calculation():
    """Ensure _safety_stock multiplies floats against numeric series without type errors."""
    mock_inventory = pd.Series([10, 20], index=['item_A', 'item_B'])
    mock_forecasts = pd.DataFrame(columns=['date', 'item_id', 'sales', 'sales_pred'])
    
    # Mock pre-calculated OOS error as a Series indexed by item_id
    mock_oos_series = pd.Series([2.5, 3.1], index=['item_A', 'item_B'])
    
    policy = InventoryPolicy(
        inventory=mock_inventory,
        forecasts=mock_forecasts,
        oos_error_list=[mock_oos_series]
    )
    
    safety_stock = policy._safety_stock()
    
    assert isinstance(safety_stock, pd.Series)
    assert len(safety_stock) == 2