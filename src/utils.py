from __future__ import annotations

import pandas as pd

from src.utils_visuals import (
    plot_item_forecast_and_inventory,
    plot_inventory_policy_bars,
    extract_item_inputs_from_dataframes,
)

__all__ = [
    "get_items_top",
    "get_items_with_min_history",
    "plot_item_forecast_and_inventory",
    "plot_inventory_policy_bars",
    "extract_item_inputs_from_dataframes",
]

def get_items_top(df,percentile):
    df_temp = df.copy()

    df_temp['revenue'] = df_temp['sales']*df_temp['sell_price']
    #aggregate
    item_revenue = df_temp.groupby('item_id',observed=True)['revenue'].sum().reset_index()

    # calculate top 80th percentile
    top_20 = item_revenue['revenue'].quantile(percentile)
    top_20_items = item_revenue[item_revenue['revenue'] >= top_20]['item_id'].tolist()

    df_temp_top = df_temp[df_temp['item_id'].isin(top_20_items)].copy()
    df_temp_top['item_id'] = df_temp_top['item_id'].cat.remove_unused_categories()

    return df_temp_top

def get_items_with_min_history(df,min_history_days=100):
    '''returns items spanning the entire history of given dataframe''' 
    total_history = (df['date'].max()-df['date'].min()).days
    print("total history: ",total_history)

    item_stats = df.groupby('item_id',observed=True)['date'].agg(min_date='min',max_date='max',total_records='nunique').reset_index()
    item_stats['history_span'] = (item_stats['max_date'] - item_stats['min_date']).dt.days

    # 4. Create a boolean mask for items present across the full timeframe
    # Option A: Spans the full start-to-end range
    full_span_mask = item_stats['history_span'] >= min_history_days

    valid_full_history_items = item_stats[full_span_mask]['item_id'].tolist()

    print(f"Total unique items: {len(item_stats)}")
    print(f"Items with min {min_history_days} days: {len(valid_full_history_items)}")

    # 6. Filter your main DataFrame to keep only items with complete historical data
    df_complete = df[df['item_id'].isin(valid_full_history_items)].copy()
    return df_complete
