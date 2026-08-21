import pandas as pd

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

def get_active_window_data(df, train_start, train_end, min_active_days=28, date_col='date', item_col='item_id', target_col='sales'):
    """
    Filters dataset for a specific backtesting window, retaining only items 
    with sufficient active history prior to train_end.
    """
    # 1. Slice training period
    train_mask = (df[date_col] >= train_start) & (df[date_col] <= train_end)
    train_slice = df[train_mask]
    
    # 2. Count active selling days (sales > 0) per item within training window
    active_counts = train_slice[train_slice[target_col] > 0].groupby(item_col)[date_col].nunique()
    
    # 3. Identify items meeting minimum history threshold
    valid_items = active_counts[active_counts >= min_active_days].index
    
    # 4. Filter dataset for valid items
    window_df = df[df[item_col].isin(valid_items)].copy()
    
    return window_df