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


def _sample_series(df:pd.DataFrame,n_series:int, seed:int,id_col:str='item_id',sales_col:str='sales',price_col:str='sell_price')->pd.DataFrame:
    ''' A stratified sampling from the main dataset based on revenue'''

    df = df.copy()

    df['revenue'] = df[price_col] * df[sales_col]

    item_stats = df.groupby(id_col,observed=True).agg(total_revenue=('revenue','sum'),
                                                      total_records= (sales_col,'count'),active_records=(sales_col,lambda x:(x>0).sum())).reset_index()

    # sales frequency ratio
    item_stats['sale_frequency'] = item_stats['active_records']/item_stats['total_records']
    # 3 revenue bins (high,medium,low)
    item_stats['rev_group'] = pd.qcut(item_stats['total_revenue'], q=3, labels=[f"Rev_Q{i+1}" for i in range(3)], duplicates='drop')
    # 2 sales bins (fast smooth moving, intermittent)
    item_stats['interm_group'] = pd.qcut(item_stats['sale_frequency'],  q=2, labels=[f"Freq_Q{i+1}" for i in range(2)],duplicates='drop')

    # Combine into a one label (e.g., "Rev_Q3_Freq_Q2")
    item_stats['combined_group'] = ( item_stats['rev_group'].astype(str) + "_" + item_stats['interm_group'].astype(str))


    # 3. Calculate target sample quota per 2D strata cell
    unique_strata = item_stats['combined_group'].unique()
    per_strata_quota = max(1, n_series // len(unique_strata))

    print(f"Sampling across {len(unique_strata)} strata combinations (~{per_strata_quota} SKUs per cell)...")

    # 4. Perform stratified sampling of item_ids
    sampled_ids = []
    for strata_label, group in item_stats.groupby('combined_group', observed=True):
        sample_size = min(len(group), per_strata_quota)
        sampled_ids.extend(
            group.sample(n=sample_size, random_state=seed)[id_col].tolist()
        )

    # Adjust to exact n_series limit if rounding caused slight mismatch
    sampled_ids = sampled_ids[:n_series]

    print(f"Successfully selected {len(sampled_ids)} unique series.")

    # 5. Filter the long-format DataFrame to return full history for selected items
    filtered_df = df[df[id_col].isin(sampled_ids)].reset_index(drop=True)
    return filtered_df


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

if __name__ == '__main__':

    data_dir = 'data/processed/'

    train_data = pd.read_parquet(f'{data_dir}sales_known_ca_1.parquet')
    test_data = pd.read_parquet(f'{data_dir}sales_future_ca_1.parquet')

    ### use the train_data from the last 3 years and spans the entire history, only needed if we implement inventory policy with error deviation

    cutoff = train_data['date'].max()-pd.Timedelta(days=1095)

    train_full_hist = get_items_with_min_history(train_data[train_data['date']>=cutoff],min_history_days=1095)

    print('items in the last 3 year with full hist: ',train_full_hist['item_id'].nunique())


    train_stratified = _sample_series(train_full_hist,n_series=300,seed=42)

    print(train_stratified['item_id'].nunique())

    # filter the test_dataset
    test_stratified = test_data[test_data['item_id'].isin(train_stratified['item_id'])]

    print(test_stratified['item_id'].nunique())

    train_stratified = train_stratified.sort_values(['item_id','date'])
    test_stratified  = test_stratified.sort_values(['item_id','date'])

    assert set(test_stratified['item_id'].unique().tolist())==set(train_stratified['item_id'].unique().tolist()),'AssersionError: test and train have different items'

    print(train_stratified.columns,test_stratified.columns)
    # ### save this as our final set 

    # train_stratified.to_parquet(f'{data_dir}train_filtered_ca1.parquet')
    # test_stratified.to_parquet(f'{data_dir}test_filtered_ca1.parquet')

    # quit()

  