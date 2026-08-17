''' load the raw data, downsize, save into chosen time splits'''

import pandas as pd 
from downsize_data import down_data
import os 

DATA_DIR = '../data/raw'
OUT_DIR  = '../data/processed'
MELTED_DATA = '../data/processed/sales_melted.parquet'

def build_long_df():
    ''' loading the raw datas, downsizing them by converting to categorical variables, adding calender features'''
    # load the raw datas
    sales_df = pd.read_csv(f'{DATA_DIR}/sales_train_evaluation.csv')
    calender_df = pd.read_csv(f'{DATA_DIR}/calendar.csv')
    prices_df = pd.read_csv(f'{DATA_DIR}/sell_prices.csv')

    # downsize the data
    sales_df = down_data(sales_df)
    calender_df = down_data(calender_df)
    prices_df   = down_data(prices_df)

    day_cols = [col for col in sales_df.columns if col.startswith('d_')]
    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

    # Melt the data
    print("... converting from wide to long format ...")
    long_df = sales_df.melt(id_vars=id_cols,value_vars=day_cols,var_name='d',value_name='sales')
    print("... conversion done ...")
    print("\n merging with calendar and prices dataframes")
    long_df = long_df.merge(calender_df, on='d', how='left') #merged calender
    long_df = long_df.merge(prices_df,on=['item_id','store_id','wm_yr_wk'],how='left')
    print("\n merging done...")


    print("\n making calander features")
    
    long_df['date'] = pd.to_datetime(long_df['date'])
    long_df['day_of_week'] = long_df['date'].dt.dayofweek
    long_df['day_of_month'] = long_df['date'].dt.day
    long_df['week_of_year'] = long_df['date'].dt.isocalendar().week.astype(int)
    long_df['month'] = long_df['date'].dt.month
    long_df['year'] = long_df['date'].dt.year

    print("saving the long format data")
    os.makedirs(OUT_DIR, exist_ok=True)
    long_df.to_parquet(MELTED_DATA, index=False)
    # print(f"Saved long-format (all stores): {len(long_df)} rows -> {}")

    return long_df.sort_values(['item_id', 'date'])

def load_data():
    if os.path.exists(MELTED_DATA):
        print(f"Loading cached long format from {MELTED_DATA}")
        return pd.read_parquet(MELTED_DATA)
    print("No cached long format found — building it now.")
    return build_long_df()

def split_and_save(store_id='CA_1', cutoff_date='2016-01-05'):
    long_df = load_data()

    # get only for a given store

    print(f"\n selecting {store_id}")
    long_df = long_df[long_df['store_id']==store_id]

    print("\n removing the dates with Nan values in sell price")
    long_df = long_df[long_df['sell_price'].notna()].copy()
  
    known = long_df[long_df['date'] <= cutoff_date].copy()
    future = long_df[long_df['date'] > cutoff_date].copy()

    known_path = f'{OUT_DIR}/sales_known_{store_id.lower()}.parquet'
    future_path = f'{OUT_DIR}/sales_future_{store_id.lower()}.parquet'

    known.sort_values(['item_id','date']).to_parquet(known_path, index=False)
    future.sort_values(['item_id','date']).to_parquet(future_path, index=False)

    print(f"[{store_id}] known: {known['date'].min().date()} to {known['date'].max().date()} "
          f"({len(known)} rows) -> {known_path}")
    print(f"[{store_id}] future: {future['date'].min().date()} to {future['date'].max().date()} "
          f"({len(future)} rows) -> {future_path}")

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--store_id', default='CA_1')
    # parser.add_argument('--cutoff_date', default='2016-01-05')
    # parser.add_argument('--rebuild_long', action='store_true',
    #                      help='Force rebuild of the cached long-format file, even if it exists')
    # args = parser.parse_args()

    # if args.rebuild_long and os.path.exists(MELTED_DATA):
    #     os.remove(MELTED_DATA)

    split_and_save()

