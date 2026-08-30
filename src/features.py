import numpy as np
import pandas as pd
import itertools
from typing import List,Dict


import pandas as pd
from typing import List

class GetLagRollFeatures:
    def __init__(self, id_col='item_id', date_col='date', target_col='sales'):
        self.id_col = id_col
        self.date_col = date_col
        self.target_col = target_col

    def add_lags(self, df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
        df = df.sort_values([self.id_col, self.date_col])
        gb = df.groupby(self.id_col, observed=True)[self.target_col]
        for lag in lags:
            df[f'lag_{lag}'] = gb.shift(lag)
        return df

    def add_rolling_mean(self, df: pd.DataFrame,  windows: List[int],source_col: str='lag_1') -> pd.DataFrame:
        """Rolling mean on any existing column, e.g. source_col='lag_1'."""
        df = df.sort_values([self.id_col, self.date_col])
        gb = df.groupby(self.id_col, observed=True)[source_col]
        for window in windows:
            df[f'rolling_mean_{window}'] = gb.rolling(window).mean().reset_index(level=0, drop=True)
        return df

    def add_rolling_max(self, df: pd.DataFrame, windows: List[int], source_col: str='lag_1') -> pd.DataFrame:
        """Rolling max on any existing column, e.g. source_col='lag_1'."""
        df = df.sort_values([self.id_col, self.date_col])

        gb = df.groupby(self.id_col, observed=True)[source_col]
        for window in windows:
            df[f'rolling_max_{window}'] = gb.rolling(window).max().reset_index(level=0, drop=True)
        return df

    def add_rolling_on_lag(self, df: pd.DataFrame, lags: List[int]=[7,28], windows: List[int]=[7,28]) -> pd.DataFrame:
        """Rolling mean computed on top of an existing lag_N column, e.g. lag=28 -> rolling_lag_28_win_7."""
        lag_col = f'lag_{lags}'
        # if lag_col not in df.columns:
        #     raise ValueError(f'{lag_col} not found — call add_lags(df, lags=[{lags}, ...]) first')
        df = df.sort_values([self.id_col, self.date_col])
        for lag in lags:
            lag_col = f'lag_{lag}'
            # print('lag col: ',lag_col)
            if lag_col not in df.columns:
                raise ValueError(f'{lag_col} not found -- call add_lags(df)')
            gb = df.groupby(self.id_col, observed=True)[lag_col]
            for window in windows:
                df[f'rolling_lag_{lag}_win_{window}'] = gb.rolling(window).mean().reset_index(level=0, drop=True)
        return df
    

def get_calendar_features(calendar_df,include_event_types=True):
    cal = calendar_df.copy()
    cal['date'] = pd.to_datetime(cal['date'])
    # derive calendar features 
    cal['day_of_week']  = cal['date'].dt.dayofweek
    cal['day_of_month'] = cal['date'].dt.day
    cal['week_of_year'] = cal['date'].dt.isocalendar().week.astype(int)

    # derive events 
    # event_name_1, event_name_2, event_type_1, event_type_2

    cal['is_event'] = (cal['event_name_1'].notna() | cal['event_name_2'].notna()).astype(int)
    # all event_dates
    event_dates = cal[cal["is_event"] == 1]["date"].drop_duplicates().sort_values()
    max_event_lookahead = 7 # days
    if event_dates.empty:
        cal["days_to_next_event"] = max_event_lookahead
        cal["days_since_last_event"] = max_event_lookahead
        cal["is_event_in_7_days"] = 0
        return cal

    def get_days_to_next(current_date):
        future_events = event_dates[event_dates >= current_date]
        if future_events.empty:
            return max_event_lookahead
        diff = (future_events.iloc[0] - current_date).days
        return min(diff, max_event_lookahead+1)
    
    # Calculate distance to nearest past event per row
    def get_days_since_last(current_date):
        past_events = event_dates[event_dates <= current_date]
        if past_events.empty:
            return max_event_lookahead
        diff = (current_date - past_events.iloc[-1]).days
        return min(diff, max_event_lookahead)
    
    cal["days_to_next_event"] = cal["date"].apply(get_days_to_next)
    cal["days_since_last_event"] = cal["date"].apply(get_days_since_last)

    # 3. Horizon Flag: Event coming up within 7 days?
    cal["is_event_in_7_days"] = (cal["days_to_next_event"] <= 7).astype(int)

    if include_event_types and "event_type_1" in cal.columns:
        # Combine type 1 and type 2 to capture overlap
        all_types = cal["event_type_1"].fillna("").astype(str) + " " + cal[
            "event_type_2"
        ].fillna("").astype(str)

        # Create binary indicators for top macro categories
        cal["is_sporting_event"] = all_types.str.contains("Sporting").astype(int)
        cal["is_cultural_event"] = all_types.str.contains("Cultural").astype(int)
        cal["is_national_event"] = all_types.str.contains("National").astype(int)
        cal["is_religious_event"] = all_types.str.contains("Religious").astype(int)

    # drop the sparse 

    return cal

def get_price_features(df,price_col='sell_price',store_col='store_id',item_col='item_id',dept_col='dept_id',week_col='wm_yr_wk'):

    df = df.copy()
    required_cols = [item_col,store_col,dept_col,week_col,price_col,'date']
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(f"missing required column: {missing_cols}")

    # sort the series to ensure correct shifting
    df = df.sort_values([store_col,item_col,'date']).reset_index(drop=True)

    # price change relative to 7 days ago
    group = [item_col,store_col]
    price_lag_7 = (df.groupby(group,observed=True)[price_col].shift(7))
    df['delta_price_weekn-1'] =((df[price_col]-price_lag_7)/price_lag_7).where(price_lag_7>0)

    # historical price , price ratio to historical mean
    df['historical_mean'] = df.groupby(group,observed=True)[price_col].transform(
        lambda x: x.shift(1).expanding().mean())
    df['historical_std'] = df.groupby(group,observed=True)[price_col].transform(
        lambda x: x.shift(1).expanding().std())

    df['price_ratio_mean'] = df[price_col].div(df['historical_mean']).where(df['historical_mean']>0)

    df["is_discounted"] = np.where(df["historical_mean"].notna(),(df[price_col] < df["historical_mean"]).astype("int8"),
        np.nan)

    # relative price compared with other products in the same group
    dept_week_group = [dept_col,week_col,store_col]
    dept_mean_price = (df.groupby(dept_week_group, observed=True)['sell_price']
                        .mean()
                        .reset_index()
                        .rename(columns={'sell_price': 'dept_mean_price'})
                    )

    df = df.merge( dept_mean_price, on=dept_week_group, how='left')

    df['delta_price_rltv_dept'] = (df['sell_price']/ df['dept_mean_price'])

    return df


def get_avg_sales(df,sales_col='sales',item_col='item_id',store_col='store_id',state_col='state_id',
                  dept_col='dept_id',cat_col='cat_id'):

    # since we are using only 'CA' and 'CA!' most of the below aggs are redundant, so we only keep what matters
    # to the current dataframe
    df = df.copy()

    required_cols = [item_col,store_col,dept_col,state_col,cat_col,'date']
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(f"missing required column: {missing_cols}")

    
    # Total average sales: average sales by different groups

    df['item_sold_avg'] = df.groupby(item_col,observed=True)[sales_col].transform('mean')
    # df['state_sold_avg'] = df.groupby(state_col,observed=True)[sales_col].transform('mean')
    # df['store_sold_avg'] = df.groupby(store_col,observed=True)[sales_col].transform('mean')
    df['dept_sold_avg'] = df.groupby(dept_col,observed=True)[sales_col].transform('mean')
    df['cat_sold_avg'] = df.groupby(cat_col,observed=True)[sales_col].transform('mean')

    # df['cat_dept_sold_avg'] = df.groupby([cat_col,dept_col],observed=True)[sales_col].transform('mean')
    # df['store_item_sold_avg'] = df.groupby([store_col,item_col],observed=True)[sales_col].transform('mean')
    # df['cat_item_sold_avg'] = df.groupby([cat_col,item_col],observed=True)[sales_col].transform('mean')
    # df['dept_item_sold_avg'] = df.groupby([dept_col,item_col],observed=True)[sales_col].transform('mean')
    # df['state_store_sold_avg'] = df.groupby([state_col,store_col],observed=True)[sales_col].transform('mean')
    # df['state_store_cat_sold_avg'] = df.groupby([state_col,store_col,cat_col],observed=True)[sales_col].transform('mean')
    # df['store_cat_dept_sold_avg'] = df.groupby([store_col,cat_col,dept_col],observed=True)[sales_col].transform('mean')

    return df 

def get_trend_features(df, mean_col_7:str='rolling_mean_7', mean_col_28:str='rolling_mean_28',
                        historical_mean_col:str='historical_mean'):

    ''' calculate the trend columns given rolling mean and historical means are calculated'''
    
    cols = [mean_col_7,mean_col_28,historical_mean_col]
    missing_cols = [c for c in cols if c not in df.columns]
    
    if missing_cols:
        raise ValueError(f"missing col: {missing_cols}")

    df['selling_trend'] = df['rolling_mean_7']/(df['rolling_mean_28']+1e-5) # to avoid divide by zero
    df['demand_vs_historical_mean'] = df['rolling_mean_7']/(df[historical_mean_col]+1e-5) 

    return df 
