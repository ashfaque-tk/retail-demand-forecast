''' Build the leakage free feature set for two type of forecasting: recursive & Direct.
Update necessarily according to the problem.

### features
# calendar features: basic calendar set with promotional events 
# price features : price dynamics taken into account 
# lag and rolling features: leakage free, never use todays sales always count up to day(t-1) for d(t) features'''

import os 
import numpy as np
import pandas as pd
from typing import List,Dict


class FeatureBuilder():



    def __init__(self, col_names:dict = {'item_col':'item_id','dept_col':'dept_id','cat_col':'cat_id',
                                         'store_col':'store_id','state_col':'state_id','price_col':'sell_price',
                                         'week_col':'wm_yr_wk','target':'sales','date_col':'date'}):

        self.id_col = col_names['item_col']
        self.date_col = col_names['date_col']
        self.dept_col = col_names['dept_col']
        self.cat_col = col_names['cat_col']
        self.price_col = col_names['price_col']
        self.week_col  = col_names['week_col']
        self.target_col  = col_names['target']
        self.store_col = col_names['store_col']
        self.state_col = col_names['state_col']
        

    # ---------- validation ----------
    def _require_datetime(self, df: pd.DataFrame, df_name: str):
        if self.date_col not in df.columns:
            raise ValueError(f"{df_name} missing required '{self.date_col}' column")
        if not np.issubdtype(df[self.date_col].dtype, np.datetime64):
            df['date'] = pd.to_datetime(df['date'])
            return df 

    def _validate_columns(self, df: pd.DataFrame, required: List[str], df_name: str):
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{df_name} missing required columns: {missing}")

    # ---------- avg sales ----------
  
    def add_avg_sales(self,df:pd.DataFrame)-> pd.DataFrame:

        df['item_sold_avg'] = df.groupby(self.id_col, observed=True)[self.target_col].transform('mean')
        df['dept_sold_avg'] = df.groupby(self.dept_col, observed=True)[self.target_col].transform('mean')
        df['cat_sold_avg'] = df.groupby(self.CAT_COL, observed=True)[self.target_col].transform('mean')
        return df
    
    # ---------- lag / rolling — lags/windows always explicit, never stored ----------
    def add_lags(self, df:pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
        if lags is None:
            lags = []

   
        gb = df.groupby(self.id_col, observed=True)[self.target_col]
        for lag in lags:
            df[f'lag_{lag}'] = gb.shift(lag)
        return df
    
    def add_rolling_mean_max(self,df:pd.DataFrame,mean_window: List[int]|None,max_window:List[int]|None) -> pd.DataFrame:

        gb = df.groupby(self.id_col, observed=True)[self.target_col]
        if mean_window is not None:
            for wdw in mean_window:
                df[f'rolling_mean_{wdw}'] = gb.shift(1).rolling(wdw).mean().reset_index(level=0, drop=True)
        if max_window is not None:
            for wdw in max_window:
                df[f'rolling_max_{wdw}'] = gb.shift(1).rolling(wdw).max().reset_index(level=0,drop=True)
        return df

    def add_rolling_on_lag(self, df:pd.DataFrame,lags: List[int], windows: List[int]) -> pd.DataFrame:
     
        for lag in lags:
            lag_col = f'lag_{lag}'
            if lag_col not in df.columns:
                raise ValueError(f'{lag_col} not found -- call add_lags(df, lags) first')
            gb = df.groupby(self.id_col, observed=True)[lag_col]
            for window in windows:
                df[f'rolling_lag_{lag}_win_{window}'] = gb.rolling(window).mean().reset_index(level=0, drop=True)
        return df

    # ---------- trend ----------
    def add_trend_features(self,df:pd.DataFrame, mean_col_7: str='rolling_mean_7', mean_col_28: str='rolling_mean_28',
                            ) -> pd.DataFrame:
        
        self._validate_columns(df, [mean_col_7, mean_col_28], "df")
        df['selling_trend'] = df[mean_col_7] / (df[mean_col_28] + 1e-5)
        # df['demand_vs_historical_mean'] = df[mean_col_7] / (df[historical_mean_col] + 1e-5)
        return df
    
    def add_sales_historical_mean(self, df: pd.DataFrame, training_window: int = 730) -> pd.DataFrame:
        """ it reads self.target_col directly.
        """
        df = df.sort_values([self.id_col, self.date_col])
        df['sales_historical_mean'] = df.groupby(self.id_col, observed=True)[self.target_col].transform(
            lambda x: x.shift(1).rolling(training_window, min_periods=1).mean())
        return df
    
    def add_price_features(self,df:pd.DataFrame):
        df = df.copy()
        # price change relative to 7 days ago
        group = [self.id_col,self.store_col]
        price_lag_7 = (df.groupby(group,observed=True)[self.price_col].shift(7))
        df['delta_price_weekn-1'] =((df[self.price_col]-price_lag_7)/price_lag_7).where(price_lag_7>0)

        # historical price , price ratio to historical mean
        df['price_historical_mean'] = df.groupby(group,observed=True)[self.price_col].transform(
            lambda x: x.shift(1).expanding().mean())
        df['price_historical_std'] = df.groupby(group,observed=True)[self.price_col].transform(
            lambda x: x.shift(1).expanding().std())

        df['price_ratio_mean'] = df[self.price_col].div(df['price_historical_mean']).where(df['price_historical_mean']>0)

        df["is_discounted"] = np.where(df["price_historical_mean"].notna(),(df[self.price_col] < df["price_historical_mean"]).astype("int8"),
            np.nan)

        # relative price compared with other products in the same group

        dept_week_group  = [self.dept_col,self.week_col,self.store_col]
        dept_mean_price = (df.groupby(dept_week_group, observed=True)['sell_price']
                            .mean()
                            .reset_index()
                            .rename(columns={'sell_price': 'dept_mean_price'})
                        )

        df = df.merge( dept_mean_price, on=dept_week_group, how='left')

        df['delta_price_rltv_dept'] = (df['sell_price']/ df['dept_mean_price'])

        return df
    
    # ---------- orchestrator (recursive-forecast default; direct will call
    # # the pieces above directly, per-horizon, instead of this) ----------
    def build(self, df:pd.DataFrame, lags: List[int]=[1,2,3,7,28,60,90] ,mean_windows: List[int] = [3,7,14,21,28,60,90],
              max_windows:List[int]=[3,7,14,21,28,60,90],rolling_on_lags:List[int]=[28]) -> pd.DataFrame:

        df = df.copy()
        # df = self.add_avg_sales(df)
        df = self.add_lags(df, lags=lags)
        df = self.add_rolling_mean_max(df, mean_window=mean_windows,max_window=max_windows)
        df = self.add_rolling_on_lag(df, lags=rolling_on_lags, windows=[7,28])
        df = self.add_trend_features(df, mean_col_7=f'rolling_mean_7', mean_col_28=f'rolling_mean_28')

        if 'price_historical_mean' not in df.columns:
            df = self.add_price_features(df)
        return df

    def next_day_feature_build(self,history_df:pd.DataFrame, next_day_df:pd.DataFrame,dynamic_feats:dict[str,List[int]|None]={'lags':[7],
                                                                                                         'rolling_mean':[7,28,60,90],
                                                                                                         'rolling_max':[7,28,60,90],
                                                                                                         'rolling_on_lag':[28],
                                                                                                         }):
        '''history_df: already cut off to recent 100 days, columns: ['item_id','date','sales']
        next_day_df: pd.DataFrame(['items':items,'date':date])
        return next_day_feats: pd.DataFrame with only dynamic features,'item_id','date'''
       
        assert (next_day_df['date'].unique()[0] - history_df['date'].max()).days==1,'AssersionError: Non-consecutive target date is given'

        lags = dynamic_feats['lags']
        rolling_means = dynamic_feats['rolling_mean']
        rolling_max = dynamic_feats['rolling_max']
        rolling_on_lag = dynamic_feats['rolling_on_lag']

        next_day =  next_day_df.copy() 
        next_day['sales'] = np.nan
        
        # append this to history
        hist_updated = pd.concat([history_df,next_day],ignore_index=True).sort_values([self.id_col,self.date_col]).reset_index(drop=True)

        if lags is not None:
            hist_updated = self.add_lags(hist_updated,lags)
        if rolling_means is not None or rolling_max is not None:
            hist_updated = self.add_rolling_mean_max(hist_updated,mean_window=rolling_means,max_window=rolling_max)
        if rolling_on_lag is not None:
            hist_updated = self.add_rolling_on_lag(hist_updated,lags=rolling_on_lag,windows=[7,28])

        hist_updated = self.add_trend_features(hist_updated)

        next_day_dynamic_feats = hist_updated[hist_updated['date']==next_day['date'].unique()[0]]

        return next_day_dynamic_feats  # row[dynamic feats]

# ---------- calendar ----------

def add_calendar_features(cal_df: pd.DataFrame, date_col:str='date') -> pd.DataFrame:
    cal = cal_df.copy()
    cal[date_col] = pd.to_datetime(cal[date_col])
    
    cal['day_of_week'] = cal[date_col].dt.dayofweek
    cal['day_of_month'] = cal[date_col].dt.day
    cal['week_of_year'] = cal[date_col].dt.isocalendar().week.astype(int)

    # Event indicators
    cal['is_event'] = (cal['event_name_1'].notna() | cal['event_name_2'].notna()).astype(int)
    
    # Efficient nearest event calculation using merge_asof or forward/backward filling on event flags
    event_df = cal[cal['is_event'] == 1][[date_col]].drop_duplicates().sort_values(date_col)
    
    if event_df.empty:
        cal["days_to_next_event"] = 7
        cal["days_since_last_event"] = 7
        cal["is_event_in_7_days"] = 0
    else:
        # Merge to find next and previous events without slow row-by-row apply loops
        cal = pd.merge_asof(
            cal.sort_values(date_col), 
            event_df.rename(columns={date_col: 'next_event_date'}), 
            left_on=date_col, 
            right_on='next_event_date', 
            direction='forward'
        )
        cal = pd.merge_asof(
            cal.sort_values(date_col), 
            event_df.rename(columns={date_col: 'prev_event_date'}), 
            left_on=date_col, 
            right_on='prev_event_date', 
            direction='backward'
        )
        
        cal["days_to_next_event"] = (cal['next_event_date'] - cal[date_col]).dt.days.fillna(7).clip(upper=7)
        cal["days_since_last_event"] = (cal[date_col] - cal['prev_event_date']).dt.days.fillna(7).clip(upper=7)
        cal["is_event_in_7_days"] = (cal["days_to_next_event"] <= 7).astype(int)
        
        # Clean up temporary merge columns
        cal = cal.drop(columns=['next_event_date', 'prev_event_date'])

    all_types = cal["event_type_1"].fillna("").astype(str) + " " + cal["event_type_2"].fillna("").astype(str)
    cal["is_sporting_event"] = all_types.str.contains("Sporting").astype(int)
    cal["is_cultural_event"] = all_types.str.contains("Cultural").astype(int)
    cal["is_national_event"] = all_types.str.contains("National").astype(int)
    cal["is_religious_event"] = all_types.str.contains("Religious").astype(int)

    return cal
    
if __name__ == '__main__':
    from pathlib import Path
    import pandas as pd

    # Define project root dynamically based on script location
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"

    # Safe loading paths
    cal_path = DATA_DIR / "raw/calendar.csv"
    sales_path = DATA_DIR / 'processed/sales_known_ca_1.parquet'
    sales_path_test = DATA_DIR / 'processed/sales_future_ca_1.parquet'

    df_calendar = pd.read_csv(cal_path)
    df_sales_train = pd.read_parquet(sales_path)
    df_sales_test = pd.read_parquet(sales_path_test)

    # 1. Idempotent Calendar Feature Addition (Calendar is universal master data known in advance)
    if 'day_of_week' not in df_calendar.columns:
        df_calendar = add_calendar_features(df_calendar, date_col='date')

    # 2. Clean Merge for Train Set (Avoiding price aggregates to prevent data leakage)
    calendar_cols_train = [col for col in df_calendar.columns if col in df_sales_train.columns and col != 'date']
    if calendar_cols_train:
        df_sales_train = df_sales_train.drop(columns=calendar_cols_train)
    df_sales_train = df_sales_train.merge(df_calendar, on=['date'], how='left')

    # 3. Clean Merge for Test Set
    calendar_cols_test = [col for col in df_calendar.columns if col in df_sales_test.columns and col != 'date']
    if calendar_cols_test:
        df_sales_test = df_sales_test.drop(columns=calendar_cols_test)
    df_sales_test = df_sales_test.merge(df_calendar, on=['date'], how='left')

    # 4. Atomic Parquet Saving
    processed_dir = DATA_DIR / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    df_calendar.to_parquet(processed_dir / 'updated_calendar.parquet', index=False)
    df_sales_train.to_parquet(processed_dir / 'sales_known_ca_1.parquet', index=False)
    df_sales_test.to_parquet(processed_dir / 'sales_future_ca_1.parquet', index=False)
    
    print("Pipeline executed successfully. Calendar features merged cleanly without price feature contamination.")

