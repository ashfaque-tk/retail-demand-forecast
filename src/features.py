''' Build the leakage free feature set for two type of forecasting: recursive & Direct.
Update necessarily according to the problem.

### features
# calendar features: basic calendar set with promotional events 
# price features : price dynamics taken into account 
# lag and rolling features: leakage free, never use todays sales always count up to day(t-1) for d(t) features'''

import os 
import numpy as np
import pandas as pd
from typing import Any

import time 
import os 
import numpy as np
import pandas as pd
from typing import Any
import time 

class FeatureBuilder():

    def __init__(
        self, 
        lags: list[int] = [7, 28, 60, 90],
        rolling_means: list[int] = [7, 28, 60, 90],
        rolling_maxs: list[int] = [7, 28, 60, 90],
        rolling_on_lags: dict[int, list[int]] = {28: [7, 28]},
        col_names: dict = {
            'item_col': 'item_id', 'dept_col': 'dept_id', 'cat_col': 'cat_id',
            'store_col': 'store_id', 'state_col': 'state_id', 'price_col': 'sell_price',
            'week_col': 'wm_yr_wk', 'target': 'sales', 'date_col': 'date'
        }
    ):
        self.lags = lags
        self.rolling_means = rolling_means
        self.rolling_maxs = rolling_maxs
        self.rolling_on_lags = rolling_on_lags
        
        # Base metadata columns that should NEVER be used as model inputs
        # any cols with direct relation with target like revenue, should also be removed
        # otherwise, code will run but with faulty predictions.
        self.excluded_metadata = {
            'id', 'weekday', 'date', 'sales', 'revenue','origin_date', 'target_date', 'target_sales',
            'd', 'wm_yr_wk', 'event_name_1', 'event_name_2', 'event_type_1', 'event_type_2',
            'store_id', 'state_id','snap_TX','snap_WI'
        }

        self.id_col = col_names['item_col']
        self.date_col = col_names['date_col']
        self.dept_col = col_names['dept_col']
        self.cat_col = col_names['cat_col']
        self.price_col = col_names['price_col']
        self.week_col = col_names['week_col']
        self.target_col = col_names['target']
        self.store_col = col_names['store_col']
        self.state_col = col_names['state_col']

    # ---------- validation ----------
    def _require_datetime(self, df: pd.DataFrame, df_name: str):
        if self.date_col not in df.columns:
            raise ValueError(f"{df_name} missing required '{self.date_col}' column")
        if not np.issubdtype(df[self.date_col].dtype, np.datetime64):
            df[self.date_col] = pd.to_datetime(df[self.date_col])
        return df 

    def _validate_columns(self, df: pd.DataFrame, required: list[str], df_name: str = "df"):
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{df_name} missing required columns: {missing}")

    # ---------- unified lags & rolling features ----------
    def add_lags_and_rollings(
        self,
        df: pd.DataFrame,
        lags: list[int] | None = None,
        rolling_means: list[int] | None = None,
        rolling_maxs: list[int] | None = None,
        rolling_on_lags: dict[int, list[int]] | None = None,
    ) -> pd.DataFrame:
        """Computes all lag and rolling features in a single pass while preserving exact row index alignment."""
        lags = self.lags if lags is None else lags
        rolling_means = self.rolling_means if rolling_means is None else rolling_means
        rolling_maxs = self.rolling_maxs if rolling_maxs is None else rolling_maxs
        rolling_on_lags = self.rolling_on_lags if rolling_on_lags is None else rolling_on_lags

        # Ensure dataset is sorted strictly by item and date
        df = df.sort_values([self.id_col, self.date_col]).reset_index(drop=True)
        gb_target = df.groupby(self.id_col, observed=True)[self.target_col]

        # 1. Standard lags
        for lag in lags:
            df[f"lag_{lag}"] = gb_target.shift(lag)

        # 2. Shifted target to prevent lookahead leakage for rolling features (t-1)
        target_shifted = gb_target.shift(1)
        # gb_shifted = target_shifted.groupby(df[self.id_col], observed=True)

        for wdw in rolling_means:
            # transform / rolling directly matches df.index without dropping levels
            df[f"rolling_mean_{wdw}"] = target_shifted.transform(lambda x: x.rolling(wdw).mean())

        for wdw in rolling_maxs:
            df[f"rolling_max_{wdw}"] = target_shifted.transform(lambda x: x.rolling(wdw).max())

        # 3. Rolling statistics on top of a specific lag (e.g., lag 28 rolling mean)
        if rolling_on_lags:
            for lag, windows in rolling_on_lags.items():
                lag_col = f"lag_{lag}"
                if lag_col not in df.columns:
                    df[lag_col] = gb_target.shift(lag)
                gb_lag = df.groupby(self.id_col, observed=True)[lag_col]
                for wdw in windows:
                    df[f"rolling_lag_{lag}_win_{wdw}"] = gb_lag.transform(lambda x: x.rolling(wdw, min_periods=1).mean())

        return df

    # ---------- trend features ----------
    def add_trend_features(
        self,
        df: pd.DataFrame,
        mean_col_short: str = "rolling_mean_7",
        mean_col_long: str = "rolling_mean_28",
    ) -> pd.DataFrame:
        if mean_col_short in df.columns and mean_col_long in df.columns:
            df["selling_trend"] = df[mean_col_short] / (df[mean_col_long] + 1e-5)
        return df

    # ---------- price features ----------
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        group = [self.id_col, self.store_col]
        
        # Price change relative to 7 days ago
        price_lag_7 = df.groupby(group, observed=True)[self.price_col].shift(7)
        df["delta_price_weekn-1"] = ((df[self.price_col] - price_lag_7) / price_lag_7).where(price_lag_7 > 0)
        
        # Expanding historical price statistics
        df["price_historical_mean"] = df.groupby(group, observed=True)[self.price_col].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        df["price_ratio_mean"] = df[self.price_col].div(df["price_historical_mean"]).where(df["price_historical_mean"] > 0)
        df["is_discounted"] = np.where(
            df["price_historical_mean"].notna(),
            (df[self.price_col] < df["price_historical_mean"]).astype("int8"),
            np.nan,
        )
        
        # Relative price compared with other products in the same department
        dept_week_group = [self.dept_col, self.week_col, self.store_col]
        if all(col in df.columns for col in dept_week_group):
            dept_mean_price = (
                df.groupby(dept_week_group, observed=True)[self.price_col]
                .mean()
                .reset_index()
                .rename(columns={self.price_col: "dept_mean_price"})
            )
            df = df.merge(dept_mean_price, on=dept_week_group, how="left")
            df["delta_price_rltv_dept"] = df[self.price_col] / df["dept_mean_price"]
        return df

    # ---------- main orchestrator ----------
    def build(
        self,
        df: pd.DataFrame,
        lags: list[int] | None = None,
        rolling_means: list[int] | None = None,
        rolling_maxs: list[int] | None = None,
        rolling_on_lags: dict[int, list[int]] | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Main entry point. Builds all features and returns (featured_df, feature_column_names)."""
        df = df.copy()
        
        # 1. Price features
        if self.price_col in df.columns and "price_historical_mean" not in df.columns:
            df = self.add_price_features(df)
            
        # 2. Unified lags & rolling statistics
        df = self.add_lags_and_rollings(
            df,
            lags=lags,
            rolling_means=rolling_means,
            rolling_maxs=rolling_maxs,
            rolling_on_lags=rolling_on_lags,
        )
        
        # 3. Short vs long-term trend
        df = self.add_trend_features(df)
        
        # 4. Feature extraction
        feature_names = [col for col in df.columns if col not in self.excluded_metadata]
        return df, feature_names

    # ---------- recursive next-day predictor ----------
    def build_next_day_features(
        self,
        history_df: pd.DataFrame,
        next_day_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Prepares lag/rolling features for a single future day during recursive forecasting."""
        target_date = next_day_df[self.date_col].unique()[0]
        max_hist_date = history_df[self.date_col].max()
        
        assert (pd.to_datetime(target_date) - pd.to_datetime(max_hist_date)).days == 1, (
            f"Non-consecutive target date given: history max is {max_hist_date}, next day is {target_date}"
        )
  
        next_day = next_day_df.copy()
        next_day[self.target_col] = np.nan
        
        # Combine history with next_day placeholder
        hist_updated = (
            pd.concat([history_df, next_day], ignore_index=True)
            .sort_values([self.id_col, self.date_col])
            .reset_index(drop=True)
        )
        
        # Re-compute lag & rolling features cleanly
        hist_updated = self.add_lags_and_rollings(
            hist_updated,
            lags=self.lags,
            rolling_means=self.rolling_means,
            rolling_maxs=self.rolling_maxs,
            rolling_on_lags=self.rolling_on_lags
        )
        
        hist_updated = self.add_trend_features(hist_updated)

        # Return only target date rows
        return hist_updated[hist_updated[self.date_col] == target_date].copy()

class TrainTestPrepare():

    """Formats features into train/test structures for different forecasting types
    Mainly for Multi-horizon direct hybrid forecast"""

    @staticmethod
    def build_direct_train_frame(known_df:pd.DataFrame, target_known_cols:list[str],
                                  original_feature_cols:list[str],horizon:int=28,
                                  id_col:str='item_id',date_col:str='date',target_col:str='sales')-> tuple[pd.DataFrame,list]:
        '''build a stacked direct-horizon training frame
        known_df: train_df with one row per item,date with static calendar features and lag and rolling features already calculated
        target_known_cols: calendar events for target date like snap_CA,is_event(target_snap_CA,target_is_event)
        horizon: forecast horizon '''
        gaps = known_df.groupby(id_col,observed=True)[date_col].diff().dropna()

        assert (gaps == pd.Timedelta(days=1)).all(),('date is not sequential, shift computed lag and rolling features'
        'will missatribute dates')
        base = known_df.sort_values([id_col,date_col]).copy()

        # rename the date as origin_date
        origin = base[[date_col,*original_feature_cols]].rename(columns={'date':'origin_date'})#id col alread in feature_cols
        # merge with targets with date renamed as target_date, on target_date
        targets = base[["item_id", "date", "sales", *target_known_cols]].rename(
            columns={
                "date": "target_date",
                "sales": "target_sales",
                **{col: f"target_{col}" for col in target_known_cols},
            }
        )
        
        frames = []

        for h in range(1,horizon+1):
            part = origin.copy()
            part['horizon'] = h
            part['target_date'] = part['origin_date']+pd.Timedelta(days=h)

            # merge 
            part = part.merge(targets,on=[id_col,'target_date'],how='inner',validate='many_to_one')
      
            frames.append(part)

        direct_train = pd.concat(frames,ignore_index=True)
        direct_feature_cols = [*original_feature_cols, "horizon", *[f"target_{col}" for col in target_known_cols]]

        return direct_train, direct_feature_cols

    @staticmethod
    def build_direct_test_frame(future_df:pd.DataFrame,
                                history_df:pd.DataFrame,
                                original_feature_cols:list[str],
                                 target_known_cols:list[str],
                                 max_horizon:int=28)->pd.DataFrame:
        '''future_df: test_data
        history_df: train_data with full features including lag and rolling features'''
        # One latest forecast origin per item.
        origins = (
            history_df.sort_values(["item_id", "date"])
            .groupby("item_id", observed=True)
            .tail(1)[["date", *original_feature_cols]]
            .rename(columns={"date": "origin_date"})
        )

        future = future_df[["item_id", "date", *target_known_cols]].rename(columns={"date": "target_date",
            **{col: f"target_{col}" for col in target_known_cols},})
        
        prediction_frame = future.merge(origins, on="item_id", how="inner", validate="many_to_one")

        prediction_frame["horizon"] = ( prediction_frame["target_date"] - prediction_frame["origin_date"]).dt.days

        prediction_frame = prediction_frame.query("horizon >= 1 and horizon <= @max_horizon").copy()

        return prediction_frame

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
        
        cal["days_to_next_event"] = (cal['next_event_date'] - cal[date_col]).dt.days.fillna(999).clip(upper=14)
        cal["days_since_last_event"] = (cal[date_col] - cal['prev_event_date']).dt.days.fillna(999).clip(upper=14)
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
    pass 