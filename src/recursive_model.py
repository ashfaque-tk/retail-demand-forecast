

'''Two type of forecast: Recursive vs Direct'''
import logging
import re
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from src.features import FeatureBuilder, TrainTestPrepare
from src.models_train import SelectModel



logger = logging.getLogger(__name__)


def _infer_dynamic_feats(full_feats) -> dict:
    """Derive which lags/rolling windows must be freshly recomputed for the newly
    appended forecast row on every recursive step, straight from the columns
    actually present in the training feature set -- so this can never silently
    drift out of sync with whatever feat_builder.build() was called with.

    (Replaces a hardcoded default of lags=[7] that used to live inside
    next_day_feature_build: it left lag_28/60/90 -- and rolling_lag_28_win_*,
    which depends on lag_28 -- as NaN for the model on every recursive step,
    silently. See the __main__ self-test below.)
    """
    lag_pat = re.compile(r'^lag_(\d+)$')
    mean_pat = re.compile(r'^rolling_mean_(\d+)$')
    max_pat = re.compile(r'^rolling_max_(\d+)$')
    on_lag_pat = re.compile(r'^rolling_lag_(\d+)_win_\d+$')

    lags = sorted({int(m.group(1)) for c in full_feats if (m := lag_pat.match(c))})
    rolling_mean = sorted({int(m.group(1)) for c in full_feats if (m := mean_pat.match(c))})
    rolling_max = sorted({int(m.group(1)) for c in full_feats if (m := max_pat.match(c))})
    rolling_on_lag = sorted({int(m.group(1)) for c in full_feats if (m := on_lag_pat.match(c))})

    missing_lag_deps = set(rolling_on_lag) - set(lags)
    if missing_lag_deps:
        raise ValueError(
            f"rolling_on_lag needs lag_{sorted(missing_lag_deps)} recomputed on every "
            f"recursive step, but those lags aren't in the training feature set's lag_* "
            f"columns ({lags}). Add them to the `lags` you pass to feat_builder.build()."
        )

    return {
        'lags': lags or None,
        'rolling_mean': rolling_mean or None,
        'rolling_max': rolling_max or None,
        'rolling_on_lag': rolling_on_lag or None,
    }


class Forecaster():

    
    def __init__(self, model:SelectModel, feature_builder:FeatureBuilder, original_features:list[str], forecast_type:str='recursive',
                 max_horizon_days:int=28):
        '''model: Already fitted SelectModel instance
        feature_builder: feature builder class'''
        self.model = model
        self.feature_builder = feature_builder
        self.forecast_type  = forecast_type
        self.full_features = original_features

        target_cols = ['snap_CA', 'is_event', 'is_event_in_7_days', 'day_of_week', 'month', 'sell_price','is_sporting_event',
                'is_cultural_event', 'is_national_event', 'is_religious_event','day_of_month','week_of_year']# fixed for now, change it if needed

        self.direct_target_cols = [col for col in target_cols if col in original_features]

        self.max_horizon = max_horizon_days

    def _get_model_feature_importance(self):
        ''' returns the fit model's feature importances'''
        importances = self.model['point'].feature_importances_

        # 2. Map to feature names for scannability
        importance_series = pd.Series(importances, index=self.full_features)
        
        # 3. Return sorted Series (highest importance first)
        return importance_series.sort_values(ascending=False)

    def forecast( self, train_df: pd.DataFrame, test_df: pd.DataFrame,  max_lookback_days: int = 100) -> pd.DataFrame:
                """Dispatches to direct or recursive forecast without engine-level logic."""
                if self.forecast_type == 'direct':
                    return self.direct_forecast(
                        train_df, test_df)
                return self.recursive_forecaster(train_df, test_df, max_lookback_days=max_lookback_days)
    
    
    def recursive_forecaster(self, 
                            train_df: pd.DataFrame, 
                            test_df: pd.DataFrame,
                            max_lookback_days: int = 100)->pd.DataFrame:
        '''train_df: entire training set
            test_df: test set with static features (should only contain the static features

            first fit the model And then 
            '''

        ##### split the train_df , to train_df[features], train_df['sales'] to fit the model.

        X_train,y_train = train_df[self.full_features], train_df['sales']
        self.model.fit(X_train,y_train)

        ##### recursive prediction, feeding back predicted sales as inputs for lag features for tomorrow
        
        full_feats = train_df.columns
        dynamic_feats = _infer_dynamic_feats(full_feats)

        cutoff_date = train_df['date'].max() - pd.Timedelta(days=max_lookback_days)
        history_slice = train_df[train_df['date'] >= cutoff_date].copy()

        dates = sorted(test_df['date'].unique())
        total_dates = len(dates)
        n_items = train_df['item_id'].nunique()

        logger.info(
            "Recursive forecast starting: %d day(s) x %d item(s), lookback=%d days, "
            "dynamic_feats=%s", total_dates, n_items, max_lookback_days, dynamic_feats
        )

        all_results = []
        for step, current_date in enumerate(dates, start=1):
            t0 = time.time()

            next_day = test_df[test_df['date'] == current_date].copy()  # all static feats
            next_day_dynamic_feats = self.feature_builder.build_next_day_features(
                history_slice, next_day, dynamic_feats=dynamic_feats
            )

            next_day_full_feats = next_day_dynamic_feats[self.full_features]

            assert set(next_day_full_feats['item_id'].unique().tolist()) == set(
                train_df['item_id'].unique().tolist()
            ), 'AssertionError: items in test_df do not match train_df'

            preds = self.model.predict(next_day_full_feats)  # point, q10, q90, ...

            day_result = pd.DataFrame({
                'item_id': next_day_full_feats['item_id'].values,
                'dept_id': next_day_full_feats['dept_id'].values,
                'cat_id' : next_day_full_feats['cat_id'].values,
                'date': current_date,
                'sales_pred': preds['point'],
            })
            for key, values in preds.items():
                if key == "point":
                    continue
                day_result[key] = values

            all_results.append(day_result)

            # append history with predicted sales, so tomorrow's lag/rolling
            # features see today's forecast rather than NaN
            history_slice = pd.concat(
                (history_slice, day_result[['item_id', 'date', 'sales_pred']]
                 .rename(columns={'sales_pred': 'sales'}))
            )

            pct = step / total_dates * 100
            logger.info(
                "[%d/%d] (%5.1f%%) date=%s done in %.2fs",
                step, total_dates, pct, pd.Timestamp(current_date).date(), time.time() - t0
            )

        preds_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
        logger.info("Recursive forecast complete: %d rows returned.", len(preds_df))
     
        return preds_df


    def direct_forecast(self,train_df:pd.DataFrame,test_df:pd.DataFrame):

        ''' train_df must have all features build already, 
          Build a direct train and test frame then predict'''
        logger.info( f"Multi horizon Direct Forecast: Horizon={self.max_horizon} days, Building Train, Test frames"
                    f'Data shapes: train_df: ({train_df.shape}), test_df:{test_df.shape} ')

        try:
             
            direct_train, direct_cols = TrainTestPrepare.build_direct_train_frame(known_df=train_df,
                                                                                      target_known_cols=self.direct_target_cols,
                                                                                      original_feature_cols=self.full_features,
                                                                                      horizon=self.max_horizon)

            logger.info("Direct train frame built with max horizon %d for each origin date",self.max_horizon)

        except ValueError as e:
            logger.error(f"Error in direct train building: {e}")
            quit()

        try: 
            X_direct_test = TrainTestPrepare.build_direct_test_frame(future_df=test_df,history_df=train_df,
                                                                                  original_feature_cols=self.full_features,
                                                                                  target_known_cols=self.direct_target_cols,
                                                                                  max_horizon=self.max_horizon)
        except ValueError as e:
            logger.error(f"Error in direct test frame building: {e}")
            quit()

        logger.info(f"Multi-horizon Direct forecast: Train, Test frame built sucessfully.  Train Frame shape:{direct_train.shape}, Test Frame shape: {X_direct_test.shape}" )
        # fit the model
        X_train, y_train = direct_train[direct_cols], direct_train['target_sales']
        X_test = X_direct_test[direct_cols]

        # fit the model
        self.model.fit(X_train,y_train)

        logger.info('Multi-horizon Direct forecast: Fit complete. Predicting')
        ## predict
        preds = self.model.predict(X_test)

        preds_df = pd.DataFrame({
                    'item_id':X_direct_test['item_id'].values,
                    'dept_id':X_direct_test['dept_id'].values if 'dept_id' in X_direct_test.columns else 'not_defined',
                    'cat_id' : X_direct_test['cat_id'].values if 'cat_id' in X_direct_test.columns else 'not_defined',
                    'date': X_direct_test['target_date'],
                    'sales_pred': preds['point'],
                })  
        for key, values in preds.items():  ## appending quantiles if enabled
                    if key == "point":
                        continue
                    preds_df[key] = values        
        logger.info(f'Multi-horizon direct forecat finished successfully...returning the predictions')
        
        return preds_df


if __name__ == '__main__':

    #### build train frame 
    pass
