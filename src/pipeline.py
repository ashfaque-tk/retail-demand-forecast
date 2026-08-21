'''functions for recursive forecasting: single item and batch'''
from typing import Dict, List


import pandas as pd
import numpy as np
from .features import get_known_future_features, GetLagRollFeatures, get_trend_features

# def recursive_forecast(models, history, future_static, feature_cols, cat_categories, id_col='item_id'):
#     """models: dict with 'point', 'q10', 'q90' fitted LGBMRegressors."""
#     history = history.copy()
#     rows = []

#     for _, day_row in future_static.sort_values('date').iterrows():
#         placeholder = pd.DataFrame({id_col: [day_row[id_col]], 'date': [day_row['date']], 'sales': [np.nan]})
#         hist_ext = get_lag_features(pd.concat([history, placeholder], ignore_index=True), id_col)
#         today_feats = hist_ext[hist_ext['date'] == day_row['date']].iloc[-1]

#         row_dict = day_row.to_dict()
#         row_dict.update({col: today_feats[col] for col in
#                           ['lag_7','lag_14','lag_28','rolling_mean_7','rolling_mean_28']})
#         row = pd.DataFrame([row_dict])
        
#         for col in ['lag_7', 'lag_14', 'lag_28', 'rolling_mean_7', 'rolling_mean_28']:
#             row[col] = today_feats[col]
#         for col, cats in cat_categories.items():
#             row[col] = pd.Categorical(row[col], categories=cats)
#         row = row[feature_cols]

#         point = max(models['point'].predict(row)[0], 0)
#         q10 = max(models['q10'].predict(row)[0], 0)
#         q90 = max(models['q90'].predict(row)[0], 0)

#         rows.append({'date': day_row['date'], 'sales_pred': point, 'q10': min(q10, point), 'q90': max(q90, point)})
#         history = pd.concat([history, pd.DataFrame({id_col: [day_row[id_col]], 'date': [day_row['date']], 'sales': [point]})], ignore_index=True)

#     return rows

from typing import Dict, List, Optional
import pandas as pd
import numpy as np

def recursive_forecast_batch(
    models: Dict,
    history_df: pd.DataFrame,
    future_static_df: pd.DataFrame,
    feature_cols: List[str],
    cat_categories: Dict,
    lags: List[int] = [1, 2, 3, 7, 14, 21],
    rolling_mean_windows: List[int] = [3, 7, 14, 21, 28, 60, 90],
    rolling_max_windows: List[int] = [2, 3, 7, 14, 21, 28, 60, 90],
    id_col: str = 'item_id', max_lookback_days:int=100
):
    '''Robust vectorized recursive forecasting loop across all items.'''

    history = history_df[[id_col, 'date', 'sales']].copy()
    all_results = []
    features_list = []

    # Keep a generous lookback buffer for max rolling windows (e.g., 120 days)
    
    cutoff_date = history['date'].max() - pd.Timedelta(days=max_lookback_days)
    history_slice = history[history['date'] >= cutoff_date].copy()

    dates = sorted(future_static_df['date'].unique())
    get_features = GetLagRollFeatures()
  
    for current_date in dates:
        # 1. Isolate static features for the current day
        day_static = future_static_df[future_static_df['date'] == current_date].copy()

        # 2. Create placeholder row for today with NaN sales
        placeholders = day_static[[id_col]].drop_duplicates().copy()
        placeholders['date'] = current_date
        placeholders['sales'] = np.nan

        # 3. Append to history slice and sort strictly
        history_slice = pd.concat([history_slice, placeholders], ignore_index=True)
        history_slice = history_slice.sort_values([id_col, 'date']).reset_index(drop=True)

        # 4. Compute lags and rolling features dynamically on the growing history
        if lags:
            history_slice = get_features.add_lags(history_slice, lags=lags)
        if rolling_max_windows:
            history_slice = get_features.add_rolling_max(history_slice, source_col='lag_1', windows=rolling_max_windows)
        if rolling_mean_windows:
            history_slice = get_features.add_rolling_mean(history_slice, source_col='lag_1', windows=rolling_mean_windows)
        
        # Optional advanced features (wrapped in try/except to prevent abrupt crashes)
        try:
            history_slice = get_features.add_rolling_on_lag(history_slice, lags=[28], windows=[7, 28])
        except Exception:
            pass

        # 5. Extract today's calculated features
        today_feats = history_slice[history_slice['date'] == current_date]

        # 6. Merge static features with today's dynamic recursive features safely
        row = day_static.merge(today_feats, on=[id_col, 'date'], how='left', suffixes=('', '_dup'))
        row = row.loc[:, ~row.columns.duplicated()] # Drop duplicate columns if any
    
        try:
            row= get_trend_features(row)
        except Exception as e:
            print(e)
            break
            
        
        # 7. SAFETY NET: Ensure every feature expected by the model exists in the row
        for col in feature_cols:
            if col not in row.columns:
                row[col] = 0.0  # Fallback zero instead of crashing with missing column error

        # 8. Handle categorical features
        for col, cats in cat_categories.items():
            if col in row.columns:
                row[col] = pd.Categorical(row[col], categories=cats)

        row_X = row[feature_cols]
        features_list.append(row.copy())

        # 9. Model Predictions
        point = np.maximum(models['point'].predict(row_X), 0)

        if models.get('q10') is None or models.get('q90') is None:
            q10 = np.zeros_like(point)
            q90 = np.zeros_like(point)
        else:
            q10 = np.maximum(models['q10'].predict(row_X), 0)
            q90 = np.maximum(models['q90'].predict(row_X), 0)

        day_result = pd.DataFrame({
            id_col: row[id_col].values,
            'date': current_date,
            'sales_pred': point,
            'q10': np.minimum(q10, point),
            'q90': np.maximum(q90, point),
        })
        all_results.append(day_result)

        # 10. Update history_slice sales for current_date using the *predicted* values
        pred_map = dict(zip(row[id_col], point))
        mask = (history_slice['date'] == current_date)
        history_slice.loc[mask, 'sales'] = history_slice.loc[mask, id_col].map(pred_map).fillna(history_slice.loc[mask, 'sales'])

    item_df = pd.concat(features_list, ignore_index=True) if features_list else pd.DataFrame()
    final_results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    return final_results, item_df



def cost_per_item(df_test,preds):
    '''pred: must contain point forecast and quantile forecasts'''
    

    item_data = df_test[['item_id','date','sell_price','sales']].copy()

    item_data = item_data.merge(preds,on=['item_id','date'],how='left')

    item_data['cost'] = 0.60*item_data['sell_price']
    item_data['profit']  = 0.40*item_data['sell_price']
    item_data['holding_cost'] = (0.25/365)*item_data['cost']
    item_data['stockout_cost'] = item_data['profit']

    item_data['safety_stock'] = item_data['q90'] - item_data['sales_pred']

    item_data['cost_holding'] = item_data['holding_cost']*item_data['safety_stock']
    item_data['expected_shortage'] = (item_data['sales'] - item_data['q90']).clip(lower=0)

    item_data['stockout_cost'] = item_data['stockout_cost']*item_data['expected_shortage']
    item_data['total_cost'] = item_data['cost_holding']  + item_data['stockout_cost']

    return item_data.groupby('item_id')['total_cost'].sum().reset_index()