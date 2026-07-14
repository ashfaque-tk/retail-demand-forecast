import pandas as pd
import numpy as np

def get_lag_rolling_features(df, id_col='item_id'):
    df = df.sort_values([id_col, 'date'])
    for lag in [7, 14, 28]:
        df[f'lag_{lag}'] = df.groupby(id_col, observed=True)['sales'].shift(lag)
    sales_shifted = df.groupby(id_col, observed=True)['sales'].shift(1)
    for window in [7, 28]:
        df[f'rolling_mean_{window}'] = (
            sales_shifted.groupby(df[id_col], observed=True)
            .rolling(window).mean().reset_index(level=0, drop=True)
        )
    return df
def get_known_future_features(item_id, dates, calendar_df, price_df, item_meta):
    cal = calendar_df[calendar_df['date'].isin(dates)].copy()
    cal['item_id'] = item_id
    cal['dept_id'] = item_meta['dept_id']
    cal['cat_id'] = item_meta['cat_id']
    cal = cal.merge(price_df[['wm_yr_wk', 'sell_price']], on='wm_yr_wk', how='left')

    # derive calendar features the same way the training notebook did
    cal['day_of_week']  = cal['date'].dt.dayofweek
    cal['day_of_month'] = cal['date'].dt.day
    cal['week_of_year'] = cal['date'].dt.isocalendar().week.astype(int)
    cal['month'] = cal['date'].dt.month
    cal['year']  = cal['date'].dt.year

    return cal

def recursive_forecast(models, history, future_static, feature_cols, cat_categories, id_col='item_id'):
    """models: dict with 'point', 'q10', 'q90' fitted LGBMRegressors."""
    history = history.copy()
    rows = []

    for _, day_row in future_static.sort_values('date').iterrows():
        placeholder = pd.DataFrame({id_col: [day_row[id_col]], 'date': [day_row['date']], 'sales': [np.nan]})
        hist_ext = get_lag_rolling_features(pd.concat([history, placeholder], ignore_index=True), id_col)
        today_feats = hist_ext[hist_ext['date'] == day_row['date']].iloc[-1]

        row_dict = day_row.to_dict()
        row_dict.update({col: today_feats[col] for col in
                          ['lag_7','lag_14','lag_28','rolling_mean_7','rolling_mean_28']})
        row = pd.DataFrame([row_dict])
        
        for col in ['lag_7', 'lag_14', 'lag_28', 'rolling_mean_7', 'rolling_mean_28']:
            row[col] = today_feats[col]
        for col, cats in cat_categories.items():
            row[col] = pd.Categorical(row[col], categories=cats)
        row = row[feature_cols]

        point = max(models['point'].predict(row)[0], 0)
        q10 = max(models['q10'].predict(row)[0], 0)
        q90 = max(models['q90'].predict(row)[0], 0)

        rows.append({'date': day_row['date'], 'sales_pred': point, 'q10': min(q10, point), 'q90': max(q90, point)})
        history = pd.concat([history, pd.DataFrame({id_col: [day_row[id_col]], 'date': [day_row['date']], 'sales': [point]})], ignore_index=True)

    return rows