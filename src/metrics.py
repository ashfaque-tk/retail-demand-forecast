import numpy as np
import pandas as pd

'''wrmsse metric '''

# Scale: per-item naive forecast error on training data
def scale(train_df):
    train_sorted = train_df.sort_values(['item_id', 'date'])
    diffs = train_sorted.groupby('item_id',observed=True)['sales'].diff()
    return train_sorted.assign(diff=diffs).groupby('item_id',observed=True).apply(
        lambda x: np.sqrt(np.mean(x['diff'].dropna()**2))
    )  # Series indexed by item_id, shape (n_items,)

# Forecast error: per-item RMSE over horizon
def forecast_error(test_df, pred_df):
    # pred_df: same structure as test_df but with 'sales' = predicted values
    merged = test_df[['item_id','date','sales']].merge(
        pred_df[['item_id','date','sales_pred']], on=['item_id','date'] )

    # assert test_df.shape[0] == pred_df.shape[0]

    return merged.groupby('item_id',observed=True).apply(lambda x: np.sqrt(np.mean((x['sales'] - x['sales_pred'])**2)),include_groups = False)
    # Series indexed by item_id
    # return np.sqrt(np.mean(test_df['sales']-pred_df)**2)

# Weights: last 28 days of training revenue share
def weights(train_df):
    last28 = train_df['date'] >= train_df['date'].max() - pd.Timedelta(days=27)
    rev = (train_df[last28]
           .assign(revenue=lambda x: x['sales'] * x['sell_price'])
           .groupby('item_id',observed=True)['revenue'].sum())
    return rev / rev.sum()

# Final WRMSSE
def wrmsse(train_df, test_df, pred_df):
    
    w = weights(train_df)
    s = scale(train_df)
    fe = forecast_error(test_df, pred_df)
    # align all three on item_id
    items = w.index
    return float(np.sum(w.loc[items] * fe.loc[items] / s.loc[items]))


def calculate_wape(y_true, y_pred):
    """Calculates weighted absolute percentage error (WAPE)"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    sum_actuals = np.sum(y_true)

    if sum_actuals == 0:
        return np.nan

    return np.sum(np.abs(y_true - y_pred)) / sum_actuals


def calculate_mape(y_true, y_pred, ignore_zeros = True):
    '''Calculates Mean Absolute  Percentage Error (MAPE)
    Filters out y_true = 0 by default to prevent division by zero '''

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if ignore_zeros:
        mask = y_true>0
        if not np.any(mask):
            return np.nan 
        return np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100 
    else:
        return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-5))) * 100


def mae(ytrue,ypred):
    y,p = np.asarray(ytrue,float),np.asarray(ypred,float)

    return abs(p.sum()-y.sum())/y.sum()


def bias(ytrue, ypred):
    y, p = np.asarray(ytrue, float), np.asarray(ypred, float)
    return (p.sum() - y.sum()) / y.sum()

def fva(baseline_wmape, model_wmape):
    return (baseline_wmape - model_wmape) / baseline_wmape