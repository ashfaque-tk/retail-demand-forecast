import numpy as np
import pandas as pd

# Scale: per-item naive forecast error on training data
def scale(train_df):
    train_sorted = train_df.sort_values(['item_id', 'date'])
    diffs = train_sorted.groupby('item_id',observed=True)['sales'].diff()
    return train_sorted.assign(diff=diffs).groupby('item_id',observed=True).apply(
        lambda x: np.sqrt(np.mean(x['diff'].dropna()**2)),include_groups=False)
     # Series indexed by item_id, shape (n_items,)

# Forecast error: per-item RMSE over horizon
def forecast_error(test_df, pred_df):
    # pred_df: same structure as test_df but with 'sales' = predicted values
    merged = test_df[['item_id','date','sales']].merge(
        pred_df[['item_id','date','sales_pred']], on=['item_id','date'] )

    # assert test_df.shape[0] == pred_df.shape[0]

    return merged.groupby('item_id',observed=True).apply(lambda x: np.sqrt(np.mean((x['sales'] - x['sales_pred'])**2)),
                                                         include_groups = False)
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


def mae(ytrue, ypred):
    y, p = np.asarray(ytrue, float), np.asarray(ypred, float)
    return np.mean(np.abs(p - y))


def mae_dept(true_df,pred_df):

    true = true_df.groupby('dept_id',observed=True)['sales'].sum().reset_index()
    preds = pred_df.groupby('dept_id',observed=True)['sales_pred'].sum().reset_index()

    return np.mean(np.abs(true['sales']-preds['sales_pred']))


def mae_cat(true_df,pred_df):

    true = true_df.groupby('cat_id',observed=True)['sales'].sum().reset_index()
    preds = pred_df.groupby('cat_id',observed=True)['sales_pred'].sum().reset_index()

    return np.mean(np.abs(true['sales']-preds['sales_pred']))

def bias(ytrue, ypred):
    y, p = np.asarray(ytrue, float), np.asarray(ypred, float)
    return ((p.sum() - y.sum()) / y.sum())*100

def fva(baseline_wmape, model_wmape):
    return round((baseline_wmape - model_wmape)*100 / baseline_wmape,2)


def get_all_metrics(
        train_df:pd.DataFrame,
        test_df:pd.DataFrame,
        pred_df:pd.DataFrame,
        risk_period:int
    )->dict:

    '''return all metrics as a dict '''

    get_wrmsse = wrmsse(train_df,test_df,pred_df)
    get_mae = mae(ytrue=test_df['sales'],ypred=pred_df['sales_pred'])
    get_bias =  bias(ytrue=test_df['sales'],ypred=pred_df['sales_pred'])

    get_mae_dept = mae_dept(test_df,pred_df)
    get_mae_cat = mae_cat(test_df,pred_df)
    get_cum_erros = compute_cumulative_metrics(test_df,pred_df,tau=risk_period)

    

    return {'wrmsse':get_wrmsse,
            'MAE':get_mae,
            'BIAS%':get_bias,
            'cum_BIAS': get_cum_erros['cumulative_bias'].mean(),
            'cum_MAE' : get_cum_erros['cumulative_mae'].mean()}

import pandas as pd

import pandas as pd

def compute_cumulative_metrics(
    test_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    tau: int = 6
) -> pd.DataFrame:
    
    # 1. Ensure date types match
    test_df = test_df.copy()
    predicted_df = predicted_df.copy()
    test_df["date"] = pd.to_datetime(test_df["date"])
    predicted_df["date"] = pd.to_datetime(predicted_df["date"])
    
    # 2. Merge and drop any unaligned null rows immediately
    df = (
        test_df
        .merge(predicted_df, on=["item_id", "date"], how="inner")
        .dropna(subset=["sales", "sales_pred"])
        .sort_values(["item_id", "date"])
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("Merged DataFrame is empty! Check item_id and date column alignment.")

    # 3. Create tau_chunk index per item
    df["tau_chunk"] = df.groupby("item_id", observed=True).cumcount() // tau
    
    # 4. Running cumulative sums per tau chunk
    df["actual_cum"] = df.groupby(["item_id", "tau_chunk"], observed=True)["sales"].cumsum()
    df["forecast_cum"] = df.groupby(["item_id", "tau_chunk"], observed=True)["sales_pred"].cumsum()
    df["cum_error"] = df["forecast_cum"] - df["actual_cum"]
    
    # 5. First aggregate per tau chunk
    chunk_metrics = (
        df.groupby(["item_id", "tau_chunk"], observed=True)
        .agg(
            final_tau_error=("cum_error", "last"),
            mean_tau_tracking_error=("cum_error", lambda x: x.abs().mean())
        )
        .reset_index()
    )
    
    # 6. Average across all tau chunks per SKU
    sku_summary = (
        chunk_metrics.groupby("item_id", observed=True)
        .agg(
            cumulative_mae=("final_tau_error", lambda x: x.abs().mean()),
            cumulative_bias=("final_tau_error", "mean"),
            tracking_trajectory_mae=("mean_tau_tracking_error", "mean")
        )
        .reset_index()
    )

    return sku_summary