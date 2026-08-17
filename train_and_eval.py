"""
train_and_eval.py

Loads known/future parquet splits for a store, trains LightGBM models on
`known`, evaluates WRMSSE on the first `eval_days` of `future`, and logs
everything to MLflow.

Usage:
    uv run python train_and_eval.py --store_id CA_1
"""

import argparse
import os
import pandas as pd
import lightgbm as lgb
import mlflow

from src.pipeline import recursive_forecast_batch
from src.features import GetLagRollFeatures, get_avg_sales, get_price_features, get_known_future_features
from src.metrics import wrmsse

import warnings

warnings.filterwarnings('ignore')

# Code that triggers warnings runs silently

DATA_DIR = 'data/processed'
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5001')

CAT_COLS = ['item_id', 'dept_id', 'cat_id']
FEATURE_COLS = [
    'lag_7', 'lag_14', 'lag_28', 'rolling_mean_7', 'rolling_mean_28',
    'day_of_week', 'day_of_month', 'week_of_year', 'month', 'year',
    'sell_price', 'item_id', 'dept_id', 'cat_id'
]


def load_known_future(store_id):
    known = pd.read_parquet(f'{DATA_DIR}/sales_known_{store_id.lower()}.parquet')
    future = pd.read_parquet(f'{DATA_DIR}/sales_future_{store_id.lower()}.parquet')
    known['date'] = pd.to_datetime(known['date'])
    future['date'] = pd.to_datetime(future['date'])
    return known.sort_values(['item_id', 'date']), future.sort_values(['item_id', 'date'])


def build_train_features(known):
    """Features for training"""
    get_lag_roll_features = GetLagRollFeatures()

    known_lag =  get_lag_roll_features.add_lags(known,[1,2,3,7,21,28])
    known_rolling = get_lag_roll_features.add_rolling_max(known_lag,windows=[3,7,14,21,28,60,90])
    known_rolling = get_lag_roll_features.add_rolling_mean(known_rolling,windows=[3,7,14,21])
    known_rolling = get_lag_roll_features.add_rolling_on_lag(known_rolling,lags=[7,28],windows=[7,28])

    return known_rolling

def build_eval_features(future, eval_days=28):
    """
    Just slices the first `eval_days` of future — static columns and REAL
    sales values only. No lag/rolling features here; those are computed
    recursively inside recursive_forecast_batch, using history (which
    already includes all of `known` via train_df).
    """
    eval_start = future['date'].min()
    eval_end = eval_start + pd.Timedelta(days=eval_days - 1)
    return future[(future['date'] >= eval_start) & (future['date'] <= eval_end)].copy()


def train_models(train_df,test_df,quantiles = False, feature_cols=FEATURE_COLS):

    train_df = train_df.dropna(subset=feature_cols).copy()

    # 1. Assert that zero NaNs remain anywhere in the feature columns
    assert (
        train_df[feature_cols].isna().sum().sum() == 0
    ), 'AssertionError: NaNs still detected in feature columns!'

    # 2. Optional safety check: ensure your drop filter didn't wipe out the whole dataframe
    assert not train_df.empty, (
        'AssertionError: train_df is empty after dropping NaNs (check if your lookback window is too large)!'
    )
    
    cat_categories = {}
    for col in CAT_COLS:
        cats = sorted(train_df[col].dropna().unique().tolist())
        train_df[col] = pd.Categorical(train_df[col], categories=cats)
        cat_categories[col] = cats


    X_train = train_df[feature_cols]
    y_train = train_df['sales']

    X_val = test_df[feature_cols]
    y_val = test_df['sales']
    
    params = dict(n_estimators=800, learning_rate=0.05, num_leaves=31,
                    subsample=0.8,  # row bagging
                    colsample_bytree=0.7,  # feature bagging (prevents reliance on a single dominant feature)
                    min_child_samples=50,
                    verbose=-1)

    model_point = lgb.LGBMRegressor(objective='tweedie', **params)
    model_point.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)],categorical_feature=CAT_COLS)   
    
    if quantiles:
        print("quantile models training.")
        model_q10 = lgb.LGBMRegressor(objective='quantile', alpha=0.1, **params)
        model_q10.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)], categorical_feature=CAT_COLS)

        model_q90 = lgb.LGBMRegressor(objective='quantile', alpha=0.9, **params)
        model_q90.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)], categorical_feature=CAT_COLS)
    else:
        model_q10 = None 
        model_q90 = None 

    return {'point': model_point, 'q10': model_q10, 'q90': model_q90}, cat_categories, params


def predict_eval_set(models, train_df, eval_df, cat_categories):
    history = train_df[['item_id','date','sales']].copy()
    predict_df= recursive_forecast_batch(models,history,eval_df,FEATURE_COLS,cat_categories)
    return predict_df[['item_id','date','sales_pred','q10','q90']]

    for item_id in eval_df['item_id'].unique():
        item_history = train_df[train_df['item_id'] == item_id][['item_id', 'date', 'sales']]
        item_future_static = eval_df[eval_df['item_id'] == item_id].copy()
        if item_history.empty or item_future_static.empty:
            continue
        results = recursive_forecast_batch(models, item_history, item_future_static, FEATURE_COLS, cat_categories)
        for r in results:
            all_preds.append({'item_id': item_id, 'date': r['date'], 'sales_pred': r['sales_pred']})
    return pd.DataFrame(all_preds)


def log_to_mlflow(store_id, models, params, score, cat_categories, training_cutoff, eval_start, eval_end,model_version, run_name=None):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(f"{store_id.lower()}-demand-forecast")

    with mlflow.start_run(run_name=run_name or f"{store_id}-v1"):
        
        mlflow.set_tag("training_cutoff",str(training_cutoff))
        mlflow.set_tag("evaluation_perios",f"{eval_start}-{eval_end}")
        mlflow.log_params(params)
        mlflow.log_param("store_id", store_id)
        mlflow.log_metric("wrmsse", score)

        for name, model in models.items():
            mlflow.lightgbm.log_model(model, name=f"model_{name}")

        mlflow.log_dict(cat_categories, "cat_categories.json")
        print(f"[{store_id}] Logged run with WRMSSE = {score:.4f}")


def run(store_id, eval_days=28, run_name=None):
    print(f"[{store_id}] Loading known/future splits...")
    known, future = load_known_future(store_id)

    print(f"[{store_id}] Building training features...")
    train_df = build_train_features(known)

    print(f"[{store_id}] Training models...")
    models, cat_categories, params = train_models(train_df)

    print(f"[{store_id}] Building eval features (first {eval_days} days of future)...")
    eval_df = build_eval_features(future, eval_days)

    print(f"[{store_id}] Predicting eval window...")
    pred_df = predict_eval_set(models, train_df, eval_df, cat_categories)

    print(f"[{store_id}] Computing WRMSSE...")
    score = wrmsse(train_df, eval_df, pred_df)
    print(f"[{store_id}] WRMSSE: {score:.4f}")

    trained_on = known['date'].max()
    eval_start = future['date'].min()
    eval_end = future['date'].max()

    log_to_mlflow(store_id, models, params, score, cat_categories,trained_on,eval_start,eval_end, run_name)

    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--store_id', default='CA_1')
    parser.add_argument('--eval_days', type=int, default=28)
    args = parser.parse_args()

    run(args.store_id, args.eval_days)