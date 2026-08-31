'''build baselines to compare ml models against.
two baselines:
    seasonal_naive: predicts sales on date d using the item's own sales on (d - lag_days)
    simple_moving_average: predicts a flat value per item — the item's mean sales over
        the last `window_days` of train — applied across every date in the test horizon'''

import pandas as pd
import numpy as np
from scipy.stats import norm


def _gaussian_quantiles(point_pred: pd.Series, spread: pd.Series,
                         quantiles=(0.10, 0.50, 0.90, 0.95, 0.97)) -> pd.DataFrame:
    """
    point_pred, spread: aligned Series — spread must already be per-row (e.g. per-item
    historical std broadcast to match point_pred's index), never a single global scalar.
    """
    if not isinstance(spread, pd.Series):
        raise TypeError(
            f"_gaussian_quantiles: spread must be a per-row pd.Series, got {type(spread)}. "
            f"A scalar would silently broadcast the same spread to every row."
        )
    if not point_pred.index.equals(spread.index):
        raise ValueError(
            "_gaussian_quantiles: point_pred and spread have mismatched indices — "
            "arithmetic would silently misalign or produce NaNs."
        )

    out = {}
    for q in quantiles:
        z = norm.ppf(q)
        out[f'q{int(q * 100)}'] = (point_pred + z * spread).clip(lower=0)
    return pd.DataFrame(out, index=point_pred.index)


def seasonal_naive(train_df: pd.DataFrame, test_df: pd.DataFrame, lag_days: int = 28) -> pd.DataFrame:
    '''
    Predicts sales on test date d using the item's own realized sales on (d - lag_days).
    Default lag_days=28 matches this project's forecast horizon. (Your original file had
    two conflicting defaults — "last week" in the module docstring, a 28-day window in the
    function body — so this is now an explicit parameter instead of implicit either way.)

    return: item_id, date, sales_pred, real_sales, q10/q50/q90/q95/q97
    '''
    train = train_df[['item_id', 'date', 'sales']].copy()
    test = test_df[['item_id', 'date']].copy()

    test['source_date'] = test['date'] - pd.Timedelta(days=lag_days)

    pred_df = test.merge(
        train.rename(columns={'date': 'source_date', 'sales': 'sales_pred'}),
        on=['item_id', 'source_date'],
        how='left'
    )

    missing = pred_df['sales_pred'].isna().sum()
    if missing > 0:
        raise ValueError(
            f"seasonal_naive: {missing} rows have no matching source date "
            f"(train history doesn't reach back {lag_days} days for some test dates/items)."
        )

    item_std = train.groupby('item_id', observed=True)['sales'].std().rename('item_std')
    pred_df = pred_df.merge(item_std, on='item_id', how='left')
    pred_df['item_std'] = pred_df['item_std'].fillna(0)

    quantiles = _gaussian_quantiles(pred_df['sales_pred'], pred_df['item_std'])
    pred_df = pd.concat([pred_df, quantiles], axis=1)

    pred_df = pred_df.merge(
        test_df[['item_id', 'date', 'sales']].rename(columns={'sales': 'real_sales'}),
        on=['item_id', 'date'], how='left'
    )

    return pred_df.drop(columns=['source_date'])


def simple_moving_average(train_df: pd.DataFrame, test_df: pd.DataFrame, window_days: int = 180) -> pd.DataFrame:
    '''
    Predicts a flat value per item — the item's mean sales over the last `window_days` of
    train — applied to every date in the test horizon. Not date-shifted like seasonal_naive,
    since a moving average has no seasonal alignment to preserve.
    window_days=180 ~ 6 months, 365 ~ 12 months.

    return: item_id, date, sales_pred, real_sales, q10/q50/q90/q95/q97
    '''
    train = train_df[['item_id', 'date', 'sales']].copy()
    cutoff = train['date'].max() - pd.Timedelta(days=window_days)
    train_window = train[train['date'] > cutoff]

    item_stats = train_window.groupby('item_id', observed=True)['sales'].agg(
        sales_pred='mean', item_std='std'
    ).reset_index()
    item_stats['item_std'] = item_stats['item_std'].fillna(0)

    test = test_df[['item_id', 'date']].copy()
    pred_df = test.merge(item_stats, on='item_id', how='left')

    missing = pred_df['sales_pred'].isna().sum()
    if missing > 0:
        raise ValueError(
            f"simple_moving_average: {missing} test rows have no matching item_id "
            f"in the last {window_days} days of train (new/unseen items)."
        )

    quantiles = _gaussian_quantiles(pred_df['sales_pred'], pred_df['item_std'])
    pred_df = pd.concat([pred_df, quantiles], axis=1)

    pred_df = pred_df.merge(
        test_df[['item_id', 'date', 'sales']].rename(columns={'sales': 'real_sales'}),
        on=['item_id', 'date'], how='left'
    )

    return pred_df