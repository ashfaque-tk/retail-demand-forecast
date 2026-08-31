'''functions to return rolling and expanding windows for backtesting expts.'''

import pandas as pd
from src.metrics import wrmsse
from train_and_eval import build_train_features, predict_eval_set, train_models

from typing import tuple,List

def split_data(df:pd.DataFrame, start_date:pd.DatetimeIndex, end_date:pd.DatetimeIndex, forecast_horizon=28)->pd.DataFrame:
    """
    Splits a DataFrame into training and testing windows based on dates.
    Encloses bitwise conditions in parentheses to prevent Operator Precedence TypeErrors.
    """
    # 1. Enclose each comparison in parentheses () before applying bitwise &
    train_mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    
    test_start = end_date
    test_end = end_date + pd.Timedelta(days=int(forecast_horizon))
    test_mask = (df['date'] > test_start) & (df['date'] <= test_end)

    train_ = df[train_mask].copy()
    test_ = df[test_mask].copy()

    return train_, test_

def generate_rolling_windows(df, training_window=365, horizon=28, step_size=120, date_col='date'):
    """
    Generates a list of fixed-size rolling window date ranges for walk-forward validation.
    """
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    
    windows = []
    window_id = 1
    
    current_train_start = min_date
    
    while True:
        current_train_end = current_train_start + pd.Timedelta(days=int(training_window))
        test_start = current_train_end
        test_end = test_start + pd.Timedelta(days=int(horizon))
        
        # Stop generating windows if the test period extends beyond available data
        if test_end > max_date:
            break        
        windows.append({
            'window_id': f"window_{window_id}",
            'train_start': current_train_start,
            'train_end': current_train_end,
            'test_start': test_start,
            'test_end': test_end
        })  
        window_id += 1
        current_train_start += pd.Timedelta(days=int(step_size))     
    return windows


def generate_expanding_windows(df, training_window=365, horizon=28, step_size=120, date_col='date'):
    """
    Generates a list of expanding window date ranges for walk-forward validation.
    """
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    
    windows = []
    window_id = 1
    
    # Train start is fixed to the absolute beginning of dataset
    train_start = min_date
    current_train_end = min_date + pd.Timedelta(days=int(training_window))
    
    while True:
        test_start = current_train_end
        test_end = test_start + pd.Timedelta(days=int(horizon))
        
        # Stop generating windows if the test period extends beyond available data
        if test_end > max_date:
            break
            
        windows.append({
            'window_id': f"window_{window_id}",
            'train_start': train_start,
            'train_end': current_train_end,
            'test_start': test_start,
            'test_end': test_end
        })
        
        window_id += 1
        current_train_end += pd.Timedelta(days=int(step_size))
        
    return windows
