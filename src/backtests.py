'''backtesting functions with rolling and expanding windows for recursive forecasting.'''

import pandas as pd
from src.metrics import wrmsse
from train_and_eval import build_train_features, predict_eval_set, train_models

def split_data(df, start_date, end_date, forecast_horizon=28):
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








def walk_forward_rolling_window(df,training_window=365,horizon=28):

    total_days_available = (df['date'].max()-df['date'].min()).days
    total_num_windows = int((total_days_available/horizon))                         
    first_window_start = df['date'].min()

    all_windows_start = [first_window_start + pd.Timedelta(days=n*horizon) for n in range(total_num_windows)]

    # selecting only a few end dates for faster calculations that spans different windows in a year
    selected_start_days = [first_window_start]

    for i,date in enumerate(all_windows_start):
        selected = selected_start_days[-1]
        if  (date-selected).days > 120:# 4 month
            if ((date+pd.Timedelta(days=training_window))< df['date'].max()):
                selected_start_days.append(date)

    print('total windows: ',total_num_windows,'selected starts: ',selected_start_days,len(selected_start_days))

    windows = {}
    metric_list = {}
    model_list = {}
    prediction_list = {}
    # starting walkforward validation
    start_date = df['date'].min()

    for i,start in enumerate(selected_start_days):
        # split the training and test datas
        end_date = start + pd.Timedelta(days=int(training_window))
        assert (end_date-start).days == training_window

        train, test =  split_data(df,start, end_date)
      
        print(f"testing window:{i+1}")
        train_feat = build_train_features(train)

        models, cat_categories, params = train_models(train_feat)
        pred_df = predict_eval_set(models,train_feat,test,cat_categories)
        # print('prediction dates', pred_df['date'].unique())
        # break
        score = wrmsse(train_feat,test,pred_df)

        print(f"Training period: {start}-{end_date} ({(end_date-start).days}) days, WRMSSE:  ",score)
        key = f'window_{i+1}' 
        windows[key] = start
        model_list[key] = models
        prediction_list[key] = pred_df
        metric_list[key] = score
    return windows,metric_list,model_list,prediction_list


def walk_forward_expanding_window(df,training_window=365,horizon=28):

    total_days_available = (df['date'].max()-df['date'].min()).days
    total_num_windows = int((total_days_available - training_window)/horizon)
    first_window_end =  df['date'].min() + pd.Timedelta(days=training_window)

    all_windows_end = [first_window_end + pd.Timedelta(days=n*horizon) for n in range(total_num_windows)]
    # selecting only a few end dates for faster calculations that spans different windows in a year
    selected_end_days = [first_window_end]

    for i,date in enumerate(all_windows_end):
        selected = selected_end_days[-1]
        if  (date-selected).days > 120:# 4 month
            selected_end_days.append(date)

    windows = {}
    metric_list = {}
    model_list = {}
    prediction_list = {}

    # starting walkforward validation
    start_date = df['date'].min()

    for i,date in enumerate(selected_end_days):
        # split the training and test datas
        train, test =  split_data(df,start_date, date)
        # building features static

        print(f"testing window:{i+1}")
        train_feat = build_train_features(train)

        models, cat_categories, params = train_models(train_feat)
        pred_df = predict_eval_set(models,train_feat,test,cat_categories)
        # print('prediction dates', pred_df['date'].unique())
        # break
        score = wrmsse(train_feat,test,pred_df)
        print(f"Training period: {start_date}-{date}, WRMSSE:  ",score)
        key = f'window_{i+1}' 

        windows[key] = date
        model_list[key] = models
        prediction_list[key] = pred_df
        metric_list[key] = score

    return windows,metric_list,model_list,prediction_list