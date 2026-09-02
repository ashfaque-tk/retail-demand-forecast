"""
A single pipeline:
    accept cleaned data -> load saved train/test data + final feature set -> run backtest
    windows (baselines + model) -> log metrics (MAE, BIAS%, WRMSSE, FVA) as tidy DataFrames.
"""
from __future__ import annotations

import json
from datetime import datetime

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import PIPELINE_CONFIG
from src.baselines import seasonal_naive, simple_moving_average
from src.metrics import get_all_metrics, fva
from src.features import FeatureBuilder
from src.models_train import SelectModel, available_models
from src.backtest_windows import generate_rolling_windows, generate_expanding_windows, split_data
from src.utils import get_items_with_min_history
from src.recursive_model import Forecaster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TRAIN_PATH = PIPELINE_CONFIG['train_data_path']
TEST_PATH = PIPELINE_CONFIG['test_data_path']
FEATURE_PATH = PIPELINE_CONFIG['final_feature_set']

RUN = PIPELINE_CONFIG['run']
MODEL_NAME = PIPELINE_CONFIG['model']
FORECAST_TYPE = PIPELINE_CONFIG['forecast_type']
TRAINING_WINDOW = PIPELINE_CONFIG['training_window']
HORIZON = PIPELINE_CONFIG['horizon_days']
BACKTEST_TYPE = PIPELINE_CONFIG['backtest_mode']
CATEGORICAL_COLS = PIPELINE_CONFIG['categorical_cols']

# None = run every backtest window. Int (e.g. 1) = debug on the first N windows only.
# This replaces the old unconditional `break` after window 0 -- same effective
# behavior right now (defaults to 1), but explicit and tunable instead of silent.
MAX_WINDOWS = PIPELINE_CONFIG.get('max_windows', 1)


def run_backtest_window(window_id: int, wnd: dict, train: pd.DataFrame, full_features: list,
                         feat_builder: FeatureBuilder, selected_models: SelectModel,
                         forecaster: Forecaster) -> tuple[list[dict], list[dict]]:
    """Run baselines + model for a single backtest window.
    Returns (metric_rows, fva_rows) -- lists of flat dicts, ready for pd.DataFrame.
    """
    train_start, train_end = wnd['train_start'], wnd['train_end']
    eval_start, eval_end = wnd['test_start'], wnd['test_end']

    full_train = train[train['date'].between(train_start, eval_end)]
    full_train_selected_items = get_items_with_min_history(full_train, min_history_days=100)
    full_train_price_feats = feat_builder.add_price_features(full_train_selected_items)
    train_wnd, eval_wnd = split_data(df=full_train_price_feats, start_date=train_start, end_date=train_end)

    logger.info(
        "window %d: train %s -> %s (%d items), eval %s -> %s",
        window_id, train_start.date(), train_end.date(),
        train_wnd['item_id'].nunique(), eval_start.date(), eval_end.date(),
    )

    # ---------- baselines ----------
    baseline_naive = seasonal_naive(train_wnd, eval_wnd)
    baseline_ma = simple_moving_average(train_wnd, eval_wnd, window_days=180)

    metrics_naive = get_all_metrics(train_wnd, eval_wnd, baseline_naive)
    metrics_ma = get_all_metrics(train_wnd, eval_wnd, baseline_ma)

    # ---------- ML model ----------
    train_wnd = feat_builder.build(
        df=train_wnd, lags=[7, 28, 60, 90], mean_windows=[7, 28, 60, 90],
        max_windows=[7, 28, 60, 90], rolling_on_lags=[28],
    )

    missing = [col for col in full_features + ['date', 'sales'] if col not in train_wnd.columns]
    if missing:
        raise ValueError(f"window {window_id}: missing features in train_wnd: {missing}")

    selected_models.fit(train_wnd[full_features], train_wnd['sales'])
    preds_df = forecaster.recursive_forecaster(train_wnd, eval_wnd)
    metrics_model = get_all_metrics(train_wnd, eval_wnd, preds_df)

    metric_rows = [
        {'window_id': window_id, 'train_start': train_start.date(), 'train_end': train_end.date(),
         'model': label, 'MAE': m['MAE'], 'BIAS%': m['BIAS%'], 'wrmsse': m['wrmsse'],'MAE-DEPT':m['MAE-DEPT-AGG'],
         'MAE-CAT':m['MAE-CAT-AGG']}
        for label, m in [
            ('seasonal_naive', metrics_naive),
            ('moving_average', metrics_ma),
            (MODEL_NAME, metrics_model),
        ]
    ]

    fva_rows = [
        {
            'window_id': window_id, 'metric': metric_key,
            'fva_moving_average_vs_naive': fva(metrics_naive[metric_key], metrics_ma[metric_key]),
            'fva_model_vs_naive': fva(metrics_naive[metric_key], metrics_model[metric_key]),
        }
        for metric_key in ['MAE', 'wrmsse']
    ]

    return metric_rows, fva_rows

RESULTS_DIR = Path(PIPELINE_CONFIG['results_dir'])
EXPERIMENTS_LOG_PATH = RESULTS_DIR / 'Experiments/Experiments.json'
DEPLOY_DIR = RESULTS_DIR / 'Deploy'


def _json_safe(obj):
    """Recursively convert an object into something json.dump can handle,
    without lying about the data: DataFrames become row-records, timestamps
    become ISO strings, numpy scalars become plain python scalars. Anything
    genuinely unrecognized is left alone and caught by json.dump's own
    TypeError (via default=str as a last-resort net), not silently dropped.
    """
    if isinstance(obj, pd.DataFrame):
        return json.loads(obj.to_json(orient='records', date_format='iso'))
    if isinstance(obj, pd.Series):
        return _json_safe(obj.to_dict())
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    return obj


def log_experiment_results(results_path: Path, record: dict) -> Path:
    """Append one experiment record to a JSON list at results_path, creating
    the file/dir on first use. Never overwrites -- refuses to append onto a
    file that isn't already a JSON list, rather than silently clobbering it.
    """
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)
        if not isinstance(existing, list):
            raise ValueError(
                f"{results_path} exists but isn't a JSON list -- refusing to "
                f"append blindly, since that would silently corrupt it."
            )
    else:
        existing = []

    existing.append(_json_safe(record))

    with open(results_path, 'w') as f:
        json.dump(existing, f, indent=2, default=str)

    logger.info("Logged experiment record to %s (%d total entries)", results_path, len(existing))
    return results_path


# def save_deploy_result(results_dir: Path, model_name: str, record: dict) -> Path:
#     """One timestamped file per deploy run, e.g. Results/Deploy/lgbm_20260901_121535.json."""
#     results_dir = Path(results_dir)
#     results_dir.mkdir(parents=True, exist_ok=True)

#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     out_path = results_dir / f"{model_name}_{timestamp}.json"

#     with open(out_path, 'w') as f:
#         json.dump(_json_safe(record), f, indent=2, default=str)

#     logger.info("Saved deploy result to %s", out_path)
#     return out_path


def main():
    t0 = time.time()

    logger.info("[1/3] Loading data + validating")
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)  # stress testing set, 137 days
    full_features = list(pd.read_pickle(FEATURE_PATH))
    logger.info("Feature set loaded: %d features", len(full_features))

    ### TODO: validate(train), validate(test) -- NaNs, negative sales/sell_price,
    ### required categorical + calendar columns present, etc.

    if RUN != "Experiment":
        logger.info("RUN=%s -- nothing configured to run for this mode yet.", RUN)
        return

    logger.info("[2/3] Generating %s backtest windows", BACKTEST_TYPE)
    if BACKTEST_TYPE == 'rolling':
        backtest_windows = generate_rolling_windows(
            train, training_window=TRAINING_WINDOW, horizon=HORIZON, step_size=120
        )
    elif BACKTEST_TYPE == 'expanding':
        backtest_windows = generate_expanding_windows(
            train, training_window=TRAINING_WINDOW, horizon=HORIZON, step_size=120
        )
    else:
        raise ValueError(f"Unknown backtest_mode: {BACKTEST_TYPE!r}")
    logger.info("Total windows: %d", len(backtest_windows))

    feat_builder = FeatureBuilder()
    selected_models = SelectModel(
        model=MODEL_NAME, quantiles=None, use_log_transform=False, categorical_cols=CATEGORICAL_COLS
    )
    forecaster = Forecaster(selected_models, feat_builder)

    windows_to_run = backtest_windows[:MAX_WINDOWS] if MAX_WINDOWS else backtest_windows
    if MAX_WINDOWS:
        logger.warning(
            "MAX_WINDOWS=%d -- running a debug subset, not the full backtest. "
            "Set PIPELINE_CONFIG['max_windows'] = None to run all %d windows.",
            MAX_WINDOWS, len(backtest_windows),
        )

    logger.info("[3/3] Running %d backtest window(s)", len(windows_to_run))
    all_metric_rows, all_fva_rows = [], []
    for window_id, wnd in enumerate(windows_to_run):
        logger.info("--- window %d/%d ---", window_id + 1, len(windows_to_run))
        try:
            metric_rows, fva_rows = run_backtest_window(
                window_id, wnd, train, full_features, feat_builder, selected_models, forecaster
            )
            all_metric_rows.extend(metric_rows)
            all_fva_rows.extend(fva_rows)
        except ValueError as e:
            logger.error("window %d failed, skipping: %s", window_id, e)

    metrics_df = pd.DataFrame(all_metric_rows)
    fva_df = pd.DataFrame(all_fva_rows)

    logger.info("Pipeline finished in %.1fs", time.time() - t0)

    if RUN == "Experiment":
        experiment_record = {
            'experiment_id': PIPELINE_CONFIG.get('experiment_name', f"experiment_{int(t0)}"),
            'timestamp': datetime.now(),
            'run': RUN,
            'model': MODEL_NAME,
            'backtest_mode': BACKTEST_TYPE,
            'training_window': TRAINING_WINDOW,
            'horizon_days': HORIZON,
            'use_log_transform': selected_models.use_log_transform,
            'categorical_cols': CATEGORICAL_COLS,
            'max_windows': MAX_WINDOWS,
            'n_windows_run': len(windows_to_run),
            'n_features': len(full_features),
            'features': full_features,
            'duration_sec': time.time() - t0,
            'metrics': metrics_df,
            'fva': fva_df,
        }
        log_experiment_results(EXPERIMENTS_LOG_PATH, experiment_record)


    print("\n=== METRICS BY WINDOW / MODEL ===")
    print(metrics_df.to_string(index=False))
    print("\n=== FVA vs SEASONAL NAIVE (positive = beat naive) ===")
    print(fva_df.to_string(index=False))

    return metrics_df, fva_df


if __name__ == '__main__':
    main()