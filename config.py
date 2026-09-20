from pathlib import Path

# Paths (hardcoded, never changing)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
DEPLOYMENT_DIR = BASE_DIR / "models/deployments"

TRAIN_DATA_PATH = DATA_DIR / "processed/train_filtered_ca1.parquet"
TEST_DATA_PATH = DATA_DIR / "processed/test_filtered_ca1.parquet"

PIPELINE_CONFIG = {
    # Run mode
    "mode": "experiment",  # "experiment", "deploy", "auto"
    
    # Backtest Setup
    "backtest_mode": "rolling",
    "training_window": 365,
    "horizon_days": 28,
    "step_size": 28,
    "max_windows": 10,  # last 8 windows, chronological

    # Model
    "model": "lgbm",
    "forecast_type": "direct",  # "direct" or "recursive"
    "use_log_transform": True, # for scaling 
    "categorical_cols": ["item_hash", "cat_hash", "dept_hash"],

    # Inventory Policy
    "lead_time": 4,
    "review_period": 7,
    "holding_cost_rate": 0.02,
}
