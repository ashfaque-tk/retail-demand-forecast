from pathlib import Path

# Dynamically resolve root relative to this config file
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / 'models'

PIPELINE_CONFIG = {
    "run"  : "Experiment" ,# "Deploy" if Deploy save the final model
    "model": "lgbm",
    "forecast_type": "recursive",
    "training_window": 730,
    "horizon_days": 28,
    "backtest_mode": "rolling",
    
    "train_data_path": str(DATA_DIR / "processed/sales_known_ca_1.parquet"),
    "test_data_path": str(DATA_DIR / "processed/sales_future_ca_1.parquet"),
    "final_feature_set":str(MODELS_DIR / "final_feature_set.parquet")
    
}

if __name__ == '__main__':
    print(PIPELINE_CONFIG['train_data_path'])
    print(PIPELINE_CONFIG['test_data_path'])