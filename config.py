from pathlib import Path

# Dynamically resolve root relative to this config file
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / 'models'

PIPELINE_CONFIG = {
    "run"  : "Deploy" ,# "Deploy" if Deploy save the final model
    "note"  : "Inventory Cost formulation with items having full history ",
    "model": "lgbm",
    "forecast_type": "recursive", #Direct
    "quantiles"    : None,
    # Offline-only comparison using known test actuals. Keep False for a true
    # production run where only the selected model should be generated.
    "deployment_evaluate_baselines": True,
    "training_window": 730,
    "horizon_days": 28,
    "backtest_mode": "rolling",  "max_windows"  : 2,
    "categorical_cols":['item_id','cat_id','dept_id'],
    "train_data_path": str(DATA_DIR / "processed/sales_known_ca_1.parquet"),
    "test_data_path": str(DATA_DIR / "processed/sales_future_ca_1.parquet"),
    "final_feature_set":str(MODELS_DIR / "final_features_ca1.pkl"),

    "results_dir"   : str(BASE_DIR/'results'),
    "deployment_dir": str(MODELS_DIR / "deployments"),
    "deployment_name":None,
    "lead_time" : 4,
    "review_period": 7,
     "deployment_evaluate_baselines": True
    
}

if __name__ == '__main__':
    print(PIPELINE_CONFIG['train_data_path'])
    print(PIPELINE_CONFIG['test_data_path'])
