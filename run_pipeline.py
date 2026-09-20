"""Production Pipeline Runner for Model-Agnostic Demand Forecasting & Inventory Optimization.

Workflow:
    1. Load and validate curated data + final feature set.
    2. Execute walk-forward validation windows via BacktestEngine (ML + Baselines).
    3. Evaluate periodic replenishment inventory policies across out-of-sample errors.
    4. Log experiment records to JSON and print executive performance tables.
"""
from __future__ import annotations

import logging
from pathlib import Path
import time
import datetime
import pandas as pd
from config import (
    BASE_DIR,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    DEPLOYMENT_DIR,
    RESULTS_DIR,
    PIPELINE_CONFIG,
)
from src.data_checks import validate_raw
from src.backtest_engine import BacktestEngine, DeploymentResult
from src.utils import log_experiment_results  
from src.features import FeatureBuilder

# logging initiation
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)




def build_engine(feature_names: list[str]) -> BacktestEngine:
    """Constructs BacktestEngine directly from the unified PIPELINE_CONFIG."""
    return BacktestEngine(
        model_name=PIPELINE_CONFIG["model"],
        forecast_type=PIPELINE_CONFIG["forecast_type"],
        training_window_days=PIPELINE_CONFIG["training_window"],
        horizon_days=PIPELINE_CONFIG["horizon_days"],
        backtest_mode=PIPELINE_CONFIG["backtest_mode"],
        step_size_days=PIPELINE_CONFIG["step_size"],
        categorical_cols=PIPELINE_CONFIG["categorical_cols"],
        feature_names=feature_names,
        lead_time_days=PIPELINE_CONFIG["lead_time"],
        review_period_days=PIPELINE_CONFIG["review_period"],
        holding_cost_rate=PIPELINE_CONFIG["holding_cost_rate"],
        max_windows=PIPELINE_CONFIG.get("max_windows"),
        use_log_transform=PIPELINE_CONFIG.get("use_log_transform", True),
    )


def save_deployment_artifacts(deployment: DeploymentResult, engine: BacktestEngine) -> Path:
    """Persists model binary, forecasts, and inventory policy."""
    deployment_id = f"deploy_{PIPELINE_CONFIG['model']}_{int(time.time())}"
    artifact_dir = DEPLOYMENT_DIR / deployment_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # 1. Save operational tables as parquet
    deployment.forecasts.to_parquet(artifact_dir / "forecasts.parquet", index=False)
    deployment.inventory_policy.to_parquet(artifact_dir / "inventory_policy.parquet", index=False)
    # 2. Save metadata summary as JSON
    summary = {
        "model": deployment.model_name,
        "run_timestamp": str(deployment.run_timestamp),
        "calibration_metrics": deployment.calibration_metrics,
        "inventory_cost_summary": deployment.inventory_cost_summary.to_dict(),
    }
    with open(artifact_dir / "metadata.json", "w", encoding="utf-8") as f:
        import json
        json.dump(summary, f, indent=2)
    logger.info("Saved deployment artifacts to %s", artifact_dir)
    return artifact_dir

def main():
    start_time = time.time()
    
    # 1. Load raw data
    logger.info("[1/3] Loading and validating data...")
    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    test_df = pd.read_parquet(TEST_DATA_PATH)
    
    validate_raw(train_df)
    validate_raw(test_df)
    # 2. Dynamically determine feature schema (NO .pkl file!)
    feat_builder = FeatureBuilder()
    
    # Take a small sample to see what features FeatureBuilder generates
    sample_df = train_df[train_df["item_id"] == train_df["item_id"].iloc[0]].tail(150)
    sample_features = feat_builder.build(sample_df)
    
    # Everything generated that isn't metadata/target is a feature
    non_feature_cols = {
        "date", "sales", "origin_date", "target_date", "target_sales",
        "store_id", "state_id", "item_id", "cat_id", "dept_id"
    }
    full_features = [col for col in sample_features.columns if col not in non_feature_cols]
    
    logger.info("Dynamically detected %d feature columns: %s", len(full_features), full_features[:5])
    # 3. Build Engine with dynamic feature names
    engine = build_engine(full_features)
    # 4. Mode routing
    run_mode = PIPELINE_CONFIG.get("mode", "experiment").lower()
    if run_mode == "deploy":
        logger.info("[2/3] Executing production deployment run...")
        deployment = engine.run_deploy(
            historical_data=train_df,
            future_static_data=test_df,
            calibration_days=PIPELINE_CONFIG.get("horizon_days", 28),
            evaluate_baselines=True,
        )
        artifact_dir = save_deployment_artifacts(deployment, engine)
        logger.info("[3/3] Deployment complete. Artifacts saved to %s", artifact_dir)
        return deployment
    elif run_mode == "experiment":
        logger.info("[2/3] Executing walk-forward backtesting (mode=%s)...", PIPELINE_CONFIG["backtest_mode"])
        metrics_df, fva_df, window_results = engine.run_all(full_data=train_df)
        
        # Save experiment records
        duration_sec = time.time() - start_time
        record = {
            "timestamp": datetime.now().isoformat(),
            "model": PIPELINE_CONFIG["model"],
            "metrics": metrics_df,
            "fva": fva_df,
            "features": full_features,
            "duration_sec": duration_sec,
        }
        log_experiment_results(RESULTS_DIR / "experiments.json", record)
        # Print executive summary
        print("\n" + "=" * 60)
        print("BACKTEST METRICS SUMMARY")
        print("=" * 60)
        print(metrics_df.groupby("model")[["MAE", "wrmsse"]].mean())
        return metrics_df, fva_df, window_results
    else:
        raise ValueError(f"Unknown mode: {run_mode}. Expected 'experiment' or 'deploy'.")


if __name__ == "__main__":

    main()
