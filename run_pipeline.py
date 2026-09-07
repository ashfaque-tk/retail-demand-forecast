"""Production Pipeline Runner for Model-Agnostic Demand Forecasting & Inventory Optimization.

Workflow:
    1. Load and validate curated data + final feature set.
    2. Execute walk-forward validation windows via BacktestEngine (ML + Baselines).
    3. Evaluate periodic replenishment inventory policies across out-of-sample errors.
    4. Log experiment records to JSON and print executive performance tables.
"""
from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
import pickle
import time
from typing import Any

import numpy as np
import pandas as pd

from config import PIPELINE_CONFIG
from src.backtest_engine import BacktestEngine, DeploymentResult, WindowResult
from src.features import FeatureBuilder
from src.models_train import SelectModel
from src.recursive_model import Forecaster
from src.utils import get_items_with_min_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration constants
TRAIN_PATH = PIPELINE_CONFIG["train_data_path"]
TEST_PATH = PIPELINE_CONFIG["test_data_path"]
FEATURE_PATH = PIPELINE_CONFIG["final_feature_set"]
RESULTS_DIR = Path(PIPELINE_CONFIG["results_dir"])
EXPERIMENTS_LOG_PATH = RESULTS_DIR / "Experiments/Experiments.json"

RUN = PIPELINE_CONFIG.get("run", "Experiment")
MODEL_NAME = PIPELINE_CONFIG.get("model", "lgbm")
TRAINING_WINDOW = PIPELINE_CONFIG.get("training_window", 730)
HORIZON = PIPELINE_CONFIG.get("horizon_days", 28)
BACKTEST_TYPE = PIPELINE_CONFIG.get("backtest_mode", "rolling")
CATEGORICAL_COLS = PIPELINE_CONFIG.get("categorical_cols", ["item_id", "cat_id", "dept_id"])
MAX_WINDOWS = PIPELINE_CONFIG.get("max_windows", 1)

DEPLOYMENT_DIR = Path(PIPELINE_CONFIG.get("deployment_dir"))

def _json_safe(obj: Any) -> Any:
    """Recursively serializes pandas DataFrames, numpy types, and Timestamps for JSON logging."""
    if isinstance(obj, pd.DataFrame):
        return json.loads(obj.to_json(orient="records", date_format="iso"))
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


def log_experiment_results(results_path: Path, record: dict[str, Any]) -> Path:
    """Appends an experiment record to a JSON log file, creating parent directories on first use."""
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if not isinstance(existing, list):
            raise ValueError(f"{results_path} exists but is not a JSON list. Refusing to append.")
    else:
        existing = []

    existing.append(_json_safe(record))

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, default=str)

    logger.info("Logged experiment record to %s (%d total entries)", results_path, len(existing))
    return results_path


def _build_engine(full_features: list[str]) -> BacktestEngine:
    """Construct the shared forecasting/inventory engine for either run mode."""
    return BacktestEngine(
        model_name=MODEL_NAME,
        training_window_days=TRAINING_WINDOW,
        horizon_days=HORIZON,
        backtest_mode=BACKTEST_TYPE,
        step_size_days=28,
        categorical_cols=CATEGORICAL_COLS,
        feature_names=full_features,
        lead_time_days=PIPELINE_CONFIG.get("lead_time", 4),
        review_period_days=PIPELINE_CONFIG.get("review_period", 7),
        holding_cost_rate=PIPELINE_CONFIG.get("holding_cost_rate", 0.02),
        min_history_days=100,
        max_windows=MAX_WINDOWS,
    )


def run_backtest_window(
    window_id: int,
    wnd: dict[str, Any],
    train: pd.DataFrame,
    full_features: list[str],
    feat_builder: FeatureBuilder | None = None,
    selected_models: SelectModel | None = None,
    forecaster: Forecaster | None = None,
    engine: BacktestEngine | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compatibility wrapper executing a single backtest window via BacktestEngine."""
    if engine is None:
        engine = BacktestEngine(
            model_name=MODEL_NAME,
            training_window_days=TRAINING_WINDOW,
            horizon_days=HORIZON,
            backtest_mode=BACKTEST_TYPE,
            categorical_cols=CATEGORICAL_COLS,
            feature_names=full_features,
            lead_time_days=PIPELINE_CONFIG.get("lead_time", 11),
            review_period_days=PIPELINE_CONFIG.get("review_period", 7),
            holding_cost_rate=PIPELINE_CONFIG.get("holding_cost_rate", 0.02),
        )
    res = engine.run_window(window_id=window_id, window_spec=wnd, full_data=train, feature_names=full_features)
    return res.metric_rows, res.fva_rows

def _save_deployment_artifacts(deployment: DeploymentResult, engine: BacktestEngine) -> Path:
    """Persist the fitted model and deployable forecast/policy outputs together."""
    deployment_id = PIPELINE_CONFIG.get("deployment_name") or f"deployment_{int(time.time())}"
    artifact_dir = DEPLOYMENT_DIR / deployment_id
    artifact_dir.mkdir(parents=True, exist_ok=False)

    with open(artifact_dir / "model.pkl", "wb") as f:
        pickle.dump(engine.model, f)

    deployment.forecasts.to_parquet(artifact_dir / "forecasts.parquet", index=False)
    deployment.inventory_policy.to_parquet(artifact_dir / "inventory_policy.parquet", index=False)
    deployment.inventory_costs.to_parquet(artifact_dir / "inventory_costs.parquet", index=False)
    deployment.error_statistics.to_parquet(artifact_dir / "error_statistics.parquet", index=False)

    # Comparison artifacts are always written in long format.  With the baseline
    # flag disabled, these contain the selected model only and retain one schema.
    def combine(attribute: str) -> pd.DataFrame:
        return pd.concat(
            [getattr(result, attribute).assign(model=model_name)
             for model_name, result in deployment.model_results.items()],
            ignore_index=True,
        )

    # Baseline helpers include ``real_sales`` while the recursive ML forecaster
    # correctly returns predictions only.  Merge one canonical actual-sales
    # column so every model has the same plotting schema.
    actuals_for_merge = deployment.test_actuals[["item_id", "dept_id", "cat_id", "date", "sales"]].rename(
        columns={"sales": "actual_sales"}
    )
    forecast_comparison = combine("forecasts").drop(columns=["real_sales"], errors="ignore").merge(
        actuals_for_merge,
        on=["item_id", "dept_id", "cat_id", "date"],
        how="left",
        validate="many_to_one",
    )
    forecast_comparison.to_parquet(artifact_dir / "forecast_comparison.parquet", index=False)
    combine("calibration_predictions").to_parquet(
        artifact_dir / "calibration_forecast_comparison.parquet", index=False
    )
    combine("inventory_policy").to_parquet(
        artifact_dir / "inventory_policy_comparison.parquet", index=False
    )
    combine("inventory_costs").to_parquet(
        artifact_dir / "inventory_cost_comparison.parquet", index=False
    )
    deployment.test_actuals.to_parquet(artifact_dir / "test_actuals.parquet", index=False)
    pd.DataFrame([
        {"model": model_name, **result.calibration_metrics}
        for model_name, result in deployment.model_results.items()
    ]).to_parquet(artifact_dir / "calibration_metrics_by_model.parquet", index=False)
    test_metric_rows = [
        {"model": model_name, **result.test_metrics}
        for model_name, result in deployment.model_results.items()
        if result.test_metrics is not None
    ]
    if test_metric_rows:
        pd.DataFrame(test_metric_rows).to_parquet(
            artifact_dir / "test_metrics_by_model.parquet", index=False
        )

    manifest = {
        "deployment_id": deployment_id,
        "created_at": datetime.now(),
        "model": MODEL_NAME,
        "training_window_days": TRAINING_WINDOW,
        "calibration_start": deployment.calibration_start,
        "calibration_end": deployment.calibration_end,
        "forecast_start": deployment.forecast_start,
        "forecast_end": deployment.forecast_end,
        "calibration_metrics": deployment.calibration_metrics,
        "inventory_cost_summary": deployment.inventory_cost_summary,
        "evaluation_models": list(deployment.model_results),
        "features": engine.feature_names,
    }
    with open(artifact_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(manifest), f, indent=2)
    return artifact_dir


def main() -> tuple[pd.DataFrame, pd.DataFrame, list[WindowResult]] | DeploymentResult:
    """Executes the end-to-end retail forecasting and inventory optimization pipeline."""
    start_time = time.time()

    # Step 1: Load and filter datasets
    logger.info("[1/3] Loading dataset and feature configuration...")
    train_df = pd.read_parquet(TRAIN_PATH)
    total_days = train_df["date"].nunique()

    # Filter items with complete historical presence
    train_df = get_items_with_min_history(train_df, min_history_days=total_days - 1).copy()
    test_df = pd.read_parquet(TEST_PATH)
    test_df = test_df[test_df["item_id"].isin(train_df["item_id"])].copy()

    logger.info(
        "Active SKUs with complete historical presence: Train=%d, Test=%d",
        train_df["item_id"].nunique(),
        test_df["item_id"].nunique(),
    )

    full_features = list(pd.read_pickle(FEATURE_PATH))
    logger.info("Loaded feature schema: %d total features", len(full_features))

    ## initilize the engine 
    engine = _build_engine(full_features)


    if RUN == "Deploy":
        history_dates = pd.Index(train_df["date"].drop_duplicates().sort_values())

        if len(history_dates) < TRAINING_WINDOW:
            raise ValueError(
                f"Deploy requires {TRAINING_WINDOW} historical dates; found {len(history_dates)}."
            )
        forecast_dates = pd.Index(test_df["date"].drop_duplicates().sort_values())
        if len(forecast_dates) < HORIZON:
            raise ValueError(f"Deploy requires {HORIZON} future dates; found {len(forecast_dates)}.")

        # Only the latest known training window and first production horizon
        # participate in deployment.  Remaining test dates may be used later for
        # offline stress tests, but must not affect today's decision.
        final_train = train_df[train_df["date"].isin(history_dates[-TRAINING_WINDOW:])].copy()
        final_test = test_df[test_df["date"].isin(forecast_dates[:HORIZON])].copy()
        logger.info(
            "[2/3] Deploying with %d historical days and %d forecast days.",
            TRAINING_WINDOW,
            HORIZON,
        )
        deployment = engine.run_deploy(
            historical_data=final_train,
            future_static_data=final_test,
            calibration_days=HORIZON,
            evaluate_baselines=PIPELINE_CONFIG.get("deployment_evaluate_baselines", False),
        )
        artifact_dir = _save_deployment_artifacts(deployment, engine)
        logger.info("[3/3] Deployment artifacts saved to %s", artifact_dir)
        return deployment

    
    if RUN != "Experiment":
            raise ValueError("PIPELINE_CONFIG['run'] must be either 'Experiment' or 'Deploy'.")


    # Step 2: Initialize BacktestEngine and execute walk-forward windows
    logger.info("[2/3] Initializing BacktestEngine (mode=%s, horizon=%d days)...", BACKTEST_TYPE, HORIZON)

    metrics_df, fva_df, window_results = engine.run_all(full_data=train_df)

    # Step 3: Log experiment results
    logger.info("[3/3] Logging experiment records and printing benchmarks...")
    duration_sec = time.time() - start_time

    experiment_record = {
        "experiment_id": PIPELINE_CONFIG.get("experiment_name", f"experiment_{int(start_time)}"),
        "timestamp": datetime.now(),
        "run": RUN,
        "model": MODEL_NAME,
        "backtest_mode": BACKTEST_TYPE,
        "training_window": TRAINING_WINDOW,
        "horizon_days": HORIZON,
        "use_log_transform": engine.use_log_transform,
        "categorical_cols": CATEGORICAL_COLS,
        "max_windows": MAX_WINDOWS,
        "n_windows_run": len(window_results),
        "n_features": len(full_features),
        "features": full_features,
        "duration_sec": duration_sec,
        "metrics": metrics_df,
        "fva": fva_df,
        "Notes": PIPELINE_CONFIG.get("note"),
    }
    log_experiment_results(EXPERIMENTS_LOG_PATH, experiment_record)

    # Print summary tables
    print("\n" + "=" * 60)
    print("BACKTEST METRICS BY WINDOW / MODEL")
    print("=" * 60)
    print(metrics_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("FORECAST VALUE ADDED (FVA) vs SEASONAL NAIVE (positive % = beat baseline)")
    print("=" * 60)
    print(fva_df.to_string(index=False))

    logger.info("Pipeline completed successfully in %.2f seconds.", duration_sec)
    return metrics_df, fva_df, window_results


if __name__ == "__main__":

    main()
