"""Production Pipeline Runner for Model-Agnostic Demand Forecasting & Inventory Optimization.

Workflow:
    1. Load and validate curated data + final feature set.
    2. Execute walk-forward validation windows via BacktestEngine (ML + Baselines).
    3. Evaluate periodic replenishment inventory policies across out-of-sample errors.
    4. Log experiment records to JSON and print executive performance tables.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time
from datetime import datetime
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
from src.backtest_windows import generate_rolling_windows
from src.utils import log_experiment_results  
from src.features import FeatureBuilder

from scripts.generate_report import generate_experiment_html_report
from scripts.populate_dbs import populate_inventory,populate_actuals,populate_predictions
# logging initiation
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_cli_args() -> argparse.Namespace:
    """Parses command line arguments to control mode, model hyperparameters, and persistence."""
    parser = argparse.ArgumentParser(
        description="Run forecasting backtests, holdout evaluations, or deployment pipelines."
    )

    # Core Execution Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["backtest", "deploy", "test"],
        default= 'backtest',
        help="Pipeline mode: 'backtest', 'deploy', or 'test'.",
    )

    # Model Parameters
    parser.add_argument(
        "--model",
        type=str,
        default='lgbm',
        help="Model engine to execute (e.g., 'lgbm', 'xgb', 'catboost', 'naive', 'ma').",
    )
    parser.add_argument(
        "--forecast-type",
        type=str,
        dest="forecast_type",
        choices=["recursive", "direct"],
        default= 'recursive',
        help="Multi-step forecasting approach.",
    )
    parser.add_argument(
        "--training-window",
        type=int,
        dest="training_window",
        default=365,
        help="Training window size in days (e.g., 365 for 1 year).",
    )
    parser.add_argument(
        "--backtest-windows",
        type = int,
        dest = 'backtest_windows',
        default=10,
        help = "Number of backtest windows to be run (default 1 for debug)")

    # Persistence Options
    parser.add_argument(
        "--persist-format",
        type=str,
        dest="persist_format",
        choices=["none", "parquet", "db"],
        default="none",
        help="Storage option for forecasts and inventory logs ('parquet', 'db', or 'none').",
    )

    return parser.parse_args()

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

def save_to_parquet(models_data:dict,filename:str):
    import os 
    output_path:str=f'{BASE_DIR}/results/tests'
    os.makedirs(output_path,exist_ok=True)
    

    processed_dfs = []

    # 1. Iterate through dictionary items
    for model_name, df_pred in models_data.items():
        # Make a copy to avoid mutating original input DataFrames
        temp_df = df_pred.copy()

        # Add identifying column for the model source
        temp_df["model"] = model_name

        processed_dfs.append(temp_df)

    # 2. Concatenate all DataFrames along rows (axis=0)
    # Pandas aligns matching columns and fills missing columns with NaN
    combined_df = pd.concat(processed_dfs, axis=0, ignore_index=True)

    # 3. Save to Parquet format
    # Requires 'pyarrow' or 'fastparquet' installed (e.g., pip install pyarrow)
    combined_df.to_parquet(output_path+'/'+filename, engine="pyarrow", index=False)

    print(f"Successfully saved combined predictions to '{output_path+'/'+filename}'")
    return combined_df

def main():

    start_time = time.time()
    
    # 0. Update config from CLI
    args = parse_cli_args()
    PIPELINE_CONFIG["mode"] = args.mode 
    PIPELINE_CONFIG["model"] = args.model
    PIPELINE_CONFIG["forecast_type"] = args.forecast_type 
    PIPELINE_CONFIG["training_window"] = args.training_window
    PIPELINE_CONFIG['max_windows'] = args.backtest_windows 



    logger.info(
        "Executing Pipeline: Mode='%s' | Model='%s' | Type='%s' | Backtest Windows= '%d'",
        PIPELINE_CONFIG["mode"],
        PIPELINE_CONFIG["model"],
        PIPELINE_CONFIG["forecast_type"],
        PIPELINE_CONFIG['max_windows']
    )

    # 1. Load raw data and validation
    logger.info("[1/3] Loading Training and Unknown data...")
    train_df = pd.read_parquet(TRAIN_DATA_PATH)
    unknown_df = pd.read_parquet(TEST_DATA_PATH) #to simulate production setup
    # split the first horizon from unknown to be holdout set, keep rest to simulate production
    horizon =  PIPELINE_CONFIG.get('horizon_days',28)
    holdout_test = unknown_df[unknown_df['date']<=(unknown_df['date'].min()+
                                                   pd.Timedelta(days=horizon))]
    
    logger.info("[1/3] Validating the data...")
    validate_raw(train_df)
    validate_raw(unknown_df)
    logger.info("[1/3] Validation Successfull.")

    # 2. Build feature schema from a small sample
    feat_builder = FeatureBuilder()
    sample_df = train_df[train_df["item_id"] == train_df["item_id"].iloc[0]].tail(150)
    _, full_features = feat_builder.build(sample_df)
    
    logger.info("Total features: %d feature columns: %s", len(full_features),full_features)

    # 3. Build Engine with dynamic feature names
    engine = build_engine(full_features)

    # 4. Mode routing
    run_mode = PIPELINE_CONFIG.get("mode",'').lower()
    # info
    model = PIPELINE_CONFIG.get('model','lgbm')
    type = PIPELINE_CONFIG.get('forecast_type','recursive')
    training_yr = PIPELINE_CONFIG.get('training_window',365)//365
    timestamp_str = pd.Timestamp.now().strftime("%Y-%m-%d_T-%H-%M-%S")
    lead_time = PIPELINE_CONFIG.get('lead_time',4)
    review_period = PIPELINE_CONFIG.get('review_period',7)
    
    if training_yr == 0:
        logger.error(f'Got Training year = 0, Provide training window in days ( for eg; 365 for 1 year)')
        quit()
    if run_mode == "test":
        logger.info("[2/3] Executing production deployment run...")
        (test_preds,
        test_inventory,
        holdout_metric,
        holdout_fva) = engine.run_holdout_test(
            full_data=train_df,
            test_holdout= holdout_test,
            training_year=training_yr)
        
        logger.info("[3/3] Holdout Evaluation completed. Results:" )
        
        holdout_results = BASE_DIR/f'results/holdout_test_results_{timestamp_str}.html'
        print(f"######  Holdout Metrics ######")
        print(pd.DataFrame(holdout_metric),'\n')
        print(f'###### Forecast Value Added ######')
        print(pd.DataFrame(holdout_fva))
        winner = generate_experiment_html_report(
                    metrics_df=pd.DataFrame(holdout_metric),
                    fva_df=pd.DataFrame(holdout_fva),
                    output_path= holdout_results,
                    lead_time=lead_time,
                    review_period=review_period,
                    model_type=type,
                    train_years=training_yr,
                    eval_mode='test')

        logger.info(f"Winning Model on Test Set: {winner}. Deploy ")
        
        print(f'champion model is {winner}, inventory: {test_inventory[winner]}')
        
        ### winning_model preds and inventory will be uploaded to dB, and also saved to parquet
        save_to_parquet(models_data=test_preds,filename=f'test_preds-{model}-{type}-{timestamp_str}.parquet')
        save_to_parquet(models_data=test_inventory,filename=f'test_inventory-{model}-{type}-{timestamp_str}.parquet')
        # save to postgres the forecasts, and inventory
        if args.persist_format == 'db':
            logger.info(f'{run_mode} champion result saving into postgres db: inventory')
            populate_inventory(
                df=test_inventory[winner],
                model_name=winner,
                forecast_type= type,
                lead_time_days= lead_time,
                review_period_days= review_period)

    
    elif run_mode == "backtest":

        logger.info("[2/3] Executing walk-forward backtesting (mode=%s)...", PIPELINE_CONFIG["backtest_mode"])
        metrics_df, fva_df, window_results = engine.run_all(full_data=train_df)
        
        # Save experiment records
        record = {
            "timestamp": datetime.now().isoformat(),
            "model": PIPELINE_CONFIG["model"],
            "metrics": metrics_df,
            "fva": fva_df,
            "features": full_features,
        }
        # for quick view, printing backtest results
        print(f"####### BACKTEST METRICS #######")
        print(metrics_df,'\n' )
        print(f'####### BACKTEST FVA_RESULTS ########')
        print(fva_df)
  
        # generate html report 
        # log experimental raw data
        log_experiment_results(RESULTS_DIR / f"expts/backtest_expt_{model}_{type}_{training_yr}yr_{timestamp_str}.json", record)
        report_out = RESULTS_DIR/f'backtest_expt_report_{model}_{type}_{training_yr}yr_{timestamp_str}.html'
        generate_experiment_html_report(metrics_df=metrics_df,fva_df=fva_df,
                                        output_path=report_out,lead_time=lead_time,
                                        review_period=review_period,
                                        model_type=type,
                                        train_years=training_yr)
        return metrics_df, fva_df, window_results

    elif run_mode == 'deploy':
        pass

    else:
        raise ValueError(f"Unknown mode: {run_mode}. Expected 'experiment' or 'deploy'.")



if __name__ == "__main__":

    main()
