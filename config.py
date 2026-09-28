from pathlib import Path

# Paths (hardcoded, never changing)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
DEPLOYMENT_DIR = BASE_DIR / "models/deployments"

TRAIN_DATA_PATH = DATA_DIR / "processed/train_filtered_ca1.parquet"
TEST_DATA_PATH = DATA_DIR / "processed/test_filtered_ca1.parquet"

# Inventory cost inputs. The safety-stock quantile is the newsvendor critical
# fractile p / (p + h) -- the probability that a unit is needed, given the cost
# of being one short versus holding it a day too long. It is derived, not typed
# in, so moving either cost moves the quantile with it. At h=0.2 and p=1.0 this
# is 0.8333, and int(level*100) names that column `q83`.
HOLDING_COST_RATE = 0.2
STOCKOUT_COST_RATE = 1


def critical_quantile(holding_cost_rate: float,
                      stockout_cost_rate: float = STOCKOUT_COST_RATE) -> float:
    """Newsvendor critical fractile p / (p + h)."""
    return stockout_cost_rate / (stockout_cost_rate + holding_cost_rate)


PIPELINE_CONFIG = {
    # Run mode
    "mode": "backtest",  # "backtest", "deploy", "auto"
    
    # Backtest Setup
    "backtest_mode": "rolling",
    "training_window": 365,
    "horizon_days": 28,
    "step_size": 28,
    "max_windows": 1,  # default debug mode

    # Model
    "model": "",
    "forecast_type": "",  # "direct" or "recursive"
    "quantiles"    : [critical_quantile(HOLDING_COST_RATE)],
    "safety_stock_policy": "quantile",
    "use_log_transform": False, # for scaling 
    "categorical_cols": ["item_id",'cat_id', "dept_id"],

    # Inventory Policy
    "lead_time": 4,
    "review_period": 7,
    "holding_cost_rate": HOLDING_COST_RATE,
    "stockout_cost_rate": STOCKOUT_COST_RATE,
}
