"""Backtest Engine for retail demand forecasting and inventory replenishment evaluation.

Orchestrates walk-forward validation windows (rolling or expanding), baseline comparisons
(Seasonal Naive, Moving Average), ML model training, recursive multi-step forecasting,
out-of-sample error tracking, and periodic inventory replenishment policy evaluation.

Eliminates mutable global state and provides clean, structured data containers.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from src.backtest_windows import (
    generate_expanding_windows,
    generate_rolling_windows,
    split_data,
)
from src.baselines import seasonal_naive, simple_moving_average
from src.features import FeatureBuilder
from src.Inventory_optimization.restock_policy_1 import (
    compute_rolling_tau_error,
    run_inventory_pipeline,
)
from src.metrics import fva, get_all_metrics
from src.models_train import SelectModel
from src.recursive_model import Forecaster
from src.utils import get_items_with_min_history

logger = logging.getLogger(__name__)


@dataclass
class WindowResult:
    """Structured container holding all artifacts and metrics for a single backtest window."""

    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp
    metric_rows: list[dict[str, Any]]
    fva_rows: list[dict[str, Any]]
    model_predictions: dict[str, pd.DataFrame]
    eval_actuals: pd.DataFrame
    inventory_costs: dict[str, dict[str, float]] = field(default_factory=dict)


class BacktestEngine:
    """Walk-forward backtesting orchestrator for model-agnostic forecasting pipelines.

    Manages data slicing, feature engineering, model fitting, recursive prediction,
    baseline generation, forecast metrics computation, and out-of-sample inventory
    policy cost tracking across sequential evaluation windows.

    Parameters
    ----------
    model_name : str, default 'lgbm'
        Name of the ML model engine ('lgbm', 'hgb', etc.).
    training_window_days : int, default 730
        Size of the historical training window in days.
    horizon_days : int, default 28
        Forecast evaluation horizon in days.
    backtest_mode : str, default 'rolling'
        Walk-forward strategy: 'rolling' or 'expanding'.
    step_size_days : int, default 28
        Cadence (days) to advance each consecutive evaluation window.
    categorical_cols : list[str] | None, optional
        List of categorical column names (e.g., ['item_id', 'cat_id', 'dept_id']).
    feature_names : list[str] | None, optional
        Ordered list of feature column names used for training and inference.
    lead_time_days : int, default 11
        Supplier replenishment lead time in days.
    review_period_days : int, default 7
        Inventory review / reorder cycle period in days.
    holding_cost_rate : float, default 0.02
        Unit holding cost per item per day.
    min_history_days : int, default 100
        Minimum historical active days required for SKUs to be evaluated.
    max_windows : int | None, optional
        Optional cap on the number of windows to execute (useful for debugging).
    use_log_transform : bool, default False
        Whether the ML model fits on log1p(target).
    """

    def __init__(
        self,
        model_name: str = "lgbm",
        training_window_days: int = 730,
        horizon_days: int = 28,
        backtest_mode: str = "rolling",
        step_size_days: int = 28,
        categorical_cols: list[str] | None = None,
        feature_names: list[str] | None = None,
        lead_time_days: int = 11,
        review_period_days: int = 7,
        holding_cost_rate: float = 0.02,
        min_history_days: int = 100,
        max_windows: int | None = None,
        use_log_transform: bool = False,
    ) -> None:
        self.model_name = model_name
        self.training_window_days = training_window_days
        self.horizon_days = horizon_days
        self.backtest_mode = backtest_mode
        self.step_size_days = step_size_days
        self.categorical_cols = categorical_cols or ["item_id", "cat_id", "dept_id"]
        self.feature_names = feature_names or []
        self.lead_time_days = lead_time_days
        self.review_period_days = review_period_days
        self.holding_cost_rate = holding_cost_rate
        self.min_history_days = min_history_days
        self.max_windows = max_windows
        self.use_log_transform = use_log_transform

        # Review period + lead time total risk horizon (tau)
        self.tau_days = self.lead_time_days + self.review_period_days

        # Encapsulated stateful error buffers (replaces mutable module globals)
        self.oos_errors: dict[str, list[pd.DataFrame]] = {
            "ml": [],
            "ma": [],
            "naive": [],
        }

        # Cache for last executed window outputs (for easy plotting / inspection)
        self.last_window_result: WindowResult | None = None

        # Core pipeline components
        self.feature_builder = FeatureBuilder()
        self.model = SelectModel(
            model=self.model_name,
            quantiles=None,
            use_log_transform=self.use_log_transform,
            categorical_cols=self.categorical_cols,
        )
        self.forecaster = Forecaster(model=self.model, feature_builder=self.feature_builder)

    def reset_state(self) -> None:
        """Clears accumulated out-of-sample errors and cached window results."""
        self.oos_errors = {"ml": [], "ma": [], "naive": []}
        self.last_window_result = None

    def generate_windows(self, full_data: pd.DataFrame) -> list[dict[str, Any]]:
        """Generates walk-forward window date ranges based on the configured mode."""
        if self.backtest_mode == "rolling":
            windows = generate_rolling_windows(
                full_data,
                training_window=self.training_window_days,
                horizon=self.horizon_days,
                step_size=self.step_size_days,
            )
        elif self.backtest_mode == "expanding":
            windows = generate_expanding_windows(
                full_data,
                training_window=self.training_window_days,
                horizon=self.horizon_days,
                step_size=self.step_size_days,
            )
        else:
            raise ValueError(f"Unknown backtest_mode: {self.backtest_mode!r}. Expected 'rolling' or 'expanding'.")

        if self.max_windows is not None and self.max_windows > 0:
            logger.info("Capping execution to first %d of %d total windows.", self.max_windows, len(windows))
            return windows[: self.max_windows]

        return windows

    def run_window(
        self,
        window_id: int,
        window_spec: dict[str, Any],
        full_data: pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> WindowResult:
        """Executes a single walk-forward evaluation window.

        Parameters
        ----------
        window_id : int
            Zero-indexed sequential window counter.
        window_spec : dict[str, Any]
            Dictionary defining 'train_start', 'train_end', 'test_start', 'test_end'.
        full_data : pd.DataFrame
            Full dataset with date, item_id, sales, and static metadata.
        feature_names : list[str] | None, optional
            List of feature names to use. Defaults to self.feature_names.

        Returns
        -------
        WindowResult
            Dataclass containing metrics, FVA, forecasts, actuals, and inventory costs.
        """
        feats = feature_names or self.feature_names
        train_start = pd.Timestamp(window_spec["train_start"])
        train_end = pd.Timestamp(window_spec["train_end"])
        eval_start = pd.Timestamp(window_spec["test_start"])
        eval_end = pd.Timestamp(window_spec["test_end"])

        logger.info(
            "--- Window %d: Train [%s -> %s] | Eval [%s -> %s] ---",
            window_id,
            train_start.date(),
            train_end.date(),
            eval_start.date(),
            eval_end.date(),
        )

        # 1. Slice and filter active items
        window_slice = full_data[full_data["date"].between(train_start, eval_end)]
        active_items_df = get_items_with_min_history(window_slice, min_history_days=self.min_history_days)
        active_items_df = self.feature_builder.add_price_features(active_items_df)

        train_wnd, eval_wnd = split_data(
            df=active_items_df,
            start_date=train_start,
            end_date=train_end,
            forecast_horizon=self.horizon_days,
        )

        # 2. Compute Benchmark Baselines
        baseline_naive = seasonal_naive(train_wnd, eval_wnd)
        baseline_ma = simple_moving_average(train_wnd, eval_wnd, window_days=180)

        metrics_naive = get_all_metrics(train_wnd, eval_wnd, baseline_naive)
        metrics_ma = get_all_metrics(train_wnd, eval_wnd, baseline_ma)

        # 3. Build Features & Fit Machine Learning Model
        train_wnd = self.feature_builder.build(
            df=train_wnd,
            lags=[7, 28, 60, 90],
            mean_windows=[7, 28, 60, 90],
            max_windows=[7, 28, 60, 90],
            rolling_on_lags=[28],
        )

        missing_features = [col for col in feats + ["date", "sales"] if col not in train_wnd.columns]
        if missing_features:
            raise ValueError(f"Window {window_id}: Missing expected features: {missing_features}")

        self.model.fit(train_wnd[feats], train_wnd["sales"])
        preds_df = self.forecaster.recursive_forecaster(train_wnd, eval_wnd)
        metrics_ml = get_all_metrics(train_wnd, eval_wnd, preds_df)

        # 4. Forecast Value Added (FVA) vs Seasonal Naive
        fva_rows = [
            {
                "window_id": window_id,
                "metric": metric_key,
                "fva_moving_average_vs_naive": fva(metrics_naive[metric_key], metrics_ma[metric_key]),
                "fva_model_vs_naive": fva(metrics_naive[metric_key], metrics_ml[metric_key]),
            }
            for metric_key in ["MAE", "wrmsse"]
        ]

        # 5. Inventory Replenishment & Holding Cost Optimization
        models_preds: dict[str, pd.DataFrame] = {
            "ml": preds_df,
            "ma": baseline_ma,
            "naive": baseline_naive,
        }
        inventory_costs: dict[str, dict[str, float]] = {"naive": {}, "ma": {}, "ml": {}}

        for short_name, current_preds in models_preds.items():
            # Rolling cumulative tau error
            error_model = compute_rolling_tau_error(eval_wnd, current_preds, tau=self.tau_days)

            # Evaluate policy only after window 0 (requires historical OOS error)
            if window_id > 0 and len(self.oos_errors[short_name]) > 0:
                cost_summary = run_inventory_pipeline(
                    current_raw=eval_wnd,
                    current_forecast=current_preds,
                    oos_error=self.oos_errors[short_name][-1],
                    review_period=self.review_period_days,
                    lead_time=self.lead_time_days,
                    holding_cost_per_unit=self.holding_cost_rate,
                )
                inventory_costs[short_name] = cost_summary.to_dict()

            # Record out-of-sample error history for this model
            self.oos_errors[short_name].append(error_model)

        # 6. Assemble tidy metric records
        name_to_label = {"naive": "seasonal_naive", "ma": "moving_average", "ml": self.model_name}
        model_metrics_map = {"naive": metrics_naive, "ma": metrics_ma, "ml": metrics_ml}

        metric_rows = [
            {
                "window_id": window_id,
                "train_start": train_start.date(),
                "train_end": train_end.date(),
                "model": name_to_label[s_name],
                "MAE": m["MAE"],
                "BIAS%": m["BIAS%"],
                "wrmsse": m["wrmsse"],
                "MAE-DEPT": m["MAE-DEPT-AGG"],
                "MAE-CAT": m["MAE-CAT-AGG"],
                **inventory_costs.get(s_name, {}),
            }
            for s_name, m in model_metrics_map.items()
        ]

        result = WindowResult(
            window_id=window_id,
            train_start=train_start,
            train_end=train_end,
            eval_start=eval_start,
            eval_end=eval_end,
            metric_rows=metric_rows,
            fva_rows=fva_rows,
            model_predictions=models_preds,
            eval_actuals=eval_wnd,
            inventory_costs=inventory_costs,
        )
        self.last_window_result = result
        return result

    def run_all(
        self,
        full_data: pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[WindowResult]]:
        """Executes the full suite of walk-forward backtesting windows.

        Parameters
        ----------
        full_data : pd.DataFrame
            Complete historical dataset.
        feature_names : list[str] | None, optional
            List of feature names to use. Defaults to self.feature_names.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, list[WindowResult]]
            (metrics_df, fva_df, list_of_window_results)
        """
        feats = feature_names or self.feature_names
        windows = self.generate_windows(full_data)
        total_windows = len(windows)
        logger.info("Executing %d walk-forward evaluation windows...", total_windows)

        all_metric_rows: list[dict[str, Any]] = []
        all_fva_rows: list[dict[str, Any]] = []
        window_results: list[WindowResult] = []

        start_time = time.time()

        for window_id, wnd_spec in enumerate(windows):
            try:
                res = self.run_window(
                    window_id=window_id,
                    window_spec=wnd_spec,
                    full_data=full_data,
                    feature_names=feats,
                )
                all_metric_rows.extend(res.metric_rows)
                all_fva_rows.extend(res.fva_rows)
                window_results.append(res)
            except Exception as exc:
                logger.error("Window %d failed: %s", window_id, exc, exc_info=True)
                raise exc

        metrics_df = pd.DataFrame(all_metric_rows)
        fva_df = pd.DataFrame(all_fva_rows)

        elapsed = time.time() - start_time
        logger.info("Completed %d windows in %.2f seconds.", len(window_results), elapsed)

        return metrics_df, fva_df, window_results

    def get_last_predictions(self) -> dict[str, Any] | None:
        """Returns the actuals and model predictions from the most recently executed window."""
        if self.last_window_result is None:
            return None
        return {
            "eval_actuals": self.last_window_result.eval_actuals,
            "predictions": self.last_window_result.model_predictions,
            "window_id": self.last_window_result.window_id,
        }
