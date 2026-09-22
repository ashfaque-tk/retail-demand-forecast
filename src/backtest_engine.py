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
from typing import Any,Optional, Mapping, Sequence

import pandas as pd

from src.backtest_windows import (
    generate_expanding_windows,
    generate_rolling_windows,
    split_data,
)
from src.baselines import seasonal_naive, simple_moving_average
from src.features import FeatureBuilder
from src.inventory_policy import (
    compute_rolling_tau_error,calculate_error_statistics,
     InventoryPolicy
)
from src.metrics import fva, get_all_metrics
from src.model_selector import SelectModel
from src.forecaster import Forecaster
from src.utils import get_items_with_min_history
from config import PIPELINE_CONFIG

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
    fva_rows: list[dict]
    model_predictions: dict[str, pd.DataFrame]
    eval_actuals: pd.DataFrame
    inventory_policy: dict[str, pd.DataFrame] = field(default_factory=dict)

@dataclass
class DeploymentResult:
    """Production deployment outputs: operational decisions, forecasts, and metadata."""
    
    # 1. Operational Decisions (What PostgreSQL & business planners need)
    forecasts: pd.DataFrame          # [item_id, date, sales_pred, (quantiles)]
    inventory_policy: pd.DataFrame   # [item_id, review_date, safety_stock, order_up_to]
    
    # 2. Financial & Performance Estimates
    inventory_cost_summary: pd.Series  # Holding cost estimate across policies
    calibration_metrics: dict[str, float]  # OOS validation metrics (MAE, WRMSSE)
    
    # 3. Execution Metadata (Audit log)
    model_name: str
    run_timestamp: pd.Timestamp
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp
    
    # 4. Optional: Baseline comparison (only populated if offline validation run)
    baseline_metrics: dict[str, dict[str, float]] | None = None

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
        forecast_type:str = 'recursive',
        training_window_days: int = 730,
        horizon_days: int = 28,
        backtest_mode: str = "rolling",
        step_size_days: int = 28, # continous rolling windows
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
        self.forecast_type = forecast_type.lower()
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
        self.oos_errors: dict[str, list[pd.Series]] = {
            "ml": [],
            "ma": [],
            "naive": [],
        }
        # inventory holder
        self.inventory: dict[str, Optional[InventoryPolicy]] = {
            "naive": None,
            "ma": None,
            "ml": None,
        }
        ### initlize a similar one for  storing actuals and preds
        self.actuals_preds :dict[str,pd.DataFrame] = {'ml':pd.DataFrame(),
                                                      'ma':pd.DataFrame(),
                                                      'naive':pd.DataFrame()}

        # Cache for last executed window outputs (for easy plotting / inspection)
        self.last_window_result: WindowResult | None = None

        # Core pipeline components
        self.feature_builder = FeatureBuilder()
        quantiles = PIPELINE_CONFIG.get("quantiles") or None # returns a list of quantiles that we want find

        self.model = SelectModel(
            model=self.model_name,
            quantiles= quantiles,
            use_log_transform=self.use_log_transform,
            categorical_cols=self.categorical_cols,
        )
        self.forecaster = Forecaster(model=self.model, feature_builder=self.feature_builder,
                                     original_features=self.feature_names,forecast_type=self.forecast_type)

        
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

        # ---------------------------------------------------------
        # 1. Prepare window data
        # ---------------------------------------------------------

        window_slice = full_data[
            full_data["date"].between(train_start, eval_end)
        ]

        active_items_df = self.feature_builder.add_price_features(
            window_slice
        )

        train_wnd, eval_wnd = split_data(
            df=active_items_df,
            start_date=train_start,
            end_date=train_end,
            forecast_horizon=self.horizon_days,
        )

        logger.debug(
            "Window %d: train=%s, eval=%s",
            window_id,
            train_wnd.shape,
            eval_wnd.shape,
        )

        # ---------------------------------------------------------
        # 2. Baseline forecasts
        # ---------------------------------------------------------

        baseline_naive = seasonal_naive(
            train_wnd,
            eval_wnd,
        )

        baseline_ma = simple_moving_average(
            train_wnd,
            eval_wnd,
            window_days=180,
        )

        # ---------------------------------------------------------
        # 3. Baseline metrics
        # ---------------------------------------------------------

        metrics_naive = get_all_metrics(
            train_wnd,
            eval_wnd,
            baseline_naive,
            risk_period=self.tau_days
        )

        metrics_ma = get_all_metrics(
            train_wnd,
            eval_wnd,
            baseline_ma,
            risk_period=self.tau_days
        )

        # ---------------------------------------------------------
        # 4. Build ML features
        # ---------------------------------------------------------

        train_wnd, _ = self.feature_builder.build(
            df=train_wnd,
            lags=[7, 28, 60, 90],
            rolling_maxs=[7, 28, 60, 90],
            rolling_means=[7, 28, 60, 90],
            rolling_on_lags={
                28: [7, 28]
            },
        )

        missing_features = [
            col
            for col in feats + ["date", "sales"]
            if col not in train_wnd.columns
        ]

        if missing_features:
            raise ValueError(
                f"Window {window_id}: "
                f"Missing expected features: {missing_features}"
            )

        # ---------------------------------------------------------
        # 5. ML forecast
        # ---------------------------------------------------------
        # # debugg for one item
        # train_wnd = train_wnd[train_wnd['item_id']==item]
        # eval_wnd = eval_wnd[eval_wnd['item_id']==item]

        forecasted_demands = self.forecaster.forecast(
            train_df=train_wnd,
            test_df=eval_wnd,
        )

        metrics_ml = get_all_metrics(
            train_wnd,
            eval_wnd,
            forecasted_demands,
            risk_period=self.tau_days
        )

        # ---------------------------------------------------------
        # 6. Collect predictions
        # ---------------------------------------------------------

        models_preds = {
            "ml": forecasted_demands,
            "ma": baseline_ma,
            "naive": baseline_naive,
        }

        # ---------------------------------------------------------
        # 7. Inventory evaluation
        # ---------------------------------------------------------

        inventory_costs: dict[str, pd.Series] = {
            "naive": pd.Series(),
            "ma": pd.Series(),
            "ml": pd.Series(),
        }

        inventory_results: dict[str, pd.DataFrame] = {
            "naive": pd.DataFrame(),
            "ma": pd.DataFrame(),
            "ml": pd.DataFrame(),
        }

        for short_name, current_preds in models_preds.items():

            forecasts_df_model = (
                eval_wnd
                .merge(
                    current_preds,
                    on=["item_id", "date"],
                    how="inner",
                )
                .sort_values(["item_id", "date"])
                .copy()
            )

            self.actuals_preds[short_name] = pd.concat(
                [
                    self.actuals_preds[short_name],
                    forecasts_df_model,
                ],
                ignore_index=True,
            )

            # ---------------------------------------------
            # First window: initialise inventory + OOS error
            # ---------------------------------------------

            if window_id == 0:

                initial_inventory = (
                    eval_wnd
                    .groupby(
                        "item_id",
                        observed=True,
                    )["sales"]
                    .mean()
                )

                error_model = compute_rolling_tau_error(
                    forecasts=forecasts_df_model,
                    tau=self.tau_days,
                )

                error_stats = calculate_error_statistics(
                    error_model
                )

                self.oos_errors[short_name].append(
                    error_stats
                )

                self.inventory[short_name] = InventoryPolicy(
                    inventory=initial_inventory,
                    forecasts=forecasts_df_model,
                    lead_time=4,
                    review_period=7,
                    daily_unit_holding_cost=0.2,
                    stockout_cost=1,
                    oos_error_list=self.oos_errors[short_name],
                )

            # ---------------------------------------------
            # Subsequent windows: simulate inventory
            # ---------------------------------------------

            elif self.oos_errors[short_name]:
                policy = self.inventory[short_name]
                if policy is not None:
                    inventory_results[short_name] = policy.daily_simulation(
                    actual_sales=eval_wnd,
                    forecasted_demand=current_preds,
                ) 
                    
                    inventory_costs[short_name] = inventory_results[short_name][
                        ['holding_cost','stockout_cost']].sum().copy()
                    inventory_costs[short_name]['total_cost'] = inventory_costs[short_name].sum().sum()

                   
        # if window_id> 0:
        #     print(inventory_costs)
        #     quit()

           
        # ---------------------------------------------------------
        # 8. Assemble metric records
        # ---------------------------------------------------------

        model_metrics_map = {
            "naive": metrics_naive,
            "ma": metrics_ma,
            "ml": metrics_ml,
        }

        metric_rows = self._build_metric_rows(
            window_id=window_id,
            train_start=train_start,
            train_end=train_end,
            model_metrics_map=model_metrics_map,
            inventory_costs=inventory_costs,
        )

        # ---------------------------------------------------------
        # 9. Calculate FVA
        # ---------------------------------------------------------

        fva_rows = self._calculate_fva(
            metric_rows
        )

        # ---------------------------------------------------------
        # 10. Return complete window result
        # ---------------------------------------------------------

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
            inventory_policy=inventory_results,
        )

        self.last_window_result = result

        return result
    
    def _calculate_fva(self,
                        metric_rows: list[dict],
                    ) -> list[dict]:

        df_metrics = pd.DataFrame(metric_rows)

        pivot_df = (
            df_metrics
            .pivot(
                index=["window_id", "train_start", "train_end"],
                columns="model",
            )
            .stack(level=0, future_stack=True)
            .reset_index()
        )

        target_metrics = [
            "MAE",
            "wrmsse",
            "cum_MAE",
            "holding_cost",
            "stockout_cost",
            "total_cost"
        ]

        fva_df = (
            pivot_df[
                pivot_df["level_3"].isin(target_metrics)
            ]
            .copy()
            .rename(columns={"level_3": "metric"})
        )

        # Make sure the metric columns are numeric
        model_cols = [
            "seasonal_naive",
            "moving_average",
            "lgbm",
        ]

        for col in model_cols:
            if col in fva_df.columns:
                fva_df[col] = pd.to_numeric(
                    fva_df[col],
                    errors="coerce",
                )

        fva_df["fva_moving_average_vs_naive"] = self.fva(
            fva_df["seasonal_naive"],
            fva_df["moving_average"],
        )

        fva_df["fva_model_vs_naive"] = self.fva(
            fva_df["seasonal_naive"],
            fva_df["lgbm"],
        )
        fva_df = fva_df[['window_id','metric','fva_moving_average_vs_naive','fva_model_vs_naive']]
        return fva_df.to_dict(orient="records")

    def fva(self, baseline: pd.Series, model: pd.Series) -> pd.Series:
        return ((baseline - model) * 100 / baseline).round(2)

    def _build_metric_rows( self,
                            window_id: int,
                            train_start: pd.Timestamp,
                            train_end: pd.Timestamp,
                            model_metrics_map: dict[str, dict],
                            inventory_costs: dict[str, pd.Series],
                        ) -> list[dict]:

        name_to_label = {
            "naive": "seasonal_naive",
            "ma": "moving_average",
            "ml": self.model_name,
        }

        rows = []
      
        for short_name, metrics in model_metrics_map.items():

            row = {
                "window_id": window_id,
                "train_start": train_start.date(),
                "train_end": train_end.date(),
                "model": name_to_label[short_name],
                "MAE": metrics["MAE"],
                "BIAS%": metrics["BIAS%"],
                "wrmsse": metrics["wrmsse"],
                "cum_BIAS" : metrics['cum_BIAS'],
                "cum_MAE"  : metrics['cum_MAE']
            }

            row.update(inventory_costs.get(short_name, {}))
            rows.append(row)

        return rows
    
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

    def run_deploy(
        self,
        historical_data: pd.DataFrame,
        future_static_data: pd.DataFrame,
        calibration_days: int | None = None,
    ) -> DeploymentResult:
        """Fit a final model on all historical data and generate production demand forecasts.

        Holds out the final `calibration_days` of historical data once to estimate
        the tau-day forecast-error distribution and safety-stock parameters.
        The model is then refit on all available historical data before forecasting
        `future_static_data`.
        """
        calibration_days = calibration_days or self.horizon_days
        if calibration_days < self.tau_days:
            raise ValueError(
                f"calibration_days ({calibration_days}) must be at least tau_days ({self.tau_days})."
            )
        if historical_data.empty or future_static_data.empty:
            raise ValueError("historical_data and future_static_data must both be non-empty.")

        # 1. Clean and sort historical and future date indexes
        history = historical_data.copy()
        future = future_static_data.copy()
        history["date"] = pd.to_datetime(history["date"])
        future["date"] = pd.to_datetime(future["date"])
        history = history.sort_values(["item_id", "date"])
        future = future.sort_values(["item_id", "date"])

        history_dates = pd.Index(history["date"].drop_duplicates().sort_values())
        forecast_dates = pd.Index(future["date"].drop_duplicates().sort_values())

        if len(history_dates) < calibration_days + 1:
            raise ValueError("Not enough historical dates for a train/calibration split.")
        if forecast_dates[0] != pd.to_datetime(history_dates[-1])+ pd.Timedelta(days=1):
            raise ValueError(
                "future_static_data must begin on the exact day after the last historical date."
            )

        # 2. Build contiguous price features before splitting windows
        combined_data = pd.concat([history, future], ignore_index=True).sort_values(["item_id", "date"])
        price_data = self.feature_builder.add_price_features(combined_data)

        # 3. Define date bounds for Calibration and Production passes
        calibration_dates = history_dates[-calibration_days:]
        calibration_start = pd.Timestamp(calibration_dates[0])
        calibration_end = pd.Timestamp(calibration_dates[-1])

        calibration_train = price_data[price_data["date"] < calibration_start].copy()
        calibration_eval = price_data[price_data["date"].isin(calibration_dates)].copy()
        final_train = price_data[price_data["date"].isin(history_dates)].copy()
        final_forecast = price_data[price_data["date"].isin(forecast_dates)].copy()

        def _build_features(raw_train: pd.DataFrame) -> pd.DataFrame:
            frame, _ = self.feature_builder.build(
                df=raw_train,
                lags=[7, 28, 60, 90],
                rolling_means=[7, 28, 60, 90],
                rolling_maxs=[7, 28, 60, 90],
                rolling_on_lags={28: [7, 28]},
            )
            missing = [col for col in self.feature_names + ["sales"] if col not in frame.columns]
            if missing:
                raise ValueError(f"Deployment training frame missing expected features: {missing}")
            return frame

        # 4. Calibration Phase: Estimate out-of-sample forecast errors
        calib_train_feats = _build_features(calibration_train)
        calib_predictions = self.forecaster.forecast(train_df=calib_train_feats, test_df=calibration_eval)
        calib_metrics = get_all_metrics(calib_train_feats, calibration_eval, calib_predictions,
                                        risk_period = self.tau_days)

        merged_calib = (
            calibration_eval.merge(calib_predictions, on=["item_id", "date"], how="inner")
            .sort_values(["item_id", "date"])
            .copy()
        )
        calib_error_df = compute_rolling_tau_error(forecasts=merged_calib, tau=self.tau_days)
        error_stats = calculate_error_statistics(calib_error_df)

        # 5. Production Phase: Re-train on 100% of historical data and forecast future horizon
        final_train_feats = _build_features(final_train)
        production_predictions = self.forecaster.forecast(
            train_df=final_train_feats, test_df=final_forecast
        )

        # 6. Construct DeploymentResult output
        return DeploymentResult(
            forecasts=production_predictions,
            inventory_policy=pd.DataFrame(),  # Optional: Place simulated inventory policy frame here if needed
            inventory_cost_summary=error_stats,
            calibration_metrics=calib_metrics,
            model_name=self.model_name,
            run_timestamp=pd.Timestamp.now(),
            forecast_start=pd.Timestamp(forecast_dates[0]),
            forecast_end=pd.Timestamp(forecast_dates[-1]),
        )

    def get_last_predictions(self) -> dict[str, Any] | None:
        """Returns the actuals and model predictions from the most recently executed window."""
        if self.last_window_result is None:
            return None
        return {
            "eval_actuals": self.last_window_result.eval_actuals,
            "predictions": self.last_window_result.model_predictions,
            "window_id": self.last_window_result.window_id,
        }
