"""Backtest Engine for retail demand forecasting and inventory replenishment evaluation.

Orchestrates walk-forward validation windows (rolling or expanding), baseline comparisons,
ML model training, recursive multi-step forecasting, out-of-sample error tracking, and
periodic inventory replenishment policy evaluation.

Baselines are supplied by name and dispatched through
:class:`src.baselines.Baseline`, so adding a method to that class requires no edit
here: the engine never names an individual baseline outside ``DEFAULT_BASELINES`` and
the per-baseline kwargs mapping. Every model that is scored - the ML model plus each
configured baseline - is carried through the same metric, FVA and inventory paths, so
a new baseline is reported on the same terms as the incumbent ones.

Eliminates mutable global state and provides clean, structured data containers.
"""
from __future__ import annotations

import inspect
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
from src.baselines import Baseline
from src.features import FeatureBuilder
from src.inventory_policy import (
    compute_rolling_tau_error,calculate_error_statistics,
     InventoryPolicy
)
from src.metrics import MetricsCalculator, fva
from src.model_selector import SelectModel
from src.forecaster import Forecaster
from src.utils import get_items_with_min_history
from config import PIPELINE_CONFIG

logger = logging.getLogger(__name__)

DEFAULT_BASELINES: tuple[str, ...] = (
    "seasonal_naive",
    "simple_moving_average",
    "seasonal_moving_average",
    "croston",
    "croston_sba"
)

FVA_METRICS: tuple[str, ...] = (
    "MAE",
    "wrmsse",
    "WAPE",
    "cum_MAE",
    "holding_cost",
    "stockout_cost",
    "total_cost",
)


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
    lead_time_days : int, default 4
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
    baselines : Sequence[str] | Mapping[str, dict] | None
        Baselines to score alongside the ML model. A list of names uses each
        method's default hyperparameters; a mapping of ``name -> kwargs`` passes
        method-specific hyperparameters (e.g. ``{"seasonal_naive": {"lag_days": 7}}``).
        Names must be :attr:`src.baselines.Baseline.METHOD_NAMES` entries that
        resolve to a callable method. Defaults to :data:`DEFAULT_BASELINES`.
    fva_reference : str | None, optional
        Baseline that every other model is measured against in the FVA table.
        Defaults to the first entry of ``baselines``. Must be one of ``baselines``.
    baseline_quantiles : tuple[float, ...] | None, optional
        Quantile levels handed to :class:`~src.baselines.Baseline`. ``None``
        (default) keeps baseline output point-only, matching the ML path.
    """

    def __init__(
        self,
        model_name: str = "lgbm",
        forecast_type:str = 'recursive',
        training_window_days: int = 365,#days 
        horizon_days: int = 28,
        backtest_mode: str = "rolling",
        step_size_days: int = 28, # continous rolling windows
        categorical_cols: list[str] | None = None,
        feature_names: list[str] | None = None,
        lead_time_days: int = 11,
        review_period_days: int = 7,
        holding_cost_rate: float = 0.02,
        stockout_rate : float = 1.0,
        min_history_days: int = 100,
        max_windows: int | None = None,
        use_log_transform: bool = False,
        baselines : Sequence[str] | Mapping[str, dict[str, Any]] | None = None,
        fva_reference : str | None = None,
        baseline_quantiles : list[float] | None = None,
        safety_stock_policy : str = 'rmse',
    ) -> None:
        self.model_name = model_name
        self.forecast_type = forecast_type.lower()
        self.training_window_days = training_window_days
        self.horizon_days = horizon_days
        self.backtest_mode = backtest_mode
        self.step_size_days = step_size_days
        self.categorical_cols = categorical_cols or ["item_id",  "dept_id"]
        self.feature_names = feature_names or []
        self.lead_time_days = lead_time_days
        self.review_period_days = review_period_days
        self.holding_cost_rate = holding_cost_rate
        self.min_history_days = min_history_days
        self.max_windows = max_windows
        self.use_log_transform = use_log_transform
        self.stockout_rate = stockout_rate
        self.baseline_quantiles = baseline_quantiles
        if safety_stock_policy not in ("rmse", "quantile"):
            raise ValueError(
                f"safety_stock_policy must be 'rmse' or 'quantile', "
                f"got {safety_stock_policy!r}"
            )
        if safety_stock_policy == "quantile" and not baseline_quantiles:
            raise ValueError(
                "safety_stock_policy='quantile' needs quantile levels. Set "
                "PIPELINE_CONFIG['quantiles'] (config.critical_quantile(holding) "
                "supplies the cost-implied one) or pass --quantiles."
            )
        self.safety_stock_policy = safety_stock_policy
        # Review period + lead time total risk horizon (tau)
        self.tau_days = self.lead_time_days + self.review_period_days

        # Baselines are resolved once, here, so an unregistered or misspelled name
        # fails at construction rather than quietly dropping out of every report.
        self.baselines, self.baseline_params = self._resolve_baselines(baselines)
        self.fva_reference = self._resolve_fva_reference(fva_reference)
        # Every model this run scores, ML first. All per-model state below is keyed
        # by these names, so a new baseline needs no edit outside _resolve_baselines.
        self.tracked_models: list[str] = [self.model_name, *self.baselines]

        # Encapsulated stateful error buffers (replaces mutable module globals)
        self._init_state_containers()

        # Cache for last executed window outputs (for easy plotting / inspection)
        self.last_window_result: WindowResult | None = None

        # Core pipeline components
        self.feature_builder = FeatureBuilder()

        self.model = SelectModel(
            model=self.model_name,
            quantiles= self.baseline_quantiles,
            use_log_transform=self.use_log_transform,
            categorical_cols=self.categorical_cols,
        )
        self.forecaster = Forecaster(model=self.model, feature_builder=self.feature_builder,
                                     original_features=self.feature_names,forecast_type=self.forecast_type)

    def _resolve_baselines(
        self,
        baselines: Sequence[str] | Mapping[str, dict[str, Any]] | None,
    ) -> tuple[list[str], dict[str, dict[str, Any]]]:
        """Normalise the baseline spec to (names, per-method kwargs).

        Validation is strict on purpose. The previous if/elif dispatch logged a
        warning and skipped an unrecognised name, which produced a metrics table
        that silently omitted a baseline the caller believed was being scored.
        """
        if baselines is None:
            baselines = DEFAULT_BASELINES

        if isinstance(baselines, Mapping):
            params = {str(name): dict(kwargs or {}) for name, kwargs in baselines.items()}
        else:
            params = {str(name): {} for name in baselines}

        names = list(params)
        if not names:
            raise ValueError("baselines is empty; nothing to compare the model against.")

        unknown = [n for n in names if n not in Baseline.METHOD_NAMES]
        if unknown:
            raise ValueError(
                f"Unknown baseline(s) {unknown}. Valid names: {Baseline.METHOD_NAMES}"
            )

        # METHOD_NAMES is a declaration; a name can be declared without a matching
        # method, in which case getattr would fail deep inside the window loop.
        unimplemented = [n for n in names if not callable(getattr(Baseline, n, None))]
        if unimplemented:
            available = [n for n in Baseline.METHOD_NAMES if callable(getattr(Baseline, n, None))]
            raise ValueError(
                f"Baseline(s) {unimplemented} are listed in Baseline.METHOD_NAMES but "
                f"no such method exists on src.baselines.Baseline. Callable methods: {available}"
            )

        logger.info("Baselines configured: %s", names)
        return names, params

    def _resolve_fva_reference(self, fva_reference: str | None) -> str:
        """Pick the baseline every other model is measured against."""
        if fva_reference is None:
            return self.baselines[0]
        if fva_reference not in self.baselines:
            raise ValueError(
                f"fva_reference {fva_reference!r} is not one of the configured "
                f"baselines {self.baselines}."
            )
        return fva_reference

    def _init_state_containers(self) -> None:
        """(Re)build the per-model buffers keyed by :attr:`tracked_models`."""
        self.oos_errors: dict[str, list[pd.Series]] = {name: [] for name in self.tracked_models}
        # inventory holder
        self.inventory: dict[str, Optional[InventoryPolicy]] = {name: None for name in self.tracked_models}
        ### initlize a similar one for  storing actuals and preds
        self.actuals_preds: dict[str, pd.DataFrame] = {name: pd.DataFrame() for name in self.tracked_models}

    def _get_feature_importance(self):
        return pd.Series(self.model.model_point.feature_importances_,index=self.model.features).sort_values(ascending=False)


    def reset_state(self) -> None:
        """Clears accumulated out-of-sample errors, predictions and cached window results."""
        self._init_state_containers()
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


    def get_forecasts(
            self,
            window_id:int,
            train_df:pd.DataFrame,
            eval_df: pd.DataFrame,
        )->tuple[dict[str,pd.DataFrame],dict[str,dict[str,Any]]]:
        '''return forecasted values and corresponding metrics for given train and eval dfs'''
        
        logger.info(f'calculating forecasts and metrics for the models')

        # ---------------------------------------------------------
        # 1. Build ML features
        # ---------------------------------------------------------

        train_wnd, _ = self.feature_builder.build(
            df=train_df,  # features are already given in feature_builder
        )

        missing_features = [
            col
            for col in self.feature_names + ["date", "sales"]
            if col not in train_wnd.columns
        ]

        if missing_features:
            raise ValueError(
                f"Window {window_id}: "
                f"Missing expected features: {missing_features}"
            )

        # ---------------------------------------------------------
        #  ML forecast
        # ---------------------------------------------------------
        forecasted_demands = self.forecaster.forecast(
            train_df=train_wnd,
            test_df=eval_df,
        )

        metrics_ml = MetricsCalculator(
            train_wnd,
            eval_df,
            forecasted_demands,
            risk_period=self.tau_days,
        ).all_metrics()

        # ----------------------------------------------------------
        # Baseline forecast and predictions 
        # ----------------------------------------------------------
        base_preds, base_metrics = self._baselines_metrics(
            train_df=train_df,
            eval_df= eval_df)

        # -----------------------------------------------------------
        # collecting the forecasts and metrics 
        # -----------------------------------------------------------
        models_preds:dict[str,pd.DataFrame] = {
            self.model_name : forecasted_demands,
            **base_preds
        }


        models_metrics:dict[str,dict[str,Any]] = {
            self.model_name : metrics_ml, #model name 
            **base_metrics
        }

    
        return models_preds, models_metrics

    def __populate_actual_vs_preds(self,
                                models_preds:dict[str,pd.DataFrame],
                                actual_df:pd.DataFrame):
        ''' populate actual_preds dictionary with values'''

        for short_name, cal_preds in models_preds.items():
                # A prediction frame may carry more than the point forecast: the
                # baselines attach their own scoring keys (dept_id, cat_id,
                # real_sales) and each method keeps whatever intermediates it
                # built. Merging those against actual_df produces dept_id_x /
                # dept_id_y suffixed columns and method-specific debris, so only
                # the join keys, the point forecast and quantile levels are kept.
                # actual_df is the single source for everything else.
                pred_cols = ["sales_pred", *[c for c in cal_preds.columns if c.startswith("q")]]

                # Merge evaluation actuals with current model predictions
                forecasts_df_model = (
                                actual_df
                                .merge(
                                    cal_preds[["item_id", "date", *pred_cols]],
                                    on=["item_id", "date"],
                                    how="inner",
                                )
                                .sort_values(["item_id", "date"])
                                .copy()
                            )

                # Store actuals vs predictions across folds
                self.actuals_preds[short_name] = pd.concat(
                    [
                        self.actuals_preds.get(short_name, pd.DataFrame()),
                        forecasts_df_model,
                    ],
                    ignore_index=True,
                )
            

    def run_window(
            self,
            window_id:int,
            train_wnd:pd.DataFrame,
            eval_wnd: pd.DataFrame, # hold out set in case of final testing
            train_start:pd.Timestamp,
            train_end: pd.Timestamp,
            eval_start:pd.Timestamp,
            eval_end : pd.Timestamp
    )-> WindowResult:
        ''' return metrics, inventory , forecasts for a current window,
        a calibration set is used for window 0 to return the forecasts needed for inventory calculations'''

        #  debug line
        logger.debug(
            "Window %d: train=%s, eval=%s",
            window_id,
            train_wnd.shape,
            eval_wnd.shape,
        )
        if 'dept_mean_price' not in train_wnd.columns:
            logging.error(f'Missing price cols. First calculate price features on both '
                          'train and eval set')    
            quit()
        # if window 0: we need forecast for a prior horizon for rmse_error in inventory calculations
        # we split the train_wnd to calibration_train, calibration_eval
        profile = self.feature_builder.build_sku_demand_profile(train_wnd)
        train_wnd = train_wnd.merge(profile,on='item_id',how='left')
        eval_wnd = eval_wnd.merge(profile,on='item_id',how='left')
        
        if window_id == 0:
            logger.debug(f'using calibration set for window_id=0')
            # build the price feature set on full train data 
            calibration_set = train_wnd.copy()
            # calibration_price = self.feature_builder.add_price_features(calibration_set)#added price
            # split into train and eval
            cal_train_start = calibration_set['date'].min()
            cal_train_end = calibration_set['date'].max()-pd.Timedelta(days=self.horizon_days)

            cal_train,cal_eval = split_data(df=calibration_set,
                                            start_date=cal_train_start,
                                            end_date=cal_train_end,
                                            forecast_horizon=self.horizon_days)

            #  return the forecasts
            cal_model_preds, _ = self.get_forecasts(window_id= window_id,
                                                    train_df=cal_train,
                                                    eval_df=cal_eval)

            self.__populate_actual_vs_preds(models_preds=cal_model_preds,actual_df=cal_eval)


        model_preds, model_metrics = self.get_forecasts(window_id=window_id,
                                                        train_df=train_wnd,
                                                        eval_df=eval_wnd)
        
        self.__populate_actual_vs_preds(models_preds=model_preds,actual_df=eval_wnd)

        # ---------------------------------------------------------
        # 7. Inventory evaluation
        # ---------------------------------------------------------

        # Reinitialize initial inventory for every window from training history
        initial_inventory = (
            train_wnd
            .groupby("item_id", observed=True)["sales"]
            .apply(lambda x: x.tail(28).mean())
        ) *self.tau_days  # Cover risk period / lead time duration

        inventory_costs: dict[str, pd.Series] = {}
        inventory_results: dict[str, pd.DataFrame] = {}

        for short_name, current_forecasts in model_preds.items():
            forecasts_model = self.actuals_preds[short_name]
            # Instantiate Inventory Policy using calibrated OOS errors & fresh initial stock
            policy = InventoryPolicy(
                inventory=initial_inventory,
                forecasts=forecasts_model,
                lead_time=self.lead_time_days,
                review_period=self.review_period_days,
                daily_unit_holding_cost=self.holding_cost_rate,
                stockout_cost=self.stockout_rate,
                safety_stock_type=self.safety_stock_policy
            )

            self.inventory[short_name] = policy

            # Run daily simulation
            sim_df = policy.daily_simulation(
                actual_sales=eval_wnd,
                forecasted_demand=current_forecasts,
                model_name=short_name,
                forecast_type=self.forecast_type,
                store_id= 'CA_1'
            )
            inventory_results[short_name] = sim_df

            # Calculate holding, stockout, and total costs
            cost_summary = sim_df[['holding_cost', 'stockout_cost']].sum()
            cost_summary['total_cost'] = cost_summary.sum()
            inventory_costs[short_name] = cost_summary
                   
        # ---------------------------------------------------------
        # 8. Assemble metric records
        # ---------------------------------------------------------

        metric_rows = self._build_metric_rows(
            window_id=window_id,
            train_start=train_start,
            train_end=train_end,
            model_metrics_map=model_metrics,
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
            model_predictions=model_preds,
            eval_actuals=eval_wnd,
            inventory_policy=inventory_results,
        )

        self.last_window_result = result

        return result

    def _calculate_fva(self,
                        metric_rows: list[dict],
                    ) -> list[dict]:
        """Forecast Value Added of every scored model against the reference baseline.

        One row per (window_id, metric) with a ``fva_<model>_vs_<reference>`` column
        per model, so a newly configured baseline appears in the FVA table without
        this method naming it. The previous implementation listed the incumbent
        baselines and the model name explicitly, which meant a new baseline got
        metrics but never an FVA column.
        """
        if not metric_rows:
            return []

        df_metrics = pd.DataFrame(metric_rows)

        key_cols = ["window_id", "train_start", "train_end", "model"]
        value_cols = [c for c in df_metrics.columns if c not in key_cols]
        target_metrics = [m for m in FVA_METRICS if m in value_cols]

        present = [m for m in self.tracked_models if m in set(df_metrics["model"])]
        if not present or not target_metrics:
            return []

        reference = self.fva_reference if self.fva_reference in present else present[0]
        if reference != self.fva_reference:
            logger.warning(
                "FVA reference baseline %r is absent from this window; falling back to %r.",
                self.fva_reference, reference,
            )

        long_df = (
            df_metrics
            .melt(
                id_vars=key_cols,
                value_vars=target_metrics,
                var_name="metric",
                value_name="value",
            )
        )
        long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")

        ref_df = (
            long_df[long_df["model"] == reference]
            [["window_id", "train_start", "train_end", "metric", "value"]]
            .rename(columns={"value": "reference_value"})
        )
        long_df = long_df.merge(
            ref_df,
            on=["window_id", "train_start", "train_end", "metric"],
            how="left",
        )

        rows: list[dict] = []
        for (window_id, metric), grp in long_df.groupby(
            ["window_id", "metric"], observed=True, sort=True
        ):
            values = dict(zip(grp["model"], grp["value"]))
            ref_value = values.get(reference, float("nan"))
            row: dict[str, Any] = {"window_id": window_id, "metric": metric}
            for model_name in present:
                row[f"fva_{model_name}_vs_{reference}"] = self.fva(
                    ref_value, values.get(model_name, float("nan"))
                )
            rows.append(row)

        return rows

    def fva(self, baseline: float, model: float) -> float:
        """Percent improvement of ``model`` over ``baseline``; NaN when undefined.

        A zero or missing baseline value has no meaningful percentage, and
        returning NaN keeps it out of the report averages instead of emitting an
        infinite or fabricated number.
        """
        if pd.isna(baseline) or pd.isna(model) or baseline == 0:
            return float("nan")
        return round((baseline - model) * 100 / baseline, 2)

    def _build_metric_rows( self,
                            window_id: int,
                            train_start: pd.Timestamp,
                            train_end: pd.Timestamp,
                            model_metrics_map: dict[str, dict],
                            inventory_costs: dict[str, pd.Series],
                        ) -> list[dict]:


        rows = []
      
        for short_name, metrics in model_metrics_map.items():

            row = {
                "window_id": window_id,
                "train_start": train_start.date(),
                "train_end": train_end.date(),
                "model": short_name,
                "MAE": metrics["MAE"],
                "BIAS%": metrics["BIAS%"],
                "wrmsse": metrics["wrmsse"],
                'WAPE'  : metrics['WAPE'],
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
                train_start = pd.Timestamp(wnd_spec["train_start"])
                train_end = pd.Timestamp(wnd_spec["train_end"])
                eval_start = pd.Timestamp(wnd_spec["test_start"])
                eval_end = pd.Timestamp(wnd_spec["test_end"])
        
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
        
                window_slice_price = self.feature_builder.add_price_features(
                    window_slice
                )# added price on whole slice
        
                train_wnd, eval_wnd = split_data(
                    df=window_slice_price,
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
            
                res = self.run_window(
                    window_id=window_id,
                    train_wnd=train_wnd,
                    eval_wnd = eval_wnd,
                    train_start= train_start,
                    train_end = train_end,
                    eval_start= eval_start,
                    eval_end= eval_end
                
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

    def run_holdout_test(
            self,
            full_data :pd.DataFrame,#fulldataframe
            test_holdout:pd.DataFrame,
            training_year:int = 1, #1 year of training data
            ):

        '''take the full train data split to the given year, and test on holdout set
        test_holdout: explicitly require the holdout set '''

        logger.info("Final Holdout Test: Model: '%s' | Type: ' %s'| Training-Period: '%d'",
                    self.model_name,self.forecast_type,training_year)

        train_window  = training_year*365 # to days
        window_id  = 0
  
        training_full = full_data[
            full_data['date']>=(full_data['date'].max()-pd.Timedelta(
                days=train_window)
                )]
                
        logger.debug(
            "Window %d: train=%s, test=%s",
            window_id,
            training_full.shape,
            test_holdout.shape,
        )

        train_start=training_full['date'].min()
        train_end = training_full['date'].max()

        test_start = test_holdout['date'].min()
        test_end = test_holdout['date'].max()
        
        
        logger.info(
                    "--- Holdout Test: Train [%s -> %s] | Eval [%s -> %s] ---",
                    train_start.date(),
                    train_end.date(),
                    test_start.date(),
                    test_end.date(),
                )

        ######## merge train and test for adding price features #######
        merged_ = pd.concat((training_full,test_holdout),axis=0)
        assert merged_['date'].max()==test_end,'AssersionError: Merging faulty'

        ### price features 
        merged_price = self.feature_builder.add_price_features(merged_)

        #now split again

        train_price,test_price = split_data(
            df=merged_price,
            start_date=train_start,
            end_date=train_end,
            forecast_horizon=self.horizon_days)

        res = self.run_window(
            window_id=window_id,
            train_wnd= train_price,
            eval_wnd = test_price,
            train_start= train_start,
            train_end = train_end,
            eval_start= test_start,
            eval_end= test_end
        
        )
        model_preds = res.model_predictions
        model_inventory = res.inventory_policy

        return model_preds,model_inventory,res.metric_rows, res.fva_rows
        


    def _baselines_metrics(
            self,
            train_df:pd.DataFrame,
            eval_df:pd.DataFrame
        ) -> tuple[dict[str,pd.DataFrame],dict[str,dict[str,Any]]]:
        """Forecast and score every configured baseline for one window.

        A single :class:`~src.baselines.Baseline` is constructed per window and
        dispatched to by name through ``Baseline.run``, replacing the previous
        if/elif chain that recognised only seasonal naive and moving average and
        skipped everything else with a warning.
        """

        baseline = Baseline(
            train_df=train_df,
            test_df=eval_df,
            quantiles=self.baseline_quantiles,
        )

        baseline_preds:dict[str,pd.DataFrame] = {}
        baseline_metrics:dict[str,dict[str,Any]] = {}

        for b_name in self.baselines:
            params = dict(self.baseline_params.get(b_name, {}))

            # Any method that forecasts a fixed horizon gets the engine's horizon
            # unless the caller pinned it, so a baseline cannot silently score a
            # 28-day default against a differently sized evaluation window.
            signature = inspect.signature(getattr(type(baseline), b_name))
            if "horizon" in signature.parameters:
                params.setdefault("horizon", self.horizon_days)

            try:
                preds = baseline.run(b_name, **params)
            except NotImplementedError as exc:
                raise NotImplementedError(
                    f"Baseline {b_name!r} is configured on this engine but raises "
                    f"NotImplementedError. Implement it in src/baselines.py or "
                    f"remove it from the `baselines` argument. Original error: {exc}"
                ) from exc
            except TypeError as exc:
                # Almost always a hyperparameter the method does not accept, e.g.
                # alpha on a statsforecast model that estimates it internally.
                raise TypeError(
                    f"Baseline {b_name!r} rejected its hyperparameters {params}. "
                    f"Accepted: {list(signature.parameters)}. Original error: {exc}"
                ) from exc

            baseline_preds[b_name] = preds

            # 1. Accuracy metrics
            baseline_metrics[b_name] = MetricsCalculator(
                train_df, eval_df, preds, risk_period=self.tau_days
            ).all_metrics()
        return baseline_preds, baseline_metrics
    
    def get_last_predictions(self) -> dict[str, Any] | None:
        """Returns the actuals and model predictions from the most recently executed window."""
        if self.last_window_result is None:
            return None
        return {
            "eval_actuals": self.last_window_result.eval_actuals,
            "predictions": self.last_window_result.model_predictions,
            "window_id": self.last_window_result.window_id,
        }
