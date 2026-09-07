"""Transactional PostgreSQL persistence for immutable deployment artifacts."""
from __future__ import annotations

from datetime import date, datetime
import json
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values, Json

from src.backtest_engine import DeploymentResult


def _value(value: Any) -> Any:
    """Convert pandas/numpy values into PostgreSQL-friendly scalars."""
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.date()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _records(df: pd.DataFrame, columns: list[str]) -> list[tuple[Any, ...]]:
    return [tuple(_value(row[col]) for col in columns) for _, row in df[columns].iterrows()]


DEPLOYMENT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS deployment_runs (
    deployment_id VARCHAR(100) PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_name VARCHAR(100) NOT NULL,
    training_window_days INTEGER NOT NULL,
    calibration_start DATE NOT NULL,
    calibration_end DATE NOT NULL,
    forecast_start DATE NOT NULL,
    forecast_end DATE NOT NULL,
    config JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE TABLE IF NOT EXISTS deployment_observations (
    deployment_id VARCHAR(100) NOT NULL REFERENCES deployment_runs(deployment_id) ON DELETE CASCADE,
    dataset_role VARCHAR(30) NOT NULL CHECK (dataset_role IN ('final_train', 'forecast_input')),
    item_id VARCHAR(100) NOT NULL,
    store_id VARCHAR(50) NOT NULL,
    cat_id VARCHAR(100), dept_id VARCHAR(100), observation_date DATE NOT NULL,
    sales_actual DOUBLE PRECISION,
    PRIMARY KEY (deployment_id, dataset_role, item_id, store_id, observation_date)
);
CREATE TABLE IF NOT EXISTS deployment_predictions (
    deployment_id VARCHAR(100) NOT NULL REFERENCES deployment_runs(deployment_id) ON DELETE CASCADE,
    prediction_stage VARCHAR(30) NOT NULL CHECK (prediction_stage IN ('calibration', 'production')),
    item_id VARCHAR(100) NOT NULL, store_id VARCHAR(50) NOT NULL,
    cat_id VARCHAR(100), dept_id VARCHAR(100), target_date DATE NOT NULL,
    sales_pred DOUBLE PRECISION NOT NULL, q10 DOUBLE PRECISION, q90 DOUBLE PRECISION, q95 DOUBLE PRECISION,
    PRIMARY KEY (deployment_id, prediction_stage, item_id, store_id, target_date)
);
CREATE TABLE IF NOT EXISTS deployment_metrics (
    deployment_id VARCHAR(100) NOT NULL REFERENCES deployment_runs(deployment_id) ON DELETE CASCADE,
    metric_name VARCHAR(50) NOT NULL, metric_value DOUBLE PRECISION,
    PRIMARY KEY (deployment_id, metric_name)
);
CREATE TABLE IF NOT EXISTS deployment_inventory_policies (
    deployment_id VARCHAR(100) NOT NULL REFERENCES deployment_runs(deployment_id) ON DELETE CASCADE,
    item_id VARCHAR(100) NOT NULL, store_id VARCHAR(50) NOT NULL,
    review_date DATE NOT NULL, protection_end_date DATE NOT NULL, policy_type VARCHAR(30) NOT NULL,
    lead_time_days INTEGER NOT NULL, review_period_days INTEGER NOT NULL,
    forecast_tau DOUBLE PRECISION NOT NULL, safety_stock DOUBLE PRECISION NOT NULL,
    order_up_to DOUBLE PRECISION NOT NULL, holding_cost DOUBLE PRECISION,
    PRIMARY KEY (deployment_id, item_id, store_id, review_date, policy_type)
);
"""


def persist_deployment(
    database_url: str,
    deployment_id: str,
    deployment: DeploymentResult,
    final_train: pd.DataFrame,
    forecast_input: pd.DataFrame,
    model_name: str,
    training_window_days: int,
    lead_time_days: int,
    review_period_days: int,
    config: dict[str, Any],
) -> None:
    """Write one deployment atomically; an existing id is rejected, never overwritten."""
    required_observation_columns = {"item_id", "store_id", "cat_id", "dept_id", "date", "sales"}
    for name, frame in (("final_train", final_train), ("forecast_input", forecast_input)):
        missing = required_observation_columns - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing database columns: {sorted(missing)}")

    store_lookup = (
        pd.concat([final_train, forecast_input], ignore_index=True)
        .sort_values("date")
        .drop_duplicates("item_id", keep="last")
        [["item_id", "store_id"]]
    )

    def observations(frame: pd.DataFrame, role: str) -> list[tuple[Any, ...]]:
        out = frame[["item_id", "store_id", "cat_id", "dept_id", "date", "sales"]].copy()
        out.insert(0, "deployment_id", deployment_id)
        out.insert(1, "dataset_role", role)
        out.columns = [
            "deployment_id", "dataset_role", "item_id", "store_id", "cat_id", "dept_id",
            "observation_date", "sales_actual",
        ]
        return _records(out, list(out.columns))

    def predictions(frame: pd.DataFrame, stage: str) -> list[tuple[Any, ...]]:
        out = frame.merge(store_lookup, on="item_id", how="left", validate="many_to_one")
        if out["store_id"].isna().any():
            raise ValueError("Could not resolve store_id for every deployment prediction.")
        out.insert(0, "deployment_id", deployment_id)
        out.insert(1, "prediction_stage", stage)
        for quantile in ("q10", "q90", "q95"):
            if quantile not in out:
                out[quantile] = None
        out = out[[
            "deployment_id", "prediction_stage", "item_id", "store_id", "cat_id", "dept_id",
            "date", "sales_pred", "q10", "q90", "q95",
        ]]
        out.columns = [
            "deployment_id", "prediction_stage", "item_id", "store_id", "cat_id", "dept_id",
            "target_date", "sales_pred", "q10", "q90", "q95",
        ]
        return _records(out, list(out.columns))

    # Costs are calculated from the policy and retain all policy fields, including
    # the review/protection dates.  Persist this joined view so each policy type
    # carries its corresponding holding-cost estimate.
    policy = deployment.inventory_costs.merge(store_lookup, on="item_id", how="left", validate="many_to_one")
    policy_rows: list[tuple[Any, ...]] = []
    for suffix, policy_type in (("rmse", "PARAMETRIC_RMSE"), ("mae", "PARAMETRIC_MAE"), ("classical", "CLASSICAL_STD")):
        subset = pd.DataFrame({
            "deployment_id": deployment_id,
            "item_id": policy["item_id"],
            "store_id": policy["store_id"],
            "review_date": policy["review_date"],
            "protection_end_date": policy["protection_end_date"],
            "policy_type": policy_type,
            "lead_time_days": lead_time_days,
            "review_period_days": review_period_days,
            "forecast_tau": policy["forecast_tau"],
            "safety_stock": policy[f"safety_stock_{suffix}"],
            "order_up_to": policy[f"order_up_to_{suffix}"],
            "holding_cost": policy[f"monthly_holding_cost_{suffix}"],
        })
        policy_rows.extend(_records(subset, list(subset.columns)))

    with psycopg2.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(DEPLOYMENT_SCHEMA_SQL)
            cur.execute(
                """INSERT INTO deployment_runs
                (deployment_id, model_name, training_window_days, calibration_start, calibration_end,
                 forecast_start, forecast_end, config)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    deployment_id, model_name, training_window_days, deployment.calibration_start.date(),
                    deployment.calibration_end.date(), deployment.forecast_start.date(),
                    deployment.forecast_end.date(), Json(config, dumps=json.dumps),
                ),
            )
            execute_values(cur, """INSERT INTO deployment_observations
                (deployment_id, dataset_role, item_id, store_id, cat_id, dept_id, observation_date, sales_actual)
                VALUES %s""", observations(final_train, "final_train") + observations(forecast_input, "forecast_input"))
            execute_values(cur, """INSERT INTO deployment_predictions
                (deployment_id, prediction_stage, item_id, store_id, cat_id, dept_id, target_date, sales_pred, q10, q90, q95)
                VALUES %s""", predictions(deployment.calibration_predictions, "calibration") + predictions(deployment.forecasts, "production"))
            execute_values(cur, "INSERT INTO deployment_metrics (deployment_id, metric_name, metric_value) VALUES %s",
                           [(deployment_id, name, _value(value)) for name, value in deployment.calibration_metrics.items()])
            execute_values(cur, """INSERT INTO deployment_inventory_policies
                (deployment_id, item_id, store_id, review_date, protection_end_date, policy_type,
                 lead_time_days, review_period_days, forecast_tau, safety_stock, order_up_to, holding_cost)
                VALUES %s""", policy_rows)
