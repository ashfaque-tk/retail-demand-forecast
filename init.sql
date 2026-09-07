CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    item_id VARCHAR(50) NOT NULL,
    store_id VARCHAR(10) NOT NULL,
    target_date DATE NOT NULL,
    prediction_made_date DATE NOT NULL,
    horizon_days INTEGER NOT NULL,
    model_version VARCHAR(50) DEFAULT 'v1',
    sales_pred FLOAT NOT NULL,
    q10 FLOAT NOT NULL,
    q90 FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS actuals (
    item_id VARCHAR(50) NOT NULL,
    cat_id VARCHAR(50) NOT NULL,
    dept_id VARCHAR(50) NOT NULL,
    store_id VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    sales_actual FLOAT NOT NULL,
    PRIMARY KEY (item_id, store_id, date)
);

CREATE TABLE IF NOT EXISTS inventory_policies (
    id BIGSERIAL PRIMARY KEY,
    item_id VARCHAR(50) NOT NULL,
    store_id VARCHAR(10) NOT NULL,
    calculated_date DATE NOT NULL,
    policy_type VARCHAR(30) NOT NULL,          -- e.g., 'PARAMETRIC_RMSE', 'QUANTILE_LOSS'
    lead_time_days INT NOT NULL DEFAULT 4,
    review_period_days INT NOT NULL DEFAULT 7,
    holding_cost_per_unit NUMERIC(10, 2) DEFAULT 0.00,
    forecasted_risk_period NUMERIC(12, 4) NOT NULL, -- Summed demand forecast over (L + R)
    safety_stock NUMERIC(12, 4) NOT NULL,
    reorder_point NUMERIC(12, 4) NOT NULL DEFAULT 0.00,
    order_up_to NUMERIC(12, 4) NOT NULL,
    holding_cost_risk_period NUMERIC(12,4) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure only one policy entry per item/store/date/type combination
    CONSTRAINT uq_inventory_policy UNIQUE (item_id, store_id, calculated_date, policy_type)
);

-- Deployment history is deliberately separate from the legacy API tables above.
-- A deployment is immutable: forecasts, calibration results, source observations,
-- and policy decisions can always be traced back to the exact model run.
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
    cat_id VARCHAR(100),
    dept_id VARCHAR(100),
    observation_date DATE NOT NULL,
    sales_actual DOUBLE PRECISION,
    PRIMARY KEY (deployment_id, dataset_role, item_id, store_id, observation_date)
);

CREATE TABLE IF NOT EXISTS deployment_predictions (
    deployment_id VARCHAR(100) NOT NULL REFERENCES deployment_runs(deployment_id) ON DELETE CASCADE,
    prediction_stage VARCHAR(30) NOT NULL CHECK (prediction_stage IN ('calibration', 'production')),
    item_id VARCHAR(100) NOT NULL,
    store_id VARCHAR(50) NOT NULL,
    cat_id VARCHAR(100),
    dept_id VARCHAR(100),
    target_date DATE NOT NULL,
    sales_pred DOUBLE PRECISION NOT NULL,
    q10 DOUBLE PRECISION,
    q90 DOUBLE PRECISION,
    q95 DOUBLE PRECISION,
    PRIMARY KEY (deployment_id, prediction_stage, item_id, store_id, target_date)
);

CREATE TABLE IF NOT EXISTS deployment_metrics (
    deployment_id VARCHAR(100) NOT NULL REFERENCES deployment_runs(deployment_id) ON DELETE CASCADE,
    metric_name VARCHAR(50) NOT NULL,
    metric_value DOUBLE PRECISION,
    PRIMARY KEY (deployment_id, metric_name)
);

CREATE TABLE IF NOT EXISTS deployment_inventory_policies (
    deployment_id VARCHAR(100) NOT NULL REFERENCES deployment_runs(deployment_id) ON DELETE CASCADE,
    item_id VARCHAR(100) NOT NULL,
    store_id VARCHAR(50) NOT NULL,
    review_date DATE NOT NULL,
    protection_end_date DATE NOT NULL,
    policy_type VARCHAR(30) NOT NULL,
    lead_time_days INTEGER NOT NULL,
    review_period_days INTEGER NOT NULL,
    forecast_tau DOUBLE PRECISION NOT NULL,
    safety_stock DOUBLE PRECISION NOT NULL,
    order_up_to DOUBLE PRECISION NOT NULL,
    holding_cost DOUBLE PRECISION,
    PRIMARY KEY (deployment_id, item_id, store_id, review_date, policy_type)
);
