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


CREATE TABLE IF NOT EXISTS inventory (
    id BIGSERIAL PRIMARY KEY,

    date DATE NOT NULL,
    item_id VARCHAR(50) NOT NULL,
    store_id VARCHAR(10) NOT NULL,

    model_name VARCHAR(50) NOT NULL,
    forecast_type VARCHAR(30) NOT NULL,

    -- Inventory state
    on_hand NUMERIC(12,4) NOT NULL DEFAULT 0.00,
    arriving_qty NUMERIC(12,4) NOT NULL DEFAULT 0.00,

    -- Inventory decision
    order_qty NUMERIC(12,4) NOT NULL DEFAULT 0.00,
    order_up_to NUMERIC(12,4) NOT NULL DEFAULT 0.00,
    safety_stock NUMERIC(12,4) NOT NULL DEFAULT 0.00,
    -- Realized outcomes
    holding_cost NUMERIC(12,4) NOT NULL DEFAULT 0.00,

    -- Policy inputs
    lead_time_days INT NOT NULL,
    review_period_days INT NOT NULL,

 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT uq_inventory_record
        UNIQUE (item_id, store_id, date, model)
);