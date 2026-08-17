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
    store_id VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    sales_actual FLOAT NOT NULL,
    PRIMARY KEY (item_id, store_id, date)
);