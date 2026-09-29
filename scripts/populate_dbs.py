import os
import sys
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

import time
import json
import subprocess


DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "forecast_db")
DB_USER = os.getenv("DB_USER", "forecast_user")
DB_PASS = os.getenv("DB_PASS", "forecast_pass")


def ensure_docker_postgres_running(container_name="your_postgres_container_name"):
    """Checks if Postgres is reachable. If not, automatically starts the Docker container."""
    try:
        # Try a quick test connection
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS, connect_timeout=2
        )
        conn.close()
        return
    except psycopg2.OperationalError:
        print("\n[INFO] Postgres is not reachable. Attempting to start Docker container automatically...")
        try:
            # Command to start your existing container
            subprocess.run(["docker", "start", container_name], check=True)
            print("[INFO] Waiting 5 seconds for PostgreSQL to start up...")
            time.sleep(5)  # Give Postgres a few seconds to boot up
        except Exception as e:
            print(f"[ERROR] Could not auto-start Docker container: {e}")
            print("Please ensure Docker Desktop is open.")

def get_db_connection():
    """Establishes connection to PostgreSQL. Throws actionable warning if Docker is off."""
    ensure_docker_postgres_running(container_name='retail-forecast-system-postgres-1')
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        return conn
    except psycopg2.OperationalError as e:
        print("\n" + "=" * 60)
        print("DATABASE CONNECTION FAILED!")
        print("-> Please start Docker Desktop and ensure your Postgres container is running.")
        print(f"-> Target URI: {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        print("=" * 60 + "\n")
        raise e


# ==========================================
# 1. POPULATE PREDICTIONS
# ==========================================
def populate_predictions(df: pd.DataFrame, store_id: str = "CA_1", model_version: str = "v1"):
    ensure_docker_postgres_running(container_name='retail-forecast-system-postgres-1')
    
    conn = get_db_connection()
    cursor = conn.cursor()

    df_to_insert = df.copy()

    # 1. Fill default metadata columns if missing
    if 'store_id' not in df_to_insert.columns:
        df_to_insert['store_id'] = store_id
    if 'date' in df_to_insert.columns:
        df_to_insert = df_to_insert.rename(columns={'date': 'target_date'})
    if 'prediction_made_date' not in df_to_insert.columns:
        df_to_insert['prediction_made_date'] = df_to_insert['target_date'].min() - pd.Timedelta(days=1)
    if 'horizon_days' not in df_to_insert.columns:
        df_to_insert['horizon_days'] = (df_to_insert['target_date'] - df_to_insert['prediction_made_date']).dt.days

    if 'model_name' not in df_to_insert.columns:
        df_to_insert['model_name'] = 'lgbm'

    # 2. Extract quantile columns (e.g. 'q10', 'q83', 'q90') dynamically
    quantile_cols = [col for col in df_to_insert.columns if col.startswith('q') and col[1:].isdigit()]

    # 3. Pack quantiles into a JSON string per row
    def pack_quantiles(row):
        q_dict = {col: float(row[col]) for col in quantile_cols if pd.notnull(row[col])}
        return json.dumps(q_dict)

    df_to_insert['quantiles'] = df_to_insert.apply(pack_quantiles, axis=1)

    # 4. Prepare target columns and tuples
    target_cols = [
        "item_id", "store_id", "target_date", "prediction_made_date",
        "horizon_days", "model_name", "sales_pred", "quantiles"
    ]
    
    df_to_insert = df_to_insert[target_cols]

    # Convert dates/timestamps to string representation for psycopg2
    df_to_insert['target_date'] = df_to_insert['target_date'].dt.strftime('%Y-%m-%d')
    df_to_insert['prediction_made_date'] = df_to_insert['prediction_made_date'].dt.strftime('%Y-%m-%d')

    records = [tuple(x) for x in df_to_insert.to_numpy()]
    cols_str = ", ".join(target_cols)

    # 5. Execute bulk insert with ::jsonb casting
    query = f"""
        INSERT INTO predictions ({cols_str})
        VALUES %s;
    """

    try:
        print(f"Uploading {len(records)} prediction rows...")
        # Cast the last column (quantiles) to JSONB dynamically in SQL template
        template = "(%s, %s, %s, %s, %s, %s, %s, %s::jsonb)"
        execute_values(cursor, query, records, template=template)
        conn.commit()
        print("Predictions successfully uploaded.")
    except Exception as e:
        conn.rollback()
        print(f"Failed to upload predictions: {e}")
        raise e
    finally:
        cursor.close()
        conn.close()


# ==========================================
# 2. POPULATE ACTUALS
# ==========================================
def populate_actuals(df: pd.DataFrame, store_id: str = 'CA_1'):
    """Inserts ground truth historical sales into actuals table."""
    ensure_docker_postgres_running('retail-forecast-system-postgres-1')
    conn = get_db_connection()
    cursor = conn.cursor()

    df_to_insert = df.copy()
    
    # 1. Standardize column names (map real_sales -> sales_actual)
    if 'real_sales' in df_to_insert.columns:
        df_to_insert = df_to_insert.rename(columns={'real_sales': 'sales_actual'})
    if 'store_id' not in df_to_insert.columns:
        df_to_insert['store_id'] = store_id

    # 2. Format date to string
    df_to_insert['date'] = pd.to_datetime(df_to_insert['date']).dt.strftime('%Y-%m-%d')

    # 3. Align target columns with PostgreSQL schema
    target_cols = ["item_id", "cat_id", "dept_id", "store_id", "date", "sales_actual"]
    df_to_insert = df_to_insert[target_cols]

    # Convert to standard Python tuples
    records = [tuple(row) for row in df_to_insert.itertuples(index=False)]
    cols_str = ", ".join(target_cols)

    query = f"""
        INSERT INTO actuals ({cols_str})
        VALUES %s
        ON CONFLICT (item_id, store_id, date)
        DO UPDATE SET sales_actual = EXCLUDED.sales_actual;
    """

    try:
        print(f"Uploading {len(records)} actuals rows...")
        execute_values(cursor, query, records)
        conn.commit()
        print("Actuals successfully uploaded.")
    except Exception as e:
        conn.rollback()
        print(f"Failed to upload actuals: {e}")
        raise e
    finally:
        cursor.close()
        conn.close()


# ==========================================
# 3. POPULATE INVENTORY
# ==========================================
def populate_inventory(
    df: pd.DataFrame,
    store_id: str = "CA_1",
    model_name: str = "lgbm",
    forecast_type: str = "direct",
    lead_time_days: int = 4,
    review_period_days: int = 7,
):
    """Inserts inventory decisions and simulation outcomes into inventory table."""
    conn = get_db_connection()
    cursor = conn.cursor()

    df_to_insert = df.copy()

    # Fill store_id and model_name only if they aren't already in the DataFrame
    if "store_id" not in df_to_insert.columns:
        df_to_insert["store_id"] = store_id

    if "model_name" not in df_to_insert.columns:
        df_to_insert["model_name"] = model_name

    # Assign forecast type and static policy parameters
    df_to_insert["forecast_type"] = forecast_type
    df_to_insert["lead_time_days"] = lead_time_days
    df_to_insert["review_period_days"] = review_period_days

    if "safety_stock" not in df_to_insert.columns:
        df_to_insert["safety_stock"] = 0.00

    target_cols = [
        "date",
        "item_id",
        "store_id",
        "model_name",
        "forecast_type",
        "on_hand",
        "arriving_qty",
        "order_qty",
        "order_up_to",
        "safety_stock",
        "holding_cost",
        "lead_time_days",
        "review_period_days",
    ]

    # Reorder and filter DataFrame columns to match target_cols exactly
    df_to_insert = df_to_insert[target_cols]
    records = [tuple(row) for row in df_to_insert.to_numpy()]
    cols_str = ", ".join(target_cols)

    query = f"""
        INSERT INTO inventory ({cols_str})
        VALUES %s
        ON CONFLICT (item_id, store_id, date, model_name, forecast_type)
        DO UPDATE SET
            on_hand = EXCLUDED.on_hand,
            arriving_qty = EXCLUDED.arriving_qty,
            order_qty = EXCLUDED.order_qty,
            order_up_to = EXCLUDED.order_up_to,
            safety_stock = EXCLUDED.safety_stock,
            holding_cost = EXCLUDED.holding_cost,
            lead_time_days = EXCLUDED.lead_time_days,
            review_period_days = EXCLUDED.review_period_days;
    """

    try:
        print(f"Uploading {len(records)} inventory rows...")
        execute_values(cursor, query, records)
        conn.commit()
        print(" Inventory successfully uploaded.")
    except Exception as e:
        conn.rollback()
        print(f" Failed to upload inventory: {e}")
        raise e
    finally:
        cursor.close()
        conn.close()

# Quick connection sanity check when running script directly
if __name__ == "__main__":
    try:
        conn = get_db_connection()
        print(" Database connection OK!")
        conn.close()
    except Exception:
        sys.exit(1)