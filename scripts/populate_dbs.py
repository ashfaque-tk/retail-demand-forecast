import os
import sys
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

import time
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
    """Inserts forecasted sales and quantiles into predictions table."""
    conn = get_db_connection()
    cursor = conn.cursor()

    df_to_insert = df.copy()
    if 'store_id' not in df_to_insert.columns:
        df_to_insert['store_id'] = store_id
    if 'model_version' not in df_to_insert.columns:
        df_to_insert['model_version'] = model_version

    target_cols = [
        "item_id", "store_id", "target_date", "prediction_made_date",
        "horizon_days", "model_version", "sales_pred", "q10", "q90"
    ]
    
    df_to_insert = df_to_insert[target_cols]
    records = [tuple(row) for row in df_to_insert.to_numpy()]
    cols_str = ", ".join(target_cols)

    query = f"""
        INSERT INTO predictions ({cols_str})
        VALUES %s;
    """

    try:
        print(f"Uploading {len(records)} prediction rows...")
        execute_values(cursor, query, records)
        conn.commit()
        print(" Predictions successfully uploaded.")
    except Exception as e:
        conn.rollback()
        print(f" Failed to upload predictions: {e}")
        raise e
    finally:
        cursor.close()
        conn.close()


# ==========================================
# 2. POPULATE ACTUALS
# ==========================================
def populate_actuals(df: pd.DataFrame):
    """Inserts ground truth historical sales into actuals table."""
    ensure_docker_postgres_running('retail-forecast-system-postgres-1')
    conn = get_db_connection()
    cursor = conn.cursor()

    target_cols = ["item_id", "cat_id", "dept_id", "store_id", "date", "sales_actual"]
    df_to_insert = df[target_cols].copy()
    records = [tuple(row) for row in df_to_insert.to_numpy()]
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
        print(" Actuals successfully uploaded.")
    except Exception as e:
        conn.rollback()
        print(f" Failed to upload actuals: {e}")
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