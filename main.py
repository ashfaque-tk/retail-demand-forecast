from fastapi import FastAPI, HTTPException, status
import os
from pydantic import BaseModel, Field
import psycopg2
from psycopg2.extras import RealDictCursor
import json

app = FastAPI(title='Retail Forecast API')

def get_db_connection():
    db_url = os.getenv("DATABASE_URL", "postgresql://forecast_user:forecast_pass@localhost:5432/forecast_db")
    return psycopg2.connect(db_url, cursor_factory=RealDictCursor)


class DailyScheduleItem(BaseModel):
    target_date: str
    sales_pred: float
    quantiles: dict[str, float] = Field(default_factory=dict)

class PolicyConfig(BaseModel):
    safety_stock: float
    holding_cost: float
    lead_time_days: int
    review_period_days: int

class CurrentInventory(BaseModel):
    on_hand: float
    arriving_qty: float
    order_qty: float
    order_up_to: float

class ForecastResponse(BaseModel):
    item_id: str
    store_id: str
    prediction_made_date: str
    policy_config: PolicyConfig
    current_inventory: CurrentInventory
    daily_schedule: list[DailyScheduleItem]


@app.get("/forecast", response_model=ForecastResponse)
def get_forecast(store_id: str = "CA_1", item_id: str = "FOODS_1_002"):
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Deduplicate rows using DISTINCT ON (target_date)
                cur.execute(
                    ''' 
                    SELECT DISTINCT ON (target_date) 
                           target_date, sales_pred, quantiles, prediction_made_date
                    FROM predictions 
                    WHERE store_id = %s AND item_id = %s
                      AND prediction_made_date = (
                          SELECT MAX(prediction_made_date)
                          FROM predictions 
                          WHERE store_id = %s AND item_id = %s
                      )  
                    ORDER BY target_date ASC;
                    ''',  
                    (store_id, item_id, store_id, item_id)
                )
                forecast_rows = cur.fetchall() 

                cur.execute(
                    """
                    SELECT on_hand, arriving_qty, order_qty, order_up_to, 
                           safety_stock, holding_cost, lead_time_days, review_period_days
                    FROM inventory
                    WHERE store_id = %s AND item_id = %s
                    ORDER BY created_at DESC LIMIT 1;
                    """,
                    (store_id, item_id),
                )
                policy_row = cur.fetchone() or {}

                if not forecast_rows:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No forecast found for item_id '{item_id}' at store '{store_id}'"
                    )

            latest_pred_date = str(forecast_rows[0]["prediction_made_date"])

            daily_schedule = []
            for r in forecast_rows:
                q_payload = r.get("quantiles") or {}
                if isinstance(q_payload, str):
                    q_payload = json.loads(q_payload)

                daily_schedule.append({
                    "target_date": str(r["target_date"]),
                    "sales_pred": float(r["sales_pred"]),
                    "quantiles": q_payload
                })

            return {
                "item_id": item_id,
                "store_id": store_id,
                "prediction_made_date": latest_pred_date,
                "policy_config": {
                    "safety_stock": float(policy_row.get("safety_stock", 0.0)),
                    "holding_cost": float(policy_row.get("holding_cost", 0.0)),
                    "lead_time_days": int(policy_row.get("lead_time_days", 0)),
                    "review_period_days": int(policy_row.get("review_period_days", 0)),
                },
                "current_inventory": {
                    "on_hand": float(policy_row.get("on_hand", 0.0)),
                    "arriving_qty": float(policy_row.get("arriving_qty", 0.0)),
                    "order_qty": float(policy_row.get("order_qty", 0.0)),
                    "order_up_to": float(policy_row.get("order_up_to", 0.0)),
                },
                "daily_schedule": daily_schedule
            }

    except psycopg2.errors.UndefinedTable as e:
        # Gracefully handle missing database table
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inventory database is not properly initialized (table 'inventory_policies' missing)."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected database error occurred: {str(e)}"
        )


