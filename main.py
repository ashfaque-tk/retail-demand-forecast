from fastapi import FastAPI, HTTPException,status
import os 
from pydantic import BaseModel
from datetime import date
import psycopg2
from psycopg2.extras import RealDictCursor


app = FastAPI(title='Retail Forecast API')

def get_db_connection():
    return psycopg2.connect(os.environ["DATABASE_URL"], cursor_factory=RealDictCursor)


class ForecastRequest(BaseModel):
    item_id: str
    horizon: int = 28

class DayForecast(BaseModel):
    date: str
    sales_pred: float
    q10: float
    q90: float
    q95: float

class InventoryPolicy(BaseModel):
    safety_stock: float 
    reorder_point: float 
    holding_cost: float
    order_up_to : float

class ForecastResponse(BaseModel):
    item_id: str
    cat_id : str
    dept_id: str
    forecast: list[DayForecast]
    inventory_policy:InventoryPolicy


@app.post("/forecast",response_model=ForecastResponse)
def get_forecast(store_id:str,item_id:str):
    '''Fetches pre-computed batch forecast and inventory policies directly from
    PostgreSQL'''
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Query precomputed daily forecasts
                cur.execute(
                    ''' SELECT target_date, sales_pred,q10,q90, prediction_made_date
                    FROM predictions 
                    WHERE store_id =%s AND item_id=%s
                        AND prediction_made_date = ( SELECT MAX(prediction_made_date)
                        FROM predictions WHERE store_id=%s AND item_id=%s)  
                        ORDER BY target_date ASC;
                        ''',  (store_id,item_id,store_id,item_id))
                forecast_rows = cur.fetchall() 

                # Query precomputed policy metrics
                cur.execute(
                    """
                    SELECT safety_stock, reorder_point, order_up_to,holding_cost_risk_period
                    FROM inventory_policies
                    WHERE store_id = %s AND item_id = %s
                    ORDER BY created_at DESC LIMIT 1;
                    """,
                    (store_id, item_id),
                )
                policy_row = cur.fetchone()

                if not forecast_rows or policy_row:
                    raise HTTPException(status_code=404,detail=f'no forecast or policy found for item_id {item_id}\
                                        at store {store_id}')
            latest_pred_date = str(forecast_rows[0]["prediction_made_date"])

            data = {"item_id": item_id,
                    "store_id": store_id,
                    "prediction_made_date": latest_pred_date,
                    "forecast": [
                        {
                            "target_date": str(r["target_date"]),
                            "sales_pred": float(r["sales_pred"]),
                            "q10": float(r["q10"]),
                            "q90": float(r["q90"]),
                        }
                        for r in forecast_rows
                    ],
                    "inventory_policy": {
                        "safety_stock": float(policy_row["safety_stock"]) if policy_row else 0.0,
                        "reorder_point": float(policy_row["reorder_point"]) if policy_row else 0.0,
                        "holding_cost": float(policy_row["holding_cost_risk_period"]) if policy_row else 0.0,
                        "order_up_to" : float(policy_row['order_up_to'])
                    }}

            return {'status':'success','data':data}

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


