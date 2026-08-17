from fastapi import FastAPI, HTTPException
import os 
from pydantic import BaseModel
from datetime import date
import psycopg2
import joblib
import pandas as pd
from src.pipeline import get_known_future_features, recursive_forecast

app = FastAPI(title='Retail Forecast API')

models = {
    'point': joblib.load('models/lgb_model_ca1_deploy.pkl'),
    'q10': joblib.load('models/lgb_q10_ca1_deploy.pkl'),
    'q90': joblib.load('models/lgb_q90_ca1_deploy.pkl'),
}
cat_categories = joblib.load('models/cat_categories_ca1.pkl')
feature_cols = joblib.load('models/feature_cols_ca1.pkl')
recent_history = pd.read_parquet('models/recent_history_ca1.parquet')
calendar_df = pd.read_csv('data/raw/calendar.csv', parse_dates=['date'])
price_df = pd.read_csv('data/raw/sell_prices.csv')

db_conn = psycopg2.connect(os.environ['DATABASE_URL'])
db_conn.autocommit = True



def log_predictions(item_id,store_id,results,prediction_made_date, model_version='v1'):
    with db_conn.cursor() as cur:
        for h,r in enumerate(results,start=1):
            cur.execute(""" INSERT INTO predictions (item_id,store_id,target_date,prediction_made_date,horizon_days,model_version,sales_pred,q10,q90) 
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (item_id,store_id,r['date'].date(),prediction_made_date,h, model_version, float(r['sales_pred']),float(r['q10']),float(r['q90'])))


class ForecastRequest(BaseModel):
    item_id: str
    horizon: int = 28

class DayForecast(BaseModel):
    date: str
    sales_pred: float
    q10: float
    q90: float

class ForecastResponse(BaseModel):
    item_id: str
    forecast: list[DayForecast]

@app.post("/forecast", response_model=ForecastResponse)
def forecast_item(req: ForecastRequest):
    item_history = recent_history[recent_history['item_id'] == req.item_id]
    if item_history.empty:
        raise HTTPException(status_code=404, detail=f"No history found for item_id {req.item_id}")

    last_date = item_history['date'].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=req.horizon)

    item_meta = {'dept_id': item_history['dept_id'].iloc[0], 'cat_id': item_history['cat_id'].iloc[0]} \
        if 'dept_id' in item_history.columns else {}
    item_price = price_df[(price_df['item_id'] == req.item_id) & (price_df['store_id'] == 'CA_1')]
    future_static = get_known_future_features(req.item_id, future_dates, calendar_df, item_price, item_meta)

    results = recursive_forecast(models, item_history, future_static, feature_cols, cat_categories)

    log_predictions(req.item_id, 'CA_1', results, date.today())

    return {'item_id': req.item_id, 'forecast': [
        {'date': str(r['date'].date()), 'sales_pred': r['sales_pred'], 'q10': r['q10'], 'q90': r['q90']}
        for r in results
    ]}