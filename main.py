from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
calendar_df = pd.read_csv('data/m5-forecasting-accuracy/calendar.csv', parse_dates=['date'])
price_df = pd.read_csv('data/m5-forecasting-accuracy/sell_prices.csv')


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
    return {'item_id': req.item_id, 'forecast': [
        {'date': str(r['date'].date()), 'sales_pred': r['sales_pred'], 'q10': r['q10'], 'q90': r['q90']}
        for r in results
    ]}