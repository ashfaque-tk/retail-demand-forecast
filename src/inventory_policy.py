''' Store replenishment policy assuming a order upto level periodic review setup
    V1:  Review Period: 7 days (for all), lead time = 4 days,
        Holding cost rate : 20% unit cost annually
        Stockout Cost     : not applied in v1, we use fill rate
    
    safety_stock_base : assuming normal distribution, classical formula: Ss = z*sigma_d * np.sqrt(L), z:
    safety_stock_v1   : Ss = sigma_error*np.sqrt(L+R)*
    '''

import pandas as pd 
import numpy as np 
import logging 


logger = logging.getLogger(__name__)


class InventoryPolicy:

    def __init__(
        self,
        inventory: pd.Series,
        forecasts: pd.DataFrame,
        lead_time: int = 4,
        review_period: int = 7,
        daily_unit_holding_cost: float = 0.2,
        stockout_cost: float = 1.0,
        
    ):
        self.forecasts_df = forecasts.copy()
        self.forecasts_df['date'] = pd.to_datetime(self.forecasts_df['date'])
        self.forecasts_df['item_id'] = self.forecasts_df['item_id'].astype(str)

        self.lead_time = lead_time
        self.review_period = review_period
        self.risk_period = lead_time + review_period

        self.daily_unit_holding_cost = daily_unit_holding_cost 
        self.stockout_penalty = stockout_cost
        self.target_service_level = 1 / (1 + (daily_unit_holding_cost / stockout_cost))

        if inventory is None:
            raise ValueError("Inventory not initialized.")

        # if not oos_error_list:
        #     raise ValueError("Out Of Sample Error not initialized.")

        self.oos_errors = [] # initialize a oos_error list

        # Align index type to string
        self.on_hand = inventory.copy()
        self.on_hand.index = self.on_hand.index.astype(str)
        self.transit_pipeline: list[tuple[pd.Timestamp, pd.Series]] = []  
        self.simulation_logs = []   

    def _safety_stock(self) -> pd.Series:
        z_score = 0.97 # for 83.33% service level
        latest_error = self.oos_errors[-1]
        
        if isinstance(latest_error, pd.DataFrame):
            latest_error = latest_error.set_index("item_id")["rmse_tau"]
            
        latest_error.index = latest_error.index.astype(str)
        return z_score * latest_error * np.sqrt(self.risk_period)

    def update_oos_errors(self, forecasts_prior_review_date: pd.DataFrame):
        rolling_error_df = compute_rolling_tau_error(forecasts=forecasts_prior_review_date, tau=self.risk_period)
        error_stats = calculate_error_statistics(rolling_error_df)
        self.oos_errors.append(error_stats)

    def generate_order(
        self,
        forecasted_risk_period: pd.DataFrame,
        on_hand_current: pd.Series,
        on_order: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        
        forecasted_risk_period['item_id'] = forecasted_risk_period['item_id'].astype(str)
        risk_demand_item = forecasted_risk_period.groupby("item_id", observed=True)["sales_pred"].sum()

        safety_stock = self._safety_stock()
        order_up_to = risk_demand_item.add(safety_stock, fill_value=0.0)

        inventory_position = (
            on_hand_current.reindex(order_up_to.index, fill_value=0.0)
            + on_order.reindex(order_up_to.index, fill_value=0.0)
        )

        order_qty = (order_up_to - inventory_position).clip(lower=0.0)
        return order_qty, order_up_to

    def daily_update(self, on_hand: pd.Series, arriving_qty: pd.Series, actual_sales_day: pd.Series):
        arriving_qty = arriving_qty.reindex(on_hand.index, fill_value=0.0)
        actual_sales_day = actual_sales_day.reindex(on_hand.index, fill_value=0.0)
        
        on_hand = on_hand.add(arriving_qty, fill_value=0.0) 
        
        sales_and_stock = pd.concat([on_hand, actual_sales_day], axis=1).fillna(0.0)
        sales_and_stock.columns = ['on_hand', 'actual_sales']
        fulfilled_sales = sales_and_stock.min(axis=1)

        lost_sales = actual_sales_day.sub(fulfilled_sales, fill_value=0.0).clip(lower=0.0)
        on_hand = on_hand.sub(fulfilled_sales, fill_value=0.0).clip(lower=0.0)

        holding_cost = on_hand * self.daily_unit_holding_cost
        stockout_cost = lost_sales * self.stockout_penalty

        return on_hand, lost_sales, holding_cost, stockout_cost

    def daily_simulation(
            self, 
            actual_sales: pd.DataFrame, 
            forecasted_demand: pd.DataFrame,
            model_name:str,
            forecast_type:str,
            store_id:str='CA_1') -> pd.DataFrame:
        actual_sales = actual_sales.copy()
        forecasted_demand = forecasted_demand.copy()
        
        actual_sales['date'] = pd.to_datetime(actual_sales['date'])
        forecasted_demand['date'] = pd.to_datetime(forecasted_demand['date'])
        actual_sales['item_id'] = actual_sales['item_id'].astype(str)
        forecasted_demand['item_id'] = forecasted_demand['item_id'].astype(str)

        dates = sorted(forecasted_demand['date'].unique())
        new_logs = []

        for idx, day in enumerate(dates):
          
            sales_today_df = actual_sales[actual_sales['date'] == day]
    
            order_qty = pd.Series(0.0, index=self.on_hand.index)
            order_up_to = pd.Series(0.0, index=self.on_hand.index)

            # --- REVIEW DAY LOGIC ---
            if idx % self.review_period == 0:
                forecasts_prior_review = self.forecasts_df[self.forecasts_df['date'] <= day]
                
                # Update OOS errors using history
                if not forecasts_prior_review.empty:
                    self.update_oos_errors(forecasts_prior_review)

                # Fix: Look ahead dynamically from current day through (day + risk_period - 1)
                risk_period_end = day + pd.Timedelta(days=self.risk_period - 1)
                risk_period_forecast = forecasted_demand[
                    forecasted_demand['date'].between(day, risk_period_end)
                ].copy()
                
                on_order = pd.Series(0.0, index=self.on_hand.index)
                for arr_date, qty_series in self.transit_pipeline:
                    if arr_date >= day:
                        on_order = on_order.add(qty_series, fill_value=0.0)

                order_qty, order_up_to = self.generate_order(
                    forecasted_risk_period=risk_period_forecast,
                    on_hand_current=self.on_hand,
                    on_order=on_order
                )

                arrival_date = day + pd.Timedelta(days=self.lead_time)
                self.transit_pipeline.append((arrival_date, order_qty))

            # --- DAILY PHYSICAL UPDATE ---
            arriving_qty = pd.Series(0.0, index=self.on_hand.index)
            for arr_date, qty_series in self.transit_pipeline:
                if arr_date == day:
                    arriving_qty = arriving_qty.add(qty_series, fill_value=0.0)

            sales_on_day = sales_today_df.groupby('item_id', observed=True)['sales'].sum()
       
            sales_on_day.index = sales_on_day.index.astype(str)
            sales_on_day = sales_on_day.reindex(self.on_hand.index, fill_value=0.0)
           
            self.on_hand, lost_sales, holding_cost, stockout_cost = self.daily_update(
                on_hand=self.on_hand,
                arriving_qty=arriving_qty,
                actual_sales_day=sales_on_day
            )

            daily_logs = pd.DataFrame({
                'date': day,
                'item_id': self.on_hand.index,
                'store_id' : store_id,
                'model_name' : model_name,
                'forecast_type': forecast_type,
                'on_hand': self.on_hand.values,
                'arriving_qty': arriving_qty.reindex(self.on_hand.index, fill_value=0.0).values,
                'actual_sales': sales_on_day.values,
                'safety_stock': self._safety_stock(),
                'lost_sales': lost_sales.values,
                'order_qty': order_qty.reindex(self.on_hand.index, fill_value=0.0).values,
                'order_up_to': order_up_to.reindex(self.on_hand.index, fill_value=0.0).values,
                'holding_cost': holding_cost.values,
                'stockout_cost': stockout_cost.values
            })  

            new_logs.append(daily_logs)
            self.simulation_logs.append(daily_logs)

        return pd.concat(new_logs, ignore_index=True)
                
def compute_rolling_tau_error(forecasts: pd.DataFrame, tau: int) -> pd.DataFrame:
    """Computes rolling tau-day cumulative actual vs forecast errors."""
    df = forecasts.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['item_id', 'date'])
    
    g = df.groupby('item_id', observed=True)
    df['_rolled_actual'] = g['sales'].transform(lambda x: x.rolling(tau, min_periods=1).sum())
    df['_rolled_forecast'] = g['sales_pred'].transform(lambda x: x.rolling(tau, min_periods=1).sum())

    df['cum_error'] = df['_rolled_actual'] - df['_rolled_forecast']
    return df.drop(columns=['_rolled_actual', '_rolled_forecast'])

def calculate_error_statistics(df_with_error: pd.DataFrame) -> pd.Series:
    """Computes per-item RMSE of cumulative TAU-day errors."""
    stats = (
        df_with_error.groupby("item_id", observed=True)["cum_error"]
        .agg(rmse_tau=lambda x: np.sqrt(np.mean(x.dropna() ** 2)))
        .reset_index()
    )
    # Ensure index is set to item_id cast as string
    stats['item_id'] = stats['item_id'].astype(str)
    return stats.set_index("item_id")["rmse_tau"]
