''' Store replenishment policy assuming a order upto level periodic review setup
    V1:  Review Period: 7 days (for all), lead time = 4 days,
        Holding cost rate : 20% unit cost annually
        Stockout Cost     : not applied in v1, we use fill rate
    
    safety_stock_base : assuming normal distribution, classical formula: Ss = z*\sigma_d * np.sqrt(L), z:
    safety_stock_v1   : Ss = \sigma_error*np.sqrt(L+R)*
    '''

import pandas as pd 
import numpy as np 
import logging 


logger = logging.getLogger(__name__)

# ASSUMPTIONS — 


 
#Safety Stock Formulas
## building the benchmark stocking asuuming a normal distribution


class InventoryPolicy():

    def __init__(self,inventory:pd.DataFrame,forecasts:pd.DataFrame,
                 lead_time:int=14,review_period:int=7,
                 daily_unit_holding_cost:float=0.2,stockout_cost:float=1,
                 oos_error_list:list[pd.Series]| None=None):
        
        '''inventory: avg of last month for starting
         forecasts: historical vs current month actuals vs preds'''
        
        self.forecasts_df = forecasts

        self.lead_time = lead_time
        self.review_period = review_period
    
        self.risk_period = lead_time+review_period

        self.daily_unit_holding_cost=daily_unit_holding_cost 
        self.stockout_penalty = stockout_cost

        # service level with given costs== 83.33% and z-score = 0.97
        self.target_service_level = 1/(1+0.2)

        #inventory data with on_hand and in_transit, for on_hand date re
        # get populated through continous day simulations,
        if inventory is None:
            logging.error(f"Inventory not initialized. Initilize the inventory to continue. ")

        # intialize the oos error list: tracking the error deviation in the previous risk_period  
        if oos_error_list is None:
            logging.error(f"Out Of Sample Error not initialized. ")

        self.oos_errors = oos_error_list # need to be updated every review period 

        # we initlize on_hand with inventory and on_arrival as empty list
        self.on_hand = inventory.copy()
        self.transit_pipeline: list[tuple[pd.Timestamp, pd.Series]] = []  

        self.simulation_logs = []   

    def get_current_state(self) -> dict:
        '''Returns a snapshot of current live inventory state.'''
        return {
            'on_hand': self.on_hand.copy(),
            'pipeline_length': len(self.pipeline),
            'latest_rmse': self.oos_errors[-1].copy()
        }

    def get_simulation_logs(self) -> pd.DataFrame:
        '''Retrieves full concatenated log history up to the current day.'''
        if not self.simulation_logs:
            return pd.DataFrame()
        return pd.concat(self.simulation_logs, ignore_index=True)

    def _safety_stock(self,type:str='rmse'):
        ''' rmse_error should be from previous risk period, not current risk period'''
        z_score = 0.97
        return z_score * self.oos_errors[-1] * np.sqrt(self.risk_period)

    def update_oos_error(self,forecasts_prior_review_date:pd.DataFrame):
        ''' calculate rolling rmse error stats from the growing self.forecast_df'''
        # self.forecasts_df contains the forecast for previous month and this month, so
        # filter first to take in dates before the current review period, ->forecasts_prior_review_date
        rolling_error_df = compute_rolling_tau_error(forecasts=forecasts_prior_review_date)
        error_stats = calculate_error_statistics(rolling_error_df)
        self.oos_errors.append(error_stats)


    def generate_order(self,forecasted_risk_period:pd.DataFrame,on_hand_current:pd.DataFrame,on_order:pd.DataFrame):
        '''forecasted_risk_period: forecasted_demand for the next risk period to generate order'''   
        # Extract latest TAU-day forecast horizon per SKU
        risk_demand_item = (forecasted_risk_period.groupby("item_id")["sales_pred"].sum())

        #calculate the safety stock
        safety_stock = self._safety_stock(type='rmse')# should be series
        # order_up_to (S)
        order_up_to = risk_demand_item.add(safety_stock,fill_value=0.0)
        inventory_position = ( on_hand_current.reindex(order_up_to.index, fill_value=0.0)
            + on_order.reindex(order_up_to.index, fill_value=0.0) )

        order_qty = ( order_up_to - inventory_position).clip(lower=0.0)

        return order_qty, order_up_to

    def daily_update(self,on_hand:pd.Series,arriving_qty:pd.Series,actual_sales_day:pd.Series):

        on_hand = on_hand.add(arriving_qty,fill_value=0.0) 
        
        fulfilled_sales = pd.concat([on_hand,actual_sales_day],axis=1).min(axis=1)

        lost_sales = (actual_sales_day.sub(fulfilled_sales,fill_value=0.0).clip(lower=0.0)) #no negative

        on_hand = (on_hand.sub(fulfilled_sales,fill_value=0.0).clip(lower=0))

        holding_cost = (on_hand*self.daily_unit_holding_cost)
        stockout_cost = (lost_sales*self.stockout_penalty)

        return (on_hand,lost_sales,holding_cost,stockout_cost)


    def daily_simulation(self,actual_sales:pd.DataFrame,forecasted_demand:pd.DataFrame,
                        ):

        dates = forecasted_demand['date'].unique().tolist()

        initial_forecast_origin = forecasted_demand['date'].min()

        # retrieve the inventory on hand and arriving on the date 
        new_logs = []

        for idx,day in enumerate(dates):

            sales_today_df = actual_sales[pd.to_datetime(actual_sales['date']) == day]
            preds_today_df = forecasted_demand[pd.to_datetime(forecasted_demand['date']) == day]

            # if the day is review day, make orders for the next period
            order_qty = pd.Series(0.0, index=self.on_hand.index)
            order_up_to = pd.Series(0.0, index=self.on_hand.index)

            if idx % self.review_period == 0:
                # get the forecasted demand in current risk period
                forecasts_prior_review = self.forecasts_df[self.forecasts_df['date']<=day]
                # update the errors
                self.update_oos_errors(forecasts_prior_review)

                forecast_origin = initial_forecast_origin + pd.Timedelta(days=self.review_period)
                risk_period_end = forecast_origin + pd.Timedelta(days=self.risk_period - 1)
                risk_period_forecast = forecasted_demand[
                    forecasted_demand['date'].between(forecast_origin, risk_period_end)
                ].copy()
                # compute current active on_order
                on_order = pd.Series(0.0, index=self.on_hand.index)
                for arr_date, qty_series in self.transit_pipeline:
                    if arr_date >= day:
                        on_order = on_order.add(qty_series, fill_value=0.0)
                # generate the order
                order_qty,order_up_to = self.generate_order(
                    forecasted_risk_period=risk_period_forecast,
                    on_hand_current=self.on_hand,
                    on_order= on_order)
                # Schedule in pipeline
                arrival_date = day + pd.Timedelta(days=self.lead_time)
                self.pipeline.append((arrival_date, order_qty))


            # 3. Process Daily Deliveries & Actual Sales
            arriving_qty = pd.Series(0.0, index=self.on_hand.index)
            for arr_date, qty_series in self.pipeline:
                if arr_date == day:
                    arriving_qty = arriving_qty.add(qty_series, fill_value=0.0)

            sales_on_day = sales_today_df.groupby('item_id')['sales'].sum()
            sales_on_day = sales_on_day.reindex(self.on_hand.index, fill_value=0.0)
            # Update live on_hand state
            self.on_hand, lost_sales, holding_cost, stockout_cost = self.daily_update(
                            on_hand=self.on_hand,
                            arriving_qty=arriving_qty,
                            actual_sales_day=sales_on_day
                        )
            # 4. Construct daily log for inventory
            daily_logs = pd.DataFrame({
                'date': day,
                'item_id': self.on_hand.index,
                'on_hand': self.on_hand.values,
                'arriving_qty': arriving_qty.values,
                'actual_sales': sales_on_day.values,
                'lost_sales': lost_sales.values,
                'order_qty': order_qty.reindex(self.on_hand.index, fill_value=0.0).values,
                'order_up_to': order_up_to.reindex(self.on_hand.index, fill_value=0.0).values,
                'holding_cost': holding_cost.values,
                'stockout_cost': stockout_cost.values
            })  
            
            new_logs.append(daily_logs)
            self.simulation_logs.append(daily_logs)

        return pd.concat(new_logs, ignore_index=True)

                
                
def compute_rolling_tau_error(forecasts:pd.DataFrame, tau: int):
    
    df = forecasts.copy()
    g = df.groupby('item_id', observed=True)

    rolled_actual = g['sales'].rolling(tau).sum().reset_index(level=0, drop=True)
    rolled_forecast = g['sales_pred'].rolling(tau).sum().reset_index(level=0, drop=True)

    df['_rolled_actual'] = rolled_actual
    df['_rolled_forecast'] = rolled_forecast

    df['cum_error'] = (
        df.groupby('item_id', observed=True)['_rolled_actual'].shift(-(tau - 1))
        - df.groupby('item_id', observed=True)['_rolled_forecast'].shift(-(tau - 1))
    )

    return df.drop(columns=['_rolled_actual', '_rolled_forecast'])


def calculate_error_statistics(df_with_error: pd.DataFrame) -> pd.DataFrame:
    """Computes RMSE and MAE across time for cumulative TAU-day errors per item."""
    error_stats = (
        df_with_error.groupby("item_id",observed=True)["cum_error"]
        .agg(
            rmse_tau=lambda x: np.sqrt((x.dropna() ** 2).mean()),
            n_obs=lambda x: x.dropna().shape[0],
        )
        .reset_index()
    )
    return error_stats

