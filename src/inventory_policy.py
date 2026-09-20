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

    def __init__(self,inventory:pd.DataFrame,lead_time:int=14,review_period:int=7,daily_unit_holding_cost:float=0.2,stockout_cost:float=1,
                 oos_error_list:list[pd.Series]| None=None):

        self.lead_time = lead_time
        self.review_period = review_period
    
        self.risk_period = lead_time+review_period

        self.daily_unit_holding_cost=daily_unit_holding_cost 
        self.stockout_penalty = stockout_cost

        # service level with given costs== 83.33% and z-score = 0.97
        self.target_service_level = 1/(1+0.2)

        #inventory data with on_hand and in_transit, for on_hand date re
        # get populated through continous day simulations,
        self.inventory = inventory 
        # intialize the oos error list: tracking the error deviation in the previous risk_period  
        if oos_error_list is None:
            logging.ERROR(f"Out Of Sample Error list is None. Provide a list ")
        self.oos_errors = oos_error_list # need to be updated every review period 
        

    def _safety_stock(self,type:str='rmse'|'quantile'):
        ''' rmse_error should be from previous risk period, not current risk period'''
        z_score = 0.97
        return z_score * self.oos_errors[-1] * np.sqrt(self.risk_period)

    def update_oos_error_list(self):
        ''' find a optimized way to update the error stats for the prior risk period'''
        pass 

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
        
    

        for day in dates:
            inventory_on_day = self.inventory[self.inventory['date']==day]

            on_hand = inventory_on_day.groupby('item_id')['on_hand']
            on_arrival = inventory_on_day.groupby('item_id')['arrival_qty']
            
            # if the day is review day, make orders for the next period
            if day % self.review_period:
                # get the forecasted demand in current risk period
                forecast_origin = initial_forecast_origin + pd.Timedelta(days=self.review_period)
                risk_period_end = forecast_origin + pd.Timedelta(days=self.risk_period - 1)
                risk_period_forecast = forecasted_demand[
                    forecasted_demand['date'].between(forecast_origin, risk_period_end)
                ].copy()

                
                order_qty,order_up_to = self.generate_order(
                    forecasted_risk_period=risk_period_forecast,on_hand_current=on_hand,
                    on_order= on_arrival)
                
                

                self

                
                








def compute_rolling_tau_error(test_df: pd.DataFrame, pred_df: pd.DataFrame, tau: int):
    df = test_df.merge(pred_df, on=['item_id', 'date'], how='inner').sort_values(['item_id', 'date']).copy()

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


def generate_inventory_policy(df: pd.DataFrame, error_stats: pd.DataFrame, lead_time: int = 4, review_period: int = 7, k_factor: float = 1.645, pred_col: str = "sales_pred",
    sales_col: str = "sales", demand_history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Generates Order-Up-To levels and Safety Stock buffers comparing RMSE, MAE,

    and Classical baselines.
    """
    tau = lead_time + review_period

    # Extract latest TAU-day forecast horizon per SKU
    latest_forecast_tau = (df.sort_values("date").groupby("item_id").tail(tau).groupby("item_id")[pred_col].sum().rename("forecast_tau") )

    policy = error_stats.merge(latest_forecast_tau, on="item_id")

    # Safety Stock and Order-Up-To calculations
    policy["safety_stock_rmse"] = k_factor * policy["rmse_tau"]
    policy["order_up_to_rmse"] = ( policy["forecast_tau"] + policy["safety_stock_rmse"] )

    # Classical baseline comparison (Standard deviation over lead time)
    # Future production rows do not have observed sales yet.  Use historical demand
    # for the classical-demand variability estimate in that case.
    demand_source = demand_history if demand_history is not None else df

    return policy

def safety_stock_base(std_raw_demand, L:int=4):
    '''std_raw_demand: standard deviation of raw demand
    L: lead time'''
    return 1.645*std_raw_demand*np.sqrt(L)

def calculate_inventory_costs(df: pd.DataFrame,
                              policy: pd.DataFrame,
                              review_period: int = 7,
                              holding_cost_per_unit_day: float = 0.02, 
                              pred_col: str = "sales_pred") -> pd.DataFrame:
    
    """Estimate holding cost for one review period from policy-implied inventory.

    This is an analytical estimate based on forecast cycle demand and safety stock;
    it is not a day-by-day inventory simulation using realised demand.
    """
    cycle_demand_r = ( df.sort_values("date").groupby("item_id").tail(review_period).groupby("item_id")[pred_col].sum().rename("cycle_demand_R"))

    policy_cost = policy.merge(cycle_demand_r, on="item_id")
    policy_cost["cycle_stock"] = policy_cost["cycle_demand_R"] / 2.0

    for method in ["rmse", "mae", "classical"]:
        policy_cost[f"avg_on_hand_{method}"] = (
            policy_cost["cycle_stock"] + policy_cost[f"safety_stock_{method}"]
        )
        policy_cost[f"monthly_holding_cost_{method}"] = (
            policy_cost[f"avg_on_hand_{method}"]
            * holding_cost_per_unit_day
            * review_period
        )

    return policy_cost


def run_inventory_pipeline(current_raw: pd.DataFrame | None,
                           forecasted_demand: pd.DataFrame,
                           oos_error: pd.DataFrame | None = None,
                           review_period:int=
                           
                           7,lead_time:int=4,
                           holding_cost_per_unit:int=0.02,
                           demand_history: pd.DataFrame | None = None,
                           return_details: bool = False) -> pd.Series | dict[str, pd.DataFrame | pd.Series]:
    '''
    current_raw: current raw sales. Optional for production forecasting, where
        future actual demand is not available yet.
    current_forecast: current forecasted sales
    oos_raw : previous month raw sales
    oos_predict: previous month predicted sales (to calculate error deviation)
    review_period: reorder period
    lead_time   : lead time,
    holding_cost_per_unit: fixed, can try different
    '''
    if oos_error is None:
        raise ValueError("oos_error is required to estimate safety stock.")
    if current_forecast.empty:
        raise ValueError("current_forecast must contain at least one forecast row.")

    tau = lead_time + review_period
    horizon = current_forecast['date'].nunique()
    initial_forecast_origin = current_forecast['date'].min()

    ##### if oos_error is not empty, then run the pipeline ##########
    full_policy = []
    full_cost  = []

    # 1. Estimate demand uncertainty from the already observed calibration period.
    historical_error_stats = calculate_error_statistics(oos_error)
    if historical_error_stats.empty:
        raise ValueError("oos_error contains no complete tau-day error observations.")

    # 2. Create a new order-up-to decision every review period.  The protection
    # horizon is tau days, but decisions advance by review_period days.
    for review_offset in range(0, horizon, review_period):
        forecast_origin = initial_forecast_origin + pd.Timedelta(days=review_offset)
        risk_period_end = forecast_origin + pd.Timedelta(days=tau - 1)
        risk_period_forecast = current_forecast[
            current_forecast['date'].between(forecast_origin, risk_period_end)
        ].copy()
        # An order-up-to level protects demand across the full tau horizon.  Do
        # not emit a final partial-horizon decision simply because an offline
        # stress-test extract ends before that protection period does.
        if risk_period_forecast.empty or risk_period_forecast['date'].nunique() < tau:
            continue

        policy_risk_period = generate_inventory_policy(
            risk_period_forecast,
            historical_error_stats,
            lead_time=lead_time,
            review_period=review_period,
            demand_history=demand_history if demand_history is not None else current_raw,
        )
        policy_risk_period["review_date"] = forecast_origin
        policy_risk_period["protection_end_date"] = risk_period_end

        inventory_cost_per_risk_period = calculate_inventory_costs(
            df=risk_period_forecast,
            policy=policy_risk_period,
            review_period=review_period,
            holding_cost_per_unit_day=holding_cost_per_unit,
        )
        inventory_cost_per_risk_period["review_date"] = forecast_origin
        full_policy.append(policy_risk_period)
        full_cost.append(inventory_cost_per_risk_period)

    if not full_cost:
        raise ValueError("No inventory policy could be generated for the forecast horizon.")

    policy = pd.concat(full_policy, ignore_index=True)
    cost = pd.concat(full_cost,ignore_index=True)

    cost_cols = ['monthly_holding_cost_mae','monthly_holding_cost_classical','monthly_holding_cost_rmse']

    total_cost = cost[cost_cols].sum()


    if return_details:
        return {
            "policy": policy,
            "costs": cost,
            "cost_summary": total_cost,
            "error_statistics": historical_error_stats,
        }
    return total_cost
