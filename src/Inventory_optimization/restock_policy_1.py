''' Store replenishment policy assuming a order upto level periodic review setup
    V1:  Review Period: 7 days (for all), lead time = 4 days,
        Holding cost rate : 20% unit cost annually
        Stockout Cost     : not applied in v1, we use fill rate
    
    safety_stock_base : assuming normal distribution, classical formula: Ss = z*\sigma_d * np.sqrt(L), z:
    safety_stock_v1   : Ss = \sigma_error*np.sqrt(L+R)*
    '''

import pandas as pd 
import numpy as np 

# ASSUMPTIONS — document these, adjust freely


 
#Safety Stock Formulas
## building the benchmark stocking asuuming a normal distribution


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
            mae_tau=lambda x: x.dropna().abs().mean(),
            n_obs=lambda x: x.dropna().shape[0],
        )
        .reset_index()
    )
    return error_stats


def generate_inventory_policy(df: pd.DataFrame, error_stats: pd.DataFrame, lead_time: int = 4, review_period: int = 7, k_factor: float = 1.645, pred_col: str = "sales_pred",
    sales_col: str = "sales") -> pd.DataFrame:
    """Generates Order-Up-To levels and Safety Stock buffers comparing RMSE, MAE,

    and Classical baselines.
    """
    tau = lead_time + review_period

    # Extract latest TAU-day forecast horizon per SKU
    latest_forecast_tau = (df.sort_values("date").groupby("item_id").tail(tau).groupby("item_id")[pred_col].sum().rename("forecast_tau") )

    policy = error_stats.merge(latest_forecast_tau, on="item_id")

    # Safety Stock and Order-Up-To calculations
    policy["safety_stock_rmse"] = k_factor * policy["rmse_tau"]
    policy["safety_stock_mae"] = k_factor * policy["mae_tau"]
    policy["order_up_to_rmse"] = ( policy["forecast_tau"] + policy["safety_stock_rmse"] )
    policy["order_up_to_mae"] = ( policy["forecast_tau"] + policy["safety_stock_mae"] )

    # Classical baseline comparison (Standard deviation over lead time)
    raw_std = df.groupby("item_id")[sales_col].std().rename("raw_demand_std")
    policy = policy.merge(raw_std, on="item_id")
    policy["safety_stock_classical"] = (
        k_factor * policy["raw_demand_std"] * np.sqrt(lead_time)
    )
    policy["order_up_to_classical"] = (
        policy["forecast_tau"] + policy["safety_stock_classical"]
    )

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
    
    """Calculates cycle stock, average on-hand inventory, and monthly holding costs

    across policy methods.
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
            * 7
        )

    return policy_cost


def run_inventory_pipeline(current_raw:pd.DataFrame,
                           current_forecast: pd.DataFrame,
                           oos_error : list|None=None,
                           review_period:int=7,lead_time:int=4,
                           holding_cost_per_unit:int=0.02)->pd.DataFrame:
    '''
    current_raw: current raw sales
    current_forecast: current forecasted sales
    oos_raw : previous month raw sales
    oos_predict: previous month predicted sales (to calculate error deviation)
    review_period: reorder period
    lead_time   : lead time,
    holding_cost_per_unit: fixed, can try different
    '''
    tau = lead_time + review_period
    horizon = current_forecast['date'].nunique()

    forecast_origin = current_forecast['date'].min()

    ##### if oos_error is not empty, then run the pipeline ##########
    full_policy = []
    full_cost  = []

    if oos_error is not None:

        ## 1. calculate the error stats per item
        historical_error_stats = calculate_error_statistics(oos_error)

        ### reorder every review days
        for risk_period in range(0,horizon,review_period):
            risk_period_end = forecast_origin + pd.Timedelta(days=tau)
            # split the actual vs demand to calculate inventory for the risk periods only
            risk_period_forecast = current_forecast[current_forecast['date'].between(forecast_origin,risk_period_end)].copy()
            risk_period_actual = current_raw[current_raw['date'].between(forecast_origin,risk_period_end)].copy()

            merged_riskperiod = risk_period_actual.merge(risk_period_forecast,on=['item_id','date'],how='inner')
            # generate policy per risk period
            policy_risk_period = generate_inventory_policy(merged_riskperiod,
                                                           historical_error_stats,
                                                           lead_time=lead_time,
                                                           review_period=review_period,
                                                           )

            inventory_cost_per_risk_period = calculate_inventory_costs(df=merged_riskperiod,
                                                                       policy=policy_risk_period,
                                                                       review_period=review_period,
                                                                       holding_cost_per_unit_day=0.02)

            full_policy.append(policy_risk_period)
            full_cost.append(inventory_cost_per_risk_period)

    cost = pd.concat(full_cost,ignore_index=True)

    cost_cols = ['monthly_holding_cost_mae','monthly_holding_cost_classical','monthly_holding_cost_rmse']

    total_cost = cost[cost_cols].sum()


    return total_cost