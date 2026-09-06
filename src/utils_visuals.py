"""Interactive Plotly visualization module for demand forecasting and inventory optimization.

Designed for seamless integration into Streamlit dashboards, Jupyter notebooks,
and portfolio presentations. Follows strict typing and PEP 8 standards.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _extract_series(data: pd.Series | pd.DataFrame, value_col: str = "sales", date_col: str = "date") -> pd.Series:
    """Helper to convert either a pd.Series or pd.DataFrame into a date-indexed pd.Series."""
    if isinstance(data, pd.DataFrame):
        if date_col in data.columns and value_col in data.columns:
            sorted_df = data.sort_values(date_col)
            return pd.Series(sorted_df[value_col].values, index=pd.to_datetime(sorted_df[date_col]), name=value_col)
        elif value_col in data.columns:
            return pd.Series(data[value_col].values, index=pd.to_datetime(data.index), name=value_col)
        else:
            raise ValueError(f"Column '{value_col}' not found in DataFrame. Available columns: {list(data.columns)}")
    elif isinstance(data, pd.Series):
        s = data.copy()
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    else:
        raise TypeError(f"Expected pd.Series or pd.DataFrame, got {type(data).__name__}")


def plot_item_forecast_and_inventory(
    raw_sales: pd.Series | pd.DataFrame,
    forecasted_demand: Mapping[str, pd.Series] | pd.Series | pd.DataFrame,
    p10: pd.Series,
    p90: pd.Series,
    inventory_values: Mapping[str, pd.Series | float | int],
    p95: pd.Series | None = None,
    item_id: str | None = None,
    sales_as_bars: bool = True,
    theme: str = "plotly_white",
) -> go.Figure:
    """Creates an interactive Plotly visualization comparing raw sales, multi-model forecasts,
    quantile prediction intervals, and inventory policy threshold levels for a single SKU.

    Designed for direct rendering in Streamlit dashboards via `st.plotly_chart(fig)`.

    Parameters
    ----------
    raw_sales : pd.Series | pd.DataFrame
        Historical / ground-truth sales demand over time for the target item.
        If Series, index should be date/datetime. If DataFrame, requires 'date' and 'sales'.
    forecasted_demand : Mapping[str, pd.Series] | pd.Series | pd.DataFrame
        Point forecasts for the item across one or more models (e.g., LightGBM,
        Moving Average, Seasonal Naive).
    p10 : pd.Series
        10th percentile (P10) demand forecast bound over time.
    p90 : pd.Series
        90th percentile (P90) demand forecast bound over time.
    inventory_values : Mapping[str, pd.Series | float | int]
        Inventory order-up-to levels or safety stock buffers from TWO distinct
        inventory policies (e.g. {'Policy 1 (RMSE)': 52.0, 'Policy 2 (Classical)': 68.0}).
    p95 : pd.Series | None, optional
        Optional 95th percentile (P95) upper bound over time.
    item_id : str | None, optional
        SKU identifier for chart titles and annotations.
    sales_as_bars : bool, default True
        If True, plots actual demand as sleek vertical bars (industry standard
        in retail/Streamlit dashboards). If False, plots as lines with markers.
    theme : str, default 'plotly_white'
        Plotly theme template ('plotly_white', 'plotly_dark', etc.).

    Returns
    -------
    go.Figure
        A Plotly figure object ready for Streamlit (`st.plotly_chart(fig)`) or `fig.show()`.
    """
    fig = go.Figure()

    # 1. Format raw sales
    sales_series = _extract_series(raw_sales, value_col="sales")
    dates = sales_series.index

    # 2. Add Ground-Truth Actual Demand (Bar or Line)
    if sales_as_bars:
        fig.add_trace(
            go.Bar(
                x=dates,
                y=sales_series.values,
                name="Actual Demand",
                marker=dict(color="#cbd5e1", line=dict(color="#94a3b8", width=1)),
                opacity=0.65,
                hovertemplate="<b>Actual Demand</b><br>Date: %{x|%Y-%m-%d}<br>Sales: %{y:.1f} units<extra></extra>",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=sales_series.values,
                name="Actual Demand",
                mode="lines+markers",
                line=dict(color="#334155", width=2),
                marker=dict(size=4),
                hovertemplate="<b>Actual Demand</b><br>Date: %{x|%Y-%m-%d}<br>Sales: %{y:.1f} units<extra></extra>",
            )
        )

    # 3. Add Quantile Prediction Band (P10 -> P90 shaded confidence area)
    p10_series = _extract_series(p10, value_col="p10")
    p90_series = _extract_series(p90, value_col="p90")
    forecast_dates = p10_series.index

    # P10 lower boundary (transparent line)
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=p10_series.values,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # P90 upper boundary filled to P10
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=p90_series.values,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(2, 132, 199, 0.15)",
            name="80% Forecast Bound (P10–P90)",
            hovertemplate="<b>80% Interval</b><br>Upper (P90): %{y:.1f}<extra></extra>",
        )
    )

    # Optional P95 Upper Bound
    if p95 is not None:
        p95_series = _extract_series(p95, value_col="p95")
        fig.add_trace(
            go.Scatter(
                x=p95_series.index,
                y=p95_series.values,
                mode="lines",
                line=dict(color="rgba(2, 132, 199, 0.5)", width=1.5, dash="dash"),
                name="P95 Extreme Demand Bound",
                hovertemplate="<b>P95 Bound</b><br>Date: %{x|%Y-%m-%d}<br>Value: %{y:.1f} units<extra></extra>",
            )
        )

    # 4. Add Forecasted Demand Lines
    model_palettes = {
        "lgbm": {"color": "#0284c7", "width": 3.0, "dash": "solid", "label": "LightGBM (Recursive)"},
        "moving_average": {"color": "#f59e0b", "width": 2.0, "dash": "dash", "label": "Moving Average (180d)"},
        "seasonal_naive": {"color": "#8b5cf6", "width": 2.0, "dash": "dot", "label": "Seasonal Naive (28d)"},
    }
    fallback_colors = ["#ec4899", "#14b8a6", "#f97316", "#6366f1"]

    # Normalize forecasted_demand into dict of {model_name: pd.Series}
    models_dict: dict[str, pd.Series] = {}
    if isinstance(forecasted_demand, Mapping):
        for m_name, m_data in forecasted_demand.items():
            models_dict[m_name] = _extract_series(m_data, value_col="sales_pred")
    elif isinstance(forecasted_demand, pd.DataFrame):
        for col in forecasted_demand.columns:
            if col not in ("date", "item_id", "dept_id", "cat_id"):
                models_dict[col] = _extract_series(forecasted_demand, value_col=col)
    elif isinstance(forecasted_demand, pd.Series):
        models_dict[forecasted_demand.name or "Model Forecast"] = _extract_series(forecasted_demand)

    for i, (m_name, m_series) in enumerate(models_dict.items()):
        norm_key = m_name.lower().replace(" ", "_")
        matched_style = None
        for key, style in model_palettes.items():
            if key in norm_key:
                matched_style = style
                break

        if matched_style is None:
            color = fallback_colors[i % len(fallback_colors)]
            style = {"color": color, "width": 2.0, "dash": "solid", "label": m_name}
        else:
            style = matched_style

        fig.add_trace(
            go.Scatter(
                x=m_series.index,
                y=m_series.values,
                mode="lines",
                name=style.get("label", m_name),
                line=dict(color=style["color"], width=style["width"], dash=style["dash"]),
                hovertemplate=f"<b>{m_name}</b><br>Date: %{{x|%Y-%m-%d}}<br>Forecast: %{{y:.2f}} units<extra></extra>",
            )
        )

    # 5. Add Inventory Values from Two Different Inventory Policies
    policy_styles = [
        {"color": "#10b981", "dash": "dot", "width": 2.5, "symbol": "diamond"},
        {"color": "#ef4444", "dash": "dash", "width": 2.5, "symbol": "square"},
    ]

    for idx, (p_name, p_val) in enumerate(inventory_values.items()):
        style = policy_styles[idx % len(policy_styles)]
        # If policy value is a time series
        if isinstance(p_val, (pd.Series, pd.DataFrame)):
            p_series = _extract_series(p_val, value_col=list(p_val.columns)[0] if isinstance(p_val, pd.DataFrame) else "inventory")
            fig.add_trace(
                go.Scatter(
                    x=p_series.index,
                    y=p_series.values,
                    mode="lines",
                    name=f"Inventory Level: {p_name}",
                    line=dict(color=style["color"], width=style["width"], dash=style["dash"]),
                    hovertemplate=f"<b>{p_name}</b><br>Date: %{{x|%Y-%m-%d}}<br>Target: %{{y:.1f}} units<extra></extra>",
                )
            )
        # If policy value is a scalar threshold across the horizon
        elif isinstance(p_val, (int, float, np.number)):
            fig.add_trace(
                go.Scatter(
                    x=[forecast_dates.min(), forecast_dates.max()],
                    y=[p_val, p_val],
                    mode="lines",
                    name=f"Inventory Buffer ({p_name}): {p_val:.1f}u",
                    line=dict(color=style["color"], width=style["width"], dash=style["dash"]),
                    hovertemplate=f"<b>{p_name} Level</b>: {p_val:.1f} units<extra></extra>",
                )
            )

    # 6. Title and Layout Styling
    title_text = f"Demand Forecast & Inventory Policy Benchmark"
    if item_id:
        title_text += f" — SKU: <b>{item_id}</b>"

    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=18, family="Inter, Roboto, sans-serif"),
            x=0.02,
            y=0.96,
        ),
        template=theme,
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.2)",
            tickformat="%b %d",
            hoverformat="%Y-%m-%d",
        ),
        yaxis=dict(
            title="Quantity (Units)",
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.2)",
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.75)" if "white" in theme else "rgba(15, 23, 42, 0.75)",
            font=dict(size=11),
        ),
        hovermode="x unified",
        margin=dict(l=50, r=40, t=110, b=50),
        height=520,
    )

    # 7. Add Interactive Buttons (Updatemenu for Streamlit / Standalone toggle)
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=1.0,
                xanchor="right",
                y=1.14,
                yanchor="top",
                buttons=[
                    dict(label="Show All", method="restyle", args=[{"visible": [True] * len(fig.data)}]),
                    dict(
                        label="ML vs Actual",
                        method="restyle",
                        args=[
                            {
                                "visible": [
                                    True if any(term in (trace.name or "").lower() for term in ("actual", "lightgbm", "bound", "p10", "p90")) else False
                                    for trace in fig.data
                                ]
                            }
                        ],
                    ),
                    dict(
                        label="Baselines Only",
                        method="restyle",
                        args=[
                            {
                                "visible": [
                                    True if any(term in (trace.name or "").lower() for term in ("actual", "moving", "naive")) else False
                                    for trace in fig.data
                                ]
                            }
                        ],
                    ),
                ],
                font=dict(size=11),
                bgcolor="rgba(241, 245, 249, 0.9)" if "white" in theme else "rgba(30, 41, 59, 0.9)",
                bordercolor="#cbd5e1",
            )
        ]
    )

    return fig


def plot_inventory_policy_bars(
    inventory_values: Mapping[str, float | int | Mapping[str, float]],
    metric_name: str = "Order-Up-To Level (Units)",
    item_id: str | None = None,
    theme: str = "plotly_white",
) -> go.Figure:
    """Creates a high-contrast Plotly bar chart comparing two inventory policies side-by-side.

    Ideal for Streamlit dashboard widgets displaying holding cost, safety stock,
    or order-up-to level comparisons.

    Parameters
    ----------
    inventory_values : Mapping[str, float | int | Mapping[str, float]]
        Dictionary containing policy names and either scalar values or metric dictionaries
        from the two policies.
        Example 1 (Scalar):
            {'Policy 1 (RMSE-Based)': 52.4, 'Policy 2 (Classical Normal)': 68.1}
        Example 2 (Multi-Metric):
            {'Policy 1 (RMSE)': {'Safety Stock': 22.4, 'Holding Cost ($)': 180.5},
             'Policy 2 (Classical)': {'Safety Stock': 38.1, 'Holding Cost ($)': 250.2}}
    metric_name : str, default "Order-Up-To Level (Units)"
        Label for the metric being compared if scalar values are supplied.
    item_id : str | None, optional
        SKU identifier for the chart title.
    theme : str, default 'plotly_white'
        Plotly theme template.

    Returns
    -------
    go.Figure
        A Plotly figure object ready for Streamlit (`st.plotly_chart(fig)`).
    """
    fig = go.Figure()
    palette = ["#0284c7", "#10b981", "#f59e0b", "#ef4444"]

    # Check if nested dictionary (multi-metric)
    first_val = next(iter(inventory_values.values())) if inventory_values else None
    if isinstance(first_val, Mapping):
        policies = list(inventory_values.keys())
        all_metrics = list(first_val.keys())

        for idx, metric in enumerate(all_metrics):
            vals = [inventory_values[p].get(metric, 0.0) for p in policies]
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=policies,
                    y=vals,
                    marker=dict(color=palette[idx % len(palette)]),
                    text=[f"{v:,.1f}" for v in vals],
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>" + metric + ": %{y:,.2f}<extra></extra>",
                )
            )
        fig.update_layout(barmode="group")
        y_label = "Metric Value"
    else:
        # Scalar comparison
        policies = list(inventory_values.keys())
        values = [float(inventory_values[p]) for p in policies]
        bar_colors = ["#0284c7", "#10b981"] if len(policies) == 2 else palette[:len(policies)]

        fig.add_trace(
            go.Bar(
                x=policies,
                y=values,
                marker=dict(color=bar_colors),
                text=[f"{v:,.1f}" for v in values],
                textposition="auto",
                hovertemplate="<b>%{x}</b><br>" + metric_name + ": %{y:,.2f}<extra></extra>",
            )
        )
        y_label = metric_name

    title_text = f"Inventory Policy Comparison"
    if item_id:
        title_text += f" — <b>{item_id}</b>"

    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=16, family="Inter, Roboto, sans-serif"),
            x=0.02,
            y=0.96,
        ),
        template=theme,
        xaxis=dict(title="Replenishment Policy", showgrid=False),
        yaxis=dict(title=y_label, showgrid=True, gridcolor="rgba(148, 163, 184, 0.2)"),
        margin=dict(l=50, r=40, t=80, b=50),
        height=420,
    )

    return fig


def extract_item_inputs_from_dataframes(
    raw_df: pd.DataFrame,
    preds_dict: Mapping[str, pd.DataFrame],
    target_item_id: str,
    policy_df: pd.DataFrame | None = None,
    policy_1_name: str = "order_up_to_rmse",
    policy_2_name: str = "order_up_to_classical",
    date_col: str = "date",
    sales_col: str = "sales",
    pred_col: str = "sales_pred",
) -> tuple[pd.Series, dict[str, pd.Series], pd.Series, pd.Series, dict[str, float | pd.Series]]:
    """Helper utility extracting the exact per-item parameters required by
    `plot_item_forecast_and_inventory` directly from pipeline evaluation DataFrames.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Ground-truth evaluation window with 'item_id', 'date', 'sales'.
    preds_dict : Mapping[str, pd.DataFrame]
        Dictionary mapping model labels ('lgbm', 'ma', 'naive') to their prediction DataFrames.
    target_item_id : str
        The SKU identifier to slice.
    policy_df : pd.DataFrame | None, optional
        Policy DataFrame from `generate_inventory_policy` containing policy columns.
    policy_1_name : str, default 'order_up_to_rmse'
        Name of first inventory policy column to compare.
    policy_2_name : str, default 'order_up_to_classical'
        Name of second inventory policy column to compare.
    date_col : str, default 'date'
        Date column name.
    sales_col : str, default 'sales'
        Actual sales column name.
    pred_col : str, default 'sales_pred'
        Forecast prediction column name.

    Returns
    -------
    tuple[pd.Series, dict[str, pd.Series], pd.Series, pd.Series, dict[str, float | pd.Series]]
        (raw_sales, forecasted_demand, p10, p90, inventory_values) ready for
        `plot_item_forecast_and_inventory(...)`.
    """
    # 1. Filter raw sales for the target item
    item_raw = raw_df[raw_df["item_id"] == target_item_id].sort_values(date_col)
    if item_raw.empty:
        raise ValueError(f"SKU '{target_item_id}' not found in raw_df.")

    raw_sales = pd.Series(item_raw[sales_col].values, index=pd.to_datetime(item_raw[date_col]), name="sales")

    # 2. Extract model forecasts
    forecasted_demand: dict[str, pd.Series] = {}
    p10_series: pd.Series | None = None
    p90_series: pd.Series | None = None

    for m_label, m_df in preds_dict.items():
        sub_df = m_df[m_df["item_id"] == target_item_id].sort_values(date_col)
        if not sub_df.empty and pred_col in sub_df.columns:
            s = pd.Series(sub_df[pred_col].values, index=pd.to_datetime(sub_df[date_col]), name=m_label)
            forecasted_demand[m_label] = s

            # Capture quantiles from model if present
            if "q10" in sub_df.columns and p10_series is None:
                p10_series = pd.Series(sub_df["q10"].values, index=pd.to_datetime(sub_df[date_col]), name="p10")
            if "q90" in sub_df.columns and p90_series is None:
                p90_series = pd.Series(sub_df["q90"].values, index=pd.to_datetime(sub_df[date_col]), name="p90")

    # Fallback Gaussian quantiles if not directly present in predictions
    if p10_series is None or p90_series is None:
        primary_pred = next(iter(forecasted_demand.values()))
        std_est = max(0.5, float(raw_sales.std()))
        p10_series = (primary_pred - 1.28 * std_est).clip(lower=0)
        p90_series = primary_pred + 1.28 * std_est

    # 3. Extract inventory policy values
    inventory_values: dict[str, float | pd.Series] = {}
    if policy_df is not None and not policy_df.empty:
        item_policy = policy_df[policy_df["item_id"] == target_item_id]
        if not item_policy.empty:
            if policy_1_name in item_policy.columns:
                inventory_values[policy_1_name] = float(item_policy[policy_1_name].iloc[0])
            if policy_2_name in item_policy.columns:
                inventory_values[policy_2_name] = float(item_policy[policy_2_name].iloc[0])

    if not inventory_values:
        # Defaults if policy_df was omitted: 1.645 * empirical vs classical
        inventory_values["Policy 1 (Empirical Buffer)"] = float(p90_series.mean() * 1.15)
        inventory_values["Policy 2 (Classical Buffer)"] = float(p90_series.mean() * 1.35)

    return raw_sales, forecasted_demand, p10_series, p90_series, inventory_values
