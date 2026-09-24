from pathlib import Path
from typing import Any, Optional, Union
import pandas as pd
import jinja2

import config

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{{ title_str }}</title>
    <style>
        :root {
            --primary: #1e293b;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --border: #e2e8f0;
            --text-main: #334155;
            --text-muted: #64748b;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            margin: 0;
            padding: 40px 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: var(--card-bg);
            border-radius: 12px;
            border: 1px solid var(--border);
            box-shadow: 0 4px 12px rgba(0,0,0,0.03);
            padding: 36px;
        }
        .header-title { font-size: 1.6rem; font-weight: 700; color: var(--primary); margin: 0 0 6px 0; }
        .meta-subtitle { font-size: 0.88rem; color: var(--text-muted); margin-bottom: 24px; line-height: 1.5; }
        .meta-tag {
            display: inline-block;
            background: #e2e8f0;
            color: #334155;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 6px;
        }
        
        /* Two-column layout hero banner */
        .hero-banner {
            background-color: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 28px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .hero-column { display: flex; flex-direction: column; }
        .hero-column:first-child { border-right: 1px solid #cbd5e1; padding-right: 20px; }
        .hero-banner .headline { font-weight: 700; color: #166534; font-size: 1.1rem; margin-bottom: 4px; }
        .hero-banner .subhead { font-weight: 600; color: #15803d; font-size: 0.82rem; margin-top: 10px; text-transform: uppercase; letter-spacing: 0.5px;}
        .hero-banner .metric-val { font-size: 1.3rem; font-weight: 700; color: #15803d; margin: 4px 0 2px 0; }
        .hero-banner ul { margin: 6px 0 0 18px; padding: 0; font-size: 0.88rem; color: #15803d; }
        .hero-banner li { margin-bottom: 6px; }
        
        .section-title {
            font-size: 1.1rem; font-weight: 600; color: var(--primary);
            margin: 28px 0 12px 0; padding-bottom: 6px; border-bottom: 1px solid var(--border);
        }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.92rem; }
        th, td { padding: 10px 14px; text-align: right; border-bottom: 1px solid var(--border); }
        th { background-color: #f8fafc; color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 0.75rem; }
        td:first-child, th:first-child { text-align: left; }
        tr:hover { background-color: #f1f5f9; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="header-title">{{ title_str }}</h1>
        <div class="meta-subtitle">
            Generated: {{ timestamp }} | Mode: <strong>{{ eval_mode_label }}</strong> ({{ num_windows }} Window/Split)<br/>
            <span class="meta-tag">Lead Time: {{ lead_time }}d</span>
            <span class="meta-tag">Review Period: {{ review_period }}d</span>
            <span class="meta-tag">Safety Stock Policy: {{ policy_type }}</span>
        </div>

        <div class="hero-banner">
            <!-- Left Column: Key Accuracy Metrics -->
            <div class="hero-column">
                <div class="headline">Winning Model: {{ winner_model_name }}</div>
                <div class="subhead">Accuracy Profile (Avg Across {{ num_windows }} Window/Split)</div>
                <ul>
                    <li><strong>cum_MAE:</strong> {{ cand_accuracy.cum_mae }}</li>
                    <li><strong>cum_BIAS:</strong> {{ cand_accuracy.cum_bias }}</li>
                    <li><strong>MAE:</strong> {{ cand_accuracy.mae }}</li>
                    <li><strong>BIAS%:</strong> {{ cand_accuracy.bias_pct }}</li>
                </ul>
            </div>

            <!-- Right Column: Business Impact & Costs -->
            <div class="hero-column">
                <div class="subhead">Business Impact</div>
                <div class="metric-val">{{ cand_cost_str }}</div>
                <div style="font-size: 0.78rem; color: #64748b;">Avg total cost across {{ num_windows }} window/split</div>

                <div class="subhead">Cost Savings vs Other Models</div>
                <ul>
                {% for b in baseline_savings %}
                    <li><strong>${{ b.savings }}</strong> vs {{ b.name }} ({{ b.fva }}% FVA)</li>
                {% endfor %}
                </ul>
            </div>
        </div>

        <div class="section-title">Mean Metrics Across {{ num_windows }} Window/Split</div>
        {{ avg_metrics_table }}

        <div class="section-title">Mean Forecast Value Added (FVA %) Across {{ num_windows }} Window/Split</div>
        {{ avg_fva_table }}
    </div>
</body>
</html>
"""

def generate_experiment_html_report(
    metrics_df: pd.DataFrame,
    fva_df: pd.DataFrame,
    output_path: Path,
    lead_time: int = 7,
    review_period: int = 7,
    policy_type: str = "RMSE (Norm Sim)",
    model_name: Optional[str] = None,
    model_type: Optional[str] = None,
    train_years: Optional[Union[str, int]] = None,
    eval_mode: str = "backtest",  # Accepts "test", or "backtest"
) -> str:
    """
    Vectorized summary generator: averages window metrics and existing FVA dataframes directly.
    Generates a dynamic HTML report selecting the lowest-cost model as winner.
    """
    # 1. Handle window aggregation gracefully for test or backtest
    if "window_id" in metrics_df.columns:
        num_windows: int = int(metrics_df["window_id"].nunique())
        num_cols_metrics = [c for c in metrics_df.select_dtypes(include="number").columns if c != "window_id"]
        avg_metrics_df: pd.DataFrame = (
            metrics_df.groupby("model", observed=True, as_index=False)[num_cols_metrics].mean()
        )
    else:
        num_windows = 1
        avg_metrics_df = metrics_df.copy()

    if "window_id" in fva_df.columns:
        num_cols_fva = [c for c in fva_df.select_dtypes(include="number").columns if c != "window_id"]
        avg_fva_df: pd.DataFrame = (
            fva_df.groupby("metric", observed=True, as_index=False)[num_cols_fva].mean()
        )
    else:
        avg_fva_df = fva_df.copy()

    # 2. Dynamic Title & Mode Resolution
    resolved_model_name: str = (
        model_name if model_name is not None else getattr(config, "MODEL_NAME", "LGBM")
    ).upper()

    resolved_model_type: str = (
        model_type if model_type is not None else getattr(config, "MODEL_TYPE", "recursive")
    ).capitalize()

    resolved_train_years: str = str(
        train_years if train_years is not None else getattr(config, "TRAIN_YEARS", "2")
    )

    eval_mode_label = eval_mode.upper()
    title_str: str = f"[{eval_mode_label}] Result: {resolved_model_name} {resolved_model_type} ({resolved_train_years}-Year Training)"

    # 3. Dynamic Winner Selection (lowest total_cost, fallback to lowest MAE)
    if "total_cost" in avg_metrics_df.columns:
        sorted_metrics = avg_metrics_df.sort_values("total_cost", ascending=True)
    elif "MAE" in avg_metrics_df.columns:
        sorted_metrics = avg_metrics_df.sort_values("MAE", ascending=True)
    else:
        sorted_metrics = avg_metrics_df.copy()

    winning_model_name: str = str(sorted_metrics.iloc[0]["model"])
    cand_mask = avg_metrics_df["model"] == winning_model_name

    # Candidate Accuracy metrics
    cand_accuracy: dict[str, str] = {
        "cum_mae": "N/A", "cum_bias": "N/A", "mae": "N/A", "bias_pct": "N/A"
    }
    cand_cost_str: str = "N/A"

    if cand_mask.any():
        cand_row = avg_metrics_df.loc[cand_mask].iloc[0]
        if "cum_MAE" in cand_row: cand_accuracy["cum_mae"] = f"{cand_row['cum_MAE']:.4f}"
        if "cum_BIAS" in cand_row: cand_accuracy["cum_bias"] = f"{cand_row['cum_BIAS']:.4f}"
        if "MAE" in cand_row: cand_accuracy["mae"] = f"{cand_row['MAE']:.4f}"
        if "BIAS%" in cand_row: cand_accuracy["bias_pct"] = f"{cand_row['BIAS%']:.2f}%"
        if "total_cost" in cand_row: cand_cost_str = f"${cand_row['total_cost']:,.2f}"

    # 4. Savings vs Other Models
    baseline_savings: list[dict[str, str]] = []
    other_models = [m for m in avg_metrics_df["model"].unique() if m != winning_model_name]

    if cand_mask.any() and "total_cost" in avg_metrics_df.columns:
        cand_cost = float(avg_metrics_df.loc[cand_mask, "total_cost"].iloc[0])

        for other_name in other_models:
            other_mask = avg_metrics_df["model"] == other_name
            if other_mask.any():
                base_cost = float(avg_metrics_df.loc[other_mask, "total_cost"].iloc[0])
                savings_val = base_cost - cand_cost
                fva_val = ((base_cost - cand_cost) / base_cost * 100.0) if base_cost > 0 else 0.0

                baseline_savings.append({
                    "name": str(other_name).replace("_", " ").title(),
                    "savings": f"{savings_val:,.2f}",
                    "fva": f"{fva_val:.2f}"
                })

    # 5. Format numeric tables cleanly for Jinja2 render
    format_dict: dict[str, str] = {
        "MAE": "{:.4f}", "BIAS%": "{:.2f}%", "wrmsse": "{:.4f}",
        "cum_BIAS": "{:.4f}", "cum_MAE": "{:.4f}",
        "holding_cost": "${:,.2f}", "stockout_cost": "${:,.2f}", "total_cost": "${:,.2f}"
    }

    avg_metrics_rendered = avg_metrics_df.copy()
    for col, fmt in format_dict.items():
        if col in avg_metrics_rendered.columns:
            avg_metrics_rendered[col] = avg_metrics_rendered[col].apply(
                lambda x: fmt.format(x) if pd.notna(x) else "-"
            )

    # 6. Render HTML
    template = jinja2.Template(HTML_TEMPLATE)
    html_out: str = template.render(
        title_str=title_str,
        timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        num_windows=num_windows,
        eval_mode_label=eval_mode_label,
        lead_time=lead_time,
        review_period=review_period,
        policy_type=policy_type,
        winner_model_name=winning_model_name.upper(),
        cand_accuracy=cand_accuracy,
        cand_cost_str=cand_cost_str,
        baseline_savings=baseline_savings,
        avg_metrics_table=avg_metrics_rendered.to_html(index=False, classes="table", border=0),
        avg_fva_table=avg_fva_df.round(2).to_html(index=False, classes="table", border=0)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_out)
    print(f"\n[INFO] Generated report [{eval_mode_label}]: {output_path}")
    return winning_model_name