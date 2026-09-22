# src/reporting.py
from pathlib import Path
from typing import Any, Optional,Union
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
            Generated: {{ timestamp }} | Backtest Windows: <strong>{{ num_windows }}</strong><br/>
            <span class="meta-tag">Lead Time: {{ lead_time }}d</span>
            <span class="meta-tag">Review Period: {{ review_period }}d</span>
            <span class="meta-tag">Safety Stock Policy: {{ policy_type }}</span>
        </div>

        <div class="hero-banner">
            <!-- Left Column: Key Accuracy Metrics -->
            <div class="hero-column">
                <div class="headline">Winning Model: {{ winner_model_name }}</div>
                <div class="subhead">Accuracy Profile (Avg Across {{ num_windows }} Windows)</div>
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
                <div style="font-size: 0.78rem; color: #64748b;">Avg total cost across {{ num_windows }} windows</div>

                <div class="subhead">Cost Savings vs Baselines</div>
                <ul>
                {% for b in baseline_savings %}
                    <li><strong>${{ b.savings }}</strong> vs {{ b.name }} ({{ b.fva }}% FVA)</li>
                {% endfor %}
                </ul>
            </div>
        </div>

        <div class="section-title">Mean Metrics Across {{ num_windows }} Window(s)</div>
        {{ avg_metrics_table }}

        <div class="section-title">Mean Forecast Value Added (FVA %) Across {{ num_windows }} Window(s)</div>
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
    model_name: Optional[str] = None,   # e.g., "LGBM", "HGB", "XGBoost"
    model_type: Optional[str] = None,   # e.g., "direct", "recursive"
    train_years: Optional[Union[str, int]] = None, # e.g., 2, "2", "3"
) -> None:
    """
    Vectorized summary generator: averages window metrics and existing FVA dataframes directly.
    Generates a fully dynamic HTML report with custom model titles and metadata.
    """
    num_windows: int = int(metrics_df["window_id"].nunique())

    # 1. Vectorized averaging of metrics across windows
    num_cols_metrics = [c for c in metrics_df.select_dtypes(include="number").columns if c != "window_id"]
    avg_metrics_df: pd.DataFrame = (
        metrics_df.groupby("model", observed=True, as_index=False)[num_cols_metrics]
        .mean()
    )

    # 2. Vectorized averaging of FVA across windows
    num_cols_fva = [c for c in fva_df.select_dtypes(include="number").columns if c != "window_id"]
    avg_fva_df: pd.DataFrame = (
        fva_df.groupby("metric", observed=True, as_index=False)[num_cols_fva]
        .mean()
    )

    # 3. Dynamic Title Resolution (Explicit parameters -> config fallback -> default)
    resolved_model_name: str = (
        model_name
        if model_name is not None
        else getattr(config, "MODEL_NAME", "LGBM")
    ).upper()

    resolved_model_type: str = (
        model_type
        if model_type is not None
        else getattr(config, "MODEL_TYPE", "recursive")
    ).capitalize()

    resolved_train_years: str = str(
        train_years
        if train_years is not None
        else getattr(config, "TRAIN_YEARS", "2")
    )

    # Fully dynamic header title
    title_str: str = f"Result: {resolved_model_name} {resolved_model_type} ({resolved_train_years}-Year Training)"

    # 4. Define model lists and candidate model cleanly
    models_list: list[str] = [str(m) for m in avg_metrics_df["model"].unique()]
    
    # Standard baseline names in retail pipelines
    standard_baselines = {"seasonal_naive", "moving_average", "croston", "ema", "naive"}
    
    candidates = [m for m in models_list if m not in standard_baselines]
    candidate_model: str = candidates[0] if len(candidates) > 0 else models_list[0]

    # Mask for candidate model metrics
    cand_mask = avg_metrics_df["model"] == candidate_model

    # Candidate Accuracy metrics for Left Column
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

    # 5. Correct FVA calculation vs each specific baseline
    baseline_savings: list[dict[str, str]] = []
    baseline_models = [m for m in models_list if m != candidate_model]

    if cand_mask.any() and "total_cost" in avg_metrics_df.columns:
        cand_cost = float(avg_metrics_df.loc[cand_mask, "total_cost"].iloc[0])

        for base_name in baseline_models:
            base_mask = avg_metrics_df["model"] == base_name
            if base_mask.any():
                base_cost = float(avg_metrics_df.loc[base_mask, "total_cost"].iloc[0])
                savings_val = base_cost - cand_cost

                # True FVA % of Candidate vs this specific baseline: (Base - Candidate) / Base * 100
                fva_val = ((base_cost - cand_cost) / base_cost * 100.0) if base_cost > 0 else 0.0

                baseline_savings.append({
                    "name": base_name.replace("_", " ").title(),
                    "savings": f"{savings_val:,.2f}",
                    "fva": f"{fva_val:.2f}"
                })

    # 6. Format numeric tables cleanly for Jinja2 render
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

    # 7. Render HTML via Jinja2
    template = jinja2.Template(HTML_TEMPLATE)
    html_out: str = template.render(
        title_str=title_str,
        timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        num_windows=num_windows,
        lead_time=lead_time,
        review_period=review_period,
        policy_type=policy_type,
        winner_model_name=candidate_model.upper(),
        cand_accuracy=cand_accuracy,
        cand_cost_str=cand_cost_str,
        baseline_savings=baseline_savings,
        avg_metrics_table=avg_metrics_rendered.to_html(index=False, classes="table", border=0),
        avg_fva_table=avg_fva_df.round(2).to_html(index=False, classes="table", border=0)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_out)
    print(f"\n[INFO] Generated report averaged across {num_windows} window(s): {output_path}")