"""Walmart M5 -- forecast accuracy and replenishment cost for store CA_1.

This is a model-selection study, not a policy search: every candidate forecast is replayed
through one identical (s, S) replenishment policy, so a difference in cost is the forecast's
doing and the question on the page is which model to deploy.

One page, four sections: (1) what was compared and what it cost, beside the demand-pattern map,
(2) what each model costs, plus the selected model's cost grouped by category/department/class,
(3) one model in detail: scope, actual vs predicted, orders, ranking, review-by-review decisions,
(4) system health, provenance and the audit trail.
Everything is read from the holdout artifacts; nothing is recomputed here.
"""

import hashlib
import json
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR, TESTS_DIR = BASE_DIR / "results", BASE_DIR / "results" / "tests"
DATA_DIR = BASE_DIR / "data" / "processed"
sys.path.insert(0, str(BASE_DIR))
from config import PIPELINE_CONFIG  # noqa: E402

LEAD_TIME, REVIEW_PERIOD = PIPELINE_CONFIG.get("lead_time", 4), PIPELINE_CONFIG.get("review_period", 7)
TAU = LEAD_TIME + REVIEW_PERIOD
HOLDING_RATE, STOCKOUT_RATE = PIPELINE_CONFIG.get("holding_cost_rate", 0.2), PIPELINE_CONFIG.get("stockout_cost_rate", 1.0)
SERVICE_TARGET = STOCKOUT_RATE / (STOCKOUT_RATE + HOLDING_RATE)  # Newsvendor critical fractile p/(p+h)
Z_SCORE = NormalDist().inv_cdf(SERVICE_TARGET)  # z multiplier in the safety-stock rule
SS_RULE = "SS = z · σ_error · √τ"
REFERENCE_MODEL = "seasonal_moving_average"  # every "vs baseline" delta on the page is measured against this
LOW_VOLUME, LOOKBACKS = 20, [30, 60, 90, 180, 365]
CLASS_ORDER, COST_GROUPS = ["Smooth", "Lumpy", "Erratic", "Intermittent"], ["Demand class", "Category", "Department"]
ORDER_BAR_MS = 1000 * 3600 * 24 * 0.7  # ~70% of a day, so one bar per review

MODEL_LABELS = {"lgbm": "LightGBM (Direct)", "seasonal_naive": "Seasonal Naive", "Naive": "Seasonal Naive",
                "simple_moving_average": "Simple Moving Average", "seasonal_moving_average": "Seasonal Moving Average",
                "Moving_Average": "Moving Average (180d)", "croston": "Croston", "croston_sba": "Croston-SBA"}
MODEL_COLORS = {"lgbm": "#ff7f0e", "seasonal_naive": "#d62728", "Naive": "#d62728",
                "simple_moving_average": "#2ca02c", "seasonal_moving_average": "#17becf",
                "Moving_Average": "#2ca02c", "croston": "#9467bd", "croston_sba": "#8c564b"}
CLASS_COLORS = {"smooth": "#1f77b4", "lumpy": "#ff7f0e", "intermittent": "#2ca02c", "erratic": "#dc2626"}
GOOD, BAD, AMBER = "#16a34a", "#dc2626", "#b45309"
GROUP_COLUMNS = ["Group", "skus", "total", "vs Base $", "Saving %", "holding", "stockout", "bought", "sold", "Bias %", "WAPE %"]
GROUP_HEADERS = {"skus": "SKUs", "total": "Total $", "vs Base $": "vs Baseline $", "Saving %": "Savings %",
                 "holding": "Holding $", "stockout": "Stockout $", "bought": "Bought", "sold": "Sold"}
GROUP_FORMATS = {"Total $": "${:,.0f}", "vs Baseline $": "${:+,.0f}", "Savings %": "{:+.0f}%", "Holding $": "${:,.0f}",
                 "Stockout $": "${:,.0f}", "Bought": "{:,.0f}", "Sold": "{:,.0f}", "Bias %": "{:+.0f}%", "WAPE %": "{:,.0f}%"}
GROUPING_COLUMN = {"Demand class": "class", "Category": "cat_id", "Department": "dept_id"}
SCORECARD_COLUMNS = ["label", "total", "holding", "stockout", "fill", "WAPE %", "Bias %", "MAE/day"]
SCORECARD_HEADERS = {"label": "Model", "fill": "Fill %", "MAE/day": "MAE (units/day)"}
SCORECARD_FORMATS = {"total": "${:,.0f}", "holding": "${:,.0f}", "stockout": "${:,.0f}", "fill": "{:.1f}%",
                     "WAPE %": "{:.0f}%", "Bias %": "{:+.0f}%", "MAE/day": "{:.2f}"}

def model_label(name):
    """'lgbm' -> 'LightGBM (Direct)'; handles list-valued cells."""
    if not isinstance(name, str):
        return ", ".join(model_label(part) for part in name)
    return MODEL_LABELS.get(name, name.replace("_", " ").title())

def tint(color, alpha):
    """`#rrggbb` + alpha -> rgba(), so a dimmed marker is pre-blended instead of stacked."""
    red, green, blue = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"

def metric_row(items):
    """One `st.metric` per entry across a row; entry is (label, value[, delta, delta_color])."""
    for col, entry in zip(st.columns(len(items)), items):
        col.metric(*entry)

# ---------------------------------------------------------------- loading
def artifact_signature():
    """Fingerprint of every file `load_artifacts` reads, used as its cache key.

    Unkeyed, Streamlit caches `load_artifacts` for the whole session, so a pipeline re-run in a
    terminal would not reach the page until a hard refresh. Source files hash by content rather
    than mtime, because `git stash` rewrites mtimes without changing anything."""
    parts = []
    for pattern in ("test_preds-*.parquet", "test_inventory-*.parquet"):
        for path in sorted(TESTS_DIR.glob(pattern), reverse=True)[:1]:
            parts.append((path.name, int(path.stat().st_mtime)))
    for path in (RESULTS_DIR / "sku_demand_classes.csv", TESTS_DIR / "source_state.json"):
        parts.append((path.name, int(path.stat().st_mtime) if path.exists() else 0))
    for name in ("run_pipeline.py", "config.py"):
        source = BASE_DIR / name
        parts.append((name, hashlib.sha256(source.read_bytes()).hexdigest()[:16] if source.exists() else ""))
    return tuple(parts)

@st.cache_data
def load_artifacts(signature):
    """Every artifact the page reads, plus which run they came from."""
    pred_files = sorted(TESTS_DIR.glob("test_preds-*.parquet"), reverse=True)
    inv_files = sorted(TESTS_DIR.glob("test_inventory-*.parquet"), reverse=True)
    train_path, classes_path = DATA_DIR / "train_filtered_ca1.parquet", RESULTS_DIR / "sku_demand_classes.csv"
    train = pd.read_parquet(train_path) if train_path.exists() else pd.DataFrame()
    preds = pd.read_parquet(pred_files[0]) if pred_files else pd.DataFrame()
    inv = pd.read_parquet(inv_files[0]) if inv_files else pd.DataFrame()
    classes = pd.read_csv(classes_path) if classes_path.exists() else pd.DataFrame()
    for frame in (train, preds, inv, classes):
        if not frame.empty:
            frame["item_id"] = frame["item_id"].astype(str)
            if "date" in frame.columns:
                frame["date"] = pd.to_datetime(frame["date"])
    implied = None
    if not inv.empty:
        positive = inv[inv["on_hand"] > 0]
        if not positive.empty:
            implied = float((positive["holding_cost"] / positive["on_hand"]).median())
    # Content hashes, not mtimes: a git checkout bumps every mtime in the tree and would report
    # drift for code nobody touched. Written by `save_to_parquet`; absence stays quiet.
    manifest_path, drift = TESTS_DIR / "source_state.json", []
    if manifest_path.exists() and inv_files:
        try:
            recorded = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            recorded = {}
        for name in ("run_pipeline.py", "config.py"):
            source = BASE_DIR / name
            digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16] if source.exists() else ""
            if name in recorded and digest != recorded[name]:
                drift.append(name)
    return {"train": train, "preds": preds, "inv": inv, "classes": classes,
            "provenance": {"preds": pred_files[0].name if pred_files else None,
                           "inv": inv_files[0].name if inv_files else None,
                           "implied_rate": implied, "drift": drift}}

def item_lookup(classes, preds):
    """One row per SKU: demand class, category, department."""
    parts = []
    if not classes.empty and "class" in classes.columns:
        parts.append(classes[["item_id", "class"]].drop_duplicates("item_id"))
    for column in ("cat_id", "dept_id"):
        if not preds.empty and column in preds.columns:
            parts.append(preds[["item_id", column]].drop_duplicates("item_id"))
    if not parts:
        return pd.DataFrame()
    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on="item_id", how="outer")
    return merged.set_index("item_id")

def unit_costs(train):
    """Mean realised unit price per SKU -- the basis for valuing stock on hand."""
    if train.empty or "sell_price" not in train.columns:
        return pd.Series(dtype=float)
    return train.groupby("item_id", observed=True)["sell_price"].mean()

# ---------------------------------------------------------------- widgets
def pick_filter(label, options, state_key, label_func=None, default=None):
    """A selectbox that survives its own option list changing."""
    if not options:
        st.selectbox(label, [], key=state_key, disabled=True)
        return None
    if st.session_state.get(state_key) not in options:
        st.session_state[state_key] = default if default in options else options[0]
    return st.selectbox(label, options, key=state_key, **({"format_func": label_func} if label_func else {}))

def changed(applied_key, value):
    """Record `value` and report whether it changed since the last run."""
    previous = st.session_state.get(applied_key, value)
    st.session_state[applied_key] = value
    return previous != value

# ---------------------------------------------------------------- frame maths
def truth_model(preds):
    """Which model's rows carry usable `real_sales`, chosen by content not name.

    Only the baselines carry real ground truth -- the lgbm rows are zero-filled. Matching a
    hardcoded name is what broke this: `MODEL_LABELS` says "Naive" but the parquet says
    `seasonal_naive`, so the filter matched nothing and the accuracy line rendered empty."""
    if preds.empty or "real_sales" not in preds.columns:
        return None
    usable = preds.groupby("model", observed=True)["real_sales"].sum()
    usable = usable[usable > 0]
    return str(usable.idxmax()) if not usable.empty else None

def scoped(frame, members, model):
    """Rows for `members` under one model."""
    return frame[(frame["item_id"].isin(members)) & (frame["model"] == model)]

def daily_sum(frame, members, model, column):
    """Total `column` per day across the members under one model."""
    subset = scoped(frame, members, model)
    return subset.groupby("date", as_index=False)[column].sum().sort_values("date") if not subset.empty else pd.DataFrame()

def daily_inventory(inv, members, model):
    """Daily inventory totals summed across the members."""
    subset = scoped(inv, members, model)
    if subset.empty:
        return pd.DataFrame()
    return subset.groupby("date", as_index=False).agg(
        order_qty=("order_qty", "sum"), safety_stock=("safety_stock", "sum"), order_up_to=("order_up_to", "sum"),
        on_hand=("on_hand", "sum"), actual_sales=("actual_sales", "sum")).sort_values("date")

def history_frame(train, members, days):
    """Total actual sales per day across the members, over the lookback."""
    subset = train[train["item_id"].isin(members)]
    recent = subset[subset["date"] >= subset["date"].max() - pd.Timedelta(days=days)]
    return recent.groupby("date", as_index=False)["sales"].sum().sort_values("date") if not subset.empty else pd.DataFrame()

def window_sum(frame_by_date, column, start, end):
    """Sum one column over an inclusive date range on a date-indexed frame."""
    mask = (frame_by_date.index >= start) & (frame_by_date.index <= end)
    return float(frame_by_date.loc[mask, column].sum())

def review_hover(inv, preds, members, model, days):
    """In-transit and risk-period demand summed across the group, per review day."""
    def by_item(frame, column):
        return [g.set_index("date") for _, g in scoped(frame, members, model).groupby("item_id", observed=True)
                if column in g.columns]

    inv_frames, pred_frames = by_item(inv, "arriving_qty"), by_item(preds, "sales_pred")
    day_one, horizon = pd.Timedelta(days=1), pd.Timedelta(days=TAU - 1)
    return {day: {"in_transit": sum(window_sum(f, "arriving_qty", day + day_one,
                                              day + pd.Timedelta(days=LEAD_TIME)) for f in inv_frames),
                  "risk_demand": sum(window_sum(f, "sales_pred", day, day + horizon) for f in pred_frames)}
            for day in days}

def service_profile(subset):
    """Demand, unmet demand, fill rate and days of supply for a slice of the inventory log."""
    if subset.empty:
        return {}
    per_item = subset.groupby("item_id", observed=True).agg(
        on_hand=("on_hand", "mean"), sold=("actual_sales", "sum"),
        holding=("holding_cost", "sum"), stockout=("stockout_cost", "sum"), days=("date", "nunique"))
    sold = float(per_item["sold"].sum())
    lost = float(subset["lost_sales"].sum()) if "lost_sales" in subset.columns else 0.0
    demand = sold + lost  # `actual_sales` is what the shelf gave out, not what was asked for
    rate = (per_item["sold"] / per_item["days"]).replace(0, np.nan)
    return {"skus": len(per_item), "demand": demand, "sold": sold, "lost": lost,
            "fill": sold / demand * 100.0 if demand > 0 else 100.0,
            "dos": float((per_item["on_hand"] / rate).mean()) if rate.notna().any() else np.nan,
            "holding": float(per_item["holding"].sum()), "stockout": float(per_item["stockout"].sum())}

def working_capital(inv, unit_cost, members, model):
    """Average inventory value: mean stock on hand priced at realised unit revenue."""
    subset = scoped(inv, members, model)
    if subset.empty or unit_cost.empty:
        return float("nan")
    mean_on_hand = subset.groupby("item_id", observed=True)["on_hand"].mean()
    return float((mean_on_hand * unit_cost.reindex(mean_on_hand.index)).sum())

def evaluate_inventory_performance(inv_df, members, model, selection_label):
    """One summary row of demand, service and stock levels over the selection."""
    profile = service_profile(scoped(inv_df, members, model))
    if not profile:
        return pd.DataFrame()
    return pd.DataFrame([{"Selection Scope": selection_label, "Total SKUs": profile["skus"],
                          "Total Demand": int(profile["demand"]), "Fill Rate (%)": f"{profile['fill']:.2f}%",
                          "Days of Supply": f"{profile['dos']:.1f} days", "Holding Cost ($)": f"${profile['holding']:,.0f}",
                          "Stockout Cost ($)": f"${profile['stockout']:,.0f}"}])

# ---------------------------------------------------------------- section 1
def pct_below(now, was):
    """Signed percentage a scope costs less than its reference, or None when no reference exists."""
    return None if not was else (was - now) / was * 100.0

def render_value_cards(summary, reference_summary, inv, unit_cost, members, model):
    """The three numbers the model choice turns on, each against a named reference model.

    Every card names the model it describes: with six forecasts competing under one shared
    policy, an unlabelled "fill rate" is not a result, it is an orphan number."""
    profile = service_profile(scoped(inv, members, model))
    if not profile:
        return
    label = model_label(model)
    st.caption(f"**{label}** replayed through the same replenishment policy as every other model, over the "
               f"28-day holdout on {len(members):,} SKUs.")
    fill = f"{profile['fill']:.1f}%"
    if not reference_summary or reference_summary.get("total", 0) <= 0:
        capital = working_capital(inv, unit_cost, members, model)
        metric_row([("Total inventory cost", f"${summary['total']:,.0f}", label, "off"),
                    ("Fill rate", fill, f"{profile['lost']:,.0f} units unmet", "off"),
                    ("Working capital tied up", f"${capital:,.0f}" if pd.notna(capital) else "n/a",
                     f"{profile['dos']:.0f} days of supply", "off")])
        return
    total_gap = pct_below(summary["total"], reference_summary["total"])
    hold_gap = pct_below(summary["holding"], reference_summary["holding"])
    metric_row([("Total inventory cost", f"${summary['total']:,.0f}",
                 f"{-total_gap:+.1f}% vs {model_label(REFERENCE_MODEL)}", "normal" if total_gap > 0 else "inverse"),
                ("of which holding cost", f"${summary['holding']:,.0f}",
                 f"{-hold_gap:+.1f}% vs {model_label(REFERENCE_MODEL)}", "normal" if hold_gap > 0 else "inverse"),
                ("Fill rate", fill, f"{profile['lost']:,.0f} units unmet", "off")])

def render_business_framing(summary, reference_summary, fill_rate, n_models):
    """Section 1 brief: the design, the fixed policy, and the outcome against the reference model.

    Every figure is computed from the artifacts rather than typed into the copy, so the brief cannot
    drift away from the cards and tables rendered directly beneath it.
    """
    st.markdown("#### What Was Compared")
    st.markdown(
        f"{n_models} demand models were backtested and then evaluated blind on a 28-day holdout for store CA_1: "
        f"**{summary['skus']:,} SKUs, {summary['sold']:,.0f} units of realised demand**.\n\n"
        f"Each model produces the same {TAU}-day multi-step forecast (**{LEAD_TIME}-day lead time + "
        f"{REVIEW_PERIOD}-day review period**) and feeds into the **same fixed periodic-review inventory policy**. "
        f"This isolates the effect of the forecast itself.\n\n"
        "Models are therefore evaluated on two dimensions: **forecast accuracy** and the **downstream inventory "
        "cost generated by that forecast**.")
    st.markdown("#### The Fixed Replenishment Rule")
    st.markdown(
        f"The inventory policy is unchanged across all runs:\n\n**{SS_RULE}**\n\n"
        f"where **z = {Z_SCORE:.2f}**, corresponding to a cost-implied critical fractile of "
        f"**{SERVICE_TARGET:.1%}**, σ_error is the out-of-sample forecast error, and **τ = {TAU} days** is the "
        f"risk period.\n\n"
        f"At each review cycle, the policy replenishes:\n\n**Q = max(0, S − IP)**\n\n"
        f"based on inventory position and realised demand, with **${HOLDING_RATE:.2f}/unit/day holding cost** and "
        f"**${STOCKOUT_RATE:g} per lost unit**.\n\n"
        "The policy itself is **not optimized** in this experiment. Only the forecast changes between runs.")
    if not reference_summary or reference_summary.get("total", 0) <= 0:
        return
    st.markdown("#### The Bottom Line")
    st.markdown(
        f"Against **{model_label(REFERENCE_MODEL)}**, using the same assortment and the same inventory policy:\n\n"
        f"- **Total cost:** ${summary['total']:,.0f} vs. ${reference_summary['total']:,.0f}\n"
        f"- **Cost reduction:** **{pct_below(summary['total'], reference_summary['total']):.1f}%**\n"
        f"- **Holding-cost reduction:** **{pct_below(summary['holding'], reference_summary['holding']):.1f}%**\n"
        f"- **Fill rate:** **{fill_rate:.1f}%**\n"
        f"- **Cost composition:** **{summary['holding'] / max(summary['total'], 1e-9) * 100:.0f}% holding cost**\n\n"
        "The result shows that forecast choice can materially affect downstream inventory cost under a fixed "
        "replenishment policy. However, this is **one policy configuration**, not an optimization of the "
        "inventory policy itself.")

def class_badges(classes):
    """SKU-share and revenue-share badges per Syntetos-Boylan class."""
    counts, revs = classes["class"].value_counts(), classes.groupby("class")["rev"].sum()
    total_rev = max(float(revs.sum()), 1e-9)
    badges = []
    for cls in CLASS_ORDER:
        key = cls.lower()
        n = int(counts.get(key, 0))
        if not n:
            continue
        color = CLASS_COLORS.get(key, "#64748b")
        badges.append(f"<span style='background:{color}1f;color:{color};border:1px solid {color}55;"
                      f"border-radius:12px;padding:3px 9px;margin:0 5px 4px 0;font-size:0.82rem'>"
                      f"<b>{cls}</b> &nbsp;{n / len(classes):.0%} SKUs &nbsp;·&nbsp; "
                      f"{revs.get(key, 0) / total_rev:.0%} rev</span>")
    st.markdown("".join(badges) or "_No demand classes available._", unsafe_allow_html=True)

def render_sku_profile(classes, inv, members, model):
    """Store assortment map (ADI vs CV2) with Syntetos-Boylan demand segmentation."""
    if classes.empty:
        st.info("No assortment class data available.")
        return
    st.markdown("#### Store Assortment Map (Syntetos-Boylan Demand Classes)")
    class_badges(classes)
    focus = st.segmented_control("Focus a demand class", ["All", *CLASS_ORDER], default="All",
                                 key="quadrant_focus", label_visibility="collapsed")
    classes = classes.assign(_hit=classes["class"].astype(str).str.lower().eq(str(focus).lower())
                             if focus != "All" else True)
    fig = go.Figure(go.Scatter(
        x=classes["adi"], y=classes["cv2"], mode="markers",
        marker=dict(size=[7 + 11 * np.sqrt(r / classes["rev"].max()) for r in classes["rev"]],
                    color=[tint(CLASS_COLORS.get(str(c).lower(), "#64748b"), 0.95 if hit else 0.22)
                           for c, hit in zip(classes["class"], classes["_hit"])],
                    line=dict(width=0.7, color="#ffffff")),
        customdata=np.stack([classes["item_id"], classes["class"], classes["mean"], classes["zero_frac"],
                             classes["rev"]], axis=-1),
        hovertemplate=("<b>%{customdata[0]}</b> (%{customdata[1]})<br>ADI %{x:.2f} · CV² %{y:.2f}"
                       "<br>Mean %{customdata[2]:.2f}/day · zero days %{customdata[3]:.0%}"
                       "<br>Revenue $%{customdata[4]:,.0f}<extra></extra>"), showlegend=False))
    fig.add_vline(x=1.32, line_dash="dash", line_color="#94a3b8", line_width=1)
    fig.add_hline(y=0.49, line_dash="dash", line_color="#94a3b8", line_width=1)
    fig.update_layout(template="plotly_white", height=330, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="ADI (Average Demand Interval in days)", yaxis_title="CV² (Demand Size Variability)")
    st.plotly_chart(fig, width="stretch")
    if focus == "All":
        erratic = classes["class"].str.lower().isin(["intermittent", "erratic"])
        st.caption(f"**{erratic.mean():.0%} of SKUs** sit in the intermittent/erratic quadrants "
                   f"(ADI > 1.32 or CV² > 0.49). Focus a class above to price that slice.")
        return
    profile = service_profile(scoped(inv, classes.loc[classes["_hit"], "item_id"].tolist(), model))
    if not profile:
        st.info(f"No inventory log for {focus} SKUs under this policy model.")
        return
    st.caption(f"**{focus}** · {profile['skus']} SKUs · demand {profile['demand']:,.0f} units · holding "
               f"\\${profile['holding']:,.0f} vs stockout \\${profile['stockout']:,.0f} · fill rate "
               f"**{profile['fill']:.1f}%** · {profile['dos']:.1f} days of supply.")

# ---------------------------------------------------------------- section 2
def rollup(per_item):
    """Scalars for one cost view, from per-SKU cost rows; `total` is what the business pays."""
    per_item = per_item.copy()
    per_item["total"] = per_item["holding"] + per_item["stockout"]
    totals = {c: float(per_item[c].sum()) for c in ("bought", "sold", "lost", "holding", "stockout", "total")}
    return dict(per_item=per_item, skus=len(per_item),
                no_stockout=int((per_item["stockout"] == 0).sum()), **totals)

def summarise(inv, model):
    """Cost and service totals for one policy model across the assortment."""
    subset = inv[inv["model"] == model]
    if subset.empty:
        return {}
    per_item = subset.groupby("item_id", observed=True).agg(
        bought=("order_qty", "sum"), sold=("actual_sales", "sum"), lost=("lost_sales", "sum"),
        holding=("holding_cost", "sum"), stockout=("stockout_cost", "sum"))
    return rollup(per_item)

def group_labels(items, lookup, group_by):
    """Label every SKU with the group it falls into for the chosen comparison."""
    items, column = pd.Index(items), GROUPING_COLUMN[group_by]
    values = lookup[column].astype(str).str.strip()
    return pd.Series(items.map(values.str.title() if column == "class" else values), index=items)

def group_costs(per_item, labels, group_by):
    """Aggregate per-item cost into the group being compared."""
    if per_item.empty:
        return pd.DataFrame()
    labelled = per_item.copy()
    labelled["group"] = labels.reindex(per_item.index).fillna("Unclassified")
    grouped = labelled.groupby("group", observed=True).agg(
        skus=("total", "size"), bought=("bought", "sum"), sold=("sold", "sum"), holding=("holding", "sum"),
        stockout=("stockout", "sum"), total=("total", "sum")).reset_index().rename(columns={"group": "Group"})
    grouped["per_sku"] = grouped["total"] / grouped["skus"]
    if group_by == "Demand class":
        order = {name: i for i, name in enumerate(CLASS_ORDER)}
        grouped = grouped.assign(_o=grouped["Group"].map(order).fillna(len(order))).sort_values("_o")
    else:
        grouped = grouped.sort_values("total", ascending=False)
    return grouped.drop(columns="_o", errors="ignore").reset_index(drop=True)

def group_accuracy(preds, labels, model, truth):
    """Holdout error on the daily series summed *inside* each group.

    Scored after aggregation rather than averaged from member scores: a category's daily series
    is far smoother than any single SKU in it, so averaging member MAEs overstates the error.
    Comparable across groups, not with section 3."""
    if preds.empty or truth is None or labels.empty:
        return pd.DataFrame()

    def by_item(name, column):
        return {item: frame.set_index("date")[column] for item, frame in
                preds[preds["model"] == name].groupby("item_id", observed=True)}

    pred_by_item, truth_by_item = by_item(model, "sales_pred"), by_item(truth, "real_sales")
    rows = []
    for label, members in labels.groupby(labels).groups.items():
        usable = [i for i in members if i in pred_by_item and i in truth_by_item]
        if not usable:
            continue
        frame = pd.concat([pd.DataFrame({i: truth_by_item[i] for i in usable}).sum(axis=1).rename("real_sales"),
                           pd.DataFrame({i: pred_by_item[i] for i in usable}).sum(axis=1).rename("sales_pred")],
                          axis=1).dropna()
        actual, predicted = frame["real_sales"].to_numpy("float64"), frame["sales_pred"].to_numpy("float64")
        total = float(actual.sum())
        if frame.empty or total <= 0:
            continue
        rows.append({"Group": label, "Bias %": float((predicted.sum() - total) / total * 100),
                     "WAPE %": float(np.sum(np.abs(actual - predicted)) / total * 100),
                     "MAE/day": float(np.mean(np.abs(actual - predicted)))})
    return pd.DataFrame(rows)

def bias_highlight(value):
    """Flag forecast bias, the primary driver of over-stocking, at the ±10% level."""
    if not pd.notna(value):
        return ""
    if value > 10:
        return f"background-color:#fef3c7;color:{AMBER};font-weight:600"
    if value < -10:
        return "background-color:#dbeafe;color:#1d4ed8;font-weight:600"
    return ""

def model_scorecard(preds, inv, members, models, truth=None):
    """One row per model: what it costs under the *same* policy, and how accurate it was.

    The rule never changes between rows, so every difference in `total` is caused by the forecast
    alone. That is the whole point of a model-selection view, and it is also why the
    holding/stockout split has to stay visible: a model can win the total by holding less while
    another loses fewer sales, and those two outcomes are not the same recommendation.
    """
    truth = truth_model(preds) if truth is None else truth
    rows = []
    for name in models:
        profile = service_profile(scoped(inv, members, name))
        if not profile:
            continue
        stats = selection_accuracy(preds, members, name, truth)
        rows.append({"model": name, "label": model_label(name),
                     "total": profile["holding"] + profile["stockout"], "holding": profile["holding"],
                     "stockout": profile["stockout"], "fill": profile["fill"], "lost": profile["lost"],
                     "WAPE %": stats.get("WAPE %", np.nan), "Bias %": stats.get("Bias %", np.nan),
                     "MAE/day": stats.get("MAE/day", np.nan)})
    return pd.DataFrame(rows).sort_values("total").reset_index(drop=True) if rows else pd.DataFrame()

def scorecard_table(scorecard, reference):
    """Per-model cost and accuracy, plus the gap to the model the user picked as reference.

    The reference model's own row reads 0 by construction. That row is the baseline itself, so a
    zero there is a definitional fact rather than a missing comparison.
    """
    frame = scorecard.copy()
    if reference in set(frame["model"]):
        base = float(frame.loc[frame["model"] == reference, "total"].iloc[0])
        frame["vs Reference $"] = frame["total"] - base
        frame["Saving %"] = -frame["vs Reference $"] / base * 100 + 0.0 if base > 0 else np.nan
        columns = ["label", "total", "vs Reference $", "Saving %", "holding", "stockout", "fill",
                   "WAPE %", "Bias %", "MAE/day"]
        headers = {"label": "Model", "vs Reference $": f"vs {model_label(reference)} $",
                   "MAE/day": "MAE (units/day)"}
        formats = {**SCORECARD_FORMATS, "vs Reference $": "${:+,.0f}", "Saving %": "{:+.0f}%"}
        return frame[columns].rename(columns=headers).style.format(formats, na_rep="--") \
            .map(bias_highlight, subset=["Bias %"])
    return frame[SCORECARD_COLUMNS].rename(columns=SCORECARD_HEADERS).style.format(
        SCORECARD_FORMATS, na_rep="--").map(bias_highlight, subset=["Bias %"])

def render_model_comparison(scorecard, selected, reference=None):
    """Every model priced under one identical policy: the comparison this project is about.

    Real measured outcomes only, sorted by total cost, with the model in view picked out. The
    caption names the two different winners when they disagree, because that disagreement is the
    honest answer rather than a defect to hide.
    """
    if scorecard.empty:
        return
    ordered = scorecard.iloc[::-1]
    fig = go.Figure()
    for column, name, color in (("holding", "Holding cost", "#3b82f6"), ("stockout", "Stockout cost", "#ef4444")):
        fig.add_trace(go.Bar(name=name, y=ordered["label"], x=ordered[column], orientation="h",
                             marker_color=[tint(color, 1.0 if m in (selected, reference) else 0.38)
                                           for m in ordered["model"]],
                             customdata=np.stack([ordered["fill"], ordered["WAPE %"], ordered["MAE/day"]], axis=-1),
                             hovertemplate=f"<b>%{{y}}</b><br>{name}: $%{{x:,.0f}}<br>Fill rate "
                                           "%{customdata[0]:.1f}%<br>WAPE %{customdata[1]:.0f}% · MAE "
                                           "%{customdata[2]:.2f} units/day<extra></extra>"))
    fig.update_layout(template="plotly_white", barmode="stack", height=max(240, 38 * len(ordered) + 80),
                      margin=dict(l=10, r=30, t=10, b=10), bargap=0.32, xaxis_title="Total inventory cost ($)",
                      yaxis_title="", legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1))
    st.plotly_chart(fig, width="stretch")
    cheapest, kindest = scorecard.iloc[0], scorecard.loc[scorecard["stockout"].idxmin()]
    shown = f" {model_label(selected)} and {model_label(reference)} are at full strength." if reference else ""
    st.caption(f"Cheapest overall: **{cheapest['label']}** at ${cheapest['total']:,.0f}. "
               f"Fewest lost sales: **{kindest['label']}** at ${kindest['stockout']:,.0f} of stockout cost"
               + ("" if kindest["model"] == cheapest["model"]
                  else f", holding {kindest['holding'] / max(kindest['total'], 1):.0%} of its budget to do it")
               + f". Same policy in every row, so the spread is the forecast's doing alone.{shown}")

def render_where_money_goes(lookup, preds, inv, unit_cost, models, default_model):
    """Section 2: one chart and one table, both scoped, both able to swap what they describe.

    Two mutually exclusive views behind the breakdown filter. On "All models" the section is the
    model comparison -- the first look -- with every candidate priced in this scope. Choosing a
    breakdown turns the same two slots to that segment for the model in view. One view at a time
    is the point: the alternative was a chart and a table per view, which is four objects in one
    section and no clear answer to "which model, against what, in which slice".

    The reference is chosen by the user, never by the model in view, so a model can never end up
    compared against itself -- that case read as a table of zeros rather than as "no difference".
    """
    st.markdown("### 2. Model Breakdown")
    model_col, reference_col, group_col, low_col = st.columns([1.2, 1.2, 1.2, 1.4])
    with model_col:
        model = pick_filter("Model in view", models, "cost_model", model_label, default_model)
    choices = [m for m in models if m != model]
    with reference_col:
        reference = pick_filter("Compare against", choices, "cost_reference", model_label,
                                REFERENCE_MODEL if REFERENCE_MODEL in choices else (choices[0] if choices else None))
    with group_col:
        group_by = pick_filter("Break down by", ["All models", *COST_GROUPS], "cost_group")
    with low_col:
        low_only = st.toggle(f"Low-volume SKUs only (< {LOW_VOLUME} units/mo)", value=False, key="cost_low_volume")
    summary = summarise(inv, model)
    if not summary or not choices:
        st.info(f"No inventory log for {model_label(model)}." if summary else "Only one model in these artifacts.")
        return
    per_item = summary["per_item"]
    if low_only:
        per_item = per_item[per_item["sold"] < LOW_VOLUME]
        if per_item.empty:
            st.info(f"No SKU sold fewer than {LOW_VOLUME} units over the holdout under "
                    f"{model_label(model)}.")
            return
    members = per_item.index.tolist()
    if group_by != "All models":
        lookup = lookup.loc[lookup.index.isin(members)]
    scorecard = model_scorecard(preds, inv, members, models)
    if scorecard.empty or model not in set(scorecard["model"]):
        st.info("No cost data for this scope.")
        return
    priced = scorecard.set_index("model")
    here = priced.loc[model]
    base = priced.loc[reference] if reference in priced.index else None
    total_gap = pct_below(here["total"], base["total"]) if base is not None else None
    st.caption(f"{len(members):,} SKUs in scope" + (f" · every percentage is against "
               f"**{model_label(reference)}** in the same scope." if base is not None else "."))
    metric_row([(f"Inventory cost @ ${HOLDING_RATE:g}/unit/day", f"${here['total']:,.0f}",
                 f"{-total_gap:+.1f}% vs {model_label(reference)}" if total_gap is not None else None,
                 "normal" if total_gap and total_gap > 0 else ("inverse" if total_gap else "off")),
                ("Holding share of cost", f"{here['holding'] / max(here['total'], 1):.0%}",
                 f"${here['stockout']:,.0f} is stockout", "inverse"),
                ("Working capital tied up",
                 (lambda c: f"${c:,.0f}" if pd.notna(c) else "n/a")(working_capital(inv, unit_cost, members, model)),
                 f"{service_profile(scoped(inv, members, model)).get('dos', float('nan')):.0f} days of supply"),
                ("Units bought vs sold", f"{per_item['bought'].sum():,.0f} / {per_item['sold'].sum():,.0f}",
                 f"{here['lost']:,.0f} lost ({here['lost'] - base['lost']:+,.0f} vs reference)"
                 if base is not None else None, "inverse")])
    plot_col, table_col = st.columns([1.1, 1.6])
    if group_by == "All models":
        with plot_col:
            st.markdown(f"**Every model, same policy, {len(members):,} SKUs**")
            render_model_comparison(scorecard, model, reference)
        with table_col:
            st.markdown("**Cost and accuracy, every model**")
            st.dataframe(scorecard_table(scorecard, reference), width="stretch", hide_index=True)
            st.caption("Cost is what the same policy charged; accuracy is the holdout error on the grouped daily "
                       "series. Nothing here is a tuned policy, so read it as model selection.")
        return
    labels = group_labels(per_item.index, lookup, group_by)
    grouped = group_costs(per_item, labels, group_by)
    if grouped.empty:
        st.info("No cost data for this grouping.")
        return
    base_groups = group_costs(summarise(inv, reference)["per_item"].reindex(per_item.index), labels, group_by)
    base_total = (base_groups.set_index("Group")["total"].reindex(grouped["Group"])
                  if not base_groups.empty else None)
    grouped["vs Base $"] = grouped["total"] - base_total.values if base_total is not None else np.nan
    grouped["Saving %"] = np.where(base_total.values > 0, -grouped["vs Base $"] / base_total.values * 100, np.nan)
    accuracy = group_accuracy(preds, labels, model, truth_model(preds))
    for column in ("Bias %", "WAPE %"):
        grouped[column] = (accuracy.set_index("Group").reindex(grouped["Group"])[column].values
                           if not accuracy.empty else np.nan)
    with plot_col:
        st.markdown(f"**{group_by} · {model_label(model)}**")
        fig = go.Figure()
        for name, column, color in (("Holding Cost", "holding", "#3b82f6"),
                                    ("Stockout Cost", "stockout", "#ef4444")):
            fig.add_trace(go.Bar(name=name, y=grouped["Group"], x=grouped[column], orientation="h", marker_color=color,
                                 hovertemplate=f"<b>%{{y}}</b><br>{name.split()[0]}: $%{{x:,.0f}}<extra></extra>"))
        if base_total is not None:
            fig.add_trace(go.Scatter(y=grouped["Group"], x=base_total.values, mode="markers", name=model_label(reference),
                                     marker=dict(symbol="line-ew-open", size=13, color="#0f172a", line=dict(width=2.5)),
                                     hovertemplate="<b>%{y}</b><br>" + model_label(reference) + ": $%{x:,.0f}<extra>Reference</extra>"))
        fig.update_layout(template="plotly_white", barmode="stack", height=max(240, 60 * len(grouped) + 80),
                          margin=dict(l=10, r=30, t=10, b=10), bargap=0.35, xaxis_title="Inventory Cost Breakdown ($)",
                          yaxis_title="", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, width="stretch")
    with table_col:
        st.markdown(f"**{group_by} breakdown**")
        headers, formats = dict(GROUP_HEADERS), dict(GROUP_FORMATS)
        headers["vs Base $"] = f"vs {model_label(reference)} $"
        formats.pop("vs Baseline $"), formats.update({headers["vs Base $"]: GROUP_FORMATS["vs Baseline $"]})
        st.dataframe(grouped[GROUP_COLUMNS].rename(columns=headers).style.format(formats, na_rep="--")
                     .map(bias_highlight, subset=["Bias %"]), width="stretch", hide_index=True)
        st.caption(f"{model_label(reference)} on the same {len(members):,} SKUs. Bias is shaded past ±10%: "
                   "persistent positive bias is what inflates the order-up-to level.")

# ---------------------------------------------------------------- section 3
def render_filter_bar(lookup):
    """Scope controls on one row; the policy terms are fixed by the run and shown as read-only pills."""
    has_cat, has_dept = "cat_id" in lookup.columns, "dept_id" in lookup.columns
    scope_a, scope_b, scope_c, span_d = st.columns(4)
    with scope_a:
        options = ["All", *sorted(lookup["cat_id"].dropna().unique())] if has_cat else ["All"]
        category = pick_filter("Category", options, "g_cat")
    if changed("g_applied_cat", category):
        st.session_state.pop("g_dept", None), st.session_state.pop("g_item", None)
    scoped_rows = lookup if category == "All" or not has_cat else lookup[lookup["cat_id"] == category]
    with scope_b:
        options = ["All", *sorted(scoped_rows["dept_id"].dropna().unique())] if has_dept else ["All"]
        department = pick_filter("Department", options, "g_dept")
    if changed("g_applied_dept", department):
        st.session_state.pop("g_item", None)
    if department != "All" and has_dept:
        scoped_rows = scoped_rows[scoped_rows["dept_id"] == department]
    with scope_c:
        item = pick_filter("Item", ["All", *sorted(scoped_rows.index.tolist())], "g_item")
    with span_d:
        lookback = pick_filter("History lookback", [f"{d}d" for d in LOOKBACKS], "g_lookback")
    st.markdown(" ".join(
        f"<span style='display:inline-block;background:#f1f5f9;border:1px solid #cbd5e1;border-radius:14px;"
        f"padding:3px 11px;margin-right:6px;font-size:0.82rem;color:#334155'>{label}: <b>{value}</b></span>"
        for label, value in (("Lead Time", f"{LEAD_TIME}d"), ("Review Cycle", f"{REVIEW_PERIOD}d"),
                             ("Protection Horizon τ", f"{TAU}d"), ("Service Target", f"{SERVICE_TARGET:.1%}"),
                             ("z", f"{Z_SCORE:.2f}"), ("Safety Stock", "z·σ·√τ"))), unsafe_allow_html=True)
    members = [item] if item != "All" else scoped_rows.index.tolist()
    label = item if item != "All" else (department if department != "All" else
                                        (category if category != "All" else f"All {len(lookup):,} SKUs"))
    return members, label, int(lookback.rstrip("d"))

def line_trace(name, frame, value, color, width, hover=None, dash="solid", markers=False):
    """One forecast/actual line trace, so the figure stays readable."""
    default = "%{x|%b %d}<br>%{y:.0f} units<extra>{name}</extra>"
    return go.Scatter(x=frame["date"], y=frame[value], mode="lines+markers" if markers else "lines", name=name,
                      line=dict(color=color, width=width, dash=dash), marker=dict(size=5) if markers else None,
                      hovertemplate=(hover or default).replace("{name}", name))

def daily_order_rate(daily, lead_time, review_period):
    """Order quantity spread over the days it is actually meant to cover.

    A review order covers a whole cycle, not a day: placed on day D it lands on D + lead_time and
    must hold the store through the next `review_period` days. As one bar on a daily axis it sits
    ~11x above a typical day of sales and reads as catastrophic over-buying when the cycle only
    carries ~1.6x what it sells."""
    rates = pd.Series(0.0, index=pd.DatetimeIndex(daily["date"]))
    for day, qty in zip(daily["date"], daily["order_qty"]):
        if qty <= 0:
            continue
        for offset in range(review_period):
            covered = day + pd.Timedelta(days=lead_time + offset)
            if covered in rates.index:
                rates.loc[covered] += qty / review_period
    return rates.reset_index(drop=True)

def shade_stockouts(fig, daily):
    """Red bands over every run of days the shelf was empty, so stockouts are unmissable."""
    empty, dates, start = (daily["on_hand"] <= 0).to_numpy(), daily["date"].tolist(), None
    for i, flag in enumerate(list(empty) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            fig.add_vrect(x0=dates[start], x1=dates[i - 1], fillcolor=BAD, opacity=0.16, layer="below", line_width=0)
            fig.add_annotation(x=dates[start], y=1, yref="paper", yanchor="bottom", showarrow=False,
                               text="stockout", font=dict(size=9, color=BAD))
            start = None

def build_inventory_figure(inv, members, policy_model):
    """On-hand sawtooth, safety stock, daily sales bars, stockout bands."""
    fig, daily = go.Figure(), daily_inventory(inv, members, policy_model)
    if daily.empty:
        return fig
    fig.add_trace(go.Bar(x=daily["date"], y=daily["actual_sales"], name="Daily Sales",
                         marker_color="rgba(245, 158, 11, 0.35)",
                         hovertemplate="%{x|%b %d}<br>%{y:.0f} units sold<extra>Sales</extra>"))
    for name, column, color, width, dash, hover in (
            ("On-Hand Inventory", "on_hand", "#16a34a", 2.5, "solid",
             "%{x|%b %d}<br>%{y:.0f} units on-hand<extra>On-Hand</extra>"),
            ("Safety Stock Target", "safety_stock", "#dc2626", 1.5, "dot",
             "%{x|%b %d}<br>%{y:.0f} units SS<extra>Safety Stock</extra>")):
        fig.add_trace(go.Scatter(x=daily["date"], y=daily[column], mode="lines+markers" if width > 2 else "lines",
                                 name=name, line=dict(color=color, width=width, dash=dash), marker=dict(size=4),
                                 hovertemplate=hover))
    shade_stockouts(fig, daily)
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=25, b=10), yaxis_title="Units",
                      xaxis_title="Date", barmode="overlay", hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig

def build_detail_figure(train, preds, inv, members, policy_model, compare, days):
    """Actual vs predicted for the selection, with the orders the policy placed."""
    fig, history = go.Figure(), history_frame(train, members, days)
    if not history.empty:
        fig.add_trace(line_trace("Actual", history, "sales", "#94a3b8", 1.5))
    for model in [policy_model, *compare]:
        frame = daily_sum(preds, members, model, "sales_pred")
        primary = model == policy_model
        if not frame.empty:
            fig.add_trace(line_trace(f"{model_label(model)} predicted", frame, "sales_pred",
                                     MODEL_COLORS.get(model, "#0f172a"), 2.5 if primary else 1.8,
                                     "%{x|%b %d}<br>%{y:.1f} units predicted<extra>" + model_label(model) + "</extra>",
                                     dash="solid" if primary else "dash"))
    daily = daily_inventory(inv, members, policy_model)
    if daily.empty:
        return fig
    fig.add_trace(line_trace("Realised in holdout", daily, "actual_sales", "#0f1720", 2.5, markers=True))
    days_on = sorted(daily["date"])[::REVIEW_PERIOD]
    ordered = daily[daily["date"].isin(set(days_on))].copy()
    hover = review_hover(inv, preds, members, policy_model, days_on)
    ordered["in_transit"] = [hover[d]["in_transit"] for d in ordered["date"]]
    ordered["risk_demand"] = [hover[d]["risk_demand"] for d in ordered["date"]]
    ordered["arrival"] = ordered["date"] + pd.Timedelta(days=LEAD_TIME)
    custom = ["order_qty", "safety_stock", "order_up_to", "on_hand", "in_transit", "risk_demand", "arrival"]
    fig.add_trace(go.Bar(x=ordered["date"], y=ordered["order_qty"], name="Order placed (whole cycle)",
                         marker_color="rgba(99,102,241,0.45)", marker_line=dict(color="#6366f1", width=1),
                         width=ORDER_BAR_MS, customdata=ordered[custom].to_numpy(),
                         hovertemplate=("<b>Review %{x|%b %d}</b><br>Order <b>%{customdata[0]:.1f}</b> units"
                                        "<br>safety stock %{customdata[1]:.1f} · S %{customdata[2]:.1f}"
                                        "<br>on hand %{customdata[3]:.1f} · in transit %{customdata[4]:.1f}"
                                        f"<br>forecast ({TAU}d) %{{customdata[5]:.1f}}"
                                        "<br>arrives %{customdata[6]|%b %d}<extra></extra>")))
    rate_frame = daily.copy()
    rate_frame["order_rate"] = daily_order_rate(daily, LEAD_TIME, REVIEW_PERIOD)
    fig.add_trace(line_trace("Order per day (cycle spread)", rate_frame, "order_rate", "#6366f1", 2,
                             "%{x|%b %d}<br>%{y:.0f} units/day<extra>Order per day</extra>", dash="dot"))
    shade_stockouts(fig, daily)
    fig.update_layout(template="plotly_white", height=470, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="Units per day", xaxis_title="Date", barmode="overlay", hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0))
    return fig

def selection_accuracy(preds, members, model, truth):
    """Bias and WAPE for one model over the whole selection, as a single synthetic group.

    Reuses `group_accuracy` instead of a second implementation: same aggregation, one code path."""
    if not truth:
        return {}
    scored = group_accuracy(preds, pd.Series("Selection", index=pd.Index(members)), model, truth)
    if scored.empty:
        return {}
    return scored[["Bias %", "WAPE %", "MAE/day"]].iloc[0].astype(float).to_dict()

def render_model_ranking(scorecard):
    """Rank the compared models by the cost they incurred under one identical policy.

    Cost leads because that is what the business pays; accuracy is the supporting detail, and
    the holding/stockout split is kept in the line so a reader can see *why* a model won. Where
    the cheapest model is not the one that lost fewest sales, the caption says so rather than
    implying the ranking settles a service question it cannot.
    """
    if scorecard.empty:
        st.info("No simulated policy cost for this selection.")
        return
    costs = scorecard.sort_values("total")
    best, cheapest = costs["model"].iloc[0], float(costs["total"].iloc[0])
    left, right = st.columns([1.6, 1])
    with left:
        for _, row in costs.iterrows():
            stats = (f" · WAPE **{row['WAPE %']:,.0f}%** · bias **{row['Bias %']:+,.0f}%**"
                     f" · MAE **{row['MAE/day']:.2f}**/day" if pd.notna(row["WAPE %"]) else "")
            st.markdown(f"**{row['label']}** — total **\\${row['total']:,.0f}** "
                        f"(holding \\${row['holding']:,.0f} + stockout \\${row['stockout']:,.0f})" + stats)
        kindest = costs.loc[costs["stockout"].idxmin()]
        st.caption(f"Ranked by total replenishment cost over the holdout; best-to-worst spread is "
                   f"{(costs['total'].iloc[-1] - cheapest) / cheapest * 100:.0f}%. Identical policy in every row. "
                   + (f"On stockout cost alone {kindest['label']} ranks first at ${kindest['stockout']:,.0f} -- the "
                      "buffer is the lever that trades one against the other, and it was not tuned."
                      if kindest["model"] != best else ""))
    with right:
        chips = [f"<span style='color:{GOOD if m == best else BAD};font-weight:700'>"
                 f"{'✓' if m == best else '✗'} {model_label(m)}</span>" for m in costs["model"]]
        st.markdown(f"<div style='text-align:right;padding-top:0.4rem'><span style='color:#64748b;font-size:0.8rem'>"
                    f"lowest cost under the same policy</span><br>{'<br>'.join(chips)}</div>",
                    unsafe_allow_html=True)

def render_financial_card(inv, unit_cost, members, model, label):
    """What the selection is worth in dollars: unit value, capital, carrying charge, service."""
    profile = service_profile(scoped(inv, members, model))
    if not profile:
        return
    capital = working_capital(inv, unit_cost, members, model)
    price = unit_cost.reindex(members).mean() if len(unit_cost) and len(members) else float("nan")
    st.markdown(f"#### Financial Summary — {label}")
    metric_row([("Avg Unit Price", f"${price:,.2f}" if pd.notna(price) else "n/a", "realised"),
                ("Holding Cost Incurred", f"${profile['holding']:,.0f}", f"over {profile['skus']} SKU(s)"),
                ("Fill Rate Achieved", f"{profile['fill']:.1f}%", f"{profile['lost']:,.0f} units unmet", "inverse"),
                ("Days of Supply", f"{profile['dos']:.1f}" if pd.notna(profile['dos']) else "n/a",
                 f"${capital:,.0f} capital at cost" if pd.notna(capital) else "avg on-hand / sell-through")])

def render_review_table(preds, inv, members, policy_model):
    """What the policy decided on each review day, plus an over-buy warning."""
    st.markdown("**What the policy decided, review by review**")
    daily = daily_inventory(inv, members, policy_model)
    if daily.empty:
        st.info("No replenishment log for this selection.")
        return
    days_on, indexed = sorted(daily["date"])[::REVIEW_PERIOD], daily.set_index("date")
    hover, rows = review_hover(inv, preds, members, policy_model, days_on), []
    for day in days_on:
        if day not in indexed.index:
            continue
        record, target = indexed.loc[day], float(indexed.loc[day, "order_up_to"])
        end = day + pd.Timedelta(days=REVIEW_PERIOD - 1)
        rows.append({"Review": day.strftime("%b %d"), f"Forecast ({TAU}d)": hover[day]["risk_demand"],
                     "Safety stock": float(record["safety_stock"]), "Order-up-to (S)": target,
                     "On hand": float(record["on_hand"]), "In transit": hover[day]["in_transit"],
                     "Order placed": float(record["order_qty"]),
                     "Arrives": (day + pd.Timedelta(days=LEAD_TIME)).strftime("%b %d"),
                     "Sold in cycle": float(indexed.loc[(indexed.index >= day) & (indexed.index <= end),
                                                        "actual_sales"].sum()),
                     "Buffer Ratio": float(record["safety_stock"]) / target if target > 0 else np.nan})
    if not rows:
        st.info("No replenishment log for this selection.")
        return
    total = {column: np.nan for column in rows[0]}
    total.update(Review="Total", **{"Order placed": float(daily["order_qty"].sum()),
                                   "Sold in cycle": float(daily["actual_sales"].sum())})
    st.dataframe(pd.concat([pd.DataFrame(rows), pd.DataFrame([total])]).round(2), width="stretch", hide_index=True)
    bought, sold = total["Order placed"], total["Sold in cycle"]
    if sold > 0 and bought > 2 * sold:
        st.warning(f"Bought {bought:,.0f} units, sold {sold:,.0f}. Safety stock is carrying the order-up-to "
                   f"level; check the holding rate and the seed window.")

def render_sku_detail(train, preds, inv, lookup, unit_cost, available):
    """Section 3: model pickers, scope, the aggregated plot, cost ranking, review table."""
    st.markdown("### 3. One Model in Detail")
    policy_col, compare_col = st.columns(2)
    with policy_col:
        policy_model = st.selectbox("Model driving the simulation", available, format_func=model_label,
                                    key="policy_model")
    with compare_col:
        compare = st.multiselect("Compare against these models", [m for m in available if m != policy_model],
                                 format_func=model_label, key="compare_models")
    members, label, days = render_filter_bar(lookup)
    if st.toggle("Show Inventory Sawtooth & Performance Metrics", value=False, key="toggle_inventory_view"):
        st.plotly_chart(build_inventory_figure(inv, members, policy_model), width="stretch")
        st.markdown(f"#### Aggregated Inventory Performance (`{label}`)")
        performance = evaluate_inventory_performance(inv, members, policy_model, label)
        if performance.empty:
            st.info("No inventory performance metrics for this selection.")
        else:
            st.dataframe(performance, width="stretch", hide_index=True)
    else:
        st.caption(f"Showing **{label}** ({len(members):,} SKUs). Red bands are days the shelf was empty.")
        st.plotly_chart(build_detail_figure(train, preds, inv, members, policy_model, compare, days), width="stretch")
        st.caption("Order bars are a whole review cycle; the dotted line spreads that same order across the "
                   "days it covers, the only one of the two comparable to a single day of sales.")
        render_model_ranking(model_scorecard(preds, inv, members, [policy_model, *compare]))
    render_financial_card(inv, unit_cost, members, policy_model, label)
    st.divider()
    render_review_table(preds, inv, members, policy_model)

# ---------------------------------------------------------------- notes
def render_notes(provenance, holding_share):
    """System health, policy mechanics, and the audit trail, behind one disclosure."""
    with st.expander("🛠️ System Health, Provenance & Audit Trail", expanded=False):
        implied, has_issue = provenance["implied_rate"], False
        if implied is not None and abs(implied - HOLDING_RATE) > 1e-9:
            has_issue = True
            st.error(f"**Cost configuration mismatch.** `config.py` sets `holding_cost_rate = {HOLDING_RATE}`, "
                     f"but these artifacts were generated under `{implied:.4f}` (recovered from holding_cost / "
                     f"on_hand). Regenerate the holdout before quoting these figures.")
        if provenance["drift"]:
            has_issue = True
            st.warning("**Stale artifacts.** `" + "`, `".join(provenance["drift"]) + "` changed after this holdout "
                       "was produced. Regenerate to restore a consistent run.")
        if not has_issue:
            st.success("**Provenance verified.** Predictions, inventory simulation and cost parameters are "
                       "mutually consistent.")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### Replenishment Policy Formulation")
            st.markdown("Dynamic order-up-to levels:\n\n$$S = \\hat{D}_{\\tau} + SS$$\n\n"
                        f"- **Protection Horizon (τ)**: L + R = {LEAD_TIME} + {REVIEW_PERIOD} = {TAU} days -- supplier "
                        "lead time (L) plus review interval (R).\n"
                        "- **Forecast Lead Demand (D̂τ)**: multi-step predictions summed over [t, t+τ−1].\n"
                        f"- **Safety Stock (SS)**: `{SS_RULE}`, with z = {Z_SCORE:.2f} from the cost-implied "
                        f"critical fractile p/(p+h) = {STOCKOUT_RATE:g}/({STOCKOUT_RATE:g}+{HOLDING_RATE:g}) = "
                        f"{SERVICE_TARGET:.1%}, σ_error the out-of-sample forecast error, and τ = {TAU} days.\n"
                        "- **Order Placed (Q)**: Q = max(0, S − IP), where IP = On-Hand + In-Transit.")
        with col_m2:
            st.markdown("#### Known Limitation & The Step Not Yet Taken")
            st.markdown(f"- **What this run does not claim**: the replenishment policy was held fixed and identical for "
                        f"every model. No search was run over service targets, buffer sizes or cost rates, so the "
                        f"numbers rank forecasts under one policy -- they are not the cost of a tuned policy. At "
                        f"{HOLDING_RATE:g}/unit/day against a {STOCKOUT_RATE:g}/lost unit, holding dominates the bill: "
                        f"**{holding_share:.0f}% of the cost this run charged is carrying stock**, not failing to sell "
                        f"it.\n"
                        f"- **The untested lever**: safety stock is the only dial that was never turned. Recalibrating "
                        f"σ_error as a rolling empirical error (out-of-sample only) and re-running the holdout at 95% "
                        f"and 99% service, then comparing the Buffer Ratio and holding-cost columns in section 3, would "
                        f"show whether the current buffer is a defensible risk choice or simply an unexamined "
                        f"inheritance. That is a re-run of the same policy, not a model rebuild.")
        st.divider()
        st.markdown("#### Lineage & Artifact Audit")
        st.caption(f"Active Artifacts: `{provenance['preds']}` · `{provenance['inv']}` | Assumptions: holding rate "
                   f"\\${HOLDING_RATE:g}/unit/day · stockout penalty \\${STOCKOUT_RATE:g}/lost unit | Evaluation: "
                   "28-day out-of-sample holdout across 300 curated SKUs in Walmart M5 Store CA_1.")

# ---------------------------------------------------------------- page
def main():
    st.set_page_config(page_title="Retail Demand Forecasting & Inventory Optimization", layout="wide")
    st.markdown("<style>.block-container{padding-top:2rem}[data-testid='stMetricValue']{font-size:1.4rem}</style>",
                unsafe_allow_html=True)
    art = load_artifacts(artifact_signature())
    train, preds, inv, classes, provenance = (art[k] for k in ("train", "preds", "inv", "classes", "provenance"))
    if preds.empty or inv.empty:
        st.error("No holdout artifacts in `results/tests/`. Remove the `quit()` in `run_pipeline.py` and re-run the "
                 "test mode.")
        return
    st.title("Retail Demand Forecasting & Inventory Optimization")
    st.markdown("Turning demand forecasts into replenishment decisions and inventory-cost trade-offs")
    st.markdown("This dashboard evaluates demand forecasting models not only by forecast accuracy, but by their "
                "downstream impact on safety stock, replenishment decisions, holding costs, and stockout costs.")
    available = sorted(preds["model"].dropna().unique().tolist())
    inv_models = [m for m in available if m in set(inv["model"].dropna())] or available
    st.session_state.setdefault("policy_model", "lgbm" if "lgbm" in available else (available[0] if available else None))
    st.session_state.setdefault("compare_models", [])
    st.session_state.setdefault("g_lookback", f"{LOOKBACKS[2]}d")
    policy_model, lookup = st.session_state["policy_model"], item_lookup(classes, preds)
    summary = summarise(inv, policy_model)
    if not summary:
        st.info("No inventory log for that policy model.")
        return
    unit_cost = unit_costs(train)
    baseline_name = REFERENCE_MODEL if REFERENCE_MODEL in inv["model"].values else None
    baseline_summary = summarise(inv, baseline_name) if baseline_name and baseline_name != policy_model else None
    members = lookup.index.tolist()
    render_value_cards(summary, baseline_summary, inv, unit_cost, members, policy_model)
    st.divider()
    st.markdown("### 1. What Was Compared, and What It Cost")
    framing, profile = st.columns([1.1, 1.3])
    with framing:
        render_business_framing(summary, baseline_summary,
                                service_profile(scoped(inv, members, policy_model)).get("fill", 100.0),
                                len(available))
    with profile:
        render_sku_profile(classes, inv, members, policy_model)
    st.divider()
    render_where_money_goes(lookup, preds, inv, unit_cost, inv_models, policy_model)
    st.divider()
    render_sku_detail(train, preds, inv, lookup, unit_cost, available)
    st.divider()
    render_notes(provenance, 100 * summary["holding"] / max(summary["total"], 1e-9))

if __name__ == "__main__":
    main()
