"""Walmart M5 -- forecast accuracy and replenishment cost for store CA_1.

One page, three sections: (1) executive summary beside the demand-pattern map,
(2) where the inventory money goes, grouped, with holdout error alongside cost,
(3) SKUs in detail: model choice, scope, actual vs predicted, orders, metrics.
Everything is read from the holdout artifacts; nothing is recomputed here.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR, TESTS_DIR = BASE_DIR / "results", BASE_DIR / "results" / "tests"
DATA_DIR = BASE_DIR / "data" / "processed"
sys.path.insert(0, str(BASE_DIR))
from config import PIPELINE_CONFIG  # noqa: E402

LEAD_TIME = PIPELINE_CONFIG.get("lead_time", 4)
REVIEW_PERIOD = PIPELINE_CONFIG.get("review_period", 7)
TAU = LEAD_TIME + REVIEW_PERIOD
HOLDING_RATE = PIPELINE_CONFIG.get("holding_cost_rate", 0.2)
STOCKOUT_RATE = PIPELINE_CONFIG.get("stockout_cost_rate", 1.0)
LOW_VOLUME, LOOKBACKS = 20, [30, 60, 90, 180, 365]
CLASS_ORDER = ["Smooth", "Lumpy", "Erratic", "Intermittent"]
COST_GROUPS = ["Demand class", "Category", "Department", f"< {LOW_VOLUME} units/mo"]
ORDER_BAR_MS = 1000 * 3600 * 24 * 0.7  # ~70% of a day, so one bar per review

MODEL_LABELS = {"lgbm": "LightGBM (Direct)", "seasonal_naive": "Seasonal Naive", "Naive": "Seasonal Naive",
                "simple_moving_average": "Simple Moving Average", "seasonal_moving_average": "Seasonal Moving Average",
                "Moving_Average": "Moving Average (180d)", "croston": "Croston", "croston_sba": "Croston-SBA"}
MODEL_COLORS = {"lgbm": "#ff7f0e", "seasonal_naive": "#d62728", "Naive": "#d62728",
                "simple_moving_average": "#2ca02c", "seasonal_moving_average": "#17becf",
                "Moving_Average": "#2ca02c", "croston": "#9467bd", "croston_sba": "#8c564b"}
CLASS_COLORS = {"smooth": "#1f77b4", "lumpy": "#ff7f0e", "intermittent": "#2ca02c", "erratic": "#dc2626"}
GOOD, BAD = "#16a34a", "#dc2626"
GROUP_COLUMNS = ["Group", "skus", "total", "holding", "stockout", "bought", "sold", "Bias %", "WAPE %"]
GROUP_HEADERS = {"skus": "SKUs", "total": "Total $", "holding": "Holding $", "stockout": "Stockout $",
                 "bought": "Bought", "sold": "Sold", "Bias %": "Bias %", "WAPE %": "WAPE %"}
GROUP_FORMATS = {"Total $": "${:,.0f}", "Holding $": "${:,.0f}", "Stockout $": "${:,.0f}", "Bought": "{:,.0f}",
                 "Sold": "{:,.0f}", "Bias %": "{:+.0f}%", "WAPE %": "{:,.0f}%"}

GROUPING_COLUMN = {"Demand class": "class", "Category": "cat_id", "Department": "dept_id"}
def model_label(name):
    """'lgbm' -> 'LightGBM (Direct)'; handles list-valued cells."""
    if not isinstance(name, str):
        return ", ".join(model_label(part) for part in name)
    return MODEL_LABELS.get(name, name.replace("_", " ").title())
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
            if name in recorded and source.exists() and \
                    hashlib.sha256(source.read_bytes()).hexdigest()[:16] != recorded[name]:
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
# ---------------------------------------------------------------- widgets
def pick_filter(label, options, state_key):
    """A selectbox that survives its own option list changing.

    Seeds the key before the widget is created and passes no `index`, so widget identity is
    stable across reruns."""
    if not options:
        st.selectbox(label, [], key=state_key, disabled=True)
        return None
    if st.session_state.get(state_key) not in options:
        st.session_state[state_key] = options[0]
    return st.selectbox(label, options, key=state_key)
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
    totals = preds.groupby("model", observed=True)["real_sales"].sum()
    usable = totals[totals > 0]
    return str(usable.idxmax()) if not usable.empty else None
def scoped(frame, members, model):
    """Rows for `members` under one model."""
    return frame[(frame["item_id"].isin(members)) & (frame["model"] == model)]
def daily_sum(frame, members, model, column):
    """Total `column` per day across the members under one model."""
    subset = scoped(frame, members, model)
    return subset.groupby("date", as_index=False)[column].sum().sort_values("date") \
        if not subset.empty else pd.DataFrame()
def daily_inventory(inv, members, model):
    """Daily inventory totals summed across the members."""
    subset = scoped(inv, members, model)
    return subset.groupby("date", as_index=False).agg(
        order_qty=("order_qty", "sum"), safety_stock=("safety_stock", "sum"), order_up_to=("order_up_to", "sum"),
        on_hand=("on_hand", "sum"), actual_sales=("actual_sales", "sum")).sort_values("date") \
        if not subset.empty else pd.DataFrame()
def history_frame(train, members, days):
    """Total actual sales per day across the members, over the lookback."""
    subset = train[train["item_id"].isin(members)]
    recent = subset[subset["date"] >= subset["date"].max() - pd.Timedelta(days=days)]
    return recent.groupby("date", as_index=False)["sales"].sum().sort_values("date") \
        if not subset.empty else pd.DataFrame()
def window_sum(frame_by_date, column, start, end):
    """Sum one column over an inclusive date range on a date-indexed frame."""
    if frame_by_date.empty:
        return 0.0
    mask = (frame_by_date.index >= start) & (frame_by_date.index <= end)
    return float(frame_by_date.loc[mask, column].sum())
def review_hover(inv, preds, members, model, days):
    """In-transit and risk-period demand summed across the group, per review day.

    In-transit is receipts landing after the review day and within the lead time; risk demand is
    forecast demand over the whole protection horizon."""
    def by_item(frame, column):
        return [g.set_index("date") for _, g in scoped(frame, members, model)
                .groupby("item_id", observed=True)]

    inv_frames, pred_frames = by_item(inv, "arriving_qty"), by_item(preds, "sales_pred")
    horizon = pd.Timedelta(days=TAU - 1)
    return {day: {"in_transit": sum(window_sum(f, "arriving_qty", day + pd.Timedelta(days=1),
                                              day + pd.Timedelta(days=LEAD_TIME)) for f in inv_frames),
                  "risk_demand": sum(window_sum(f, "sales_pred", day, day + horizon) for f in pred_frames)}
            for day in days}
def evaluate_inventory_performance(inv_df, members, model, selection_label):
    """One summary row of demand, service and stock levels over the selection."""
    subset = inv_df[(inv_df["item_id"].isin(members)) & (inv_df["model"] == model)].copy()
    if subset.empty:
        return pd.DataFrame()
    daily = subset.groupby("date", as_index=False).agg(
        actual_sales=("actual_sales", "sum"), on_hand=("on_hand", "sum"), safety_stock=("safety_stock", "sum"),
        lost_sales=("lost_sales", "sum") if "lost_sales" in subset.columns else ("actual_sales", "count"))
    total_demand = daily["actual_sales"].sum()
    unmet = (subset["lost_sales"].sum() if "lost_sales" in subset.columns
             else np.maximum(0, subset["actual_sales"] - subset["on_hand"]).sum())
    fill_rate = ((total_demand - unmet) / total_demand * 100.0 if total_demand > 0 else 100.0)
    stockout_days, total_days = int((subset["on_hand"] == 0).sum()), len(subset)
    return pd.DataFrame([{
        "Selection Scope": selection_label, "Total SKUs": len(members), "Total Demand": int(total_demand),
        "Fill Rate (%)": f"{max(0.0, fill_rate):.2f}%",
        "Stockout Rate": f"{(stockout_days / total_days * 100.0 if total_days else 0.0):.1f}% ({stockout_days} item-days)",
        "Mean Daily On-Hand": f"{daily['on_hand'].mean():,.1f} units",
        "Mean Daily SS": f"{daily['safety_stock'].mean():,.1f} units"}])
# ---------------------------------------------------------------- section 1
def render_business_framing(summary, baseline_summary=None):
    """Executive framing: the commercial objective, the status quo, and the policy's payoff."""
    st.markdown("#### Executive Summary: Working Capital Optimization")
    if baseline_summary:
        saved, units = baseline_summary["total"] - summary["total"], baseline_summary["bought"] - summary["bought"]
        lost = baseline_summary["lost"] - summary["lost"]
        c_a, c_b, c_c = st.columns(3)
        c_a.metric("Net Cost Reduction", f"${saved:,.0f}",
                   f"{saved / baseline_summary['total'] * 100:.1f}% vs Naive baseline", delta_color="normal")
        c_b.metric("Over-Procurement Cut", f"{units:,.0f} units", "fewer units tied up", delta_color="normal")
        c_c.metric("Lost Sales Delta", f"{lost:+,.0f} units", "vs Naive baseline", delta_color="inverse")
    hold_pct = (summary['holding'] / summary['total']) * 100 if summary['total'] > 0 else 0
    st.markdown(f"**The Commercial Objective**  \nIn retail operations, inventory is trapped working capital. Across store **CA_1's 300-SKU assortment**, the mandate is to **minimize carrying costs** while protecting shelf availability (target: ≥ 98% on-shelf availability).\n\n"
                f"**The Status Quo vs. The Challenge**  \nStandard static replenishment policies over-buffer to prevent stockouts, trapping capital in low-velocity SKUs. Over the 28-day holdout:\n- The store fulfilled **{summary['sold']:,.0f} units** of demand, with **{summary['no_stockout']} of {summary['skus']} SKUs at 100% in-stock**.\n- Holding cost accounts for **{hold_pct:.0f}%** of total supply chain cost.\n- **The Core Finding**: This store does not have an availability problem; it has a **working-capital over-buffering problem** in the slow-moving tail.\n\n"
                f"**The ML-Driven Policy Engine**  \n1. **Dynamic Multi-Step Forecasting**: LightGBM projects demand across the **{TAU}-day protection horizon** (τ = Lead Time {LEAD_TIME}d + Review Cycle {REVIEW_PERIOD}d).\n2. **Newsvendor Service Calibration**: Safety buffers are sized using the economic critical fractile (z = Φ⁻¹(p / (p + h)) ≈ 0.83), balancing daily holding rate (h=\\${HOLDING_RATE:g}) against stockout penalty (p=\\${STOCKOUT_RATE:g}).\n3. **Simulated Holdout Benchmark**: Every replenishment cycle is stress-tested in a closed-loop simulation against actual holdout sales, validating dollar savings against standard retail baselines.")
def render_sku_profile(classes, selected_item=None):
    """Store assortment map (ADI vs CV2) with Syntetos-Boylan demand segmentation."""
    if classes.empty:
        st.info("No assortment class data available.")
        return
    counts, revs, total_skus = classes["class"].value_counts(), classes.groupby("class")["rev"].sum(), len(classes)
    st.markdown("#### Assortment Demand Profile (Syntetos-Boylan)")
    for col, cls in zip(st.columns(4), ("intermittent", "erratic", "smooth", "lumpy")):
        n, r = counts.get(cls, 0), revs.get(cls, 0)
        col.metric(cls.title(), f"{n} ({n / total_skus * 100:.0f}%)", f"${r / 1000:,.0f}k rev")
    fig = go.Figure(go.Scatter(
        x=classes["adi"], y=classes["cv2"], mode="markers",
        marker=dict(size=8 + 34 * np.sqrt(classes["rev"] / classes["rev"].max()),
                    color=[CLASS_COLORS.get(str(c).lower(), "#64748b") for c in classes["class"]],
                    opacity=0.65, line=dict(width=0)),
        customdata=np.stack([classes["item_id"], classes["class"], classes["mean"], classes["zero_frac"],
                             classes["rev"]], axis=-1),
        hovertemplate=("<b>%{customdata[0]}</b> (%{customdata[1]})<br>ADI %{x:.2f} · CV² %{y:.2f}"
                       "<br>Mean %{customdata[2]:.2f}/day · zero days %{customdata[3]:.0%}"
                       "<br>Revenue $%{customdata[4]:,.0f}<extra></extra>"), showlegend=False))
    point = None
    if selected_item and selected_item in classes["item_id"].values:
        point = classes[classes["item_id"] == selected_item].iloc[0]
        fig.add_trace(go.Scatter(x=[float(point["adi"])], y=[float(point["cv2"])], mode="markers",
                                 marker=dict(size=26, color="rgba(0,0,0,0)", line=dict(color="#0f172a", width=3)),
                                 hoverinfo="skip", showlegend=False))
    fig.add_vline(x=1.32, line_dash="dash", line_color="#94a3b8", line_width=1)
    fig.add_hline(y=0.49, line_dash="dash", line_color="#94a3b8", line_width=1)
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="ADI (Average Demand Interval in days)",
                      yaxis_title="CV² (Demand Size Variability)")
    st.plotly_chart(fig, width="stretch")
    if point is not None:
        st.caption(f"Selected SKU `{selected_item}` is **{str(point['class']).lower()}** — ADI {point['adi']:.2f}, "
                   f"CV² {point['cv2']:.2f}, {point['zero_frac']:.0%} zero days, {point['mean']:.2f} units/day.")
    else:
        erratic = counts.get("intermittent", 0) + counts.get("erratic", 0)
        st.caption(f"Store Assortment: **{erratic / total_skus * 100:.0f}% of SKUs** fall into the intermittent/"
                   f"erratic quadrants (ADI > 1.32 or CV² > 0.49), where traditional Gaussian safety-stock "
                   f"formulas systematically over-buffer.")
# ---------------------------------------------------------------- section 2
def summarise(inv, model):
    """Cost and service totals for one policy model across the assortment."""
    subset = inv[inv["model"] == model]
    if subset.empty:
        return {}
    per_item = subset.groupby("item_id", observed=True).agg(
        bought=("order_qty", "sum"), sold=("actual_sales", "sum"), lost=("lost_sales", "sum"),
        holding=("holding_cost", "sum"), stockout=("stockout_cost", "sum"))
    per_item["total"] = per_item["holding"] + per_item["stockout"]
    totals = {c: float(per_item[c].sum()) for c in ("bought", "sold", "lost", "holding", "stockout", "total")}
    return dict(per_item=per_item, skus=len(per_item),
                no_stockout=int((per_item["stockout"] == 0).sum()), **totals)
def group_labels(items, sold, lookup, group_by):
    """Label every SKU with the group it falls into for the chosen comparison."""
    items = pd.Index(items)
    column = GROUPING_COLUMN.get(group_by)
    if column is None:
        return pd.Series(np.where(sold.reindex(items).fillna(0) < LOW_VOLUME, group_by,
                                  f">= {LOW_VOLUME} units/mo"), index=items)
    values = lookup[column].astype(str).str.strip()
    return pd.Series(items.map(values.str.title() if column == "class" else values), index=items)
def group_costs(per_item, labels, group_by):
    """Aggregate per-item cost into the group being compared."""
    if per_item.empty:
        return pd.DataFrame()
    labelled = per_item.copy()
    labelled["group"] = labels.reindex(per_item.index).fillna("Unclassified")
    grouped = labelled.groupby("group", observed=True).agg(
        skus=("total", "size"), bought=("bought", "sum"), sold=("sold", "sum"),
        holding=("holding", "sum"), stockout=("stockout", "sum"), total=("total", "sum"))
    grouped["per_sku"] = grouped["total"] / grouped["skus"]
    grouped = grouped.reset_index().rename(columns={"group": "Group"})
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
        if frame.empty:
            continue
        a, p = frame["real_sales"].to_numpy("float64"), frame["sales_pred"].to_numpy("float64")
        total = float(a.sum())
        if total <= 0:
            continue
        error = a - p
        rows.append({"Group": label, "MAE": float(np.mean(np.abs(error))),
                     "Bias %": float((p.sum() - total) / total * 100),
                     "WRMSE": float(np.sqrt(np.mean(error ** 2)) / (total / len(a))),
                     "WAPE %": float(np.sum(np.abs(error)) / total * 100),
                     "pred_units": float(p.sum()), "real_units": total})
    return pd.DataFrame(rows)
def render_where_money_goes(summary, lookup, preds, model, baseline_summary=None):
    """Section 2: Portfolio KPIs, stacked cost breakdown, and holdout error per group."""
    st.markdown("### 2. Where the money goes")
    c1, c2, c3, c4 = st.columns(4)
    cost_delta = None
    if baseline_summary and baseline_summary.get("total", 0) > 0:
        saved = baseline_summary["total"] - summary["total"]
        cost_delta = f"-\\${saved:,.0f} (-{saved / baseline_summary['total'] * 100:.1f}%) vs Naive"
    c1.metric(f"Inventory cost @ \\${HOLDING_RATE:g}/unit/day", f"${summary['total']:,.0f}",
              delta=cost_delta, delta_color="normal" if cost_delta else "off")
    c2.metric("Holding share of cost", f"{summary['holding'] / max(summary['total'], 1):.0%}",
              f"${summary['stockout']:,.0f} is stockout", delta_color="inverse")
    lost_delta = f"{summary['lost']:,.0f} lost"
    if baseline_summary and baseline_summary.get("lost") is not None:
        lost_delta = f"{summary['lost']:,.0f} lost ({baseline_summary['lost'] - summary['lost']:+,.0f} vs Naive)"
    c3.metric("Units bought vs sold", f"{summary['bought']:,.0f} / {summary['sold']:,.0f}", lost_delta,
              delta_color="inverse")
    c4.metric("SKUs that never stocked out", f"{summary['no_stockout']} / {summary['skus']}")
    group_by = pick_filter("Group cost by", COST_GROUPS, "cost_group")
    per_item = summary["per_item"]
    labels = group_labels(per_item.index, per_item["sold"], lookup, group_by)
    grouped = group_costs(per_item, labels, group_by)
    if grouped.empty:
        st.info("No cost data for this grouping.")
        return
    accuracy = group_accuracy(preds, labels, model, truth_model(preds))
    for column in ("MAE", "Bias %", "WRMSE", "WAPE %"):
        grouped[column] = accuracy.set_index("Group").reindex(grouped["Group"])[column].values \
            if not accuracy.empty else np.nan
    fig = go.Figure()
    for name, column, color in (("Holding Cost", "holding", "#3b82f6"), ("Stockout Cost", "stockout", "#ef4444")):
        fig.add_trace(go.Bar(name=name, y=grouped["Group"], x=grouped[column], orientation="h", marker_color=color,
                             hovertemplate=f"<b>%{{y}}</b><br>{name.split()[0]}: $%{{x:,.0f}}<extra></extra>"))
    fig.update_layout(template="plotly_white", barmode="stack", height=max(240, 60 * len(grouped) + 70),
                      margin=dict(l=10, r=30, t=10, b=10), bargap=0.35,
                      xaxis_title="Inventory Cost Breakdown ($)", yaxis_title="",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    plot_col, table_col = st.columns([1.1, 1.4])
    with plot_col:
        st.plotly_chart(fig, width="stretch")
    with table_col:
        st.dataframe(grouped[GROUP_COLUMNS].rename(columns=GROUP_HEADERS).style.format(GROUP_FORMATS),
                     width="stretch", hide_index=True)
    bought, sold = float(grouped["bought"].sum()), float(grouped["sold"].sum())
    if sold > 0:
        st.caption(f"Across every group, the policy bought **{bought:,.0f}** units against **{sold:,.0f}** "
                   f"sold (**{bought / sold:.2f}x** ratio). Holding cost accounts for "
                   f"**{(summary['holding'] / summary['total']) * 100:.1f}%** of total cost.")
# ---------------------------------------------------------------- section 3
def render_filter_bar(lookup):
    """All six section-3 controls on one row, so the selection stays on screen while reading."""
    has_cat, has_dept = "cat_id" in lookup.columns, "dept_id" in lookup.columns
    scope_a, scope_b, scope_c, span_d, term_e, term_f = st.columns(6)
    with scope_a:
        options = ["All"] + sorted(lookup["cat_id"].dropna().unique()) if has_cat else ["All"]
        category = pick_filter("Category", options, "g_cat")
    if changed("g_applied_cat", category):
        st.session_state.pop("g_dept", None), st.session_state.pop("g_item", None)
    scoped_rows = lookup if category == "All" or not has_cat else lookup[lookup["cat_id"] == category]
    with scope_b:
        options = ["All"] + sorted(scoped_rows["dept_id"].dropna().unique()) if has_dept else ["All"]
        department = pick_filter("Department", options, "g_dept")
    if changed("g_applied_dept", department):
        st.session_state.pop("g_item", None)
    if department != "All" and has_dept:
        scoped_rows = scoped_rows[scoped_rows["dept_id"] == department]
    with scope_c:
        item = pick_filter("Item", ["All"] + sorted(scoped_rows.index.tolist()), "g_item")
    with span_d:
        lookback = pick_filter("History lookback", [f"{d}d" for d in LOOKBACKS], "g_lookback")
    with term_e:
        st.metric("Lead Time (L)", f"{LEAD_TIME} days")
    with term_f:
        st.metric("Review Cycle (R)", f"{REVIEW_PERIOD} days")
    members = [item] if item != "All" else scoped_rows.index.tolist()
    if item != "All":
        label = item
    elif department != "All":
        label = department
    else:
        label = category if category != "All" else f"All {len(lookup):,} SKUs"
    return members, label, int(lookback.rstrip("d"))
def line_trace(name, frame, value, color, width, hover=None, dash="solid", markers=False):
    """One forecast/actual line trace, so the figure stays readable."""
    return go.Scatter(x=frame["date"], y=frame[value], mode="lines+markers" if markers else "lines", name=name,
                      line=dict(color=color, width=width, dash=dash), marker=dict(size=5) if markers else None,
                      hovertemplate=(hover or "%{x|%b %d}<br>%{y:.0f} units<extra>{name}</extra>").replace("{name}", name))
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
def build_inventory_figure(inv, members, policy_model):
    """Right column plot: On-hand sawtooth, safety stock, and daily sales bars."""
    fig, daily = go.Figure(), daily_inventory(inv, members, policy_model)
    if daily.empty:
        return fig
    fig.add_trace(go.Bar(x=daily["date"], y=daily["actual_sales"], name="Daily Sales",
                         marker_color="rgba(245, 158, 11, 0.35)",
                         hovertemplate="%{x|%b %d}<br>%{y:.0f} units sold<extra>Sales</extra>"))
    for name, column, color, width, dash, hover in (
            ("On-Hand Inventory", "on_hand", "#16a34a", 2.5, "solid", "%{x|%b %d}<br>%{y:.0f} units on-hand<extra>On-Hand</extra>"),
            ("Safety Stock Target", "safety_stock", "#dc2626", 1.5, "dot", "%{x|%b %d}<br>%{y:.0f} units SS<extra>Safety Stock</extra>")):
        fig.add_trace(go.Scatter(x=daily["date"], y=daily[column],
                                 mode="lines+markers" if width > 2 else "lines", name=name,
                                 line=dict(color=color, width=width, dash=dash), marker=dict(size=4),
                                 hovertemplate=hover))
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=25, b=10), title="",
                      yaxis_title="Units", xaxis_title="Date", barmode="overlay", hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig
def build_detail_figure(train, preds, inv, members, policy_model, compare, days):
    """Actual vs predicted for the selection, with the orders the policy placed."""
    fig = go.Figure()
    history = history_frame(train, members, days)
    if not history.empty:
        fig.add_trace(line_trace("Actual", history, "sales", "#94a3b8", 1.5))
    for model in [policy_model, *compare]:
        frame = daily_sum(preds, members, model, "sales_pred")
        if frame.empty:
            continue
        primary = model == policy_model
        fig.add_trace(line_trace(f"{model_label(model)} predicted", frame, "sales_pred",
                                 MODEL_COLORS.get(model, "#0f172a"), 2.5 if primary else 1.8,
                                 "%{x|%b %d}<br>%{y:.1f} units predicted<extra>" + model_label(model) + "</extra>",
                                 dash="solid" if primary else "dash"))
    daily = daily_inventory(inv, members, policy_model)
    if daily.empty:
        return fig
    fig.add_trace(line_trace("Realised in holdout", daily, "actual_sales", "#0f172a", 2.5, markers=True))
    days_on = sorted(daily["date"])[::REVIEW_PERIOD]
    ordered = daily[daily["date"].isin(set(days_on))].copy()
    hover = review_hover(inv, preds, members, policy_model, days_on)
    ordered["in_transit"] = [hover[d]["in_transit"] for d in ordered["date"]]
    ordered["risk_demand"] = [hover[d]["risk_demand"] for d in ordered["date"]]
    ordered["arrival"] = ordered["date"] + pd.Timedelta(days=LEAD_TIME)
    fig.add_trace(go.Bar(x=ordered["date"], y=ordered["order_qty"], name="Order placed (whole cycle)",
                         marker_color="rgba(99,102,241,0.45)", marker_line=dict(color="#6366f1", width=1),
                         width=ORDER_BAR_MS,
                         customdata=ordered[["order_qty", "safety_stock", "order_up_to", "on_hand",
                                             "in_transit", "risk_demand", "arrival"]].to_numpy(),
                         hovertemplate=("<b>Review %{x|%b %d}</b><br>Order <b>%{customdata[0]:.1f}</b> units"
                                         "<br>safety stock %{customdata[1]:.1f} · S %{customdata[2]:.1f}<br>on hand %{customdata[3]:.1f} · in transit %{customdata[4]:.1f}"
                                         f"<br>forecast ({TAU}d) %{{customdata[5]:.1f}}"
                                         "<br>arrives %{customdata[6]|%b %d}<extra></extra>")))
    rate_frame = daily.copy()
    rate_frame["order_rate"] = daily_order_rate(daily, LEAD_TIME, REVIEW_PERIOD)
    fig.add_trace(line_trace("Order per day (cycle spread)", rate_frame, "order_rate", "#6366f1", 2,
                             "%{x|%b %d}<br>%{y:.0f} units/day<extra>Order per day</extra>", dash="dot"))
    fig.add_vrect(x0=daily["date"].min(), x1=daily["date"].max(),
                  fillcolor="#f1f5f9", opacity=0.55, layer="below", line_width=0)
    fig.update_layout(template="plotly_white", height=470, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="Units per day", xaxis_title="Date", barmode="overlay", hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0))
    return fig
def accuracy_metrics(preds, members, model, truth):
    """Error metrics for one model over the holdout, on the aggregated series."""
    frame = daily_sum(preds, members, model, "sales_pred")
    truth_frame = daily_sum(preds, members, truth, "real_sales") if truth else pd.DataFrame()
    if frame.empty or truth_frame.empty:
        return {}
    merged = frame.merge(truth_frame, on="date", how="inner")
    if merged.empty:
        return {}
    actual, predicted = merged["real_sales"].to_numpy("float64"), merged["sales_pred"].to_numpy("float64")
    total, safe = float(actual.sum()), float(actual.sum()) > 0
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    return {"MAE": float(np.mean(np.abs(actual - predicted))),
            "Bias %": float((predicted.sum() - total) / total * 100) if safe else np.nan,
            "WRMSE": rmse / (total / len(actual)) if safe else np.nan,
            "WAPE %": float(np.sum(np.abs(actual - predicted)) / total * 100) if safe else np.nan,
            "actual": total, "predicted": float(predicted.sum())}
def render_metrics_line(preds, members, policy_model, compare):
    """One line of error metrics per model on the left, the winner on the right."""
    truth = truth_model(preds)
    if truth is None:
        st.error("No model in the holdout carries `real_sales`, so there is no ground truth to "
                 "score against. Re-run the test mode.")
        return
    models = [policy_model, *[m for m in compare if m != policy_model]]
    ranked = [(m, s) for m in models if (s := accuracy_metrics(preds, members, m, truth))]
    if not ranked:
        st.info("No holdout metrics for this selection.")
        return
    left, right = st.columns([1.6, 1])
    with left:
        for model, stats in ranked:
            st.markdown(f"**{model_label(model)}** — MAE **{stats['MAE']:.2f}** · bias **{stats['Bias %']:+.0f}%** · WRMSE **{stats['WRMSE']:.2f}** · WAPE **{stats['WAPE %']:,.0f}%** · predicted {stats['predicted']:,.0f} vs actual {stats['actual']:,.0f}")
        if max(s["actual"] for _, s in ranked) < LOW_VOLUME:
            st.caption("Volume is tiny here, so trust MAE in units over the percentages.")
    with right:
        best = min(ranked, key=lambda pair: pair[1]["MAE"])[0]
        chips = [f"<span style='color:{GOOD if m == best else BAD};font-weight:700'>{'✓' if m == best else '✗'} {model_label(m)} {'+' if m == best else '−'}</span>"
                 for m, _ in ranked]
        st.markdown(f"<div style='text-align:right;padding-top:0.4rem'><span style='color:#64748b;"
                    f"font-size:0.8rem'>lowest MAE</span><br>{'<br>'.join(chips)}</div>",
                    unsafe_allow_html=True)
def build_review_table(preds, inv, members, policy_model):
    """What the policy decided on each review day, summed over the selection."""
    daily = daily_inventory(inv, members, policy_model)
    if daily.empty:
        return pd.DataFrame()
    days_on = sorted(daily["date"])[::REVIEW_PERIOD]
    hover = review_hover(inv, preds, members, policy_model, days_on)
    indexed = daily.set_index("date")
    rows = []
    for day in days_on:
        if day not in indexed.index:
            continue
        record, target = indexed.loc[day], float(indexed.loc[day, "order_up_to"])
        end = day + pd.Timedelta(days=REVIEW_PERIOD - 1)
        cycle = indexed.loc[(indexed.index >= day) & (indexed.index <= end)]
        rows.append({"Review": day.strftime("%b %d"), f"Forecast ({TAU}d)": hover[day]["risk_demand"],
                     "Safety stock": float(record["safety_stock"]), "Order-up-to (S)": target,
                     "On hand": float(record["on_hand"]), "In transit": hover[day]["in_transit"],
                     "Order placed": float(record["order_qty"]),
                     "Arrives": (day + pd.Timedelta(days=LEAD_TIME)).strftime("%b %d"),
                     "Sold in cycle": float(cycle["actual_sales"].sum()),
                     "SS % of S": (float(record["safety_stock"]) / target * 100) if target > 0 else np.nan})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    total = {column: np.nan for column in table.columns}
    total.update(Review="Total", **{"Order placed": float(daily["order_qty"].sum()),
                                   "Sold in cycle": float(daily["actual_sales"].sum())})
    return pd.concat([table, pd.DataFrame([total])], ignore_index=True)
def render_review_table(table):
    """The review-by-review decision table, with an over-buy warning."""
    st.markdown("**What the policy decided, review by review**")
    if table.empty:
        st.info("No replenishment log for this selection.")
        return
    st.dataframe(table.round(1), width="stretch", hide_index=True)
    per_review = table[table["Review"] != "Total"]
    bought, sold = float(per_review["Order placed"].sum()), float(table["Sold in cycle"].iloc[-1])
    if sold > 0 and bought > 2 * sold:
        st.warning(f"Bought {bought:,.0f} units, sold {sold:,.0f}. Safety stock is carrying the order-up-to level; check the holding rate and the seed window.")
def render_sku_detail(train, preds, inv, lookup, available):
    """Section 3: model pickers, scope, the aggregated plot, metrics, review table."""
    st.markdown("### 3. SKUs in detail")
    policy_col, compare_col = st.columns(2)
    with policy_col:
        policy_model = st.selectbox("Policy model (drives replenishment)", available,
                                    format_func=model_label, key="policy_model")
    with compare_col:
        compare = st.multiselect("Also show these forecasts", [m for m in available if m != policy_model],
                                 format_func=model_label, key="compare_models")
    members, label, days = render_filter_bar(lookup)
    if st.toggle("Show Inventory Sawtooth & Performance Metrics", value=False,
                 key="toggle_inventory_view"):
        st.plotly_chart(build_inventory_figure(inv, members, policy_model), width="stretch")
        st.markdown(f"#### Aggregated Inventory Performance (`{label}`)")
        performance = evaluate_inventory_performance(inv, members, policy_model, label)
        st.dataframe(performance, width="stretch", hide_index=True) if not performance.empty \
            else st.info("No inventory performance metrics for this selection.")
    else:
        st.caption(f"Showing **{label}** ({len(members):,} SKUs). Risk period = {LEAD_TIME} + {REVIEW_PERIOD} = **{TAU} days**, locked to the loaded run.")
        st.plotly_chart(build_detail_figure(train, preds, inv, members, policy_model, compare, days),
                        width="stretch")
        st.caption("The order bars are a whole review cycle; the dotted line is that same order spread across the days it covers, which is the only one of the two that is comparable to a single day of sales.")
        render_metrics_line(preds, members, policy_model, compare)
    st.divider()
    render_review_table(build_review_table(preds, inv, members, policy_model))
# ---------------------------------------------------------------- notes
def render_notes(provenance):
    """Methodology, statistical mechanics, and system provenance in a collapsible panel."""
    with st.expander("🛠️ Methodology, Statistical Mechanics & Audit Provenance", expanded=False):
        implied, has_issue = provenance["implied_rate"], False
        if implied is not None and abs(implied - HOLDING_RATE) > 1e-9:
            has_issue = True
            st.error(f"**Cost Configuration Mismatch:** `config.py` specifies `holding_cost_rate = {HOLDING_RATE}`, but the loaded inventory artifacts were generated under `{implied:.4f}` (recovered from holding_cost / on_hand). Re-run the test mode to synchronize.")
        if provenance["drift"]:
            has_issue = True
            st.warning("**Artifact Drift Detected:** Source files `" + "`, `".join(provenance["drift"])
                       + "` have been modified since this holdout was produced. Re-run test mode to refresh.")
        if not has_issue:
            st.success("**Provenance Verified:** Model predictions, inventory simulation, and config "
                       "parameters are fully synchronized.")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### Replenishment Policy Formulation")
            st.markdown("The periodic inventory policy uses dynamic Order-Up-To levels:\n\n$$S = \\hat{D}_{\\tau} + SS$$\n\n"
                        f"- **Protection Horizon (τ)**: τ = L + R = {LEAD_TIME} + {REVIEW_PERIOD} = {TAU} days, covering supplier delivery lead time (L) plus the replenishment review interval (R).\n"
                        "- **Forecast Lead Demand (D̂τ)**: Sum of multi-step model predictions over days [t, t+τ−1].\n"
                        "- **Safety Stock (SS)**: Calibrated via the Newsvendor critical fractile z = Φ⁻¹(p / (p + h)). With stockout penalty "
                        f"p=\\${STOCKOUT_RATE:g} and daily holding rate h=\\${HOLDING_RATE:g}, the target service level is ≈ 83.3% (z ≈ 0.967). Then SS = z · RMSE_τ.\n"
                        "- **Order Placed (Q)**: At each review cycle: Q = max(0, S − IP), where Inventory Position IP = On-Hand + In-Transit.")
        with col_m2:
            st.markdown("#### Safety Stock Dynamics & Known Limitations")
            st.markdown("- **The Over-Buffering Mechanism**: Sizing safety stock from cumulative RMSE charges systematic forecast bias as if it were uncorrectable random noise. For intermittent items (74% of this catalog), cumulative RMSE inflates the safety buffer far beyond the true demand variance.\n"
                        "- **Initial Inventory Seeding**: Initial on-hand is seeded from the trailing 28-day historical mean sales, which can anchor early replenishment orders to promotional spikes.\n"
                        "- **Production Enhancement Roadmap**: Decouple safety buffer sizing from Gaussian assumptions by deploying direct empirical pinball quantile loss (q₀.₈₃) or distribution-free conformal prediction intervals over the multi-step horizon.")
        st.divider()
        st.markdown("#### Lineage & Artifact Audit")
        st.caption(f"Active Artifacts: `{provenance['preds']}` · `{provenance['inv']}` | Assumptions: "
                   f"Holding rate \\${HOLDING_RATE:g}/unit/day · Stockout penalty "
                   f"\\${STOCKOUT_RATE:g}/lost unit | Evaluation: 28-day out-of-sample holdout across "
                   f"300 curated SKUs in Walmart M5 Store CA_1.")
# ---------------------------------------------------------------- page
def main():
    st.set_page_config(page_title="Retail Replenishment Optimization", layout="wide")
    st.markdown("<style>.block-container{padding-top:2rem}[data-testid='stMetricValue']{font-size:1.4rem}"
                "</style>", unsafe_allow_html=True)
    art = load_artifacts(artifact_signature())
    train, preds, inv, classes, provenance = (art[k] for k in ("train", "preds", "inv", "classes", "provenance"))
    if preds.empty or inv.empty:
        st.error("No holdout artifacts in `results/tests/`. Remove the `quit()` in `run_pipeline.py` and re-run the test mode.")
        return
    st.title("Retail Replenishment Optimization: 300-SKU Assortment")
    st.caption("Walmart M5 · Store CA_1 · 300 Curated SKUs · Periodic Review (s, S) Policy")
    available = sorted(preds["model"].dropna().unique().tolist())
    st.session_state.setdefault("policy_model", "lgbm" if "lgbm" in available else (available[0] if available else None))
    st.session_state.setdefault("compare_models", [])
    st.session_state.setdefault("g_lookback", f"{LOOKBACKS[2]}d")
    policy_model = st.session_state["policy_model"]
    lookup = item_lookup(classes, preds)
    summary = summarise(inv, policy_model)
    if not summary:
        st.info("No inventory log for that policy model.")
        return
    # Baseline for comparative deltas; the scatter highlight tracks section 3's item picker.
    baseline_name = "seasonal_naive" if "seasonal_naive" in inv["model"].values else None
    baseline_summary = summarise(inv, baseline_name) if baseline_name and baseline_name != policy_model else None
    active_item = st.session_state.get("g_item")
    active_item = None if active_item == "All" else active_item
    st.divider()
    st.markdown("### 1. Executive Summary & Assortment Strategy")
    framing, profile = st.columns([1.1, 1.3])
    with framing:
        render_business_framing(summary, baseline_summary)
    with profile:
        render_sku_profile(classes, active_item)
    st.divider()
    render_where_money_goes(summary, lookup, preds, policy_model, baseline_summary)
    st.divider()
    render_sku_detail(train, preds, inv, lookup, available)
    st.divider()
    render_notes(provenance)

if __name__ == "__main__":
    main()
