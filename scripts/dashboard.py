"""Walmart M5 -- forecast accuracy and replenishment cost for store CA_1.

One page, three sections:
  1. the business framing beside the demand-pattern map of the store
  2. where the inventory money goes, grouped
  3. SKUs in detail: actual vs predicted, the orders placed, and one line of metrics

Everything is read from the holdout artifacts; nothing is recomputed here.

Section 1 is `render_business_framing` + `render_sku_profile`, section 2 is
`render_where_money_goes`, section 3 is `render_sku_detail`. Edit those four.
"""

import glob
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
LOW_VOLUME = 20
LOOKBACKS = [30, 60, 90, 180, 365]
CLASS_ORDER = ["Smooth", "Lumpy", "Erratic", "Intermittent"]
COST_GROUPS = ["Demand class", "Category", "Department", f"< {LOW_VOLUME} units/mo"]

MODEL_LABELS = {
    "lgbm": "LightGBM (Direct)", "Naive": "Seasonal Naive",
    "Moving_Average": "Moving Average (180d)", "croston": "Croston",
    "croston_sba": "Croston-SBA",
}
MODEL_COLORS = {
    "lgbm": "#ff7f0e", "Naive": "#d62728", "Moving_Average": "#2ca02c",
    "croston": "#9467bd", "croston_sba": "#8c564b",
}
CLASS_COLORS = {
    "smooth": "#1f77b4", "lumpy": "#ff7f0e",
    "intermittent": "#2ca02c", "erratic": "#d62728",
}
GOOD, BAD = "#16a34a", "#dc2626"
ORDER_BAR_MS = 1000 * 3600 * 24 * 0.7  # ~70% of a day, so one bar per review


def model_label(name):
    """'lgbm' -> 'LightGBM (Direct)'; handles list-valued cells."""
    if not isinstance(name, str):
        return ", ".join(model_label(part) for part in name)
    return MODEL_LABELS.get(name, name.replace("_", " ").title())


# ============================================================
# Loading
# ============================================================

@st.cache_data
def load_artifacts():
    """Every artifact the page reads, plus which run they came from."""
    train_path = DATA_DIR / "train_filtered_ca1.parquet"
    train = pd.read_parquet(train_path) if train_path.exists() else pd.DataFrame()
    if not train.empty:
        train["date"] = pd.to_datetime(train["date"])
        train["item_id"] = train["item_id"].astype(str)

    pred_files = sorted(glob.glob(str(TESTS_DIR / "test_preds-*.parquet")), reverse=True)
    inv_files = sorted(glob.glob(str(TESTS_DIR / "test_inventory-*.parquet")), reverse=True)
    preds = pd.read_parquet(pred_files[0]) if pred_files else pd.DataFrame()
    inv = pd.read_parquet(inv_files[0]) if inv_files else pd.DataFrame()
    for frame in (preds, inv):
        if not frame.empty:
            frame["item_id"] = frame["item_id"].astype(str)
            frame["date"] = pd.to_datetime(frame["date"])

    classes_path = RESULTS_DIR / "sku_demand_classes.csv"
    classes = pd.read_csv(classes_path) if classes_path.exists() else pd.DataFrame()
    if not classes.empty:
        classes["item_id"] = classes["item_id"].astype(str)

    implied = None
    if not inv.empty:
        positive = inv[inv["on_hand"] > 0]
        if not positive.empty:
            implied = float((positive["holding_cost"] / positive["on_hand"]).median())

    return {"train": train, "preds": preds, "inv": inv, "classes": classes,
            "provenance": {
                "preds": Path(pred_files[0]).name if pred_files else None,
                "inv": Path(inv_files[0]).name if inv_files else None,
                "implied_rate": implied}}


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


# ============================================================
# Widgets
# ============================================================

def pick_filter(label, options, state_key):
    """A selectbox that survives its own option list changing.

    The value is seeded into the widget key *before* the widget is created and no
    `index` is passed, so the widget identity is stable across reruns.
    """
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


# ============================================================
# Frame maths
# ============================================================

def daily_sum(frame, members, model, column):
    """Total `column` per day across the members under one model."""
    scoped = frame[(frame["item_id"].isin(members)) & (frame["model"] == model)]
    if scoped.empty:
        return pd.DataFrame()
    return scoped.groupby("date", as_index=False)[column].sum().sort_values("date")


def daily_inventory(inv, members, model):
    """Daily inventory totals summed across the members."""
    scoped = inv[(inv["item_id"].isin(members)) & (inv["model"] == model)]
    if scoped.empty:
        return pd.DataFrame()
    return scoped.groupby("date", as_index=False).agg(
        order_qty=("order_qty", "sum"), safety_stock=("safety_stock", "sum"),
        order_up_to=("order_up_to", "sum"), on_hand=("on_hand", "sum"),
        actual_sales=("actual_sales", "sum")).sort_values("date")


def history_frame(train, members, days):
    """Total actual sales per day across the members, over the lookback."""
    scoped = train[train["item_id"].isin(members)]
    if scoped.empty:
        return pd.DataFrame()
    recent = scoped[scoped["date"] >= scoped["date"].max() - pd.Timedelta(days=days)]
    return recent.groupby("date", as_index=False)["sales"].sum().sort_values("date")


def window_sum(frame_by_date, column, start, end):
    """Sum one column over an inclusive date range on a date-indexed frame."""
    if frame_by_date.empty:
        return 0.0
    mask = (frame_by_date.index >= start) & (frame_by_date.index <= end)
    return float(frame_by_date.loc[mask, column].sum())


def review_hover(inv, preds, members, model, days):
    """In-transit and risk-period demand summed across the group, per review day.

    In-transit is receipts landing after the review day and within the lead time;
    risk demand is forecast demand over the whole protection horizon.
    """
    inv_frames = [g.set_index("date") for _, g in
                  inv[(inv["item_id"].isin(members)) & (inv["model"] == model)]
                  .groupby("item_id", observed=True)]
    pred_frames = [g.set_index("date") for _, g in
                   preds[(preds["item_id"].isin(members)) & (preds["model"] == model)]
                   .groupby("item_id", observed=True)]
    hover = {}
    for day in days:
        hover[day] = {
            "in_transit": sum(window_sum(f, "arriving_qty", day + pd.Timedelta(days=1),
                                         day + pd.Timedelta(days=LEAD_TIME))
                              for f in inv_frames),
            "risk_demand": sum(window_sum(f, "sales_pred", day,
                                          day + pd.Timedelta(days=TAU - 1))
                               for f in pred_frames)}
    return hover


# ============================================================
# Section 1 -- business framing (left) and SKU profile (right)
# ============================================================

def render_business_framing(summary):
    """The ask, the policy's behaviour, and the one caveat. Edit here."""
    st.markdown(f"""
**The ask.** Cut inventory cost across a {summary['skus']}-SKU assortment without
losing sales.

**What the policy does today.** Every SKU is replenished on a fixed
{REVIEW_PERIOD}-day cycle up to forecast demand over the {TAU}-day risk period
(lead time + review period) plus an error-based buffer. Replayed over the 28-day
holdout that buys **{summary['bought']:,.0f} units** against **{summary['sold']:,.0f}**
of demand, and **{summary['no_stockout']} of {summary['skus']} SKUs never stock out**.

**So this is not a service problem.** Almost none of the cost is missed sales; it is
stock that was bought and never sold. The question is not "how do we forecast better"
-- it is "how much stock does the error model tell us to hold, and is that justified".

**What we built.** Forecast 28 days out, turn the *cumulative* forecast error over the
{TAU}-day risk period into a safety-stock buffer, raise the inventory position at each
review, then replay the month against realised demand to price the decision in dollars.

**The one caveat.** Unlike the VN2 competition there is **no simple benchmark for this
optimisation to beat**. VN2 shipped a closed-form baseline, so the only question was
whether we cleared it. Here there is nothing to clear, so "is this policy good?" is
answered by replaying the same month under different models and comparing error and
dollars. That compares *policies*; it does not prove optimality.
""")


def render_sku_profile(classes, item_id):
    """ADI vs CV2 for the whole store, with the selected SKU ringed."""
    selected = classes[classes["item_id"] == item_id] if item_id else classes.iloc[:0]
    if selected.empty:
        st.info("No SKU selected.")
        return
    point = selected.iloc[0]

    fig = go.Figure(go.Scatter(
        x=classes["adi"], y=classes["cv2"], mode="markers",
        marker=dict(
            size=8 + 34 * np.sqrt(classes["rev"] / classes["rev"].max()),
            color=[CLASS_COLORS.get(str(c).lower(), "#64748b") for c in classes["class"]],
            opacity=0.65, line=dict(width=0)),
        customdata=np.stack([classes["item_id"], classes["class"], classes["mean"],
                             classes["zero_frac"], classes["rev"]], axis=-1),
        hovertemplate=(
            "<b>%{customdata[0]}</b> (%{customdata[1]})<br>ADI %{x:.2f} · CV² %{y:.2f}"
            "<br>Mean %{customdata[2]:.2f}/day · zero days %{customdata[3]:.0%}"
            "<br>Revenue $%{customdata[4]:,.0f}<extra></extra>"),
        showlegend=False))
    fig.add_trace(go.Scatter(
        x=[float(point["adi"])], y=[float(point["cv2"])], mode="markers",
        marker=dict(size=26, color="rgba(0,0,0,0)", line=dict(color="#0f172a", width=3)),
        hoverinfo="skip", showlegend=False))
    fig.add_vline(x=1.32, line_dash="dash", line_color="#94a3b8", line_width=1)
    fig.add_hline(y=0.49, line_dash="dash", line_color="#94a3b8", line_width=1)
    fig.update_layout(template="plotly_white", height=430,
                      margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="ADI (days between sales)", yaxis_title="CV²")
    st.plotly_chart(fig, width="stretch")

    demand_class = str(point["class"]).lower()
    st.caption(f"`{item_id}` is **{demand_class}** -- ADI {point['adi']:.2f}, "
               f"CV² {point['cv2']:.2f}, {point['zero_frac']:.0%} zero days, "
               f"{point['mean']:.2f} units/day.")
    if demand_class in ("intermittent", "erratic"):
        st.caption("**Warning:** a buffer sized from cumulative forecast error is wide "
                   "and unstable on a series like this, and one promotional week can set "
                   "the level it anchors to.")


# ============================================================
# Section 2 -- where the money goes
# ============================================================

def summarise(inv, model):
    """Cost and service totals for one policy model across the assortment."""
    scoped = inv[inv["model"] == model]
    if scoped.empty:
        return {}
    per_item = scoped.groupby("item_id", observed=True).agg(
        bought=("order_qty", "sum"), sold=("actual_sales", "sum"),
        lost=("lost_sales", "sum"), holding=("holding_cost", "sum"),
        stockout=("stockout_cost", "sum"))
    per_item["total"] = per_item["holding"] + per_item["stockout"]
    totals = {column: float(per_item[column].sum())
              for column in ("bought", "sold", "lost", "holding", "stockout", "total")}
    return dict(per_item=per_item, skus=int(len(per_item)), no_stockout=int(
        (per_item["stockout"] == 0).sum()), **totals)


def group_costs(per_item, lookup, group_by):
    """Aggregate per-item cost into the group being compared."""
    if per_item.empty:
        return pd.DataFrame()
    if group_by == "Demand class":
        column = "class"
    elif group_by in ("Category", "Department"):
        column = "cat_id" if group_by == "Category" else "dept_id"
    else:
        column = None

    if column is None:
        labels = pd.Series(np.where(per_item["sold"] < LOW_VOLUME, group_by,
                                     f">= {LOW_VOLUME} units/mo"), index=per_item.index)
    else:
        values = lookup[column].astype(str).str.strip()
        labels = pd.Series(per_item.index.map(values.str.title() if column == "class" else values),
                           index=per_item.index)

    labelled = per_item.copy()
    labelled["group"] = labels.fillna("Unclassified")
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


def render_where_money_goes(summary, lookup, model):
    """Section 2: portfolio KPIs plus cost grouped the way a planner reads it."""
    st.markdown("### 2. Where the money goes")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total inventory cost", f"${summary['total']:,.0f}")
    c2.metric("Holding share of cost",
              f"{summary['holding'] / max(summary['total'], 1):.0%}",
              f"${summary['stockout']:,.0f} is stockout", delta_color="inverse")
    c3.metric("Units bought vs sold", f"{summary['bought']:,.0f} / {summary['sold']:,.0f}",
              f"{summary['lost']:,.0f} lost", delta_color="inverse")
    c4.metric("SKUs that never stocked out", f"{summary['no_stockout']} / {summary['skus']}")

    group_by = pick_filter("Group cost by", COST_GROUPS, "cost_group")
    grouped = group_costs(summary["per_item"], lookup, group_by)
    if grouped.empty:
        st.info("No cost data for this grouping.")
        return

    fig = go.Figure(go.Bar(
        x=grouped["total"], y=grouped["Group"], orientation="h",
        marker=dict(color=[CLASS_COLORS.get(str(g).lower(), "#6366f1")
                           for g in grouped["Group"]],
                    line=dict(color="rgba(15,23,42,0.18)", width=1)),
        text=[f"${v:,.0f}" for v in grouped["total"]], textposition="outside",
        cliponaxis=False,
        customdata=np.stack([grouped["holding"], grouped["stockout"], grouped["skus"],
                             grouped["bought"], grouped["sold"], grouped["per_sku"]], axis=-1),
        hovertemplate=(
            "<b>%{y}</b> — $%{x:,.0f} total"
            "<br>holding $%{customdata[0]:,.0f} · stockout $%{customdata[1]:,.0f}"
            "<br>%{customdata[2]:,.0f} SKUs · bought %{customdata[3]:,.0f}"
            " / sold %{customdata[4]:,.0f}<br>$%{customdata[5]:,.0f} per SKU<extra></extra>"),
        showlegend=False))
    fig.update_layout(template="plotly_white", height=max(230, 66 * len(grouped) + 70),
                      margin=dict(l=10, r=80, t=10, b=10), bargap=0.4,
                      xaxis_title="Total inventory cost over the 28-day holdout ($)",
                      yaxis_title="")

    plot_col, table_col = st.columns([1.5, 1])
    with plot_col:
        st.plotly_chart(fig, width="stretch")
    with table_col:
        money = grouped[["total", "holding", "stockout", "per_sku"]].round(0)
        st.dataframe(money.style.format("${:,.0f}"), width="stretch", hide_index=True)


# ============================================================
# Section 3 -- SKUs in detail
# ============================================================

def group_filters(lookup):
    """Cascading Category -> Department -> Item pickers. Returns members + label."""
    has_cat = "cat_id" in lookup.columns
    has_dept = "dept_id" in lookup.columns
    cats = ["All"] + sorted(lookup["cat_id"].dropna().unique()) if has_cat else ["All"]
    category = pick_filter("Category", cats, "g_cat")
    if changed("g_applied_cat", category):
        st.session_state.pop("g_dept", None)
        st.session_state.pop("g_item", None)

    scoped = lookup if category == "All" or not has_cat else lookup[lookup["cat_id"] == category]
    depts = ["All"] + sorted(scoped["dept_id"].dropna().unique()) if has_dept else ["All"]
    department = pick_filter("Department", depts, "g_dept")
    if changed("g_applied_dept", department):
        st.session_state.pop("g_item", None)

    scoped = scoped if department == "All" or not has_dept else scoped[scoped["dept_id"] == department]
    item = pick_filter("Item", ["All"] + sorted(scoped.index.tolist()), "g_item")

    if item != "All":
        return [item], item
    if department != "All":
        return scoped.index.tolist(), department
    if category != "All":
        return scoped.index.tolist(), category
    return lookup.index.tolist(), f"All {len(lookup):,} SKUs"


def render_policy_windows():
    """Lead time and review period, read-only at the loaded run's values."""
    left, right = st.columns(2)
    left.slider("Lead time (days)", LEAD_TIME, LEAD_TIME + 1, LEAD_TIME,
                disabled=True, key="w_lead")
    right.slider("Review period (days)", REVIEW_PERIOD, REVIEW_PERIOD + 1, REVIEW_PERIOD,
                 disabled=True, key="w_review")
    st.caption(f"Risk period = {LEAD_TIME} + {REVIEW_PERIOD} = **{TAU} days**, locked.")


def line_trace(name, frame, value, color, width, hover=None, dash="solid", markers=False):
    """One forecast/actual line trace, so the figure stays readable."""
    template = hover or "%{x|%b %d}<br>%{y:.0f} units<extra>{name}</extra>"
    return go.Scatter(
        x=frame["date"], y=frame[value],
        mode="lines+markers" if markers else "lines", name=name,
        line=dict(color=color, width=width, dash=dash),
        marker=dict(size=5) if markers else None,
        hovertemplate=template.replace("{name}", name))


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
        fig.add_trace(line_trace(
            f"{model_label(model)} predicted", frame, "sales_pred",
            MODEL_COLORS.get(model, "#0f172a"), 2.5 if primary else 1.8,
            "%{x|%b %d}<br>%{y:.1f} units predicted<extra>" + model_label(model) + "</extra>",
            dash="solid" if primary else "dash"))

    daily = daily_inventory(inv, members, policy_model)
    if daily.empty:
        return fig
    fig.add_trace(line_trace("Realised in holdout", daily, "actual_sales",
                             "#0f172a", 2.5, markers=True))

    days_on = sorted(daily["date"])[::REVIEW_PERIOD]
    ordered = daily[daily["date"].isin(set(days_on))].copy()
    hover = review_hover(inv, preds, members, policy_model, days_on)
    ordered["in_transit"] = [hover[d]["in_transit"] for d in ordered["date"]]
    ordered["risk_demand"] = [hover[d]["risk_demand"] for d in ordered["date"]]
    ordered["arrival"] = ordered["date"] + pd.Timedelta(days=LEAD_TIME)
    fig.add_trace(go.Bar(
        x=ordered["date"], y=ordered["order_qty"], name="Order placed",
        marker_color="rgba(99,102,241,0.45)", marker_line=dict(color="#6366f1", width=1),
        width=ORDER_BAR_MS,
        customdata=ordered[["order_qty", "safety_stock", "order_up_to", "on_hand",
                            "in_transit", "risk_demand", "arrival"]].to_numpy(),
        hovertemplate=(
            "<b>Review %{x|%b %d}</b><br>Order <b>%{customdata[0]:.1f}</b> units"
            "<br>safety stock %{customdata[1]:.1f} · S %{customdata[2]:.1f}"
            "<br>on hand %{customdata[3]:.1f} · in transit %{customdata[4]:.1f}"
            f"<br>forecast ({TAU}d) %{{customdata[5]:.1f}}"
            "<br>arrives %{customdata[6]|%b %d}<extra></extra>")))
    fig.add_vrect(x0=daily["date"].min(), x1=daily["date"].max(),
                  fillcolor="#f1f5f9", opacity=0.55, layer="below", line_width=0)
    fig.update_layout(
        template="plotly_white", height=470, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Units", xaxis_title="Date", barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
        hovermode="x unified")
    return fig


def accuracy_metrics(preds, members, model):
    """Error metrics for one model over the holdout, on the aggregated series."""
    frame = daily_sum(preds, members, model, "sales_pred")
    truth = daily_sum(preds, members, "Naive", "real_sales")
    if frame.empty or truth.empty:
        return {}
    merged = frame.merge(truth, on="date", how="inner")
    if merged.empty:
        return {}
    actual = merged["real_sales"].to_numpy(dtype="float64")
    predicted = merged["sales_pred"].to_numpy(dtype="float64")
    total = float(actual.sum())
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    safe = total > 0
    return {
        "MAE": float(np.mean(np.abs(actual - predicted))),
        "Bias %": float((predicted.sum() - total) / total * 100) if safe else np.nan,
        "WRMSE": rmse / (total / len(actual)) if safe else np.nan,
        "WAPE %": float(np.sum(np.abs(actual - predicted)) / total * 100) if safe else np.nan,
        "actual": total, "predicted": float(predicted.sum())}


def render_metrics_line(preds, members, policy_model, compare):
    """One line of error metrics per model on the left, the winner on the right."""
    models = [policy_model, *[m for m in compare if m != policy_model]]
    scored = {m: accuracy_metrics(preds, members, m) for m in models}
    scored = {m: s for m, s in scored.items() if s}
    if not scored:
        st.info("No holdout metrics for this selection.")
        return

    left, right = st.columns([1.6, 1])
    with left:
        for model in models:
            if model not in scored:
                continue
            stats = scored[model]
            st.markdown(
                f"**{model_label(model)}** — MAE **{stats['MAE']:.2f}** · "
                f"bias **{stats['Bias %']:+.0f}%** · WRMSE **{stats['WRMSE']:.2f}** · "
                f"WAPE **{stats['WAPE %']:,.0f}%** · predicted {stats['predicted']:,.0f} "
                f"vs actual {stats['actual']:,.0f}")
        if max(s["actual"] for s in scored.values()) < LOW_VOLUME:
            st.caption("Volume is tiny here, so trust MAE in units over the percentages.")

    with right:
        best = min(scored, key=lambda m: scored[m]["MAE"])
        chips = []
        for model in models:
            if model not in scored:
                continue
            win = model == best
            chips.append(f"<span style='color:{GOOD if win else BAD};font-weight:700'>"
                         f"{'✓' if win else '✗'} {model_label(model)} "
                         f"{'+' if win else '−'}</span>")
        st.markdown(
            f"<div style='text-align:right;padding-top:0.4rem'>"
            f"<span style='color:#64748b;font-size:0.8rem'>lowest MAE</span><br>"
            f"{'<br>'.join(chips)}</div>", unsafe_allow_html=True)


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
        record = indexed.loc[day]
        cycle = indexed.loc[(indexed.index >= day)
                            & (indexed.index <= day + pd.Timedelta(days=REVIEW_PERIOD - 1))]
        target = float(record["order_up_to"])
        rows.append({
            "Review": day.strftime("%b %d"),
            f"Forecast ({TAU}d)": hover[day]["risk_demand"],
            "Safety stock": float(record["safety_stock"]),
            "Order-up-to (S)": target,
            "On hand": float(record["on_hand"]),
            "In transit": hover[day]["in_transit"],
            "Order placed": float(record["order_qty"]),
            "Arrives": (day + pd.Timedelta(days=LEAD_TIME)).strftime("%b %d"),
            "Sold in cycle": float(cycle["actual_sales"].sum()),
            "SS % of S": (float(record["safety_stock"]) / target * 100) if target > 0 else np.nan})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    total = {column: np.nan for column in table.columns}
    total["Review"] = "Total"
    total["Order placed"] = float(daily["order_qty"].sum())
    total["Sold in cycle"] = float(daily["actual_sales"].sum())
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
        st.warning(f"Bought {bought:,.0f} units, sold {sold:,.0f}. Safety stock is carrying "
                   "the order-up-to level; check the holding rate and the seed window.")


def render_sku_detail(train, preds, inv, lookup, policy_model, compare):
    """Section 3: group by, the aggregated plot, one metrics line, the review table."""
    st.markdown("### 3. SKUs in detail")
    st.markdown("**Group by**")
    members, label = group_filters(lookup)
    st.caption(f"Showing **{label}** ({len(members):,} SKUs).")
    render_policy_windows()
    days = st.selectbox("History lookback", LOOKBACKS, index=2, key="g_lookback")
    st.plotly_chart(build_detail_figure(train, preds, inv, members,
                                        policy_model, compare, days), width="stretch")
    render_metrics_line(preds, members, policy_model, compare)
    render_review_table(build_review_table(preds, inv, members, policy_model))


# ============================================================
# Notes
# ============================================================

def render_notes(provenance):
    """Where the numbers came from and what to distrust in them."""
    st.markdown("### Notes on these numbers")
    implied = provenance["implied_rate"]
    if implied is not None and abs(implied - HOLDING_RATE) > 1e-9:
        st.error(f"**Cost configuration mismatch.** `config.py` sets `holding_cost_rate = "
                 f"{HOLDING_RATE}`, but the loaded inventory log was produced with "
                 f"`{implied:.4f}` (recovered from holding_cost / on_hand). Every dollar "
                 "figure here comes from the older configuration. Re-run the holdout "
                 "before quoting these to a client.")
    if "quit()" in (BASE_DIR / "run_pipeline.py").read_text():
        st.warning("**Stale artifact.** `run_pipeline.py` calls `quit()` before "
                   "`to_parquet`, so the holdout cannot be regenerated until that is removed.")
    st.markdown(
        f"**How the policy computes it.** `S = D({TAU}d) + SS`; `D` is forecast demand "
        f"over lead time + review period and `SS = z * RMSE_tau` is RMSE of the "
        f"*cumulative* {TAU}-day error from the out-of-sample calibration window, with "
        f"`z = Phi^-1(h / (h + p))` taken from the cost inputs. The order is "
        "`max(0, S - IP)` where `IP` is on-hand plus in-transit. Holding is "
        "`on_hand * h` per unit-day and stockout is `lost_sales * p`.")
    st.caption(
        f"Sources: `{provenance['preds']}`, `{provenance['inv']}`. Policy: review every "
        f"{REVIEW_PERIOD}d, lead time {LEAD_TIME}d, risk period {TAU}d, holding rate "
        f"{HOLDING_RATE}/unit/day, stockout penalty 1.0/unit. Safety stock uses `z * RMSE` "
        "of cumulative error, and RMSE charges a systematic miss as if it were noise, so "
        "the buffer is larger than the underlying variability warrants. One 28-day "
        "calibration window feeds it, only about two non-overlapping risk periods. Initial "
        "on-hand is seeded from the trailing 28-day mean, so a promotional spike sets the "
        "first order.")


# ============================================================
# Page
# ============================================================

def main():
    st.set_page_config(page_title="Forecast & Inventory", layout="wide")
    st.markdown("<style>.block-container{padding-top:2rem}"
                "[data-testid='stMetricValue']{font-size:1.4rem}</style>",
                unsafe_allow_html=True)

    art = load_artifacts()
    train, preds, inv = art["train"], art["preds"], art["inv"]
    classes, provenance = art["classes"], art["provenance"]
    if preds.empty or inv.empty:
        st.error("No holdout artifacts in `results/tests/`. Remove the `quit()` in "
                 "`run_pipeline.py` and re-run the test mode.")
        return

    st.title("Reducing inventory cost across 300 SKUs")
    st.caption("Walmart M5 · store `CA_1` · 300 curated SKUs · periodic review")

    available = sorted(preds["model"].dropna().unique().tolist())
    policy_col, compare_col = st.columns(2)
    with policy_col:
        policy_model = st.selectbox(
            "Policy model (drives replenishment)", available,
            index=available.index("lgbm") if "lgbm" in available else 0,
            format_func=model_label, key="policy_model")
    with compare_col:
        compare = st.multiselect("Also show these forecasts",
                                 [m for m in available if m != policy_model],
                                 format_func=model_label, key="compare_models")

    lookup = item_lookup(classes, preds)
    summary = summarise(inv, policy_model)
    if not summary:
        st.info("No inventory log for that policy model.")
        return

    st.divider()
    st.markdown("### 1. The question on this dataset")
    focus = classes.loc[classes["item_id"].isin(lookup.index), "item_id"].iloc[0] \
        if not classes.empty else None
    framing, profile = st.columns([1, 1.15])
    with framing:
        render_business_framing(summary)
    with profile:
        render_sku_profile(classes, focus)

    st.divider()
    render_where_money_goes(summary, lookup, policy_model)

    st.divider()
    render_sku_detail(train, preds, inv, lookup, policy_model, compare)

    st.divider()
    render_notes(provenance)


if __name__ == "__main__":
    main()
