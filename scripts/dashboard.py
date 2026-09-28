"""Walmart M5 -- forecast accuracy and replenishment cost for store CA_1.

One page, three sections:
  1. the business framing beside the demand-pattern map of the store
  2. where the inventory money goes, grouped, with holdout error alongside cost
  3. SKUs in detail: model choice, scope, actual vs predicted, orders, metrics

Everything is read from the holdout artifacts; nothing is recomputed here.

Section 1 is `render_business_framing` + `render_sku_profile`, section 2 is
`render_where_money_goes`, section 3 is `render_sku_detail`. Edit those four.
"""

import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

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

def artifact_signature():
    """Cheap fingerprint of everything `load_artifacts` reads.

    `load_artifacts` takes this as its only argument so the cache key moves when
    the files move. With no argument at all Streamlit would cache the whole
    session and keep serving the first load's numbers, so re-running the pipeline
    in a terminal would not show up on the page until a hard refresh -- which is
    exactly the loop this page gets used in. Source files go in by content hash
    so the staleness check re-evaluates too; a `git stash` rewrites mtimes
    without changing content, and must not count as an edit.
    """
    parts = []
    for pattern in ("test_preds-*.parquet", "test_inventory-*.parquet"):
        for path in sorted(TESTS_DIR.glob(pattern), reverse=True)[:1]:
            parts.append((path.name, int(path.stat().st_mtime)))
    for path in (RESULTS_DIR / "sku_demand_classes.csv", TESTS_DIR / "source_state.json"):
        parts.append((path.name, int(path.stat().st_mtime) if path.exists() else 0))
    for name in ("run_pipeline.py", "config.py"):
        source = BASE_DIR / name
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16] if source.exists() else ""
        parts.append((name, digest))
    return tuple(parts)


@st.cache_data
def load_artifacts(signature):
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

    # Content hashes, not mtimes. A git checkout or stash rewrites every file in
    # the tree and bumps all the mtimes, so an mtime comparison reports "stale"
    # for a tree nobody edited. The manifest is written by `save_to_parquet` at
    # run time; its absence just means the check stays quiet.
    manifest_path = TESTS_DIR / "source_state.json"
    drift = []
    if manifest_path.exists() and inv_files:
        try:
            recorded = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            recorded = {}
        for name in ("run_pipeline.py", "config.py"):
            source = BASE_DIR / name
            if name in recorded and source.exists():
                current = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
                if current != recorded[name]:
                    drift.append(name)

    return {"train": train, "preds": preds, "inv": inv, "classes": classes,
            "provenance": {
                "preds": Path(pred_files[0]).name if pred_files else None,
                "inv": Path(inv_files[0]).name if inv_files else None,
                "implied_rate": implied,
                "drift": drift}}


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
def evaluate_inventory_performance(
    inv_df: pd.DataFrame,
    members: list[str],
    model: str,
    selection_label: str,
) -> pd.DataFrame:
    """Calculates aggregated inventory metrics over all selected items in the scope

    returning a single summary row.
    """
    scoped_inv = inv_df[
        (inv_df["item_id"].isin(members)) & (inv_df["model"] == model)
    ].copy()

    if scoped_inv.empty:
        return pd.DataFrame()

    # Aggregate daily totals across all selected SKUs
    daily_agg = scoped_inv.groupby("date", as_index=False).agg(
        actual_sales=("actual_sales", "sum"),
        on_hand=("on_hand", "sum"),
        safety_stock=("safety_stock", "sum"),
        lost_sales=(
            "lost_sales",
            "sum",
        )
        if "lost_sales" in scoped_inv.columns
        else ("actual_sales", "count"),
    )

    total_demand = daily_agg["actual_sales"].sum()

    # Unmet demand calculation
    if "lost_sales" in scoped_inv.columns:
        unmet_demand = scoped_inv["lost_sales"].sum()
    else:
        unmet_demand = np.maximum(
            0, scoped_inv["actual_sales"] - scoped_inv["on_hand"]
        ).sum()

    # Aggregate Fill Rate
    fill_rate = (
        ((total_demand - unmet_demand) / total_demand) * 100.0
        if total_demand > 0
        else 100.0
    )

    # Item-days with stockouts
    stockout_item_days = int((scoped_inv["on_hand"] == 0).sum())
    total_item_days = len(scoped_inv)
    stockout_pct = (
        (stockout_item_days / total_item_days * 100.0)
        if total_item_days > 0
        else 0.0
    )

    avg_on_hand = daily_agg["on_hand"].mean()
    avg_safety_stock = daily_agg["safety_stock"].mean()

    metrics = [
        {
            "Selection Scope": selection_label,
            "Total SKUs": len(members),
            "Total Demand": int(total_demand),
            "Fill Rate (%)": f"{max(0.0, fill_rate):.2f}%",
            "Stockout Rate": f"{stockout_pct:.1f}% ({stockout_item_days} item-days)",
            "Mean Daily On-Hand": f"{avg_on_hand:,.1f} units",
            "Mean Daily SS": f"{avg_safety_stock:,.1f} units",
        }
    ]

    return pd.DataFrame(metrics)
# ============================================================
# Frame maths
# ============================================================

def truth_model(preds):
    """Which model's rows carry usable `real_sales`, chosen by content not name.

    The artifact writes `real_sales` on the baseline rows only -- the lgbm rows
    are zero-filled -- so ground truth has to be lifted off a baseline. Matching
    on a hardcoded name is what broke this: `MODEL_LABELS` calls it "Naive" but
    the parquet column says `seasonal_naive`, so the filter matched zero rows and
    the whole accuracy line silently rendered nothing.
    """
    if preds.empty or "real_sales" not in preds.columns:
        return None
    totals = preds.groupby("model", observed=True)["real_sales"].sum()
    usable = totals[totals > 0]
    return str(usable.idxmax()) if not usable.empty else None


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


def group_labels(items, sold, lookup, group_by):
    """Label every SKU with the group it falls into for the chosen comparison."""
    items = pd.Index(items)
    if group_by == "Demand class":
        column = "class"
    elif group_by in ("Category", "Department"):
        column = "cat_id" if group_by == "Category" else "dept_id"
    else:
        column = None

    if column is None:
        return pd.Series(np.where(sold.reindex(items).fillna(0) < LOW_VOLUME, group_by,
                                  f">= {LOW_VOLUME} units/mo"), index=items)
    values = lookup[column].astype(str).str.strip()
    return pd.Series(items.map(values.str.title() if column == "class" else values),
                     index=items)


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

    Scored after aggregation rather than averaged from the members' scores: a
    category's daily series is far smoother than any single SKU inside it, so
    averaging member MAEs and calling it the category MAE would overstate the
    error. These numbers are only comparable with other groups, not with the
    per-SKU numbers in section 3.
    """
    if preds.empty or truth is None or labels.empty:
        return pd.DataFrame()
    pred_by_item = {item: frame.set_index("date")["sales_pred"] for item, frame in
                    preds[preds["model"] == model].groupby("item_id", observed=True)}
    truth_by_item = {item: frame.set_index("date")["real_sales"] for item, frame in
                     preds[preds["model"] == truth].groupby("item_id", observed=True)}

    rows = []
    for label, members in labels.groupby(labels).groups.items():
        usable = [item for item in members if item in pred_by_item and item in truth_by_item]
        if not usable:
            continue
        actual = pd.DataFrame({i: truth_by_item[i] for i in usable}).sum(axis=1)
        predicted = pd.DataFrame({i: pred_by_item[i] for i in usable}).sum(axis=1)
        frame = pd.concat([actual.rename("real_sales"), predicted.rename("sales_pred")],
                          axis=1).dropna()
        if frame.empty:
            continue
        a = frame["real_sales"].to_numpy(dtype="float64")
        p = frame["sales_pred"].to_numpy(dtype="float64")
        total = float(a.sum())
        if total <= 0:
            continue
        error = a - p
        rows.append({
            "Group": label,
            "MAE": float(np.mean(np.abs(error))),
            "Bias %": float((p.sum() - total) / total * 100),
            "WRMSE": float(np.sqrt(np.mean(error ** 2)) / (total / len(a))),
            "WAPE %": float(np.sum(np.abs(error)) / total * 100),
            "pred_units": float(p.sum()),
            "real_units": total})
    return pd.DataFrame(rows)


def render_where_money_goes(summary, lookup, preds, model):
    """Section 2: Portfolio KPIs, stacked cost breakdown, and holdout error per group."""
    st.markdown("### 2. Where the money goes")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total inventory cost", f"${summary['total']:,.0f}")
    c2.metric(
        "Holding share of cost",
        f"{summary['holding'] / max(summary['total'], 1):.0%}",
        f"${summary['stockout']:,.0f} is stockout",
        delta_color="inverse",
    )
    c3.metric(
        "Units bought vs sold",
        f"{summary['bought']:,.0f} / {summary['sold']:,.0f}",
        f"{summary['lost']:,.0f} lost",
        delta_color="inverse",
    )
    c4.metric(
        "SKUs that never stocked out",
        f"{summary['no_stockout']} / {summary['skus']}",
    )

    group_by = pick_filter("Group cost by", COST_GROUPS, "cost_group")
    per_item = summary["per_item"]
    labels = group_labels(per_item.index, per_item["sold"], lookup, group_by)
    grouped = group_costs(per_item, labels, group_by)
    if grouped.empty:
        st.info("No cost data for this grouping.")
        return

    accuracy = group_accuracy(preds, labels, model, truth_model(preds))
    if not accuracy.empty:
        grouped = grouped.merge(accuracy, on="Group", how="left")
    else:
        for column in ("MAE", "Bias %", "WRMSE", "WAPE %"):
            grouped[column] = np.nan

    # Clean executive table format focusing on actionable metrics
    table = grouped[
        [
            "Group",
            "skus",
            "total",
            "holding",
            "stockout",
            "bought",
            "sold",
            "Bias %",
            "WAPE %",
        ]
    ].rename(
        columns={
            "skus": "SKUs",
            "total": "Total $",
            "holding": "Holding $",
            "stockout": "Stockout $",
            "bought": "Bought",
            "sold": "Sold",
            "Bias %": "Bias %",
            "WAPE %": "WAPE %",
        }
    )

    # Stacked Bar Chart: Holding vs Stockout Cost
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Holding Cost",
            y=grouped["Group"],
            x=grouped["holding"],
            orientation="h",
            marker_color="#3b82f6",
            hovertemplate="<b>%{y}</b><br>Holding: $%{x:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Stockout Cost",
            y=grouped["Group"],
            x=grouped["stockout"],
            orientation="h",
            marker_color="#ef4444",
            hovertemplate="<b>%{y}</b><br>Stockout: $%{x:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        barmode="stack",
        height=max(240, 60 * len(grouped) + 70),
        margin=dict(l=10, r=30, t=10, b=10),
        bargap=0.35,
        xaxis_title="Inventory Cost Breakdown ($)",
        yaxis_title="",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )

    plot_col, table_col = st.columns([1.1, 1.4])
    with plot_col:
        st.plotly_chart(fig, width="stretch")
    with table_col:
        styled = table.style.format(
            {
                "Total $": "${:,.0f}",
                "Holding $": "${:,.0f}",
                "Stockout $": "${:,.0f}",
                "Bought": "{:,.0f}",
                "Sold": "{:,.0f}",
                "Bias %": "{:+.0f}%",
                "WAPE %": "{:,.0f}%",
            }
        )
        st.dataframe(styled, width="stretch", hide_index=True)

    bought, sold = float(grouped["bought"].sum()), float(grouped["sold"].sum())
    if sold > 0:
        st.caption(
            f"Across every group, the policy bought **{bought:,.0f}** units against "
            f"**{sold:,.0f}** sold (**{bought / sold:.2f}x** ratio). "
            f"Holding cost accounts for **{(summary['holding']/summary['total'])*100:.1f}%** of total cost."
        )



# ============================================================
# Section 3 -- SKUs in detail
# ============================================================

def render_filter_bar(lookup):
    """All six section-3 controls on one row: scope, lookback, policy timing.

    One row of columns rather than stacked full-width boxes. Stacked, the scope
    selectors and the horizon they describe end up in different parts of the
    screen and push the plot below the fold; together, the selection that
    produced the numbers stays visible while you read them.
    """
    has_cat = "cat_id" in lookup.columns
    has_dept = "dept_id" in lookup.columns
    scope_a, scope_b, scope_c, span_d, term_e, term_f = st.columns(6)

    with scope_a:
        cats = ["All"] + sorted(lookup["cat_id"].dropna().unique()) if has_cat else ["All"]
        category = pick_filter("Category", cats, "g_cat")
    if changed("g_applied_cat", category):
        st.session_state.pop("g_dept", None)
        st.session_state.pop("g_item", None)

    scoped = lookup if category == "All" or not has_cat else lookup[lookup["cat_id"] == category]
    with scope_b:
        depts = ["All"] + sorted(scoped["dept_id"].dropna().unique()) if has_dept else ["All"]
        department = pick_filter("Department", depts, "g_dept")
    if changed("g_applied_dept", department):
        st.session_state.pop("g_item", None)

    scoped = scoped if department == "All" or not has_dept else scoped[scoped["dept_id"] == department]
    with scope_c:
        item = pick_filter("Item", ["All"] + sorted(scoped.index.tolist()), "g_item")
    with span_d:
        lookback = pick_filter("History lookback", [f"{d}d" for d in LOOKBACKS], "g_lookback")
    with term_e:
        st.slider("Lead time (days)", LEAD_TIME, LEAD_TIME + 1, LEAD_TIME,
                  disabled=True, key="w_lead")
    with term_f:
        st.slider("Review period (days)", REVIEW_PERIOD, REVIEW_PERIOD + 1, REVIEW_PERIOD,
                  disabled=True, key="w_review")

    if item != "All":
        members, label = [item], item
    elif department != "All":
        members, label = scoped.index.tolist(), department
    elif category != "All":
        members, label = scoped.index.tolist(), category
    else:
        members, label = lookup.index.tolist(), f"All {len(lookup):,} SKUs"
    return members, label, int(lookback.rstrip("d"))


def line_trace(name, frame, value, color, width, hover=None, dash="solid", markers=False):
    """One forecast/actual line trace, so the figure stays readable."""
    template = hover or "%{x|%b %d}<br>%{y:.0f} units<extra>{name}</extra>"
    return go.Scatter(
        x=frame["date"], y=frame[value],
        mode="lines+markers" if markers else "lines", name=name,
        line=dict(color=color, width=width, dash=dash),
        marker=dict(size=5) if markers else None,
        hovertemplate=template.replace("{name}", name))


def daily_order_rate(daily, lead_time, review_period):
    """Order quantity spread over the days it is actually meant to cover.

    A review order covers a whole cycle, not a day: placed on day D it lands on
    D + lead_time and has to hold the store through the next `review_period`
    days. Plotted as one bar on a daily axis it therefore sits ~11x above a
    typical day of sales, and reads as catastrophic over-buying when the cycle
    only carries ~1.6x what it sells. Dividing it back out puts the order and the
    sales line on the same per-day footing, so the gap left on screen is the
    real one.
    """
    rates = pd.Series(0.0, index=pd.DatetimeIndex(daily["date"]))
    for day, qty in zip(daily["date"], daily["order_qty"]):
        if qty <= 0:
            continue
        start = day + pd.Timedelta(days=lead_time)
        for offset in range(review_period):
            covered = start + pd.Timedelta(days=offset)
            if covered in rates.index:
                rates.loc[covered] += qty / review_period
    return rates.reset_index(drop=True)

def build_inventory_figure(inv, members, policy_model):
    """Right column plot: On-hand sawtooth, safety stock, and daily sales bars."""
    fig = go.Figure()
    daily = daily_inventory(inv, members, policy_model)

    if daily.empty:
        return fig

    # Daily Sales Bars
    fig.add_trace(
        go.Bar(
            x=daily["date"],
            y=daily["actual_sales"],
            name="Daily Sales",
            marker_color="rgba(245, 158, 11, 0.35)",
            hovertemplate="%{x|%b %d}<br>%{y:.0f} units sold<extra>Sales</extra>",
        )
    )

    # On-Hand Inventory Line
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["on_hand"],
            mode="lines+markers",
            name="On-Hand Inventory",
            line=dict(color="#16a34a", width=2.5),
            marker=dict(size=4),
            hovertemplate="%{x|%b %d}<br>%{y:.0f} units on-hand<extra>On-Hand</extra>",
        )
    )

    # Safety Stock Target Line
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["safety_stock"],
            mode="lines",
            name="Safety Stock Target",
            line=dict(color="#dc2626", width=1.5, dash="dot"),
            hovertemplate="%{x|%b %d}<br>%{y:.0f} units SS<extra>Safety Stock</extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=10, r=10, t=25, b=10),
        title="",
        yaxis_title="Units",
        xaxis_title="Date",
        barmode="overlay",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
        ),
        hovermode="x unified",
    )
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
        x=ordered["date"], y=ordered["order_qty"], name="Order placed (whole cycle)",
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

    rate_frame = daily.copy()
    rate_frame["order_rate"] = daily_order_rate(daily, LEAD_TIME, REVIEW_PERIOD)
    fig.add_trace(line_trace(
        "Order per day (cycle spread)", rate_frame, "order_rate", "#6366f1", 2,
        "%{x|%b %d}<br>%{y:.0f} units/day<extra>Order per day</extra>", dash="dot"))

    fig.add_vrect(x0=daily["date"].min(), x1=daily["date"].max(),
                  fillcolor="#f1f5f9", opacity=0.55, layer="below", line_width=0)
    fig.update_layout(
        template="plotly_white", height=470, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Units per day", xaxis_title="Date", barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
        hovermode="x unified")
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
    truth = truth_model(preds)
    if truth is None:
        st.error("No model in the holdout carries `real_sales`, so there is no ground "
                 "truth to score against. Re-run the test mode.")
        return
    models = [policy_model, *[m for m in compare if m != policy_model]]
    scored = {m: accuracy_metrics(preds, members, m, truth) for m in models}
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


def render_sku_detail(train, preds, inv, lookup, available):
    """Section 3: model pickers, scope, the aggregated plot, metrics, review table."""
    st.markdown("### 3. SKUs in detail")

    policy_col, compare_col = st.columns(2)
    with policy_col:
        policy_model = st.selectbox(
            "Policy model (drives replenishment)", available,
            format_func=model_label, key="policy_model")
    with compare_col:
        compare = st.multiselect("Also show these forecasts",
                                 [m for m in available if m != policy_model],
                                 format_func=model_label, key="compare_models")

    members, label, days = render_filter_bar(lookup)
    # --- Toggle Switch Controls ---
    show_inventory = st.toggle(
        "Show Inventory Sawtooth & Performance Metrics",
        value=False,
        key="toggle_inventory_view",
    )

    if not show_inventory:
        # Full-width Demand Plot View
        st.caption(f"Showing **{label}** ({len(members):,} SKUs). Risk period = {LEAD_TIME} + "
                    f"{REVIEW_PERIOD} = **{TAU} days**, locked to the loaded run.")
        st.plotly_chart(build_detail_figure(train, preds, inv, members,
                                                policy_model, compare, days), width="stretch")
        st.caption("The order bars are a whole review cycle; the dotted line is that same order "
                    "spread across the days it covers, which is the only one of the two that is "
                    "comparable to a single day of sales.")
        render_metrics_line(preds, members, policy_model, compare)


    else:
        # Full-width Inventory Plot & Aggregated Metrics View
        st.plotly_chart(
            build_inventory_figure(inv, members, policy_model), width="stretch"
        )

        st.markdown(f"#### Aggregated Inventory Performance (`{label}`)")
        perf_df = evaluate_inventory_performance(
            inv, members, policy_model, label
        )

        if not perf_df.empty:
            st.dataframe(perf_df, width="stretch", hide_index=True)
        else:
            st.info("No inventory performance metrics for this selection.")

    # Review-by-review decision log table below
    st.divider()
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
    if provenance["drift"]:
        st.warning("**Stale artifacts.** `"
                   + "`, `".join(provenance["drift"])
                   + "` changed after this holdout was generated, so the code and the numbers "
                   "on this page are out of step. Re-run the test mode before quoting these.")
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

    art = load_artifacts(artifact_signature())
    train, preds, inv = art["train"], art["preds"], art["inv"]
    classes, provenance = art["classes"], art["provenance"]
    if preds.empty or inv.empty:
        st.error("No holdout artifacts in `results/tests/`. Remove the `quit()` in "
                 "`run_pipeline.py` and re-run the test mode.")
        return

    st.title("Reducing inventory cost across 300 SKUs")
    st.caption("Walmart M5 · store `CA_1` · 300 curated SKUs · periodic review")

    available = sorted(preds["model"].dropna().unique().tolist())
    # The model pickers live in section 3, but sections 1 and 2 are scored under
    # the chosen policy, and Streamlit only exposes a widget's value after the
    # widget is created. Seeding both keys up front lets every section read the
    # current choice on the same rerun, so nothing goes stale when they change.
    st.session_state.setdefault("policy_model",
                                "lgbm" if "lgbm" in available else (available[0] if available else None))
    st.session_state.setdefault("compare_models", [])
    st.session_state.setdefault("g_lookback", f"{LOOKBACKS[2]}d")
    policy_model = st.session_state["policy_model"]

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
    render_where_money_goes(summary, lookup, preds, policy_model)

    st.divider()
    render_sku_detail(train, preds, inv, lookup, available)

    st.divider()
    render_notes(provenance)


if __name__ == "__main__":
    main()
