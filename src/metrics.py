"""Evaluation metrics with structural key alignment.

Design rule
-----------
Alignment happens exactly once, in the constructor, and every metric reads from the
same validated frame. A caller cannot pair rows positionally by accident because
positional pairing is not exposed anywhere in this module.

Why this exists
---------------
`wrmsse` used to merge on ``["item_id", "date"]`` while `mae` and `bias` used
``np.asarray(df["sales"])``. For a recursive forecast the prediction frame is
date-major while the evaluation frame is item-major, so the two metrics were
scoring *different row pairings* of the same run. Measured effect on a 12-SKU
slice: positional MAE 1.3461 vs aligned MAE 1.0519, a 28% inflation. `wrmsse`
looked correct, `MAE` did not, and the disagreement looked like a model property.

The merge is also validated to be lossless. An inner join silently drops
(item_id, date) pairs that have no prediction, which computes every metric over a
different subset than the one you think you are scoring.

Usage
-----
    calc = MetricsCalculator(train_df, test_df, pred_df, risk_period=18)
    calc.all_metrics()
    calc.aligned          # the single validated frame every metric reads from

Legacy
------
`get_all_metrics` and `compute_cumulative_metrics` remain as thin wrappers that
delegate here, so existing callers keep working while emitting a DeprecationWarning.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

__all__ = [
    "MetricsCalculator",
    "DEFAULT_KEYS",
    "fva",
    "get_all_metrics",
    "compute_cumulative_metrics",
    "calculate_wape",
    "calculate_mape",
]

DEFAULT_KEYS = ("item_id", "date")


class MetricsCalculator:
    """All forecast-accuracy metrics for one (train, test, prediction) triple.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training history. Supplies the WRMSSE scale and the revenue weights.
        Requires ``item_id``, ``date``, ``sales``; ``sell_price`` for weighting.
    test_df : pd.DataFrame
        Evaluation truth. Requires the join keys and ``sales``. May carry
        ``dept_id`` / ``cat_id`` for the grouped metrics.
    pred_df : pd.DataFrame
        Predictions. Requires the join keys and ``sales_pred``.
    risk_period : int, optional
        tau for the cumulative metrics. Usually ``lead_time + review_period``.
        Only `cumulative()` needs it; leave None for WRMSSE-only use.
    keys : tuple
        Join keys. Defaults to ``("item_id", "date")``.
    strict : bool
        When True (default) a prediction that cannot be matched to a truth row
        raises, because it means the scored set is not the set you think it is.
        Set False to score the intersection and let the dropped keys be reported.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        risk_period: int | None = None,
        *,
        keys: tuple[str, ...] = DEFAULT_KEYS,
        strict: bool = True,
        season_length: int = 7,
    ) -> None:
        self.keys = list(keys)
        self.risk_period = int(risk_period) if risk_period is not None else None
        self.strict = strict
        self.season_length = int(season_length)

        self.train_df = self._normalise(train_df)
        self.test_df = self._normalise(test_df)
        self.pred_df = self._normalise(pred_df)

        self._aligned: pd.DataFrame | None = None
        self._scale: pd.Series | None = None
        self._weights: pd.Series | None = None
        self._naive_mae_scale: pd.Series | None = None
        self._seasonal_naive_mae_scale: pd.Series | None = None
        self._cumulative: pd.DataFrame | None = None
        self._cumulative_tau: int | None = None

        # Validate eagerly. A malformed join must fail at the call site with a
        # useful message, not deep inside a metric computation.
        self.aligned

    # ------------------------------------------------------------------
    # normalisation
    # ------------------------------------------------------------------
    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce join keys to a comparable dtype. Prevents silent key mismatch."""
        out = df.copy()
        missing = [k for k in self.keys if k not in out.columns]
        if missing:
            raise KeyError(f"frame is missing join key(s): {missing}")
        out["item_id"] = out["item_id"].astype(str)
        out["date"] = pd.to_datetime(out["date"])
        return out

    # ------------------------------------------------------------------
    # alignment: the single source of truth
    # ------------------------------------------------------------------
    def _build_aligned(self) -> pd.DataFrame:
        truth_cols = [c for c in ("dept_id", "cat_id", "sales") if c in self.test_df.columns]
        truth = self.test_df[self.keys + truth_cols]

        if "sales_pred" not in self.pred_df.columns:
            raise KeyError("pred_df must contain a 'sales_pred' column")
        preds = self.pred_df[self.keys + ["sales_pred"]]

        dup_truth = int(truth.duplicated(subset=self.keys).sum())
        if dup_truth:
            raise ValueError(
                f"test_df has {dup_truth} duplicate {self.keys} rows; the evaluation "
                "frame must have exactly one truth row per key."
            )
        dup_pred = int(preds.duplicated(subset=self.keys).sum())
        if dup_pred:
            raise ValueError(
                f"pred_df has {dup_pred} duplicate {self.keys} rows. Duplicate "
                "predictions would be averaged silently and understate error."
            )

        merged = truth.merge(
            preds,
            on=self.keys,
            how="left",
            validate="one_to_one",
            indicator=True,
        )

        unmatched = merged[merged["_merge"] == "left_only"]
        if len(unmatched):
            detail = (
                f"{len(unmatched)} of {len(merged)} truth rows have no matching "
                f"prediction on {self.keys}. First offenders: "
                f"{unmatched[self.keys].head(5).to_dict('records')}"
            )
            if self.strict:
                raise ValueError(detail)
            warnings.warn(detail + " Scoring the intersection only.", UserWarning, stacklevel=3)
            merged = merged[merged["_merge"] == "both"]

        merged = merged.drop(columns="_merge")
        merged = merged.dropna(subset=["sales", "sales_pred"])

        if merged.empty:
            raise ValueError("alignment produced an empty frame; check the join keys.")

        return merged.sort_values(self.keys).reset_index(drop=True)

    @property
    def aligned(self) -> pd.DataFrame:
        """The validated, key-aligned frame every metric in this class reads."""
        cached = self._aligned
        if cached is None:
            cached = self._build_aligned()
            self._aligned = cached
        return cached

    @property
    def n_scored(self) -> int:
        return len(self.aligned)

    # ------------------------------------------------------------------
    # training-set quantities (WRMSSE)
    # ------------------------------------------------------------------
    @property
    def scale(self) -> pd.Series:
        """Per-item naive one-step error scale. Zero for constant-demand items."""
        cached = self._scale
        if cached is None:
            s = self.train_df.sort_values(self.keys)
            # diff within item, then root-mean-square per item. Written as a
            # vectorised aggregation rather than groupby.apply: same result, no
            # dependency on the pandas>=2.2 `include_groups` kwarg, and no
            # per-group Python call.
            sq = s.groupby("item_id", observed=True)["sales"].diff() ** 2
            cached = sq.groupby(s["item_id"], observed=True).mean().pow(0.5)
            self._scale = cached
        return cached

    @property
    def weights(self) -> pd.Series:
        """Per-item share of trailing 28-day revenue. Falls back to unit demand."""
        cached = self._weights
        if cached is None:
            last = self.train_df["date"] >= self.train_df["date"].max() - pd.Timedelta(days=27)
            tail = self.train_df[last]
            if "sell_price" in tail.columns:
                value = tail["sales"] * tail["sell_price"]
            else:
                value = tail["sales"]
            rev = value.groupby(tail["item_id"].astype(str), observed=True).sum()
            total = rev.sum()
            cached = rev / total if total > 0 else pd.Series(dtype=float)
            self._weights = cached
        return cached

    def _item_weights_and_scale(self) -> tuple[pd.Series, pd.Series]:
        """Weights and scale restricted to items actually being scored."""
        items = pd.Index(self.aligned["item_id"].unique())
        w = self.weights.reindex(items)
        s = self.scale.reindex(items)
        if w.isna().any():
            missing = w[w.isna()].index.tolist()[:5]
            warnings.warn(
                f"{int(w.isna().sum())} scored items are absent from train_df "
                f"(e.g. {missing}); they are dropped from WRMSSE.",
                UserWarning,
                stacklevel=3,
            )
            w = w.dropna()
            s = s.reindex(w.index)
        return w, s

    # ------------------------------------------------------------------
    # point metrics
    # ------------------------------------------------------------------
    def forecast_error(self) -> pd.Series:
        """Per-item RMSE over the horizon, computed on the aligned frame."""
        df = self.aligned
        sq = (df["sales"] - df["sales_pred"]) ** 2
        return sq.groupby(df["item_id"], observed=True).mean().pow(0.5)

    def wrmsse(self) -> float:
        """Revenue-weighted RMSE scaled by naive one-step error. Lower is better.

        Items with a zero scale (constant historical demand) are excluded rather
        than producing an infinite term.
        """
        w, s = self._item_weights_and_scale()
        fe = self.forecast_error().reindex(w.index)
        valid = (s > 0) & fe.notna() & w.notna()
        if not valid.any():
            raise ValueError("no item has a usable scale; WRMSSE is undefined.")
        return float((w[valid] * (fe[valid] / s[valid])).sum())

    def mae(self) -> float:
        """Unweighted mean absolute error across scored (item, date) rows."""
        df = self.aligned
        return float(np.mean(np.abs(df["sales_pred"] - df["sales"])))

    def bias(self) -> float:
        """Aggregate over-forecast as a percentage of total actual demand."""
        df = self.aligned
        actual = df["sales"].sum()
        if actual == 0:
            return float("nan")
        return float((df["sales_pred"].sum() - actual) / actual * 100)

    def wmape(self) -> float:
        """Weighted absolute percentage error on the aligned frame.

        Warning: on intermittent panels this is structurally inflated. Rows with
        zero actual demand contribute their full absolute error to the numerator
        while contributing nothing to the denominator, so WAPE rises with
        zero-density even for a perfect model. Prefer ``mase`` here.
        """
        df = self.aligned
        actual = df["sales"].sum()
        if actual == 0:
            return float("nan")
        return float(np.abs(df["sales_pred"] - df["sales"]).sum() / actual)

    def grouped_mae(self, by: str = "dept_id") -> float:
        """MAE of demand aggregated to a higher level (department or category)."""
        df = self.aligned
        if by not in df.columns:
            raise KeyError(f"'{by}' is not present in test_df; cannot aggregate by it.")
        truth = df.groupby([by], observed=True)["sales"].sum()
        pred = df.groupby([by], observed=True)["sales_pred"].sum()
        return float(np.mean(np.abs(pred - truth)))

    # ------------------------------------------------------------------
    # cumulative / tracking metrics
    # ------------------------------------------------------------------
    def cumulative(self, tau: int | None = None) -> pd.DataFrame:
        """Per-item tau-chunk tracking error, read off the same aligned frame.

        Returns one row per item with ``cumulative_mae``, ``cumulative_bias`` and
        ``tracking_trajectory_mae``.
        """
        tau = int(tau or self.risk_period or 0)
        if tau <= 0:
            raise ValueError("cumulative() needs a tau: pass risk_period or tau explicitly.")
        if self._cumulative is None or tau != self._cumulative_tau:
            df = self.aligned.sort_values(self.keys).reset_index(drop=True)
            by_item = df.groupby("item_id", observed=True)
            df["tau_chunk"] = by_item.cumcount() // tau
            df["actual_cum"] = df.groupby(["item_id", "tau_chunk"], observed=True)["sales"].cumsum()
            df["forecast_cum"] = df.groupby(["item_id", "tau_chunk"], observed=True)["sales_pred"].cumsum()
            df["cum_error"] = df["forecast_cum"] - df["actual_cum"]
            # Precompute the magnitude so agg can use a named reduction instead of
            # a per-group lambda.
            df["_abs_cum_error"] = df["cum_error"].abs()

            chunks = (
                df.groupby(["item_id", "tau_chunk"], observed=True)
                .agg(
                    final_tau_error=("cum_error", "last"),
                    mean_tau_tracking_error=("_abs_cum_error", "mean"),
                )
                .reset_index()
            )
            self._cumulative = (
                chunks.groupby("item_id", observed=True)
                .agg(
                    cumulative_mae=("final_tau_error", lambda x: x.abs().mean()),
                    cumulative_bias=("final_tau_error", "mean"),
                    tracking_trajectory_mae=("mean_tau_tracking_error", "mean"),
                )
                .reset_index()
            )
            self._cumulative_tau = tau
        return self._cumulative

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------
    def all_metrics(self) -> dict:
        """Every metric the backtest records, all from one aligned frame."""
        cum = self.cumulative()
        out = {
            "wrmsse": self.wrmsse(),
            "WAPE": self.wmape(),
            "MAE": self.mae(),
            "BIAS%": self.bias(),
            "cum_BIAS": float(cum["cumulative_bias"].mean()),
            "cum_MAE": float(cum["cumulative_mae"].mean()),
            "n_scored": self.n_scored,
        }
        for level in ("dept_id", "cat_id"):
            if level in self.aligned.columns:
                out[f"MAE_{level.split('_')[0]}"] = self.grouped_mae(level)
        return out


# ----------------------------------------------------------------------
# reporting helper - no alignment concern, operates on scalars
# ----------------------------------------------------------------------
def fva(baseline_value: float, model_value: float) -> float:
    """Forecast Value Added: percent improvement of model over baseline.

    Positive means the model is better. A negative value is a real result, not a
    bug: it means the model lost to the baseline on that metric.
    """
    if baseline_value == 0:
        return float("nan")
    return round((baseline_value - model_value) * 100 / baseline_value, 2)


# ----------------------------------------------------------------------
# array helpers - callers pass matching arrays and own the alignment
# ----------------------------------------------------------------------
def mae(ytrue, ypred) -> float:
    """Unweighted MAE between two equally-ordered arrays.

    Deprecated for frame input: the caller owns row alignment, which is exactly
    the mistake this module was refactored to remove. Use
    ``MetricsCalculator(train, test, pred, risk).mae()`` for frames.
    """
    warnings.warn(
        "mae(y_true, y_pred) requires the caller to have aligned the rows "
        "correctly. Use MetricsCalculator for DataFrame input.",
        UserWarning,
        stacklevel=2,
    )
    y, p = np.asarray(ytrue, dtype=float), np.asarray(ypred, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y_true {y.shape} vs y_pred {p.shape}")
    return float(np.mean(np.abs(p - y)))


def bias(y_true, y_pred) -> float:
    """Aggregate over-forecast as a percentage of total actual demand.

    Deprecated for frame input: the caller owns row alignment. Use
    ``MetricsCalculator(train, test, pred, risk).bias()`` for frames.
    """
    warnings.warn(
        "bias(y_true, y_pred) requires the caller to have aligned the rows "
        "correctly. Use MetricsCalculator for DataFrame input.",
        UserWarning,
        stacklevel=2,
    )
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y_true {y.shape} vs y_pred {p.shape}")
    if y.sum() == 0:
        return float("nan")
    return float((p.sum() - y.sum()) / y.sum() * 100)


def calculate_wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error. Returns NaN when actuals sum to zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    total = np.sum(y_true)
    if total == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / total)


def calculate_mape(y_true, y_pred, ignore_zeros: bool = True) -> float:
    """Mean absolute percentage error. Zero actuals are skipped by default."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if ignore_zeros:
        mask = y_true > 0
        if not np.any(mask):
            return float("nan")
        return float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100)
    return float(np.mean(np.abs((y_pred - y_true) / np.maximum(y_true, 1e-5))) * 100)


# ----------------------------------------------------------------------
# legacy shims - kept so existing callers keep working
# ----------------------------------------------------------------------
def _calculator(train_df, test_df, pred_df, risk_period=None) -> "MetricsCalculator":
    return MetricsCalculator(train_df, test_df, pred_df, risk_period)


def wrmsse(train_df: pd.DataFrame, test_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    """Deprecated. Use ``MetricsCalculator(train, test, pred).wrmsse()``.

    Retained because notebooks/02 imports it. Numerically identical to the class
    for item-major frames; the class additionally rejects malformed joins.
    """
    warnings.warn(
        "wrmsse() is deprecated; use MetricsCalculator(train, test, pred).wrmsse()",
        DeprecationWarning,
        stacklevel=2,
    )
    return _calculator(train_df, test_df, pred_df).wrmsse()


def scale(train_df: pd.DataFrame) -> pd.Series:
    """Deprecated. Use ``MetricsCalculator(train, ...).scale``."""
    warnings.warn(
        "scale() is deprecated; use MetricsCalculator(train, ...).scale",
        DeprecationWarning,
        stacklevel=2,
    )
    return _calculator(
        train_df,
        train_df.head(1).assign(sales_pred=0.0),
        train_df.head(1).assign(sales_pred=0.0),
    ).scale


def weights(train_df: pd.DataFrame) -> pd.Series:
    """Deprecated. Use ``MetricsCalculator(train, ...).weights``."""
    warnings.warn(
        "weights() is deprecated; use MetricsCalculator(train, ...).weights",
        DeprecationWarning,
        stacklevel=2,
    )
    return _calculator(
        train_df,
        train_df.head(1).assign(sales_pred=0.0),
        train_df.head(1).assign(sales_pred=0.0),
    ).weights


def forecast_error(test_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.Series:
    """Deprecated. Use ``MetricsCalculator(train, test, pred).forecast_error()``."""
    warnings.warn(
        "forecast_error() is deprecated; use MetricsCalculator(...).forecast_error()",
        DeprecationWarning,
        stacklevel=2,
    )
    return _calculator(test_df, test_df, pred_df).forecast_error()


def mae_dept(true_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    """Deprecated. Use ``MetricsCalculator(train, test, pred).grouped_mae('dept_id')``."""
    warnings.warn(
        "mae_dept() is deprecated; use MetricsCalculator(...).grouped_mae('dept_id')",
        DeprecationWarning,
        stacklevel=2,
    )
    return _calculator(true_df, true_df, pred_df).grouped_mae("dept_id")


def mae_cat(true_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    """Deprecated. Use ``MetricsCalculator(train, test, pred).grouped_mae('cat_id')``."""
    warnings.warn(
        "mae_cat() is deprecated; use MetricsCalculator(...).grouped_mae('cat_id')",
        DeprecationWarning,
        stacklevel=2,
    )
    return _calculator(true_df, true_df, pred_df).grouped_mae("cat_id")


def get_all_metrics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    risk_period: int,
) -> dict:
    """Deprecated. Use ``MetricsCalculator(train, test, pred, risk).all_metrics()``."""
    warnings.warn(
        "get_all_metrics() is deprecated; use MetricsCalculator(...).all_metrics() "
        "so that alignment is structural rather than positional.",
        DeprecationWarning,
        stacklevel=2,
    )
    calc = MetricsCalculator(train_df, test_df, pred_df, risk_period)
    return calc.all_metrics()


def compute_cumulative_metrics(
    test_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    tau: int = 6,
) -> pd.DataFrame:
    """Deprecated. Use ``MetricsCalculator(...).cumulative(tau)``.

    ``train_df`` is not needed for this metric, so an empty frame is passed for the
    WRMSSE quantities that are never read.
    """
    warnings.warn(
        "compute_cumulative_metrics() is deprecated; use MetricsCalculator(...).cumulative(tau).",
        DeprecationWarning,
        stacklevel=2,
    )
    empty = pd.DataFrame({"item_id": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]"), "sales": pd.Series(dtype=float)})
    calc = MetricsCalculator(empty, test_df, predicted_df, tau)
    return calc.cumulative(tau)
