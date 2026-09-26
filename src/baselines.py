"""Forecast baselines for the retail demand panel.

The module exposes a single :class:`Baseline` class with one method per
baseline, rather than a class per baseline. Dispatch is a name lookup, so
``backtest_engine`` can iterate over baseline names without an ``if/elif``
chain that silently skips unrecognised entries.

Method status
-------------
``seasonal_naive`` and ``simple_moving_average`` are complete. The remaining
methods raise :class:`NotImplementedError`; each docstring specifies the
algorithm, the failure modes specific to this panel, and the output
validation the method is expected to satisfy.

Return contract
---------------
Every method returns one row per ``test_df`` row with the columns
``item_id``, ``date``, ``sales_pred`` and ``real_sales``.

Quantile columns are opt-in. Constructing ``Baseline`` without a ``quantiles``
argument leaves them absent, which is the default. When supplied, every method
emits Gaussian intervals derived from its own point forecast, so no method is
required to implement interval logic.
"""
from __future__ import annotations

import inspect
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm


class Baseline:
    """Univariate per-SKU forecast baselines.

    Invariants
    ----------
    * ``train_df`` is the full training window; its maximum date is the
      forecast origin. All conditioning information derives from it.
    * Forecasts are never conditioned on ``test_df['sales']``. The
      :meth:`_attach_actuals` merge supplies actuals for scoring only.
    * ``item_id`` is a pandas Categorical declaring all 3,049 store SKUs while
      a given frame materialises only a subset. Every groupby requires
      ``observed=True``; omitting it yields thousands of all-NaN phantom
      groups and incorrect aggregates.
    * Predictions are clipped at zero. Negative demand is not a forecast.
    * Levels are computed per item. A global aggregate baseline is not a
      valid comparator.

    Parameter placement
    -------------------
    ``train_df`` and ``test_df`` are constructor arguments because every
    method consumes the same two frames. This permits validation once and
    reuse of derived per-item structures via :meth:`_sorted_train`.

    Method hyperparameters remain method arguments, since their meaning is
    method-specific: ``alpha`` for Croston, ``lag_days`` for seasonal naive.

    Instance scope
    --------------
    A separate instance is required per backtest window. Constructing once
    outside the window loop would carry per-item state across windows.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        quantiles: tuple[float, ...] | None = None,
    ) -> None:
        self._check_inputs(train_df, test_df)

        # Minimal copies, so a method that sorts or mutates cannot corrupt the
        # caller's frames for subsequent methods.
        self.train = train_df[["item_id", "date", "sales"]].copy()
        self.test = test_df[["item_id", "date"]].copy()
        self.test = self.test.drop_duplicates(subset=["item_id", "date"])

        # Retained solely so _attach_actuals can recover dept_id, cat_id and
        # real_sales. Method implementations must not read sales from this.
        self._scoring_src = test_df

        self.horizon = int(self.test["date"].nunique())
        if self.horizon < 1:
            raise ValueError("test_df contains no dates to forecast")

        self._sorted_cache: pd.DataFrame | None = None

        # Quantile output is opt-in. When enabled, every method receives
        # quantile columns derived from its own point forecast, so no method
        # has to implement interval logic.
        self.quantiles = self._check_quantiles(quantiles)
        self._item_std: pd.Series | None = None
        if self.quantiles is not None:
            self._item_std = (
                self.train.groupby("item_id", observed=True)["sales"]
                .std()
                .fillna(0.0)
            )

    def _check_quantiles(
        self, quantiles: tuple[float, ...] | None
    ) -> tuple[float, ...] | None:
        """Validate requested quantile levels. ``None`` disables quantile output."""
        if quantiles is None:
            return None
        levels = tuple(sorted(set(float(q) for q in quantiles)))
        if not levels:
            raise ValueError("quantiles was supplied but contains no levels")
        invalid = [q for q in levels if not 0.0 < q < 1.0]
        if invalid:
            raise ValueError(
                f"quantile levels must lie strictly between 0 and 1, got {invalid}"
            )
        return levels

    def _sorted_train(self) -> pd.DataFrame:
        """Return the training frame sorted by ``(item_id, date)``, computed once.

        Methods that group, shift, or difference within an item require a
        stable per-item date order, which pandas does not guarantee on the
        caller's input.
        """
        if self._sorted_cache is None:
            self._sorted_cache = self.train.sort_values(
                ["item_id", "date"], kind="mergesort"
            ).reset_index(drop=True)
        return self._sorted_cache

    def seasonal_naive(self, lag_days: int = 28) -> pd.DataFrame:
        """Forecast date *d* from the item's own sales on *d* minus ``lag_days``.

        Parameters
        ----------
        lag_days : int
            Seasonal offset. The default of 28 matches the project's forecast
            horizon. Note that the panel is weekly-seasonal and the feature
            set includes ``lag_7``; offsets of 7 and 28 are both defensible
            and reporting both would clarify how much of the model's
            advantage is attributable to weekly structure.
        """
        if lag_days < 1:
            raise ValueError(f"lag_days must be >= 1, got {lag_days}")

        test = self.test.copy()
        test["source_date"] = test["date"] - pd.Timedelta(days=lag_days)

        pred_df = test.merge(
            self.train.rename(columns={"date": "source_date", "sales": "sales_pred"}),
            on=["item_id", "source_date"],
            how="left",
        )
        if pred_df["sales_pred"].isna().any():
            n = int(pred_df["sales_pred"].isna().sum())
            raise ValueError(
                f"seasonal_naive: {n} rows have no source date. The training "
                f"window does not reach back {lag_days} days for all test "
                f"dates and items. A smaller lag_days or a longer training "
                f"window resolves this."
            )

        return self._finish(pred_df, "seasonal_naive").drop(columns=["source_date"])

    def simple_moving_average(self, window_days: int = 180) -> pd.DataFrame:
        """Flat per-item forecast: mean sales over the last ``window_days`` of training data.

        Parameters
        ----------
        window_days : int
            Trailing window length. 180 is approximately six months.

        Notes
        -----
        Retained as an ablation rather than a headline baseline. A flat average
        cannot represent day-of-week structure, and the panel is
        weekly-seasonal, so this understates what a non-seasonal baseline
        achieves. Comparing it against
        :meth:`seasonal_moving_average` isolates the contribution of the
        seasonal component.
        """
        if window_days < 1:
            raise ValueError(f"window_days must be >= 1, got {window_days}")

        train_window = self.train[
            self.train["date"] > self.train["date"].max() - pd.Timedelta(days=window_days)
        ]
        item_stats = (
            train_window.groupby("item_id", observed=True)["sales"]
            .agg(sales_pred="mean")
            .reset_index()
        )

        pred_df = self.test.merge(item_stats, on="item_id", how="left")
        if pred_df["sales_pred"].isna().any():
            n = int(pred_df["sales_pred"].isna().sum())
            raise ValueError(
                f"simple_moving_average: {n} test rows have no matching item_id "
                f"in the last {window_days} days of training data, indicating "
                f"new or fully dormant items."
            )

        return self._finish(pred_df, "simple_moving_average")

    def seasonal_moving_average(
        self,
        season_length: int = 7,
        extrapolate_trend: bool = False,
    ) -> pd.DataFrame:
        """Classical multiplicative decomposition: decouple, then re-couple.

        Parameters
        ----------
        season_length : int
            Period *m* of the seasonal cycle. 7 for a weekly retail panel.
        extrapolate_trend : bool
            When False the final available centred trend is held flat across
            the horizon. When True a line is fitted through the non-NaN trend
            points and extrapolated.

        Algorithm
        ---------
        1. Trend. 
        2. Decouple. ``detrended = sales / trend``. Periods where the trend
           is zero, negative, or undefined require explicit handling, since
           they otherwise propagate NaN into the seasonal means.
        3. Seasonal index. Average ``detrended`` within each seasonal slot
           (day-of-week when *m* = 7). Slots with no observations remain NaN.
        4. Normalise. ``index = index * m / index.sum()``. Omitting this
           step biases every forecast by the normalisation error across the
           full horizon. The indices must sum to *m* after normalisation.
        5. Re-couple. ``forecast = trend_extrapolated * index[slot]``.
           Centring leaves the final *m*/2 periods without a trend value, so
           the most recent available trend sits at ``origin - m/2`` rather
           than at the origin.
        6. Clip at zero.

        """
        train = self._sorted_train() 
        train= train[
                    train["date"] > train["date"].max() - pd.Timedelta(days=365)
                ]
        grp = train.groupby('item_id',observed=True)['sales']
        # 
        level = grp.transform(lambda s: s.rolling(season_length,min_periods=1).mean())
        # detrending, 
        detrended = train['sales']/ level 

        ## season
        season = (detrended.groupby([train['item_id'],train['date'].dt.dayofweek],
        observed=True).mean().rename('season_factor').reset_index())

        ## normalize 
        season['season_factor'] = (season['season_factor']*season_length
                                   /season.groupby('item_id')['season_factor'].transform('sum'))

        # calculate current base level using deseasonalized recent sales 
        train['dayofweek'] = train['date'].dt.dayofweek 
        train = train.merge(season,on=['item_id','dayofweek'],how='left')
        train['sales_deseason'] = train['sales']/train['season_factor']

        item_base = (
            train.groupby("item_id", observed=True)["sales_deseason"]
            .agg(sales_pred="mean")
            .reset_index() # this serves as base 
        )

        preds = (self.test
                 .assign(dayofweek=lambda d:d['date'].dt.dayofweek)
                 .merge(item_base,on='item_id',how='left')
                 .merge(season,on=['item_id','dayofweek'],how='left'))

        preds['sales_pred'] = (preds['sales_pred']*preds['season_factor']).clip(lower=0)

        return self._finish(preds,'seasonal_moving_average')

    def croston(self, alpha: float = 0.1) -> pd.DataFrame:
        """Croston's method for intermittent demand.

        Croston is the primary non-seasonal baseline for this panel: 93.3% of
        items are classified intermittent or erratic under the Syntetos-Boylan
        demand classification. The method requires no dependency beyond
        pandas and numpy.

        Parameters
        ----------
        alpha : float
            Exponential smoothing weight shared by both smoothers. Sitter's
            rule of thumb is 0.1. ``alpha = 1`` degenerates to the naive
            estimator.

        Algorithm
        ---------
        Let the non-zero observations be ``(t_i, z_i)`` for ``i = 1..n``, and
        let ``q_i = t_i - t_{i-1}`` be the number of periods elapsed since the
        preceding non-zero observation.

        1. Mask non-zero periods. Zero periods define the gaps but are never
           treated as demand observations.
        2. Compute gaps. The first gap is not observable within the window;
           seed it with the median observed gap or with 1, and record which
           convention was used.
        3. Seed the smoothers at ``z_hat = z_1`` and ``q_hat = seed``.
           Seeding from zero distorts the initial forecast, and the
           distortion decays slowly at typical alpha values.
        4. Update both smoothers on non-zero periods only, using a common
           alpha against their own previous values::

               z_hat = alpha * z_i + (1 - alpha) * z_hat
               q_hat = alpha * q_i + (1 - alpha) * q_hat

           Zero periods leave both quantities unchanged.
        5. Forecast ``z_hat / q_hat``, applied uniformly across the horizon.
           Croston carries neither seasonality nor trend by construction.

        Panel-specific considerations
        -----------------------------
        - Items with ``n == 0`` admit no forecast. Emit zero and record the
          count; NaN would propagate silently through groupby means and
          metrics.
        - Items with ``n == 1`` have no observable gap. Fall back to ``z_1``
          and record the count. This case is realised in the current panel.
        - ``q_hat == 0`` requires a guard before division.
        - A moving average on this panel reports a near-zero level and
          supports little inventory. Croston addresses demand size and
          demand frequency separately, which is the behaviour intermittent
          inventory actually requires.
        - Group with ``observed=True``.

        Validation
        ----------
        - No NaN values and no negative values.
        - ``yhat <= max(z_i)`` for every item.
        - ``z_hat / q_hat`` is materially smaller than the mean of the
          non-zero spikes; a ratio near that mean indicates the gaps are
          being counted incorrectly.
        - The fraction of items routed to the ``n <= 1`` fallback is reported.
        """
        
        raise NotImplementedError(
            "croston: not implemented. The class docstring specifies the "
            "smoothing recursion, the seeding convention, and the n == 0, "
            "n == 1 and q_hat == 0 guards."
        )

    def croston_sba(self, alpha: float = 0.1) -> pd.DataFrame:
        """Syntetos-Boylan approximation of Croston.

        Identical to :meth:`croston` except for the bias correction applied to
        the final forecast::

            yhat = (1 - alpha/2) * z_hat / q_hat

        Plain Croston is positively biased on lumpy series, so SBA is the
        more defensible choice on this panel. An SBA result that does not
        outperform Croston indicates a defect in the Croston implementation.

        Parameters
        ----------
        alpha : float
            Smoothing weight, as in :meth:`croston`.
        """
        raise NotImplementedError(
            "croston_sba: not implemented. Reuse the croston smoothing and "
            "apply the (1 - alpha/2) correction to the final ratio."
        )

    def tsb(self, alpha: float = 0.1, beta: float = 0.1) -> pd.DataFrame:
        """Teunter-Syntetos-Babai: Croston extended with demand obsolescence.

        The distinguishing feature is a second smoothing weight applied to
        demand probability, allowing the forecast to decay as a SKU
        approaches discontinuation::

            p_hat = beta * 1{sales_t > 0} + (1 - beta) * p_hat
            yhat = p_hat * z_hat / q_hat

        ``p_hat`` is updated on every period, including zero-sales periods,
        while ``z_hat`` and ``q_hat`` update only on non-zero periods. That
        asymmetry is the substance of the method.

        Parameters
        ----------
        alpha : float
            Smoothing weight for demand size and interval, as in
            :meth:`croston`.
        beta : float
            Smoothing weight for demand probability. ``beta = 0`` reduces the
            method to plain Croston, which provides a direct correctness
            check on the implementation.

        Notes
        -----
        This is the only method in the module that responds to discontinued
        demand. ``get_items_with_min_history`` in ``src/utils.py`` supports
        constructing that case study.
        """
        raise NotImplementedError(
            "tsb: not implemented. The class docstring specifies the p_hat "
            "smoother and the beta = 0 equivalence check against croston."
        )

    def theta(
        self,
        season_length: int = 7,
        deseasonalize: bool = True,
    ) -> pd.DataFrame:
        """Theta method, the primary statistical baseline for this panel.

        Theta outperformed competing methods in the M3 and M4 forecasting
        competitions and requires minimal tuning. It is the recommended
        statistical reference point for evaluating the machine-learning
        models.

        Parameters
        ----------
        season_length : int
            Period passed to ``ThetaModel``. Set to 1 with
            ``deseasonalize=False`` where the training window cannot support
            the requested period.
        deseasonalize : bool
            Whether the series is deseasonalised before the Theta components
            are fitted.

        Algorithm
        ---------
        Theta is the sum of a simple exponential smoothing of the original
        series and half a simple exponential smoothing of the linearly
        detrended series, with the drift contribution halved. Deseasonalisation
        is performed internally when ``deseasonalize`` is True.

        Panel-specific considerations
        -----------------------------
        - The method is univariate, so this entails one fit per item across
          300 items. statsmodels raises on degenerate input; each fit must be
          guarded independently and the failure count recorded. A high
          failure rate read as poor performance is a defect, not a result.
        - All-zero items admit no fit and require the same fallback as
          :meth:`croston`.
        - ``statsmodels`` 0.14.6 and
          ``statsmodels.tsa.forecasting.theta.ThetaModel`` are available in
          the project environment.

        Validation
        ----------
        - Per-item fit failure count is reported.
        - Output row count matches the test frame for every item.
        - No negative values after forecasting.
        """
        raise NotImplementedError(
            "theta: not implemented. The class docstring specifies the "
            "per-item fit procedure, the required exception handling, and the "
            "degenerate-series fallbacks."
        )

    def holt_winters(
        self,
        season_length: int = 7,
        damped_trend: bool = True,
    ) -> pd.DataFrame:
        """Triple exponential smoothing (Holt-Winters) with additive seasonality.

        Parameters
        ----------
        season_length : int
            Number of periods in a seasonal cycle. The method requires at
            least two complete cycles.
        damped_trend : bool
            Whether the trend component is damped. Damping is generally
            preferable at this forecast horizon.

        Panel-specific considerations
        -----------------------------
        - 93.3% of items are intermittent or erratic, a regime for which
          Holt-Winters was not designed. Additive seasonality fitted to a
          series that is 56% zero risks estimating a seasonal pattern from
          noise. A shortfall relative to :meth:`croston` is the expected
          result and should be reported as evidence that the method is
          unsuited to intermittent demand, rather than as a ranking.
        - Restricting the comparison to the smooth and lumpy demand classes
          provides a more favourable setting if a fair-contest result is
          required.
        - Multiplicative seasonality is unusable on this panel, since it
          divides by a seasonal factor that zeros render degenerate.
        - As with :meth:`theta`, one fit is required per item, convergence
          failures must be caught individually, and the count recorded.
        - ``statsmodels`` 0.14.6 and
          ``statsmodels.tsa.holtwinters.ExponentialSmoothing`` are available
          in the project environment.

        Validation
        ----------
        - Fit failure count is reported.
        - No negative values after forecasting.
        """
        raise NotImplementedError(
            "holt_winters: not implemented. The class docstring specifies the "
            "model configuration, the minimum-cycle requirement, and the "
            "expected behaviour on intermittent items."
        )

    # ==================================================================
    # DISPATCH
    # ==================================================================

    METHOD_NAMES = [
        "seasonal_naive",
        "simple_moving_average",
        "seasonal_moving_average",
        "croston",
        "croston_sba",
        "tsb",
        "theta",
        "holt_winters",
    ]

    def implemented(self) -> list[str]:
        """Return the names of methods that are complete.

        Detection inspects the method body for an explicit
        ``raise NotImplementedError``. The docstrings are not a reliable
        indicator, since unimplemented methods carry their specification
        there.
        """
        out = []
        for name in self.METHOD_NAMES:
            source = inspect.getsource(getattr(self, name))
            if "raise NotImplementedError" not in source:
                out.append(name)
        return out

    def run(self, method: str, **kwargs: Any) -> pd.DataFrame:
        """Invoke a baseline method by name.

        Parameters
        ----------
        method : str
            One of :attr:`METHOD_NAMES`.
        **kwargs
            Method-specific hyperparameters.

        Raises
        ------
        KeyError
            If ``method`` is not a registered baseline name.

        Examples
        --------
        One instance per backtest window, then iterate baseline names::

            bl = Baseline(window_train, window_test)
            for name in baseline_names:
                preds = bl.run(name, **params[name])
        """
        if method not in self.METHOD_NAMES:
            raise KeyError(f"Unknown baseline {method!r}. Known: {self.METHOD_NAMES}")
        return getattr(self, method)(**kwargs)

    # ==================================================================
    # SHARED PLUMBING
    # ==================================================================

    def _check_inputs(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Validate column presence and warn on temporal overlap. Runs once, from __init__."""
        for frame, label, required in (
            (train_df, "train_df", ["item_id", "date", "sales"]),
            (test_df, "test_df", ["item_id", "date"]),
        ):
            missing = [c for c in required if c not in frame.columns]
            if missing:
                raise KeyError(f"Baseline: {label} is missing column(s) {missing}")

        if train_df["date"].max() >= test_df["date"].min():
            warnings.warn(
                f"Baseline: the training window ends {train_df['date'].max().date()}, "
                f"on or after the forecast window start {test_df['date'].min().date()}. "
                f"This overlap risks leakage.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _attach_actuals(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """Merge ``real_sales`` and the ``dept_id`` / ``cat_id`` keys for scoring."""
        source = self._scoring_src
        keep = [c for c in ("dept_id", "cat_id", "sales") if c in source.columns]
        if not keep:
            return pred_df
        return pred_df.merge(
            source[["item_id", "date", *keep]].rename(columns={"sales": "real_sales"}),
            on=["item_id", "date"],
            how="left",
        )

    def _validate_output(self, pred_df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Assert the invariants that would otherwise fail silently downstream.

        The conditions checked here do not raise inside the method
        implementations, but each corrupts groupby aggregates and metrics
        without surfacing an error at the point of origin.
        """
        expected = len(self.test)
        if len(pred_df) != expected:
            raise AssertionError(
                f"{method}: produced {len(pred_df)} rows against an expected "
                f"{expected}. A merge must not drop or duplicate test rows."
            )

        n_nan = int(pred_df["sales_pred"].isna().sum())
        if n_nan:
            raise AssertionError(
                f"{method}: {n_nan} NaN predictions. NaN propagates silently "
                f"through groupby means and metrics; undefined cases require "
                f"explicit handling."
            )

        n_negative = int((pred_df["sales_pred"] < 0).sum())
        if n_negative:
            raise AssertionError(
                f"{method}: {n_negative} negative predictions. Predictions must "
                f"be clipped at zero."
            )

        n_zero = int((pred_df["sales_pred"] == 0).sum())
        if n_zero > 0.9 * len(pred_df):
            warnings.warn(
                f"{method}: {n_zero} of {len(pred_df)} predictions are exactly "
                f"zero. This is expected for Croston on dormant items and "
                f"indicates a defect elsewhere.",
                RuntimeWarning,
                stacklevel=2,
            )

        return pred_df

    def _finish(self, pred_df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Standard method tail: quantiles, actuals, then validation."""
        pred_df = self._attach_quantiles(pred_df, method)
        pred_df = self._attach_actuals(pred_df)
        pred_df = self._validate_output(pred_df, method)
        return self._validate_quantiles(pred_df, method)

    # ------------------------------------------------------------------
    # QUANTILE OUTPUT (opt-in)
    # ------------------------------------------------------------------

    def _spread_for(self, pred_df: pd.DataFrame) -> pd.Series:
        """Return the per-row dispersion used to form Gaussian intervals.

        The default is the per-item standard deviation of training sales,
        broadcast to the prediction rows. Methods whose error structure
        differs from the raw series may override this.

        Intermittent methods are the relevant case: Croston forecasts a
        per-period rate, and the standard deviation of the raw series is
        inflated by the zero periods that method already accounts for
        explicitly. An override using the dispersion of non-zero demand
        observations is more appropriate there.

        Returns
        -------
        pd.Series aligned to ``pred_df.index``. Must be per-row; a scalar
        would apply one dispersion to every item.
        """
        if self._item_std is None:
            raise RuntimeError("_spread_for called with quantile output disabled")
        spread = pred_df["item_id"].map(self._item_std)
        return spread.fillna(0.0).astype(float)

    def _attach_quantiles(self, pred_df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Append Gaussian quantile columns derived from the point forecast.

        Column names use ``q`` followed by the level as an integer percentage,
        so 0.8333 becomes ``q83``. Levels that do not map to distinct integers
        collide; the requested levels are validated and de-duplicated in
        __init__, so callers should avoid sub-integer precision.
        """
        if self.quantiles is None:
            return pred_df

        spread = self._spread_for(pred_df)
        if not isinstance(spread, pd.Series):
            raise TypeError(
                f"{method}: _spread_for must return a per-row pd.Series, got "
                f"{type(spread)}. A scalar would apply one dispersion to "
                f"every item."
            )
        if not pred_df.index.equals(spread.index):
            raise ValueError(
                f"{method}: _spread_for returned an index that does not match "
                f"the prediction frame."
            )

        for level in self.quantiles:
            column = f"q{int(level * 100)}"
            pred_df[column] = (
                pred_df["sales_pred"] + norm.ppf(level) * spread
            ).clip(lower=0.0)
        return pred_df

    def _validate_quantiles(self, pred_df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Check quantile columns for ordering, sign, and point agreement."""
        if self.quantiles is None:
            return pred_df

        columns = [f"q{int(level * 100)}" for level in self.quantiles]
        present = [c for c in columns if c in pred_df.columns]
        if len(present) < 2:
            return pred_df

        ordered = pred_df[present].to_numpy()
        if (ordered[:, 1:] < ordered[:, :-1]).any():
            raise AssertionError(
                f"{method}: quantile columns are not monotonically increasing "
                f"in level. Intervals must satisfy q_lo <= q_hi on every row."
            )
        if (pred_df[present] < 0).any().any():
            raise AssertionError(
                f"{method}: negative quantile values. Intervals must be clipped "
                f"at zero."
            )

        if "q50" in pred_df.columns and not np.allclose(
            pred_df["q50"], pred_df["sales_pred"], atol=1e-8
        ):
            raise AssertionError(
                f"{method}: q50 does not equal sales_pred. The Gaussian interval "
                f"is centred on the point forecast, so the median level must "
                f"reproduce it exactly."
            )
        return pred_df



# ======================================================================
# Backward-compatible wrappers.
# src/backtest_engine.py imports these at module scope. They are intended for
# removal once the engine's if/elif dispatch is replaced by Baseline.run.
# ======================================================================
def seasonal_naive(
    train_df: pd.DataFrame, test_df: pd.DataFrame, lag_days: int = 28
) -> pd.DataFrame:
    return Baseline(train_df, test_df).seasonal_naive(lag_days=lag_days)


def simple_moving_average(
    train_df: pd.DataFrame, test_df: pd.DataFrame, window_days: int = 180
) -> pd.DataFrame:
    return Baseline(train_df, test_df).simple_moving_average(window_days=window_days)
