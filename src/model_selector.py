"""
Training module for gradient boosting regressors (HistGradientBoostingRegressor & LightGBM).
Handles target transformation (log1p), categorical features, and quantile forecasting.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DEFAULT_QUANTILES = [0.10, 0.90]


def _make_model(
    kind: str,
    quantile: float | None = None,
    categorical_cols: list[str] | None = None,
) -> Any:
    has_cats = bool(categorical_cols)

    if kind == "hgb":
        loss = 'quantile' if quantile is not None else 'squared_error'

        return HistGradientBoostingRegressor(
            loss=loss,
            quantile=quantile,
            max_iter=400,
            learning_rate=0.05,
            max_depth=None,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            categorical_features="from_dtype" if has_cats else None,
            random_state=42,
        )

    if kind == "lgbm":
        from lightgbm import LGBMRegressor  # optional dependency
        objective = 'quantile' if quantile is not None else 'tweedie'

        return LGBMRegressor(
            objective=objective,
            quantile=quantile,
            tweedie_variance_power=1.2,
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )

    raise ValueError(f'Unknown model: {kind}')


def available_models() -> list[str]:
    models = ["hgb"]
    try:
        import lightgbm  # noqa: F401
        models.append("lgbm")
    except ImportError:
        pass
    return models


def _check_features(X: pd.DataFrame, categorical_cols: list[str] | None) -> None:
    """
    Validates that declared categorical columns are pandas 'category' dtype
    and all other features are numeric.
    """
    categorical_cols = categorical_cols or []
    for col in categorical_cols:
        if col not in X.columns:
            raise ValueError(f"declared categorical_cols has '{col}', not present in X")
        if not isinstance(X[col].dtype, pd.CategoricalDtype):
            raise TypeError(
                f"'{col}' declared as categorical but dtype is {X[col].dtype}, "
                f"not pandas 'category'. Cast it explicitly: X['{col}'].astype('category')."
            )

    non_cat_cols = [c for c in X.columns if c not in categorical_cols]
    bad_cols = [c for c in non_cat_cols if not pd.api.types.is_numeric_dtype(X[c])]
    if bad_cols:
        raise TypeError(
            f"SelectModel received non-numeric, non-declared columns: {bad_cols}. "
            f"Either encode them (.cat.codes) or add them to categorical_cols."
        )


class SelectModel:
    """Thin wrapper: optional log1p target transform + chosen GBM engine +
    optional native categorical handling.
    """

    def __init__(
        self,
        model: str = "lgbm",
        quantiles: list[float] | None = None,
        use_log_transform: bool = True,
        categorical_cols: list[str] | None = None,
    ):
        self.kind = model
        self.use_log_transform = use_log_transform
        self.categorical_cols = categorical_cols

        self.features: list[str] | None = None

        self.quantiles = quantiles is not None
        self.quantile_levels = (
            quantiles if quantiles is not None else DEFAULT_QUANTILES.copy()
        )

        # Point forecast model
        self.model_point = _make_model(
            kind=self.kind, quantile=None, categorical_cols=self.categorical_cols
        )

        # Quantile forecast models
        self.quantile_models: Dict[str, Any] = {}

        if self.quantiles:
            if any(q <= 0 or q >= 1 for q in self.quantile_levels):
                raise ValueError("All quantile levels must be strictly between 0 and 1.")

            self.quantile_models = {
                f'q{int(q*100)}': _make_model(
                    kind=self.kind,
                    quantile=q,
                    categorical_cols=self.categorical_cols,
                )
                for q in self.quantile_levels
            }

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        _check_features(X, self.categorical_cols)
        if self.categorical_cols:
            X = X.copy()
            for col in self.categorical_cols:
                X[col] = X[col].astype("category")
        return X

    def fit(self, X: pd.DataFrame, y: Any) -> SelectModel:
        X_prep = self._prep(X)
        self.features = list(X_prep.columns)
        target = (
            np.log1p(np.asarray(y, dtype=float))
            if self.use_log_transform
            else np.asarray(y, dtype=float)
        )

        # Fit point model
        if self.kind == "lgbm" and self.categorical_cols:
            getattr(self.model_point, "fit")(X_prep, target, categorical_feature=self.categorical_cols)
        else:
            self.model_point.fit(X_prep, target)

        # Fit quantile models
        if self.quantiles:
            for qval, quantmodel in self.quantile_models.items():
                if self.kind == "lgbm" and self.categorical_cols:
                    getattr(quantmodel, "fit")(X_prep, target, categorical_feature=self.categorical_cols)
                else:
                    quantmodel.fit(X_prep, target)

        return self

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        if self.features is not None:
            X = X[self.features]

        X_prep = self._prep(X)
        preds: dict[str, np.ndarray] = {}

        # Point forecast
        raw_point = np.asarray(self.model_point.predict(X_prep), dtype=float)
        pred_point = np.expm1(raw_point) if self.use_log_transform else raw_point
        preds['point'] = np.clip(pred_point, 0.0, None)

        # Quantile forecasts
        if self.quantiles:
            for qval, quantmodel in self.quantile_models.items():
                raw_q = np.asarray(quantmodel.predict(X_prep), dtype=float)
                pred_quantile = np.expm1(raw_q) if self.use_log_transform else raw_q
                preds[qval] = np.clip(pred_quantile, 0.0, None)

        return preds