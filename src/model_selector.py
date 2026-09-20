"""
Training. HistGradientBoostingRegressor is the default engine: fast, handles NaN
natively, no external deps. XGBoost / LightGBM are used automatically IF installed
-- same interface, so `pip install  lightgbm` lights them up with no code
change.

Two independent options, both explicit, neither hardcoded:
  - use_log_transform: model log1p(units) and invert on predict. Stabilises variance
    for right-skewed, non-negative demand data -- but whether it actually helps is an
    empirical question per dataset, same as the 730-day window was. Test with/without.
  - categorical_cols: if given, those columns are passed through as native pandas
    'category' dtype and each engine's own categorical-split logic is used (no
    encoding). If None, every column must already be plain numeric (e.g. your own
    .cat.codes convention) -- this is checked and raised on, not assumed.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DEFAULT_QUANTILES = [0.10,0.90]

def _make_model(kind: str, quantile: float | None = None, categorical_cols: list[str] | None = None):
    has_cats = bool(categorical_cols)

    if kind == "hgb":
        # sklearn >=1.4 required for categorical_features="from_dtype"
        if quantile is not None:
            loss = 'quantile'
        else:
            loss = 'squared_error'

        return HistGradientBoostingRegressor(loss=loss, quantile=quantile,
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
        from lightgbm import LGBMRegressor  # optional
        if quantile is not None:
            objective = 'quantile'
            
        else:
            objective = 'tweedie'

        return LGBMRegressor(objective=objective,quantile=quantile,
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


def _check_features(X: pd.DataFrame, categorical_cols: list[str] | None):
    """
    If categorical_cols is given: those columns must be pandas 'category' dtype
    (native mode) -- everything else must be numeric.
    If categorical_cols is None: every column must be numeric (your .cat.codes
    convention) -- silently receiving a leftover category-dtype column here would
    break native handling that was never requested.
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
    optional native categorical handling. No notion of recursive vs. direct
    forecasting -- that's orchestration (pipeline.py), not a model concern.
    """

    def __init__(self, model: str = "lgbm", quantiles:list[float] | None=None,use_log_transform: bool = True,
                 categorical_cols: list[str] | None = None):
        
        self.kind = model
        self.use_log_transform = use_log_transform
        self.categorical_cols = categorical_cols
        
        self.features: list[str] | None = None

        self.quantiles = quantiles is not None
        self.quantile_levels = (quantiles if quantiles is not None else DEFAULT_QUANTILES.copy() )

        # point forecast model            
        self.model_point = _make_model(kind=self.kind,quantile=None, categorical_cols=self.categorical_cols)

        ### get the quantile models--> optional
        self.quantile_models = {}

        if self.quantiles:
            if any(q <= 0 or q >= 1 for q in self.quantile_levels):
                raise ValueError("All quantile levels must be strictly between 0 and 1.")

            self.quantile_models = { f'q{int(q*100)}': _make_model( kind = self.kind, 
                                                quantile= q,
                                                categorical_cols=self.categorical_cols)
                                                for q in self.quantile_levels 
                                                }
        


    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        _check_features(X, self.categorical_cols)
        if self.categorical_cols:
            X = X.copy()
            for col in self.categorical_cols:
                X[col] = X[col].astype("category")
        return X

    def fit(self, X, y):
        X = self._prep(X)
        self.features = list(X.columns)
        target = np.log1p(np.asarray(y, dtype=float)) if self.use_log_transform else np.asarray(y, dtype=float)

        if self.kind == "lgbm" and self.categorical_cols:
            # point forecast
            self.model_point.fit(X, target, categorical_feature=self.categorical_cols)

        else:
            self.model_point.fit(X, target)

        # quantile forecasts
            
        if self.quantiles:
            for qval, quantmodel in self.quantile_models.items():
                if self.kind=='lgbm' and self.categorical_cols:
                    self.quantile_models[qval] = quantmodel.fit(X,target,
                                                        categorical_feature=self.categorical_cols)
                else:
                    self.quantile_models[qval] = quantmodel.fit(X,target)

        return self

    def predict(self, X)->dict[str,]:

        if self.features is not None:
            X = X[self.features]

        X = self._prep(X)

        preds = {}

        # point forecast
        raw_pred = self.model_point.predict(X)
        pred_point = np.expm1(raw_pred) if self.use_log_transform else raw_pred
        preds['point']  = np.clip(pred_point,0.0,None)
                
        if self.quantiles:
            for qval,quantmodel in self.quantile_models.items():
                raw_pred = quantmodel.predict(X)
                pred_quantile = np.expm1(raw_pred) if self.use_log_transform else raw_pred
                preds[qval] = np.clip(pred_quantile,0.0,None)

       
        return preds# demand can't be negative