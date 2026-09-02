

'''functions for recursive forecasting: all items'''
import logging
import re
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from .features import FeatureBuilder
from .models_train import SelectModel

logger = logging.getLogger(__name__)


def _infer_dynamic_feats(full_feats) -> dict:
    """Derive which lags/rolling windows must be freshly recomputed for the newly
    appended forecast row on every recursive step, straight from the columns
    actually present in the training feature set -- so this can never silently
    drift out of sync with whatever feat_builder.build() was called with.

    (Replaces a hardcoded default of lags=[7] that used to live inside
    next_day_feature_build: it left lag_28/60/90 -- and rolling_lag_28_win_*,
    which depends on lag_28 -- as NaN for the model on every recursive step,
    silently. See the __main__ self-test below.)
    """
    lag_pat = re.compile(r'^lag_(\d+)$')
    mean_pat = re.compile(r'^rolling_mean_(\d+)$')
    max_pat = re.compile(r'^rolling_max_(\d+)$')
    on_lag_pat = re.compile(r'^rolling_lag_(\d+)_win_\d+$')

    lags = sorted({int(m.group(1)) for c in full_feats if (m := lag_pat.match(c))})
    rolling_mean = sorted({int(m.group(1)) for c in full_feats if (m := mean_pat.match(c))})
    rolling_max = sorted({int(m.group(1)) for c in full_feats if (m := max_pat.match(c))})
    rolling_on_lag = sorted({int(m.group(1)) for c in full_feats if (m := on_lag_pat.match(c))})

    missing_lag_deps = set(rolling_on_lag) - set(lags)
    if missing_lag_deps:
        raise ValueError(
            f"rolling_on_lag needs lag_{sorted(missing_lag_deps)} recomputed on every "
            f"recursive step, but those lags aren't in the training feature set's lag_* "
            f"columns ({lags}). Add them to the `lags` you pass to feat_builder.build()."
        )

    return {
        'lags': lags or None,
        'rolling_mean': rolling_mean or None,
        'rolling_max': rolling_max or None,
        'rolling_on_lag': rolling_on_lag or None,
    }


class Forecaster():
    def __init__(self, model, feature_builder=FeatureBuilder):
        '''model: Already fitted SelectModel instance
        feature_builder: feature builder class'''
        self.model = model
        self.feature_builder = feature_builder

    def recursive_forecaster(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                              max_lookback_days: int = 100):
        '''train_df: entire training set
            test_df: test set with static features (should only contain the static features)
            '''
        full_feats = train_df.columns
        dynamic_feats = _infer_dynamic_feats(full_feats)

        cutoff_date = train_df['date'].max() - pd.Timedelta(days=max_lookback_days)
        history_slice = train_df[train_df['date'] >= cutoff_date].copy()

        dates = sorted(test_df['date'].unique())
        total_dates = len(dates)
        n_items = train_df['item_id'].nunique()

        logger.info(
            "Recursive forecast starting: %d day(s) x %d item(s), lookback=%d days, "
            "dynamic_feats=%s", total_dates, n_items, max_lookback_days, dynamic_feats
        )

        all_results = []
        for step, current_date in enumerate(dates, start=1):
            t0 = time.time()

            next_day = test_df[test_df['date'] == current_date].copy()  # all static feats
            next_day_dynamic_feats = self.feature_builder.next_day_feature_build(
                history_slice, next_day, dynamic_feats=dynamic_feats
            )

            next_day_full_feats = next_day_dynamic_feats[full_feats]

            assert set(next_day_full_feats['item_id'].unique().tolist()) == set(
                train_df['item_id'].unique().tolist()
            ), 'AssertionError: items in test_df do not match train_df'

            preds = self.model.predict(next_day_full_feats)  # point, q10, q90, ...

            day_result = pd.DataFrame({
                'item_id': next_day_full_feats['item_id'].values,
                'dept_id': next_day_full_feats['dept_id'].values,
                'cat_id' : next_day_full_feats['cat_id'].values,
                'date': current_date,
                'sales_pred': preds['point'],
            })
            for key, values in preds.items():
                if key == "point":
                    continue
                day_result[key] = values

            all_results.append(day_result)

            # append history with predicted sales, so tomorrow's lag/rolling
            # features see today's forecast rather than NaN
            history_slice = pd.concat(
                (history_slice, day_result[['item_id', 'date', 'sales_pred']]
                 .rename(columns={'sales_pred': 'sales'}))
            )

            pct = step / total_dates * 100
            logger.info(
                "[%d/%d] (%5.1f%%) date=%s done in %.2fs",
                step, total_dates, pct, pd.Timestamp(current_date).date(), time.time() - t0
            )

        preds_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
        logger.info("Recursive forecast complete: %d rows returned.", len(preds_df))
     
        return preds_df


def cost_per_item(df_test, preds, safety_level='q90'):
    '''pred: must contain point forecast and quantile forecasts'''
    item_data = df_test[['item_id', 'date', 'sell_price', 'sales']].copy()
    item_data = item_data.merge(preds, on=['item_id', 'date'], how='left')

    item_data['cost'] = 0.60 * item_data['sell_price']
    item_data['profit'] = 0.40 * item_data['sell_price']
    item_data['holding_cost'] = (0.25 / 365) * item_data['cost']
    item_data['stockout_cost'] = item_data['profit']

    S = item_data[safety_level]
    D = item_data['sales']
    item_data['cost_holding'] = item_data['holding_cost'] * np.maximum(S - D, 0)
    item_data['stockout_cost'] = item_data['stockout_cost'] * np.maximum(D - S, 0)
    item_data['total_cost'] = item_data['cost_holding'] + item_data['stockout_cost']

    return item_data.groupby('item_id')['total_cost'].sum().reset_index()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    print("Running self-test for Forecaster.recursive_forecaster ...\n")

    # synthetic data run through the *real* FeatureBuilder pipeline, so the test
    # exercises exactly the feature set production trains on.
    rng = pd.date_range('2020-01-01', periods=100, freq='D')
    items = ['ITEM_A', 'ITEM_B']
    rows = []
    for item in items:
        base = 0.0 if item == 'ITEM_A' else 100.0
        for i, d in enumerate(rng):
            rows.append({'item_id': item, 'date': d, 'sales': float(i % 10) + base,
                         'store_id': 'CA_1', 'dept_id': 'FOODS_1', 'sell_price': 3.0,
                         'wm_yr_wk': int(d.strftime('%y%V'))})
    raw = pd.DataFrame(rows)

    fb = FeatureBuilder()
    train_df = fb.build(raw, lags=[7, 28], mean_windows=[7, 28],
                         max_windows=[7, 28], rolling_on_lags=[28])

    future_dates = pd.date_range(rng[-1] + pd.Timedelta(days=1), periods=10, freq='D')
    test_df = pd.DataFrame([{'item_id': it, 'date': d} for it in items for d in future_dates])

    # ---- test 1: structural correctness ------------------------------------
    class ConstantModel:
        def predict(self, X):
            return {'point': np.full(len(X), 42.0)}

    forecaster = Forecaster(model=ConstantModel(), feature_builder=fb)
    preds = forecaster.recursive_forecaster(train_df, test_df, max_lookback_days=100)

    expected_rows = len(items) * len(future_dates)
    assert len(preds) == expected_rows, f"expected {expected_rows} rows, got {len(preds)}"
    assert not preds.duplicated(subset=['item_id', 'date']).any(), "duplicate (item_id, date) rows"
    assert set(preds['date']) == set(future_dates), "forecasted dates don't match requested test dates"
    assert set(preds['item_id']) == set(items), "item coverage mismatch"
    assert (preds['sales_pred'] == 42.0).all(), "prediction path broken -- model output not passed through"
    print("[PASS] structural correctness: shape, uniqueness, date/item coverage")

    # ---- test 2: every lag_* feature stays fresh (non-NaN) across the whole
    # recursive horizon -- this is the one that actually catches real bugs.
    class LagFreshnessProbe:
        def __init__(self):
            self.nan_lag_cols_seen = set()

        def predict(self, X):
            for c in [c for c in X.columns if c.startswith('lag_')]:
                if X[c].isna().any():
                    self.nan_lag_cols_seen.add(c)
            return {'point': np.full(len(X), 1.0)}

    probe = LagFreshnessProbe()
    forecaster2 = Forecaster(model=probe, feature_builder=fb)
    _ = forecaster2.recursive_forecaster(train_df, test_df, max_lookback_days=100)

    assert not probe.nan_lag_cols_seen, (
        f"lag feature(s) {probe.nan_lag_cols_seen} were NaN during recursive prediction -- "
        f"next_day_feature_build isn't recomputing every lag the model was trained on."
    )
    print("[PASS] all lag_* features stay fresh (non-NaN) throughout the recursive horizon")

    # ---- test 3: recursion actually steps through dates, one call per day --
    class DayIndexModel:
        def __init__(self):
            self.calls = 0
        def predict(self, X):
            self.calls += 1
            return {'point': np.full(len(X), self.calls * 10.0)}

    day_model = DayIndexModel()
    forecaster3 = Forecaster(model=day_model, feature_builder=fb)
    preds3 = forecaster3.recursive_forecaster(train_df, test_df, max_lookback_days=100)
    assert day_model.calls == len(future_dates), (
        f"expected {len(future_dates)} model calls, got {day_model.calls} -- "
        f"recursion loop isn't stepping through dates correctly"
    )
    print(f"[PASS] model invoked once per day across all {len(future_dates)} recursive steps")

    print("\nAll recursive_forecaster self-tests passed.")