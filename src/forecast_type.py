

'''Two type of forecast: Recursive vs Direct'''
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

    _NON_FEATURE_COLS= ('item_id', 'dept_id', 'cat_id', 'date', 'sales')
    def __init__(self, model, feature_builder=FeatureBuilder()):
        '''model: Already fitted SelectModel instance
        feature_builder: feature builder class'''
        self.model = model
        self.feature_builder = feature_builder

    def recursive_forecaster(self, 
                            train_df: pd.DataFrame, 
                            test_df: pd.DataFrame,
                            max_lookback_days: int = 100)->pd.DataFrame:
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
    
    def _build_train_frame(
        self,
        known_df: pd.DataFrame,
        horizons: int = 28,
        exclude_cols: tuple = _NON_FEATURE_COLS,
        assume_sorted: bool = False,
        validate: bool = True,
    ) -> tuple[pd.DataFrame, list]:
        """Build a stacked direct-horizon training frame.

        known_df: one row per (item_id, date=origin), with static +
            dynamic (lag/rolling, already anchored to origin date) feature
            columns and a 'sales' column giving actual demand at origin.
        horizons: number of forecast horizons (1..horizons) to stack.
        exclude_cols: columns in known_df that are not model features.
        assume_sorted: skip the sort-by (item_id, date) step if the caller
            already guarantees this ordering (saves time on large frames).
        validate: run a cheap monotonic-date-per-item sanity check.
            Adds an extra pass over the data; off by default.

        Assumes a dense daily calendar per item_id (no missing dates),
        since horizon h is implemented as a row-position shift of h
        within each item's sorted history, not a calendar-day shift.
        If your panel has gaps, reindex to a full daily calendar upstream.

        Returns (train_df, features_direct):
            train_df has id/date columns + feature columns + 'h' + 'target'
            features_direct is the list of column names to feed the model
            (feature columns plus 'h'; excludes id/date/target).
        """
        t0 = time.time()

        df = known_df
        if not assume_sorted:
            df = df.sort_values(['item_id', 'date'], kind='mergesort')
        df = df.reset_index(drop=True)

        if validate:
            gaps = df.groupby('item_id', observed=True)['date'].diff().dropna()
            assert (gaps > pd.Timedelta(0)).all(), (
                'date is not strictly increasing within at least one '
                'item_id; horizon shifting by row-position will be wrong.'
            )
         

        feature_cols = [c for c in df.columns if c not in exclude_cols]

        sales = df['sales'].to_numpy(dtype='float32')
        n_rows = len(df)

        gb = df.groupby('item_id', observed=True, sort=False)
        group_sizes = gb['item_id'].transform('size').to_numpy()
        pos_in_group = gb.cumcount().to_numpy()
        steps_remaining = group_sizes - pos_in_group - 1

        row_idx_parts = []
        h_parts = []
        target_parts = []

        for h in range(1, horizons + 1):
            valid = steps_remaining >= h
            valid_idx = np.nonzero(valid)[0]
            if valid_idx.size == 0:
                continue
            row_idx_parts.append(valid_idx)
            h_parts.append(np.full(valid_idx.size, h, dtype='int16'))
            target_parts.append(sales[valid_idx + h])

        row_idx_full = np.concatenate(row_idx_parts)
        h_full = np.concatenate(h_parts)
        target_full = np.concatenate(target_parts)

        train_df = df.iloc[row_idx_full].reset_index(drop=True)
        train_df['h'] = h_full
        train_df['target'] = target_full

        features_direct = feature_cols + ['h']

        logger.info(
            'Direct train frame built: %d origin rows -> %d stacked rows '
            '(horizons=%d), %d feature cols, in %.2fs',
            n_rows, len(train_df), horizons, len(features_direct),
            time.time() - t0,
        )

        return train_df, features_direct
    
    def _make_direct_training_dataset(
        self,
        known_df: pd.DataFrame,
        origin_feature_cols: list[str],
        target_known_cols: list[str],
        max_horizon: int = 28,
        chunk_size: int = 7,  # Process 7 horizons at a time
    ) -> tuple[pd.DataFrame, list[str]]:

        base = known_df.sort_values(["item_id", "date"]).copy()

        clean_feature_cols = [ c for c in origin_feature_cols if c not in ["item_id", "date"] ]
                                        # 1. Build features ONCE at origin level
        origin = self.feature_builder.build(
            df=base,
            lags=[7, 28, 60, 90],
            mean_windows=[7, 28, 60, 90],
            max_windows=[7, 28, 60, 90],
            rolling_on_lags=[28],
        )

        origin = origin.loc[:, ~origin.columns.duplicated()]
        origin = origin[["item_id", "date", *clean_feature_cols]].rename(
            columns={"date": "origin_date"}
        )

        # Prepare lookup targets
        targets = base[["item_id", "date", "sales"]].rename(
            columns={"date": "target_date", "sales": "target_sales"}
        )
        target_covariates = base[["item_id", "date", *target_known_cols]].rename(
            columns={
                "date": "target_date",
                **{col: f"target_{col}" for col in target_known_cols},
            }
        )

        # 2. Process horizons in chunks to keep RAM low
        chunks = []
        for h_start in range(1, max_horizon + 1, chunk_size):
            h_end = min(h_start + chunk_size, max_horizon + 1)
            horizons_chunk = np.arange(h_start, h_end)

            # Expand ONLY for this chunk of horizons
            expanded_chunk = origin.loc[
                origin.index.repeat(len(horizons_chunk))
            ].copy()
            expanded_chunk["horizon"] = np.tile(horizons_chunk, len(origin))
            expanded_chunk["target_date"] = expanded_chunk[
                "origin_date"
            ] + pd.to_timedelta(expanded_chunk["horizon"], unit="D")

            # Merge for this chunk
            merged = expanded_chunk.merge(
                targets, on=["item_id", "target_date"], how="inner"
            )
            merged = merged.merge(
                target_covariates,
                on=["item_id", "target_date"],
                how="left",
                validate="many_to_one",
            )

            chunks.append(merged)
            logger.info(f"{100*h_start/(max_horizon/chunk_size)}% chunks processed.")

        # 3. Combine chunked results
        direct_train = pd.concat(chunks, ignore_index=True)

        direct_feature_cols = [
            *origin_feature_cols,
            "horizon",
            *[f"target_{col}" for col in target_known_cols],
        ]


        return direct_train, direct_feature_cols

    
    def _build_test_frame(
        self,
        test_origin_df: pd.DataFrame,
        horizons: int = 28,
        exclude_cols: tuple = _NON_FEATURE_COLS,validate=True,
    ) -> tuple[pd.DataFrame, list]:
        """Build a stacked direct-horizon inference frame.

        test_origin_df: one row per (item_id, origin_date), with static
            feats + dynamic feats anchored to the origin date (same
            structure as known_df in _build_train_frame, but no future
            target is available). No steps_remaining filtering is applied
            since every horizon is scored regardless of how much future
            history exists in known_df.
        horizons: number of forecast horizons (1..horizons) to expand.

        Returns (test_df, features_direct), same shape convention as
        _build_train_frame but without a 'target' column.
        """
        t0 = time.time()


        feature_cols = [
            c for c in test_origin_df.columns if c not in exclude_cols
        ]
        n_rows = len(test_origin_df)

        row_idx = np.repeat(np.arange(n_rows), horizons)
        h_full = np.tile(
            np.arange(1, horizons + 1, dtype='int16'), n_rows
        )

        test_df = test_origin_df.iloc[row_idx].reset_index(drop=True)

        if validate:
            gaps = test_origin_df.groupby('item_id', observed=True)['date'].diff().dropna()
            assert (gaps > pd.Timedelta(0)).all(), (
                'date is not strictly increasing within at least one '
                'item_id; horizon shifting by row-position will be wrong.'
            )

        test_df['h'] = h_full

        features_direct = feature_cols + ['h']

        logger.info(
            'Direct test frame built: %d origin rows -> %d stacked rows '
            '(horizons=%d) in %.2fs',
            n_rows, len(test_df), horizons, time.time() - t0,
        )

        return test_df, features_direct
    
import numpy as np
import pandas as pd

def optimize_dtypes(df: pd.DataFrame,int8_cols) -> pd.DataFrame:
    # 1. Downcast binary / small integer flags to int8 / uint8
    df = df.copy()

    for col in int8_cols:
        if col in df.columns:
            df[col] = df[col].astype('int8')
            
    # 2. Downcast year and days_to_next_event to int16 (up to 32,767)
    int16_cols = ['year', 'wm_yr_wk', 'days_to_next_event']
    for col in int16_cols:
        if col in df.columns:
            df[col] = df[col].astype('int16')

    obj_cols = ['id', 'd', 'weekday', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for col in obj_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    return df

def _check_date_gaps(df, id_col='item_id', date_col='date', freq='D'):
    """Return item_ids whose date sequence has a gap at `freq`.

    Cheap O(n) diagnostic, does not raise. Run once against any frame
    you rely on row-position shifts for (train_df, history_slice,
    test_df) before trusting groupby().shift()/rolling() on it.
    """
    diffs = (
        df.sort_values([id_col, date_col])
        .groupby(id_col, observed=True)[date_col]
        .diff()
    )
    bad_mask = diffs.notna() & (diffs != pd.Timedelta(1, unit=freq))
    return df.loc[bad_mask, id_col].unique()

if __name__ == '__main__':

    train_path = 'data/processed/sales_known_ca_1.parquet'
    
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet('data/processed/sales_future_ca_1.parquet')
    test_df_horizon = test_df[test_df['date']<=(test_df['date'].min()+pd.Timedelta(days=28))]


    print(_check_date_gaps(train_df))
    print(_check_date_gaps(test_df))
    print(_check_date_gaps(test_df_horizon))

    from .utils import get_items_with_min_history, get_items_top

    cutoff = train_df['date'].max()-pd.Timedelta(days=1095)

    train_last_3 = train_df[train_df['date']>=cutoff] 

    mem_usage = lambda df : df.memory_usage().sum()/1024**2

    print(f"train_last_3 memory: {mem_usage(train_last_3)}")

    # filter to items only surviving 3 years

    train_ = get_items_top( get_items_with_min_history(train_last_3,1095),0.8)

    print(f'items: {train_['item_id'].nunique()}, mem_usage: {mem_usage(train_)}')

    #### build train frame 

    forecaster = Forecaster(model='model')


    train_direct, feature_direct = forecaster._build_train_frame(train_)

    print(f'direct train memory: {mem_usage(train_direct)}')





    test_path = 'data/processed/sales_future_ca_1.parquet'