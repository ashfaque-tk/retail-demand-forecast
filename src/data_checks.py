'''basic data checks that need to be passed before running the pipeline'''

from __future__ import annotations

import pandas as pd
import numpy as np

#mandatory columns
SCHEMA = {
    "date": "datetime64[ns]",
    "store_id": "object",
    "item_id": "object",
    "dept_id" :"object",
    "cat_id" : "object",
    "sell_price": "float_32",
    'sales'  : 'float_32'  #if float_16, averaging would give error
}


def validate_raw(df: pd.DataFrame) -> None:
    """Raise ValueError if any basic sanity check fails."""
    required = {'item_id', 'date', 'sales', 'sell_price'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # dates must be parseable
    if not np.issubdtype(df['date'].dtype, np.datetime64):
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception as exc:
            raise ValueError("Column 'date' cannot be parsed as datetime") from exc

    # sales should be non‑negative
    if (df['sales'] < 0).any():
        raise ValueError("Negative sales values detected")

    # duplicate (item_id, date) pairs are not allowed
    if df.duplicated(subset=['item_id', 'date']).any():
        raise ValueError("Duplicate (item_id, date) rows found")