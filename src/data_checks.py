'''basic data checks that need to be passed before running the pipeline'''

from __future__ import annotations

import pandas as pd

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


def check_data(df:pd.DataFrame) -> pd.DataFrame:
    failures = list[str] = []

