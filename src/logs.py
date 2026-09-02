
from __future__ import annotations 
import json 
from pathlib import Path
import logging 

import numpy as np 
import pandas as pd 
from datetime import datetime


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

def _json_safe(obj):
    """Recursively convert an object into something json.dump can handle,
    without lying about the data: DataFrames become row-records, timestamps
    become ISO strings, numpy scalars become plain python scalars. Anything
    genuinely unrecognized is left alone and caught by json.dump's own
    TypeError (via default=str as a last-resort net), not silently dropped.
    """
    if isinstance(obj, pd.DataFrame):
        return json.loads(obj.to_json(orient='records', date_format='iso'))
    if isinstance(obj, pd.Series):
        return _json_safe(obj.to_dict())
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    return obj


def log_experiment_results(results_path: Path, record: dict) -> Path:
    """Append one experiment record to a JSON list at results_path, creating
    the file/dir on first use. Never overwrites -- refuses to append onto a
    file that isn't already a JSON list, rather than silently clobbering it.
    """
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)
        if not isinstance(existing, list):
            raise ValueError(
                f"{results_path} exists but isn't a JSON list -- refusing to "
                f"append blindly, since that would silently corrupt it."
            )
    else:
        existing = []

    existing.append(_json_safe(record))

    with open(results_path, 'w') as f:
        json.dump(existing, f, indent=2, default=str)

    logger.info("Logged experiment record to %s (%d total entries)", results_path, len(existing))
    return results_path
