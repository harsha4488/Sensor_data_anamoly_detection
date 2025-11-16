"""
insights_nlp.py - Lightweight anomaly & log-correlation utilities (no LSTM)

Exports:
- preprocess_sensor_data
- numeric_summary
- run_all_detectors
- correlate_anomalies_with_logs
- map_window_mask_to_index_mask
- TF_AVAILABLE
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

# TensorFlow flag kept for compatibility (no LSTM in this version)
TF_AVAILABLE = False

# -------------------------
# Preprocessing
# -------------------------
def preprocess_sensor_data(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    value_cols: Optional[List[str]] = None,
    resample_rule: Optional[str] = None,
    fill_method: str = "ffill"
) -> pd.DataFrame:
    if df is None:
        raise ValueError("Input dataframe is None")
    df = df.copy()

    # Convert timestamp to index if provided
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.dropna(subset=[timestamp_col]).set_index(timestamp_col).sort_index()
    elif not isinstance(df.index, pd.DatetimeIndex):
        # attempt to coerce index to datetime
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            raise ValueError("No datetime index and timestamp_col missing or invalid")

    # choose numeric columns
    if value_cols:
        missing = [c for c in value_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing value columns: {missing}")
        df_vals = df[value_cols].astype(float)
    else:
        df_vals = df.select_dtypes(include=[np.number])
        if df_vals.shape[1] == 0:
            raise ValueError("No numeric columns found for anomaly detection.")

    if resample_rule:
        df_vals = df_vals.resample(resample_rule).mean()

    if fill_method == "ffill":
        df_vals = df_vals.ffill().bfill()
    elif fill_method == "interpolate":
        df_vals = df_vals.interpolate().bfill()
    else:
        df_vals = df_vals.fillna(0)

    return df_vals

# -------------------------
# Numeric summary
# -------------------------
def numeric_summary(series: pd.Series) -> Dict[str, Any]:
    s = series.dropna()
    return {
        "count": int(s.shape[0]),
        "mean": float(s.mean()) if s.shape[0] else None,
        "std": float(s.std()) if s.shape[0] else None,
        "min": float(s.min()) if s.shape[0] else None,
        "50%": float(s.median()) if s.shape[0] else None,
        "max": float(s.max()) if s.shape[0] else None,
    }

# -------------------------
# Simple detectors (no heavy imports at top)
# -------------------------
def _zscore_mask(arr: np.ndarray, z_thresh: float = 3.0) -> np.ndarray:
    if arr.ndim != 1:
        arr = arr.ravel()
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if std == 0 or np.isnan(std):
        return np.zeros_like(arr, dtype=bool)
    z = np.abs((arr - mean) / std)
    return z > z_thresh

def _iqr_mask(arr: np.ndarray, k: float = 1.5) -> np.ndarray:
    if arr.ndim != 1:
        arr = arr.ravel()
    q1 = np.nanpercentile(arr, 25)
    q3 = np.nanpercentile(arr, 75)
    iqr = q3 - q1
    if iqr == 0:
        return np.zeros_like(arr, dtype=bool)
    return (arr < (q1 - k * iqr)) | (arr > (q3 + k * iqr))

# -------------------------
# run_all_detectors: returns dictionary expected by app.py
# -------------------------
def run_all_detectors(
    df_values: pd.DataFrame,
    column: Optional[str] = None,
    rescale: bool = True,
) -> Dict[str, Any]:
    """
    Run simple detectors and return masks.
    This implementation purposely avoids heavy imports; placeholders are provided
    for more advanced detectors.
    """
    if df_values is None or df_values.shape[0] == 0:
        raise ValueError("df_values must be non-empty DataFrame")

    if column:
        if column not in df_values.columns:
            raise ValueError(f"Column {column} not found in df_values")
        arr = df_values[[column]].values.flatten()
    else:
        arr = df_values.iloc[:, 0].values.flatten()

    n = arr.shape[0]

    results: Dict[str, Any] = {}
    # basic statistical detectors
    results["zscore"] = _zscore_mask(arr)
    results["iqr"] = _iqr_mask(arr)

    # placeholders for classical ML detectors (length = n)
    zero_mask = np.zeros(n, dtype=bool)
    results["isolation_forest"] = zero_mask.copy()
    results["local_outlier_factor"] = zero_mask.copy()
    results["one_class_svm"] = zero_mask.copy()
    results["elliptic_envelope"] = zero_mask.copy()
    results["dbscan"] = zero_mask.copy()
    results["pca_recon"] = zero_mask.copy()

    # No LSTM here — return skipped metadata
    results["lstm_autoencoder"] = {
        "skipped": True,
        "reason": "LSTM removed in this build",
        "tf_available": TF_AVAILABLE,
        "window_errors": np.array([]),
        "window_mask": np.array([]),
        "window_size": 0
    }

    results["meta"] = {"n_rows": n, "n_cols": df_values.shape[1] if hasattr(df_values, "shape") else 1}
    return results

# -------------------------
# Log correlation (TF-IDF)
# -------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def correlate_anomalies_with_logs(anomaly_indices: List[int], logs: List[str], top_k: int = 5) -> Dict[int, List[tuple]]:
    if not logs:
        return {}
    if not isinstance(logs, list):
        raise ValueError("logs must be a list of strings")

    vect = TfidfVectorizer(max_features=2000, stop_words="english")
    tfidf = vect.fit_transform(logs)  # shape (n_logs, n_features)

    out: Dict[int, List[tuple]] = {}
    n_logs = len(logs)
    for idx in anomaly_indices:
        if not isinstance(idx, int):
            try:
                idx = int(idx)
            except Exception:
                continue
        if idx < 0 or idx >= n_logs:
            continue
        sims = cosine_similarity(tfidf[idx], tfidf).flatten()
        order = np.argsort(-sims)
        top = []
        for o in order:
            if o == idx:
                continue
            top.append((int(o), float(sims[o])))
            if len(top) >= top_k:
                break
        out[int(idx)] = top
    return out

# -------------------------
# Utility: map window mask (not used but kept for compatibility)
# -------------------------
def map_window_mask_to_index_mask(n_rows: int, window_size: int, window_mask: np.ndarray) -> np.ndarray:
    mask = np.zeros(n_rows, dtype=bool)
    if window_mask is None or window_mask.size == 0:
        return mask
    for i, v in enumerate(window_mask):
        if v:
            mask[i:i + window_size] = True
    return mask

# explicit exports
__all__ = [
    "preprocess_sensor_data",
    "numeric_summary",
    "run_all_detectors",
    "correlate_anomalies_with_logs",
    "map_window_mask_to_index_mask",
    "TF_AVAILABLE",
]
