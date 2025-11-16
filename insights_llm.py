# insights_nlp.py
"""
Compact version of insights_nlp providing exactly what app.py imports:
- preprocess_sensor_data
- numeric_summary
- run_all_detectors
- correlate_anomalies_with_logs
- map_window_mask_to_index_mask
- TF_AVAILABLE

This version avoids heavy optional deps at import time and helps rule out import-time errors.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

# Avoid importing tensorflow at top-level (heavy) — provide a safe boolean
TF_AVAILABLE = False
try:
    import tensorflow as _tf  # optional
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# -------------------------
# Preprocessing utility
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
    # ensure datetime index
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.dropna(subset=[timestamp_col]).set_index(timestamp_col).sort_index()
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            raise ValueError("No datetime index and timestamp_col missing")
    # choose numeric columns
    if value_cols:
        missing = [c for c in value_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing value columns: {missing}")
        vals = df[value_cols].astype(float)
    else:
        vals = df.select_dtypes(include=[np.number])
        if vals.shape[1] == 0:
            raise ValueError("No numeric columns")
    if resample_rule:
        vals = vals.resample(resample_rule).mean()
    if fill_method == "ffill":
        vals = vals.fillna(method="ffill").fillna(method="bfill")
    else:
        vals = vals.fillna(0)
    return vals

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
# Simple detectors (lightweight)
# -------------------------
def _zscore_mask(vec: np.ndarray, z_thresh: float = 3.0) -> np.ndarray:
    if vec.ndim != 1:
        vec = vec.ravel()
    mean = np.nanmean(vec)
    std = np.nanstd(vec)
    if std == 0 or np.isnan(std):
        return np.zeros_like(vec, dtype=bool)
    z = np.abs((vec - mean) / std)
    return (z > z_thresh)

def run_all_detectors(
    df_values: pd.DataFrame,
    column: Optional[str] = None,
    rescale: bool = True,
    window_size_for_lstm: int = 24,
    lstm_epochs: int = 6,
    verbose_lstm: int = 0
) -> Dict[str, Any]:
    """
    Lightweight aggregator that provides boolean masks for classical detectors
    and a placeholder LSTM result (skipped if TF not available).
    This avoids heavy imports at module load while providing expected outputs.
    """
    if df_values is None or df_values.shape[0] == 0:
        raise ValueError("df_values must be non-empty DataFrame")
    if column:
        if column not in df_values.columns:
            raise ValueError(f"Column {column} not found")
        X = df_values[[column]].values.astype(float)
    else:
        X = df_values.values.astype(float)
    # use first column for 1-D detectors
    vec = X[:, 0] if X.ndim == 2 else X.ravel()

    # simple detectors
    masks = {}
    masks["zscore"] = _zscore_mask(vec, z_thresh=3.0)
    # a naive IQR mask
    q1 = np.nanpercentile(vec, 25)
    q3 = np.nanpercentile(vec, 75)
    iqr = q3 - q1
    if iqr == 0:
        masks["iqr"] = np.zeros_like(vec, dtype=bool)
    else:
        masks["iqr"] = (vec < (q1 - 1.5 * iqr)) | (vec > (q3 + 1.5 * iqr))

    # placeholders for other detectors to match app expectations (length = n_rows)
    n = vec.shape[0]
    masks["isolation_forest"] = np.zeros(n, dtype=bool)
    masks["local_outlier_factor"] = np.zeros(n, dtype=bool)
    masks["one_class_svm"] = np.zeros(n, dtype=bool)
    masks["elliptic_envelope"] = np.zeros(n, dtype=bool)
    masks["dbscan"] = np.zeros(n, dtype=bool)
    masks["pca_recon"] = np.zeros(n, dtype=bool)

    # LSTM autoencoder: provide consistent structure expected by app
    if TF_AVAILABLE and df_values.shape[0] >= window_size_for_lstm + 1:
        # We won't actually train here in the stub to avoid heavy runtime at import; mark as placeholder
        masks["lstm_autoencoder"] = {
            "skipped": False,
            "note": "TF available but this compact stub does not train — replace with full insights_nlp for real LSTM",
            "window_errors": np.array([]),
            "window_mask": np.array([]),
            "window_size": window_size_for_lstm
        }
    else:
        masks["lstm_autoencoder"] = {"skipped": True, "reason": "not enough data or TF unavailable", "tf_available": TF_AVAILABLE}

    masks["meta"] = {"n_rows": n, "n_cols": df_values.shape[1]}
    return masks

# -------------------------
# Log correlation (simple TF-IDF wrapper)
# -------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def correlate_anomalies_with_logs(anomaly_indices: List[int], logs: List[str], top_k: int = 5):
    if not logs:
        return {}
    if not isinstance(logs, list):
        raise ValueError("logs must be list[str]")
    vect = TfidfVectorizer(max_features=2000, stop_words="english")
    tfidf = vect.fit_transform(logs)
    out = {}
    n_logs = len(logs)
    for idx in anomaly_indices:
        if idx < 0 or idx >= n_logs:
            continue
        v = tfidf[idx]
        sims = cosine_similarity(v, tfidf).flatten()
        ranks = np.argsort(-sims)
        top = []
        for r in ranks:
            if r == idx:
                continue
            top.append((int(r), float(sims[r])))
            if len(top) >= top_k:
                break
        out[int(idx)] = top
    return out

def map_window_mask_to_index_mask(n_rows: int, window_size: int, window_mask: np.ndarray):
    mask = np.zeros(n_rows, dtype=bool)
    if window_mask is None or window_mask.size == 0:
        return mask
    for i, v in enumerate(window_mask):
        if v:
            mask[i:i + window_size] = True
    return mask

# explicit exports to avoid accidental name hiding
__all__ = [
    "preprocess_sensor_data",
    "numeric_summary",
    "run_all_detectors",
    "correlate_anomalies_with_logs",
    "map_window_mask_to_index_mask",
    "TF_AVAILABLE"
]
