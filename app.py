"""
app.py - Streamlit UI with detector dropdown (single or Run ALL).
Includes groq API sidebar key, robust logs ingestion, timestamp-aware + TF-IDF log analysis,
and a robust "Analyze logs now" button that runs correlation even if detectors haven't been run.
"""

import os
import re
from datetime import datetime, timedelta
from typing import Optional, List
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from insights_nlp import (
    preprocess_sensor_data,
    numeric_summary,
    run_single_detector,
    run_all_detectors,
    correlate_anomalies_with_logs,
    TF_AVAILABLE,
)

st.set_page_config(page_title="Sensor Anomaly Explorer", layout="wide")

# -------------------------
# Helpers
# -------------------------
def load_csv_uploader(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)

def plot_timeseries(df: pd.DataFrame, column: str, anomalies_mask: Optional[np.ndarray] = None, title: str = ""):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df.index, df[column].values, label=column)
    if anomalies_mask is not None:
        mask = np.asarray(anomalies_mask, dtype=bool)
        if mask.shape[0] == df.shape[0]:
            ax.scatter(df.index[mask], df[column].values[mask], color="red", s=20, label="anomaly")
    ax.set_title(title or f"Time series: {column}")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# -------------------------
# Sidebar controls (ensure visible)
# -------------------------
st.sidebar.title("Controls")

uploaded_file = st.sidebar.file_uploader("Upload sensor CSV", type=["csv"])
timestamp_col = st.sidebar.text_input("Timestamp column", value="timestamp")
value_col = st.sidebar.text_input("Value column (leave blank to use first numeric)", value="")
resample_rule = st.sidebar.text_input("Optional resample rule (e.g., 1T, 5T, 1H)", value="")

# Detector UI block (must be present)
st.sidebar.markdown("### Detector Settings")
detector_choices = [
    "Z-Score",
    "IQR",
    "Isolation Forest",
    "Local Outlier Factor (LOF)",
    "One-Class SVM",
    "Elliptic Envelope",
    "DBSCAN",
    "PCA Reconstruction",
    "Run ALL Detectors",
]
selected_detector = st.sidebar.selectbox("Detector", detector_choices, index=0)

pca_components = st.sidebar.number_input("PCA components (for PCA Reconstruction)", min_value=1, value=1, step=1)
run_button = st.sidebar.button("Run detector")

# GROQ API key block
st.sidebar.markdown("---")
st.sidebar.subheader("🔑 Groq API Key")
groq_key = st.sidebar.text_input("Enter GROQ_API_KEY", type="password")
# if groq_key:
#     os.environ["GROQ_API_KEY"] = groq_key
# api_key = os.getenv("GROQ_API_KEY")
if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key.strip()

api_key = os.getenv("GROQ_API_KEY")

if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key.strip()
    st.sidebar.success("GROQ_API_KEY set ✔")
else:
    st.sidebar.warning("Enter your GROQ_API_KEY above")

_missing_chat_deps = []
try:
    from langchain_groq import ChatGroq
    from langchain.schema import HumanMessage, AIMessage
except Exception:
    _missing_chat_deps.append("langchain_groq")
try:
    import langgraph
except Exception:
    _missing_chat_deps.append("langgraph")
try:
    import langchain_community
except Exception:
    _missing_chat_deps.append("langchain_community")

chat_enabled = bool(api_key) and (len(_missing_chat_deps) == 0)
if chat_enabled:
    st.sidebar.success("Groq Chat Enabled ✔")
else:
    st.sidebar.info("Groq Chat disabled. Paste key above and install deps.")

# logs uploader
st.sidebar.markdown("---")
logs_file = st.sidebar.file_uploader("Upload logs.txt (one line per entry)", type=["txt"])

def read_logs_from_uploader_or_local(uploader, local_path="logs.txt"):
    if uploader:
        try:
            raw = uploader.read()
            try:
                s = raw.decode("utf-8")
            except Exception:
                s = raw.decode("latin-1", errors="ignore")
            return [line.strip() for line in s.splitlines() if line.strip()]
        except Exception:
            pass
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
            s = f.read()
        return [line.strip() for line in s.splitlines() if line.strip()]
    return None

logs_text = read_logs_from_uploader_or_local(logs_file)
if logs_text:
    st.sidebar.success(f"Loaded {len(logs_text)} log lines")
    for l in logs_text[:5]:
        st.sidebar.write(l)
else:
    st.sidebar.warning("No logs loaded. Upload logs.txt or place it next to app.py")

# -------------------------
# Timestamp parsing helper
# -------------------------
iso_like = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")

def _try_parse_log_ts(line: str) -> Optional[datetime]:
    m = iso_like.search(line)
    if m:
        try:
            s = m.group(0).replace("T", " ")
            return datetime.fromisoformat(s)
        except Exception:
            pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%d/%b/%Y:%H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            ts = datetime.strptime(line[:len(fmt)], fmt)
            return ts
        except Exception:
            continue
    return None

# -------------------------
# Improved Analyze logs (verbose + timestamp-aware)
# -------------------------
def analyze_logs_verbose(logs_text, df_vals=None, anomaly_indices=None, window_minutes=5, top_k=5):
    meta = {"n_logs": 0, "n_rows": 0, "mode": None, "used_indices": []}
    if not logs_text:
        return {"meta": meta, "results": {}}

    meta["n_logs"] = len(logs_text)

    parsed_logs = []
    logs_have_ts = False
    for i, line in enumerate(logs_text):
        ts = _try_parse_log_ts(line)
        parsed_logs.append((i, line, ts))
        if ts:
            logs_have_ts = True

    if df_vals is not None and isinstance(df_vals, pd.DataFrame):
        meta["n_rows"] = int(df_vals.shape[0])

    if logs_have_ts and (df_vals is not None) and isinstance(df_vals.index, pd.DatetimeIndex):
        meta["mode"] = "timestamp_window"
        results = {}

        if anomaly_indices:
            idxs = anomaly_indices
        else:
            try:
                col0 = df_vals.columns[0]
                vals = df_vals[col0].values
                med = np.nanmedian(vals)
                dev = np.abs(vals - med)
                idxs = list(np.argsort(-dev)[:min(20, len(vals))])
            except Exception:
                idxs = list(range(min(20, len(df_vals))))

        meta["used_indices"] = idxs

        for ai in idxs:
            try:
                ai = int(ai)
                if ai < 0 or ai >= len(df_vals):
                    continue
                at = df_vals.index[ai]
            except Exception:
                continue

            window_start = at - timedelta(minutes=window_minutes)
            window_end = at + timedelta(minutes=window_minutes)
            matches = []
            for i, line, lts in parsed_logs:
                if lts is None:
                    continue
                if window_start <= lts <= window_end:
                    matches.append((i, line, float((lts - at).total_seconds())))
            matches = sorted(matches, key=lambda x: abs(x[2]))[:top_k]
            results[int(ai)] = [{"log_index": m[0], "delta_seconds": m[2], "text": m[1]} for m in matches]
        return {"meta": meta, "results": results}

    # TF-IDF fallback
    meta["mode"] = "tfidf_index_based"
    if anomaly_indices:
        indices = list(map(int, anomaly_indices))
    else:
        if df_vals is not None:
            try:
                col0 = df_vals.columns[0]
                vals = df_vals[col0].values
                med = np.nanmedian(vals)
                dev = np.abs(vals - med)
                indices = list(np.argsort(-dev)[:min(20, len(vals))])
            except Exception:
                indices = list(range(min(20, df_vals.shape[0])))
        else:
            indices = list(range(min(20, len(logs_text))))

    n_logs = len(logs_text)
    clamped = [max(0, min(int(i), n_logs - 1)) for i in indices]
    seen = set()
    clamped_unique = []
    for i in clamped:
        if i not in seen:
            clamped_unique.append(i)
            seen.add(i)

    meta["used_indices"] = clamped_unique
    corr = correlate_anomalies_with_logs(clamped_unique, logs_text, top_k=top_k)
    return {"meta": meta, "results": corr}

# -------------------------
# "Analyze logs now" button
# -------------------------
st.sidebar.markdown("---")
if logs_text:
    if st.sidebar.button("Analyze logs now (verbose)"):
        st.subheader("Logs Analysis — verbose mode")

        local_df = globals().get("df_vals", None)
        st.write(f"Dataset rows: {local_df.shape[0] if isinstance(local_df, pd.DataFrame) else 'N/A'}")
        st.write(f"Loaded log lines: {len(logs_text)}")

        preferred_indices = None
        if 'consensus' in globals() and isinstance(globals()['consensus'], (np.ndarray, list)) and np.any(globals()['consensus']):
            preferred_indices = list(np.where(globals()['consensus'])[0][:20])
            st.write("Using consensus indices:", preferred_indices[:10])
        elif 'mask_bool' in globals() and isinstance(globals()['mask_bool'], (np.ndarray, list)) and np.any(globals()['mask_bool']):
            preferred_indices = list(np.where(globals()['mask_bool'])[0][:20])
            st.write("Using last detector indices:", preferred_indices[:10])
        else:
            st.write("Using fallback indices.")

        out = analyze_logs_verbose(logs_text, df_vals=local_df, anomaly_indices=preferred_indices, window_minutes=5, top_k=5)

        st.write("Mode:", out["meta"]["mode"])
        st.write("Used indices:", out["meta"]["used_indices"][:20])
        if out["meta"]["mode"] == "timestamp_window":
            st.write("Timestamp-based matches:")
            for ai, matches in out["results"].items():
                st.write(f"Anomaly row {ai}: {len(matches)} matches")
                for m in matches:
                    st.write(f"  log_idx={m['log_index']} delta_s={m['delta_seconds']:.1f}  {m['text']}")
        else:
            st.write("TF-IDF matches:")
            st.json(out["results"])
else:
    st.sidebar.info("Upload logs.txt then click 'Analyze logs now'.")

# -------------------------
# Main UI
# -------------------------
st.title("Sensor Anomaly Detection & Log Correlation")

if uploaded_file is None:
    st.info("Upload a sensor CSV to begin.")
    st.stop()

# Load CSV
try:
    df_raw = load_csv_uploader(uploaded_file)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

st.subheader("Raw data preview")
st.dataframe(df_raw.head(200))

# Preprocess
try:
    df_vals = preprocess_sensor_data(df_raw, timestamp_col=timestamp_col, value_cols=[value_col] if value_col else None, resample_rule=resample_rule or None)
except Exception as e:
    st.error(f"Preprocessing failed: {e}")
    st.stop()

st.subheader("Preprocessed data (index is timestamp)")
st.dataframe(df_vals.head(200))

first_col = value_col if value_col else df_vals.columns[0]
st.subheader("Numeric summary (first column)")
st.json(numeric_summary(df_vals[first_col]))

# -------------------------
# Groq Chat
# -------------------------
if chat_enabled:
    st.subheader("💬 Groq LLM Chat")
    try:
        llm = ChatGroq(groq_api_key=api_key, model_name="llama3-8b-8192", temperature=0.2)
    except Exception as e:
        st.error(f"Failed to initialize ChatGroq: {e}")
        llm = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.write(f"🧑‍💻 **You:** {msg.content}")
        else:
            st.write(f"🤖 **AI:** {msg.content}")

    user_msg = st.text_input("Type your message:")
    if user_msg and llm is not None:
        st.session_state.chat_history.append(HumanMessage(content=user_msg))
        with st.spinner("Waiting for AI..."):
            resp = llm(st.session_state.chat_history)
            st.session_state.chat_history.append(AIMessage(content=resp.content))
            st.experimental_rerun()

# -------------------------
# Run detector(s)
# -------------------------
if run_button:
    st.subheader("Running detector(s)...")
    try:
        if selected_detector == "Run ALL Detectors":
            results = run_all_detectors(df_vals, pca_n_components=pca_components)

            detector_names = [k for k in results.keys() if k not in ("meta",)]
            cols = st.columns(2)
            for i, name in enumerate(detector_names):
                mask = np.asarray(results[name], dtype=bool)
                with cols[i % 2]:
                    st.metric(label=name, value=f"{int(mask.sum())} anomalies")
                    st.write("Example indices:", list(np.where(mask)[0][:10]))

            masks = []
            for name in detector_names:
                m = np.asarray(results[name], dtype=bool)
                if m.shape[0] == df_vals.shape[0]:
                    masks.append(m)

            if masks:
                M = np.stack(masks).astype(int)
                threshold = max(1, len(masks) // 2)
                consensus = M.sum(axis=0) > threshold
                globals()['consensus'] = consensus
                st.subheader("Consensus anomalies")
                st.write(f"Total: {int(consensus.sum())}")
                plot_timeseries(df_vals, first_col, consensus, "Consensus anomalies")
            else:
                st.info("Not enough detector masks.")

        else:
            map_name = {
                "Z-Score": "zscore",
                "IQR": "iqr",
                "Isolation Forest": "isolation_forest",
                "Local Outlier Factor (LOF)": "lof",
                "One-Class SVM": "one_class_svm",
                "Elliptic Envelope": "elliptic_envelope",
                "DBSCAN": "dbscan",
                "PCA Reconstruction": "pca_recon",
            }
            det_key = map_name[selected_detector]

            if det_key == "pca_recon":
                mask = run_single_detector(df_vals, det_key, n_components=pca_components)
            else:
                mask = run_single_detector(df_vals, det_key)

            mask_bool = np.asarray(mask, dtype=bool)
            globals()['mask_bool'] = mask_bool

            st.metric(label=selected_detector, value=f"{int(mask_bool.sum())} anomalies")
            st.write("Example indices:", list(np.where(mask_bool)[0][:20]))
            plot_timeseries(df_vals, first_col, mask_bool, f"Anomalies: {selected_detector}")

    except Exception as e:
        st.error(f"Detector run failed: {e}")
        st.stop()

# Footer
st.markdown("---")
st.write("Notes: TensorFlow (LSTM) is not used.")
st.write("Groq Chat enabled:", bool(chat_enabled))
