"""
app.py - Streamlit app using insights_nlp.py (no LSTM).
Run:
    streamlit run app.py
"""

import os
from typing import Optional, List
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import functions from insights_nlp (must be in same folder)
from insights_nlp import (
    preprocess_sensor_data,
    numeric_summary,
    run_all_detectors,
    correlate_anomalies_with_logs,
    map_window_mask_to_index_mask,
    TF_AVAILABLE,
)

st.set_page_config(layout="wide", page_title="Sensor Anomaly Explorer (No LSTM)", initial_sidebar_state="auto")

# -------------------------
# Helpers
# -------------------------
def load_csv_uploader(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
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
# Sidebar controls
# -------------------------
st.sidebar.title("Controls")

uploaded_file = st.sidebar.file_uploader("Upload sensor CSV (must have timestamp column)", type=["csv"])
timestamp_col = st.sidebar.text_input("Timestamp column name", value="timestamp")
value_col = st.sidebar.text_input("Value column (leave blank to use first numeric)", value="")
resample_rule = st.sidebar.text_input("Optional resample rule (pandas, e.g., 1T, 5T, 1H)", value="")

run_button = st.sidebar.button("Run detectors")

# Logs uploader
logs_file = st.sidebar.file_uploader("Upload logs TXT (one log per line)", type=["txt"])

# Chat dependencies & key handling
st.sidebar.markdown("---")
st.sidebar.header("Chat (Groq LLM)")

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

user_key = st.sidebar.text_input("Enter GROQ_API_KEY", type="password")
if user_key:
    os.environ["GROQ_API_KEY"] = user_key

api_key = os.getenv("GROQ_API_KEY")
chat_enabled = (len(_missing_chat_deps) == 0) and (api_key is not None and len(api_key) > 0)

if not chat_enabled:
    st.sidebar.error(
        "Chat unavailable: missing dependencies or GROQ_API_KEY.\n"
        "Install: langgraph, langchain_groq, langchain_community or paste key above."
    )
else:
    st.sidebar.success("Chat enabled ✔")

# -------------------------
# Main UI
# -------------------------
st.title("Sensor Anomaly Detection & Log Correlation — No LSTM")

if uploaded_file is None:
    st.info("Upload a sensor CSV to begin. You can also paste GROQ_API_KEY in sidebar to enable chat.")
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

# Numeric summary
first_col = value_col if value_col else df_vals.columns[0]
st.subheader("Numeric summary (first column)")
st.json(numeric_summary(df_vals[first_col]))

# ---------- Robust logs reader ----------
def read_logs_from_uploader_or_local(uploader, local_path="logs.txt"):
    logs_list = None
    if uploader is not None:
        try:
            raw = uploader.read()
            try:
                s = raw.decode("utf-8")
            except Exception:
                s = raw.decode("latin-1", errors="ignore")
            logs_list = [line.strip() for line in s.splitlines() if line.strip()]
            return logs_list
        except Exception as e:
            print("uploader read failed:", e)
            logs_list = None

    if os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
                s = f.read()
            logs_list = [line.strip() for line in s.splitlines() if line.strip()]
            return logs_list
        except Exception as e:
            print("local logs read failed:", e)
            return None

    return None

logs_text = read_logs_from_uploader_or_local(logs_file, local_path="logs.txt")
if logs_text is None:
    st.sidebar.warning("No logs loaded. Upload logs.txt in the sidebar or place logs.txt next to app.py.")
else:
    st.sidebar.success(f"Loaded {len(logs_text)} log lines.")
    st.sidebar.write("Sample logs (first 5):")
    for l in logs_text[:5]:
        st.sidebar.write(l)

# Chat UI (main area) when enabled
if chat_enabled:
    st.subheader("💬 Chat — Groq LLM")
    try:
        llm = ChatGroq(groq_api_key=api_key, model_name="llama3-8b-8192", temperature=0.2)
    except Exception as e:
        st.error(f"Failed to initialize ChatGroq: {e}")
        llm = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Render history
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.write(f"🧑‍💻 **You:** {msg.content}")
        else:
            st.write(f"🤖 **AI:** {msg.content}")

    user_msg = st.text_input("Type your message:")
    if user_msg and llm is not None:
        st.session_state.chat_history.append(HumanMessage(content=user_msg))
        with st.spinner("Waiting for AI..."):
            try:
                resp = llm(st.session_state.chat_history)
                st.session_state.chat_history.append(AIMessage(content=resp.content))
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Chat request failed: {e}")

# Run detectors
if run_button:
    st.subheader("Running detectors (this may take a short while)...")
    with st.spinner("Running models..."):
        try:
            results = run_all_detectors(
                df_values=df_vals,
                column=None if not value_col else value_col,
            )
        except Exception as e:
            st.error(f"Error running detectors: {e}")
            st.stop()

    st.success("Detectors finished. Showing results below.")

    # Show results for classical detectors
    detector_names = [k for k in results.keys() if k not in ("meta", "lstm_autoencoder")]
    cols = st.columns(2)
    for i, name in enumerate(detector_names):
        mask = results.get(name)
        mask_bool = None
        try:
            mask_bool = np.asarray(mask, dtype=bool)
        except Exception:
            mask_bool = None
        with cols[i % 2]:
            st.metric(label=name, value=(str(int(mask_bool.sum())) + " anomalies") if (mask_bool is not None and mask_bool.size > 0) else "N/A")
            if mask_bool is not None and mask_bool.size > 0:
                anomaly_idx = np.where(mask_bool)[0][:10].tolist()
                st.write("Top indices:", anomaly_idx)

    # Inform user LSTM removed
    st.info("LSTM removed — using only classical detectors.")

    # Consensus
    st.subheader("Consensus anomalies (majority vote across detectors)")
    masks_list = []
    for k in detector_names:
        m = results.get(k)
        try:
            mb = np.asarray(m, dtype=bool)
            if mb.shape[0] == df_vals.shape[0]:
                masks_list.append(mb)
        except Exception:
            pass

    if masks_list:
        sum_mask = np.sum(np.stack(masks_list, axis=0).astype(int), axis=0)
        threshold = max(1, int(len(masks_list) / 2))
        consensus = sum_mask > threshold
        st.write(f"Consensus anomalies count: {int(consensus.sum())}")
        plot_timeseries(df_vals, first_col, anomalies_mask=consensus, title="Consensus anomalies")
    else:
        st.info("Not enough detector masks available to compute consensus.")

    # Correlate with logs if present
    if logs_text:
        st.subheader("Correlate anomalies with logs (TF-IDF)")
        anomaly_indices = []
        if 'consensus' in locals() and isinstance(consensus, (list, np.ndarray)) and np.any(consensus):
            anomaly_indices = list(np.where(consensus)[0][:20])
        else:
            # fallback: pick first 20 anomalies from any detector
            for k in detector_names:
                m = results.get(k)
                try:
                    mb = np.asarray(m, dtype=bool)
                    if mb.sum() > 0:
                        anomaly_indices = list(np.where(mb)[0][:20])
                        break
                except Exception:
                    continue

        n_logs = len(logs_text)
        # Clamp indices
        clamped = []
        for idx in anomaly_indices:
            try:
                idx_i = int(idx)
            except Exception:
                continue
            if n_logs > 0:
                clamped_idx = max(0, min(idx_i, n_logs - 1))
                clamped.append(clamped_idx)

        if not clamped:
            st.info("No valid anomaly indices to correlate with logs (after clamping).")
        else:
            st.write("Using anomaly indices (clamped to log length):", clamped[:10])
            corr = correlate_anomalies_with_logs(clamped, logs_text, top_k=5)
            st.write("Top log matches for each anomaly index:")
            st.json(corr)
    else:
        st.info("Logs not provided — skipping correlation step.")

# Footer notes
st.markdown("---")
st.write("Notes:")
if not TF_AVAILABLE:
    st.warning("TensorFlow not available (LSTM removed).")
else:
    st.success("TensorFlow available (but LSTM not used).")

st.write("Paste GROQ_API_KEY in the sidebar to enable chat or install the required packages: langgraph, langchain_groq, langchain_community.")
