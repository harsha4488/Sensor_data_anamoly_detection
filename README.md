# Sensor_data_anamoly_detection

# Sensor Anomaly Detection & Log Intelligence System

This project is a full analytical workflow designed for identifying anomalies in raw sensor data and automatically correlating them with human-readable log messages. It aims to bridge the gap between numeric metrics and textual observability.

---

## Motivation

Engineering teams often analyze metrics (like temperature, voltage, pressure) separately from textual logs.  
This tool connects both worlds:

 *If a temperature spike happens at 10:14:23, which log messages are most relevant?*  
This application helps answer that question.

---

## 🚦 Key Features

### 1. Sensor Data Processing
- CSV upload (timestamp + numeric values)
- Automatic cleaning
- Resampling (per minute, per hour, etc.)
- Summary statistics

### 2. Anomaly Detection (Statistical)
- Z-score filtering  
- IQR-based outlier detection  
- Combined consensus detector

### 3. Log Correlation Engine
- Load logs from upload or from file
- One log entry per line
- TF-IDF vectorization
- Similarity ranking
- Show top related logs per anomaly

### 4. (Optional) Groq LLM Chat
If `GROQ_API_KEY` is provided, you can:
- Ask follow-up questions
- Analyze logs in natural language
- Request summaries or improvements

---

##  Setup

### Install
```bash
pip install -r requirements.txt
