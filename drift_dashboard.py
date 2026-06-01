import json
import pandas as pd
import streamlit as st
import os

st.set_page_config(
    page_title="Drift Monitoring Dashboard",
    layout="wide"
)

st.title("Drift Monitoring Dashboard")

file_path = "monitoring/drift_logs.json"

# Safe loading
if not os.path.exists(file_path):
    st.warning("No drift logs found yet. Run the API and generate some data.")
    st.stop()

with open(file_path, "r") as f:
    try:
        logs = json.load(f)
    except json.JSONDecodeError:
        st.warning("Drift log file is corrupted or empty.")
        st.stop()

if len(logs) == 0:
    st.warning("No drift data available.")
    st.stop()

df = pd.DataFrame(logs)

# Metrics Cards
col1, col2, col3 = st.columns(3)

col1.metric("Average Groundedness", round(df["groundedness"].mean(), 3))
col2.metric("Average Relevance", round(df["relevance"].mean(), 3))
col3.metric("Total Queries", len(df))

st.divider()

st.subheader("Groundedness Trend")
st.line_chart(df["groundedness"])

st.subheader("Relevance Trend")
st.line_chart(df["relevance"])

st.divider()

st.subheader("Historical Logs")
st.dataframe(df)