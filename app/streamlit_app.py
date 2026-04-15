import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
import plotly.express as px

from src.preprocess import preprocess_climate_data
from src.anomaly_detection import detect_temperature_anomalies

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Climate Intelligence Dashboard", layout="wide")
st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background-color: #0E1117;
    }

    /* Main area headings */
    h1, h2, h3 {
        color: #FFFFFF !important;
    }

    /* Main area paragraph text */
    .stApp p {
        color: #CFCFCF !important;
    }

    /* Metric labels */
    div[data-testid="stMetricLabel"] {
        color: #AAAAAA !important;
    }

    /* Metric values */
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 28px !important;
        font-weight: bold;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #F0F2F6 !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] span {
        color: #111111 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Custom Styling (Premium Look)
# -------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Title Section
# -------------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>🌍 Climate Intelligence Dashboard</h1>
    <p style='text-align: center;'>Advanced climate trend analysis and forecasting system</p>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.title("⚙️ Controls")

data_path = PROJECT_ROOT / "data" / "processed" / "cleaned_climate_data.csv"

# -------------------------------
# Load Data
# -------------------------------
try:
    df = pd.read_csv(data_path, low_memory=False)
    st.success("Dataset loaded successfully.")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Convert date
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# -------------------------------
# Apply Preprocessing + Anomaly Detection
# -------------------------------
df = preprocess_climate_data(df)
df, temp_anomalies = detect_temperature_anomalies(df)

# -------------------------------
# Sidebar Filter
# -------------------------------
year_range = st.sidebar.slider(
    "Select Year Range",
    int(df["year"].min()),
    int(df["year"].max()),
    (int(df["year"].min()), int(df["year"].max()))
)

df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]

# -------------------------------
# Metrics Section (Premium Cards)
# -------------------------------
st.subheader("📊 Climate Overview")

col1, col2, col3 = st.columns(3)

col1.metric("🌡 Avg Temperature", round(df["avg_temp"].mean(), 2))
col2.metric("🚨 Anomalies Detected", len(temp_anomalies))
col3.metric("🌧 Total Rainfall", round(df["precipitation"].sum(), 0))

# -------------------------------
# Dataset Preview
# -------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# Temperature Trend
# -------------------------------
yearly_temp = df.groupby("year", as_index=False)["avg_temp"].mean()

fig1 = px.line(
    yearly_temp,
    x="year",
    y="avg_temp",
    title="📈 Long-Term Temperature Trend",
    markers=True,
)
st.plotly_chart(fig1, use_container_width=True)

# -------------------------------
# Rainfall Trend
# -------------------------------
yearly_rain = df.groupby("year", as_index=False)["precipitation"].sum()

fig2 = px.line(
    yearly_rain,
    x="year",
    y="precipitation",
    title="🌧 Rainfall Pattern Over Time",
    markers=True,
)
st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# Monthly Pattern
# -------------------------------
monthly_temp = (
    df.groupby(["month", "month_name"], as_index=False)["avg_temp"]
    .mean()
    .sort_values("month")
)

fig3 = px.bar(
    monthly_temp,
    x="month_name",
    y="avg_temp",
    title="📊 Average Monthly Temperature Pattern",
)

st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# Anomaly Table
# -------------------------------
st.subheader("🚨 Detected Anomalies")

if len(temp_anomalies) > 0:
    st.dataframe(temp_anomalies[["date", "avg_temp", "z_score"]].head(20))
else:
    st.info("No anomalies detected.")