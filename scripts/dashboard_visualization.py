import sys
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# Automatically add the `src` folder to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")

if src_path not in sys.path:
    sys.path.append(src_path)

# Function to load data using SQLAlchemy
def load_data_using_sqlalchemy(query):
    try:
        # Replace with your actual database connection string
        engine = create_engine("postgresql://username:password@localhost/dbname")
        with engine.connect() as connection:
            return pd.read_sql(query, connection)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# SQL Query
query = """
    SELECT 
        "Bearer Id", "Start", "End", "Dur. (ms)", "IMSI", "MSISDN/Number", "IMEI", 
        "Last Location Name", "Avg RTT DL (ms)", "Avg RTT UL (ms)", "Avg Bearer TP DL (kbps)",
        "Avg Bearer TP UL (kbps)", "TCP DL Retrans. Vol (Bytes)", "TCP UL Retrans. Vol (Bytes)",
        "DL TP < 50 Kbps (%)", "50 Kbps < DL TP < 250 Kbps (%)", "250 Kbps < DL TP < 1 Mbps (%)",
        "DL TP > 1 Mbps (%)", "UL TP < 10 Kbps (%)", "10 Kbps < UL TP < 50 Kbps (%)", 
        "50 Kbps < UL TP < 300 Kbps (%)", "UL TP > 300 Kbps (%)", "HTTP DL (Bytes)", 
        "HTTP UL (Bytes)", "Activity Duration DL (ms)", "Activity Duration UL (ms)", 
        "Handset Manufacturer", "Handset Type", "Social Media DL (Bytes)", "Social Media UL (Bytes)",
        "Google DL (Bytes)", "Google UL (Bytes)", "Email DL (Bytes)", "Email UL (Bytes)", 
        "Youtube DL (Bytes)", "Youtube UL (Bytes)", "Netflix DL (Bytes)", "Netflix UL (Bytes)",
        "Gaming DL (Bytes)", "Gaming UL (Bytes)", "Other DL (Bytes)", "Other UL (Bytes)", 
        "Total DL (Bytes)", "Total UL (Bytes)"
    FROM "xdr_data"
"""

# Load data from the database
try:
    df = load_data_using_sqlalchemy(query)
    if df is None or df.empty:
        raise ValueError("No data returned from the database query.")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    sys.exit(1)

# Standardize column names if data is valid
if not df.empty:
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("[().]", "", regex=True)
    )
else:
    st.error("The loaded DataFrame is empty.")
    sys.exit(1)

# Helper function to validate columns
def validate_columns(required_columns, df):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.warning(f"Missing columns: {missing}")
        return False
    return True

# Streamlit Page Setup
st.set_page_config(page_title="Telecom Data Dashboard", layout="wide")
st.title("Telecom Data Dashboard")
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Device Usage",
        "Network Performance",
        "Data Usage Trends",
        "User Activity",
        "Network Throughput",
        "Customer Experience",
    ],
)

# Device Usage Analysis
if page == "Device Usage":
    st.header("Device Usage Analysis")
    if validate_columns(["handset_manufacturer"], df):
        device_counts = df["handset_manufacturer"].value_counts().head(10)
        fig, ax = plt.subplots()
        device_counts.plot(kind="bar", ax=ax, color="skyblue")
        ax.set_title("Top 10 Device Manufacturers")
        ax.set_xlabel("Manufacturer")
        ax.set_ylabel("Count")
        st.pyplot(fig)

# Network Performance Analysis
elif page == "Network Performance":
    st.header("Network Performance Analysis")
    if validate_columns(["start", "avg_rtt_dl_ms", "avg_rtt_ul_ms"], df):
        fig, ax = plt.subplots()
        ax.plot(
            pd.to_datetime(df["start"]), df["avg_rtt_dl_ms"], color="blue", label="Avg RTT DL (ms)"
        )
        ax.plot(
            pd.to_datetime(df["start"]), df["avg_rtt_ul_ms"], color="red", label="Avg RTT UL (ms)"
        )
        ax.set_title("Network Latency Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("RTT (ms)")
        ax.legend()
        st.pyplot(fig)

# Data Usage Trends
elif page == "Data Usage Trends":
    st.header("Data Usage Trends")
    services = [
        "social_media_dl_bytes",
        "youtube_dl_bytes",
        "netflix_dl_bytes",
        "gaming_dl_bytes",
    ]
    if validate_columns(["start"] + services, df):
        df_grouped = df.groupby("start")[services].sum()
        fig, ax = plt.subplots()
        df_grouped.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Data Usage by Service")
        ax.set_xlabel("Date")
        ax.set_ylabel("Data Volume (Bytes)")
        st.pyplot(fig)

# User Activity Analysis
elif page == "User Activity":
    st.header("User Activity Analysis")
    if validate_columns(["activity_duration_dl_ms", "total_dl_bytes"], df):
        fig, ax = plt.subplots()
        ax.scatter(
            df["activity_duration_dl_ms"],
            df["total_dl_bytes"],
            color="green",
            alpha=0.6,
        )
        ax.set_title("Activity Duration vs. Data Usage")
        ax.set_xlabel("Activity Duration DL (ms)")
        ax.set_ylabel("Total DL (Bytes)")
        st.pyplot(fig)

# Network Throughput Analysis
elif page == "Network Throughput":
    st.header("Network Throughput Analysis")
    if validate_columns(["avg_bearer_tp_dl_kbps", "avg_bearer_tp_ul_kbps"], df):
        fig, ax = plt.subplots()
        df[["avg_bearer_tp_dl_kbps", "avg_bearer_tp_ul_kbps"]].boxplot(ax=ax)
        ax.set_title("Throughput Distribution")
        ax.set_ylabel("Throughput (kbps)")
        st.pyplot(fig)

# Customer Experience Analysis
elif page == "Customer Experience":
    st.header("Customer Experience Analysis")
    categories = ["dl_tp_<50_kbps_percent", "dl_tp_>1_mbps_percent"]
    if validate_columns(categories, df):
        df_grouped = df[categories].mean()
        fig, ax = plt.subplots()
        df_grouped.plot(kind="bar", ax=ax, color="purple")
        ax.set_title("Percentage of Users in Throughput Categories")
        ax.set_ylabel("Percentage")
        st.pyplot(fig)
