import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Analytics Dashboard", layout="wide")

st.title("🚀 Enterprise Customer Segmentation & Churn Dashboard")

rfm = pd.read_csv("data/rfm_final.csv")

st.subheader("📌 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(rfm))
col2.metric("Avg Revenue", round(rfm["Monetary"].mean(),2))
col3.metric("Churn Risk Customers", (rfm["Recency"] > 90).sum())

st.subheader("📊 Segment Distribution")
st.bar_chart(rfm["Cluster"].value_counts())

st.subheader("⚠️ High Churn Risk Customers")
st.dataframe(rfm[rfm["Recency"] > 90].head(10))

st.success("✅ Dashboard Loaded Successfully!")
