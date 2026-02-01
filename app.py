import streamlit as st
import pandas as pd
import os

st.title("🚀 Customer Segmentation Dashboard")

# File name in repo
file_path = "rfm_final.csv"

# Safe loader
if os.path.exists(file_path):
    rfm = pd.read_csv(file_path)

    st.success("✅ Data Loaded Successfully!")
    st.dataframe(rfm.head())

    st.subheader("📊 Segment Distribution")
    st.bar_chart(rfm["frequency"])

else:
    st.error("❌ File not found: rfm_final.csv")
    st.write("Please make sure the dataset is uploaded to GitHub.")
