import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("Customer Segmentation & Retention Dashboard")

# Load data
rfm = pd.read_csv("rfm_final.csv")
transactions = pd.read_csv("transactions_final.csv")

st.subheader("RFM Data Preview")
st.dataframe(rfm.head())

st.subheader("Transactions Preview")
st.dataframe(transactions.head())

st.success("Dashboard loaded successfully 🚀")
