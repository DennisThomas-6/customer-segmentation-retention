import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Enterprise Customer Analytics Dashboard",
    layout="wide"
)

st.title("🚀 Enterprise Customer Segmentation & Churn Dashboard")
st.write(
    "This dashboard performs Customer Segmentation + Churn Risk Prediction "
    "using RFM Analytics and Machine Learning."
)

# ---------------------------------------------------------
# SIDEBAR MODE SELECTION
# ---------------------------------------------------------
st.sidebar.title("⚙️ Dashboard Input Mode")

mode = st.sidebar.radio(
    "Select Data Mode:",
    ["✅ Use Demo Dataset (Recommended)", "📂 Upload My Dataset"]
)

# ---------------------------------------------------------
# LOAD DATA BASED ON MODE
# ---------------------------------------------------------
if mode == "📂 Upload My Dataset":

    uploaded_file = st.file_uploader(
        "Upload Transactions Dataset (CSV)",
        type=["csv"]
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Custom Dataset Uploaded Successfully!")

    else:
        st.warning("⚠️ Please upload a dataset to continue.")
        st.stop()

else:
    st.info("✅ Demo Dataset Loaded Automatically for Portfolio Showcase")

    # ✅ DEMO DATASET FILE FROM YOUR REPO
    df = pd.read_csv("transactions_final.csv")

    st.write("Preview of Demo Dataset:")
    st.dataframe(df.head())


# ---------------------------------------------------------
# AUTO COLUMN STANDARDIZATION
# ---------------------------------------------------------
df.columns = [c.lower() for c in df.columns]

# Rename columns if needed
if "customer_id" in df.columns:
    df.rename(columns={"customer_id": "CustomerID"}, inplace=True)

if "amount" not in df.columns and "monetary" in df.columns:
    df.rename(columns={"monetary": "Amount"}, inplace=True)

if "invoicedate" in df.columns:
    df.rename(columns={"invoicedate": "InvoiceDate"}, inplace=True)

# Required columns check
required_cols = ["CustomerID", "InvoiceDate", "Amount"]

if not all(col in df.columns for col in required_cols):
    st.error("❌ Dataset must contain: CustomerID, InvoiceDate, Amount")
    st.stop()


# ---------------------------------------------------------
# STEP 1: BUILD RFM FEATURES
# ---------------------------------------------------------
st.subheader("✅ Step 1: RFM Feature Engineering")

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "CustomerID": "count",
    "Amount": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

st.dataframe(rfm.head())


# ---------------------------------------------------------
# STEP 2: CUSTOMER SEGMENTATION (KMeans)
# ---------------------------------------------------------
st.subheader("✅ Step 2: Customer Segmentation (Clustering)")

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

k = st.slider("Select Number of Customer Segments", 2, 6, 4)

kmeans = KMeans(n_clusters=k, random_state=42)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

st.write("📊 Cluster Distribution")
st.bar_chart(rfm["Cluster"].value_counts())


# ---------------------------------------------------------
# STEP 3: BUSINESS SEGMENT NAMING
# ---------------------------------------------------------
st.subheader("✅ Step 3: Segment Naming (VIP / Loyal / At Risk)")

def label_segment(row):

    if row["Recency"] < 30 and row["Monetary"] > rfm["Monetary"].quantile(0.75):
        return "VIP Customers 💎"

    elif row["Recency"] > 90:
        return "At Risk Customers ⚠️"

    elif row["Frequency"] > rfm["Frequency"].quantile(0.75):
        return "Loyal Customers ❤️"

    else:
        return "New / Regular Customers 🆕"


rfm["Segment_Name"] = rfm.apply(label_segment, axis=1)

st.write("Segment Examples:")
st.dataframe(rfm.head())


# ---------------------------------------------------------
# STEP 4: CHURN PREDICTION
# ---------------------------------------------------------
st.subheader("✅ Step 4: Churn Risk Prediction (ML Model)")

# Define churn label: inactive > 90 days
rfm["Churn"] = (rfm["Recency"] > 90).astype(int)

X = rfm[["Recency", "Frequency", "Monetary"]]
y = rfm["Churn"]

model = RandomForestClassifier()
model.fit(X, y)

rfm["Churn_Risk_Score"] = model.predict_proba(X)[:, 1]

st.write("⚠️ Top 10 Customers Most Likely to Churn")
st.dataframe(
    rfm.sort_values("Churn_Risk_Score", ascending=False).head(10)
)


# ---------------------------------------------------------
# KPI DASHBOARD SUMMARY
# ---------------------------------------------------------
st.subheader("📌 Executive KPI Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", len(rfm))
col2.metric("VIP Customers", (rfm["Segment_Name"] == "VIP Customers 💎").sum())
col3.metric("At Risk Customers", (rfm["Segment_Name"] == "At Risk Customers ⚠️").sum())
col4.metric("Avg Churn Risk", round(rfm["Churn_Risk_Score"].mean(), 2))


# ---------------------------------------------------------
# CLUSTER VISUALIZATION
# ---------------------------------------------------------
st.subheader("📊 Cluster Visualization (Frequency vs Monetary)")

fig = plt.figure()
plt.scatter(rfm["Frequency"], rfm["Monetary"], c=rfm["Cluster"])
plt.xlabel("Frequency")
plt.ylabel("Monetary Value")
plt.title("Customer Segments Visualization")

st.pyplot(fig)


# ---------------------------------------------------------
# DOWNLOAD REPORT
# ---------------------------------------------------------
st.subheader("⬇️ Download Full Customer Segmentation Report")

csv = rfm.to_csv().encode("utf-8")

st.download_button(
    label="Download Customer Segmentation Results",
    data=csv,
    file_name="customer_segmentation_report.csv",
    mime="text/csv"
)

st.success("✅ Enterprise Dashboard Successfully Generated!")
