import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Enterprise Customer Analytics",
    layout="wide"
)

st.title("🚀 Enterprise Customer Segmentation & Churn Dashboard")
st.write("Upload transaction-level data and get customer segments + churn risk insights.")

# ----------------------------
# FILE UPLOADER
# ----------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Transactions Dataset (CSV)",
    type=["csv"]
)

# ----------------------------
# HELPER FUNCTION: Build RFM
# ----------------------------
def build_rfm(df):
    """
    Builds Recency, Frequency, Monetary features.
    Dataset must contain:
    CustomerID, InvoiceDate, Amount
    """

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "CustomerID": "count",
        "Amount": "sum"
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]
    return rfm


# ----------------------------
# MAIN APP
# ----------------------------
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.success("✅ Dataset Uploaded Successfully!")
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # Validate Required Columns
    required_cols = ["CustomerID", "InvoiceDate", "Amount"]

    if not all(col in df.columns for col in required_cols):
        st.error("❌ Dataset must contain columns: CustomerID, InvoiceDate, Amount")
        st.stop()

    # ----------------------------
    # RFM FEATURE ENGINEERING
    # ----------------------------
    st.subheader("✅ Step 1: RFM Feature Engineering")

    rfm = build_rfm(df)
    st.dataframe(rfm.head())

    # ----------------------------
    # SCALING
    # ----------------------------
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    # ----------------------------
    # CUSTOMER SEGMENTATION (KMeans)
    # ----------------------------
    st.subheader("✅ Step 2: Customer Segmentation (KMeans Clustering)")

    k = st.slider("Select number of clusters (segments)", 2, 6, 4)

    kmeans = KMeans(n_clusters=k, random_state=42)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    st.write("Cluster Distribution:")
    st.bar_chart(rfm["Cluster"].value_counts())

    # ----------------------------
    # SEGMENT NAMING (Business Labels)
    # ----------------------------
    st.subheader("✅ Step 3: Business Segment Naming")

    def label_segment(row):
        if row["Recency"] < 30 and row["Monetary"] > rfm["Monetary"].quantile(0.75):
            return "VIP Customers 💎"
        elif row["Recency"] > 90:
            return "At Risk Customers ⚠️"
        elif row["Frequency"] > rfm["Frequency"].quantile(0.75):
            return "Loyal Customers ❤️"
        else:
            return "New/Regular Customers 🆕"

    rfm["Segment_Name"] = rfm.apply(label_segment, axis=1)

    st.write("Segment Sample:")
    st.dataframe(rfm.head())

    # ----------------------------
    # CHURN PREDICTION MODEL
    # ----------------------------
    st.subheader("✅ Step 4: Churn Risk Prediction")

    # Define churn = inactive > 90 days
    rfm["Churn"] = (rfm["Recency"] > 90).astype(int)

    X = rfm[["Recency", "Frequency", "Monetary"]]
    y = rfm["Churn"]

    model = RandomForestClassifier()
    model.fit(X, y)

    rfm["Churn_Risk"] = model.predict_proba(X)[:, 1]

    st.write("Top 10 High Churn Risk Customers:")
    st.dataframe(
        rfm.sort_values("Churn_Risk", ascending=False)
        .head(10)
    )

    # ----------------------------
    # DASHBOARD KPIs
    # ----------------------------
    st.subheader("📌 Executive Summary KPIs")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", len(rfm))
    col2.metric("VIP Customers", (rfm["Segment_Name"] == "VIP Customers 💎").sum())
    col3.metric("At Risk Customers", (rfm["Segment_Name"] == "At Risk Customers ⚠️").sum())
    col4.metric("Avg Churn Risk", round(rfm["Churn_Risk"].mean(), 2))

    # ----------------------------
    # VISUALIZATION: Frequency vs Monetary
    # ----------------------------
    st.subheader("📊 Cluster Visualization (Frequency vs Monetary)")

    fig = plt.figure()
    plt.scatter(rfm["Frequency"], rfm["Monetary"], c=rfm["Cluster"])
    plt.xlabel("Frequency")
    plt.ylabel("Monetary Value")
    plt.title("Customer Segments")

    st.pyplot(fig)

    # ----------------------------
    # DOWNLOAD RESULTS
    # ----------------------------
    st.subheader("⬇️ Download Final Customer Segmentation Report")

    csv = rfm.to_csv().encode("utf-8")

    st.download_button(
        "Download Segmentation Report",
        csv,
        "customer_segmentation_results.csv",
        "text/csv"
    )

    st.success("✅ Enterprise Segmentation & Churn Dashboard Ready!")

else:
    st.warning("Please upload a dataset to start the analysis.")
