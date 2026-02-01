import streamlit as st
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("🚀 Customer Segmentation Dashboard")
st.write("Fast customer segmentation using RFM + KMeans clustering.")

# ---------------------------------------------------------
# Sidebar Option
# ---------------------------------------------------------
mode = st.sidebar.radio(
    "Choose Input Mode:",
    ["Use Demo Dataset", "Upload My Dataset"]
)

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
if mode == "Upload My Dataset":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset Uploaded Successfully!")
    else:
        st.warning("Please upload a dataset to continue.")
        st.stop()

else:
    df = pd.read_csv("transactions_final.csv")
    st.info("✅ Demo dataset loaded automatically.")

# ---------------------------------------------------------
# ✅ SPEED FIX: Limit dataset size
# ---------------------------------------------------------
if len(df) > 5000:
    df = df.sample(5000, random_state=42)
    st.warning("⚡ Large dataset detected → Using a 5000-row sample for fast execution.")

# ---------------------------------------------------------
# Preview Dataset
# ---------------------------------------------------------
st.subheader("📌 Dataset Preview")
st.dataframe(df.head(15))

# ---------------------------------------------------------
# Fix Column Names Automatically
# ---------------------------------------------------------
df.columns = [c.strip().lower() for c in df.columns]
df = df.loc[:, ~df.columns.str.contains("unnamed")]

if "customer_id" in df.columns:
    df.rename(columns={"customer_id": "CustomerID"}, inplace=True)

if "order_date" in df.columns:
    df.rename(columns={"order_date": "InvoiceDate"}, inplace=True)

if "order_amount" in df.columns:
    df.rename(columns={"order_amount": "Amount"}, inplace=True)

# Required column check
if not all(col in df.columns for col in ["CustomerID", "InvoiceDate", "Amount"]):
    st.error("Dataset must contain CustomerID, InvoiceDate, Amount columns.")
    st.stop()

# ---------------------------------------------------------
# Button-Based Execution (FAST FIX)
# ---------------------------------------------------------
st.subheader("Run Customer Segmentation")

if st.button("✅ Run Segmentation Analysis"):

    with st.spinner("Processing... Please wait"):

        # Convert Date
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

        # Build RFM Table
        rfm = df.groupby("CustomerID").agg({
            "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
            "CustomerID": "count",
            "Amount": "sum"
        })

        rfm.columns = ["Recency", "Frequency", "Monetary"]

        st.subheader("📊 RFM Feature Table")
        st.dataframe(rfm.head(10))

        # ✅ Cluster count limited for speed
        max_k = min(4, len(rfm))
        k = st.slider("Number of Customer Segments", 2, max_k, 3)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(rfm)

        # Apply KMeans
        kmeans = KMeans(n_clusters=k, random_state=42)
        rfm["Cluster"] = kmeans.fit_predict(scaled)

        # Show Cluster Distribution
        st.subheader("🎯 Customer Segment Distribution")
        st.bar_chart(rfm["Cluster"].value_counts())

        st.success("✅ Segmentation Completed Successfully!")

else:
    st.info("Click the button above to run segmentation analysis.")
