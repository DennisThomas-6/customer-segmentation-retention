import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Customer Analytics Dashboard", layout="wide")

st.title("🚀 Enterprise Customer Segmentation & Churn Dashboard")
st.write(
    "A real-world customer analytics system with segmentation, churn prediction, "
    "and business insights using RFM + Machine Learning."
)

# ---------------------------------------------------------
# SIDEBAR MODE
# ---------------------------------------------------------
st.sidebar.title("Data Input")
mode = st.sidebar.radio(
    "Choose dataset mode:",
    ["Use Demo Dataset", "Upload My Dataset"]
)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
if mode == "Upload My Dataset":

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset uploaded successfully!")
    else:
        st.warning("Please upload a dataset to continue.")
        st.stop()

else:
    df = pd.read_csv("transactions_final.csv")
    st.info("✅ Demo dataset loaded automatically for showcase.")

st.subheader("📌 Dataset Preview (First 20 rows)")
st.dataframe(df.head(20))

# ---------------------------------------------------------
# UNIVERSAL COLUMN FIX
# ---------------------------------------------------------
df.columns = [c.strip().lower() for c in df.columns]
df = df.loc[:, ~df.columns.str.contains("unnamed")]

if "customer_id" in df.columns:
    df.rename(columns={"customer_id": "CustomerID"}, inplace=True)

if "order_date" in df.columns:
    df.rename(columns={"order_date": "InvoiceDate"}, inplace=True)

if "order_amount" in df.columns:
    df.rename(columns={"order_amount": "Amount"}, inplace=True)

if not all(col in df.columns for col in ["CustomerID", "InvoiceDate", "Amount"]):
    st.error("Dataset must contain CustomerID, InvoiceDate, Amount columns.")
    st.stop()

# ---------------------------------------------------------
# ✅ CACHED RFM FUNCTION (FAST)
# ---------------------------------------------------------
@st.cache_data
def build_rfm(df):
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "CustomerID": "count",
        "Amount": "sum"
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]
    return rfm

rfm = build_rfm(df)

st.subheader("📊 RFM Customer Features")
st.dataframe(rfm.head(15))

# ---------------------------------------------------------
# ✅ CACHED KMEANS CLUSTERING (FAST)
# ---------------------------------------------------------
@st.cache_data
def run_kmeans(rfm, k):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm)

    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(scaled)

    return clusters

# Ensure cluster slider never exceeds customer count
max_k = min(6, len(rfm))

k = st.slider("Number of Customer Segments", 2, max_k, min(4, max_k))

rfm["Cluster"] = run_kmeans(rfm, k)

st.subheader("🎯 Customer Segment Distribution")
st.bar_chart(rfm["Cluster"].value_counts())

# ---------------------------------------------------------
# BUSINESS SEGMENT LABELS
# ---------------------------------------------------------
def segment_label(row):
    if row["Recency"] < 30 and row["Monetary"] > rfm["Monetary"].quantile(0.75):
        return "VIP Customers 💎"
    elif row["Recency"] > 90:
        return "At Risk Customers ⚠️"
    elif row["Frequency"] > rfm["Frequency"].quantile(0.75):
        return "Loyal Customers ❤️"
    return "Regular Customers 🆕"

rfm["Segment"] = rfm.apply(segment_label, axis=1)

st.subheader("📌 Segment Summary Table")
segment_table = rfm["Segment"].value_counts().reset_index()
segment_table.columns = ["Segment", "Customer Count"]
st.dataframe(segment_table)

# Pie Chart
st.subheader("📊 Segment Split")

fig1 = plt.figure()
plt.pie(
    segment_table["Customer Count"],
    labels=segment_table["Segment"],
    autopct="%1.1f%%"
)
plt.title("Customer Segments")
st.pyplot(fig1)

# ---------------------------------------------------------
# ✅ CACHED CHURN MODEL TRAINING (FAST)
# ---------------------------------------------------------
@st.cache_resource
def train_churn_model(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

rfm["Churn"] = (rfm["Recency"] > 90).astype(int)

X = rfm[["Recency", "Frequency", "Monetary"]]
y = rfm["Churn"]

model = train_churn_model(X, y)

rfm["Churn_Risk"] = model.predict_proba(X)[:, 1]

# Histogram
st.subheader("📉 Churn Risk Distribution")

fig2 = plt.figure()
plt.hist(rfm["Churn_Risk"], bins=10)
plt.xlabel("Churn Risk Score")
plt.ylabel("Customers")
plt.title("Churn Risk Histogram")
st.pyplot(fig2)

# Top risky customers
st.subheader("⚠️ Top 10 Customers Likely to Churn")
st.dataframe(rfm.sort_values("Churn_Risk", ascending=False).head(10))

# ---------------------------------------------------------
# KPI SUMMARY
# ---------------------------------------------------------
st.subheader("📌 Executive Dashboard KPIs")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", len(rfm))
col2.metric("VIP Customers", (rfm["Segment"] == "VIP Customers 💎").sum())
col3.metric("At Risk Customers", (rfm["Segment"] == "At Risk Customers ⚠️").sum())
col4.metric("Avg Churn Risk", round(rfm["Churn_Risk"].mean(), 2))

# Scatter plot
st.subheader("📍 Cluster Visualization")

fig3 = plt.figure()
plt.scatter(rfm["Frequency"], rfm["Monetary"], c=rfm["Cluster"])
plt.xlabel("Frequency")
plt.ylabel("Monetary Value")
plt.title("Customer Clusters")
st.pyplot(fig3)

# ---------------------------------------------------------
# DOWNLOAD REPORT
# ---------------------------------------------------------
st.subheader("⬇️ Download Full Customer Report")

csv = rfm.to_csv().encode("utf-8")

st.download_button(
    "Download Segmentation Report",
    csv,
    "customer_segmentation_report.csv",
    "text/csv"
)

st.success("✅ Dashboard Generated Successfully (Fast & Optimized)!")
