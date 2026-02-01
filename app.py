import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(page_title="Customer Analytics Dashboard", layout="wide")

st.title("🚀 Enterprise Customer Segmentation & Churn Dashboard")
st.write(
    "Interactive analytics dashboard for customer segmentation, churn risk prediction, "
    "and business insights using RFM + Machine Learning."
)


st.sidebar.title("Data Input")
mode = st.sidebar.radio(
    "Choose dataset:",
    ["Use Demo Dataset", "Upload My Dataset"]
)


if mode == "Upload My Dataset":

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset uploaded successfully.")
    else:
        st.warning("Please upload a dataset to continue.")
        st.stop()

else:
    df = pd.read_csv("transactions_final.csv")
    st.info("✅ Demo dataset loaded automatically for portfolio showcase.")


st.subheader("Dataset Preview")
st.dataframe(df.head())


df.columns = [c.strip().lower() for c in df.columns]
df = df.loc[:, ~df.columns.str.contains("unnamed")]

if "customer_id" in df.columns:
    df.rename(columns={"customer_id": "CustomerID"}, inplace=True)

if "order_date" in df.columns:
    df.rename(columns={"order_date": "InvoiceDate"}, inplace=True)

if "order_amount" in df.columns:
    df.rename(columns={"order_amount": "Amount"}, inplace=True)

if not all(col in df.columns for col in ["CustomerID", "InvoiceDate", "Amount"]):
    st.error("Dataset must contain customer, transaction date, and amount columns.")
    st.stop()


df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "CustomerID": "count",
    "Amount": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

st.subheader("RFM Customer Features")
st.dataframe(rfm.head())


k = st.slider("Number of Segments", 2, 6, 4)

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

kmeans = KMeans(n_clusters=k, random_state=42)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

st.subheader("Customer Segment Distribution")
st.bar_chart(rfm["Cluster"].value_counts())


def segment_label(row):
    if row["Recency"] < 30 and row["Monetary"] > rfm["Monetary"].quantile(0.75):
        return "VIP Customers 💎"
    elif row["Recency"] > 90:
        return "At Risk Customers ⚠️"
    elif row["Frequency"] > rfm["Frequency"].quantile(0.75):
        return "Loyal Customers ❤️"
    return "Regular Customers 🆕"


rfm["Segment"] = rfm.apply(segment_label, axis=1)

st.subheader("Business Segment Examples")
st.dataframe(rfm.head())


rfm["Churn"] = (rfm["Recency"] > 90).astype(int)

X = rfm[["Recency", "Frequency", "Monetary"]]
y = rfm["Churn"]

model = RandomForestClassifier()
model.fit(X, y)

rfm["Churn_Risk"] = model.predict_proba(X)[:, 1]

st.subheader("Top High-Risk Customers")
st.dataframe(rfm.sort_values("Churn_Risk", ascending=False).head(10))


st.subheader("Executive Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", len(rfm))
col2.metric("VIP Customers", (rfm["Segment"] == "VIP Customers 💎").sum())
col3.metric("At Risk Customers", (rfm["Segment"] == "At Risk Customers ⚠️").sum())
col4.metric("Avg Churn Risk", round(rfm["Churn_Risk"].mean(), 2))


st.subheader("Cluster Visualization")

fig = plt.figure()
plt.scatter(rfm["Frequency"], rfm["Monetary"], c=rfm["Cluster"])
plt.xlabel("Frequency")
plt.ylabel("Monetary Value")
plt.title("Customer Segmentation Clusters")
st.pyplot(fig)


st.subheader("Download Report")

csv = rfm.to_csv().encode("utf-8")

st.download_button(
    "Download Customer Segmentation Results",
    csv,
    file_name="customer_segmentation_report.csv",
    mime="text/csv"
)

st.success("✅ Dashboard generated successfully!")
