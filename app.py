import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(page_title="Enterprise Customer Dashboard", layout="wide")

st.title("🚀 Enterprise Customer Segmentation & Churn Dashboard")
st.write("Customer analytics system with segmentation, churn prediction, and business insights.")

# ---------------- Sidebar ----------------
st.sidebar.title("Data Input")

mode = st.sidebar.radio(
    "Choose dataset mode:",
    ["Use Demo Dataset", "Upload My Dataset"]
)

# ---------------- Load Dataset ----------------
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

# ---------------- Preview Dataset ----------------
st.subheader("📌 Dataset Preview (First 20 rows)")
st.dataframe(df.head(20))

# ---------------- Column Fixing ----------------
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

# ---------------- RFM Feature Engineering ----------------
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "CustomerID": "count",
    "Amount": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

st.subheader("📊 RFM Customer Features")
st.dataframe(rfm.head(15))

# ---------------- Clustering ----------------
st.subheader("🎯 Customer Segmentation")

if len(rfm) < 2:
    st.error("Not enough customers for segmentation.")
    st.stop()

max_k = min(6, len(rfm))

k = st.slider(
    "Number of Segments",
    2,
    max_k,
    min(4, max_k)
)

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

kmeans = KMeans(n_clusters=k, random_state=42)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)


# Segment Labels
def segment_label(row):
    if row["Recency"] < 30 and row["Monetary"] > rfm["Monetary"].quantile(0.75):
        return "VIP Customers 💎"
    elif row["Recency"] > 90:
        return "At Risk Customers ⚠️"
    elif row["Frequency"] > rfm["Frequency"].quantile(0.75):
        return "Loyal Customers ❤️"
    return "Regular Customers 🆕"

rfm["Segment"] = rfm.apply(segment_label, axis=1)

# ---------------- Segment Summary Table ----------------
st.subheader("📌 Segment Summary Table")

segment_table = rfm["Segment"].value_counts().reset_index()
segment_table.columns = ["Segment Name", "Customer Count"]

st.dataframe(segment_table)

# ---------------- Segment Pie Chart ----------------
st.subheader("📊 Segment Distribution (Pie Chart)")

fig1 = plt.figure()
plt.pie(
    segment_table["Customer Count"],
    labels=segment_table["Segment Name"],
    autopct="%1.1f%%"
)
plt.title("Customer Segments")

st.pyplot(fig1)

# ---------------- Churn Risk Prediction ----------------
st.subheader("⚠️ Churn Risk Prediction")

rfm["Churn"] = (rfm["Recency"] > 90).astype(int)

X = rfm[["Recency", "Frequency", "Monetary"]]
y = rfm["Churn"]

model = RandomForestClassifier()
model.fit(X, y)

rfm["Churn_Risk"] = model.predict_proba(X)[:, 1]

# Histogram of churn risk
st.subheader("📉 Churn Risk Score Distribution")

fig2 = plt.figure()
plt.hist(rfm["Churn_Risk"], bins=10)
plt.xlabel("Churn Risk Score")
plt.ylabel("Number of Customers")
plt.title("Churn Risk Distribution")

st.pyplot(fig2)

# ---------------- VIP + At-Risk Tables ----------------
st.subheader("💎 Top VIP Customers")
st.dataframe(
    rfm[rfm["Segment"] == "VIP Customers 💎"]
    .sort_values("Monetary", ascending=False)
    .head(10)
)

st.subheader("⚠️ Customers Most Likely to Churn")
st.dataframe(
    rfm.sort_values("Churn_Risk", ascending=False).head(10)
)

# ---------------- Cluster Scatter Plot ----------------
st.subheader("📍 Cluster Visualization")

fig3 = plt.figure()
plt.scatter(rfm["Frequency"], rfm["Monetary"], c=rfm["Cluster"])
plt.xlabel("Frequency")
plt.ylabel("Monetary Value")
plt.title("Customer Clusters")

st.pyplot(fig3)

# ---------------- Download Report ----------------
st.subheader("⬇️ Download Full Report")

csv = rfm.to_csv().encode("utf-8")

st.download_button(
    "Download Customer Segmentation Report",
    csv,
    file_name="customer_segmentation_report.csv",
    mime="text/csv"
)

st.success("✅ Dashboard Generated Successfully!")
