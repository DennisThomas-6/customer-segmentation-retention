import pandas as pd

def build_rfm(transactions):
    rfm = transactions.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (transactions["InvoiceDate"].max() - x.max()).days,
        "InvoiceNo": "count",
        "Amount": "sum"
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]

    return rfm
