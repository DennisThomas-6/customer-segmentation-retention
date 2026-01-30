
# Customer Segmentation & Retention Analysis

## Objective
The goal of this project is to segment customers based on purchasing behavior and analyze retention patterns to improve customer loyalty and reduce churn.

## Dataset
Simulated transactional customer data containing:
- Customer ID
- Order date
- Order amount

## Methodology
- RFM feature engineering (Recency, Frequency, Monetary)
- Customer segmentation using K-Means clustering
- Churn identification based on recency
- Cohort-based retention analysis

## Key Insights
- Identified high-value and churn-prone customer segments
- Observed retention patterns across customer cohorts
- Provided actionable business recommendations

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Matplotlib
## Interactive Dashboard (Streamlit)

The project includes an interactive Streamlit dashboard for:
- Customer Segmentation (RFM)
- Cohort-based Retention Analysis

### Dashboard Preview
![Dashboard](dashboard_home.png)
![RFM](rfm_view.png)
![Retention](retention_view.png)

To run locally:
```bash
pip install streamlit pandas matplotlib seaborn
streamlit run app.py
