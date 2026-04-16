from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
)

ARTIFACT_PATH = Path("model") / "customer_segmentation_artifacts.joblib"
MAX_INVOICE_DATE = pd.Timestamp("2010-12-09 20:01:00")

caps = {}
model_log_features = []
CLUSTER_PROFILES = {
    0: {
        "name": "High-Value Loyal Power Customers",
        "profile": (
            "**Profile:** Highest spenders with the longest tenure, most invoices, and the most recent activity.\n"
            "**Suggested action:** VIP retention, exclusive benefits, and proactive churn prevention."
        ),
    },
    1: {
        "name": "High-Price One-Offs (At-Risk)",
        "profile": (
            "**Profile:** Low overall spend and frequency, but the highest item price, with long inactivity.\n"
            "**Suggested action:** Win-back offers and second-purchase incentives."
        ),
    },
    2: {
        "name": "Frequent Emerging Regulars",
        "profile": (
            "**Profile:** Moderately valued customers who buy relatively often but are still newer.\n"
            "**Suggested action:** Nurture toward loyalty with bundles and personalized upsell."
        ),
    },
    3: {
        "name": "One-Time Bulk Buyers",
        "profile": (
            "**Profile:** Single-transaction customers with large baskets, but no repeat behavior.\n"
            "**Suggested action:** Post-purchase follow-up to convert to repeat."
        ),
    },
    4: {
        "name": "Long-Tenure Low-Frequency Customers",
        "profile": (
            "**Profile:** Moderate spenders with long tenure but the lowest purchase frequency.\n"
            "**Suggested action:** Reactivation nudges and cadence reminders."
        ),
    },
}

def quantile_cap(X):
    X_df = X.copy()
    for c, upper in caps.items():
        X_df[c] = X_df[c].clip(upper=upper)
    return X_df

def apply_log_transform(X):
    X_df = X.copy()
    for c in model_log_features:
        X_df[c] = np.log1p(np.clip(X_df[c], a_min=0, a_max=None))
    return X_df

@st.cache_resource
def load_artifacts():
    return joblib.load(ARTIFACT_PATH)

artifacts = load_artifacts()
model_pipeline = artifacts["model_pipeline"]
selected_features = artifacts["selected_features"]
caps = artifacts["caps"]
model_log_features = artifacts["model_log_features"]
data_max_invoice_date = MAX_INVOICE_DATE

def predict_segment(customer_row: dict) -> int:
    X = pd.DataFrame([customer_row], columns=selected_features)
    return int(model_pipeline.predict(X)[0])

def build_model_features(original_row: dict) -> dict:
    total_spend = float(original_row["TotalSpend"])
    total_quantity = float(original_row["TotalQuantity"])
    invoice_count = int(original_row["InvoiceCount"])
    unique_products = int(original_row["UniqueProducts"])
    recency_days = int(original_row["RecencyDays"])
    tenure_days = int(original_row["TenureDays"])

    active_months = max(1, math.ceil((tenure_days + 1) / 30))
    return {
        "TotalSpend": total_spend,
        "InvoiceCount": invoice_count,
        "UniqueProducts": unique_products,
        "AverageItemPrice": float(original_row["AverageItemPrice"]),
        "RecencyDays": recency_days,
        "TenureDays": tenure_days,
        "AvgBasketValue": total_spend / invoice_count,
        "SpendPerProduct": total_spend / unique_products,
        "QuantityPerInvoice": total_quantity / invoice_count,
        "InvoicePerMonth": invoice_count / active_months,
        "IsOneTransactionCustomer": int(invoice_count == 1),
    }

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 1.2rem;}
      .stButton > button {width: 100%; font-weight: 600; border-radius: 10px;}
      div[data-testid="stMetric"] {
          background: #0f172a10;
          border: 1px solid #64748b33;
          border-radius: 12px;
          padding: 10px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛍️ Customer Segmentation")
st.caption("This page is for transactions of **one customer**.")
st.caption("Supported transaction date range: **2009-12-01 to 2010-12-09**.")


default_transaction = {
    "Invoice": "",
    "StockCode": "",
    "Quantity": None,
    "InvoiceDate": data_max_invoice_date.date(),
    "Price": None,
}

if "transactions" not in st.session_state:
    st.session_state.transactions = [default_transaction.copy()]

delete_index = None
for idx, txn in enumerate(st.session_state.transactions):
    with st.container(border=True):
        head_col, delete_col = st.columns([6, 2])
        with head_col:
            st.markdown(f"**Transaction {idx + 1}**")
        with delete_col:
            if st.button("Delete Transaction", key=f"delete_txn_{idx}"):
                delete_index = idx

        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            invoice = st.text_input(
                "Invoice",
                value=str(txn["Invoice"]),
                placeholder="enter the value",
                key=f"invoice_{idx}",
            )
            quantity_value = None if txn["Quantity"] is None else float(txn["Quantity"])
            quantity = st.number_input(
                "Quantity",
                min_value=1.0,
                value=quantity_value,
                step=1.0,
                placeholder="enter the value",
                key=f"quantity_{idx}",
            )
        with row1_col2:
            stock_code = st.text_input(
                "StockCode",
                value=str(txn["StockCode"]),
                placeholder="enter the value",
                key=f"stock_{idx}",
            )
            price_value = None if txn["Price"] is None else float(txn["Price"])
            price = st.number_input(
                "Price",
                min_value=0.01,
                value=price_value,
                step=0.01,
                format="%.2f",
                placeholder="enter the value",
                key=f"price_{idx}",
            )

        invoice_date = st.date_input(
            "InvoiceDate",
            value=pd.Timestamp(txn["InvoiceDate"]).date(),
            format="YYYY-MM-DD",
            key=f"date_{idx}",
        )

        st.session_state.transactions[idx] = {
            "Invoice": invoice,
            "StockCode": stock_code,
            "Quantity": None if quantity is None else float(quantity),
            "InvoiceDate": invoice_date,
            "Price": None if price is None else float(price),
        }

if delete_index is not None:
    if len(st.session_state.transactions) == 1:
        st.error("You cannot delete all transactions. At least one transaction is required.")
    else:
        st.session_state.transactions.pop(delete_index)
        st.rerun()

if st.button("Add New Transaction"):
    st.session_state.transactions.append(default_transaction.copy())
    st.rerun()

if st.button("Predict"):
    fe_df = pd.DataFrame(st.session_state.transactions).copy()
    fe_df["Invoice"] = fe_df["Invoice"].astype(str).str.strip()
    fe_df["StockCode"] = fe_df["StockCode"].astype(str).str.strip()

    if (fe_df["Invoice"] == "").any() or (fe_df["StockCode"] == "").any():
        st.error("Please enter Invoice and StockCode for all transactions.")
        st.stop()
    if fe_df["Quantity"].isna().any() or fe_df["Price"].isna().any():
        st.error("Please enter Quantity and Price for all transactions.")
        st.stop()
    if (fe_df["Quantity"] <= 0).any() or (fe_df["Price"] <= 0).any():
        st.error("Quantity and Price must both be greater than 0.")
        st.stop()

    fe_df["InvoiceDate"] = pd.to_datetime(fe_df["InvoiceDate"], errors="coerce")
    if fe_df["InvoiceDate"].isna().any():
        st.error("There is an invalid InvoiceDate value.")
        st.stop()
    fe_df["line_amount"] = fe_df["Quantity"] * fe_df["Price"]
    customer_row = {
        "TotalSpend": float(fe_df["line_amount"].sum()),
        "TotalQuantity": int(fe_df["Quantity"].sum()),
        "InvoiceCount": int(fe_df["Invoice"].nunique()),
        "UniqueProducts": int(fe_df["StockCode"].nunique()),
        "AverageItemPrice": float(fe_df["Price"].mean()),
        "FirstPurchase": pd.Timestamp(fe_df["InvoiceDate"].min()),
        "LastPurchase": pd.Timestamp(fe_df["InvoiceDate"].max()),
    }

    recency_days = int((data_max_invoice_date - customer_row["LastPurchase"]).days)
    tenure_days = int((customer_row["LastPurchase"] - customer_row["FirstPurchase"]).days)

    original_row = {
        "TotalSpend": customer_row["TotalSpend"],
        "TotalQuantity": customer_row["TotalQuantity"],
        "InvoiceCount": customer_row["InvoiceCount"],
        "UniqueProducts": customer_row["UniqueProducts"],
        "AverageItemPrice": customer_row["AverageItemPrice"],
        "RecencyDays": recency_days,
        "TenureDays": tenure_days,
    }

    model_row = build_model_features(original_row)
    if not np.isfinite(np.array(list(model_row.values()), dtype=float)).all():
        st.error("Computed features contain invalid values. Please check your inputs.")
        st.stop()

    cluster = predict_segment(model_row)
    st.success(f"The customer belongs to **Cluster {cluster}**")
    cluster_info = CLUSTER_PROFILES.get(cluster)
    if cluster_info is not None:
        st.info(f"**Cluster {cluster} — {cluster_info['name']}**\n\n{cluster_info['profile']}")
