# 🛍️ Customer Segmentation (Online Retail)

An end-to-end customer segmentation project that clusters retail customers based on transaction behavior. It includes exploratory analysis and model training in a notebook, plus a Streamlit app for interactive segmentation.

---

## 📌 Problem Statement

Given transaction-level retail data, group customers into **behavioral segments** using unsupervised learning.  
The goal is to produce actionable clusters (e.g., loyal, at‑risk, one‑off buyers) based on spend, frequency, recency, and tenure patterns.

---

## 🗂️ Project Structure

```
customer-segmentation/
│
├── app.py                                      # Streamlit web application
├── data/
│   └── online_retail_II.xlsx                   # Raw dataset
├── model/
│   └── customer_segmentation_artifacts.joblib  # Serialized model pipeline + metadata
├── notebook/
│   └── online-retail-customers-clustering.ipynb # End-to-end notebook
├── requirements.txt                            # Python dependencies
└── README.md
```

---

## 📊 Dataset

| Property   | Detail |
|-----------|--------|
| File      | `data/online_retail_II.xlsx` |
| Rows (raw)| 525,461 |
| Columns   | 8 |
| Date range| 2009-12-01 07:45:00 → 2010-12-09 20:01:00 |

**Raw features (per transaction):**

| Feature | Type | Description |
|---------|------|-------------|
| `Invoice` | Categorical | Invoice identifier |
| `StockCode` | Categorical | Product code |
| `Description` | Categorical | Product description |
| `Quantity` | Numerical | Units purchased |
| `InvoiceDate` | Datetime | Transaction timestamp |
| `Price` | Numerical | Unit price |
| `Customer ID` | Numerical | Customer identifier |
| `Country` | Categorical | Customer country |

---

## 🔬 Methodology

### 1. Data Preparation
Transactions are cleaned and transformed into customer-level aggregates.

### 2. Feature Engineering (per customer)
The model uses behavioral features such as:

| Feature | Description |
|---------|-------------|
| `TotalSpend` | Total revenue per customer |
| `InvoiceCount` | Number of invoices |
| `UniqueProducts` | Distinct products purchased |
| `AverageItemPrice` | Mean item price |
| `RecencyDays` | Days since last purchase |
| `TenureDays` | Days between first and last purchase |
| `AvgBasketValue` | Spend per invoice |
| `SpendPerProduct` | Spend per unique product |
| `QuantityPerInvoice` | Items per invoice |
| `InvoicePerMonth` | Invoices per active month |
| `IsOneTransactionCustomer` | 1 if only one invoice |

### 3. Clustering Algorithms (K‑Means vs DBSCAN)
Two clustering approaches are explored and compared:

| Algorithm | Goal | Notes |
|----------|------|-------|
| **K‑Means** | Partition customers into *k* stable segments | Works well for compact, separable clusters |
| **DBSCAN** | Discover density-based groups and outliers | Useful for arbitrary shapes + noise detection |

**K‑Means setup**
- Standardized features + log transform on skewed variables
- `k` evaluated with **Silhouette**, **Davies‑Bouldin**, **Calinski‑Harabasz**, and **Elbow (WCSS)**
- Final model trained with **k = 5**

**DBSCAN setup**
- Standardized features
- `eps` and `min_samples` tuned using k‑distance plots
- Noise points labeled as `-1`

### 4. Model Comparison & Selection
Both algorithms are evaluated on the same engineered features.  
**K‑Means is selected** for business usability and interpretability: DBSCAN achieved stronger clustering metrics in this experiment, but produced less actionable segment structure for downstream business use.

### 5. Interpretation
Clusters are profiled in the notebook with business labels and suggested actions.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+

### Installation

```bash
pip install -r requirements.txt
```

### Run the Web App

```bash
python -m streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Re‑train the Model

Open and run all cells in the notebook:

```bash
jupyter notebook "notebook/online-retail-customers-clustering.ipynb"
```

Re-running the notebook regenerates `model/customer_segmentation_artifacts.joblib`.

---

## 🖥️ Web App

The Streamlit app lets you input multiple transactions for a **single customer** and returns the predicted cluster with a short business interpretation.

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| Data Processing | `pandas`, `numpy` |
| Machine Learning | `scikit-learn` |
| Model Serialization | `joblib` |
| Web App | `streamlit` |
| Visualisation (notebook) | `matplotlib`, `seaborn` |

---