
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📊",
    layout="wide"
)

# Load model and dataset
@st.cache_resource
def load_model():
    return joblib.load("retail_demand_model.pkl")

@st.cache_data
def load_data():
    data = pd.read_csv("data/processed_sales.csv")
    data["date"] = pd.to_datetime(data["date"])
    return data

model = load_model()
data = load_data()

# Dashboard title
st.title("📊 Retail Demand Forecasting")
st.subheader("Smart Inventory Alert System")

st.write(
    "Predict next-day product demand and manage "
    "retail inventory using Machine Learning."
)

# Prepare latest records
latest_data = data.sort_values("date").groupby(
    ["store", "item"]
).tail(1).copy()

# Update date
latest_data["date"] = latest_data["date"] + pd.Timedelta(days=1)

latest_data["year"] = latest_data["date"].dt.year
latest_data["month"] = latest_data["date"].dt.month
latest_data["day"] = latest_data["date"].dt.day
latest_data["day_of_week"] = latest_data["date"].dt.day_of_week

# Model features
features = [
    "store", "item", "year", "month", "day",
    "day_of_week", "lag_1", "lag_7", "lag_28",
    "rolling_mean_7", "rolling_mean_28"
]

# Predict demand
latest_data["predicted_demand"] = np.maximum(
    0, model.predict(latest_data[features])
).round().astype(int)

# Simulated inventory
np.random.seed(42)

latest_data["current_stock"] = np.random.randint(
    20, 120, size=len(latest_data)
)

# Inventory calculations
lead_time = 3

latest_data["safety_stock"] = latest_data["predicted_demand"]

latest_data["reorder_point"] = (
    latest_data["predicted_demand"] * lead_time
    + latest_data["safety_stock"]
)

latest_data["recommended_order"] = np.maximum(
    0,
    latest_data["reorder_point"] - latest_data["current_stock"]
)

latest_data["reorder_quantity"] = np.maximum(
    0,
    latest_data["predicted_demand"] - latest_data["current_stock"]
)

# Inventory status
def inventory_status(row):
    if row["current_stock"] <= row["predicted_demand"]:
        return "URGENT: REORDER NOW"
    elif row["current_stock"] <= row["predicted_demand"] * lead_time:
        return "LOW STOCK"
    elif row["current_stock"] <= row["reorder_point"]:
        return "REORDER NOW"
    else:
        return "STOCK AVAILABLE"

latest_data["inventory_status"] = latest_data.apply(
    inventory_status, axis=1
)

# Dashboard metrics
st.divider()
st.header("📈 Inventory Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Store-Item Pairs", len(latest_data))
col2.metric("Average Predicted Demand", round(latest_data["predicted_demand"].mean(), 1))
col3.metric("Urgent Reorders", (latest_data["inventory_status"] == "URGENT: REORDER NOW").sum())
col4.metric("Low Stock Items", (latest_data["inventory_status"] == "LOW STOCK").sum())

# Inventory table
st.divider()
st.header("📦 Smart Inventory Recommendations")

st.dataframe(
    latest_data[
        [
            "store",
            "item",
            "predicted_demand",
            "current_stock",
            "inventory_status",
            "recommended_order"
        ]
    ],
    use_container_width=True
)

# Sales trend
st.divider()
st.header("📊 Monthly Sales Trend")

monthly_sales = data.groupby(
    data["date"].dt.to_period("M")
)["sales"].sum()

monthly_sales.index = monthly_sales.index.astype(str)

st.line_chart(monthly_sales)

# Alert section
st.divider()
st.header("⚠️ Urgent Inventory Alerts")

urgent_items = latest_data[
    latest_data["inventory_status"] == "URGENT: REORDER NOW"
]

if len(urgent_items) > 0:
    st.warning(f"{len(urgent_items)} products need urgent reordering.")
    st.dataframe(
        urgent_items[
            ["store", "item", "predicted_demand",
             "current_stock", "recommended_order"]
        ],
        use_container_width=True
    )
else:
    st.success("No urgent inventory alerts.")

st.caption(
    "Note: Current inventory is simulated for demonstration."
)
