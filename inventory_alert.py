
import pandas as pd
import numpy as np
import joblib

# Load the trained machine learning model
model = joblib.load("retail_demand_model.pkl")

# Load the processed sales dataset
data = pd.read_csv("data/processed_sales.csv")
data["date"] = pd.to_datetime(data["date"])

print("SMART INVENTORY ALERT SYSTEM")
print("--------------------------------")
print("Trained model loaded successfully!")
print("Sales dataset loaded successfully!")
print("\nDataset shape:", data.shape)

# Get the latest sales record for each store and item
latest_data = data.sort_values("date").groupby(
    ["store", "item"]
).tail(1).copy()

# Predict demand for the next day
latest_data["date"] = latest_data["date"] + pd.Timedelta(days=1)

latest_data["year"] = latest_data["date"].dt.year
latest_data["month"] = latest_data["date"].dt.month
latest_data["day"] = latest_data["date"].dt.day
latest_data["day_of_week"] = latest_data["date"].dt.day_of_week

# Prepare features
features = [
    "store", "item", "year", "month", "day", "day_of_week",
    "lag_1", "lag_7", "lag_28",
    "rolling_mean_7", "rolling_mean_28"
]

X_latest = latest_data[features]

# Predict next-day demand
predictions = model.predict(X_latest)

latest_data["predicted_demand"] = np.maximum(
    0, predictions
).round().astype(int)

print("\nNEXT-DAY DEMAND PREDICTIONS:")
print(
    latest_data[
        ["store", "item", "predicted_demand"]
    ].head(10)
)

# Generate sample inventory quantities
np.random.seed(42)

latest_data["current_stock"] = np.random.randint(
    20, 120, size=len(latest_data)
)

print("\nCURRENT INVENTORY:")
print(
    latest_data[
        ["store", "item", "predicted_demand", "current_stock"]
    ].head(10)
)

# Inventory planning parameters
lead_time = 3

# Calculate reorder quantity
latest_data["reorder_quantity"] = np.maximum(
    0,
    latest_data["predicted_demand"] - latest_data["current_stock"]
)

# Calculate safety stock (simplified demonstration assumption)
latest_data["safety_stock"] = latest_data["predicted_demand"]

# Calculate reorder point
latest_data["reorder_point"] = (
    latest_data["predicted_demand"] * lead_time
    + latest_data["safety_stock"]
)

# Calculate recommended order quantity
latest_data["recommended_order"] = np.maximum(
    0,
    latest_data["reorder_point"] - latest_data["current_stock"]
)

# Generate inventory alerts
def updated_inventory_status(row):
    if row["current_stock"] <= row["predicted_demand"]:
        return "URGENT: REORDER NOW"
    elif row["current_stock"] <= row["predicted_demand"] * lead_time:
        return "LOW STOCK"
    elif row["current_stock"] <= row["reorder_point"]:
        return "REORDER NOW"
    else:
        return "STOCK AVAILABLE"

# Create inventory status column
latest_data["inventory_status"] = latest_data.apply(
    updated_inventory_status, axis=1
)

# Print reorder recommendations
print("\nREORDER RECOMMENDATIONS:")
print(
    latest_data[
        [
            "store", "item", "predicted_demand",
            "current_stock", "inventory_status", "reorder_quantity"
        ]
    ].head(15)
)

# Print smart reorder plan
print("\nSMART REORDER PLAN:")
print(
    latest_data[
        [
            "store", "item", "predicted_demand",
            "current_stock", "inventory_status",
            "safety_stock", "reorder_point", "recommended_order"
        ]
    ].head(15)
)
