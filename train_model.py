import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the processed dataset
data = pd.read_csv("data/processed_sales.csv")

# Convert date column
data["date"] = pd.to_datetime(data["date"])

# Select input features
features = [
    "store",
    "item",
    "year",
    "month",
    "day",
    "day_of_week",
    "lag_1",
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28"
]

# Select target column
X = data[features]
y = data["sales"]

# Split data based on date
split_date = "2017-10-01"

train_mask = data["date"] < split_date
validation_mask = data["date"] >= split_date

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[validation_mask]
y_val = y[validation_mask]

print("TRAINING ROWS:", len(X_train))
print("VALIDATION ROWS:", len(X_val))

# Create the machine learning model
model = HistGradientBoostingRegressor(
    max_iter=50,
    learning_rate=0.1,
    max_leaf_nodes=15,
    random_state=42
)

# Train the model
print("\nTraining the model...")
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "retail_demand_model.pkl")

print("Trained model saved successfully!")

# Make predictions
predictions = model.predict(X_val)

# Evaluate the model
mae = mean_absolute_error(y_val, predictions)
rmse = np.sqrt(mean_squared_error(y_val, predictions))

print("\nMODEL PERFORMANCE:")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))

# Compare with a simple baseline
baseline_predictions = X_val["lag_7"]

baseline_mae = mean_absolute_error(y_val, baseline_predictions)

print("\nBASELINE PERFORMANCE:")
print("Baseline MAE:", round(baseline_mae, 2))
