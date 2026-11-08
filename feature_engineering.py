
import pandas as pd

# Load dataset
data = pd.read_csv("data/train.csv")

# Convert date column
data["date"] = pd.to_datetime(data["date"])

# Sort data before creating lag features
data = data.sort_values(["store", "item", "date"])

# Date-based features
data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.month
data["day"] = data["date"].dt.day
data["day_of_week"] = data["date"].dt.day_of_week

# Lag features
data["lag_1"] = data.groupby(["store", "item"])["sales"].shift(1)
data["lag_7"] = data.groupby(["store", "item"])["sales"].shift(7)
data["lag_28"] = data.groupby(["store", "item"])["sales"].shift(28)


# Rolling average features
data["rolling_mean_7"] = (
    data.groupby(["store", "item"])["sales"]
    .transform(lambda x: x.shift(1).rolling(window=7).mean())
)

data["rolling_mean_28"] = (
    data.groupby(["store", "item"])["sales"]
    .transform(lambda x: x.shift(1).rolling(window=28).mean())
)


# Remove rows with missing feature values
data = data.dropna()

# Reset row index
data = data.reset_index(drop=True)

# Save processed dataset
data.to_csv("data/processed_sales.csv", index=False)

print("\nProcessed dataset saved successfully!")

# Check the cleaned dataset
print("\nCLEANED DATASET:")
print(data.head())

print("\nCLEANED DATASET SHAPE:")
print(data.shape)

print("\nMISSING VALUES AFTER CLEANING:")
print(data.isnull().sum())


# Display results
print("DATASET WITH LAG FEATURES:")
print(data.head(35))

print("\nUPDATED DATASET SHAPE:")
print(data.shape)

print("\nLAG FEATURE COLUMNS:")
print(data[["lag_1", "lag_7", "lag_28"]].head(35))

