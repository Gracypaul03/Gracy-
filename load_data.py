
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data/train.csv")

# Convert date column
data["date"] = pd.to_datetime(data["date"])

# Calculate monthly total sales
monthly_sales = data.groupby(
    pd.Grouper(key="date", freq="MS")
)["sales"].sum()

# Display monthly sales
print("MONTHLY SALES:")
print(monthly_sales.head(12))

# Create sales trend graph
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index, monthly_sales.values)

plt.title("Monthly Retail Sales Trend (2013-2017)")
plt.xlabel("Month")
plt.ylabel("Total Units Sold")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Save graph
plt.savefig("monthly_sales_trend.png")

# Display graph
plt.show()
