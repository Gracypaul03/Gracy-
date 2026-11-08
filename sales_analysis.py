
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("data/train.csv")

# Find total sales for each product
product_sales = data.groupby("item")["sales"].sum()
top_products = product_sales.sort_values(ascending=False).head(10)

print("TOP 10 BEST-SELLING PRODUCTS:")
print(top_products)

# Find total sales for each store
store_sales = data.groupby("store")["sales"].sum()
top_stores = store_sales.sort_values(ascending=False)

print("\nSTORE-WISE TOTAL SALES:")
print(top_stores)

# Graph 1: Top 10 products
plt.figure(figsize=(10, 5))
top_products.sort_values().plot(kind="barh")

plt.title("Top 10 Best-Selling Products")
plt.xlabel("Total Units Sold")
plt.ylabel("Product ID")
plt.tight_layout()
plt.savefig("top_products.png")
plt.show()

# Graph 2: Store-wise sales
plt.figure(figsize=(10, 5))
store_sales.sort_values().plot(kind="barh")

plt.title("Store-Wise Total Sales")
plt.xlabel("Total Units Sold")
plt.ylabel("Store ID")
plt.tight_layout()
plt.savefig("store_sales.png")
plt.show()
