import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Question 3
## 1. Load dataset of airbnb rental properties for NYC
airbnb = pd.read_csv('./data/airbnb_hw.csv')
airbnb['Price'] = airbnb['Price'].replace('[\$,]', '', regex=True).astype(float)

## 2. Dimensions of the data, number of observations, and variables
print("Shape:", airbnb.shape)
print("Columns:", airbnb.columns.tolist())
print(airbnb.head())

## 3. Cross tabulate Room Type and Property Type
airbnb.columns = airbnb.columns.str.strip()
room_property = pd.crosstab(airbnb['Room Type'], airbnb['Property Type'])
print(room_property)

# For property type, Apartment dominates the dataset and are available in entire home, private, room, shared room.
# Across different property type, shared room has the lowest number. For bed and breakfast, private rooms are more common


# 4.
airbnb['Price'] = airbnb['Price'].replace('[\$,]', '', regex=True).astype(float)
airbnb_filtered = airbnb[(airbnb['Price'] >= 10) & (airbnb['Price'] <= 1000)].copy()

# Histogram
plt.figure(figsize=(10, 4))
sns.histplot(airbnb_filtered['Price'], bins=50, kde=False)
plt.title("Histogram of Airbnb Price (Filtered: $10–$1000)")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# KDE Plot
plt.figure(figsize=(10, 4))
sns.kdeplot(airbnb_filtered['Price'], fill=True)
plt.title("Kernel Density Estimate of Price")
plt.xlabel("Price")
plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()
plt.show()

# Boxplot
plt.figure(figsize=(10, 1.5))
sns.boxplot(x=airbnb_filtered['Price'])
plt.title("Boxplot of Airbnb Price (Filtered)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Statistical Description
price_description = airbnb_filtered['Price'].describe()
print("Descriptive statistics for Airbnb Price (Filtered):")
print(price_description)

airbnb_filtered['price_log'] = np.log1p(airbnb_filtered['Price'])

# Price_log
# Histogram of log(price)
plt.figure(figsize=(10, 4))
sns.histplot(airbnb_filtered['price_log'], bins=50, kde=False)
plt.title("Histogram of Log(Price)")
plt.xlabel("Log(Price)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# KDE of log(price)
plt.figure(figsize=(10, 4))
sns.kdeplot(airbnb_filtered['price_log'], fill=True)
plt.title("Kernel Density Estimate of Log(Price)")
plt.xlabel("Log(Price)")
plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()
plt.show()

# Boxplot of log(price)
plt.figure(figsize=(10, 1.5))
sns.boxplot(x=airbnb_filtered['price_log'])
plt.title("Boxplot of Log(Price)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Statistical description for log(price)
log_price_description = airbnb_filtered['price_log'].describe()
print("Descriptive statistics for Log(Price):")
print(log_price_description)

plt.figure(figsize=(8, 5))
sns.scatterplot(data=airbnb_filtered, x='Beds', y='price_log')
plt.title("Scatterplot of Log(Price) vs Beds")
plt.xlabel("Beds")
plt.ylabel("Log(Price)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Groupby Beds and describe Price
bed_price_summary = airbnb_filtered.groupby('Beds')['Price'].agg(['mean', 'std', 'count']).reset_index()
print("Price summary grouped by Beds:")
print(bed_price_summary)

# The plot shows a positive relationship between the number of beds and Airbnb price, but the returns show diminishing.
# The price does increase with more beds, but not linear relationship. The pattern of the average price increases with
# the bed count. The standard deviation of the price also increases with bed counts.

## 6
# Scatterplot colored by Room Type and Property Type
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=airbnb_filtered,
    x='Beds',
    y='price_log',
    hue='Room Type',
    style='Property Type'
)
plt.title("Log(Price) vs Beds Colored by Room & Property Type")
plt.xlabel("Beds")
plt.ylabel("Log(Price)")
plt.tight_layout()
plt.show()

# Grouped descriptive stats
grouped_stats = airbnb_filtered.groupby(['Room Type', 'Property Type'])['Price'].agg(['mean', 'median', 'std', 'count']).reset_index()
grouped_stats_sorted = grouped_stats.sort_values(by='mean', ascending=False)

# Display top of the sorted result
grouped_stats_sorted.head(10)

# The pattern of the scatterplot includes color indicating room time and marker style indicating property type.
# Entire home and apartments are the blue dots and are the one that has the most span in the scatterplot.
# They also have the higher log price espcially as bed count increases. Private rooms are concentrated in lower bed counts and
# lower price ranges. Shared rooms are rare across the graph and low priced. Property that has more bed have higher prices but
# also higher variability. Because the prices are skew and presents variability, median would be a better indicator than mean.

## 7
sns.jointplot(
    data=airbnb_filtered,
    x='Beds',
    y='price_log',
    kind='hex',
    height=8,
    color='steelblue'
)

plt.suptitle("Hexbin Plot of Beds vs Log(Price)", y=1.02)
plt.show()

# Based on the graph, most listings are concentrated where bedroom count are 1-2 and log orice 4.5-5.5.
# It appears to be evenly spread across a range of bed counts and prices, and higher bed listings are visually
# overrepresented in scatterplots.

