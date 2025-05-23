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

# Question 4
import pandas as pd

drill = pd.read_csv('./data/drilling_rigs.csv')

## 1.
drill['time'] = pd.to_datetime(drill['Month'], format='%Y %B', errors='coerce')

# STEP 2: Identify and convert numeric columns with potential 'Not Available' strings
cols_to_convert = drill.columns[1:-1].tolist() + [drill.columns[-1]]  # all columns except 'Month' and 'time'

# STEP 3: Convert all identified columns to numeric, forcing 'Not Available' to NaN
for col in cols_to_convert:
    drill[col] = pd.to_numeric(drill[col], errors='coerce')

# STEP 4: Summary of cleaning results
print("✅ Data cleaned successfully!")
print("Shape of cleaned dataset:", drill.shape)
print("Data types:\n", drill.dtypes)
print("\nMissing values per column:\n", drill.isna().sum())
print("\nFirst few rows:\n", drill.head())

# There are 623 observations with 11 variables. Some variables are correctly interpreted, some are not. Some cells contain string
# that shows "Not Available" which prevents pandas from reading correctly. By coercing invalid numeric values to NaN, we are
# able to clean the data.

## 2.
drill['time'] = pd.to_datetime(drill['Month'], format='%Y %B')

# Convert the rig count column to numeric
col = 'Active Well Service Rig Count (Number of Rigs)'
drill[col] = pd.to_numeric(drill[col], errors='coerce')

# Create line plot
plt.figure(figsize=(12, 6))
plt.plot(drill['time'], drill[col], color='tab:blue')
plt.title('Active Well Service Rig Count Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Rigs')
plt.grid(True)
plt.tight_layout()
plt.show()

# The line plot shows a gradually decreasing trend with a spike of number of rigs between 1980 and 1990. After 2020, it
# reaches the lowest.

## 3.
drill['time'] = pd.to_datetime(drill['Month'], format='%Y %B')

# Convert rig count column to numeric
col = 'Active Well Service Rig Count (Number of Rigs)'
drill[col] = pd.to_numeric(drill[col], errors='coerce')

# Compute first difference (month-to-month change)
drill['Rig Change'] = drill[col].diff()

# Plot the first difference
plt.figure(figsize=(12, 6))
plt.plot(drill['time'], drill['Rig Change'], color='darkorange')
plt.title('Monthly Change in Active Well Service Rig Count')
plt.xlabel('Date')
plt.ylabel('Change in Number of Rigs')
plt.grid(True)
plt.tight_layout()
plt.show()

# There are some notable decrease in the line graph. Between 1980-1990, the change in number of rigs
# decreases to more than -1000, and between 2000-2010, the change in number of rigs also almost reach to -1000.
# There are 2 steep increase in 1980-1990 that the change in number of rigs exceeds 500.

## 4.
onshore_col = drill.columns[1]
offshore_col = drill.columns[2]

# Convert these columns to numeric
drill[onshore_col] = pd.to_numeric(drill[onshore_col], errors='coerce')
drill[offshore_col] = pd.to_numeric(drill[offshore_col], errors='coerce')

# Melt the dataset
melted = drill.melt(
    id_vars='time',
    value_vars=[onshore_col, offshore_col],
    var_name='Rig Type',
    value_name='Count'
)

# Plot the melted data
plt.figure(figsize=(12, 6))
sns.lineplot(data=melted, x='time', y='Count', hue='Rig Type')
plt.title('Onshore vs Offshore Rig Counts Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Rigs')
plt.grid(True)
plt.tight_layout()
plt.show()