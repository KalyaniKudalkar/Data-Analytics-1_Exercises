from google.colab import files
uploaded = files.upload()

import pandas as pd

# Loaded the CSV file into a DataFrame
file_path = "Electronic_sales_Sep2023-Sep2024.csv"  
data = pd.read_csv(file_path)

print("Data Preview:")
print(data.head())

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

column_name = 'Total Price'

# Calculated Mean, Median, and Mode
mean = data[column_name].mean()
median = data[column_name].median()
mode = data[column_name].mode()[0]

# Calculated Variance, Standard Deviation, and Range
variance = np.var(data[column_name], ddof=1)
std_dev = np.std(data[column_name], ddof=1)
range = max(data[column_name]) - min(data[column_name])



# Results
print("Measure of Central Tendency:")
print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Mode: {mode}")

print("Measure of Dispersion:")
print(f"Variance: {variance}")
print(f"Standard Deviation: {std_dev}")
print(f"Range: {range}")

# Histogram
plt.hist(data[column_name], bins=8, color='blue', alpha=0.7, edgecolor='black')
plt.title("Histogram of Total Price")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# Boxplot
plt.boxplot(data[column_name], vert=False)
plt.title("Box Plot of Total Price")
plt.xlabel("Value")
plt.show()

