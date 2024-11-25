import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway, pearsonr
import matplotlib.pyplot as plt
import numpy as np

# Loaded the Iris dataset
iris = sns.load_dataset("iris")

# T-Test: Compared mean petal lengths of Setosa and Versicolor
setosa_petal_length = iris[iris['species'] == 'setosa']['petal_length']
versicolor_petal_length = iris[iris['species'] == 'versicolor']['petal_length']

t_stat, p_value = ttest_ind(setosa_petal_length, versicolor_petal_length)

print("\nT-Test: Comparing Petal Lengths (Setosa vs. Versicolor)")
print(f"T-Statistic: {t_stat:.2f}, P-Value: {p_value:.4f}")
if p_value < 0.05:
    print("The means are significantly different.")
else:
    print("The means are not significantly different.")

# Z-Test: Tested if the mean sepal length of Setosa equals 5.0
setosa_sepal_length = iris[iris['species'] == 'setosa']['sepal_length']
mean_setosa = setosa_sepal_length.mean()
std_setosa = setosa_sepal_length.std(ddof=1)
z_stat = (mean_setosa - 5.0) / (std_setosa / np.sqrt(len(setosa_sepal_length)))

print("\nZ-Test: Is the Mean Sepal Length of Setosa 5.0?")
print(f"Mean: {mean_setosa:.2f}, Z-Statistic: {z_stat:.2f}")

# ANOVA: Compared petal widths across all three species
setosa_petal_width = iris[iris['species'] == 'setosa']['petal_width']
versicolor_petal_width = iris[iris['species'] == 'versicolor']['petal_width']
virginica_petal_width = iris[iris['species'] == 'virginica']['petal_width']

f_stat, p_value_anova = f_oneway(setosa_petal_width, versicolor_petal_width, virginica_petal_width)

print("\nANOVA: Comparing Petal Widths Across All Species")
print(f"F-Statistic: {f_stat:.2f}, P-Value: {p_value_anova:.4f}")
if p_value_anova < 0.05:
    print("At least one species has a significantly different mean.")
else:
    print("All species have similar means.")

# Correlation: Relationship between sepal length and petal length
sepal_length = iris['sepal_length']
petal_length = iris['petal_length']
correlation, p_value_corr = pearsonr(sepal_length, petal_length)

print("\nCorrelation: Sepal Length vs. Petal Length")
print(f"Correlation Coefficient: {correlation:.2f}, P-Value: {p_value_corr:.4f}")
if p_value_corr < 0.05:
    print("There is a significant correlation.")
else:
    print("No significant correlation.")

# Plotted the relationship
plt.figure(figsize=(8, 6))
plt.scatter(sepal_length, petal_length, color='blue', alpha=0.6, label="Data Points")
plt.title("Scatter Plot: Sepal Length vs. Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.grid()
plt.legend()
plt.show()
