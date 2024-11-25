import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import builtins 


# Loaded the Iris dataset using Seaborn
iris = sns.load_dataset("iris")

print("First few rows of the Iris dataset:")
print(iris.head())

# Random Sampling
random_sample = iris.sample(n=30, random_state=42)

print("\nRandomly sampled 30 observations:")
print(random_sample)

# Sample Mean Distribution Analysis
sample_size = 30
num_samples = 100
sample_means = []


for _ in builtins.range(num_samples):
    sample = iris.sample(n=sample_size)
    sample_mean = sample["sepal_length"].mean()
    sample_means.append(sample_mean)

# Distribution of the 100 sample means
plt.figure(figsize=(8, 6))
plt.hist(sample_means, bins=15, color="skyblue", edgecolor="black", alpha=0.7)
plt.title("Distribution of Sample Means (Central Limit Theorem)")
plt.xlabel("Sample Mean of Sepal Length")
plt.ylabel("Frequency")
plt.grid()
plt.show()

# Systematic Sampling
def systematic_sampling(data, percentage):
    """Perform systematic sampling."""
    interval = int(len(data) / (len(data) * percentage))
    sampled_data = data.iloc[::interval, :]
    return sampled_data

# 20% of the dataset using systematic sampling
sampled_data_systematic = systematic_sampling(iris, percentage=0.2)

plt.figure(figsize=(14, 6))

# Original dataset histogram
plt.subplot(1, 2, 1)
plt.hist(iris["sepal_length"], bins=15, color="blue", alpha=0.6, edgecolor="black")
plt.title("Original Dataset: Sepal Length Distribution")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.grid()

# Sampled dataset histogram
plt.subplot(1, 2, 2)
plt.hist(sampled_data_systematic["sepal_length"], bins=15, color="green", alpha=0.6, edgecolor="black")
plt.title("Systematic Sampled Dataset: Sepal Length Distribution")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.grid()

plt.tight_layout()
plt.show()
