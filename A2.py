import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("stealthphisher2025.csv")

def gini_index(series):
    values, counts = np.unique(series, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities ** 2)

gini_value = gini_index(df['Label'])
print("Gini Index of Label =", gini_value)
