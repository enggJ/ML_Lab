import pandas as pd
import numpy as np
from math import log2

# Load dataset
df = pd.read_csv("stealthphisher2025.csv")

# Equal width binning
def equal_width_binning(series, bins=4):
    return pd.cut(series, bins=bins, labels=False)

# Entropy
def entropy(series):
    values, counts = np.unique(series, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum([p * log2(p) for p in probabilities if p > 0])

# Information gain
def information_gain(df, feature, target):
    total_entropy = entropy(df[target])
    values, counts = np.unique(df[feature], return_counts=True)
    weighted_entropy = np.sum([
        (counts[i] / np.sum(counts)) * entropy(df[df[feature] == values[i]][target])
        for i in range(len(values))
    ])
    return total_entropy - weighted_entropy

# Convert continuous to categorical
categorical_df = df.copy()
for col in df.columns:
    if df[col].dtype != 'object' and col != 'Label':
        categorical_df[col] = equal_width_binning(df[col], bins=4)

# Find root node
ig_scores = {col: information_gain(categorical_df, col, 'Label') for col in categorical_df.columns if col != 'Label'}
root_node = max(ig_scores, key=ig_scores.get)
print("Root Node =", root_node)
print("Information Gains:", ig_scores)
