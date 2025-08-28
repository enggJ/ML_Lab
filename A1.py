import pandas as pd
import numpy as np
from math import log2

df = pd.read_csv("stealthphisher2025.csv")

# Equal width binning function
def equal_width_binning(series, bins=4):
    return pd.cut(series, bins=bins, labels=False)

# Entropy function 
def entropy(series):
    values, counts = np.unique(series, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum([p * log2(p) for p in probabilities if p > 0])

target_entropy = entropy(df['Label'])
print("Entropy of Label =", target_entropy)
