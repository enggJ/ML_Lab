import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

df = pd.read_csv("StealthPhisher2025.csv")
df_numeric = df.select_dtypes(include=[np.number])

# Select two vectors
vec1 = df_numeric.iloc[0]
vec2 = df_numeric.iloc[1]

# Minkowski distances
minkowski_distances = [distance.minkowski(vec1, vec2, p=r) for r in range(1, 11)]

# Plot
plt.plot(range(1, 11), minkowski_distances, marker='o')
plt.title("Minkowski Distance from r=1 to 10")
plt.xlabel("r")
plt.ylabel("Distance")
plt.grid(True)
plt.show()
