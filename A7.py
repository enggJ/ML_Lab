import pandas as pd
import numpy as np  
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("stealthphisher2025.csv")

# Keep only numeric features and drop classification target
df_numeric = df.select_dtypes(include=[np.number]).copy()
X = df_numeric.drop(columns=['ShannonEntropy'])  # Remove pseudo regression target

# Elbow plot calculation
distortions = []
k_values = range(2, 20)  # As per instructions

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X)
    distortions.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(6, 4))
plt.plot(k_values, distortions, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Distortion (Inertia)")
plt.grid(True)
plt.show()
