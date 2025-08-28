import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np

# Load the dataset
df = pd.read_csv("stealthphisher2025.csv")

# Drop non-numeric columns + the label (classification target)
df_numeric = df.select_dtypes(include=[np.number]).copy()
X = df_numeric.drop(columns=['ShannonEntropy'])  # Optional: remove pseudo-regression target too

# Split is not mandatory for clustering, but we can use full dataset for clustering
# Function to perform k-means clustering
def perform_kmeans(X, n_clusters=2, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    kmeans.fit(X)
    return kmeans

# Main clustering call
kmeans_model = perform_kmeans(X)

# Get clustering outputs
cluster_labels = kmeans_model.labels_
cluster_centers = kmeans_model.cluster_centers_

# Print results
print("Cluster Labels (first 10):", cluster_labels[:10])
print("\nCluster Centers (shape):", cluster_centers.shape)
print("\nCluster Centers (first 2):")
print(cluster_centers[:2])
