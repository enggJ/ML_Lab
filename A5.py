import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np

# Load the dataset
df = pd.read_csv("stealthphisher2025.csv")

# Keep only numeric features and remove classification target
df_numeric = df.select_dtypes(include=[np.number]).copy()
X = df_numeric.drop(columns=['ShannonEntropy'])  # Remove pseudo regression target if used earlier

# Optional: sample for faster metric computation
sample_frac = 0.2  # Use 20% of data for metrics
X_sample = X.sample(frac=sample_frac, random_state=42)

# Function to perform k-means clustering
def perform_kmeans(X, n_clusters=2, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    kmeans.fit(X)
    return kmeans

# Function to calculate clustering metrics
def evaluate_clustering(X, labels):
    sil_score = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    return sil_score, ch_score, db_score

# Run k-means on full dataset
kmeans_model = perform_kmeans(X)

# Predict labels for sample
sample_labels = kmeans_model.predict(X_sample)

# Get metrics (on sample for speed)
silhouette, ch, db = evaluate_clustering(X_sample, sample_labels)

print("Clustering Evaluation Metrics (k=2) — using sample:")
print(f"  Silhouette Score       : {silhouette:.4f}")
print(f"  Calinski-Harabasz Score: {ch:.4f}")
print(f"  Davies-Bouldin Index   : {db:.4f}")
