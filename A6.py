import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("stealthphisher2025.csv")

# Keep only numeric columns and remove classification target
df_numeric = df.select_dtypes(include=[np.number]).copy()
X = df_numeric.drop(columns=['ShannonEntropy'])

# Take a fixed-size sample for metric calculation
X_sample = X.sample(n=min(1000, len(X)), random_state=42)

def evaluate_kmeans_for_k(X_full, X_sample, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X_full)
    sample_labels = kmeans.predict(X_sample)

    if len(np.unique(sample_labels)) < 2:
        sil = np.nan  # Can't compute silhouette if only one cluster
    else:
        sil = silhouette_score(X_sample, sample_labels)

    ch = calinski_harabasz_score(X_sample, sample_labels) if len(np.unique(sample_labels)) > 1 else np.nan
    db = davies_bouldin_score(X_sample, sample_labels) if len(np.unique(sample_labels)) > 1 else np.nan

    return sil, ch, db

# Loop through k values
k_values = range(2, 11)
sil_scores = []
ch_scores = []
db_scores = []

for k in k_values:
    sil, ch, db = evaluate_kmeans_for_k(X, X_sample, k)
    sil_scores.append(sil)
    ch_scores.append(ch)
    db_scores.append(db)

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(k_values, sil_scores, marker='o')
plt.title("Silhouette Score vs k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")

plt.subplot(1, 3, 2)
plt.plot(k_values, ch_scores, marker='o', color='green')
plt.title("Calinski-Harabasz Score vs k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("CH Score")

plt.subplot(1, 3, 3)
plt.plot(k_values, db_scores, marker='o', color='red')
plt.title("Davies-Bouldin Index vs k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("DB Index")

plt.tight_layout()
plt.show()
fgb

