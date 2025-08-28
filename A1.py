import pandas as pd
import numpy as np

df = pd.read_csv("StealthPhisher2025.csv")

# Drop non-numeric columns
df_numeric = df.select_dtypes(include=[np.number])
labels = df['Label']

class1 = df_numeric[labels == 'Phishing']
class2 = df_numeric[labels == 'Legitimate']

# Mean vector (centroid) for each class
centroid1 = class1.mean(axis=0)
centroid2 = class2.mean(axis=0)

# Spread (standard deviation) for each class
spread1 = class1.std(axis=0)
spread2 = class2.std(axis=0)

# Euclidean distance between centroids
interclass_distance = np.linalg.norm(centroid1 - centroid2)

print("Centroid Phishing:\n", centroid1)
print("Centroid Legitimate:\n", centroid2)
print("Intraclass Spread Phishing:\n", spread1)
print("Intraclass Spread Legitimate:\n", spread2)
print("Interclass Distance:", interclass_distance)
