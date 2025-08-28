import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("StealthPhisher2025.csv")

# Convert labels to binary
df['Label'] = df['Label'].map({'Legitimate': 0, 'Phishing': 1})

# Select numeric features and target
X = df.select_dtypes(include=[np.number])
y = df['Label']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train kNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

accuracy = knn.score(X_test, y_test)
print("Test Accuracy (k=3):", accuracy)
