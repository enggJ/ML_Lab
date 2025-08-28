import pandas as pd
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("StealthPhisher2025.csv")

# Encode 'Label' column to 0 and 1
df['Label'] = df['Label'].map({'Legitimate': 0, 'Phishing': 1})

# Select numeric columns for features (X) and target label (y)
X = df.select_dtypes(include=[np.number])
y = df['Label']

# Split into train and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the kNN classifier with k = 3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

print(" kNN model trained successfully with k = 3.")
