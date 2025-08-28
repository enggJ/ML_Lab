# A5_knn_grid_varying_k.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('StealthPhisher2025.csv')

# Encode labels as integers (0, 1)
df['Label'] = LabelEncoder().fit_transform(df['Label'])

# Use two selected features and first 20 rows
features = ['LengthOfURL', 'DigitCntInURL']
X_train = df[features].head(20)
y_train = df['Label'].head(20)

# Create test grid
x_min, x_max = X_train[features[0]].min() - 1, X_train[features[0]].max() + 1
y_min, y_max = X_train[features[1]].min() - 1, X_train[features[1]].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                     np.arange(y_min, y_max, 0.5))
test_points = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=features)

# Test different k values
for k in [1, 3, 5, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    Z = knn.predict(test_points).astype(int).reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X_train[features[0]], X_train[features[1]], c=y_train, edgecolor='k')
    plt.title(f"A5: kNN Decision Boundary (k={k})")
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.grid(True)
    plt.show()
