# A6_knn_project_data.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('StealthPhisher2025.csv')

# Encode labels
df['Label'] = LabelEncoder().fit_transform(df['Label'])

# Use all data and two features
features = ['LengthOfURL', 'DigitCntInURL']
X = df[features]
y = df['Label']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit kNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Generate meshgrid
x_min, x_max = X[features[0]].min() - 1, X[features[0]].max() + 1
y_min, y_max = X[features[1]].min() - 1, X[features[1]].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                     np.arange(y_min, y_max, 0.5))
grid_points = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=features)

# Predict and reshape
Z = knn.predict(grid_points).astype(int).reshape(xx.shape)

# Plot
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_train[features[0]], X_train[features[1]], c=y_train, edgecolor='k')
plt.title("A6: kNN Decision Boundary on Full Dataset (k=3)")
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.grid(True)
plt.show()
