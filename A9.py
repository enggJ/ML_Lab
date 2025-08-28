import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("StealthPhisher2025.csv")

# Encode labels: Legitimate → 0, Phishing → 1
df['Label'] = df['Label'].map({'Legitimate': 0, 'Phishing': 1})

# Separate numeric features and target label
X = df.select_dtypes(include=[np.number])
y = df['Label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train kNN (k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict test set
y_pred = knn.predict(X_test)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

report = classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"])
print("\nClassification Report:\n", report)
