import pandas as pd
import numpy as np  
from sklearn.model_selection import train_test_split

df = pd.read_csv("StealthPhisher2025.csv")

# Encode class labels: 'Legitimate' -> 0, 'Phishing' -> 1
df['Label'] = df['Label'].map({'Legitimate': 0, 'Phishing': 1})

# Select numeric features (input) and target labels (output)
X = df.select_dtypes(include=[np.number])
y = df['Label']

# Split the dataset: 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training Set Size:", X_train.shape)
print("Testing Set Size:", X_test.shape)
