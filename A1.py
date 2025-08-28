import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load your dataset
df = pd.read_csv('StealthPhisher2025.csv')  

df = df.drop(columns=['URL', 'Domain', 'TLD'])  # These are string columns
df['Label'] = LabelEncoder().fit_transform(df['Label'])  # Phishing=1, Legitimate=0
X = df.drop(columns=['Label'])
y = df['Label']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
# Predictions
y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)
# Confusion matrix and classification report
print("Training Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))
print("\nTraining Classification Report:\n", classification_report(y_train, y_pred_train))
print("Testing Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
print("\nTesting Classification Report:\n", classification_report(y_test, y_pred_test))
