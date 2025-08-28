import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("stealthphisher2025.csv")

# Encode target
df['Label'] = LabelEncoder().fit_transform(df['Label'])  # Phishing=1, Legitimate=0

# Drop non-numeric columns (URL, Domain)
df = df.drop(columns=["URL", "Domain"])

# Encode TLD (string categorical)
if df['TLD'].dtype == "object":
    df['TLD'] = LabelEncoder().fit_transform(df['TLD'])

# Features and target
X = df.drop(columns=["Label"])
y = df["Label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Perceptron model
model = Perceptron(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
