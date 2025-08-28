# A2.py
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("stealthphisher2025.csv")

df['Label'] = LabelEncoder().fit_transform(df['Label'])
df = df.drop(columns=["URL", "Domain"])

if df['TLD'].dtype == "object":
    df['TLD'] = LabelEncoder().fit_transform(df['TLD'])

X = df.drop(columns=["Label"])
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define parameter grid
param_grid = {
    'penalty': ['l2', 'l1', 'elasticnet', None],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'max_iter': [500, 1000, 1500],
    'eta0': [0.1, 0.5, 1.0]
}

search = RandomizedSearchCV(
    estimator=Perceptron(random_state=42),
    param_distributions=param_grid,
    n_iter=10,
    scoring='accuracy',
    cv=5,
    verbose=1,
    random_state=42
)

search.fit(X_train, y_train)

print("Best Parameters:", search.best_params_)
print("Best CV Score:", search.best_score_)

# Evaluate best model
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
