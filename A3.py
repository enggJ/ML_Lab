import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np

# Load dataset
df = pd.read_csv("stealthphisher2025.csv")

# Drop non-numeric columns and the classification target
df_numeric = df.select_dtypes(include=[np.number]).copy()

# Define features and target (use all numeric columns except 'ShannonEntropy' as features)
X = df_numeric.drop(columns=['ShannonEntropy'])
y = df_numeric['ShannonEntropy']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Function to train linear regression
def train_linear_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to make predictions
def predict(model, X):
    return model.predict(X)

# Function to calculate evaluation metrics
def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

# Train model and predict
model = train_linear_model(X_train, y_train)
y_train_pred = predict(model, X_train)
y_test_pred = predict(model, X_test)

# Evaluate
train_metrics = evaluate_regression(y_train, y_train_pred)
test_metrics = evaluate_regression(y_test, y_test_pred)

# Print outputs
print("Train Metrics (Multiple Features):")
print(f"  MSE  : {train_metrics[0]:.4f}")
print(f"  RMSE : {train_metrics[1]:.4f}")
print(f"  MAPE : {train_metrics[2]:.4f}")
print(f"  R^2  : {train_metrics[3]:.4f}")

print("\nTest Metrics (Multiple Features):")
print(f"  MSE  : {test_metrics[0]:.4f}")
print(f"  RMSE : {test_metrics[1]:.4f}")
print(f"  MAPE : {test_metrics[2]:.4f}")
print(f"  R^2  : {test_metrics[3]:.4f}")
