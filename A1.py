import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("stealthphisher2025.csv")

# Select one numerical column as input and one as target
# Use 'LengthOfURL' to predict 'ShannonEntropy' as per instructions
X = df[['LengthOfURL']]          # Feature (independent variable)
y = df['ShannonEntropy']         # Pseudo target for regression

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train linear model
def train_linear_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to predict using model
def predict(model, X):
    return model.predict(X)


model = train_linear_model(X_train, y_train)
y_train_pred = predict(model, X_train)
y_test_pred = predict(model, X_test)
print("Train Predictions Sample:", y_train_pred[:5])
print("Test Predictions Sample:", y_test_pred[:5])
