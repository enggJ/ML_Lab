import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

df = pd.read_csv('StealthPhisher2025.csv')  

# Drop irrelevant columns and assume 'PathLength' is the regression target
df = df.drop(columns=['URL', 'Domain', 'TLD'])
X = df.drop(columns=['Label', 'PathLength'])
y = df['PathLength']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("MAPE:", mape)
print("R² Score:", r2)
