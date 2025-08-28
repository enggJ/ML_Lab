# A7_minimal_knn_tuning.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('StealthPhisher2025.csv')

# Encode target label
df['Label'] = LabelEncoder().fit_transform(df['Label'])

# Select 2 features and take only 500 rows to speed up grid search
features = ['LengthOfURL', 'DigitCntInURL']
df_sample = df[features + ['Label']].sample(n=500, random_state=42)
X = df_sample[features]
y = df_sample['Label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Limit the range of k values
param_grid = {'n_neighbors': list(range(1, 6))}  # Try k = 1 to 5

# Use GridSearchCV with 3-fold CV, no parallel jobs to reduce system load
grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    cv=3,
    verbose=1
)

# Fit the model
grid_search.fit(X_train, y_train)

# Output the best k
print(" A7: Best k =", grid_search.best_params_['n_neighbors'])
print(" A7: Best Cross-Validation Accuracy =", grid_search.best_score_)

# Optional: Evaluate on test set
best_knn = grid_search.best_estimator_
test_score = best_knn.score(X_test, y_test)
print(" Test Accuracy with Best k =", test_score)
