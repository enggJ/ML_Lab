import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("stealthphisher2025.csv")

df_encoded = df.copy()
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        df_encoded[col] = pd.factorize(df_encoded[col])[0]

X = df_encoded.drop('Label', axis=1)
y = df_encoded['Label']

clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X, y)

print("Decision Tree model trained successfully.")
