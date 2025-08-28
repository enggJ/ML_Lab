import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

df = pd.read_csv("stealthphisher2025.csv")

df_encoded = df.copy()
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        df_encoded[col] = pd.factorize(df_encoded[col])[0]

X = df_encoded.drop('Label', axis=1)
y = df_encoded['Label']

clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X, y)

# Plot
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=['Legitimate', 'Phishing'], filled=True)
plt.title("Decision Tree Visualization (A6)")
plt.show()
