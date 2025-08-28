import pandas as pd
import numpy as np
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("stealthphisher2025.csv")

# Encode target
df['Label'] = LabelEncoder().fit_transform(df['Label'])

# Drop text columns
df = df.drop(columns=["URL", "Domain"])

# Encode categorical column (TLD)
if df['TLD'].dtype == "object":
    df['TLD'] = LabelEncoder().fit_transform(df['TLD'])

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

# -------------------------------
# Train RandomForest model
# -------------------------------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# -------------------------------
# LIME Explainer
# -------------------------------
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns,
    class_names=["Legitimate", "Phishing"],
    mode='classification'
)

# Pick one sample instance to explain
i = 5
exp = explainer.explain_instance(X_test[i], clf.predict_proba, num_features=10)

# Save interactive HTML
exp.save_to_file("lime_explanation.html")
print("LIME explanation saved as lime_explanation.html (open it in your browser)")


fig = exp.as_pyplot_figure()
plt.tight_layout()
plt.savefig("lime_explanation_plot.png", dpi=300)
plt.show()
print("Static plot saved as lime_explanation_plot.png")

