import pandas as pd
import shap
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


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# Global feature importance (summary plot)
plt.title("SHAP Summary Plot (Global Feature Importance)")
shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
plt.savefig("shap_summary_plot.png", dpi=300, bbox_inches="tight")
plt.show()
print("SHAP summary plot saved as shap_summary_plot.png")

# Local explanation for one instance
i = 5
shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][i,:],
    X_test[i,:],
    feature_names=X.columns,
    matplotlib=True
)
plt.savefig("shap_force_plot.png", dpi=300, bbox_inches="tight")
plt.show()
print("SHAP force plot (local explanation) saved as shap_force_plot.png")
