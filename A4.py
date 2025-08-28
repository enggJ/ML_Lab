import pandas as pd

# Load dataset
df = pd.read_csv("stealthphisher2025.csv")

# Binning function
def binning(series, bins=4, method="equal_width"):
    if method == "equal_width":
        return pd.cut(series, bins=bins, labels=False)
    elif method == "frequency":
        # Allow dropping duplicate bin edges if too many identical values exist
        return pd.qcut(series, q=bins, labels=False, duplicates="drop")
    else:
        raise ValueError("Unknown method. Use 'equal_width' or 'frequency'.")

# Apply binning to PathLength
df['PathLength_binned'] = binning(df['PathLength'], bins=4, method="frequency")
print("PathLength after frequency binning:")
print(df[['PathLength', 'PathLength_binned']].head())
