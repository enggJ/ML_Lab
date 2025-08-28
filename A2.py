import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("StealthPhisher2025.csv")

feature = 'LengthOfURL'
data = df[feature]

# Plot the histogram
plt.hist(data, bins=20, edgecolor='black')
plt.title(f'Histogram of {feature}')
plt.xlabel(feature)
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

print("Mean:", data.mean())
print("Variance:", data.var())
