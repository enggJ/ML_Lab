import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.random.uniform(1, 10, size=(20, 2))
labels = np.random.choice([0, 1], size=20)

class0 = X[labels == 0]
class1 = X[labels == 1]

plt.figure(figsize=(8, 6))
plt.scatter(class0[:, 0], class0[:, 1], color='blue', label='Class 0 (Blue)')
plt.scatter(class1[:, 0], class1[:, 1], color='red', label='Class 1 (Red)')
plt.xlabel('Feature X')
plt.ylabel('Feature Y')
plt.title('Scatter Plot of Training Data (20 Points)')
plt.legend()
plt.grid(True)
plt.show()
