import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = load_iris()
x = data.data
y = data.target
target_names = data.target_names

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

plt.figure(figsize=(8, 6))
for target, colour, name in zip([0, 1, 2], ['r', 'g', 'b'], target_names):
    plt.scatter(x_pca[y == target, 0], x_pca[y == target, 1], lable=name, color=colour)

plt.xlabel("Principal Component 1 --->")
plt.ylabel("Principal Component 2 --->")
plt.legend()
plt.title("PCA on the Iris Dataset")
plt.grid(True)
plt.tight_layout()
plt.show()