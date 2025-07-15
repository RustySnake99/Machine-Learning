from sklearn.datasets import load_wine
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

data = load_wine()
x = data.data
y = data.target

tsne = TSNE(n_components=2, random_state=42)
x_embedded = tsne.fit_transform(x)

plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=y, cmap='viridis')
plt.title("t-SNE Visualization based on Wine-Quality Dataset")
plt.xlabel("t-SNE 1 ---->")
plt.ylabel("t-SNE 2 ---->")
plt.colorbar()
plt.show()