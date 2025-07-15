import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
import joblib as jb

iris = load_iris()
x = iris.data
features = iris.feature_names

df = pd.DataFrame(x, columns=features)
km = KMeans(n_clusters=3, random_state=42)
km.fit(x)

df['Cluster'] = km.labels_
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x=features[2], y=features[3], hue='Cluster', palette='Set2', s=70)
plt.title("K-Means Clustering on Iris Dataset")
plt.grid(True)
plt.show()

jb.dump(km, "Models and Datasets\\K_Means_Clustering.pkl")