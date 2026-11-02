"""Customer segmentation with PCA + K-Means.

Usage: python segment.py [path/to/Mall_Customers.csv]
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "Mall_Customers.csv")
X = StandardScaler().fit_transform(df[FEATURES])

# PCA: project to 2 components for visualisation
pca = PCA(n_components=2).fit(X)
X2 = pca.transform(X)
print("PCA explained variance:", pca.explained_variance_ratio_.round(3))

# Pick k by silhouette score
ks = range(2, 11)
scores = [silhouette_score(X2, KMeans(k, n_init=10, random_state=42).fit_predict(X2)) for k in ks]
best_k = ks[scores.index(max(scores))]
print(f"Best k = {best_k} (silhouette = {max(scores):.3f})")

# Final clustering
df["Cluster"] = KMeans(best_k, n_init=10, random_state=42).fit_predict(X2)
print(df.groupby("Cluster")[FEATURES].mean().round(1))

# Plots
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
a.plot(ks, scores, marker="o")
a.set(title="Silhouette score vs k", xlabel="k", ylabel="Silhouette")
b.scatter(X2[:, 0], X2[:, 1], c=df["Cluster"], cmap="tab10", s=25)
b.set(title=f"Customer segments (k={best_k})", xlabel="PC1", ylabel="PC2")
plt.tight_layout()
plt.savefig("segments.png", dpi=150)
