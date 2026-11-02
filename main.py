import sys
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "Mall_Customers.csv")

features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = StandardScaler().fit_transform(df[features])

# PCA dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Explained variance:", pca.explained_variance_ratio_.round(3))
print("Total variance:", round(pca.explained_variance_ratio_.sum(), 3))

# Find optimal K using silhouette score
k_values = range(2, 11)
scores = [
    silhouette_score(
        X_pca,
        KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_pca)
    )
    for k in k_values
]

best_k = list(k_values)[scores.index(max(scores))]
print(f"\nOptimal clusters: {best_k}")
print(f"Silhouette score: {max(scores):.3f}")

# Final clustering
model = KMeans(n_clusters=best_k, n_init=10, random_state=42)
df["Cluster"] = model.fit_predict(X_pca)

# Cluster analysis
print("\nCustomer segments:")
print(df.groupby("Cluster")[features].mean().round(2))

# Save results
df.to_csv("customer_segments.csv", index=False)

# Visualisation
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot(k_values, scores, marker="o")
ax[0].set(title="Silhouette Analysis", xlabel="Number of Clusters", ylabel="Score")

ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=df["Cluster"], cmap="viridis")
ax[1].set(title="Customer Segmentation", xlabel="PC1", ylabel="PC2")

plt.tight_layout()
plt.savefig("segments.png", dpi=150)
plt.show()
