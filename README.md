# Customer Segmentation using K-Means Clustering and PCA

Segments mall customers by age, annual income and spending score.

## Method
1. Standardise the features.
2. Reduce to 2 dimensions with PCA.
3. Choose the number of clusters `k` (2-10) with the silhouette score.
4. Cluster with K-Means and plot the segments.

## Run
Download the [Mall Customers dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) and save it as `Mall_Customers.csv` in this folder.

```bash
python main.py
```

Output: explained variance, best `k`, per-cluster feature means, and `segments.png`.

## License
GPL-3.0
