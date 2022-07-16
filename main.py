from k_centers import k_centers
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

def main(filename, p):
    data = pd.read_csv(filename)
    data = data.astype(float)
    k = data["cat"].nunique()
    dataset = (data.iloc[:, 1:-1]).to_numpy()
    radius, center_points, inertia = k_centers(dataset, k, p)
    k_means = KMeans(n_clusters=k, random_state=0, init = "k-means++").fit(dataset)
    print(radius)
    print(inertia)
    print(k_means.inertia_)



