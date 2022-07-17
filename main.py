from k_centers import k_centers, calc_minkowski_distance
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import time
import math

def main(filename):
    print("MANHATTAN DISTANCE")
    generate_stats(filename, 1)
    print("EUCLIDEAN DISTANCE")
    generate_stats(filename, 2)

def generate_stats(filename, p):
    data = pd.read_csv(filename)
    k = data["cat"].nunique()
    data_numeric = (data.iloc[:, 0:-2]).astype(float)
    dataset = (data_numeric).to_numpy()
    execution_time_k_centers = []
    execution_time_kmeans = []
    radius_k_centers = []
    radius_kmeans = []
    silhouette_scores_k_centers = []
    silhouette_scores_kmeans = []
    rand_scores_k_centers = []
    rand_scores_kmeans = []
    metric = "euclidean" if p == 2 else "manhattan"

    for _ in range(30):
        start = time.time()
        radix, centroids = k_centers(dataset, k, p)
        radius_k_centers.append(radix)
        _, labels = get_radius(dataset, np.array(centroids), p)
        silhouette_scores_k_centers.append(silhouette_score(dataset, labels, metric=metric))
        rand_scores_k_centers.append(adjusted_rand_score(data["cat"], labels))
        end = time.time()
        execution_time_k_centers.append(end - start)

        start = time.time()
        kmeans = KMeans(n_clusters=k, random_state=0).fit(dataset)
        radix = get_radius(dataset, kmeans.cluster_centers_, p)[0]
        radius_kmeans.append(radix)
        silhouette_scores_kmeans.append(silhouette_score(dataset, kmeans.labels_, metric=metric))
        rand_scores_kmeans.append(adjusted_rand_score(data["cat"], kmeans.labels_))
        end = time.time()
        execution_time_kmeans.append(end - start)
    

    print("============== STATS ==============")
    print("🐢🐢 Section: K-centers: 🐢🐢")
    print("Iterations:", len(execution_time_k_centers))
    print("Avg. Execution time:", np.mean(np.array(execution_time_k_centers)))
    print("Avg. Radius:", np.mean(np.array(radius_k_centers)))
    print("SD. Radius:", np.std(np.array(radius_k_centers)))
    print("Avg. Silhouette score:", np.mean(np.array(silhouette_scores_k_centers)))
    print("SD. Silhouette score:", np.std(np.array(silhouette_scores_k_centers)))
    print("Avg. Rand score:", np.mean(np.array(rand_scores_k_centers)))
    print("SD. Rand score:", np.std(np.array(rand_scores_k_centers)))


    print("\n🐇🐇 Section: K-means: 🐇🐇")
    print("Iterations:", len(execution_time_kmeans))
    print("Avg. Execution time:", np.mean(np.array(execution_time_kmeans)))
    print("Avg. Radius:", np.mean(np.array(radius_kmeans)))
    print("SD. Radius:", np.mean(np.array(radius_kmeans)))
    print("Avg. Silhouette score:", np.mean(np.array(silhouette_scores_kmeans)))
    print("SD. Silhouette score:", np.std(np.array(silhouette_scores_kmeans)))
    print("Avg. Rand score:", np.mean(np.array(rand_scores_kmeans)))
    print("SD. Rand score:", np.std(np.array(rand_scores_kmeans)))
    print("===================================")


def get_distance_from_point_to_centroid(point, centroids, p):
    minimum_distance = math.inf
    center_index = 0
    for i in range(centroids.shape[0]):
        distance = calc_minkowski_distance(point, centroids[i], p)
        if distance < minimum_distance:
            minimum_distance = distance
            center_index = i
    return (center_index, minimum_distance)

def get_radius(points, centroids, p):
    radius = 0
    labels = []
    for i in range(points.shape[0]):
        center, distance = get_distance_from_point_to_centroid(points[i], centroids, p)
        labels.append(center)
        if distance > radius:
            radius = distance
    return radius, labels

files = ["/Users/ufmg_thiagomiarelli/Downloads/data_alg/abalone.csv", "/Users/ufmg_thiagomiarelli/Downloads/data_alg/adult.csv", "/Users/ufmg_thiagomiarelli/Downloads/data_alg/allhyper.csv", "/Users/ufmg_thiagomiarelli/Downloads/data_alg/arrhythmia.csv", "/Users/ufmg_thiagomiarelli/Downloads/data_alg/australian.dat.csv", "/Users/ufmg_thiagomiarelli/Downloads/data_alg/auto-mpg.csv", "/Users/ufmg_thiagomiarelli/Downloads/data_alg/bands.csv", "/Users/ufmg_thiagomiarelli/Downloads/data_alg/CalIt2.csv", "/Users/ufmg_thiagomiarelli/Downloads/data_alg/cmc.csv"]

for i in files:
    try:
        print("Analyzing file:", i)
        main(i)
    except:
        print("Error on file:", i)
        continue