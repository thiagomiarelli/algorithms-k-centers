import numpy as np
import random
import math

def calc_minkowski_distance(X, Y, p):
    """
    Calculates the Minkowski distance between two points.
    """
    return np.power(np.sum(np.power(np.abs(X - Y), p)), 1/p)

def build_distance_matrix(X, p):
    """
    Builds a distance matrix for a dataset.
    """
    distance_matrix = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            distance_matrix[i][j] = calc_minkowski_distance(X[i], X[j], p)
    return distance_matrix

def maximize_center_distances(distances, centroids):
    """
    Maximizes the distances between each point and its closest center.
    """
    further_center = 0
    further_point = 0
    further_distance = 0

    for j in range(distances.shape[0]):
        minimum_distance = math.inf
        minimum_center = 0
        for i in range(len(centroids)):
            if distances[j][centroids[i]] < minimum_distance:
                minimum_distance = distances[j][centroids[i]]
                minimum_center = centroids[i]

        if minimum_distance > further_distance:
            further_distance = minimum_distance
            further_point = j
            further_center = minimum_center

    return(further_point, further_center)

def calcInertia(distances, centroids):
    """
    Calculates the inertia of a dataset.
    """
    inertia = 0
    for j in range(distances.shape[0]):
        minimum_distance = math.inf
        minimum_center = 0
        for i in range(len(centroids)):
            if distances[j][centroids[i]] < minimum_distance:
                minimum_distance = distances[j][centroids[i]]
                minimum_center = centroids[i]
        inertia += minimum_distance**2
    return inertia

def set_k_centers(distances, k, p):
    """
    Calculates the k-centers of a dataset.
    """
    centroids = []
    centroids.append(random.randint(0, distances.shape[0]-1))

    for i in range(k - 1):
        centroids.append(maximize_center_distances(distances, centroids)[0])

    return centroids

def k_centers(dataset, k, p):

    if dataset.shape[0] < k:
        return dataset
    
    distances = build_distance_matrix(dataset, p)

    centroids = set_k_centers(distances, k, p)
    radius = calc_minkowski_distance(*maximize_center_distances(distances, centroids), p)
    center_points = []
    center_points.append(dataset[centroids])
    inertia = calcInertia(distances, centroids)
    
    return(radius, center_points, inertia)
