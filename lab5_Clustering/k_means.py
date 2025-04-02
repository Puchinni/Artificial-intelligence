import numpy as np

def initialize_centroids_forgy(data, k):
    # TODO implement random initialization
    return data[np.random.choice(range(data.shape[0]), replace=False, size=k), :]


def initialize_centroids_kmeans_pp(data, k):
    # TODO implement kmeans++ initizalization
    centroids = np.zeros((k, data.shape[1]))
    first_centroid = data[np.random.choice(range(data.shape[0]), replace=False), :]
    centroids[0, :] = first_centroid
    for i in range(1, k):
        distances = np.linalg.norm(data - centroids[i-1, :], axis=1)
        probability = distances**2 / np.sum(distances**2)
        next_centroid = data[np.random.choice(range(data.shape[0]), replace=False, p=probability), :]
        centroids[i, :] = next_centroid
    return centroids


def assign_to_cluster(data, centroid):
    # TODO find the closest cluster for each data point
    distances = np.linalg.norm(data[:, np.newaxis, :] - centroid, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(data, assignments):
    # TODO find new centroids based on the assignments
    new_centroids = np.zeros((len(np.unique(assignments)), data.shape[1]))
    for i in range(len(np.unique(assignments))):
        points_in_cluster = data[assignments==i, :]
        if points_in_cluster.shape[0] > 0:
            new_centroids[i, :] = np.mean(points_in_cluster, axis=0)
    return new_centroids
        

def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))

def k_means(data, num_centroids, kmeansplusplus= False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else: 
        centroids = initialize_centroids_forgy(data, num_centroids)

    
    assignments  = assign_to_cluster(data, centroids)
    for i in range(100): # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)         

