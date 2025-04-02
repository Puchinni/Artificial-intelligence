from k_means import k_means
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_iris():
    data = pd.read_csv("data/iris.data", names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
    print(data)
    classes = data["class"].to_numpy()
    features = data.drop("class", axis=1).to_numpy()
    return features, classes

def evaluate(clusters, labels):
    for cluster in np.unique(clusters):
        labels_in_cluster = labels[clusters==cluster]
        print(f"Cluster: {cluster}")
        for label_type in np.unique(labels):
            print(f"Num of {label_type}: {np.sum(labels_in_cluster==label_type)}")
    

def clustering(kmeans_pp):
    data = load_iris()
    features, classes = data
    intra_class_variance = []
    for i in range(100):
        assignments, centroids, error = k_means(features, 3, kmeans_pp)
        evaluate(assignments, classes)
        intra_class_variance.append(error)
        # plt.scatter(features[assignments==0, 0], features[assignments==0, 1], c="r", label="Cluster 1")
        # plt.scatter(features[assignments==1, 0], features[assignments==1, 1], c="g", label="Cluster 2")
        # plt.scatter(features[assignments==2, 0], features[assignments==2, 1], c="b", label="Cluster 3")
        # plt.scatter(centroids[:, 0], centroids[:, 1], c="black", label="Centroids")
        # plt.legend()
        # plt.show()
        

    # plt.scatter(features[assignments==0, 0], features[assignments==0, 1], c="r", label="Cluster 1")
    # plt.scatter(features[assignments==1, 0], features[assignments==1, 1], c="g", label="Cluster 2")
    # plt.scatter(features[assignments==2, 0], features[assignments==2, 1], c="b", label="Cluster 3")
    # plt.scatter(centroids[:, 0], centroids[:, 1], c="black", label="Centroids")
    # plt.legend()
    # plt.show()
            
    print(f"Mean intra-class variance: {np.mean(intra_class_variance)}")

if __name__=="__main__":
    
    clustering(kmeans_pp = True)
    #clustering(kmeans_pp = False)
