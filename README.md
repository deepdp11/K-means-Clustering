# K-means-Clustering
1. Implementing K-means
1.1 Finding Closest Centroids
Let's start by implementing the function to find the closest centroids for each data point. This involves calculating the Euclidean distance between each data point and all centroids, and then assigning each point to the closest centroid.

import numpy as np

# Implementing function to find closest centroids
def find_closest_centroids(X, centroids):
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)
    
    for i in range(X.shape[0]):
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        idx[i] = np.argmin(distances)
    
    return idx
1.2 Computing Centroid Means
Next, let's implement the function to compute the centroid means. After assigning data points to centroids, we'll update the centroids based on the mean of all points assigned to them.

# Implementing function to compute centroid means
def compute_centroid_means(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    
    for k in range(K):
        centroids[k] = np.mean(X[idx == k], axis=0)
    
    return centroids
2. K-means on a Sample Dataset
Now, let's apply our implementation of K-means to a sample dataset to visualize how the algorithm works.


# Sample dataset
X = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])

# Initial centroids
initial_centroids = np.array([[1, 2], [8, 8]])

# Finding closest centroids
idx = find_closest_centroids(X, initial_centroids)
print("Closest centroids:", idx)

# Computing centroid means
centroids = compute_centroid_means(X, idx, initial_centroids.shape[0])
print("Updated centroids:", centroids)
3. Random Initialization
To ensure robustness and avoid local optima, we'll incorporate random initialization of centroids. This step is crucial for achieving reliable results with K-means.

4. Image Compression with K-means
4.1 Dataset
Let's load an image dataset to explore the practical application of K-means in image compression.

4.2 K-Means on Image Pixels
Applying K-means to image pixels, we'll identify the most representative colors by clustering similar pixels together.

4.3 Compress the Image
Finally, using the centroids obtained from K-means, we'll compress the image by replacing each pixel with the nearest centroid color. This significantly reduces the number of colors in the image, resulting in efficient compression without compromising on image quality.
