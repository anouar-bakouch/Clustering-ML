import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# Load the data
data = np.loadtxt("res/patients.csv", delimiter=",")

# Create the clustering object
clusterer = AgglomerativeClustering(n_clusters=None, affinity="euclidean", linkage="ward")

# Fit the clustering object to the data
clusterer.fit(data)

# Get the cluster labels
labels = clusterer.labels_

# Plot the dendrogram
plt.figure()
plt.title("Dendrogram")
plt.xlabel("Patient ID")
plt.ylabel("Distance")
plt.plot(clusterer.distance_matrix_)
plt.show()

# Print the cluster labels
print(labels)
