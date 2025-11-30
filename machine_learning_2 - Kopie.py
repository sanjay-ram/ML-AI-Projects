import matplotlib.pyplot as plt
import hdbscan
from machine_learning import data

hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size= 2)
hdbscan_labels =  hdbscan_clusterer.fit_predict(data)

plt.scatter(data[:, 0], data[:, 1], c=hdbscan_labels, cmap='rainbow')
plt.title('HDBSCAN Clustering')
plt.show()