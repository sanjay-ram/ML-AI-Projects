import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

data = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
dbscan = DBSCAN(eps=1.5, min_samples=2)
labels = dbscan.fit_predict(data)

plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.title('DBSCAN Clustering')
plt.show()