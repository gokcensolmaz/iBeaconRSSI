import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from preprocessing import data

dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(data)

print("DBSCAN Labels: ", dbscan_labels)

dbscan_score = silhouette_score(data[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1])
print("Silhouette Score - AGNES:", dbscan_score)