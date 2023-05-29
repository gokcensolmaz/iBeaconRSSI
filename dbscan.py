import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from preprocessing import data

agnes = DBSCAN(eps=0.3)
agnes_labels = agnes.fit_predict(data)

print("AGNES Labels: ", agnes_labels)

agnes_score = silhouette_score(data, agnes_labels)
print("Silhouette Score - AGNES:", agnes_score)