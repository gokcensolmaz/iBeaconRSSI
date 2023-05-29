import pandas as pd
import numpy as np

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from preprocessing import data


kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
kmeans_labels = kmeans.labels_

print("K-Means Labels:", kmeans_labels)

kmeans_score = silhouette_score(data, kmeans_labels)
print("Silhouette Score - K-Means:", kmeans_score)



