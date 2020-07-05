import numpy as np
from collections import defaultdict

class KMeans():
    def __init__(self, n_clusters, epochs = 10):
        self.n_clusters = n_clusters
        self.epochs = epochs
        self.means = []
        self.labels = [c for c in range(n_clusters)]

    def transform_data(self, data):
        return [{'k': np.random.randint(0, self.n_clusters), 'data': data_point} for data_point in data]

    def map_labels(self, points):
        point_dict = defaultdict(list)
        for p in points:
            point_dict[p['k']] = point_dict[p['k']] + [p['data']]
        return point_dict

    def calc_k_means(self, point_dict):
        return [np.mean(point_dict[k], axis=0) for k in self.labels]

    def predict(self, point):
        # calculate the distances to all the means
        pass

    def update_k(self, points):
        for p in points:
            distances = [np.linalg.norm(self.means[k] - p['data']) for k in self.labels]
            p['k'] = np.argmin(distances)

    def fit(self, data):
        points = self.transform_data(data)
        for epoch in range(self.epochs):
            point_dict = self.map_labels(points)
            self.means = self.calc_k_means(point_dict)
            self.update_k(points)