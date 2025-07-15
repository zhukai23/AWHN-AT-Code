from scipy.io import loadmat
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


class BuildGraph:
    def __init__(self, n_neighbors):
        self.path1 = r''
        self.neighbors = n_neighbors

    def load_data(self):
        data1 = loadmat(self.path1)
        pc1 = torch.from_numpy(data1['BrainNetSet']).float()
        label = torch.tensor(np.concatenate((np.ones(), np.zeros()))).long()

        return pc1, label

    def create_knn_graph(self, pc, num_nodes):
        adj_matrix = np.zeros((len(pc), num_nodes, num_nodes))
        for subject in range(len(pc)):
            corr_matrix = pc[subject]
            kNN = NearestNeighbors(n_neighbors=self.neighbors, metric='euclidean')
            kNN.fit(corr_matrix)
            distances, indices = kNN.kneighbors(corr_matrix)
            for node, neighbors in enumerate(indices):
                for neighbor, distance in zip(neighbors[1:], distances[node][1:]):
                    adj_matrix[subject, node, neighbor] = distance
                    adj_matrix[subject, neighbor, node] = distance
            return adj_matrix



