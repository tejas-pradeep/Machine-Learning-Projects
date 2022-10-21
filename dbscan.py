import numpy as np
from kmeans import pairwise_dist 

class DBSCAN(object):
    def __init__(self, eps, minPts, dataset):
        self.eps = eps
        self.minPts = minPts
        self.dataset = dataset
    def fit(self):
        """Fits DBSCAN to dataset and hyperparameters defined in init().
        Args:
            None
        Return:
            cluster_idx: (N, ) int numpy array of assignment of clusters for each point in dataset
        Hint: Using sets for visitedIndices may be helpful here 
        """
        n = len(self.dataset)
        cluster_idx = [-1] * n
        C = 0
        visitedIndices = set()
        for i in range(n):
            if i in visitedIndices:
                continue
            visitedIndices.add(i)
            neighbourPts = self.regionQuery(i)
            if len(neighbourPts) < self.minPts:
                cluster_idx[i] = -1
            else:
                self.expandCluster(i, neighbourPts, C, cluster_idx, visitedIndices)
                C += 1
                # print(len(visitedIndices))
        return np.array(cluster_idx)
    def expandCluster(self, index, neighborIndices, C, cluster_idx, visitedIndices):
        """Expands cluster C using the point P, its neighbors, and any points density-reachable to P and updates indices visited, cluster assignments accordingly

        Args:
            index: index of point P in dataset (self.dataset)
            neighborIndices: (N, ) int numpy array, indices of all points witin P's eps-neighborhood
            C: current cluster
            cluster_idx: (N, ) int numpy array of current assignment of clusters for each point in dataset
            visitedIndices: set of indices in dataset visited so far
        Return:
            None
        Hint: np.concatenate(), np.unique(), and np.take() may be helpful here
        """
        # if not neighborIndices.any() or neighborIndices == np.array([]):
        #     return
        cluster_idx[index] = C
        i = 0
        # added_neighbours = np.array([])
        while True:
            if i == len(neighborIndices):
                break
            idx = neighborIndices[i]
            if idx in visitedIndices:
                i += 1
                if cluster_idx[idx] == -1:
                    cluster_idx[idx] = C
                continue
            visitedIndices.add(idx)
            new_neightbours = self.regionQuery(idx)
            if len(new_neightbours) >= self.minPts:
                neighborIndices = np.unique(np.concatenate((new_neightbours, neighborIndices)))
                neighborIndices = np.take(neighborIndices, np.unique(neighborIndices, return_index=True)[1])
                i = 0
            if cluster_idx[idx] == -1:
                cluster_idx[idx] = C
            i += 1
        # print(added_neighbours)
        # self.expandCluster(index, added_neighbours, C, cluster_idx, visitedIndices)

    def regionQuery(self, pointIndex):
        """Returns all points within P's eps-neighborhood (including P)

        Args:
            pointIndex: index of point P in dataset (self.dataset)
        Return:
            indices: (N, ) int numpy array, indices of all points witin P's eps-neighborhood
        Hint: pairwise_dist (implemented above) and np.argwhere may be helpful here
        """
        pairwise_dists = pairwise_dist(self.dataset, self.dataset[pointIndex])
        # print(np.where(pairwise_dists[:, 0] < self.eps))
        return np.where(pairwise_dists[:, 0] < self.eps)[0]