# !/usr/bin/env python3
# encoding=utf-8

# from sklearn.cluster import DBSCAN
from sklearn.neighbors import *
import numpy as np
import os, sys
import argparse as ap
from utils.preprocess import load_data
from utils.metrics import silhouette_score_
from utils.visualize import visualize, MDS_visualize
from utils.config import args

class dbscan:
    def __init__(self, args):
        self.eps = args.eps
        self.min_samples = args.min_samples
        self.reduction = args.reduction
        self.random_state = args.random_state
        self.algo = args.algo

    def euclidean(self, X, y, square = False):
        if not square:
            return np.sqrt(np.transpose([np.sum(X* X, axis=1)]) + np.sum(y* y, axis=1) - 2* X.dot(y.T))
        else:
            return np.transpose([np.sum(X* X, axis=1)]) + np.sum(y* y, axis=1) - 2* X.dot(y.T)

    def  neighbors(self, X):
        neighborhoods = []
        dissimilarity = self.euclidean(X, X)
        for i in range(X.shape[0]):
            neighborhoods.append(np.where(dissimilarity[i] <= self.eps)[0])
        return np.array(neighborhoods)    


    def dbscan_inner(self, is_core, neighborhoods):
        stack = list()
        labels = self.y
        label_num = 0

        for i in range(labels.shape[0]):
            if labels[i] != -1 or not is_core[i]:
                continue
            while True:
                if labels[i] == -1:
                    labels[i] = label_num
                    if is_core[i]:
                        neighbors = neighborhoods[i]
                        for n in range(neighbors.shape[0]):
                            neighb = neighbors[n]
                            if labels[neighb] == -1:
                                stack.append(neighb)

                if len(stack) == 0:
                    break
                i = stack.pop()
            
            label_num += 1
        
        return labels
    
    def get_clusters(self, X):
        means = []
        for i in range(self.n_clusters):
            mean = np.mean(X[np.where(self.y == i)], axis=0)
            means.append(mean)
        return means

    def run(self, raw_data):

        X = np.array(raw_data)
    
        if self.algo and self.algo == 'standard':
            # clusters = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(X)
            # self.y = clusters.labels_
            pass

        else:
            neighbors_model = NearestNeighbors(radius=self.eps, algorithm='auto',
                                           leaf_size=30,
                                           metric='minkowski',
                                           metric_params=None, p=2,
                                           n_jobs=None)
            neighbors_model.fit(X)
            # This has worst case O(n^2) memory complexity
            neighborhoods = neighbors_model.radius_neighbors(X, self.eps,
                                                         return_distance=False)
            
            n_neighbors = np.array([neighbors.shape[0] for neighbors in neighborhoods])
            # print(n_neighbors)

            # Initially, all samples are noise.
            self.y = np.full(X.shape[0], -1)

            # A list of all core samples found.
            core_samples = np.asarray(n_neighbors >= self.min_samples, dtype=np.uint8)
            # print(core_samples)
            # dbscan_inner(core_samples, neighborhoods, self.y)
            self.y = self.dbscan_inner(core_samples, neighborhoods)

        clusters = set(self.y)
        clusters.discard(-1)
        self.n_clusters = len(clusters)

        self.means = self.get_clusters(X)
        print('Means for clusters: %s' % str(self.means))
        intra_distance, inter_distance, s_score = silhouette_score_(X, self.y)
        # print('#samples in clusters, %d, %d, %d, %d, %d' %(self.clusters[0], self.))
        print('intra distance: %f' %intra_distance)
        print('inter_distance: %f' %inter_distance)
        print('silhouette score: %f' %s_score)
        kargs = {'intra': intra_distance, 'inter': inter_distance, 'score': s_score}
        
        res_path = os.path.normpath(os.path.join(sys.path[0], args.output_dir, 'DBSCAN_%d.txt' %self.n_clusters))
        write_data = np.c_[X, self.y]
        # print(write_data[:1])
        np.savetxt(res_path, write_data, fmt='%d',delimiter=' ')
        print('Prediction results saved in %s' % res_path)
                
        vis_path = os.path.normpath(os.path.join(sys.path[0], args.output_dir, 'DBSCAN_%d.png' %self.n_clusters))
        visualize(raw_data, X, self.y, self.n_clusters, 'DBSCAN', vis_path, self.reduction, self.random_state, **kargs)



if __name__ == "__main__":
    data = load_data(os.path.join(sys.path[0], args.data_dir, 'cluster_data.txt'))
    args=parser.parse_args()
    dbscan = dbscan(args)
    dbscan.run(data)