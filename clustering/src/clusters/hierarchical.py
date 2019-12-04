# !/usr/bin/env python3
# encoding=utf-8

# from sklearn.cluster import AgglomerativeClustering
import numpy as np
import os, sys
import argparse as ap
import heapq
from utils.preprocess import load_data
from utils.visualize import visualize, MDS_visualize, visualize_proc
from utils.metrics import silhouette_score_
from scipy.cluster.hierarchy import ward
from utils.config import args

output_dir = "../output"
data_dir = '../data'

parser = ap.ArgumentParser()
parser.add_argument("--n_clusters", default=None, help="numbers of clusters", type=int)
parser.add_argument("--threshold", default=None, help="The linkage distance threshold above which, clusters will not be merged", type=float)
parser.add_argument('--linkage', default='ward', help='linkage methods. Only support ward now.')
# min, max, group average, ward
'''
- ward minimizes the variance of the clusters being merged.
- average uses the average of the distances of each observation of the two sets.
- max linkage uses the maximum distances between all observations of the two sets.
- min uses the minimum of the distances between all observations of the two sets.
'''

class hierarchical:

    def __init__(self, args):
        if (args.n_clusters is None) and (args.threshold is None):
            args.n_clusters = 5
            # raise ValueError('Either number of clusters or distance threshold must be assigned.')
        if (args.n_clusters is not None) and (args.threshold is not None):
            raise ValueError('number of clusters and distance threshold cannot be assigned simultaneously.')
        
        self.linkage = args.linkage
        self.n_clusters = args.n_clusters
        self.distance_threshold = args.threshold
        self.reduction = args.reduction
        self.random_state = args.random_state


    def euclidean(self, X, y, square = False):
        if not square:
            return np.sqrt(np.transpose([np.sum(X* X, axis=1)]) + np.sum(y* y, axis=1) - 2* X.dot(y.T))
        else:
            return np.transpose([np.sum(X* X, axis=1)]) + np.sum(y* y, axis=1) - 2* X.dot(y.T)


    def buildHeap(self, matrix):
        distance_list = []
        distance_dict = {}
        # distance list: (dist, (i, j))
        for i in range(matrix.shape[0]):
            distance_list += list(zip(list(matrix[i][i + 1:]), list(zip(matrix.shape[1]*[i],np.arange(i+1,matrix.shape[1])))))
            # distance_list = list(np.concatenate(distance_list))
            distance_dict.update(zip(list(zip(matrix.shape[1]*[i],np.arange(i+1,matrix.shape[1]))), list(matrix[i][i + 1:])))
        # print(distance_list[0])
        heapq.heapify(distance_list)
        return distance_list, distance_dict

    def updateHeap(self, pairs, new_cluster_label):
        new_dissimilarity = []
        if self.linkage == 'ward':
            for i in self.clusters:
                if (i != pairs[0]) and (i != pairs[1]):
                    num_i = self.clusters[i].shape[0]
                    num_pairs0 = self.clusters[pairs[0]].shape[0]
                    num_pairs1 = self.clusters[pairs[1]].shape[0]
                    num_sum = num_i + num_pairs0 + num_pairs1
                    new_dist = \
                        (num_i + num_pairs0)/ num_sum * self.distance_dict[min(i, pairs[0]), max(i, pairs[0])]\
                            + (num_i + num_pairs1)/num_sum * self.distance_dict[min(i, pairs[1]), max(i, pairs[1])]\
                                - num_i/num_sum * self.distance_dict[pairs[0], pairs[1]]
                    new_pair = (min(i, new_cluster_label), max(i, new_cluster_label))
                    heapq.heappush(self.heap, (new_dist, new_pair))
                    self.distance_dict[new_pair] = new_dist
        elif self.linkage == 'min':
            pass
        elif self.linkage == 'max':
            pass
                


    def run(self, raw_data):
        X = np.array(raw_data)

        if self.linkage == 'standard':
            # clusters = AgglomerativeClustering(n_clusters=5).fit(X)
            # self.y = clusters.labels_
            pass
        else:
            dissimilarity = self.euclidean(X, X)
            self.heap, self.distance_dict = self.buildHeap(dissimilarity)

            
            self.num_samples = X.shape[0]
            self.y = np.zeros(self.num_samples)
            self.clusters = dict(zip(np.transpose(np.arange(self.num_samples)), np.transpose([np.arange(self.num_samples)])))
            self.means = []

            if self.distance_threshold is not None:
                self.n_clusters = 1
            cluster_label = self.num_samples
            while (len(self.clusters) > self.n_clusters) \
                and (self.distance_threshold is None or self.heap[0][0] > self.distance_threshold):
                dist, pairs = heapq.heappop(self.heap)
                # to check if the pair is valid (may be already merged)
                if(self.clusters.get(pairs[0]) is None) or (self.clusters.get(pairs[1]) is None):
                    continue
                if len(self.clusters) % 100 == 0:
                    print('Clusters: %d' %len(self.clusters))
                self.updateHeap(pairs, cluster_label)
                self.clusters[cluster_label] = np.r_[self.clusters[pairs[0]], self.clusters[pairs[1]]]
                self.clusters.pop(pairs[0])
                self.clusters.pop(pairs[1])
                cluster_label += 1

            num_class = 0
            for c in self.clusters:
                self.y[self.clusters[c]] = num_class
                self.means.append(np.mean(X[self.clusters[c]], axis=0))
                num_class += 1

            if(self.distance_threshold is None and num_class != self.n_clusters):
                raise ValueError('Unimplemented Error')
        
        print('Means for clusters: %s' % str(self.means))
        intra_distance, inter_distance, s_score = silhouette_score_(X, self.y)
        # print('#samples in clusters, %d, %d, %d, %d, %d' %(self.clusters[0], self.))
        print('intra distance: %f' %intra_distance)
        print('inter_distance: %f' %inter_distance)
        print('silhouette score: %f' %s_score)
        kargs = {'intra': intra_distance, 'inter': inter_distance, 'score': s_score}


        vis_path = os.path.normpath(os.path.join(sys.path[0], output_dir, 'Hierarchical_%d.png' %self.n_clusters))
        visualize(raw_data, X, self.y, self.n_clusters, 'hierarchical',vis_path, self.reduction, self.random_state, **kargs)

        vis_proc_path = os.path.normpath(os.path.join(sys.path[0], output_dir, 'Hierarchical_proc_%d.png' %self.n_clusters))
        visualize_proc(X, self.y, self.n_clusters, vis_proc_path)


if __name__ == "__main__":
    data = load_data(os.path.join(sys.path[0], data_dir, 'cluster_data.txt'))
    args=parser.parse_args()
    hierarchical = hierarchical(args)
    hierarchical.run(data)
