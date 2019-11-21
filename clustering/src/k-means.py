# !/usr/bin/env python3
# encoding=utf-8

from sklearn.cluster import KMeans as KMeans_standard
from sklearn.metrics import silhouette_score
import numpy as np
import os, sys
import argparse as ap
from utils.preprocess import load_data
from utils.visualize import visualize, MDS_visualize
from utils.metrics import silhouette_score_

output_dir = "../output"
data_dir = '../data'

parser = ap.ArgumentParser()
parser.add_argument("--n_clusters", default=5, help="numbers of clusters", type=int)
parser.add_argument("--init", default='kmeans++', help='method for init clusters', type=str)
args=parser.parse_args()

class KMeans:
    def __init__(self, n_clusters, init='kmeans++'):
        self.eps = 10e-3
        self.n_clusters = n_clusters
        self.init = init
        self.means = []
        # self.n_clusters = 5
        
    def euclidean(self, X, y, square = False):
        if not square:
            return np.sqrt(np.transpose([np.sum(X* X, axis=1)]) + np.sum(y* y, axis=1) - 2* X.dot(y.T))
        else:
            return np.transpose([np.sum(X* X, axis=1)]) + np.sum(y* y, axis=1) - 2* X.dot(y.T)

    def rinit_clusters(self, X, centers = None):
        num_samples = X.shape[0]
        if centers is None:
            print('Initializing clusters by random selection')
            centers_id = np.random.randint(0, num_samples, size=self.n_clusters)
            centers = X[centers_id]
        
        # dissimilarity = np.zeros((num_samples, self.n_clusters), dtype='float64')
        # for i in range(num_samples):
        #     dissimilarity[i] = np.linalg.norm(X[i] - centers, axis = 1)
        dissimilarity = self.euclidean(X, centers)
        
        # print(dissimilarity[:50])
        # print(np.sqrt(np.sum(np.square(X[0] - points[0]))))
        
        y = np.argmin(dissimilarity, axis = 1)
        dissimilarity_y = np.min(dissimilarity, axis = 1)
        clusters = []
        means = []
        for j in range(self.n_clusters):
            points_in_cluster = np.where(y == j)
            clusters.append(points_in_cluster)
            means.append(np.average(X[points_in_cluster], axis = 0))
            # print(points_in_cluster)
        # print(means)
        return y, means, clusters

    def kinit_clusters(self, X):
        '''
        Selects initial cluster centers for k-mean clustering in a smart way
        to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
        "k-means++: the advantages of careful seeding". ACM-SIAM symposium
        on Discrete algorithms. 2007
        '''
        print('Initializing clusters by kmeans++ selection')
        num_samples = X.shape[0]
        num_features = X.shape[1]
        # centers = np.zeros((self.n_clusters, num_features))
        # clusters = []

        first_center_id = np.random.randint(0, num_samples)
        first_center = X[first_center_id]
        centers = np.array([first_center])
        n_local_trials = 2 + int(np.log(self.n_clusters))
        print(n_local_trials)

        
        for c in range(1, self.n_clusters):
            dissimilarity_square = self.euclidean(X, centers, square = True)# np.transpose([np.sum(X* X, axis=1)]) + np.sum(centers* centers, axis=1) - 2* X.dot(centers.T)
            dissimilarity_square.reshape(num_samples, -1)
            closest_dissimilarity_square = np.min(dissimilarity_square, axis=1)
            possibility_matrix = np.cumsum(closest_dissimilarity_square) # a[i]=sum(a[j]) (0 <= j <= i)
            candidate_seeds = np.random.random_sample(n_local_trials) * np.sum(closest_dissimilarity_square)
            candidate_ids = np.searchsorted(possibility_matrix, candidate_seeds)
            # print(candidate_ids)
            # select 1 in candidates
            dissimilarity_to_candidates = self.euclidean(X[candidate_ids], X, square=True)
            center_id = None
            best_potential = np.sum(closest_dissimilarity_square)
            for trial in range(n_local_trials):
                new_dissimilarity_square = np.minimum(closest_dissimilarity_square, dissimilarity_to_candidates[trial])
                # print(new_dissimilarity_square.shape)
                new_potential = np.sum(new_dissimilarity_square)
                if (center_id == None) or (new_potential < best_potential):
                    best_potential = new_potential
                    center_id = candidate_ids[trial]
                    # print(trial)
            centers = np.r_[centers, np.array([X[center_id]])]
            # print(centers.shape)
            # print(dissimilarity_square.shape)
            # print(closest_dissimilarity_square.shape)
            # print(possibility_matrix[1999])
            # print(np.sum(closest_dissimilarity_square))
            # print(center_seed)
        return self.rinit_clusters(X, centers = centers)

    def update_clusters(self, X):
        dissimilarity = np.zeros((X.shape[0], self.n_clusters))
        for i in range(X.shape[0]):
            dissimilarity[i] = np.linalg.norm(X[i] - self.means, axis = 1)

        y = np.argmin(dissimilarity, axis = 1)
        
        change = True
        if(np.all(y == self.y)): # there are label changes for some points
            change = False
        # else:
            # print(np.where((y - self.y) != 0))
        dissimilarity_y = np.min(dissimilarity, axis = 1)
        clusters = []
        means = []
        for j in range(self.n_clusters):
            points_in_cluster = np.where(y == j)
            clusters.append(points_in_cluster)
            means.append(np.average(X[points_in_cluster], axis = 0))
        return change, y, means, clusters
        
    
    def run(self, raw_data):
        X = np.array(raw_data)
        # print(X[:10])
        num_samples = X.shape[0]
        if(self.init == 'kmeans++'):
            self.y, self.means, self.clusters = self.kinit_clusters(X)
        elif(self.init == 'random'):
            self.y, self.means, self.clusters = self.rinit_clusters(X)
        elif(self.init == 'standard'):
            km = KMeans_standard(n_clusters = self.n_clusters).fit(X)
            self.y = km.labels_
        else:
            raise ValueError('Unknown init method: %s' %args.init)

        if (self.init == 'kmeans++') or (self.init == 'random'):
            change = True # if any point in clusters change its class
            iterations = 0
            while(change):
                change, self.y, self.means, self.clusters = self.update_clusters(X)
                iterations += 1
                print("iterations: %d" %iterations)
                # change, self.clusters = update_clusters(x)
            print('iterations over')

        # km = KMeans(n_clusters = n_clusters, random_state = 1244).fit(x)
        # print(km.cluster_centers_)
        # print(km.labels_)
        
        # scores = silhouette_score(X, self.y, metric='euclidean')
        # print('silhouette scores: %f' %scores)

        intra_distance, inter_distance, s_score = silhouette_score_(X, self.y)
        # print('#samples in clusters, %d, %d, %d, %d, %d' %(self.clusters[0], self.))
        print('intra distance: %f' %intra_distance)
        print('inter_distance: %f' %inter_distance)
        print('silhouette score: %f' %s_score)

        res_path = os.path.normpath(os.path.join(sys.path[0], output_dir, 'Kmeans_%d.txt' %self.n_clusters))
        write_data = np.c_[X, self.y]
        # print(write_data[:1])
        np.savetxt(res_path, write_data, fmt='%d',delimiter=' ')
        print('Prediction results saved in %s' % res_path)
                

        vis_path = os.path.normpath(os.path.join(sys.path[0], output_dir, 'Kmeans_%d.png' %self.n_clusters))
        visualize(raw_data, X, self.y, self.n_clusters, 'kmeans', vis_path)
        


if __name__ == "__main__":
    # print(sys.path)
    data = load_data(os.path.join(sys.path[0], data_dir, 'cluster_data.txt'))
    # print(data)
    kmeans_clustering = KMeans(n_clusters=args.n_clusters, init=args.init)
    kmeans_clustering.run(data)
