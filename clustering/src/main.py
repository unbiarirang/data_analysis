# !/usr/bin/env python3
# encoding=utf-8
from clusters.kmeans import KMeans
from clusters.DBSCAN import dbscan
from clusters.hierarchical import hierarchical
from utils.preprocess import load_data
from utils.visualize import visualize, MDS_visualize, visualize_proc
from utils.metrics import silhouette_score_
import os, sys
from utils.config import args

if __name__ == "__main__":
    
    data = load_data(os.path.join(sys.path[0], args.data_dir, 'cluster_data.txt'))

    if args.method == 'kmeans':
        args.init = 'random'
        kmeans_clustering = KMeans(args)
        kmeans_clustering.run(data)
    elif args.method == 'kmeans++':
        kmeans_clustering = KMeans(args)
        kmeans_clustering.run(data)
    elif args.method == 'hierarchical':
        hierarchical = hierarchical(args)
        hierarchical.run(data)
    elif args.method == 'DBSCAN':
        dbscan = dbscan(args)
        dbscan.run(data)
    elif args.method == 'test_kmeans':
        args.init = 'random'
        kmeans_clustering = KMeans(args)
        for i in range(args.repeat):
            kmeans_clustering.run(data)
    elif args.method == 'test_kmeans++':
        kmeans_clustering = KMeans(args)
        for i in range(args.repeat):
            kmeans_clustering.run(data)   
    else:
        raise ValueError('Unimplemented Method. Please choose from kmeans, hierarchical and DBSCAN')

