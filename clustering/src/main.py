from clusters.kmeans import KMeans
from clusters.DBSCAN import dbscan
from clusters.hierarchical import hierarchical
from utils.preprocess import load_data
from utils.visualize import visualize, MDS_visualize, visualize_proc
from utils.metrics import silhouette_score_
import os, sys
import argparse as ap


parser = ap.ArgumentParser()
parser.add_argument("--method", default='kmeans', help="clustering method", type=str)
parser.add_argument("--eps", default=8.5, help="The maximum distance between two samples for one to be considered \
        as in the neighborhood of the other.", type=float)
parser.add_argument("--min_samples", default=5, help="The number of samples (or total weight) in a neighborhood for a point \
        to be considered as a core point.", type=int)
parser.add_argument("--algo", default='self-implemented', help="", type=str)

parser.add_argument("--n_clusters", default=5, help="numbers of clusters", type=int)
parser.add_argument("--threshold", default=None, help="The linkage distance threshold above which, clusters will not be merged", type=float)
parser.add_argument('--linkage', default='ward', help='')

parser.add_argument("--init", default='kmeans++', help='method for init clusters', type=str)
# parser.add_argument('--linkage', default='ward', help='')

parser.add_argument("--reduction", default='TSNE', help='method for dimension reduction', type=str)
parser.add_argument("--random_state", default=10, help='random seed for dimension reduction', type=int)

parser.add_argument("--outpit_dir", default='../output', type=str)
parser.add_argument("--data_dir", default='../data', type=str)

args=parser.parse_args()

if __name__ == "__main__":
    # print(sys.path)
    data = load_data(os.path.join(sys.path[0], args.data_dir, 'cluster_data.txt'))
    # print(data)

    if args.method == 'kmeans':
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
        for i in range(12):
            kmeans_clustering.run(data)
    elif args.method == 'test_kmeans++':
        kmeans_clustering = KMeans(args)
        for i in range(12):
            kmeans_clustering.run(data)   

    else:
        raise ValueError('Unimplemented Method. Please choose from kmeans, hierarchical and DBSCAN')

