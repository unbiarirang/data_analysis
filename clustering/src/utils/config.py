# !/usr/bin/env python3
# encoding=utf-8

import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument("--method", default='kmeans', help="clustering method", type=str)
parser.add_argument("--eps", default=9, help="The maximum distance between two samples for one to be considered \
        as in the neighborhood of the other.", type=float)
parser.add_argument("--min_samples", default=6, help="The number of samples (or total weight) in a neighborhood for a point \
        to be considered as a core point.", type=int)
parser.add_argument("--algo", default='self-implemented', help="", type=str)

parser.add_argument("--n_clusters", default=5, help="numbers of clusters", type=int)
parser.add_argument("--threshold", default=None, help="The linkage distance threshold above which, clusters will not be merged", type=float)
parser.add_argument('--linkage', default='ward', help='linkage method.')

parser.add_argument("--init", default='kmeans++', help='method for init clusters', type=str)
# parser.add_argument('--linkage', default='ward', help='')

parser.add_argument("--reduction", default='TSNE', help='method for dimension reduction', type=str)
parser.add_argument("--random_state", default=10, help='random seed for dimension reduction', type=int)

parser.add_argument("--output_dir", default='../output', type=str)
parser.add_argument("--data_dir", default='../data', type=str)

parser.add_argument("--repeat", default=12, help='repeat for n times', type=int)

args=parser.parse_args()